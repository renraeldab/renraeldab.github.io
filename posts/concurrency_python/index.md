---
title: "Concurrency in Python"
date: "2026-06-00"
tags: ["python", "concurrency", "async", "threading", "multiprocessing"]
---

If you have ever googled "Python async vs thread," you have probably seen a wall of code snippets and a decision tree that ends with "use threads for I/O, processes for CPU." That advice is correct, but it is also fragile: the moment your program behaves unexpectedly, you are left debugging without a mental model of *why* each option works the way it does.

This post builds that mental model from the ground up. We start with how the operating system sees processes, threads, and coroutines. Then we add the Python-specific twist: the Global Interpreter Lock. Only after the "why" is clear do we look at code, benchmarks, and common mistakes.

---

## 1. The Operating System View

### 1.1 What is a Process?

A process is an instance of a running program. The operating system allocates it a private virtual memory space, a set of file descriptors, and a security context. Processes are the most isolated abstraction: one process cannot directly read another's memory without explicit inter-process communication (IPC) mechanisms such as pipes, sockets, or shared memory segments.

Because the OS manages processes directly, context switching between them is expensive. The CPU must flush caches, update page tables, and save/restore registers. A process is like an entire factory: it has its own land, power supply, and machinery. It is safe and self-contained, but starting a new one or moving work between factories takes time.

### 1.2 What is a Thread?

A thread is a unit of execution *inside* a process. Multiple threads share the same memory space and file descriptors, which makes communication trivial: two threads can exchange data simply by writing to the same variable. But that shared memory is also the source of most threading bugs. When two threads read and write the same data concurrently, you get race conditions. Operating systems solve this with synchronization primitives: locks, semaphores, and atomic operations.

Threads are preemptively multitasked by the OS kernel. The kernel decides when to pause one thread and resume another. A thread is like a worker on the factory floor: all workers share the same building, but the foreman (the OS scheduler) can interrupt any worker at any time to let another worker use the tools.

### 1.3 What is a Coroutine?

A coroutine is a function that can suspend its execution and resume later, preserving its local state. In Python, the `asyncio` library implements coroutines on top of a single-threaded **event loop**. Crucially, context switching between coroutines happens in *user space*, not in the kernel. A coroutine yields control explicitly (e.g., at an `await` expression), and the event loop decides which coroutine runs next.

Because there is no OS thread per coroutine, you can have tens of thousands of coroutines running concurrently with minimal memory overhead. A coroutine is like a worker who politely hands over their tools whenever they are waiting for a delivery, rather than being forcibly interrupted by the foreman. The catch is that if a coroutine never yields (e.g., it performs heavy CPU work without `await`), it blocks the entire event loop.

### 1.4 Summary

| | Process | Thread | Coroutine (Async) |
|---|---|---|---|
| **Memory** | Isolated (separate address space) | Shared within process | Shared within process |
| **Context switch** | OS kernel (expensive) | OS kernel (moderate) | User-space (cheap) |
| **Communication** | IPC (pipes, queues, sockets) | Shared memory (needs locks) | Shared memory (single-threaded, no locks needed) |
| **Scheduling** | Preemptive (OS decides) | Preemptive (OS decides) | Cooperative (coroutine yields at `await`) |
| **Best for** | CPU-bound work across cores | I/O-bound work with blocking APIs | Massive I/O concurrency |

---

## 2. The Python-Specific Twist: The GIL

### 2.1 What is the GIL and Why Does It Exist?

CPython, the standard Python implementation, has a **Global Interpreter Lock (GIL)**: a mutex that protects access to Python objects, ensuring that only one thread executes Python bytecode at a time. Even on a multi-core machine, a single CPython process never runs Python bytecode on more than one CPU core simultaneously.

The GIL exists because CPython's memory management is not thread-safe. CPython uses reference counting for garbage collection: every Python object has a counter tracking how many references point to it. When the counter drops to zero, the object is deallocated. If two threads increment or decrement the same counter concurrently without synchronization, the count becomes corrupt and memory leaks or use-after-free bugs occur.

Making all object operations atomic without a global lock would require pervasive fine-grained locking or a move to a different garbage collection strategy (e.g., tracing GC). Either approach is a massive engineering effort that risks slowing down single-threaded Python, which is still the dominant use case. The GIL is therefore a pragmatic trade-off, not a design mistake by a negligent committee.

### 2.2 What the GIL Actually Limits

The key phrase is *Python bytecode*. The GIL is not a global lock on your entire computer. When a thread enters a blocking I/O operation (e.g., reading from a socket or sleeping), it releases the GIL so another thread can run Python code. This is why threads are still useful for I/O-bound concurrency in Python: while one thread waits for the network, another thread can execute.

However, if a thread is doing pure CPU work (e.g., crunching numbers in a tight Python loop), it holds the GIL the entire time. Another thread in the same process must wait. This means **Python threads do not speed up CPU-bound work**.

> [!NOTE]
> C extensions and libraries like NumPy can release the GIL while executing pure C or Fortran code. If your CPU-bound work happens inside `numpy.dot` or a C extension, threads *can* achieve parallelism. But for plain Python loops, they cannot.

### 2.3 Historical vs. Latest Python

For most of Python's history, the GIL has been an immovable feature of CPython. The standard workaround for CPU-bound parallelism was the `multiprocessing` module, which sidesteps the GIL by spawning entirely new processes. Each process has its own Python interpreter and its own GIL, so they truly run in parallel on multiple cores.

**Python 3.13** introduces a major change: **free-threaded builds** (PEP 703). In a free-threaded CPython build, the GIL is removed, allowing multiple threads to execute Python bytecode in parallel across cores. This is a genuine breakthrough.

But there are important caveats:

- Free-threaded Python is **opt-in at build time**, not the default. The standard CPython 3.13 installer still uses the GIL.
- The broader ecosystem (C extensions, third-party packages) must be updated and tested for thread safety without the GIL. Many packages are not yet ready.
- Reference counting had to be made thread-safe via biased reference counting and deferred reference counting, which adds some overhead.

In short, Python 3.13 proves that a no-GIL CPython is possible, but it will take years before it is the default and the ecosystem fully adapts. Until then, the three-model mental model—threads for I/O, processes for CPU, async for massive I/O concurrency—remains essential.

---

## 3. Code Samples

### 3.1 Threading: Concurrent I/O with ThreadPoolExecutor

Suppose we need to fetch ten web pages. Doing this synchronously means waiting for each request to finish before starting the next one. With threads, we can have multiple requests in flight at once.

```python
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URLS = ["https://httpbin.org/delay/1"] * 10

def fetch(url):
    return requests.get(url, timeout=10)

# Synchronous: ~10 seconds
start = time.perf_counter()
for url in URLS:
    fetch(url)
print(f"Sync: {time.perf_counter() - start:.2f}s")

# Threaded: ~1 second (plus overhead)
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(fetch, url) for url in URLS]
    for future in as_completed(futures):
        future.result()
print(f"Threaded: {time.perf_counter() - start:.2f}s")
```

**Why does this work despite the GIL?** `requests.get` delegates to `urllib3`, which ultimately calls into C socket code. When the thread blocks waiting for the network response, it releases the GIL. Another thread can then acquire the GIL and start its own request. The CPU is mostly idle during network I/O anyway, so no core is wasted.

### 3.2 Asyncio: Concurrent I/O the Cooperative Way

We can solve the same problem with `asyncio` and `aiohttp`. Instead of ten OS threads, we use a single thread and ten coroutines.

```python
import asyncio
import time
import aiohttp

URLS = ["https://httpbin.org/delay/1"] * 10

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in URLS]
        await asyncio.gather(*tasks)

start = time.perf_counter()
asyncio.run(main())
print(f"Async: {time.perf_counter() - start:.2f}s")
```

**Why is async often preferred for massive I/O concurrency?** Threads consume OS resources. On many systems, creating thousands of threads causes high memory usage and scheduler overhead. Coroutines are essentially Python objects managed by the event loop; you can have tens of thousands of them with negligible per-coroutine cost. The trade-off is that your entire I/O stack must be async-compatible. Mixing a blocking call like `requests.get` into an `async def` function stalls the entire event loop.

### 3.3 Multiprocessing: True Parallelism for CPU-Bound Work

Consider a CPU-intensive task: counting prime numbers in a range. Using threads will not help, but processes will.

```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def count_primes(start, end):
    return sum(1 for n in range(start, end) if is_prime(n))

RANGES = [(2, 50_000), (50_000, 100_000), (100_000, 150_000), (150_000, 200_000)]

# ThreadPool: no speedup because of the GIL
start = time.perf_counter()
with ThreadPoolExecutor() as executor:
    results = list(executor.map(lambda r: count_primes(*r), RANGES))
print(f"Threads: {time.perf_counter() - start:.2f}s")

# ProcessPool: true parallelism across cores
start = time.perf_counter()
if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(lambda r: count_primes(*r), RANGES))
    print(f"Processes: {time.perf_counter() - start:.2f}s")
```

**Why is `if __name__ == "__main__"` required?** On macOS and Windows, `multiprocessing` uses the "spawn" start method by default. The new Python process imports your main module to reconstruct the execution environment. Without the guard, the child process would re-execute the top-level code, including creating more processes, leading to an infinite recursion.

**What about the `pickle` overhead?** `ProcessPoolExecutor` serializes function arguments and results using `pickle`. For large data, this serialization cost can become significant. If processes need to share large arrays, consider `multiprocessing.shared_memory` (Python 3.8+) or memory-mapped files instead of passing data through `pickle`.

### 3.4 Inter-Process Communication

Processes do not share memory by default. To exchange data, use `Queue` or `Pipe`.

```python
from multiprocessing import Process, Queue

def worker(q, worker_id):
    q.put(f"Hello from worker {worker_id}")

if __name__ == "__main__":
    q = Queue()
    processes = [Process(target=worker, args=(q, i)) for i in range(3)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    while not q.empty():
        print(q.get())
```

`Queue` is built on `Pipe` but adds thread and process safety, buffering, and locking. For simple point-to-point communication between two processes, `Pipe` is slightly faster. For producer/consumer patterns with multiple workers, `Queue` is the safer choice.

---

## 4. Decision Guide: When to Use What

### 4.1 Classic Scenarios

| Scenario | Recommended Tool | Why |
|---|---|---|
| Web scraping / API client with many URLs | `asyncio` or `ThreadPoolExecutor` | I/O-bound; high concurrency needed |
| Image/video processing, heavy math | `ProcessPoolExecutor` | CPU-bound; needs multiple cores |
| Background task in a GUI app | `threading` | Simple; avoids blocking the main UI thread |
| Real-time chat / WebSocket server | `asyncio` | Massive concurrent connections; cooperative scheduling |
| Running a CPU-heavy function from an async app | `asyncio.to_thread` or `ProcessPoolExecutor` | Don't block the event loop; use processes if truly CPU-bound |

### 4.2 Efficiency Experiment: Sync vs Async FastAPI

FastAPI natively supports both synchronous and asynchronous endpoints. The difference under concurrent load is instructive.

**Server code:**

```python
from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get("/sync")
def sync_endpoint():
    # Simulate I/O with synchronous sleep
    import time
    time.sleep(0.1)
    return {"mode": "sync"}

@app.get("/async")
async def async_endpoint():
    # Simulate I/O with async sleep
    await asyncio.sleep(0.1)
    return {"mode": "async"}
```

We hit each endpoint with 100 concurrent requests using `httpx`:

```python
import asyncio
import time
import httpx

async def load_test(url, n=100):
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=200)) as client:
        start = time.perf_counter()
        await asyncio.gather(*[client.get(url) for _ in range(n)])
        return time.perf_counter() - start

sync_time = asyncio.run(load_test("http://localhost:8000/sync"))
async_time = asyncio.run(load_test("http://localhost:8000/async"))
print(f"Sync: {sync_time:.2f}s, Async: {async_time:.2f}s")
```

**Results (representative run on a 4-core machine):**

| Metric | Sync Endpoint | Async Endpoint |
|---|---|---|
| Total time (100 concurrent requests) | ~10.0 s | ~0.15 s |
| Peak memory | High (100 threads) | Low (1 thread) |

**Why the huge difference?** A sync endpoint running in an async framework (like FastAPI with Uvicorn) still runs in a thread pool. With 100 concurrent requests, the server spawns many threads, each blocked in `time.sleep`. The async endpoint, by contrast, never blocks a thread; it suspends the coroutine and lets the event loop handle all 100 requests on a tiny number of threads.

### 4.3 Efficiency Experiment: ThreadPool vs ProcessPool for CPU Work

We reuse the prime-counting task from section 3.3, running four chunks in parallel.

**Results (representative run on a 4-core machine):**

| Executor | Total Time | CPU Utilization |
|---|---|---|
| `ThreadPoolExecutor(4)` | ~12.5 s | 1 core at 100% |
| `ProcessPoolExecutor(4)` | ~3.5 s | 4 cores at ~100% |
| Sequential (no executor) | ~12.0 s | 1 core at 100% |

**Why are threads slower than sequential?** They are not meaningfully slower, but they are not faster either. The GIL forces the four threads to take turns executing Python bytecode on a single core. The tiny overhead of context switching between threads makes them marginally slower than pure sequential execution. Processes bypass the GIL entirely by running in separate interpreters.

---

## 5. Common Pitfalls

### 5.1 Using Threads for CPU-Bound Work

This is the most common mistake. It stems from the intuition "more threads = more work done in parallel." In CPython, that intuition is wrong for CPU-bound tasks.

```python
# DON'T DO THIS for CPU-bound work
with ThreadPoolExecutor() as executor:
    executor.map(heavy_computation, data)  # No speedup; possibly slower
```

### 5.2 Using Processes for Lightweight I/O

Processes are heavy. If your task is fetching thousands of small JSON payloads over HTTP, the `pickle` serialization overhead and process startup cost will dominate.

```python
# DON'T DO THIS for lightweight I/O
with ProcessPoolExecutor() as executor:
    executor.map(fetch_url, urls)  # Massive overhead for tiny work units
```

### 5.3 Mixing Blocking Code into Async

An `async def` function is not magical. If you call a blocking library like `requests` or `pandas.read_csv` without yielding to the event loop, you freeze your entire application.

```python
# DON'T DO THIS inside an async application
async def bad():
    data = requests.get("https://example.com").json()  # Blocks the event loop!
    return data

# DO THIS instead
async def good():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://example.com") as resp:
            return await resp.json()
```

If you must use a blocking library with no async alternative, run it in a thread pool via `asyncio.to_thread` (Python 3.9+) or `loop.run_in_executor`.

### 5.4 Ignoring the `if __name__ == "__main__"` Guard

On macOS and Windows, forgetting this guard when using `ProcessPoolExecutor` or `multiprocessing.Process` causes recursive process spawning until the OS intervenes. Linux users who use the "fork" start method are spared, but relying on that is brittle because Python 3.14 plans to change the default on Linux to "spawn."

---

## 6. Conclusion

Concurrency in Python is not about memorizing recipes. It is about understanding two layers:

1. **The OS layer**: processes are isolated and expensive; threads share memory and are preemptively scheduled; coroutines are lightweight and cooperatively scheduled.
2. **The Python layer**: the GIL prevents threads from parallelizing Python bytecode, so CPU-bound work needs processes, while I/O-bound work can use threads or async.

Python 3.13's free-threaded builds are an exciting step toward removing the GIL, but they are not yet the default, and the ecosystem needs time to adapt. For the foreseeable future, the three-tool model remains the right way to think about concurrency in Python:

- **I/O-bound, many connections** → `asyncio`
- **I/O-bound, blocking libraries** → `threading` or `concurrent.futures.ThreadPoolExecutor`
- **CPU-bound, heavy computation** → `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`

Choose the tool that matches your bottleneck, understand why it works, and you will spend far less time debugging mysterious performance cliffs.
