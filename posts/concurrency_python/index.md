---
title: "Concurrency in Python"
date: "2026-06-00"
tags: ["python", "concurrency", "async", "threading", "multiprocessing"]
---

In this post, We start with how the operating system sees processes, threads, and coroutines. Then we add the Python-specific 
twist: the Global Interpreter Lock. Only after the "why" is clear do we look at code, benchmarks, and common mistakes. By the end, 
you should be able to answer common concurrency interview questions and know exactly which tool to reach for.

## Overview

### Processes, Threads, and Coroutines

| | Process | Thread | Coroutine (Async) |
|---|---|---|---|
| **Memory** | Isolated (separate address space) | Shared within process | Shared within process |
| **Context switch** | OS kernel (expensive) | OS kernel (moderate) | User-space (cheap) |
| **Communication** | IPC | Shared memory | Shared memory |
| **Scheduling** | Preemptive (OS decides) | Preemptive (OS decides) | Cooperative (coroutine yields at `await`) |
| **Best for** | CPU-bound work across cores | I/O-bound work with blocking APIs | Massive I/O concurrency |

A **process** is an instance of a running program with its own virtual memory and file descriptors. It is the most isolated 
and most expensive to create.

A **thread** is a unit of execution inside a process. Multiple threads share the same memory, which makes communication 
easy but requires synchronization (locks, semaphores) to prevent race conditions.

A **coroutine** is a function that can suspend and resume while preserving its local state. In Python, `asyncio` runs coroutines 
on a single-threaded **event loop**. Because there is no OS thread per coroutine, you can have tens of thousands of them 
with minimal memory overhead. The catch is that a coroutine must explicitly yield control at `await`; if it never yields, 
it blocks the entire loop.

### The Global Interpreter Lock (GIL)

CPython, the standard Python implementation, has a **Global Interpreter Lock (GIL)**: a mutex that protects access to Python 
objects, ensuring that only one thread executes Python bytecode at a time. Even on a multi-core machine, a single CPython 
process never runs Python bytecode on more than one CPU core simultaneously.

The GIL exists because CPython uses reference counting for garbage collection. If two threads incremented or decremented 
the same counter concurrently without synchronization, memory would corrupt. Removing the GIL entirely would require pervasive 
fine-grained locking or a different GC strategy, both of which risk slowing down single-threaded Python. The GIL is therefore 
a pragmatic trade-off.

When a thread enters a blocking I/O operation (e.g., reading from a socket), it **releases the GIL** so another thread can 
run. This is why threads are still useful for I/O-bound work in Python. However, if a thread is doing pure CPU work in a 
tight Python loop, it holds the GIL the entire time, and another thread in the same process must wait. Python threads do 
not speed up CPU-bound pure-Python work.

> [!NOTE]
> C extensions and libraries like NumPy can release the GIL while executing pure C or Fortran code. If your CPU-bound work 
> happens inside `numpy.dot` or a C extension, threads *can* achieve parallelism.

### Historical vs. Latest Python

For most of Python's history, the GIL has been immovable. The standard workaround for CPU-bound parallelism was the `multiprocessing` 
module, which sidesteps the GIL by spawning entirely new processes.

**Python 3.13** introduces **free-threaded builds** ([PEP 703](https://peps.python.org/pep-0703/)). In a free-threaded CPython 
build, the GIL is removed, allowing multiple threads to execute Python bytecode in parallel. But it is opt-in at build time, 
not the default, and the broader ecosystem of C extensions still needs time to adapt.

## Threading — I/O-Bound Concurrency

### Theory and Suitable Scenarios

Threading is ideal for **I/O-bound** work: network requests, disk reads, database queries, and any operation where the program 
spends most of its time waiting on an external resource.

Because the OS scheduler preemptively switches between threads, a thread that is waiting for I/O can be paused while another 
thread runs Python code. Crucially, the waiting thread **releases the GIL** during the blocking call, so other threads are 
not starved. The CPU is mostly idle during network I/O anyway, so no core is wasted.

Threading is also the easiest concurrency model to retrofit into existing synchronous code. If your codebase already uses 
blocking libraries like `requests` or `pandas`, wrapping calls in a `ThreadPoolExecutor` often requires minimal changes.

### Classic Use Cases

- Web scraping or API clients with many URLs
- Background tasks in GUI or desktop applications
- Parallel file or database I/O
- Calling multiple blocking services and aggregating results

### Experiment: Fetching HTTP Requests

We start a local server that sleeps for a fixed delay to simulate network latency, then issue 500 requests using three approaches.

```python
import asyncio
import concurrent.futures
import subprocess
import sys
import time

import httpx

SERVER_URL = "http://127.0.0.1:9001"
NUM_REQUESTS = 500
DELAY_SECONDS = 0.1


def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-c", f"""
import time
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/delay/{{seconds}}")
def delay(seconds: float):
    time.sleep(seconds)
    return {{"delayed": seconds}}

uvicorn.run(app, host="127.0.0.1", port=9001, log_level="warning")
"""],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for _ in range(50):
        try:
            httpx.get(f"{SERVER_URL}/delay/0.01", timeout=0.2)
            break
        except Exception:
            time.sleep(0.1)
    return proc


def run_sync():
    url = f"{SERVER_URL}/delay/{DELAY_SECONDS}"
    start = time.perf_counter()
    with httpx.Client() as client:
        for _ in range(NUM_REQUESTS):
            client.get(url)
    return time.perf_counter() - start


def run_threads():
    url = f"{SERVER_URL}/delay/{DELAY_SECONDS}"
    start = time.perf_counter()
    with httpx.Client() as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(lambda _: client.get(url), range(NUM_REQUESTS)))
    return time.perf_counter() - start


async def run_async():
    url = f"{SERVER_URL}/delay/{DELAY_SECONDS}"
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[client.get(url) for _ in range(NUM_REQUESTS)])
    return time.perf_counter() - start


def main():
    print("Starting local FastAPI server...")
    proc = start_server()
    try:
        print(f"Benchmarking {NUM_REQUESTS} requests with {DELAY_SECONDS}s delay each\n")

        t_sync = run_sync()
        t_threads = run_threads()
        t_async = asyncio.run(run_async())

        results = [
            ("Synchronous", t_sync),
            ("ThreadPoolExecutor (10 workers)", t_threads),
            ("Asyncio + httpx", t_async),
        ]

        max_label = max(len(label) for label, _ in results)
        print("-" * (max_label + 20))
        print(f"{'Approach':<{max_label}} | {'Time (s)':>10}")
        print("-" * (max_label + 20))
        for label, elapsed in results:
            print(f"{label:<{max_label}} | {elapsed:>10.4f}")
        print("-" * (max_label + 20))

        baseline = results[0][1]
        print("\nSpeedup relative to baseline:")
        for label, elapsed in results:
            print(f"  {label}: {baseline / elapsed:.2f}x")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
```

Results:

```
---------------------------------------------------
Approach                        |   Time (s)
---------------------------------------------------
Synchronous                     |    49.4284
ThreadPoolExecutor (10 workers) |     5.4921
Asyncio + httpx                 |     3.6578
---------------------------------------------------

Speedup relative to baseline:
  Synchronous: 1.00x
  ThreadPoolExecutor (10 workers): 9.00x
  Asyncio + httpx: 13.51x
```

## Asyncio — Massive I/O Concurrency

### Theory and Suitable Scenarios

Asyncio is ideal when you need **massive I/O concurrency**: thousands of simultaneous connections, WebSocket servers, or 
high-throughput API gateways.

Whereas threading relies on the OS kernel to preemptively switch between threads, asyncio uses a single-threaded **event loop** 
and **cooperative scheduling**. A coroutine explicitly yields control at an `await` expression, and the event loop immediately 
switches to another ready coroutine. Because there is no OS thread per connection, the memory overhead is tiny.

The trade-off is that your entire I/O stack must be async-compatible. Mixing a blocking call like `requests.get` or `time.sleep` 
into an `async def` function stalls the entire event loop and defeats the purpose.

> [!NOTE]
> If you must call a blocking library inside an async app, run it in a thread pool via `asyncio.to_thread` (Python 3.9+) 
> or `loop.run_in_executor`.

### Classic Use Cases

- Real-time chat or WebSocket servers
- High-concurrency API clients (e.g., scraping thousands of pages)
- ASGI web servers (FastAPI, Starlette, Sanic)
- Connection pools and proxies

### Experiment: FastAPI Async vs Sync Endpoints

FastAPI natively supports both synchronous and asynchronous endpoints. The difference under concurrent load is instructive 
because it reveals what happens when blocking code enters an async framework.

Server code (`server.py`):

```python
import asyncio
import time

from fastapi import FastAPI

app = FastAPI()
SLEEP_SECONDS = 0.1


@app.get("/async")
async def async_endpoint():
    await asyncio.sleep(SLEEP_SECONDS)
    return {"type": "async"}


@app.get("/sync")
def sync_endpoint():
    time.sleep(SLEEP_SECONDS)
    return {"type": "sync"}
```

Benchmark code (`benchmark.py`):

```python
import asyncio
import statistics
import subprocess
import sys
import time

import httpx

URL_ASYNC = "http://127.0.0.1:9002/async"
URL_SYNC = "http://127.0.0.1:9002/sync"
TOTAL_REQUESTS = 300
CONCURRENT_REQUESTS = 100


def start_server():
    import pathlib
    cwd = pathlib.Path(__file__).parent
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", "9002", "--log-level", "warning"],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for _ in range(50):
        try:
            httpx.get(URL_ASYNC, timeout=0.2)
            break
        except Exception:
            time.sleep(0.1)
    return proc


async def fetch_all(url):
    latencies = []

    async def fetch(client):
        start = time.perf_counter()
        await client.get(url)
        latencies.append(time.perf_counter() - start)

    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def bounded_fetch():
            async with semaphore:
                await fetch(client)

        wall_start = time.perf_counter()
        await asyncio.gather(*[bounded_fetch() for _ in range(TOTAL_REQUESTS)])
        wall_elapsed = time.perf_counter() - wall_start
    return latencies, wall_elapsed


async def benchmark():
    print("Starting server...")
    proc = start_server()
    try:
        print(f"Firing {TOTAL_REQUESTS} requests ({CONCURRENT_REQUESTS} concurrent)\n")

        latencies_async, elapsed_async = await fetch_all(URL_ASYNC)
        latencies_sync, elapsed_sync = await fetch_all(URL_SYNC)

        def summarize(name, latencies, elapsed):
            print(f"{name} endpoint:")
            print(f"  Total time  : {elapsed:.3f}s")
            print(f"  Throughput  : {TOTAL_REQUESTS / elapsed:.1f} req/s")
            print(f"  Mean latency: {statistics.mean(latencies)*1000:.1f}ms")
            print(f"  P99 latency : {sorted(latencies)[int(len(latencies)*0.99)]*1000:.1f}ms")
            print()

        summarize("Async", latencies_async, elapsed_async)
        summarize("Sync", latencies_sync, elapsed_sync)
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    asyncio.run(benchmark())
```

Results:

```
Async endpoint:
  Total time  : 0.518s
  Throughput  : 579.6 req/s
  Mean latency: 146.9ms
  P99 latency : 212.6ms

Sync endpoint:
  Total time  : 0.961s
  Throughput  : 312.1 req/s
  Mean latency: 269.1ms
  P99 latency : 417.7ms
```

**Why the difference?** A sync endpoint running in an async framework (FastAPI with Uvicorn) still runs in a thread pool. 
With 100 concurrent requests, many threads are blocked in `time.sleep`. Once the pool saturates, new requests queue up, 
spiking latency. The async endpoint never blocks a thread; it suspends the coroutine and lets the event loop handle all 
requests on a tiny number of threads.

## Multiprocessing — CPU-Bound Parallelism

### Theory and Suitable Scenarios

Multiprocessing is the right tool for **CPU-bound** work: heavy computation, brute-force algorithms, image processing, and 
any task where the CPU is the bottleneck.

Because the GIL prevents multiple threads from executing Python bytecode in parallel, threads are useless for pure-Python 
CPU work. Multiprocessing sidesteps the GIL entirely by spawning **separate processes**, each with its own Python interpreter 
and its own GIL. The OS can schedule these processes on different CPU cores simultaneously, achieving true parallelism.

The trade-offs are higher memory usage (each process has its own memory space), slower startup time, and the cost of inter-process 
communication (IPC). `ProcessPoolExecutor` serializes arguments and results using `pickle`, which can become expensive for 
large data. For large arrays, consider `multiprocessing.shared_memory` (Python 3.8+) instead.

### Classic Use Cases

- Heavy mathematical computation
- Image and video processing pipelines
- Data transformation on large datasets
- Running CPU-heavy functions from a synchronous or async application

### Experiment 1: Pure-Python CPU Work

We use a classic CPU-bound task—counting prime numbers with a brute-force algorithm written in pure Python—to demonstrate 
that threads provide almost no speedup while processes scale with core count.

```python
import concurrent.futures
import math
import os
import time

NUM_TASKS = 16
PRIME_LIMIT = 500000


def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    limit = int(math.isqrt(n)) + 1
    for i in range(3, limit, 2):
        if n % i == 0:
            return False
    return True


def count_primes(limit):
    count = 0
    for n in range(2, limit):
        if is_prime(n):
            count += 1
    return count


def run_sync():
    start = time.perf_counter()
    for _ in range(NUM_TASKS):
        count_primes(PRIME_LIMIT)
    return time.perf_counter() - start


def run_threads():
    workers = os.cpu_count() or 4
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(count_primes, [PRIME_LIMIT] * NUM_TASKS))
    return time.perf_counter() - start


def run_processes():
    workers = os.cpu_count() or 4
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(count_primes, [PRIME_LIMIT] * NUM_TASKS))
    return time.perf_counter() - start


def main():
    print(f"Benchmarking prime counting up to {PRIME_LIMIT} ({NUM_TASKS} tasks)\n")
    t_sync = run_sync()
    t_threads = run_threads()
    t_processes = run_processes()

    results = [
        ("Synchronous", t_sync),
        ("ThreadPoolExecutor", t_threads),
        ("ProcessPoolExecutor", t_processes),
    ]

    max_label = max(len(label) for label, _ in results)
    print("-" * (max_label + 20))
    print(f"{'Approach':<{max_label}} | {'Time (s)':>10}")
    print("-" * (max_label + 20))
    for label, elapsed in results:
        print(f"{label:<{max_label}} | {elapsed:>10.4f}")
    print("-" * (max_label + 20))

    baseline = results[0][1]
    print("\nSpeedup relative to baseline:")
    for label, elapsed in results:
        print(f"  {label}: {baseline / elapsed:.2f}x")


if __name__ == "__main__":
    main()
```

> [!NOTE]
> On macOS and Windows, `multiprocessing` uses the "spawn" start method by default. The new process imports your main module 
> to reconstruct the environment. Without `if __name__ == "__main__"`, the child process re-executes the top-level code, 
> creating more processes in an infinite recursion. See "Safe importing of main module" in [the official documentation](https://docs.python.org/3/library/multiprocessing.html).

Results:

```
---------------------------------------
Approach            |   Time (s)
---------------------------------------
Synchronous         |     4.1210
ThreadPoolExecutor  |     3.9364
ProcessPoolExecutor |     0.6353
---------------------------------------

Speedup relative to baseline:
  Synchronous: 1.00x
  ThreadPoolExecutor: 1.05x
  ProcessPoolExecutor: 6.49x
```

### Experiment 2: NumPy and the GIL

NumPy's core routines are implemented in C and Fortran (BLAS/LAPACK). When NumPy executes these routines, it **releases the GIL**. 
This means multiple Python threads can execute heavy NumPy operations in parallel on different CPU cores.

```python
import concurrent.futures
import os
import time

import numpy as np

NUM_TASKS = 20
MATRIX_SIZE = 1600


def matrix_task(_i):
    a = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
    b = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
    for _ in range(5):
        a = a @ b
    return a


def run_sync():
    start = time.perf_counter()
    for i in range(NUM_TASKS):
        matrix_task(i)
    return time.perf_counter() - start


def run_threads():
    workers = os.cpu_count() or 4
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(matrix_task, range(NUM_TASKS)))
    return time.perf_counter() - start


def run_processes():
    workers = os.cpu_count() or 4
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(matrix_task, range(NUM_TASKS)))
    return time.perf_counter() - start


def main():
    print(f"Benchmarking {NUM_TASKS} matrix multiplications ({MATRIX_SIZE}x{MATRIX_SIZE})\n")
    t_sync = run_sync()
    t_threads = run_threads()
    t_processes = run_processes()

    results = [
        ("Synchronous", t_sync),
        ("ThreadPoolExecutor", t_threads),
        ("ProcessPoolExecutor", t_processes),
    ]

    max_label = max(len(label) for label, _ in results)
    print("-" * (max_label + 20))
    print(f"{'Approach':<{max_label}} | {'Time (s)':>10}")
    print("-" * (max_label + 20))
    for label, elapsed in results:
        print(f"{label:<{max_label}} | {elapsed:>10.4f}")
    print("-" * (max_label + 20))

    baseline = results[0][1]
    print("\nSpeedup relative to baseline:")
    for label, elapsed in results:
        print(f"  {label}: {baseline / elapsed:.2f}x")


if __name__ == "__main__":
    main()
```

Results:

```
---------------------------------------
Approach            |   Time (s)
---------------------------------------
Synchronous         |     1.2704
ThreadPoolExecutor  |     0.9427
ProcessPoolExecutor |     1.0350
---------------------------------------

Speedup relative to baseline:
  Synchronous: 1.00x
  ThreadPoolExecutor: 1.35x
  ProcessPoolExecutor: 1.23x
```

**Interpretation:** Threads achieve real parallelism because NumPy releases the GIL during the C-level matrix multiplication. 
Processes are close but slightly slower due to the cost of spawning workers and pickling large arrays back and forth.

### A Note on PyTorch

The same GIL-release principle applies to PyTorch: its CPU tensor operations also release the GIL, so multiple threads can 
run CPU-based tensor math in parallel. But PyTorch's concurrency story diverges from NumPy's once GPUs enter the picture, 
and that is where most PyTorch workloads live.

When you call `.cuda()` and launch a kernel, the actual computation is performed by the GPU, not by Python threads. The 
CPU's job is merely to enqueue work into a **CUDA stream**. Because GPU execution is asynchronous with respect to the CPU, 
a single Python thread can keep the GPU saturated by launching kernels fast enough. The bottleneck is rarely Python bytecode 
execution; it is **keeping the GPU fed** with data and maximizing GPU utilization.

## Mixed Workloads — Putting It All Together

### Theory and Suitable Scenarios

Real-world applications rarely fit neatly into "only I/O" or "only CPU" categories. A typical pipeline looks like this:

1. Fetch a batch of data over the network (I/O)
2. Transform or analyze the batch (CPU)
3. Repeat

### Classic Use Cases

- ETL pipelines that fetch data from APIs and then transform it
- ML inference servers that receive requests over HTTP and run model inference
- Data enrichment pipelines (fetch → parse → compute → store)

### Experiment: I/O + CPU Pipeline

We simulate 40 batches, each requiring a short I/O wait and a pure-Python CPU calculation.

```python
import asyncio
import concurrent.futures
import os
import time

NUM_BATCHES = 40
IO_DELAY = 0.05
CPU_WORK_ITERATIONS = 5000000


def cpu_work():
    total = 0
    for i in range(CPU_WORK_ITERATIONS):
        total += i * i
    return total


async def io_fetch(batch_id):
    await asyncio.sleep(IO_DELAY)
    return batch_id


async def run_async_only():
    async def task(batch_id):
        await io_fetch(batch_id)
        cpu_work()

    start = time.perf_counter()
    await asyncio.gather(*[task(i) for i in range(NUM_BATCHES)])
    return time.perf_counter() - start


def process_task(_batch_id):
    time.sleep(IO_DELAY)
    cpu_work()


def run_process_only():
    workers = os.cpu_count() or 4
    start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process_task, range(NUM_BATCHES)))
    return time.perf_counter() - start


async def run_hybrid():
    workers = os.cpu_count() or 4
    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        async def task(batch_id):
            await io_fetch(batch_id)
            await loop.run_in_executor(executor, cpu_work)

        await asyncio.gather(*[task(i) for i in range(NUM_BATCHES)])
    return time.perf_counter() - start


def main():
    print(f"Benchmarking {NUM_BATCHES} batches ({IO_DELAY}s I/O + CPU work each)\n")
    t_async = asyncio.run(run_async_only())
    t_process = run_process_only()
    t_hybrid = asyncio.run(run_hybrid())

    results = [
        ("Pure asyncio (CPU blocks loop)", t_async),
        ("Pure multiprocessing", t_process),
        ("Hybrid (asyncio + ProcessPool)", t_hybrid),
    ]

    max_label = max(len(label) for label, _ in results)
    print("-" * (max_label + 20))
    print(f"{'Approach':<{max_label}} | {'Time (s)':>10}")
    print("-" * (max_label + 20))
    for label, elapsed in results:
        print(f"{label:<{max_label}} | {elapsed:>10.4f}")
    print("-" * (max_label + 20))

    baseline = results[0][1]
    print("\nSpeedup relative to baseline:")
    for label, elapsed in results:
        print(f"  {label}: {baseline / elapsed:.2f}x")


if __name__ == "__main__":
    main()
```

Results:

```
--------------------------------------------------
Approach                       |   Time (s)
--------------------------------------------------
Pure asyncio (CPU blocks loop) |     4.4142
Pure multiprocessing           |     0.7197
Hybrid (asyncio + ProcessPool) |     0.6147
--------------------------------------------------

Speedup relative to baseline:
  Pure asyncio (CPU blocks loop): 1.00x
  Pure multiprocessing: 6.13x
  Hybrid (asyncio + ProcessPool): 7.18x
```

**Interpretation:** Pure asyncio is the slowest because CPU work blocks the event loop, serializing both I/O and computation. 
Pure multiprocessing improves on the CPU side but cannot overlap I/O across batches inside a single process. The hybrid 
wins by overlapping all I/O concurrently while keeping all CPU cores busy.

## Conclusion

Concurrency in Python is not about memorizing recipes. It is about understanding two layers:

**The OS layer:**

- processes are isolated and expensive;
- threads share memory and are preemptively scheduled;
- coroutines are lightweight and cooperatively scheduled.

**The Python layer:**

- the GIL prevents threads from parallelizing Python bytecode;
- so CPU-bound pure-Python work needs processes;
- while I/O-bound work can use threads or async.

Python 3.13's free-threaded builds are an exciting step toward removing the GIL, but they are not yet the default, and the 
ecosystem needs time to adapt. For the foreseeable future, the three-tool model remains the right way to think about concurrency 
in Python:

- **I/O-bound, many connections:** `asyncio`
- **I/O-bound, blocking libraries:** `threading` or `concurrent.futures.ThreadPoolExecutor`
- **CPU-bound, heavy computation:** `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`
- **Mixed workloads:** compose `asyncio` with `ProcessPoolExecutor`

Choose the tool that matches your bottleneck, and understand why it works.