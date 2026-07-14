---
title: "Sandboxing for LLM Agents"
date: "2026-07-14"
tags: ["llm", "agents", "sandbox"]
---

Modern LLM agents need to run code, inspect files, and interact with systems. How that environment is set up—what we call the **sandbox**—varies widely across frameworks and tools. The choice of sandbox shapes what an agent can do, how safely it can do it, and even which operating systems it runs well on.

## The Spectrum of Sandboxes

### Unrestricted Terminal Access

The simplest approach is no sandbox at all: give the agent direct access to a terminal and let it run shell commands freely. This is easy to set up and mirrors how developers actually work, but it carries obvious risks. A mistaken `rm -rf` or an overly broad `find` command can damage the host system.

### Remote Service Sandboxes

On the opposite end are fully managed remote sandboxes. The agent connects to a server that handles environment provisioning, resource limits, and security isolation. The framework does not need to worry about what runs where; it simply sends commands and receives results. This offloads complexity but introduces latency and dependency on an external service.

### Simulated Sandboxes

A middle ground is the **simulated sandbox**, where the agent believes it is interacting with a local filesystem and shell, but the commands are actually translated into API requests. For example, when the agent runs `ls`, `grep`, or `find`, a virtual filesystem layer intercepts the call and forwards it to a remote server. The agent gets the familiar bash experience without needing a real local environment.

## Operating System Matters

Sandboxes are not equally portable. Most implementations assume a Linux environment, and macOS is usually supported as a close second. Windows lags behind, which is worth noting because many developers work on Windows machines. OpenAI's Codex agent, for instance, has drawn complaints about its bash tool behaving poorly on Windows. This raises a real question: is Windows support an afterthought because most agent workloads target Linux servers, or is it a genuine gap that limits adoption?

## The "Everything Is Bash" Trend

A growing philosophy in agent design is to give the agent a single tool—bash—and let all other capabilities emerge as command-line utilities. Instead of adding specialized tools for file search, code analysis, or web fetching, the framework installs CLI programs and teaches the agent to invoke them through shell commands. This naturally ties back to sandbox design, because those CLI tools must exist in the sandbox environment, and their availability differs across platforms.

## Closing Thoughts

There is no one-size-fits-all sandbox for LLM agents. Direct terminal access is simple but risky. Remote services are safe but add friction. Simulated environments try to bridge the gap. Meanwhile, OS differences and the shift toward "everything is bash" continue to shape what agents can realistically do.
