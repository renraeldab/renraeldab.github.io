---
title: "Terminal, Shell, CLI, and TUI: What Is What"
date: 2026-07-25
tags: ["terminal", "shell", "cli", "beginner"]
---

This post explains the difference between a terminal, a shell, and a CLI; why the same command can behave differently on different operating systems; and how a script file becomes a command you can run by name.

## A Brief History: CLI, GUI, and the Comeback

In the 1960s and 1970s, the command-line interface (CLI) was the only way to interact with a computer. Operators typed commands into teletypes — literally typewriters wired to mainframes — and read the printed output. There were no windows, icons, or menus.

The 1980s and 1990s brought the graphical user interface (GUI) revolution. Macintosh and Windows made computers accessible to non-experts by replacing typed commands with point-and-click interactions. For decades, the GUI was the default interface for mainstream users, while the CLI remained a niche tool for developers and system administrators.

The CLI never disappeared for developers because it is efficient and composable: you can chain small tools together to perform complex tasks. But in the 2020s, the CLI is experiencing a broader comeback. AI agent tools, such as Claude Code and Copilot CLI, present themselves as "terminal apps" with rich, interactive interfaces. Newcomers now encounter powerful CLI tools before they understand the stack beneath them, which creates confusion.

## The Three Layers

When you open a terminal window and type `ls`, three different pieces of software are working together. Understanding which does what is the key to understanding everything else.

### Terminal Emulator

The terminal emulator is the GUI application you are looking at: Terminal.app, iTerm2, Windows Terminal, Ghostty, Alacritty, or WezTerm. It is responsible for rendering text on the screen, handling your keyboard input, managing the scrollback buffer, and communicating with the underlying program via a pseudo-terminal (PTY).

Think of the terminal emulator as a web browser. It displays content and captures your input, but it does not interpret the commands itself.

### Shell

The shell is the interpreter that runs inside the terminal emulator: Bash, Zsh, Fish, or PowerShell. It reads the text you type, parses it, and translates it into system calls. When you type `ls`, the shell understands that you want to list directory contents, finds the `ls` executable, and runs it.

You can change the shell without changing the terminal emulator. On macOS, Terminal.app can run Zsh, Bash, or Fish. On Windows, Windows Terminal can run PowerShell, Command Prompt, or WSL shells. The shell is a separate program.

### CLI

CLI is not a program. It is a pattern — an interface style where you communicate with software through text commands and receive text output. Both the terminal emulator and the shell participate in providing a CLI, but neither is the CLI. It is the contract between you and the software: text in, text out.

## Terminal Emulators Are Not Interchangeable

Here is where the practical differences come from. Terminal emulators implement feature sets differently. Modern terminal emulators support advanced escape sequences, true color, GPU-accelerated rendering, and special protocols for images and clipboard access. Older or built-in terminals often do not.

### Why Paste Image Works on macOS but Not on Windows

When you paste an image into a terminal, the terminal emulator must know what to do with non-text data. Modern terminal emulators on macOS implement protocols that allow applications to receive or display image data — or at least handle the paste gracefully without garbling the bytes. Historically, the legacy Windows console host (`conhost.exe`) treated the terminal as a strict text environment and did not support such protocols. Microsoft has since released Windows Terminal, which closes much of this gap, but the difference persists because terminal emulator capabilities are not universal.

The key point: this is a terminal emulator difference, not a shell difference. Bash on Windows Terminal may handle the paste better than Bash on the legacy console host, even with the same shell.

### Why Agent CLI Apps Suggest Ghostty

AI agent tools often render rich interfaces: colored panels, progress bars, live-updating text, and keyboard shortcuts. These are not plain text streams; they require the terminal emulator to support specific escape sequences for cursor movement, color, and screen clearing. They also benefit from fast rendering to feel responsive.

Built-in terminals like Terminal.app or the legacy Windows console host either lack support for certain escape sequences or render them slowly. Modern terminals like Ghostty, Alacritty, and WezTerm are built to handle these features efficiently, often using GPU acceleration. When an agent app suggests installing Ghostty, it is not because your old terminal is "broken" — it is because the app was built assuming a modern terminal emulator that speaks the same dialect of escape sequences.

## Streaming Text vs. Terminal UI

What is the difference between a basic Python script that uses `input()` and `print()`, and a "CLI app" with colors, tables, and progress bars?

A simple Python program is a CLI program in the strictest sense:

```python
name = input("Enter your name: ")
print(f"Hello, {name}")
```

This streams text line by line. The terminal prints what the program outputs and sends keyboard input back. There is no screen manipulation; the cursor simply moves down.

A colorful CLI app — built with libraries like Rich, Textual, or Bubble Tea — is technically a TUI (Terminal User Interface). It uses escape sequences to move the cursor around, clear lines, change colors, and draw boxes. The terminal emulator interprets these sequences and updates the screen accordingly.

```python
from rich.progress import track
import time

for step in track(range(100), description="Processing..."):
    time.sleep(0.05)
```

Both programs run inside the same terminal emulator and shell. The difference is what the program emits: plain text versus escape sequences that manipulate the terminal screen.

## Making a Script Into a Command

When you type `ls`, the shell finds a compiled executable and runs it. But you can also turn a text file into a command. On macOS and Linux, a script needs two things.

First, the file must begin with a **shebang** line that tells the shell which interpreter to use:

```python
#!/usr/bin/env python3
name = input("Enter your name: ")
print(f"Hello, {name}")
```

Second, the file needs the **executable permission**. You grant it with `chmod +x hello.py`. Once the file is executable and located in a directory listed in your `PATH`, you can type `hello.py` and the shell will run it like any other command.

Notice that the command name includes the `.py` extension only because the file is named that way. On Unix-like systems, the shell does not require an extension. You can rename the file to plain `hello` (no extension) and run it with `hello` as long as the shebang line is still inside. This is why commands you install via npm or Homebrew appear as clean names like `eslint` or `fzf` rather than `eslint.js` or `fzf.rb`. Package managers typically place a small wrapper script or a symlink with a clean name into a directory on your `PATH`, while the actual implementation files live elsewhere.

This mechanism does not work on Windows. Windows decides how to run a file by its extension (`.py`, `.bat`, `.exe`) and does not use the shebang line or the executable permission bit. On Windows, you typically run a Python script by typing `python hello.py` or rely on file associations rather than making the script itself a standalone command.

This is why setup instructions for command-line tools often look different across operating systems: the Unix world expects a script to become a first-class command, while Windows handles the same task through different conventions.

## Summary

- A **terminal emulator** is the application window that renders text and captures keystrokes. It is not the same as the shell.
- A **shell** is the interpreter that reads what you type, parses it, and runs programs. You can swap shells without changing terminal emulators.
- **CLI** is the interface pattern itself — text commands in, text output out — not a specific program.
- Terminal emulators differ in capabilities. Rich agent apps and TUIs need modern terminals that support advanced escape sequences and fast rendering.
- A **TUI** differs from a simple CLI program by using escape sequences to manipulate the screen — colors, tables, and live updates — rather than just streaming plain text.
- On Unix-like systems, a script becomes a command through a shebang line, executable permissions, and placement on the `PATH`. Windows uses file extensions and associations instead, which is why the same tool often has different setup instructions per operating system.
