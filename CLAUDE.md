# Empório da Música — customer service agent

Python text-message agent that answers customers of Empório da Música, a fictional musical-instrument store in Campo Grande/MS, using the store's operational data (`data/*.csv`) and its policy manual (`data/store_policies.pdf`).

## Workflow

- Plan before building. Technology, framework, model and architecture are chosen and documented in the README before any implementation code.
- Build in small blocks, one file at a time. Never drop hundreds of lines at once: every block must be small enough for me to read and fully understand.
- Before writing code, explain what will be done and why. After, summarize what was done.
- Every technical decision needs a justification (why this value, this library, this approach).
- Do not move to the next step until I say so.
- Every agreed decision is recorded in full in `decisions_details.md` (local, not versioned) before moving on; the README keeps only the essentials.

## Code rules

1. **Readability comes first**, above performance, cleverness or brevity. Names say where a value comes from and what it does. Prefer separate, named steps over clever one-liners. Code reads on its own, without large comment blocks.
2. **One function, one job.** Each function does exactly one thing. A function that only orchestrates others (call step A, then B, then C) is doing one thing: orchestrating.
3. **No loose code at module level.** Importing a module must never execute anything (no I/O, no reading `.env`, no creating clients). Module level holds only imports, class and function definitions, constants (fixed values that never change, in `UPPER_CASE`) and, in entry-point scripts, the `if __name__ == "__main__":` guard calling `main()`. Configuration, meaning anything read from the environment or from files, is loaded by a function.
4. **Short functions:** 40 lines maximum, counted from `def` to the function's last line.
5. **One-line docstring** at the start of every function, summarizing what it does. A module docstring at the top of the file is fine.
6. **No improvised, fragile code.** Nothing written in a rush or patched around: if something does not fit, fix the design. Each pipeline step is explicit, has a clear input and output, and can change without breaking the others.
7. **Clean architecture.** Folders and files are separated by responsibility; unrelated concerns never share a file.
8. **Type hints** on every function's parameters and return value, so what goes in and what comes out is readable without opening the function body.

## Language

- Code, file names, README, docs and commit messages: English.
- Everything the store's customer sees (agent replies, example conversations): Brazilian Portuguese.
- "Empório da Música" is a proper noun and is never translated.
- Use exact technical terms, consistently across code, README and explanations. No loose analogies: name what a thing is, and what it is not when confusion is likely.

## Commits and review

- Commit at the end of each significant block or file, keeping a real incremental history.
- Before each commit:
  1. Run a separate review subagent on the diff, giving it only the code (not the reasoning behind it) so it brings an independent perspective. I decide which findings get fixed.
  2. Check the diff for secrets (API keys, `.env` content) and personal data. When in doubt, do not commit.
