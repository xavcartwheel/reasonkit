#!/usr/bin/env python3
# Run a prompt through ReasonKit using Claude Code as the model.
from __future__ import annotations

import subprocess
import sys

import reasonkit


def call_llm(prompt: str) -> str:
    # Claude Code is already authenticated; -p runs one non-interactive turn.
    out = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() or sys.stdin.read().strip()
    if not prompt:
        print("usage: reasonkit_run.py <prompt>", file=sys.stderr)
        return 2
    smart = reasonkit.enhance(call_llm, max_cycles=2, branches=3)
    print(smart(prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
