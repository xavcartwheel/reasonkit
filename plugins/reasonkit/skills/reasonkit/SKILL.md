---
name: reasonkit
description: Improve an answer with classification, critique, and self-revision via the reasonkit library, using Claude Code itself as the model.
---

# ReasonKit

Use when the user wants a more thorough, self-checked answer to a decision,
planning, or code question.

ReasonKit wraps a `call_llm(prompt) -> str` with a short pipeline and runs it
through that same function:

- **decision**: classify -> generate approaches -> critique -> merge
- **code**: generate -> verify -> fix
- **direct**: answer -> refine

## How to run

The plugin script uses Claude Code (`claude -p`) as the wrapped `call_llm`, so
no API key is needed:

```
python "${CLAUDE_PLUGIN_ROOT}/scripts/reasonkit_run.py" "<the user's prompt>"
```

It prints the final answer to stdout. Requires `pip install reasonkit` and the
`claude` CLI on PATH.
