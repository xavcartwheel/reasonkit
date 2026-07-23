from __future__ import annotations

import re
from pathlib import Path

from ._sanitize import strip_post_answer_meta
from .trace import Trace

_PROMPT_PATH = Path(__file__).parent / "prompts" / "critique_all.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_critique_prompt(approaches: list[str], goal_and_assumptions: str) -> str:
    numbered = "\n\n".join(
        f"APPROACH {i + 1}:\n{a}" for i, a in enumerate(approaches)
    )
    return _load_prompt().format(approaches=numbered, goal_and_assumptions=goal_and_assumptions)


def _split_critique(raw: str, n_approaches: int) -> list[list[str]]:
    # Parse critique output into one issue-list per approach (length n_approaches).
    if n_approaches <= 1:
        body = re.sub(r"(?im)^\s*ISSUES\s*:\s*", "", raw, count=1)
        return [_extract_bullets(body)]

    blocks = re.split(r"(?im)(?:^\s*\d+\s*[\.\)]\s*)?\**\s*APPROACH\s*\d+\s*[:.)-]?\s*\**", raw)
    raw_blocks = blocks[1:] if len(blocks) > 1 else []

    if not raw_blocks:
        shared = _extract_bullets(re.sub(r"(?im)^\s*ISSUES\s*:\s*", "", raw, count=1))
        return [list(shared) for _ in range(n_approaches)]

    result: list[list[str]] = []
    for i in range(n_approaches):
        block = raw_blocks[i] if i < len(raw_blocks) else ""
        result.append(_extract_bullets(block))
    return result


_NONE_MARKERS = {"none", "no issues", "no issue", "empty", "n/a", "(none)", "nil"}


def _normalize(token: str) -> str:
    # Lowercase + strip surrounding punctuation for none-marker comparison.
    return token.strip().lower().strip(".!?;:-")


def _extract_bullets(block: str) -> list[str]:
    # Pull issue items out of a block of free-form text.
    block = block.strip()
    if not block:
        return []
    # Strip leading "ISSUES:" header so "(none)" / "none" is recognized below.
    block = re.sub(r"(?im)^\s*ISSUES\s*:\s*", "", block, count=1).strip()
    if _normalize(block) in _NONE_MARKERS:
        return []
    # Strip post-ISSUES meta-commentary that small models sometimes add.
    block = strip_post_answer_meta(block)
    items = []
    for line in block.replace(";", "\n").split("\n"):
        raw_line = line.strip()
        if re.match(r"^\*?\*?APPROACH\s+\d+", raw_line, re.IGNORECASE):
            continue  # drop leaked bold approach-title lines
        line = raw_line.lstrip("-*•").strip()
        line = line.lstrip("0123456789").lstrip(".)").strip().strip("*").strip()
        if line and _normalize(line) not in _NONE_MARKERS:
            items.append(line)
    return items

async def critique_all(
    fn, approaches: list[str], goal_and_assumptions: str, trace: Trace
) -> list[list[str]]:
    rendered = build_critique_prompt(approaches, goal_and_assumptions)
    raw = await fn(rendered)
    trace.add_call("critique", rendered, raw, n_approaches=len(approaches))
    return _split_critique(raw, len(approaches))
