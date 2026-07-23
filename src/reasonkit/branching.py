from __future__ import annotations

import re
from pathlib import Path

from ._sanitize import strip_post_answer_meta
from .trace import Trace

_PROMPT_PATH = Path(__file__).parent / "prompts" / "generate_approaches.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_generate_prompt(
    goal_and_assumptions: str,
    n: int,
    previous_issues: list[str] | None = None,
) -> str:
    if n > 1:
        instruction = (
            f"Generate {n} GENUINELY DISTINCT approaches to address it -- real "
            "alternative strategies, not rephrasings of each other. Each must "
            "take a different angle (e.g. different audience, different risk "
            "posture, different first move)."
        )
        fmt = (
            "Label each approach exactly 'APPROACH 1:', 'APPROACH 2:', etc., in "
            "order, with a blank line between them. Do not wrap them in extra "
            "sections or a summary."
        )
        # Feed previous-cycle critique back into generation so new approaches
        # avoid known flaws.
        if previous_issues:
            bullets = "\n".join(f"- {iss}" for iss in previous_issues[:8])
            instruction += (
                "\n\n**Avoid previous mistakes.** The previous cycle's approaches "
                "had these issues. Your NEW approaches must NOT repeat them:\n"
                f"{bullets}"
            )
    else:
        instruction = (
            "Write ONE direct, human reply to that message in plain language "
            "(the way a knowledgeable friend would answer)."
        )
        fmt = "Write the reply as a single continuous answer. No 'APPROACH' labels."
    return _load_prompt().format(
        goal_and_assumptions=goal_and_assumptions,
        n_instruction=instruction,
        n_format=fmt,
    )


def _split_approaches(raw: str) -> list[str]:
    # Split the numbered output into distinct approach blocks.
    parts = re.split(r"(?im)^\s*APPROACH\s*\d+\s*[:.)-]?\s*", raw)
    approaches = [p.strip() for p in parts[1:] if p.strip()]
    if not approaches:
        return [raw.strip()] if raw.strip() else []  # no markers: one approach
    # Strip meta-commentary that leaks into any approach.
    return [strip_post_answer_meta(a) for a in approaches]


async def generate_approaches(
    fn,
    goal_and_assumptions: str,
    n: int,
    trace: Trace,
    previous_issues: list[str] | None = None,
) -> list[str]:
    rendered = build_generate_prompt(goal_and_assumptions, n, previous_issues=previous_issues)
    raw = await fn(rendered)
    trace.add_call("generate", rendered, raw)
    return _split_approaches(raw)
