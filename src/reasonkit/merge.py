from __future__ import annotations

import re
from pathlib import Path

from ._sanitize import sanitize_final, strip_preamble, strip_post_answer_meta
from ._utils import _specificity, _structure_score, _regressed
from .trace import Trace

_PROMPT_PATH = Path(__file__).parent / "prompts" / "merge.txt"
_DEEPEN_PATH = Path(__file__).parent / "prompts" / "deepen_concrete.txt"
_REFINE_MERGE_PATH = Path(__file__).parent / "prompts" / "refine_merge.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _load_deepen_prompt() -> str:
    return _DEEPEN_PATH.read_text(encoding="utf-8")


def _load_refine_merge_prompt() -> str:
    return _REFINE_MERGE_PATH.read_text(encoding="utf-8")


def build_merge_prompt(
    approaches: list[str],
    issues_by_approach: list[list[str]],
    goal_and_assumptions: str,
) -> str:
    single = len(approaches) == 1
    parts = []
    for i, (approach, issues) in enumerate(zip(approaches, issues_by_approach), start=1):
        issue_text = "\n".join(f"- {iss}" for iss in issues) if issues else "(no issues flagged)"
        label = "DRAFT REPLY" if single else f"APPROACH {i}"  # don't nudge toward strategy-deck output
        parts.append(f"{label}:\n{approach}\n\nCRITIQUE:\n{issue_text}")
    combined = "\n\n---\n\n".join(parts)
    return _load_prompt().format(
        approaches_with_critiques=combined,
        goal_and_assumptions=goal_and_assumptions,
    )


async def merge(
    fn,
    approaches: list[str],
    issues_by_approach: list[list[str]],
    goal_and_assumptions: str,
    trace: Trace,
) -> str:
    rendered = build_merge_prompt(approaches, issues_by_approach, goal_and_assumptions)
    raw = await fn(rendered)
    trace.add_call("merge", rendered, raw)
    merged = sanitize_final(raw)

    # Fallback: if the merge regressed vs the best single approach (known
    # small-model failure), ship the best approach instead.
    if approaches:
        best = max(approaches, key=lambda a: _specificity(a) + _structure_score(a) * 20)
        if _regressed(merged, best):
            trace.notes.append(
                "merge: merged output regressed vs best approach "
                f"(spec={_specificity(merged):.0f}/{_specificity(best):.0f}, "
                f"struct={_structure_score(merged):.0f}/{_structure_score(best):.0f}); "
                "falling back to best approach"
            )
            return best

    return merged


async def deepen(
    fn,
    answer: str,
    goal_and_assumptions: str,
    trace: Trace,
) -> str:
    # Deepen stage: force concretization of the final answer. Adds specific
    # examples, numbers, named trade-offs where the answer is generic.
    if not answer:
        return answer

    rendered = _load_deepen_prompt().format(
        answer=answer,
        goal_and_assumptions=goal_and_assumptions,
    )
    raw = await fn(rendered)
    trace.add_call("deepen", rendered, raw)
    result = sanitize_final(raw)

    # Never return empty — keep the original if deepen hollowed it out.
    if not result.strip():
        trace.notes.append("deepen: output was empty; keeping original answer")
        return answer

    # Regression guard: if the output is significantly shorter or structurally
    # weaker, the model probably dropped content instead of deepening it.
    if len(result) < len(answer) * 0.6 or _regressed(result, answer):
        trace.notes.append(
            "deepen: output regressed vs input; keeping original answer"
        )
        return answer

    return result


async def refine_merge(
    fn,
    answer: str,
    goal_and_assumptions: str,
    trace: Trace,
) -> str:
    # Refinement pass: check the merged answer for quality gaps (organization,
    # reasoning depth, assumptions, honesty) and strengthen them without
    # rewriting. Regression guard prevents degradation.
    if not answer:
        return answer

    rendered = _load_refine_merge_prompt().format(
        answer=answer,
        goal_and_assumptions=goal_and_assumptions,
    )
    raw = await fn(rendered)
    trace.add_call("refine_merge", rendered, raw)
    result = sanitize_final(raw)

    # Never return empty.
    if not result.strip():
        trace.notes.append("refine_merge: output was empty; keeping original answer")
        return answer

    # Regression guard: if the output degraded, keep the original.
    if len(result) < len(answer) * 0.6 or _regressed(result, answer):
        trace.notes.append(
            "refine_merge: output regressed vs input; keeping original answer"
        )
        return answer

    return result
