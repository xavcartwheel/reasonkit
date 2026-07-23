from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Union

from ._sanitize import sanitize_final, strip_preamble, strip_post_answer_meta, strip_scaffold_lines
from ._utils import _looks_like_code, _specificity, _structure_score, _regressed
from .classifier import classify
from .branching import generate_approaches
from .critique import critique_all
from .merge import merge, deepen, refine_merge
from .codegen import code_pipeline
from .stop_conditions import should_stop
from .trace import Trace
from .errors import ConfigurationError, ModelCallError

logger = logging.getLogger(__name__)

LLMCallable = Callable[[str], str]
AsyncLLMCallable = Callable[[str], Awaitable[str]]
StreamingLLMCallable = Callable[[str], AsyncIterator[str]]

DEFAULT_CONFIG = {
    "max_cycles": 2,
    "branches": 3,   # single direct draft -> critique path
    "return_trace": False,
    "verbose": False,
    "test_hook": None,   # used by code verify pass
    "timeout": None,
    "retries": 0,
    "stream_chunk_mode": "sentence",
}


def _is_coroutine_function(fn: Callable) -> bool:
    # partial chains keep the underlying function on .func
    target = fn.func if isinstance(fn, functools.partial) else fn
    return inspect.iscoroutinefunction(target)


def _callable_kind(fn: Callable) -> str:
    # Return 'async-gen', 'async', or 'sync'.
    target = fn.func if isinstance(fn, functools.partial) else fn
    if inspect.isasyncgenfunction(target):
        return "async-gen"
    if inspect.iscoroutinefunction(target):
        return "async"
    return "sync"


def _validate_config(cfg: dict) -> None:
    branches = cfg.get("branches", DEFAULT_CONFIG["branches"])
    max_cycles = cfg.get("max_cycles", DEFAULT_CONFIG["max_cycles"])
    if not isinstance(branches, int) or branches < 1:
        raise ConfigurationError(f"branches must be an int >= 1, got {branches!r}")
    if not isinstance(max_cycles, int) or max_cycles < 1:
        raise ConfigurationError(f"max_cycles must be an int >= 1, got {max_cycles!r}")
    chunk = cfg.get("stream_chunk_mode", "sentence")
    if chunk not in ("once", "sentence"):
        raise ConfigurationError(
            f"stream_chunk_mode must be 'once' or 'sentence', got {chunk!r}"
        )
    retries = cfg.get("retries", 0)
    if not isinstance(retries, int) or retries < 0:
        raise ConfigurationError(f"retries must be an int >= 0, got {retries!r}")


@dataclass
class EnhanceResult:
    # Returned when return_trace=True. Plain string otherwise.

    answer: str
    trace: "Trace"
    mode: str
    stopped_reason: str

    def __str__(self) -> str:
        return self.answer


def enhance(fn: Union[LLMCallable, AsyncLLMCallable, StreamingLLMCallable], **config: Any):
    """Wrap fn with the ReasonKit reasoning pipeline. The wrapper keeps fn's shape.

    Parameters
    ----------
    fn : str -> str | async str -> str | async generator
        Your LLM callable. Must accept (prompt: str) and return/produce str.
    **config
        max_cycles (2), branches (3), return_trace (False), verbose (False),
        test_hook ((code)->list[str] for code mode), timeout (None), retries (0),
        stream_chunk_mode ("once"|"sentence").

    Returns
    -------
    A callable with the same signature as fn. When return_trace=True, each call
    returns an EnhanceResult with .answer, .trace, .mode, .stopped_reason.
    """
    cfg = {**DEFAULT_CONFIG, **config}
    _validate_config(cfg)
    kind = _callable_kind(fn)
    if kind == "async-gen":
        return _make_streaming_wrapper(fn, cfg)
    if kind == "async":
        return _make_async_wrapper(fn, cfg)
    return _make_sync_wrapper(fn, cfg)


# Internal async helpers -- everything runs through the wrapped fn.


def _to_async(fn):
    # Return an awaitable version of fn.
    if _is_coroutine_function(fn):
        return fn

    async def shim(prompt: str) -> str:
        return fn(prompt)

    return shim


async def _classify_call(afn, prompt, trace):
    return await classify(afn, prompt, trace)


async def _call(afn, stage, prompt, trace, *, critical=False, timeout=None, retries=0):
    last_exc: BaseException | None = None
    for attempt in range(1 + retries):
        try:
            coro = afn(prompt)
            if timeout is not None:
                coro = asyncio.wait_for(coro, timeout=timeout)
            result = await coro
            if isinstance(result, str) and not result.strip():
                last_exc = ValueError("wrapped fn returned empty response")
                trace.notes.append(f"{stage}: empty response (attempt {attempt})")
                continue
            recorded = result if isinstance(result, str) else str(result)
            trace.add_call(stage, prompt, recorded)
            return recorded
        except (OSError, RuntimeError, ValueError) as exc:
            last_exc = exc
            trace.notes.append(f"{stage}: call failed (attempt {attempt}): {exc}")
    if critical:
        raise ModelCallError(
            f"wrapped fn failed for critical stage '{stage}' after {1 + retries} attempts",
            stage=stage,
            cause=last_exc,
        )
    trace.notes.append(f"{stage}: degraded after {1 + retries} attempts")
    return ""


def _goal_and_assumptions(classification: dict, original_prompt: str = "") -> str:
    # Context block for downstream stages. Puts the user's actual prompt first so
    # stages reason from their real words, not the classifier's inferred goal/assumptions.
    lines = []
    if original_prompt:
        p = original_prompt.strip()
        if len(p) > 4000:
            p = p[:4000] + " …(truncated)"
        lines.append("The user's actual message:\n" + p)
    goal = classification.get("goal", "")
    assumptions = classification.get("assumptions", [])
    clarifying = classification.get("clarifying_questions", [])
    lines.append(f"Goal: {goal}" if goal else "Goal: (not specified)")
    if assumptions:
        lines.append("Assumptions to revisit: " + "; ".join(assumptions))
    if clarifying:
        lines.append("Open clarifying questions: " + "; ".join(clarifying))
    return "\n".join(lines)


async def _decision_pipeline(afn, prompt, classification, cfg, trace):
    # Decision path: classify -> (generate -> critique -> merge) loop, then
    # refine + deepen passes on the final answer.
    goal_ctx = _goal_and_assumptions(classification, original_prompt=prompt)
    branches = int(cfg["branches"])
    max_cycles = int(cfg["max_cycles"])
    verbose = cfg["verbose"]

    previous_issue_count: int | None = None
    previous_issues_flat: list[str] | None = None  # fed back into next generate
    final_answer = ""
    stopped_reason = "max_cycles"

    for cycle in range(1, max_cycles + 1):
        trace.cycles = cycle
        if verbose:
            logger.info("decision cycle %s/%s", cycle, max_cycles)

        approaches = await generate_approaches(
            afn, goal_ctx, branches, trace,
            previous_issues=previous_issues_flat,
        )
        # Cap at requested count — some models produce 11+ approaches.
        approaches = approaches[:branches]
        if not approaches:
            trace.notes.append("decision: no approaches generated; skipping cycle")
            continue

        issues = await critique_all(afn, approaches, goal_ctx, trace)

        # No issues found: return best approach + clarifying note (no merge/
        # deepen needed since the approach is already sound). Applies on
        # every cycle: cycle 1 starts fresh, cycle 2+ generates approaches
        # that already avoid the previous cycle's known flaws, so the best
        # individual approach is a sound stopping point.
        if all(len(i) == 0 for i in issues):
            best = max(
                approaches,
                key=lambda a: _specificity(a) + _structure_score(a) * 20,
            )
            note = _clarifier_note(classification)
            final_answer = best.strip() + note
            trace.notes.append(
                "decision: per-approach critique found no issues; "
                "returning best approach + clarifying note"
            )
            stopped_reason = "no_issues"
            break

        # Merge all approaches into a coherent answer (synthesizes best ideas).
        merged = await merge(afn, approaches, issues, goal_ctx, trace)
        final_answer = merged.strip() or approaches[0].strip()

        stop, reason = should_stop(cycle, max_cycles, issues, previous_issue_count)
        if stop:
            stopped_reason = reason
            break
        previous_issue_count = sum(len(i) for i in issues)
        # Feed critiques back into the next generation cycle.
        previous_issues_flat = [iss for approach_issues in issues for iss in approach_issues]

    if stopped_reason == "no_issues":
        trace.stopped_reason = stopped_reason
        return sanitize_final(final_answer)

    # Single refine pass: check merged answer for quality gaps.
    if final_answer:
        refined = await refine_merge(afn, final_answer, goal_ctx, trace)
        if refined.strip():
            final_answer = refined.strip()

    # Single deepen pass: force concretization on the final answer.
    if final_answer:
        deepened = await deepen(afn, final_answer, goal_ctx, trace)
        if deepened.strip():
            final_answer = deepened.strip()

    # Final sanitization pass: strip any leaked meta-commentary from the final
    # answer before returning it to the user.
    trace.stopped_reason = stopped_reason
    return sanitize_final(final_answer)


def _clarifier_note(classification: dict) -> str:
    # A 'Worth clarifying:' block from the classifier's questions/assumptions,
    # or '' if none.
    clar = [c for c in (classification.get("clarifying_questions") or []) if c.strip()]
    assume = [a for a in (classification.get("assumptions") or []) if a.strip()]
    items = clar or assume  # prefer explicit questions; fall back to assumptions
    if not items:
        return ""
    bullets = "\n".join(f"- {it.rstrip(".")}." for it in items[:3])
    return "\n\nWorth clarifying:\n" + bullets


async def _direct_pipeline(afn, prompt, classification, cfg, trace):
    # Answer once, giving fn the classifier's goal/assumptions up front so the
    # single draft already reflects that context. A refine pass then sharpens it.
    # Falls back to the raw draft if refine degrades.
    #
    # Note: the classifier already produced goal/assumptions/clarifying-questions.
    # We feed those into the *first* draft call rather than answering blind and
    # re-feeding context in the refine step -- that avoids a wasted blind draft.
    note = _clarifier_note(classification)
    goal_ctx = _goal_and_assumptions(classification, original_prompt=prompt)

    raw = await _call(
        afn, "answer", goal_ctx, trace, critical=True,
        timeout=cfg.get("timeout"), retries=cfg.get("retries", 0),
    )

    # Skip refine when the raw answer is substantive and the classifier
    # didn't flag missing info. The answer prompt already asks for honesty
    # about assumptions and uncertainty, and _clarifier_note handles the
    # "Worth clarifying" section. Refine's unique value -- catching
    # invented specifics -- is already covered by the answer prompt saying
    # "If something is unclear or underspecified, say so honestly. Do not
    # guess." A second look at a substantive, informed answer is unlikely
    # to catch anything the first pass didn't.
    if len(raw.strip()) > 150 and not classification.get("needs_clarification", False):
        trace.notes.append("direct: raw answer is substantive; skipping refine")
        trace.stopped_reason = "no_issues"
        return sanitize_final(raw) + note

    refined = await _refine(afn, raw, classification, cfg, trace)
    if refined:
        trace.stopped_reason = "no_issues"
        return refined + note
    trace.notes.append("direct mode: refine degraded; returning raw answer")
    trace.stopped_reason = "no_issues"
    return raw.strip() + note


async def _refine(afn, raw, classification, cfg, trace) -> str:
    from pathlib import Path

    prompt_path = Path(__file__).parent / "prompts" / "refine_direct.txt"
    template = prompt_path.read_text(encoding="utf-8")
    goal_ctx = _goal_and_assumptions(classification, original_prompt="")
    rendered = template.format(draft=raw, goal_and_assumptions=goal_ctx)
    refined = await _call(
        afn, "refine", rendered, trace, critical=False,
        timeout=cfg.get("timeout"), retries=cfg.get("retries", 0),
    )
    if not refined:
        return ""

    # Apply shared sanitization: preamble, scaffold lines, and post-answer meta.
    refined = strip_preamble(refined)
    refined = strip_scaffold_lines(refined)
    refined = strip_post_answer_meta(refined)

    return refined.strip()


async def _run_pipeline(afn, prompt, classification, cfg, trace) -> str:
    # Route to the matching pipeline. CODE with no usable code + a clarification
    # flag reroutes to DECISION so a vague "build me an app" gets a real answer.
    category = classification["category"]

    if category == "CODE":
        code_input = (classification.get("code_input") or "").strip()
        has_code = bool(code_input) and _looks_like_code(code_input)
        if not has_code and classification.get("needs_clarification", False):
            trace.notes.append(
                "routing: CODE with no usable code + needs_clarification; "
                "rerouting to DECISION"
            )
            classification = {**classification, "category": "DECISION"}
            category = "DECISION"
            trace.mode = "decision"

    if category == "DECISION":
        return await _decision_pipeline(afn, prompt, classification, cfg, trace)
    if category == "CODE":
        return await code_pipeline(
            afn, prompt, classification,
            int(cfg["max_cycles"]), cfg["test_hook"], trace,
        )
    return await _direct_pipeline(afn, prompt, classification, cfg, trace)


async def _full_run(afn, prompt, cfg, trace) -> str:
    # Classify, then run the matching pipeline.
    classification = await _classify_call(afn, prompt, trace)
    return await _run_pipeline(afn, prompt, classification, cfg, trace)


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #


def _make_sync_wrapper(fn, cfg):
    afn = _to_async(fn)

    @functools.wraps(fn)
    def wrapper(prompt: str, *args, **kwargs):
        trace = Trace()
        loop = asyncio.new_event_loop()
        try:
            answer = loop.run_until_complete(_full_run(afn, prompt, cfg, trace))
        finally:
            loop.close()

        if cfg["return_trace"]:
            return EnhanceResult(
                answer=answer, trace=trace, mode=trace.mode, stopped_reason=trace.stopped_reason
            )
        return answer

    return wrapper


def _make_async_wrapper(fn, cfg):
    afn = _to_async(fn)

    @functools.wraps(fn)
    async def wrapper(prompt: str, *args, **kwargs):
        trace = Trace()
        answer = await _full_run(afn, prompt, cfg, trace)
        if cfg["return_trace"]:
            return EnhanceResult(
                answer=answer, trace=trace, mode=trace.mode, stopped_reason=trace.stopped_reason
            )
        return answer

    return wrapper


def _split_sentences(text: str) -> list[str]:
    # Split into sentence-ish chunks for progressive streaming.
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _make_streaming_wrapper(fn, cfg):
    # Buffer the raw stream, run the full pipeline on it, then yield the refined
    # answer ('once' or 'sentence' mode). Degrades to raw if the pipeline fails.
    chunk_mode = cfg.get("stream_chunk_mode", "sentence")

    async def _collect(stream) -> str:
        chunks = []
        async for piece in stream:
            chunks.append(piece if isinstance(piece, str) else str(piece))
        return "".join(chunks)

    async def wrapper(prompt: str, *args, **kwargs):
        raw_stream = fn(prompt, *args, **kwargs)
        raw = await _collect(raw_stream)

        async def _afn(p: str) -> str:
            return await _collect(fn(p))

        async def afn_call(p: str) -> str:
            return await _call(
                _afn, "pipeline", p, Trace(),
                timeout=cfg.get("timeout"), retries=cfg.get("retries", 0),
            )

        trace = Trace()
        try:
            classification = await _classify_call(afn_call, prompt, trace)
            refined = await _run_pipeline(afn_call, prompt, classification, cfg, trace)
        except Exception as exc:  # noqa: BLE001 - degrade to raw on any failure
            trace.notes.append(f"streaming pipeline failed; yielding raw: {exc}")
            refined = raw

        if not refined:
            refined = raw

        if chunk_mode == "once":
            yield refined
        else:
            for sentence in _split_sentences(refined):
                yield sentence

    return wrapper
