from __future__ import annotations

import re
from pathlib import Path
from typing import Awaitable, Callable, Optional, Union

from ._utils import _line_looks_code, _looks_like_code
from .critique import _extract_bullets
from .stop_conditions import should_stop
from .trace import Trace

TestHook = Callable[[str], list[str]]
LLMCallable = Callable[[str], str]

_GEN_PATH = Path(__file__).parent / "prompts" / "generate_code.txt"
_FIX_PATH = Path(__file__).parent / "prompts" / "fix_code.txt"
_VERIFY_PATH = Path(__file__).parent / "prompts" / "verify_code.txt"


def _load(path: Path) -> str:
    return path.read_text(encoding="utf-8")


_FENCED_CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_+.#-]+)?\s*\n([\s\S]*?)```", re.MULTILINE)


def _normalize_code_output(text: str) -> str:
    # Extract code when a model wraps it in markdown/prose: largest fenced
    # block, else the original text trimmed.
    t = (text or "").strip()
    if not t:
        return ""

    blocks = [b.strip() for b in _FENCED_CODE_RE.findall(t) if b.strip()]
    if blocks:
        return max(blocks, key=len)

    # Some models prepend one explanatory line before unfenced inline code.
    lines = t.splitlines()
    for i, ln in enumerate(lines):
        if _line_looks_code(ln):
            tail = "\n".join(lines[i:]).strip()
            if _looks_like_code(tail):
                return tail
            break

    return t


def build_generate_code_prompt(prompt: str) -> str:
    return _load(_GEN_PATH).format(prompt=prompt)


def build_fix_prompt(code: str, issues: list[str]) -> str:
    issue_text = "\n".join(f"- {iss}" for iss in issues) if issues else "(none)"
    return _load(_FIX_PATH).format(code=code, issues=issue_text)


def build_verify_prompt(code: str, test_results: Optional[str] = None) -> str:
    test_block = f"Test results: {test_results}\n" if test_results else ""
    return _load(_VERIFY_PATH).format(code=code, test_results=test_block)


async def generate_code(fn, prompt: str, trace: Trace) -> str:
    rendered = build_generate_code_prompt(prompt)
    raw = await fn(rendered)
    trace.add_call("generate", rendered, raw)
    normalized = _normalize_code_output(raw)
    return normalized if _looks_like_code(normalized) else raw.strip()


async def verify(
    fn,
    code: str,
    test_hook: Optional[TestHook],
    trace: Trace,
    test_results: Optional[str] = None,
) -> list[str]:
    # The verify pass: a plain list of concrete issues (never a score). With a
    # test_hook, failures come from real tests; otherwise an LLM self-review call.
    if test_hook is not None:
        issues = list(test_hook(code))
        trace.add_call(
            "verify", f"<test_hook({len(code)} chars)>", str(issues), source="test_hook"
        )
        return issues

    normalized = _normalize_code_output(code)
    code_to_verify = normalized if _looks_like_code(normalized) else code
    rendered = build_verify_prompt(code_to_verify, test_results)
    raw = await fn(rendered)
    trace.add_call("verify", rendered, raw, source="llm")
    return _extract_bullets(raw)


async def fix(fn, code: str, issues: list[str], trace: Trace) -> str:
    rendered = build_fix_prompt(code, issues)
    raw = await fn(rendered)
    trace.add_call("fix", rendered, raw)
    normalized = _normalize_code_output(raw)
    return normalized if _looks_like_code(normalized) else raw.strip()


def _is_refusal_or_note(text: str) -> bool:
    # True when generation returned a plain 'I need more info' note instead of
    # code. Feeding that into verify->fix would make the model confabulate a
    # program, so we stop and return the honest note instead.
    if not text or _looks_like_code(text):
        return False
    low = text.lower()
    signals = (
        "too vague", "vague", "need more", "need to know", "not specified",
        "unspecified", "missing", "unclear", "can't", "cannot", "unable to",
        "provide", "clarif", "what x", "what does", "no specific",
        "don't know", "do not know", "more detail", "more information",
    )
    return any(s in low for s in signals)


async def code_pipeline(
    afn: Union[LLMCallable, Callable[[str], Awaitable[str]]],
    prompt: str,
    classification: dict,
    max_cycles: int,
    test_hook: Optional[TestHook],
    trace: Trace,
) -> str:
    # Generate (once) -> Verify -> Fix loop. If the request is underspecified,
    # generation returns an honest note and we stop -- no fabricated program.
    code_input = (classification.get("code_input") or "").strip()

    if code_input and _looks_like_code(code_input):
        trace.notes.append("code mode: user-supplied code; skipping generate")
        code = code_input
    else:
        if code_input:
            # Codegen prompt got prose (not code); ignore it and generate from
            # the prompt rather than guessing from a note.
            trace.notes.append(
                "code mode: CODE_INPUT was prose, not code; generating from prompt"
            )
        code = await generate_code(afn, prompt, trace)

    trace.code_versions.append(code)
    trace.mode = "code"

    if _is_refusal_or_note(code):
        trace.notes.append(
            "code mode: generation returned a clarification note, not code; "
            "stopping instead of guessing"
        )
        trace.stopped_reason = "needs_clarification"
        questions = classification.get("clarifying_questions") or []
        if questions:
            note = code.rstrip() + "\n\nWhat I need to proceed:\n" + "\n".join(
                f"- {q.rstrip(".")}." for q in questions[:3] if q.strip()
            )
            return note
        return code

    previous_issue_count: Optional[int] = None
    stopped_reason = "max_cycles"

    for cycle in range(1, max_cycles + 1):
        trace.cycles = cycle
        issues = await verify(afn, code, test_hook, trace)

        stop, reason = should_stop(cycle, max_cycles, [issues], previous_issue_count)
        if stop:
            stopped_reason = reason
            break

        fixed = await fix(afn, code, issues, trace)

        # Guard: if fix returned non-code (issue bullets, prose, meta-commentary),
        # keep the original code and stop -- prevents verify->fix cascading where
        # the model keeps outputting issues instead of fixed code each cycle.
        if _looks_like_code(fixed):
            code = fixed
        else:
            trace.notes.append(
                "code mode: fix output was not code; keeping previous version and stopping"
            )
            trace.stopped_reason = "no_improvement"
            break

        trace.code_versions.append(code)
        previous_issue_count = len(issues)

    trace.stopped_reason = stopped_reason
    return code
