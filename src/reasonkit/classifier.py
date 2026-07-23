from __future__ import annotations

from pathlib import Path

from .trace import Trace

_PROMPT_PATH = Path(__file__).parent / "prompts" / "classify_and_assumptions.txt"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_classify_prompt(prompt: str) -> str:
    return _load_prompt().format(prompt=prompt)


# Structured-output field headers the classify prompt asks for. DIRECT_ANSWER is
# intentionally absent -- the classifier routes only.
_FIELD_NAMES = (
    "CATEGORY",
    "COMPLEXITY",
    "NEEDS_CLARIFICATION",
    "GOAL",
    "ASSUMPTIONS",
    "CLARIFYING_QUESTIONS",
    "CODE_INPUT",
)


def _read_field(text: str, field_name: str) -> str:
    # Extract FIELD_NAME: up to the next known field header. '' if absent.
    import re

    others = [f for f in _FIELD_NAMES if f != field_name]
    stop = r"|".join(re.escape(o) for o in others)
    pattern = (
        rf"^\s*{re.escape(field_name)}\s*:\s*(.*?)"
        rf"(?=^\s*(?:{stop})\s*:|\Z)"
    )
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _normalize_category(value: str) -> str:
    v = value.strip().upper()
    if v in ("DECISION", "CODE", "DIRECT"):
        return v
    if "CODE" in v:
        return "CODE"
    if "DECIS" in v:
        return "DECISION"
    return "DIRECT"  # safest fallback


def _normalize_complexity(value: str) -> str:
    return "COMPLEX" if "COMPLEX" in value.upper() else "SIMPLE"


def _normalize_needs_clarification(value: str) -> bool:
    v = value.strip().upper()
    if v.startswith("Y"):
        return True
    if v.startswith("N"):
        return False
    return "YES" in v and "NO" not in v  # tolerate free text


def _try_parse_json(text: str) -> dict | None:
    # Try to parse text as JSON. Handles ```json fences and bare objects.
    import json
    import re

    t = text.strip()

    # Strip ```json ... ``` fences if present.
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", t)
    if m:
        t = m.group(1).strip()

    # Strip outer code fences without language tag.
    m = re.search(r"```\s*\n?([\s\S]*?)```", t)
    if m:
        t = m.group(1).strip()

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Also try to find a top-level JSON object anywhere in the text (some models
    # wrap JSON in prose).
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def parse_classification(raw: str) -> dict:
    # Parse structured classify output into a dict.
    # 
    # Handles BOTH formats models actually output:
    #   1. Field-per-line  (CATEGORY: DECISION\\nGOAL: ...)
    #   2. JSON             ({"CATEGORY": "DECISION", "goal": "..."})
    # The classify prompt asks for format 1, but many models (codestral, deepseek)
    # emit JSON, sometimes inside a ```json code block. JSON keys can be uppercase
    # (CATEGORY) or lowercase (category); we accept both.
    import re

    # Try JSON parsing first (handles ```json blocks, bare JSON, prose-wrapped).
    json_obj = _try_parse_json(raw)

    if json_obj is not None:
        # Normalise: accept both "CATEGORY" and "category" as keys.
        def _jget(*keys: str) -> str:
            for k in keys:
                v = json_obj.get(k)
                if v is not None and v != "":
                    return str(v).strip()
            return ""

        def _jget_list(*keys: str) -> list[str]:
            for k in keys:
                v = json_obj.get(k)
                if isinstance(v, list):
                    return [str(x).strip() for x in v if x]
            return []

        category = _normalize_category(_jget("CATEGORY", "category"))
        complexity = _normalize_complexity(_jget("COMPLEXITY", "complexity"))
        needs_clar = _jget("NEEDS_CLARIFICATION", "needs_clarification")
        assumptions = _jget_list("ASSUMPTIONS", "assumptions")
        clarifying = _jget_list("CLARIFYING_QUESTIONS", "clarifying_questions")
        goal = _jget("GOAL", "goal")
        code_input = _jget("CODE_INPUT", "code_input")

        if not code_input:
            code_input = _read_field(raw, "CODE_INPUT")  # fallback: regex

        return {
            "category": category,
            "complexity": complexity,
            "needs_clarification": _normalize_needs_clarification(needs_clar),
            "goal": goal,
            "assumptions": assumptions if assumptions else _split_list(_read_field(raw, "ASSUMPTIONS")),
            "clarifying_questions": clarifying if clarifying else _split_list(_read_field(raw, "CLARIFYING_QUESTIONS")),
            "code_input": code_input,
        }

    # Fall back to line-based field parsing (format 1 in the prompt).
    category = _normalize_category(_read_field(raw, "CATEGORY"))
    complexity = _normalize_complexity(_read_field(raw, "COMPLEXITY"))
    assumptions = _split_list(_read_field(raw, "ASSUMPTIONS"))
    clarifying = _split_list(_read_field(raw, "CLARIFYING_QUESTIONS"))
    code_input = _read_field(raw, "CODE_INPUT")

    return {
        "category": category,
        "complexity": complexity,
        "needs_clarification": _normalize_needs_clarification(
            _read_field(raw, "NEEDS_CLARIFICATION")
        ),
        "goal": _read_field(raw, "GOAL"),
        "assumptions": assumptions,
        "clarifying_questions": clarifying,
        "code_input": code_input,
    }


def _split_list(value: str) -> list[str]:
    # Split a free-form field into list items on newlines or ';' or '-' bullets.
    if not value:
        return []
    items = []
    for part in value.replace(";", "\n").split("\n"):
        part = part.strip().lstrip("-").strip()
        part = part.lstrip("0123456789").lstrip(".)").strip()
        if part:
            items.append(part)
    return items


async def classify(fn, prompt: str, trace: Trace) -> dict:
    rendered = build_classify_prompt(prompt)
    raw = await fn(rendered)
    trace.add_call("classify", rendered, raw)
    result = parse_classification(raw)
    trace.classification = result
    trace.mode = result["category"].lower()
    if result["category"] not in ("DECISION", "CODE", "DIRECT"):
        trace.notes.append("classify: unrecognized category, fell back to DIRECT")
    return result
