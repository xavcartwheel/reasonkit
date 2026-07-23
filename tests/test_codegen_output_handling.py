# Code-output normalization tests for code mode.
from reasonkit.codegen import _normalize_code_output
from reasonkit._utils import _looks_like_code


def test_normalize_code_output_extracts_fenced_block():
    raw = (
        "Here is the corrected code:\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "This now works."
    )
    out = _normalize_code_output(raw)
    assert out.startswith("def add")
    assert "return a + b" in out
    assert "Here is the corrected code" not in out


def test_normalize_code_output_keeps_plain_note_when_no_code():
    raw = "The request is too vague. Please specify what X should do."
    out = _normalize_code_output(raw)
    assert out == raw
    assert _looks_like_code(out) is False
