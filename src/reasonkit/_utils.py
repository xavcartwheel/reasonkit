from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# Code-detection helpers (extracted from codegen.py to break circular imports)
# --------------------------------------------------------------------------- #

def _line_looks_code(line: str) -> bool:
    """Heuristic: does a single line look like source code?"""
    low = line.strip().lower()
    if not low:
        return False
    starters = (
        "def ", "class ", "if ", "for ", "while ", "try:", "except", "return ",
        "import ", "from ", "#include", "public ", "private ", "fn ", "let ",
        "const ", "var ", "function ", "package ", "interface ", "struct ",
    )
    if any(low.startswith(s) for s in starters):
        return True
    return any(ch in line for ch in ("{", "}", "(", ")", "=>", "::", ";", "[", "]"))


def _looks_like_code(text: str) -> bool:
    """Heuristic: is this source code, or prose the classifier dumped into
    CODE_INPUT? A "the user did not provide code" sentence must not be treated
    as code to verify/fix -- that's what makes the pipeline hallucinate."""
    if not text:
        return False
    if "```" in text or text.lstrip().startswith("#!"):
        return True
    low = text.lower()
    seen = sum(
        1
        for kw in (
            "def ", "class ", "func ", "function ", "public ", "private ",
            "void ", "int ", "return ", "import ", "include", "package ",
            "fn ", "let ", "var ", "const ", "struct ", "end", "begin",
        )
        if kw in low
    )
    prose_markers = sum(
        1 for m in ("note:", "since the user", "please provide", "it's essential")
        if m in low
    )
    # Need >= 2 code-ish keywords and no prose markers, so a plain sentence isn't
    # misread as code but short snippets (def add(a,b): return a+b) still pass.
    return seen >= 2 and prose_markers == 0


# --------------------------------------------------------------------------- #
# Heuristic scoring helpers (extracted from merge.py)
# --------------------------------------------------------------------------- #

def _specificity(text: str) -> float:
    """Cheap proxy for how concrete an answer is: length + developed points +
    a bottom-line marker. Used only to catch clear merge regressions, not to
    judge quality in general."""
    if not text:
        return 0.0
    low = text.lower()
    sentences = text.count(".") + text.count("!") + text.count("?")
    bottom_line = 1.0 if any(
        k in low for k in ("bottom line", "recommend", "in short", "overall,")
    ) else 0.0
    return float(len(text)) + sentences * 8.0 + bottom_line * 40.0


# Markers a well-organized answer carries (headings, bullets, ordinals, bold).
# A merge that collapses a 40-section draft into 4 items has lost structure even
# if still "long enough"; _structure_score catches that where length alone won't.
_STRUCTURE_RE = re.compile(
    r"^\s*(?:"
    r"#{1,6}\s+"
    r"|[-*]\s+\S"
    r"|\d+[.)]\s+\S"
    r"|\*\*[^*]+\*\*"
    r"|(?i:first|second|third|fourth|fifth|next|then|finally|lastly)\b[,:.\s]"
    r")",
    re.MULTILINE,
)


def _structure_score(text: str) -> float:
    """Count of structural markers per line. 0 for a flat prose blob."""
    if not text:
        return 0.0
    return float(len(_STRUCTURE_RE.findall(text)))


# Flag only clear regressions: merged below 0.5x on either axis, or below 0.7x
# on both at once. A merge that tightens a few sentences but keeps structure
# stays strictly better and is not flagged.
def _regressed(merged: str, best: str) -> bool:
    if not merged:
        return True
    spec_ratio = _specificity(merged) / max(_specificity(best), 1.0)
    struct_ratio = _structure_score(merged) / max(_structure_score(best), 1.0)
    if spec_ratio < 0.5 or struct_ratio < 0.5:
        return True
    if spec_ratio < 0.7 and struct_ratio < 0.7:
        return True
    return False
