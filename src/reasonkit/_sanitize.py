from __future__ import annotations

import re


# --------------------------------------------------------------------------- #
# Preamble patterns -- meta-preambles small models prepend despite the prompt
# telling them not to.
# --------------------------------------------------------------------------- #

_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"here(?:'s| is)|below is|this is|"
    r"i'?ve\s+(?:revised|improved|updated|corrected|rewritten|reworked)|"
    r"after\s+(?:considering|reviewing|reading|analyzing)|"
    r"based\s+on\s+(?:the\s+)?(?:critique|above|draft|issues|feedback)|"
    r"here'?s\s+my\s+(?:revised|improved|refined|better|updated)"
    r")\b[^\n:]{0,80}:\s*",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Scaffold labels from _goal_and_assumptions that leak into the final answer.
# --------------------------------------------------------------------------- #

_SCAFFOLD_LINE_RE = re.compile(
    r"^\s*(?:the user's actual message|assumptions to revisit|"
    r"open clarifying questions?|draft reply(?: \+ critique)?|"
    r"goal / things to keep in mind)\s*:.*$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Post-answer meta-commentary: after the actual answer, some models continue
# with reasoning about their own writing process ("We need to...", "We must...",
# "Let's produce..."). These phrases would never be in a real answer (they
# address the writer, not the user), so we can safely strip everything from the
# first such marker onwards.
# --------------------------------------------------------------------------- #

# "we (need to|must|should|can|can now) [not] <action verb>"
# Handles "we should not claim", "we can provide", "we can say", etc.
_META_WE_ACTION_RE = re.compile(
    r"\n{2,}we\s+(?:need\s+to|must|should|can(?:\s+now)?)\s+"
    r"(?:not\s+)?(?:produce|create|write|generate|output|"
    r"craft|do|make|put|try|go\s+ahead|wrap|start|give|summarize|"
    r"fix|repair|correct|revise|rewrite|rework|begin|improve|keep|"
    r"add|include|ask|ensure|mention|consider|note|be|talk|"
    r"say|provide|claim)\b.*",
    re.IGNORECASE | re.DOTALL,
)

# "we must preserve|keep|maintain|..." -- a different class of instruction
_META_WE_MUST_RE = re.compile(
    r"\n{2,}we\s+must\s+(?:preserve|keep|maintain|include|add|use|follow|adhere|"
    r"comply|obey|respect|retain|not|avoid|never)\b.*",
    re.IGNORECASE | re.DOTALL,
)

# "let's <action verb>" (with optional interjection prefix like "ok. let's")
_META_LET_RE = re.compile(
    r"\n{2,}(?:ok[a-z]*[.!?]+\s+)?let['´`]?s\s+(?:produce|output|write|craft|do|make|put|try|"
    r"go\s+ahead|wrap|start|give|summarize|begin|get|work|fix|create|"
    r"repair|correct|revise|rewrite|rework|answer)\b.*",
    re.IGNORECASE | re.DOTALL,
)

# "i/we (will|shall|'ll) [not|also|...] <action verb>"
_META_PERSONAL_WILL_RE = re.compile(
    r"\n{2,}(?:i|we)(?:['´`]?\s+(?:will|shall)|['´`]ll)\s+"
    r"(?:\w+\s+)*?(?:"
    r"output|produce|write|provide|give|craft|fix|revise|rewrite|"
    r"present|offer|explain|keep|mention|make|comply|answer|try|do|"
    r"add|include|ensure|note|consider|ask"
    r")\b.*",
    re.IGNORECASE | re.DOTALL,
)

# "however|but,? the instruction|the prompt|the rules|..."
_META_INSTRUCTION_RE = re.compile(
    r"\n{2,}(?:however|but)\s*,?\s*(?:the instruction|the prompt|the rules|"
    r"the guidelines|our task|i(?:'m|\s+am)\s+(?:supposed|asked|told|instructed)"
    r"|as\s+(?:an?\s+)?(?:AI|language\s+model|assistant)).*",
    re.IGNORECASE | re.DOTALL,
)

# Revision / process notes: "i've revised...", "after considering...", etc.
_META_REVISION_RE = re.compile(
    r"\n{2,}"
    r"(?:"
    r"i'?ve\s+(?:revised|improved|updated|corrected|rewritten|reworked)\s+"
    r"(?:the|this|it|above|answer|draft)"
    r"|after\s+(?:considering|reviewing|reading)\s+(?:the\s+)?"
    r"(?:critique|above|draft|issues|feedback|analysis)"
    r"|here'?s\s+my\s+(?:revised|improved|refined|better|updated|answer)"
    r"|the\s+(?:user|prompt|request|question)\s+(?:asks|wants|says|needs|requests)\s+(?:(?:me\s+)?to\b)?"
    r"|User\s+wants\s+(?:a|an|the|to|me)\b"
    r"|User's\s+(?:request|question|prompt)\s+"
    r"|(?:now|so)\s+(?:let\s+me|i['´`]?ll)\s+(?:provide|give|present|offer|"
    r"produce|write|craft|explain)"
    r"|i['´`]?ll\s+(?:provide|give|present|offer|produce|write|craft|explain|revise)"
    r"|my\s+(?:revised|improved|refined|final)\s+(?:answer|draft|version)"
    r").*",
    re.IGNORECASE | re.DOTALL,
)

# All post-answer meta patterns, applied in order.
_POST_ANSWER_PATTERNS = (
    _META_WE_ACTION_RE,
    _META_WE_MUST_RE,
    _META_LET_RE,
    _META_INSTRUCTION_RE,
    _META_REVISION_RE,
    _META_PERSONAL_WILL_RE,
)


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def strip_preamble(text: str) -> str:
    """Strip leading meta-preambles like 'Here is my revised answer:'."""
    if not text:
        return text
    return _PREAMBLE_RE.sub("", text, count=1)


def strip_scaffold_lines(text: str) -> str:
    """Strip lines that look like internal pipeline context labels (Goal:,
    Assumptions to revisit:, etc.)."""
    if not text:
        return text
    kept = [ln for ln in text.splitlines() if not _SCAFFOLD_LINE_RE.match(ln)]
    return "\n".join(kept).strip()


def strip_post_answer_meta(text: str) -> str:
    """Strip post-answer meta-commentary about the writing process.

    Applies all known meta-commentary patterns in order and returns the
    clean text. Each pattern strips from its match to the end of the string,
    so the first hit removes everything after it.
    """
    if not text:
        return text
    for pattern in _POST_ANSWER_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


def sanitize_final(text: str) -> str:
    """Full sanitization: preambles, scaffold lines, and post-answer meta.

    This is the main entry point for final-answer sanitization. Strips:
    1. Leading meta-preambles ("Here is my revised answer:")
    2. Internal scaffold labels ("Goal:", "Assumptions to revisit:")
    3. Post-answer commentary ("We need to fix...", "Let's rewrite...")
    """
    if not text:
        return text
    text = strip_preamble(text)
    text = strip_scaffold_lines(text)
    text = strip_post_answer_meta(text)
    return text.strip()
