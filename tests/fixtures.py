# Fake stub `call_llm` for tests -- deterministic canned responses per stage.
#
# No real API, no network, no key. The stub inspects the prompt text to decide
# which canned response to return, mimicking what a real model would emit for
# each pipeline stage. This lets the automated suite verify orchestration logic:
# call counts, stop conditions, structured-output parsing, trace shape.
from __future__ import annotations

from typing import Any, Awaitable, Callable, Union


# --------------------------------------------------------------------------- #
# Fixture prompts: Decision-mode ones (SaaS, Coffee, Plan) and C++ code-mode
# --------------------------------------------------------------------------- #

FIXTURE_SAAS = "build me a SaaS that will make me a millionaire"
FIXTURE_COFFEE = "should I start a coffee shop?"
FIXTURE_PLAN = "help me assess this plan for a project"
FIXTURE_CPP = "generate a C++ app that does X"

# Placeholder context for the "assess this plan" fixture.
PLACEHOLDER_PLAN = (
    "Plan: Launch a mobile app for local event discovery within 3 months, "
    "no marketing budget, solo founder, targeting a city of 200k people."
)


# --------------------------------------------------------------------------- #
# Canned responses per stage
# --------------------------------------------------------------------------- #

CLASSIFY_DECISION = """CATEGORY: DECISION
COMPLEXITY: COMPLEX
NEEDS_CLARIFICATION: NO
GOAL: Maximize the chance of building a profitable, durable business.
ASSUMPTIONS:
1. That a million dollars is a realistic near-term outcome rather than a vanity target.
2. That "SaaS" is the right vehicle and not just a trend label.
3. That the user has the capital and time to sustain a startup before revenue.
CLARIFYING_QUESTIONS:
1. What skills and capital can you actually commit?
2. Who is the specific customer and what job are they hiring the software to do?
CODE_INPUT:"""

CLASSIFY_DIRECT = """CATEGORY: DIRECT
COMPLEXITY: SIMPLE
NEEDS_CLARIFICATION: NO
GOAL:
ASSUMPTIONS:
CLARIFYING_QUESTIONS:
CODE_INPUT:"""


GENERATE = """APPROACH 1: Validate a painful, paid problem before building. Interview 20 potential buyers, pre-sell a waitlist, and build only the thinnest tool that automates the core job.

APPROACH 2: Target a niche vertical with real willingness to pay (e.g. dental, HVAC) and sell via direct outreach instead of burning cash on ads.

APPROACH 3: License an existing internal tool or wrap a commodity API as a branded product to cut build risk and lean on a distribution partner.

Recommendation: pick one narrow, well-funded niche and pre-sell before you build anything."""


CRITIQUE_WITH_ISSUES = """ISSUES:
- Pre-selling is slow and may not reach a million in revenue quickly.
- Assumes buyers exist and will talk to a stranger.
- Narrow niches cap total revenue well below a million unless priced very high.
- Direct outreach does not scale without a team."""


CRITIQUE_EMPTY = """ISSUES: (none)"""


MERGE = """Bottom line: a million dollars from a SaaS is realistic only if you solve a painful, paid problem for a specific customer — not by chasing the "SaaS millionaire" label.

The real considerations:
- Validate a painful, paid problem first: interview 20 potential buyers before writing code, pre-sell a waitlist, and build only the thinnest tool that automates the core job.
- A niche vertical with real willingness to pay (e.g. dental, HVAC) beats a broad consumer app: you can sell via direct outreach instead of burning cash on ads.
- Licensing an existing internal tool or wrapping a commodity API as a branded product cuts build risk and leans on a distribution partner.

Recommendation: pick one narrow, well-funded niche and pre-sell before you build anything.

Worth clarifying: what skills and capital can you actually commit, and who is the specific paying customer?"""


# --------------------------------------------------------------------------- #
# Code-mode canned responses (Phase 2)
# --------------------------------------------------------------------------- #

CLASSIFY_CODE = """CATEGORY: CODE
COMPLEXITY: COMPLEX
NEEDS_CLARIFICATION: YES
GOAL: Produce a correct C++ application.
ASSUMPTIONS:
1. That X is well-defined enough to implement.
CLARIFYING_QUESTIONS:
1. What should "X" actually do?
CODE_INPUT:"""

# A vague "build me a SaaS / generate a C++ app that does X" request: the model
# correctly classifies CODE but flags that there is nothing concrete to build
# from. This is the misroute case -- the pipeline should reroute to DECISION.
CLASSIFY_CODE_VAGUE = """CATEGORY: CODE
COMPLEXITY: COMPLEX
NEEDS_CLARIFICATION: YES
GOAL: Produce a C++ application, but the spec ("does X") is undefined.
ASSUMPTIONS:
1. That "X" is well-defined enough to implement (it is not).
CLARIFYING_QUESTIONS:
1. What concrete behavior should the app have?
2. Which language and platform?
CODE_INPUT:"""

CLASSIFY_CODE_WITH_INPUT = """CATEGORY: CODE
COMPLEXITY: SIMPLE
NEEDS_CLARIFICATION: NO
GOAL: Fix the supplied function.
ASSUMPTIONS:
CLARIFYING_QUESTIONS:
CODE_INPUT: def add(a, b):
    return a - b  # bug: subtracts instead of adds"""

# A concrete code request that GENERATES (no pasted code, but a real spec so it
# stays in CODE mode and the pipeline runs generate -> verify -> fix).
CLASSIFY_CODE_GENERATE = """CATEGORY: CODE
COMPLEXITY: COMPLEX
NEEDS_CLARIFICATION: NO
GOAL: Write a Python function that returns the sum of a list of numbers.
ASSUMPTIONS:
1. The input is a list of numbers.
CLARIFYING_QUESTIONS:
CODE_INPUT:"""

GENERATE_CODE = """#include <iostream>
int main() {
    std::cout << "X";
    return 0;
}"""

VERIFY_WITH_ISSUES = """- Missing error handling for invalid input.
- Off-by-one in the loop boundary.
- No tests cover the edge case."""

VERIFY_EMPTY = """None."""

FIX_CODE = """#include <iostream>
int main() {
    try {
        std::cout << "X";
    } catch (...) {
        return 1;
    }
    return 0;
}"""


# --------------------------------------------------------------------------- #
# Stub builder
# --------------------------------------------------------------------------- #

def make_stub(
    *,
    classify: str = CLASSIFY_DECISION,
    generate: str = GENERATE,
    critique_first: str = CRITIQUE_WITH_ISSUES,
    critique_later: str = CRITIQUE_EMPTY,
    merge: str = MERGE,
    direct_answer: str = "The capital of France is Paris.",
    generate_code: str = GENERATE_CODE,
    verify_first: str = VERIFY_WITH_ISSUES,
    verify_later: str = VERIFY_EMPTY,
    fix_code: str = FIX_CODE,
) -> Callable[[str], str]:
    """Return a sync fake call_llm.

    The stub routes by prompt content markers:
        * classify prompt contains "Classify the following prompt"
        * decision generate prompt contains "Write a direct, human reply to that message"
        * decision critique prompt contains "Read the reply below"
        * decision merge prompt contains "best, most honest version"
        * code generate prompt contains "Write complete, working code"
        * code verify prompt contains "Review this code for correctness"
        * code fix prompt contains "Issues to fix"
    On the first critique/verify call it returns the *_first variant; on
    subsequent calls the *_later variant (so loops can converge to no_issues).
    Anything else (e.g. a fresh direct-mode answer call) returns direct_answer.
    """
    state = {"critique_calls": 0, "verify_calls": 0}

    def stub(prompt: str) -> str:
        p = prompt
        if "Classify the following prompt" in p:
            return classify
        if "GENUINELY DISTINCT approaches" in p or "direct, human reply" in p:
            return generate
        if "Read the reply below" in p:
            state["critique_calls"] += 1
            return critique_first if state["critique_calls"] == 1 else critique_later
        if "best, most honest version" in p:
            return merge
        if "Write complete, working code" in p:
            return generate_code
        if "Review this code for correctness" in p:
            state["verify_calls"] += 1
            return verify_first if state["verify_calls"] == 1 else verify_later
        if "Issues to fix" in p:
            return fix_code
        return direct_answer

    return stub


def make_async_stub(**kwargs: Any) -> Callable[[str], Awaitable[str]]:
    import asyncio

    sync_stub = make_stub(**kwargs)

    async def astub(prompt: str) -> str:
        return sync_stub(prompt)

    return astub
