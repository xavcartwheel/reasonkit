# Routing tests: the CODE->DECISION reroute for vague "build X" prompts
import reasonkit
from reasonkit.core import EnhanceResult
from reasonkit._utils import _regressed, _structure_score, _specificity
from tests.fixtures import (
    make_stub,
    CLASSIFY_CODE_VAGUE,
    FIXTURE_SAAS,
    FIXTURE_CPP,
)


def _run(prompt, classify=CLASSIFY_CODE_VAGUE, **kwargs):
    stub = make_stub(classify=classify)
    wrapped = reasonkit.enhance(stub, return_trace=True, **kwargs)
    res = wrapped(prompt)
    assert isinstance(res, EnhanceResult)
    return res


def test_vague_code_reroutes_to_decision_not_code():
    # "generate a C++ app that does X" -> classifier says CODE + needs_clarification
    # -> pipeline must reroute to DECISION (not run code mode).
    res = _run(FIXTURE_CPP, max_cycles=2)
    assert res.mode == "decision"
    assert any("rerouting to DECISION" in n for n in res.trace.notes)
    # It should NOT have run a code-mode verify/fix call.
    stages = [c.stage for c in res.trace.calls]
    assert "verify" not in stages
    assert "fix" not in stages
    # Crucially the answer is a real strategy reply, not a clarification stub.
    assert len(res.answer) > 100


def test_vague_saas_reroutes_to_decision():
    # "build me a SaaS that will make me a millionaire" historically went CODE and
    # collapsed to a 465-char stub. With the reroute it produces a decision answer.
    res = _run(FIXTURE_SAAS, classify=CLASSIFY_CODE_VAGUE, max_cycles=2)
    assert res.mode == "decision"
    assert len(res.answer) > 200


def test_code_with_real_input_still_runs_code_mode():
    # A genuine CODE request with pasted code must NOT be rerouted.
    from tests.fixtures import CLASSIFY_CODE_WITH_INPUT

    res = _run("fix this function", classify=CLASSIFY_CODE_WITH_INPUT, max_cycles=2)
    assert res.mode == "code"


def test_merge_structure_guard_detects_collapse():
    # A 40-section structured draft collapsed to a 4-item list must be flagged as
    # regressed (the qwq coffee_shop failure: 40 -> 1 section).
    structured = "\n".join(
        f"### Section {i}\n- point about {i}\n- another point about {i}"
        for i in range(1, 41)
    )
    collapsed = "1. Location\n2. Competition\n3. Money\n4. Experience"
    assert _regressed(collapsed, structured) is True
    # The structured draft is not regressed vs itself.
    assert _regressed(structured, structured) is False


def test_merge_structure_guard_allows_tightening():
    # A legitimate tighten (slightly shorter, same structure) is NOT a regression.
    draft = (
        "### Key Considerations\n- Location matters a lot.\n- Competition is fierce.\n"
        "- Money is the biggest risk.\n### Next Steps\n1. Research\n2. Plan\n3. Fund"
    )
    tightened = (
        "### Key Considerations\n- Location: high foot traffic beats low rent.\n"
        "- Competition: 5+ nearby shops changes the math.\n"
        "- Money: budget $100k-$200k upfront.\n### Next Steps\n1. Research\n2. Plan\n3. Fund"
    )
    assert _regressed(tightened, draft) is False


def test_structure_score_counts_markers():
    flat = "This is just a paragraph with no structure at all really."
    structured = "### Heading\n- bullet one\n- bullet two\n1. first\n2. second"
    assert _structure_score(flat) == 0.0
    assert _structure_score(structured) >= 4.0
    assert _structure_score(structured) > _structure_score(flat)
