# Decision-mode pipeline tests: orchestration, call count, stop conditions, trace.
import reasonkit
from reasonkit.core import EnhanceResult
from tests.fixtures import (
    make_stub,
    FIXTURE_SAAS,
    FIXTURE_COFFEE,
    FIXTURE_PLAN,
    PLACEHOLDER_PLAN,
)


def _run(prompt, **kwargs):
    stub = make_stub()
    wrapped = reasonkit.enhance(stub, return_trace=True, **kwargs)
    res = wrapped(prompt)
    assert isinstance(res, EnhanceResult)
    return res


def test_decision_uses_one_call_per_stage_per_cycle():
    # Pipeline: classify(1) + per-cycle: generate(1), critique_all(1), augment(1), deepen(1).
    # With max_cycles=2: each stage appears at most 2× (except classify).
    res = _run(FIXTURE_SAAS, max_cycles=2)
    stages = [c.stage for c in res.trace.calls]
    assert stages[0] == "classify"
    assert stages.count("classify") == 1
    gen = stages.count("generate")
    crit = stages.count("critique")
    augment = stages.count("augment")
    deepen = stages.count("deepen")
    assert gen <= 2  # max_cycles
    assert crit <= 2  # one batch critique per cycle
    assert augment <= 2  # max_cycles
    assert deepen <= 2  # max_cycles


def test_decision_stops_on_empty_issues():
    res = _run(FIXTURE_SAAS, max_cycles=2)
    # Our stub: first critique has issues, second (cycle 2) is empty -> no_issues.
    assert res.stopped_reason == "no_issues"


def test_decision_trace_records_every_call():
    res = _run(FIXTURE_COFFEE, max_cycles=2)
    assert res.trace.call_count >= 4
    # Every call has a non-empty prompt and response.
    for c in res.trace.calls:
        assert c.prompt
        assert c.response is not None


def test_decision_saas_fixture_runs():
    res = _run(FIXTURE_SAAS, max_cycles=2)
    assert res.mode == "decision"
    assert "validation" in res.answer.lower() or len(res.answer) > 20


def test_decision_coffee_fixture_runs():
    res = _run(FIXTURE_COFFEE, max_cycles=2)
    assert res.mode == "decision"


def test_decision_plan_fixture_with_context():
    # The plan fixture needs context; we embed it in the prompt for the test.
    prompt = f"{FIXTURE_PLAN}\n\n{PLACEHOLDER_PLAN}"
    res = _run(prompt, max_cycles=2)
    assert res.mode == "decision"
    assert res.trace.call_count >= 4


def test_no_confidence_score_in_trace():
    res = _run(FIXTURE_SAAS, max_cycles=2)
    blob = str(res.trace.to_dict()).lower()
    assert "confidence" not in blob


def test_decision_default_explores_multiple_approaches():
    """Default branches=3 must flow into generate_approaches (real exploration
    at defaults, not just an opt-in). Capture the n passed to the prompt builder."""
    import reasonkit.branching as br

    captured = {}
    orig = br.build_generate_prompt

    def spy(goal, n, previous_issues=None):
        captured["n"] = n
        return orig(goal, n, previous_issues=previous_issues)

    br.build_generate_prompt = spy
    try:
        _run(FIXTURE_SAAS, max_cycles=2)
    finally:
        br.build_generate_prompt = orig
    assert captured.get("n") == 3


def test_decision_merge_fallback_when_regressed():
    """Regression guard: merge returns weak output, ensuring the best single
    approach survives instead of a degraded merge."""
    from tests.fixtures import make_stub

    base = make_stub()

    def stub(prompt: str) -> str:
        if "best, most honest version" in prompt:
            return "Maybe try something."  # clearly weaker than the 3 approaches
        return base(prompt)

    wrapped = reasonkit.enhance(stub, return_trace=True, max_cycles=1)
    res = wrapped(FIXTURE_SAAS)
    assert any("merge" in n and "regressed" in n for n in res.trace.notes)
    # The best approach (niche vertical / pre-sell) content must survive.
    assert "niche" in res.answer.lower() or "pre-sell" in res.answer.lower()
