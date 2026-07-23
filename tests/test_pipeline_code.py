# Code-mode pipeline tests: Generate -> Verify -> Fix loop.
import reasonkit
from reasonkit.core import EnhanceResult
from tests.fixtures import (
    make_stub,
    CLASSIFY_CODE,
    CLASSIFY_CODE_WITH_INPUT,
    CLASSIFY_CODE_GENERATE,
    FIXTURE_CPP,
)


def _run(prompt, classify=CLASSIFY_CODE_WITH_INPUT, test_hook=None, **kwargs):
    # Default to CLASSIFY_CODE_WITH_INPUT (a real code request) so code mode runs.
    # A *vague* CODE request (CLASSIFY_CODE / CLASSIFY_CODE_VAGUE) is now rerouted
    # to DECISION by core._run_pipeline -- that is the fix for the SaaS/C++ misroute.
    stub = make_stub(classify=classify)
    wrapped = reasonkit.enhance(stub, return_trace=True, test_hook=test_hook, **kwargs)
    res = wrapped(prompt)
    assert isinstance(res, EnhanceResult)
    return res


def test_code_mode_runs_pipeline_not_direct_fallback():
    res = _run(FIXTURE_CPP, classify=CLASSIFY_CODE_GENERATE)
    assert res.mode == "code"
    stages = [c.stage for c in res.trace.calls]
    assert "classify" in stages
    assert "generate" in stages
    assert "verify" in stages
    assert not any("falling back to direct" in n for n in res.trace.notes)


def test_code_stops_on_empty_issues():
    res = _run(FIXTURE_CPP, classify=CLASSIFY_CODE_GENERATE, max_cycles=2)
    assert res.stopped_reason == "no_issues"


def test_code_call_count_discipline():
    # generate(1) + per cycle: verify + fix. max_cycles=2, converges at cycle 2.
    # classify(1) + generate(1) + cycle1: verify, fix (2) + cycle2: verify (stops) (1) = 5.
    # Per-cycle discipline: <= 2 calls/cycle (verify+fix), and generate happens once.
    res = _run(FIXTURE_CPP, classify=CLASSIFY_CODE_GENERATE, max_cycles=2)
    stages = [c.stage for c in res.trace.calls]
    assert stages.count("generate") == 1        # generated once
    assert stages.count("verify") == 2
    assert stages.count("fix") == 1             # no fix after empty verify
    # No more than 2 calls per cycle (verify+fix); the setup calls are 1+1.
    # Total should be 5: classify + generate + 2 (cycle1) + 1 (cycle2 verify).
    assert res.trace.call_count == 5


def test_code_versions_recorded():
    res = _run(FIXTURE_CPP, classify=CLASSIFY_CODE_GENERATE)
    assert len(res.trace.code_versions) >= 2  # initial + at least one fix


def test_code_uses_test_hook_and_skips_llm_verify():
    captured = {}

    def hook(code: str) -> list[str]:
        captured["called"] = True
        captured["code"] = code
        return ["hook found a bug"]

    res = _run(FIXTURE_CPP, classify=CLASSIFY_CODE_GENERATE, test_hook=hook, max_cycles=2)
    assert captured.get("called") is True
    # No LLM verify call happened; the verify stage source must be test_hook.
    verify_calls = [c for c in res.trace.calls if c.stage == "verify"]
    assert verify_calls, "expected a verify call"
    assert all(c.metadata.get("source") == "test_hook" for c in verify_calls)


def test_code_user_supplied_input_skips_generate():
    res = _run(
        "fix this function",
        classify=CLASSIFY_CODE_WITH_INPUT,
        max_cycles=1,
    )
    stages = [c.stage for c in res.trace.calls]
    assert "generate" not in stages  # used CODE_INPUT instead
    assert "verify" in stages
    assert any("user-supplied code" in n for n in res.trace.notes)


def test_code_cpp_fixture_runs():
    res = _run(FIXTURE_CPP, classify=CLASSIFY_CODE_GENERATE, max_cycles=2)
    assert res.mode == "code"
    assert res.answer  # final code string returned


def test_code_underspecified_stops_instead_of_guessing():
    """Phase-1 exit criterion: when code mode DOES run (real code/spec present)
    and generation returns a 'I need X' note, the pipeline must STOP there and
    return that honest note -- never invent a program and feed it to verify/fix.

    Regression guard: an earlier version passed the vague note into verify/fix,
    which fabricated a full application to satisfy the vagueness.
    """
    vague_note = (
        "The request is too vague, as it does not specify what X is. "
        "Please provide a clear description of what the application should do."
    )
    stub = make_stub(
        classify=CLASSIFY_CODE_GENERATE,
        generate_code=vague_note,
    )
    wrapped = reasonkit.enhance(stub, return_trace=True, max_cycles=2, branches=1)
    res = wrapped(FIXTURE_CPP)

    assert res.mode == "code"
    assert res.stopped_reason == "needs_clarification"
    # One generate, NO verify, NO fix -- we never manufacture an artifact.
    stages = [c.stage for c in res.trace.calls]
    assert stages.count("generate") == 1
    assert "verify" not in stages
    assert "fix" not in stages
    assert len(res.trace.code_versions) == 1  # no fabricated revisions
    # The answer is the honest note, not a guessed program.
    assert "too vague" in res.answer
    assert "```cpp" not in res.answer
    assert any("stopping instead of guessing" in n for n in res.trace.notes)


def test_vague_code_reroutes_to_decision_not_code():
    """The core fix: a 'generate a C++ app that does X' request with no code and
    no concrete spec must NOT run code mode. It reroutes to DECISION and produces
    a real strategy answer instead of a clarification stub."""
    res = _run(FIXTURE_CPP, classify=CLASSIFY_CODE, max_cycles=2)
    assert res.mode == "decision"
    assert any("rerouting to DECISION" in n for n in res.trace.notes)
    assert len(res.answer) > 100
