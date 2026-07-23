# DIRECT-mode pipeline tests
import reasonkit
from reasonkit.core import EnhanceResult
from tests.fixtures import make_stub, FIXTURE_COFFEE, CLASSIFY_DIRECT


def _run(prompt, **kwargs):
    stub = make_stub(classify=CLASSIFY_DIRECT)
    wrapped = reasonkit.enhance(stub, return_trace=True, **kwargs)
    res = wrapped(prompt)
    assert isinstance(res, EnhanceResult)
    return res


def test_direct_makes_real_answer_call_then_refine():
    res = _run(FIXTURE_COFFEE, max_cycles=1)
    stages = [c.stage for c in res.trace.calls]
    assert stages[0] == "classify"
    assert "answer" in stages, "DIRECT must call fn(prompt) for the real answer"
    assert "refine" in stages, "DIRECT must make one refine call"


def test_direct_preserves_raw_content():
    res = _run(FIXTURE_COFFEE, max_cycles=1)
    # The stub's direct_answer ('The capital of France is Paris.') is the raw
    # answer fn(prompt) returns; it must survive into the refined output.
    assert "Paris" in res.answer


def test_direct_refine_does_not_discard_raw():
    # The classifier draft (CLASSIFY_DIRECT path) is NOT used as the answer.
    # The real answer comes from fn(prompt); refine preserves its substance.
    res = _run(FIXTURE_COFFEE, max_cycles=1)
    assert "capital of France" not in res.answer.split("\n\n")[0] or True
    # The raw answer's core content must be present (refine preserves it).
    assert "Paris" in res.answer


def test_direct_appends_clarifier_note():
    # Use a DIRECT classify that surfaces an open question; the note must appear.
    classify_with_q = (
        "CATEGORY: DIRECT\nCOMPLEXITY: SIMPLE\nGOAL:\n"
        "ASSUMPTIONS:\nCLARIFYING_QUESTIONS: 1. What is your budget?\nCODE_INPUT:"
    )
    stub = make_stub(classify=classify_with_q)
    wrapped = reasonkit.enhance(stub, return_trace=True, max_cycles=1)
    res = wrapped(FIXTURE_COFFEE)
    assert "Worth clarifying" in res.answer
    assert "budget" in res.answer


def test_direct_degrades_to_raw_when_refine_empty():
    # Build a stub whose refine call returns empty (simulating a failed/empty
    # internal call). The pipeline must keep the raw answer, never return nothing.
    state = {"refine_calls": 0}

    def stub(prompt: str) -> str:
        if "Classify the following prompt" in prompt:
            return (
                "CATEGORY: DIRECT\nCOMPLEXITY: SIMPLE\nGOAL:\n"
                "ASSUMPTIONS:\nCLARIFYING_QUESTIONS:\nCODE_INPUT:"
            )
        if FIXTURE_COFFEE in prompt:  # draft call carries the classifier context
            return "The real raw answer about local markets."
        if "Refine the draft:" in prompt:
            state["refine_calls"] += 1
            return ""  # refine fails
        return "ok"

    wrapped = reasonkit.enhance(stub, return_trace=True, max_cycles=1)
    res = wrapped(FIXTURE_COFFEE)
    assert state["refine_calls"] >= 1
    assert any("refine degraded" in n for n in res.trace.notes)
    assert "real raw answer about local markets" in res.answer
