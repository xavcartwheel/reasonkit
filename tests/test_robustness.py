# Robustness tests: config validation, typed errors, safe-degrade.
#
# ReasonKit must fail fast on bad config (ConfigurationError at enhance() time,
# never mid-pipeline) and must never emit silent wrong output when the wrapped
# function misbehaves (empty responses, raises). Critical-stage failures raise
# ModelCallError; non-critical stages degrade with a trace note.
import pytest

import reasonkit
from reasonkit.errors import (
    ReasonKitError,
    ConfigurationError,
    ModelCallError,
)
from reasonkit.core import EnhanceResult
from tests.fixtures import make_stub, FIXTURE_COFFEE


@pytest.mark.parametrize(
    "bad_cfg",
    [
        {"branches": 0},
        {"branches": -1},
        {"max_cycles": 0},
        {"stream_chunk_mode": "nonsense"},
        {"retries": -1},
    ],
)
def test_config_validation_raises(bad_cfg):
    with pytest.raises(ConfigurationError):
        reasonkit.enhance(make_stub(), **bad_cfg)


def test_configuration_error_is_reasonkit_error():
    with pytest.raises(ReasonKitError):
        reasonkit.enhance(make_stub(), branches=0)


def test_critical_stage_failure_raises_model_call_error():
    # If the raw ANSWER call fails (critical), the pipeline must raise rather
    # than return nothing or a fabricated answer.
    def stub(prompt: str) -> str:
        if "Classify the following prompt" in prompt:
            return (
                "CATEGORY: DIRECT\nCOMPLEXITY: SIMPLE\nGOAL:\n"
                "ASSUMPTIONS:\nCLARIFYING_QUESTIONS:\nCODE_INPUT:"
            )
        if FIXTURE_COFFEE in prompt:  # draft call carries the classifier context
            raise RuntimeError("provider 500")
        return "ok"

    wrapped = reasonkit.enhance(stub, return_trace=True, max_cycles=1, retries=1)
    with pytest.raises(ModelCallError):
        wrapped(FIXTURE_COFFEE)


def test_non_critical_empty_degrades_with_note():
    # The refine call returns empty (non-critical); DIRECT mode degrades to the
    # raw answer instead of crashing or returning nothing.
    state = {"refine": 0}

    def stub(prompt: str) -> str:
        if "Classify the following prompt" in prompt:
            return (
                "CATEGORY: DIRECT\nCOMPLEXITY: SIMPLE\nGOAL:\n"
                "ASSUMPTIONS:\nCLARIFYING_QUESTIONS:\nCODE_INPUT:"
            )
        if FIXTURE_COFFEE in prompt:  # draft call carries the classifier context
            return "The real raw answer about markets."
        if "Refine the draft:" in prompt:
            state["refine"] += 1
            return ""
        return "ok"

    wrapped = reasonkit.enhance(stub, return_trace=True, max_cycles=1, retries=1)
    res = wrapped(FIXTURE_COFFEE)
    assert isinstance(res, EnhanceResult)
    assert state["refine"] >= 1
    assert any("refine degraded" in n for n in res.trace.notes)
    assert "real raw answer about markets" in res.answer


def test_no_confidence_string_in_trace():
    wrapped = reasonkit.enhance(make_stub(), return_trace=True, max_cycles=1)
    res = wrapped(FIXTURE_COFFEE)
    blob = str(res.trace.to_dict()).lower()
    assert "confidence" not in blob
