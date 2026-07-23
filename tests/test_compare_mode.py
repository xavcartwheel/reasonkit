# compare() tests: baseline vs enhanced, side by side, using the stub.
import reasonkit
from reasonkit.compare import Comparison
from tests.fixtures import make_stub, FIXTURE_SAAS, FIXTURE_COFFEE


def test_compare_returns_both_answers():
    stub = make_stub()
    comp = reasonkit.compare(stub, FIXTURE_SAAS)
    assert isinstance(comp, Comparison)
    assert comp.baseline_answer
    assert comp.reasonkit_answer
    assert comp.baseline_answer != comp.reasonkit_answer  # pipeline changed the output


def test_compare_baseline_is_direct_call():
    # Baseline must be exactly fn(prompt) unchanged (classify prompt NOT sent).
    baseline_calls = []
    pipeline_stub = make_stub()

    def baseline_fn(prompt):
        baseline_calls.append(prompt)
        # The "real" model would answer here; for the test, delegate to the stub.
        return pipeline_stub(prompt)

    comp = reasonkit.compare(baseline_fn, FIXTURE_COFFEE)
    # compare() uses the same fn for baseline and pipeline; the FIRST call is the
    # raw baseline (original prompt, no classify wrapper).
    assert baseline_calls[0] == FIXTURE_COFFEE
    assert "Classify the following prompt" not in comp.baseline_answer


def test_compare_trace_available():
    stub = make_stub()
    comp = reasonkit.compare(stub, FIXTURE_SAAS)
    assert comp.trace is not None
    assert comp.trace.call_count >= 4


def test_compare_async_callable():
    import asyncio
    from tests.fixtures import make_async_stub

    stub = make_async_stub()
    comp = reasonkit.compare(stub, FIXTURE_COFFEE)
    assert comp.baseline_answer
    assert comp.reasonkit_answer
    assert comp.trace is not None


def test_comparison_str_uses_instance_prompt():
    comp = Comparison(
        prompt="hello",
        baseline_answer="base",
        reasonkit_answer="rk",
    )
    s = str(comp)
    assert "PROMPT:\nhello" in s
    assert "BASELINE:\nbase" in s
    assert "REASONKIT:\nrk" in s
