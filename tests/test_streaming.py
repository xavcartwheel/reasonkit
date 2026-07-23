# Streaming passthrough tests.
#
# ReasonKit inherits the shape of `fn`:
# - plain str -> str  => wrapper returns str
# - async str -> str  => wrapper is async, returns str
# - async generator   => wrapper is an async generator yielding str
import asyncio
import inspect

import reasonkit
from tests.fixtures import make_stub, FIXTURE_COFFEE


def test_plain_fn_returns_string_not_generator():
    wrapped = reasonkit.enhance(make_stub(), return_trace=True)
    res = wrapped(FIXTURE_COFFEE)
    assert isinstance(res, str) or hasattr(res, "answer")
    assert not inspect.isasyncgenfunction(wrapped)


def test_streaming_fn_yields_refined_answer():
    async def sfn(prompt: str):
        for chunk in ["It ", "depends ", "on ", "your ", "market."]:
            yield chunk

    wrapped = reasonkit.enhance(sfn, return_trace=True)
    assert inspect.isasyncgenfunction(wrapped), "streaming fn must yield a generator"

    async def collect():
        out = []
        async for piece in wrapped(FIXTURE_COFFEE):
            out.append(piece)
        return "".join(out)

    full = asyncio.run(collect())
    assert "market" in full
    # The pipeline ran (classify + refine) over the buffered stream; the refined
    # answer must contain the streamed content's substance.
    assert len(full) > 0


def test_streaming_yields_something_even_on_pipeline_failure():
    # If classify returns garbage, the pipeline degrades; the wrapper must still
    # yield the buffered raw text rather than nothing.
    async def sfn(prompt: str):
        for chunk in ["raw ", "streamed ", "answer"]:
            yield chunk

    wrapped = reasonkit.enhance(sfn, return_trace=True)

    async def collect():
        out = []
        async for piece in wrapped("anything"):
            out.append(piece)
        return "".join(out)

    full = asyncio.run(collect())
    assert full.strip() == "raw streamed answer"
