from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

from .core import enhance, EnhanceResult, LLMCallable, AsyncLLMCallable, _is_coroutine_function


@dataclass
class Comparison:
    prompt: str
    baseline_answer: str
    reasonkit_answer: str
    trace: Any = None  # Trace or None

    def __str__(self) -> str:
        return (
            f"PROMPT:\n{self.prompt}\n\n"
            f"BASELINE:\n{self.baseline_answer}\n\n"
            f"REASONKIT:\n{self.reasonkit_answer}"
        )


def _collect_sync_or_async(result):
    # Resolve str | coroutine | async-gen to a plain string.
    import asyncio
    import inspect

    if inspect.isasyncgen(result):
        parts = []

        async def _g():
            async for piece in result:
                parts.append(piece if isinstance(piece, str) else str(piece))

        asyncio.run(_g())
        return "".join(parts)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def compare(
    fn: Union[LLMCallable, AsyncLLMCallable],
    prompt: str,
    **config: Any,
) -> Comparison:
    # Run fn directly (baseline) and enhance(fn)(prompt), resolved to strings.
    enhanced = enhance(fn, return_trace=True, **config)

    baseline = fn(prompt)
    baseline = _collect_sync_or_async(baseline)
    result = _collect_sync_or_async(enhanced(prompt))

    if isinstance(result, EnhanceResult):
        rk_answer = result.answer
        trace = result.trace
    else:
        rk_answer = result
        trace = None

    return Comparison(
        prompt=prompt,
        baseline_answer=baseline if isinstance(baseline, str) else str(baseline),
        reasonkit_answer=rk_answer,
        trace=trace,
    )
