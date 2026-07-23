from __future__ import annotations

import os

import reasonkit


def call_llm(prompt: str) -> str:
    """The developer's own Anthropic call. ReasonKit only ever sees str -> str."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


async def acall_llm(prompt: str) -> str:
    """Async equivalent -- enhance() returns an async wrapper to match."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


smart_llm = reasonkit.enhance(call_llm, max_cycles=2, branches=3)
async_smart_llm = reasonkit.enhance(acall_llm, max_cycles=2, branches=3)


if __name__ == "__main__":
    prompt = "should I start a coffee shop?"
    baseline = call_llm(prompt)
    enhanced = smart_llm(prompt)

    print("BASELINE:\n", baseline)
    print("\nREASONKIT:\n", enhanced)
