# Example: wrap an OpenAI chat call with ReasonKit.
#
# This file is NOT part of the reasonkit package and is NOT an automated test.
# It exists only to prove that enhance() works against a real provider -- the
# wrapped function is *your* function; ReasonKit never imports openai itself.
#
# To run:
#     1. pip install openai
#     2. export OPENAI_API_KEY=sk-...
#     3. python examples/wrap_openai_example.py
#
# ReasonKit's entire dependency on OpenAI is the one `call_llm` function below.
from __future__ import annotations

import os

import reasonkit


def call_llm(prompt: str) -> str:
    """The developer's own OpenAI call. ReasonKit only ever sees this signature."""
    from openai import OpenAI  # imported here so the package itself never depends on it

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


# Wrap it once. Same signature in, same signature out.
smart_llm = reasonkit.enhance(call_llm, max_cycles=2, branches=3)


if __name__ == "__main__":
    prompt = "should I start a coffee shop?"
    baseline = call_llm(prompt)
    enhanced = smart_llm(prompt)

    print("BASELINE:\n", baseline)
    print("\nREASONKIT:\n", enhanced)
