<div align="center">
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./img/logo-dark.png">
    <img src="./img/logo-light.png" alt="ReasonKit">
    </picture>
</div>

<p align="center">
  <strong>Your LLM's first answer is not your best option. One wrapper later, your model checks its work and improves.</strong>
</p>

<p align="center">
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square" alt="Python version"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue?style=flat-square" alt="License"></a>
</p>

---

Single-pass LLM calls sound confident. They ship their first draft without review. They do not catch the assumptions they made or the questions they should have asked.

**ReasonKit** wraps your `call_llm(prompt) -> str` and returns a function with the same signature. The wrapped function runs classification, generates alternative approaches, critiques them, merges the best ideas, and deepens the final answer. Every call goes through your model, your credentials, and your rate limits. No second provider. No new API.

```python
import reasonkit

def call_llm(prompt: str) -> str:
    ...  # your existing LLM call

better_llm = reasonkit.enhance(call_llm)
answer = better_llm("Should I start a coffee shop?")
```

The **benchmark** across 2 providers (Cloudflare, LLM7.io) and 6 models shows the results:

| Model | RQI raw | RQI ReasonKit | Delta | Catch rate | SC lift |
|-------|---------|---------------|-------|------------|---------|
| Llama 3.1 8B | 21 | 24 | +3.0 | 80% | +6.8 |
| Llama 4 Scout 17B | 23 | 28 | +4.8 | 100% | +8.5 |
| Qwen QwQ 32B | 23 | 29 | +6.0 | 100% | +8.7 |
| GPT-OSS 20B | 23 | 30 | +7.4 | 100% | +8.9 |
| Codestral Latest | 21 | 29 | +8.6 | 100% | +8.6 |
| MiniMax M2.7 | 21 | 22 | +0.6 | 100% | +9.0 |
| **Average** | **22** | **27** | **+5.1** | **97%** | **+8.4** |

RQI is a 0-40 composite of reasoning, accuracy, questioning, and structure. A fixed judge model scores each answer blindly. Self-correction (SC) is trace-derived and reported separately. Read the full methodology.

**Cost warning.** `enhance()` multiplies your token usage by about 3 to 4 times. Every internal call goes through your function. You pay for every one. Use ReasonKit for prompts where quality matters. For high-volume traffic, lower branches and max_cycles to reduce cost.

## Features

- **Self-correcting output.** The pipeline critiques its own draft, finds concrete flaws, and fixes them. It catches 80% to 100% of issues a raw call would ship without review — 97% average across 6 models.
- **Surfaces hidden assumptions.** The critique pass forces the model to name its own assumptions. You see the clarifying questions a single-pass call omits.
- Drop-in wrapper. enhance() keeps your existing str -> str signature. Async and streaming callables work too.
- **Zero dependencies.** Standard library only. No third-party packages are installed.
- Full trace. Opt-in to see every internal call, classification, and revision.

## Install

```bash
pip install reasonkit
```

Requires Python 3.9 or later. No third-party packages are installed by default.

For development:

```bash
pip install -e ".[test]"
```

## Quick start

```python
import reasonkit

def call_llm(prompt: str) -> str:
    ...  # your existing LLM call

better_llm = reasonkit.enhance(call_llm)
answer = better_llm("Should I start a coffee shop?")
```

Enable the trace for full visibility:

```python
better_llm = reasonkit.enhance(call_llm, return_trace=True)
result = better_llm("Should I start a coffee shop?")
result.answer          # str
result.trace           # every internal call
result.mode            # decision, code, or direct
result.stopped_reason  # why the pipeline stopped
```

Compare raw and wrapped output side by side:

```python
comparison = reasonkit.compare(call_llm, "Should I start a coffee shop?")
comparison.baseline_answer    # raw call_llm result
comparison.reasonkit_answer   # wrapped enhance result
```

A decorator form is also supported:

```python
@reasonkit.enhance
def call_llm(prompt: str) -> str:
    ...
```

## How ReasonKit works

Every stage is a differently-worded prompt sent through your own function. ReasonKit never calls a model directly.

```
User Prompt
    |
    v
Classify + Surface Assumptions
    |
    +--direct--> Answer --> Refine
    |
    +--decision--> Generate Approaches --> Critique --> Merge
    |
    +--code--> Generate Code --> Verify --> Fix Issues
    |
    v
Final answer with caveats
```

Decision mode routes advisory questions through generate, critique, and merge. The model surfaces assumptions, weighs trade-offs, and appends open questions.

Code mode generates code, then runs verify and fix loops. Pass a test_hook to use your own tests instead of LLM self-review.

Direct mode answers plain questions in one pass with a refinement step.

It stops when the issue list is empty, stops shrinking, or hits max_cycles. There is no fabricated confidence score.

## Configuration

enhance() accepts these keyword arguments:

| Option | Default | Description |
|--------|---------|-------------|
| max_cycles | 2 | Max decision or code-loop cycles |
| branches | 3 | Approaches generated per cycle |
| return_trace | False | Return EnhanceResult with .answer, .trace, .mode, .stopped_reason |
| verbose | False | Print cycle progress |
| test_hook | None | Function (code: str) -> list[str] for code verify pass |

```python
better_llm = reasonkit.enhance(call_llm, max_cycles=2, branches=3, test_hook=my_tests)
```

## API

### reasonkit.enhance(fn, **config)

Wrap fn with the reasoning pipeline. fn must accept (prompt: str) -> str. Sync, async, and async generator functions are all supported. The wrapper keeps the same callable shape.

When return_trace=True, each call returns an EnhanceResult instead of a plain string.

### class EnhanceResult

| Attribute | Type | Description |
|-----------|------|-------------|
| .answer | str | The final output |
| .trace | Trace | Record of every internal call |
| .mode | str | decision, code, or direct |
| .stopped_reason | str | no_issues, no_improvement, max_cycles, or similar |

### reasonkit.compare(fn, prompt, **config)

Run fn(prompt) directly and enhance(fn)(prompt) side by side. Returns a Comparison with .baseline_answer, .reasonkit_answer, and .trace.

### class Trace

Records every call ReasonKit made through the wrapped function.

| Method or Attribute | Type | Description |
|--------------------|------|-------------|
| .calls | list[CallRecord] | Stage, prompt, and response per call |
| .call_count | int | Total internal calls |
| .mode | str | Pipeline mode |
| .stopped_reason | str | Why the pipeline stopped |
| .notes | list[str] | Diagnostic notes from the run |
| .code_versions | list[str] | Each draft or fix produced in code mode |
| .to_dict() | dict | Serialize the full trace |

### Error types

All errors inherit from ReasonKitError.

| Exception | Raised when |
|-----------|-------------|
| ConfigurationError | Invalid config at enhance() time |
| ModelCallError | Critical pipeline stage exhausted all retries |

## Cost and call-reduction notes

The largest lever on spend is branches multiplied by max_cycles. The defaults (3 x 2) favor quality over frugality. Several mechanisms already reduce call count:

- Decision mode skips the merge on a clean first draft. If cycle 1 critique finds zero issues, the pipeline returns the draft plus a clarifying note and stops.
- Direct mode uses the classifier's output in the first draft. It does not answer blind and then refine. The classify pass produces goal, assumptions, and clarifying questions. ReasonKit feeds these into the first draft call.
- Code mode reuses your supplied code. If you paste code, generation is skipped. The verify and fix loop works on what you provided.

To reduce cost further, lower the settings:

```python
better_llm = reasonkit.enhance(call_llm, branches=1, max_cycles=1)
```

branches=1 collapses approach generation to a single draft. max_cycles=1 runs exactly one generate, critique, and merge pass. Wrap only the prompts where the quality lift is worth the extra calls.

## Examples

Two examples show enhance() wrapping real providers:

- wrap_openai_example.py wraps an OpenAI gpt-4o-mini call.
- wrap_anthropic_example.py wraps Anthropic Claude in sync and async forms.

Provider SDK imports appear only in these examples. They are not package dependencies.

## Benchmark

Results use a 5-axis rubric scored by LLM-as-judge. A fixed judge model evaluates each answer blindly. The judge does not know which condition produced the answer. The four text axes (reasoning, accuracy, questioning, structure) are blind-scored. Self-correction is trace-derived.

Benchmarked across 14 fixture-runs on 6 models (2 providers):

| Model | Condition | RQI | SC |
|-------|-----------|-----|----|
| Llama 3.1 8B | Raw | 21 | 0.2 |
| | ReasonKit | 24 | 7.0 |
| | Delta | **+3.0** | **+6.8** |
| Llama 4 Scout 17B | Raw | 23 | 0.5 |
| | ReasonKit | 28 | 9.0 |
| | Delta | **+4.8** | **+8.5** |
| Qwen QwQ 32B | Raw | 23 | 0.3 |
| | ReasonKit | 29 | 9.0 |
| | Delta | **+6.0** | **+8.7** |
| GPT-OSS 20B | Raw | 23 | 0.1 |
| | ReasonKit | 30 | 9.0 |
| | Delta | **+7.4** | **+8.9** |
| Codestral Latest | Raw | 21 | 0.4 |
| | ReasonKit | 29 | 9.0 |
| | Delta | **+8.6** | **+8.6** |
| MiniMax M2.7 | Raw | 21 | 0.0 |
| | ReasonKit | 22 | 9.0 |
| | Delta | **+0.6** | **+9.0** |
| **Average** | Raw | 22 | 0.3 |
| | ReasonKit | 27 | 8.7 |
| | **Delta** | **+5.1** | **+8.4** |

Catch rate. The fraction of fixtures where ReasonKit caught a concrete flaw the raw call shipped without review: Llama 3.1 8B 80%, Llama 4 Scout 17B 100%, Qwen QwQ 32B 100%, GPT-OSS 20B 100%, Codestral Latest 100%, MiniMax M2.7 100%. Average 97%.

| Axis | What it measures | Judge criterion |
|------|-----------------|-----------------|
| Reasoning | Multiple angles weighed, trade-offs named | Several distinct considerations, not a single narrow recommendation |
| Accuracy | Addresses what was asked | Honest about missing input, does not fabricate specifics |
| Questioning | Surfaces assumptions, asks what would change the answer | Names assumptions clearly, asks targeted questions |
| Structure | Organized and scannable | Clear headings, logical flow, not a wall of text |
| Self-Correction | Caught and fixed its own flaw | Trace-derived. Did the pipeline revise its own draft? |

Each axis is scored 0 to 10. RQI is the 0 to 40 composite of RE plus AC plus QU plus ST. SC is reported separately. A single-shot raw call structurally cannot earn SC points.

Key takeaways:

- **Self-correction is the largest difference.** Raw calls get SC equal to 0 by definition. They cannot revise what they already emitted. ReasonKit catches and fixes concrete flaws in 97% of fixtures. Average SC lift is +8.4. This axis is where no single-pass call competes.
- **Questioning lifts consistently across all models.** The pipeline forces the model to surface its own assumptions and ask clarifying questions. Raw calls rarely do this.
- **The lift grows with model capability.** Codestral Latest gains +8.6 RQI. MiniMax M2.7 gains +0.6 RQI. The pipeline amplifies stronger models further. It is not a crutch for weak models.
- **No axis is inflated.** Accuracy on small models shows occasional slight negative deltas. The longer pipeline sometimes loses detail on weaker base models. These numbers are reported as-is.

Methodology. The four text axes are scored blind. The judge receives no metadata about which run produced an answer. Per-fixture reports label the two outputs RUN 1 and RUN 2 in randomized order. Execution order is also randomized per fixture to avoid fixed-order provider drift. The judge is nemotron-3-ultra-free (OpenCode), a fixed model independent of the models being benchmarked. Underperformance is stated plainly per axis. No result is adjusted. Full per-fixture output and traces are in benchmarks/stats_output/ and benchmarks/stats_output.llm7io/.

## Claude Code plugin

Turn Claude Code into a self-checking reasoner. The plugin wraps the claude CLI as the model behind reasonkit.enhance. You get the full pipeline without an API key.

### Install

```
/plugin marketplace add xavcartwheel/reasonkit
/plugin install reasonkit@reasonkit-marketplace
```

### Use

```
/reasonkit Should I start a coffee shop?
```

Or run the bundled script directly:

```
python plugins/reasonkit/scripts/reasonkit_run.py "Should I start a coffee shop?"
```

See plugins/reasonkit/ for the full plugin source.

Requires pip install reasonkit and the claude CLI on your PATH.

## Contributing

Contributions are welcome. Open an issue to discuss substantial changes first. See CONTRIBUTING.md for setup and conventions. The test suite must pass, and the package must stay dependency-free beyond the standard library.

## License

Released under the GNU General Public License v3.0.
