from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallRecord:
    # One call ReasonKit made through the wrapped function.

    stage: str          # classify | generate | critique | merge | answer
    prompt: str         # exact prompt sent to fn
    response: str       # raw string fn returned
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    # Everything that happened inside one enhance() run.

    mode: str = "unknown"                       # decision | code | direct
    calls: list[CallRecord] = field(default_factory=list)
    cycles: int = 0
    stopped_reason: str = "unknown"             # no_issues | no_improvement | max_cycles
    classification: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    code_versions: list[str] = field(default_factory=list)  # each draft/fix

    def add_call(self, stage: str, prompt: str, response: str, **metadata: Any) -> CallRecord:
        rec = CallRecord(stage=stage, prompt=prompt, response=response, metadata=dict(metadata))
        self.calls.append(rec)
        return rec

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cycles": self.cycles,
            "stopped_reason": self.stopped_reason,
            "call_count": self.call_count,
            "classification": self.classification,
            "notes": self.notes,
            "code_versions": self.code_versions,
            "calls": [
                {
                    "stage": c.stage,
                    "prompt": c.prompt,
                    "response": c.response,
                    "metadata": c.metadata,
                }
                for c in self.calls
            ],
        }
