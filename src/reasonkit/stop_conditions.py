from __future__ import annotations

from typing import Optional


def count_issues(issues_by_approach: list[list[str]]) -> int:
    # Total number of concrete issues across all approaches.
    return sum(len(iss) for iss in issues_by_approach)


def should_stop(
    cycle: int,
    max_cycles: int,
    issues_by_approach: list[list[str]],
    previous_issue_count: Optional[int],
) -> tuple[bool, str]:
    # Return (stop, reason) with reason in no_issues/no_improvement/max_cycles.
    total = count_issues(issues_by_approach)

    if total == 0:
        return True, "no_issues"

    if previous_issue_count is not None and total >= previous_issue_count:
        return True, "no_improvement"  # list didn't shrink

    if cycle >= max_cycles:
        return True, "max_cycles"

    return False, ""
