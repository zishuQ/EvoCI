"""Configurable capability utility scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evoci.capability.models import SkillStats


class SkillUtilityPolicy(Protocol):
    def score(self, stats: SkillStats) -> float: ...


@dataclass(frozen=True, slots=True)
class WeightedUtilityPolicy:
    success_weight: float = 0.6
    tool_reduction_weight: float = 0.2
    retry_reduction_weight: float = 0.2
    baseline_tool_calls: float = 20.0
    baseline_attempts: float = 3.0

    def score(self, stats: SkillStats) -> float:
        outcomes = stats.success_count + stats.failure_count
        success_rate = stats.success_count / outcomes if outcomes else 0.0
        tool_reduction = 0.0
        if stats.avg_tool_calls_when_used is not None:
            tool_reduction = max(
                0.0, min(1.0, 1.0 - stats.avg_tool_calls_when_used / self.baseline_tool_calls)
            )
        retry_reduction = 0.0
        if stats.avg_attempts_when_used is not None:
            retry_reduction = max(
                0.0, min(1.0, 1.0 - stats.avg_attempts_when_used / self.baseline_attempts)
            )
        return (
            success_rate * self.success_weight
            + tool_reduction * self.tool_reduction_weight
            + retry_reduction * self.retry_reduction_weight
        )
