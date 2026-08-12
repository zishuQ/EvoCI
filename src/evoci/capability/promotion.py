"""Evidence-based trial promotion and failure rejection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from evoci.capability.models import RegisteredSkill, SkillStats, SkillVersionRef
from evoci.capability.registry import CapabilityRegistry

PromotionAction = Literal["hold", "promote", "reject"]


@dataclass(frozen=True, slots=True)
class TrialPromotionPolicy:
    min_uses: int = 2
    min_successes: int = 2
    min_success_rate: float = 0.75
    max_failures: int = 2
    max_exposures_without_use: int = 5

    def evaluate(self, stats: SkillStats) -> PromotionAction:
        if stats.failure_count >= self.max_failures:
            return "reject"
        if stats.use_count == 0 and stats.selected_count >= self.max_exposures_without_use:
            return "reject"
        outcomes = stats.success_count + stats.failure_count
        success_rate = stats.success_count / outcomes if outcomes else 0.0
        if (
            stats.use_count >= self.min_uses
            and stats.success_count >= self.min_successes
            and success_rate >= self.min_success_rate
        ):
            return "promote"
        return "hold"

    def apply(
        self,
        registry: CapabilityRegistry,
        record: RegisteredSkill,
        *,
        operation_key: str | None = None,
    ) -> PromotionAction:
        if operation_key:
            previous = registry.operation_result(operation_key)
            if previous is not None:
                return str(previous["action"])  # type: ignore[return-value]
        current = registry.get(record.manifest.skill_id, record.manifest.version)
        if current is None or current.manifest.status != "trial":
            if operation_key:
                registry.record_operation(operation_key, {"action": "hold"})
            return "hold"
        record = current
        action = self.evaluate(registry.stats(record.manifest.skill_id, record.manifest.version))
        if action == "reject":
            registry.transition(record.manifest.skill_id, record.manifest.version, "rejected")
            if operation_key:
                registry.record_operation(operation_key, {"action": action})
            return action
        if action != "promote":
            if operation_key:
                registry.record_operation(operation_key, {"action": action})
            return action
        registry.transition(record.manifest.skill_id, record.manifest.version, "active")
        lineage = list(record.manifest.supersedes)
        if record.manifest.parent_version is not None:
            lineage.append(
                SkillVersionRef(
                    skill_id=record.manifest.skill_id,
                    version=record.manifest.parent_version,
                )
            )
        for source in lineage:
            old = registry.get(source.skill_id, source.version)
            if old is not None and old.manifest.status in {"active", "stale"}:
                registry.transition(source.skill_id, source.version, "superseded")
        if operation_key:
            registry.record_operation(operation_key, {"action": action})
        return action
