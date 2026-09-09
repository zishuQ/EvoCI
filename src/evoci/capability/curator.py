"""Deterministic lifecycle maintenance and validated merge finalization."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evoci.capability.models import SkillCandidate, SkillVersionRef
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.utility import SkillUtilityPolicy, WeightedUtilityPolicy
from evoci.capability.validator import CandidateValidator
from evoci.model.gateway import ModelGateway
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder


@dataclass(frozen=True, slots=True)
class CuratorDecision:
    ref: SkillVersionRef
    action: str
    reason: str


class CuratorReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["none", "review", "merge"]
    rationale: str
    sources: list[SkillVersionRef] = Field(default_factory=list)
    merged_candidate: SkillCandidate | None = None

    @model_validator(mode="after")
    def merge_has_candidate(self) -> CuratorReview:
        if self.action == "merge" and self.merged_candidate is None:
            raise ValueError("merge review requires merged_candidate")
        return self


class CuratedCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    ref: SkillVersionRef
    sources: list[SkillVersionRef]
    validation_passed: bool


class CuratorPipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    maintenance: list[dict[str, object]] = Field(default_factory=list)
    shortlisted_groups: list[list[SkillVersionRef]] = Field(default_factory=list)
    reviews: list[CuratorReview] = Field(default_factory=list)
    candidates: list[CuratedCandidate] = Field(default_factory=list)


class ModelCurator:
    """Use an auxiliary model for semantic overlap and merge proposals only."""

    def __init__(self, registry: CapabilityRegistry, gateway: ModelGateway) -> None:
        self.registry = registry
        self.gateway = gateway

    def package_summary(self, ref: SkillVersionRef) -> dict[str, object]:
        record = self.registry.get(ref.skill_id, ref.version)
        if record is None:
            raise KeyError(f"unknown skill: {ref.skill_id} v{ref.version}")
        package = Path(record.package_path)
        files: list[dict[str, object]] = []
        for declared in record.manifest.files:
            target = package / declared.path
            files.append(
                {
                    "path": declared.path,
                    "sha256": declared.sha256,
                    "executable": declared.executable,
                    "content": target.read_text(encoding="utf-8", errors="replace"),
                }
            )
        return {
            "manifest": record.manifest.model_dump(mode="json"),
            "stats": self.registry.stats(ref.skill_id, ref.version).model_dump(mode="json"),
            "files": files,
            "package_shape": {
                category: sum(
                    file.path.startswith(f"{category}/") for file in record.manifest.files
                )
                for category in ("scripts", "references", "templates", "tests")
            },
            "provenance": record.manifest.source_run_ids,
        }

    async def review(
        self,
        refs: list[SkillVersionRef],
        *,
        run_id: str | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> CuratorReview:
        payload = [self.package_summary(ref) for ref in refs]
        if run_id is not None and recorder is not None:
            key = ",".join(f"{ref.skill_id}:v{ref.version}" for ref in refs)
            recorder.emit(
                run_id=run_id,
                event_type=EventType.MODEL_CALL,
                agent_id="capability-curator",
                event_key=key,
                payload={
                    "phase": "curation",
                    "skills": len(refs),
                    "budget_scope": "post_run",
                },
            )
        return await self.gateway.complete(
            system_prompt=(
                "Review capabilities for semantic duplication, conflict, drift, or useful "
                "generalization. Inspect manifests, all declared package files, tests, utility, "
                "and provenance. Return none/review/merge. A merge must be a complete new "
                "candidate; never overwrite or deactivate a source."
            ),
            user_prompt=json.dumps(payload, default=str),
            response_model=CuratorReview,
            agent_id="capability-curator",
        )


class DeterministicCurator:
    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        utility_policy: SkillUtilityPolicy | None = None,
        active_stale_days: int = 30,
        stale_archive_days: int = 60,
        max_exposures_without_use: int = 5,
    ) -> None:
        self.registry = registry
        self.utility_policy = utility_policy or WeightedUtilityPolicy()
        self.active_stale_days = active_stale_days
        self.stale_archive_days = stale_archive_days
        self.max_exposures_without_use = max_exposures_without_use

    def run(
        self,
        *,
        now: datetime | None = None,
        operation_prefix: str | None = None,
    ) -> list[CuratorDecision]:
        current = now or datetime.now(UTC)
        decisions: list[CuratorDecision] = []
        for record in self.registry.list({"trial", "active", "stale"}):
            manifest = record.manifest
            ref = SkillVersionRef(skill_id=manifest.skill_id, version=manifest.version)
            stats = self.registry.stats(ref.skill_id, ref.version)
            utility = self.utility_policy.score(stats)
            self.registry.set_utility(ref, utility)
            reference_time = stats.last_used_at or stats.created_at
            age = current - reference_time
            operation_key = (
                f"{operation_prefix}:{ref.skill_id}:v{ref.version}" if operation_prefix else None
            )
            if operation_key and self.registry.operation_result(operation_key) is not None:
                continue
            if (
                manifest.status == "trial"
                and stats.use_count == 0
                and stats.selected_count >= self.max_exposures_without_use
            ):
                self.registry.transition(ref.skill_id, ref.version, "rejected")
                decisions.append(
                    CuratorDecision(ref, "reject", "trial was repeatedly selected but unused")
                )
            elif manifest.status == "active" and age >= timedelta(days=self.active_stale_days):
                self.registry.transition(ref.skill_id, ref.version, "stale")
                decisions.append(CuratorDecision(ref, "stale", "active skill has not been used"))
            elif manifest.status == "stale" and age >= timedelta(days=self.stale_archive_days):
                self.registry.transition(ref.skill_id, ref.version, "archived")
                decisions.append(CuratorDecision(ref, "archive", "stale skill remained unused"))
            elif stats.failure_count >= 3 and stats.failure_count > stats.success_count:
                decisions.append(CuratorDecision(ref, "review", "failure association is elevated"))
            if operation_key:
                decision = next((item for item in reversed(decisions) if item.ref == ref), None)
                self.registry.record_operation(
                    operation_key,
                    {
                        "action": decision.action if decision else "hold",
                        "reason": decision.reason if decision else "no lifecycle change",
                    },
                )
        return decisions

    def supersede_sources(self, sources: list[SkillVersionRef], target: SkillVersionRef) -> None:
        target_record = self.registry.get(target.skill_id, target.version)
        if target_record is None or target_record.manifest.status != "active":
            raise ValueError("merge target must be active")
        for source in sources:
            record = self.registry.get(source.skill_id, source.version)
            if record and record.manifest.status in {"active", "stale"}:
                self.registry.transition(source.skill_id, source.version, "superseded")


def _capability_tokens(record_text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{3,}", record_text.lower()))


def duplicate_shortlist(
    registry: CapabilityRegistry,
    *,
    limit: int = 10,
    threshold: float = 0.45,
) -> list[list[SkillVersionRef]]:
    """Use an inverted token index before any bounded semantic comparison."""

    records = registry.list({"trial", "active", "stale"})
    texts: dict[tuple[str, int], str] = {}
    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for record in records:
        manifest = record.manifest
        key = (manifest.skill_id, manifest.version)
        text = " ".join(
            [
                manifest.name,
                manifest.description,
                *manifest.triggers,
                *manifest.task_families,
            ]
        ).lower()
        texts[key] = text
        for token in _capability_tokens(text):
            buckets[token].append(key)
    pair_overlap: dict[tuple[tuple[str, int], tuple[str, int]], int] = defaultdict(int)
    for keys in buckets.values():
        bounded = sorted(set(keys))[:100]
        for index, left in enumerate(bounded):
            for right in bounded[index + 1 :]:
                if left[0] != right[0]:
                    pair_overlap[(left, right)] += 1
    scored: list[tuple[float, tuple[str, int], tuple[str, int]]] = []
    for (left, right), overlap in pair_overlap.items():
        if overlap < 2:
            continue
        ratio = SequenceMatcher(None, texts[left], texts[right]).ratio()
        score = min(1.0, ratio + min(overlap, 5) * 0.05)
        if score >= threshold:
            scored.append((score, left, right))
    scored.sort(reverse=True)
    return [
        [
            SkillVersionRef(skill_id=left[0], version=left[1]),
            SkillVersionRef(skill_id=right[0], version=right[1]),
        ]
        for _, left, right in scored[:limit]
    ]


class CuratorPipeline:
    """Deterministic maintenance followed by bounded model-assisted merge trials."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        validator: CandidateValidator,
        *,
        model_curator: ModelCurator | None = None,
        shortlist_limit: int = 10,
        max_exposures_without_use: int = 5,
    ) -> None:
        self.registry = registry
        self.validator = validator
        self.model_curator = model_curator
        self.shortlist_limit = shortlist_limit
        self.max_exposures_without_use = max_exposures_without_use

    async def run(
        self,
        *,
        run_id: str | None = None,
        recorder: TrajectoryRecorder | None = None,
    ) -> CuratorPipelineResult:
        final_key = f"curator:{run_id}:result" if run_id else None
        if final_key:
            existing = self.registry.operation_result(final_key)
            if existing is not None:
                return CuratorPipelineResult.model_validate(existing)
        maintenance = DeterministicCurator(
            self.registry,
            max_exposures_without_use=self.max_exposures_without_use,
        ).run(operation_prefix=f"curator:{run_id}:maintenance" if run_id else None)
        groups = duplicate_shortlist(self.registry, limit=self.shortlist_limit)
        reviews: list[CuratorReview] = []
        candidates: list[CuratedCandidate] = []
        if self.model_curator is not None:
            for group in groups:
                group_key = ",".join(f"{ref.skill_id}:v{ref.version}" for ref in group)
                review_key = f"curator:{run_id}:review:{group_key}" if run_id else None
                prior_review = self.registry.operation_result(review_key) if review_key else None
                if prior_review is not None:
                    review = CuratorReview.model_validate(prior_review)
                else:
                    review = await self.model_curator.review(
                        group, run_id=run_id, recorder=recorder
                    )
                    if review_key:
                        self.registry.record_operation(review_key, review.model_dump(mode="json"))
                reviews.append(review)
                if review.action != "merge" or review.merged_candidate is None:
                    continue
                sources = review.sources or group
                allowed = {(ref.skill_id, ref.version) for ref in group}
                sources = [ref for ref in sources if (ref.skill_id, ref.version) in allowed]
                if len(sources) < 2:
                    continue
                created = self.registry.create_candidate(
                    review.merged_candidate,
                    supersedes=sources,
                    operation_key=(f"curator:{run_id}:candidate:{group_key}" if run_id else None),
                )
                validation = await self.validator.avalidate_to_trial(
                    created.manifest.skill_id, created.manifest.version
                )
                candidates.append(
                    CuratedCandidate(
                        ref=SkillVersionRef(
                            skill_id=created.manifest.skill_id,
                            version=created.manifest.version,
                        ),
                        sources=sources,
                        validation_passed=validation.passed,
                    )
                )
        result = CuratorPipelineResult(
            maintenance=[
                {
                    "ref": decision.ref.model_dump(),
                    "action": decision.action,
                    "reason": decision.reason,
                }
                for decision in maintenance
            ],
            shortlisted_groups=groups,
            reviews=reviews,
            candidates=candidates,
        )
        if final_key:
            self.registry.record_operation(final_key, result.model_dump(mode="json"))
        return result
