"""Reducer-backed parent graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from evoci.domain.models import (
    CIFailure,
    EvidenceItem,
    FailedCandidateRef,
    FailureClass,
    FixerOutput,
    MemoryHit,
    RepoSpec,
    ReviewResult,
    SkillCatalogEntry,
    SkillHit,
    SkillRef,
    SkillUsage,
    SupervisorDecision,
    VerificationResult,
    WorkerExecutionResult,
    WorkerTask,
)
from evoci.runtime.events import RunEvent

ORCHESTRATION_SCHEMA_VERSION = 3


class LegacyOrchestrationError(RuntimeError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "Old graph checkpoint detected. Start a new run or campaign; "
                "supervisor-worker orchestration does not migrate previous checkpoints."
            )
        )


class EvoCIState(TypedDict, total=False):
    schema_version: int
    run_id: str
    task_id: str
    repo: RepoSpec
    ci_failure: CIFailure
    workspace_path: str
    campaign_provenance: dict[str, object]

    phase: str
    supervisor_batch: int
    decision: SupervisorDecision | None
    decision_history: list[str]
    pending_tasks: list[WorkerTask]
    current_wave: list[WorkerTask]
    remaining_waves: list[list[WorkerTask]]
    worker_task: WorkerTask
    completed_task_ids: Annotated[list[str], operator.add]
    worker_results: Annotated[list[WorkerExecutionResult], operator.add]
    evidence: Annotated[list[EvidenceItem], operator.add]
    budget_failures: Annotated[list[str], operator.add]

    batch_snapshot_path: str | None
    batch_snapshot_id: str | None
    integrated_snapshot_id: str | None
    verification_snapshot_id: str | None
    worker_baseline_id: str | None
    worker_memories: list[MemoryHit]
    recommended_skills: list[SkillCatalogEntry]
    resume_artifact_ref: str | None
    previous_attempt_summary: str | None

    fixer_output: FixerOutput | None
    verification: VerificationResult | None
    verification_history: Annotated[list[VerificationResult], operator.add]
    review: ReviewResult | None
    approved: bool | None
    attempt_baseline: dict[str, dict[str, str | int | None] | str | None]
    attempt_written: dict[str, str | None]
    attempt_targets: dict[str, str | None]
    failed_candidates: list[FailedCandidateRef]
    batch_conflict: str | None

    retrieved_memories: list[MemoryHit]
    retrieved_skills: list[SkillHit]
    skill_catalog: list[SkillCatalogEntry]
    selected_memory_ids: Annotated[list[str], operator.add]
    used_memory_ids: Annotated[list[str], operator.add]
    selected_skill_refs: Annotated[list[SkillRef], operator.add]
    used_skill_refs: Annotated[list[SkillRef], operator.add]
    used_skills: list[SkillUsage]
    recommended_skill_ids: Annotated[list[str], operator.add]
    events: Annotated[list[RunEvent], operator.add]

    episode_id: str | None
    learning_decision: dict[str, object] | None
    candidate_skill_id: str | None
    learning_errors: list[dict[str, str]]
    learning_deferred: bool
    learning_payload_chars: int
    usage_complete: bool

    status: Literal["running", "waiting_approval", "success", "failed"]
    failure_reason: str | None
    failure_class: FailureClass | None
    failure_classes: Annotated[list[str], operator.add]
    failure_stage: str | None
