"""Reducer-backed parent graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from evoci.domain.models import (
    CIFailure,
    Diagnosis,
    EvidenceItem,
    FixerOutput,
    InvestigationPlan,
    InvestigationTask,
    MemoryHit,
    RepoSpec,
    ReviewResult,
    SkillHit,
    SkillRef,
    SkillUsage,
    VerificationResult,
)
from evoci.runtime.events import RunEvent


class EvoCIState(TypedDict, total=False):
    run_id: str
    task_id: str
    repo: RepoSpec
    ci_failure: CIFailure
    workspace_path: str

    phase: str
    investigation_round: int
    repair_attempt: int
    investigation_task_count: int
    investigation_plan: InvestigationPlan | None
    worker_task: InvestigationTask
    completed_task_ids: Annotated[list[str], operator.add]
    evidence: Annotated[list[EvidenceItem], operator.add]
    budget_failures: Annotated[list[str], operator.add]
    diagnosis: Diagnosis | None

    fixer_output: FixerOutput | None
    verification: VerificationResult | None
    verification_history: Annotated[list[VerificationResult], operator.add]
    review: ReviewResult | None
    approved: bool | None
    attempt_baseline: dict[str, str | None]
    attempt_written: dict[str, str | None]
    previous_review_blockers: list[str]
    previous_attempt_summary: str | None

    retrieved_memories: list[MemoryHit]
    retrieved_skills: list[SkillHit]
    selected_memory_ids: Annotated[list[str], operator.add]
    used_memory_ids: Annotated[list[str], operator.add]
    selected_skill_refs: Annotated[list[SkillRef], operator.add]
    used_skill_refs: Annotated[list[SkillRef], operator.add]
    used_skills: list[SkillUsage]
    events: Annotated[list[RunEvent], operator.add]

    episode_id: str | None
    learning_decision: dict[str, object] | None
    candidate_skill_id: str | None
    learning_errors: list[dict[str, str]]
    learning_deferred: bool

    status: Literal["running", "waiting_approval", "success", "failed"]
    failure_reason: str | None
