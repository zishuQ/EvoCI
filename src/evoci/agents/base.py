"""Role-specific interfaces used by the orchestration graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from evoci.domain.models import (
    CIFailure,
    FailedCandidateRef,
    FileEdit,
    MemoryHit,
    RepoSpec,
    SkillCatalogEntry,
    SupervisorDecision,
    VerificationCommandSpec,
    VerificationResult,
    WorkerExecutionResult,
    WorkerTask,
    parse_verification_plan,
)
from evoci.tools.policy import WorkerCapabilities


@dataclass(frozen=True, slots=True)
class SupervisorContext:
    run_id: str
    repo: RepoSpec
    failure: CIFailure
    workspace_path: str
    invocation_id: str
    memories: tuple[MemoryHit, ...] = ()
    skills: tuple[SkillCatalogEntry, ...] = ()
    previous_decisions: tuple[str, ...] = ()
    worker_result_summaries: tuple[dict[str, object], ...] = ()
    last_verification: VerificationResult | None = None
    verification_snapshot_id: str | None = None
    remaining_batches: int = 0
    remaining_model_calls: int = 0
    remaining_tool_calls: int = 0
    failed_candidates: tuple[FailedCandidateRef, ...] = ()
    usage_complete: bool = True


@dataclass(frozen=True, slots=True)
class WorkerContext:
    run_id: str
    repo: RepoSpec
    failure: CIFailure
    workspace_path: str
    invocation_id: str
    task: WorkerTask
    memories: tuple[MemoryHit, ...] = ()
    recommended_skills: tuple[SkillCatalogEntry, ...] = ()
    baseline_snapshot_id: str = ""
    previous_attempt_summary: str | None = None
    resume_artifact_ref: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerRun:
    result: WorkerExecutionResult
    edits: list[FileEdit] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    verification_plan: list[VerificationCommandSpec] = field(default_factory=list)
    risk: str = "low"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verification_plan", parse_verification_plan(self.verification_plan)
        )


class Supervisor(Protocol):
    async def decide(self, *, context: SupervisorContext) -> SupervisorDecision: ...


class Worker(Protocol):
    async def execute(
        self,
        *,
        context: WorkerContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerRun: ...


@dataclass(frozen=True, slots=True)
class AgentSuite:
    supervisor: Supervisor
    worker: Worker
