"""Role-specific interfaces used by the orchestration graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    VerificationResult,
    WorkerResult,
)
from evoci.tools.policy import WorkerCapabilities


@dataclass(frozen=True, slots=True)
class AgentContext:
    run_id: str
    repo: RepoSpec
    failure: CIFailure
    workspace_path: str
    invocation_id: str
    memories: tuple[MemoryHit, ...] = ()
    skills: tuple[SkillHit, ...] = ()
    previous_review_blockers: tuple[str, ...] = ()
    previous_attempt_summary: str | None = None


class Coordinator(Protocol):
    async def plan(
        self,
        *,
        context: AgentContext,
        evidence: list[EvidenceItem],
        round_number: int,
        remaining_task_budget: int,
    ) -> InvestigationPlan: ...


class Investigator(Protocol):
    async def run(
        self,
        *,
        task: InvestigationTask,
        context: AgentContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerResult: ...


class Diagnoser(Protocol):
    async def diagnose(
        self, *, context: AgentContext, evidence: list[EvidenceItem]
    ) -> Diagnosis: ...


class Fixer(Protocol):
    async def propose(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        evidence: list[EvidenceItem],
        previous_verification: VerificationResult | None,
    ) -> FixerOutput: ...


class Reviewer(Protocol):
    async def review(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        patch: FixerOutput,
        verification: VerificationResult,
    ) -> ReviewResult: ...


@dataclass(frozen=True, slots=True)
class AgentSuite:
    coordinator: Coordinator
    investigator: Investigator
    diagnoser: Diagnoser
    fixer: Fixer
    reviewer: Reviewer
