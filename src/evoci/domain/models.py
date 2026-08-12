"""Core domain models for investigation, repair, verification, and review."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

InvestigationRole = Literal["log", "repository", "workflow", "dependency", "test", "specialist"]
EvidenceKind = Literal[
    "ci_log",
    "source_code",
    "git_history",
    "git_diff",
    "workflow",
    "dependency",
    "test_result",
    "runtime",
]


class RepoSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    owner: str = "local"
    name: str
    base_commit: str = "HEAD"

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class CIFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    log_excerpt: str
    failed_commands: list[list[str]] = Field(default_factory=list)
    workflow_yaml: str = ""
    workflow_path: str | None = None
    task_family: str = "unknown"


class InvestigationTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    role: InvestigationRole
    objective: str
    expected_evidence: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=10)


class InvestigationPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    tasks: list[InvestigationTask]
    reasoning_summary: str

    @model_validator(mode="after")
    def unique_task_ids(self) -> InvestigationPlan:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("investigation task IDs must be unique")
        return self


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source_agent: str
    kind: EvidenceKind
    claim: str
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    command: str | None = None
    excerpt: str | None = None
    confidence: float = Field(ge=0, le=1)
    artifact_ref: str | None = None


class SkillRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    version: int


class WorkerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    summary: str
    evidence: list[EvidenceItem]
    used_memory_ids: list[str] = Field(default_factory=list)
    used_skill_refs: list[SkillRef] = Field(default_factory=list)


class Hypothesis(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_cause: str
    evidence_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    affected_files: list[str]
    proposed_action: str


class Diagnosis(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary: Hypothesis
    alternatives: list[Hypothesis] = Field(default_factory=list)
    needs_more_evidence: bool
    missing_evidence: list[str] = Field(default_factory=list)


class FileEdit(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    content: str | None = None
    delete: bool = False
    expected_sha256: str | None = None

    @model_validator(mode="after")
    def content_matches_operation(self) -> FileEdit:
        if self.delete == (self.content is not None):
            raise ValueError("exactly one of delete=true or content must be provided")
        return self


class PatchProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    changed_files: list[str]
    commands_run: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"]
    verification_plan: list[list[str]]


class FixerOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    proposal: PatchProposal
    edits: list[FileEdit]
    used_memory_ids: list[str] = Field(default_factory=list)
    used_skill_refs: list[SkillRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def edits_match_changed_files(self) -> FixerOutput:
        edit_paths = [edit.path for edit in self.edits]
        if sorted(edit_paths) != sorted(self.proposal.changed_files):
            raise ValueError("proposal changed_files must exactly match edit paths")
        return self


class VerificationCommandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    level: Literal["targeted", "repository", "full_ci"] = "targeted"
    commands: list[VerificationCommandResult]


class ReviewResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    accepted: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    used_memory_ids: list[str] = Field(default_factory=list)
    used_skill_refs: list[SkillRef] = Field(default_factory=list)


class MemoryHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    namespace: str
    content: str
    score: float


class SkillHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    version: int
    name: str
    description: str
    skill_md: str
    score: float
    resources: list[str] = Field(default_factory=list)


class SkillUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    version: int
    agent_id: str
    used_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
