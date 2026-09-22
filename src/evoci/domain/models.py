"""Core domain models for investigation, repair, verification, and review."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

InvestigationRole = Literal["log", "repository", "workflow", "dependency", "test", "specialist"]
FailureClass = Literal["repair", "model", "budget", "policy", "infrastructure"]
WorkerKind = Literal["investigate", "repair"]
SupervisorAction = Literal["dispatch", "stop"]
WorkerTaskStatus = Literal["completed", "blocked", "budget_exhausted"]
ORCHESTRATION_SCHEMA_VERSION = 3
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


class TaskBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_model_calls: int = Field(ge=1)
    max_tool_calls: int = Field(ge=1)


class WorkerTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    kind: WorkerKind
    objective: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    hypothesis: str | None = None
    read_scope: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    fact_refs: list[str] = Field(default_factory=list)
    recommended_skill_refs: list[SkillRef] = Field(default_factory=list)
    budget: TaskBudget | None = None
    resume_candidate_id: str | None = None

    @model_validator(mode="after")
    def scope_matches_kind(self) -> WorkerTask:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if self.kind == "investigate" and self.write_scope:
            raise ValueError("investigate tasks cannot declare write_scope")
        if self.depends_on:
            raise ValueError("depends_on is removed; dispatch exactly one independent task")
        return self


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: SupervisorAction
    reasoning_summary: str
    tasks: list[WorkerTask] = Field(default_factory=list)
    stop_reason: str | None = None

    @model_validator(mode="after")
    def action_payload_is_complete(self) -> SupervisorDecision:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("worker task IDs must be unique")
        if self.action == "dispatch" and len(self.tasks) != 1:
            raise ValueError("dispatch requires exactly one task")
        if self.action == "stop":
            if self.tasks:
                raise ValueError("stop cannot include tasks")
            if not (self.stop_reason or "").strip():
                raise ValueError("stop requires stop_reason")
        return self


class WorkerExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    status: WorkerTaskStatus
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    base_revision: str | None = None
    snapshot_id: str | None = None
    patch_artifact_ref: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    command_result_refs: list[str] = Field(default_factory=list)
    used_memory_ids: list[str] = Field(default_factory=list)
    used_skill_refs: list[SkillRef] = Field(default_factory=list)


class FailedCandidateRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    task_id: str
    batch: int
    baseline_snapshot_id: str
    artifact_ref: str
    changed_files: list[str] = Field(default_factory=list)


class SkillCatalogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    name: str
    description: str


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


class VerificationCommandSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    argv: list[str] = Field(min_length=1, max_length=32)
    cwd: str = "."

    @field_validator("argv")
    @classmethod
    def argv_items_are_strings(cls, value: list[str]) -> list[str]:
        cleaned = [str(item) for item in value]
        if any(not item.strip() for item in cleaned):
            raise ValueError("verification argv items must be non-empty")
        return cleaned

    @field_validator("cwd")
    @classmethod
    def normalize_cwd(cls, value: str) -> str:
        return normalize_repo_cwd(value)


def normalize_repo_cwd(cwd: str | None) -> str:
    raw = "." if cwd is None else str(cwd).strip() or "."
    if Path(raw).is_absolute() or raw.startswith("/"):
        raise ValueError(f"verification cwd must be workspace-relative: {cwd}")
    parts = [part for part in PurePosixPath(raw.replace("\\", "/")).parts if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ValueError(f"verification cwd escapes workspace: {cwd}")
    if not parts:
        return "."
    return str(PurePosixPath(*parts))


def parse_verification_command(raw: object) -> VerificationCommandSpec:
    if isinstance(raw, VerificationCommandSpec):
        return raw
    if isinstance(raw, dict):
        return VerificationCommandSpec.model_validate(raw)
    if isinstance(raw, (list, tuple)) and raw and all(isinstance(item, str) for item in raw):
        return VerificationCommandSpec(argv=list(raw), cwd=".")
    raise ValueError(f"invalid verification command: {raw!r}")


def parse_verification_plan(raw: object) -> list[VerificationCommandSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("verification_plan must be a list")
    return [parse_verification_command(item) for item in raw]


class PatchProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    changed_files: list[str]
    commands_run: list[str] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"]
    verification_plan: list[VerificationCommandSpec] = Field(default_factory=list)

    @field_validator("verification_plan", mode="before")
    @classmethod
    def coerce_verification_plan(cls, value: object) -> list[VerificationCommandSpec]:
        return parse_verification_plan(value)


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


VerificationCommandSource = Literal["mandatory", "supplementary"]
VerificationStatus = Literal[
    "passed", "failed", "incomplete", "unavailable", "infra_error", "inconclusive"
]
ReviewStatus = Literal["passed", "failed", "not_performed"]


class VerificationCommandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    source: VerificationCommandSource = "mandatory"
    executed: bool = True
    skip_reason: str | None = None
    cwd: str = "."


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    status: VerificationStatus = "failed"
    level: Literal["targeted", "repository", "full_ci"] = "targeted"
    commands: list[VerificationCommandResult]
    expected_count: int = 0
    executed_count: int = 0
    incomplete_reason: str | None = None
    incomplete_cause: Literal["budget", "execution"] | None = None
    oracle_source: Literal["harness", "none"] = "harness"

    @model_validator(mode="before")
    @classmethod
    def normalize_completeness(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        commands = data.get("commands") or []
        passed = bool(data.get("passed"))
        status = data.get("status", "failed")
        if passed and status == "failed" and not data.get("incomplete_reason"):
            status = "passed"
        expected = data.get("expected_count") or len(commands)
        if data.get("executed_count"):
            executed_count = data["executed_count"]
        else:
            executed_count = 0
            for command in commands:
                if isinstance(command, dict):
                    executed_count += int(command.get("executed", True))
                else:
                    executed_count += int(getattr(command, "executed", True))
        payload = dict(data)
        payload["status"] = status
        payload["expected_count"] = expected
        payload["executed_count"] = executed_count
        return payload

    @model_validator(mode="after")
    def verify_success_contract(self) -> VerificationResult:
        if self.passed and self.status != "passed":
            raise ValueError("passed=true is only valid for verified success")
        if self.status == "passed" and not self.passed:
            raise ValueError("status=passed requires passed=true")
        return self


class ReviewResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    accepted: bool
    performed: bool = True
    status: ReviewStatus = "failed"
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    used_memory_ids: list[str] = Field(default_factory=list)
    used_skill_refs: list[SkillRef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_review_status(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        performed = data.get("performed", True)
        accepted = bool(data.get("accepted"))
        payload = dict(data)
        if not performed:
            payload["accepted"] = False
            payload["status"] = "not_performed"
        elif accepted:
            payload["status"] = "passed"
        else:
            payload["status"] = "failed"
        return payload

    @model_validator(mode="after")
    def unperformed_review_is_not_accepted(self) -> ReviewResult:
        if not self.performed and self.accepted:
            raise ValueError("unperformed review cannot be accepted")
        return self


class MemoryHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_id: str
    namespace: str
    content: str
    score: float


class SkillHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    name: str
    description: str
    skill_md: str
    memory: list[str] = Field(default_factory=list)
    score: float
    resources: list[str] = Field(default_factory=list)


class SkillCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    entries: list[SkillCatalogEntry] = Field(default_factory=list)
    omitted_count: int = 0
    context_chars: int = 0


class SkillUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    agent_id: str
    used_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
