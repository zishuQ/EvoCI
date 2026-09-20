"""Typed split between agent-visible task data and evaluator-only truth."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BenchmarkVariant = Literal["single", "multi", "multi-memory", "evo"]
BenchmarkVerificationStatus = Literal["passed", "failed", "not_available", "infra_error"]
BenchmarkPreflightStatus = Literal["reproduced", "not_reproduced", "not_available", "infra_error"]
BenchmarkTaskStatus = Literal[
    "resolved", "unresolved", "not_evaluable", "skipped", "error", "infra_error"
]


class FailedStep(BaseModel):
    """One normalized failed CI step from a dataset row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = None
    log: str
    command: list[str] | None = None


class NormalizedCIFailure(BaseModel):
    """Canonical agent/evaluator input derived from heterogeneous CI data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_yaml: str
    log_text: str
    failed_steps: list[FailedStep]
    candidate_failed_commands: list[list[str]]


class TaskMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    repo: str
    task_family: str | None = None


class AgentTaskView(BaseModel):
    """The only benchmark model that may enter worker context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    repo_owner: str
    repo_name: str
    workflow_name: str
    workflow_path: str
    sha_fail: str
    ci_failure: NormalizedCIFailure


class GroundTruth(BaseModel):
    """Evaluator-only fields; never nested in AgentTaskView."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    sha_success: str
    diff: str
    changed_files: list[str]
    error_type: str


class DockerTaskSpec(BaseModel):
    """Evaluator-only container configuration; never nested in AgentTaskView."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    official_image: str
    fail_to_pass: list[str]
    pass_to_pass: list[str] = Field(default_factory=list)
    test_patch: str = ""
    protected_files: list[str] = Field(default_factory=list)
    reference_patch: str = ""
    replay_commands: list[list[str]] = Field(default_factory=list)


class PreparedTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_view: AgentTaskView
    workspace_path: str | None = None


class BenchmarkManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    category: str
    source: str = "ci-repair-bench"
    skipped: bool = False
    skip_reason: str | None = None


class FinalWorkspaceChanges(BaseModel):
    """Final patch statistics measured only from the real benchmark workspace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    changed_files: list[str]
    lines_added: int = Field(default=0, ge=0)
    lines_deleted: int = Field(default=0, ge=0)


class BenchmarkCommandResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    command: list[str]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class EvalAttempt(BaseModel):
    """One evaluator container run; retries are recorded, not overwritten."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str
    ftp_results: dict[str, str] = Field(default_factory=dict)
    ptp_results: dict[str, str] = Field(default_factory=dict)
    infra_reason: str | None = None


class BenchmarkVerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: BenchmarkVerificationStatus
    commands: list[BenchmarkCommandResult] = Field(default_factory=list)
    details: str
    failure_class: str | None = None
    ftp_results: dict[str, str] = Field(default_factory=dict)
    ptp_results: dict[str, str] = Field(default_factory=dict)
    attempts: list[EvalAttempt] = Field(default_factory=list)


class BenchmarkPreflightResult(BaseModel):
    """Evaluator-owned evidence that the candidate oracle fails before repair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: BenchmarkPreflightStatus
    commands: list[BenchmarkCommandResult] = Field(default_factory=list)
    details: str
    failure_class: str | None = None
    ftp_results: dict[str, str] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_declared_success: bool
    targeted_verification_passed: bool
    review_passed: bool
    review_performed: bool = False
    benchmark_verification: BenchmarkVerificationResult
    benchmark_verification_status: BenchmarkVerificationStatus
    benchmark_resolved: bool
    gold_file_overlap: float | None = Field(default=None, ge=0, le=1)
    final_workspace_changes: FinalWorkspaceChanges
    attempted_files: list[str] = Field(default_factory=list)
    wall_time: float = Field(ge=0)
    llm_calls: int = Field(default=0, ge=0)
    repair_model_calls: int = Field(default=0, ge=0)
    post_run_model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    repair_tool_calls: int = Field(default=0, ge=0)
    post_run_tool_calls: int = Field(default=0, ge=0)
    failed_tool_calls: int = Field(default=0, ge=0)
    repair_attempts: int = Field(default=0, ge=0)
    files_changed: int = Field(default=0, ge=0)
    lines_changed: int = Field(default=0, ge=0)
    workers_spawned: int = Field(default=0, ge=0)
    parallel_rounds: int = Field(default=0, ge=0)
    investigation_rounds: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    memory_retrieval_count: int = Field(default=0, ge=0)
    memory_selected_count: int = Field(default=0, ge=0)
    memory_used_count: int = Field(default=0, ge=0)
    skills_retrieved: int = Field(default=0, ge=0)
    skills_selected: int = Field(default=0, ge=0)
    skills_used: int = Field(default=0, ge=0)
    skill_created: int = Field(default=0, ge=0)
    skill_updated: int = Field(default=0, ge=0)
    skills_promoted: int = Field(default=0, ge=0)
    skills_rejected: int = Field(default=0, ge=0)
    skills_superseded: int = Field(default=0, ge=0)
    skill_registry_size: int = Field(default=0, ge=0)
    active_skill_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def benchmark_status_is_consistent(self) -> RunMetrics:
        if self.benchmark_verification_status != self.benchmark_verification.status:
            raise ValueError("benchmark verification status fields disagree")
        if self.benchmark_resolved != (self.benchmark_verification_status == "passed"):
            raise ValueError("benchmark_resolved must be derived from benchmark verification")
        if self.files_changed != len(self.final_workspace_changes.changed_files):
            raise ValueError("files_changed must come from final workspace changes")
        final_line_count = (
            self.final_workspace_changes.lines_added + self.final_workspace_changes.lines_deleted
        )
        if self.lines_changed != final_line_count:
            raise ValueError("lines_changed must come from final workspace changes")
        return self


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    variant: BenchmarkVariant
    status: BenchmarkTaskStatus
    category: str | None = None
    metrics: RunMetrics | None = None
    skipped: bool = False
    skip_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def task_status_is_consistent(self) -> BenchmarkResult:
        if self.status == "skipped" and not self.skipped:
            raise ValueError("skipped status requires skipped=true")
        if self.status == "error" and not self.error_message:
            raise ValueError("error status requires error_message")
        if self.status in {"resolved", "unresolved", "not_evaluable", "infra_error"}:
            if self.metrics is None:
                raise ValueError("completed task status requires metrics")
            expected = {
                "passed": "resolved",
                "failed": "unresolved",
                "not_available": "not_evaluable",
                "infra_error": "infra_error",
            }[self.metrics.benchmark_verification_status]
            if self.status != expected:
                raise ValueError("task status disagrees with benchmark verification")
        return self
