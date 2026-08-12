"""Capability package, lifecycle, and usage models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SkillStatus = Literal[
    "candidate",
    "trial",
    "active",
    "stale",
    "archived",
    "rejected",
    "superseded",
]


class SkillPermissions(BaseModel):
    model_config = ConfigDict(frozen=True)

    read_workspace: bool = True
    write_workspace: bool = False
    execute: bool = False
    network: bool = False


class SkillVersionRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    version: int


class SkillFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    executable: bool = False


class SkillManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    version: int
    name: str
    description: str
    status: SkillStatus
    triggers: list[str]
    task_families: list[str]
    permissions: SkillPermissions
    source_run_ids: list[str]
    parent_version: int | None = None
    supersedes: list[SkillVersionRef] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    files: list[SkillFile]
    verification_commands: list[list[str]] = Field(default_factory=list)
    operation_key: str | None = None


class GeneratedFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    content: str
    executable: bool = False


class SkillCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    triggers: list[str]
    task_families: list[str]
    skill_md: str
    scripts: list[GeneratedFile] = Field(default_factory=list)
    references: list[GeneratedFile] = Field(default_factory=list)
    templates: list[GeneratedFile] = Field(default_factory=list)
    tests: list[GeneratedFile] = Field(default_factory=list)
    source_run_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    permissions: SkillPermissions = Field(default_factory=SkillPermissions)
    verification_commands: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_skill_sections(self) -> SkillCandidate:
        required = (
            "# Purpose",
            "# When to Use",
            "# Procedure",
            "# Pitfalls",
            "# Verification",
            "# Bundled Resources",
        )
        if not self.skill_md.startswith("---\n") or any(
            section not in self.skill_md for section in required
        ):
            raise ValueError("SKILL.md is missing frontmatter or required sections")
        return self


class RegisteredSkill(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: SkillManifest
    package_path: str


class SkillStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    version: int
    retrieval_count: int = 0
    selected_count: int = 0
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    patch_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = None
    last_modified_at: datetime | None = None
    avg_tool_calls_when_used: float | None = None
    avg_attempts_when_used: float | None = None
    utility_score: float | None = None


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    errors: list[str] = Field(default_factory=list)
    tests_run: int = 0


class ScriptExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
