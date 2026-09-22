"""Capability package, usage, and skill-memory models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkillPermissions(BaseModel):
    model_config = ConfigDict(frozen=True)

    read_workspace: bool = True
    write_workspace: bool = False
    execute: bool = False
    network: bool = False


class SkillFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    executable: bool = False


class SkillManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    name: str
    description: str
    triggers: list[str]
    task_families: list[str]
    permissions: SkillPermissions
    source_run_ids: list[str]
    enabled: bool = True
    files: list[SkillFile]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GeneratedFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    content: str
    executable: bool = False


class SkillSpec(BaseModel):
    """Structured skill content visible to the model. The runtime renders SKILL.md."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Short skill name")
    description: str = Field(description="One-line description of what the skill does")
    purpose: str = Field(description="Why the skill exists")
    when_to_use: str = Field(description="When an agent should select this skill")
    procedure: str = Field(description="Step-by-step procedure")
    pitfalls: str = Field(description="Mistakes to avoid")
    verification: str = Field(
        description=(
            "Repository-level guidance for future agents on how to verify a repair after using "
            "this skill. Written into SKILL.md; never executed during skill installation."
        )
    )
    bundled_resources: str = Field(
        default="No bundled files are required.",
        description="Declared scripts, references, and templates",
    )


class SkillCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    triggers: list[str]
    task_families: list[str]
    skill_md: str = ""
    spec: SkillSpec | None = None
    scripts: list[GeneratedFile] = Field(default_factory=list)
    references: list[GeneratedFile] = Field(default_factory=list)
    templates: list[GeneratedFile] = Field(default_factory=list)
    tests: list[GeneratedFile] = Field(default_factory=list)
    source_run_ids: list[str]
    confidence: float = Field(ge=0, le=1)
    permissions: SkillPermissions = Field(default_factory=SkillPermissions)

    @model_validator(mode="before")
    @classmethod
    def render_structured_spec(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        spec = data.get("spec")
        if spec is None:
            return data
        from evoci.capability.render import render_skill_markdown

        parsed = spec if isinstance(spec, SkillSpec) else SkillSpec.model_validate(spec)
        payload = dict(data)
        payload["skill_md"] = render_skill_markdown(parsed)
        payload["name"] = payload.get("name") or parsed.name
        payload["description"] = payload.get("description") or parsed.description
        return payload

    @model_validator(mode="after")
    def required_skill_sections(self) -> SkillCandidate:
        from evoci.capability.render import REQUIRED_SKILL_SECTIONS

        if not self.skill_md.startswith("---\n") or any(
            section not in self.skill_md for section in REQUIRED_SKILL_SECTIONS
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
    retrieval_count: int = 0
    selected_count: int = 0
    use_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    patch_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = None


class SkillMemoryEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    repository: str
    task_summary: str
    outcome: Literal["success", "failure"]
    lesson: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    errors: list[str] = Field(default_factory=list)
    tests_run: int = 0
    test_files: int = 0
    behavior_verified: bool = False


class ScriptExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    observed_revision: int | None = None
