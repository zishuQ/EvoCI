"""Long-term memory models kept separate from full trajectories."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Episode(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    repo: str
    task_family: str
    failure_summary: str
    root_cause: str | None = None
    important_evidence: list[str] = Field(default_factory=list)
    attempts: int = Field(ge=0)
    successful_fix_summary: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    hypotheses_attempted: list[str] = Field(default_factory=list)
    verification_failures: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    success: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SemanticMemory(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    namespace: str
    content: str
    importance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    source_run_ids: list[str]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime | None = None

    @model_validator(mode="after")
    def known_namespace(self) -> SemanticMemory:
        if not self.namespace.startswith(("repo:", "family:", "global:")):
            raise ValueError("memory namespace must be repo, family, or global")
        return self


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["semantic", "none"]
    content: str | None = None
    namespace: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence_event_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def semantic_fields_are_complete(self) -> MemoryCandidate:
        if self.type == "semantic" and (not self.content or not self.namespace):
            raise ValueError("semantic candidate requires content and namespace")
        return self


class RetrievalTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    retrieved_ids: list[str]
    selected_ids: list[str]
    used_ids: list[str] = Field(default_factory=list)
    context_chars: int
