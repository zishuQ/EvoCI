"""Typed trajectory events."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    RUN_STARTED = "RunStarted"
    AGENT_STARTED = "AgentStarted"
    AGENT_COMPLETED = "AgentCompleted"
    MODEL_CALL = "ModelCall"
    MODEL_USAGE = "ModelUsage"
    TOOL_CALL = "ToolCall"
    TOOL_RESULT = "ToolResult"
    EVIDENCE_CREATED = "EvidenceCreated"
    ATTEMPT_FAILED = "AttemptFailed"
    DIAGNOSIS_CREATED = "DiagnosisCreated"
    PATCH_CREATED = "PatchCreated"
    VERIFICATION_STARTED = "VerificationStarted"
    VERIFICATION_COMPLETED = "VerificationCompleted"
    INTERRUPT_CREATED = "InterruptCreated"
    INTERRUPT_RESOLVED = "InterruptResolved"
    MEMORY_RETRIEVED = "MemoryRetrieved"
    MEMORY_SELECTED = "MemorySelected"
    MEMORY_USED = "MemoryUsed"
    MEMORY_CREATED = "MemoryCreated"
    SKILL_RETRIEVED = "SkillRetrieved"
    SKILL_SELECTED = "SkillSelected"
    SKILL_INVOCATION_REJECTED = "SkillInvocationRejected"
    SKILL_USED = "SkillUsed"
    SKILL_CANDIDATE_CREATED = "SkillCandidateCreated"
    SKILL_UPDATED = "SkillUpdated"
    LEARNING_ERROR = "LearningError"
    RUN_COMPLETED = "RunCompleted"
    RUN_FAILED = "RunFailed"


class RunEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: EventType
    agent_id: str | None = None
    invocation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
