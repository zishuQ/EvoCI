"""Authoritative event recording and trajectory projection."""

from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from evoci.domain.models import SkillRef, VerificationResult
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.events import EventType, RunEvent


class AgentTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    model_calls: int
    tool_calls: int
    failed_tool_calls: int


class ToolTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    agent_id: str
    invocation_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool | None = None
    exit_code: int | None = None
    error: str | None = None
    duration_seconds: float | None = None
    result: Any = None


class AttemptSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal[
        "tool_failure",
        "verification_failure",
        "diagnosis_rejected",
        "review_rejected",
        "patch_rejected",
    ]
    agent_id: str | None = None
    reason: str
    iteration: int | None = None


class SkillUseTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    version: int
    agent_id: str
    invocation_id: str | None = None
    resource: str | None = None
    execution_success: bool | None = None


class TrajectoryView(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    agents: list[AgentTrace]
    tool_calls: list[ToolTrace]
    failed_attempts: list[AttemptSummary]
    evidence: list[dict[str, Any]]
    diagnosis_history: list[dict[str, Any]]
    patch_history: list[dict[str, Any]]
    created_files: list[str]
    modified_files: list[str]
    verification_history: list[VerificationResult]
    skills_retrieved: list[SkillRef]
    skills_selected: list[SkillRef]
    skills_used: list[SkillRef]
    skill_use_traces: list[SkillUseTrace] = Field(default_factory=list)
    memories_retrieved: list[str]
    memories_selected: list[str]
    memories_used: list[str]
    final_status: Literal["success", "failed"]
    failure_reason: str | None = None

    @property
    def model_call_count(self) -> int:
        return sum(agent.model_calls for agent in self.agents)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def reusable_script_created(self) -> bool:
        helper_markers = ("helper", "diagnostic", "inspect", "analy", "script", "tool")
        return any(
            path.endswith((".py", ".sh"))
            and any(marker in path.lower() for marker in helper_markers)
            for path in self.created_files
        )


class TrajectoryRecorder:
    """Append events once and project metrics from those same events."""

    def __init__(self, store: SQLiteEventStore | None = None) -> None:
        self.store = store
        self._events: dict[str, list[RunEvent]] = defaultdict(list)
        self._lock = RLock()

    def emit(
        self,
        *,
        run_id: str,
        event_type: EventType,
        agent_id: str | None = None,
        invocation_id: str | None = None,
        payload: dict[str, Any] | None = None,
        event_key: str | None = None,
    ) -> RunEvent:
        event_payload = dict(payload or {})
        if invocation_id is not None:
            event_payload["invocation_id"] = invocation_id
        event = RunEvent(
            event_id=(
                f"{run_id}:{agent_id or 'harness'}:{invocation_id or 'default'}:"
                f"{event_type.value}:{event_key}"
                if event_key
                else str(uuid4())
            ),
            run_id=run_id,
            type=event_type,
            agent_id=agent_id,
            invocation_id=invocation_id,
            payload=event_payload,
        )
        with self._lock:
            if not any(existing.event_id == event.event_id for existing in self._events[run_id]):
                self._events[run_id].append(event)
            if self.store is not None:
                self.store.append(event)
        return event

    def events(self, run_id: str) -> list[RunEvent]:
        if self.store is not None:
            return self.store.list(run_id)
        with self._lock:
            return list(self._events.get(run_id, []))

    def count(
        self,
        run_id: str,
        event_type: EventType,
        *,
        agent_id: str | None = None,
    ) -> int:
        return sum(
            event.type == event_type and (agent_id is None or event.agent_id == agent_id)
            for event in self.events(run_id)
        )

    def build_view(
        self,
        *,
        run_id: str,
        verification_history: list[VerificationResult],
        final_status: Literal["success", "failed"],
        failure_reason: str | None,
    ) -> TrajectoryView:
        events = self.events(run_id)
        per_agent: dict[str, dict[str, int]] = defaultdict(
            lambda: {"model": 0, "tool": 0, "failed": 0}
        )
        calls: dict[str, ToolTrace] = {}
        attempts: list[AttemptSummary] = []
        evidence: list[dict[str, Any]] = []
        diagnoses: list[dict[str, Any]] = []
        patches: list[dict[str, Any]] = []
        created_files: set[str] = set()
        modified_files: set[str] = set()
        memory_ids: dict[EventType, set[str]] = defaultdict(set)
        skill_refs: dict[EventType, set[tuple[str, int]]] = defaultdict(set)
        skill_use_traces: list[SkillUseTrace] = []
        for event in events:
            agent = event.agent_id or "harness"
            if event.type == EventType.MODEL_CALL:
                per_agent[agent]["model"] += 1
            elif event.type == EventType.TOOL_CALL:
                per_agent[agent]["tool"] += 1
                call_id = str(event.payload.get("call_id", event.event_id))
                invocation_id = event.invocation_id
                trace_key = f"{agent}:{invocation_id or 'default'}:{call_id}"
                calls[trace_key] = ToolTrace(
                    call_id=call_id,
                    agent_id=agent,
                    invocation_id=(str(invocation_id) if invocation_id is not None else None),
                    tool_name=str(event.payload.get("tool_name", "unknown")),
                    arguments=dict(event.payload.get("arguments", {})),
                )
            elif event.type == EventType.TOOL_RESULT:
                call_id = str(event.payload.get("call_id", ""))
                invocation_id = event.invocation_id
                trace_key = f"{agent}:{invocation_id or 'default'}:{call_id}"
                current = calls.get(trace_key)
                if current is not None:
                    calls[trace_key] = current.model_copy(
                        update={
                            "success": bool(event.payload.get("success")),
                            "exit_code": event.payload.get("exit_code"),
                            "error": event.payload.get("error"),
                            "duration_seconds": event.payload.get("duration_seconds"),
                            "result": event.payload.get("result"),
                        }
                    )
                if not bool(event.payload.get("success")):
                    per_agent[agent]["failed"] += 1
                    attempts.append(
                        AttemptSummary(
                            kind="tool_failure",
                            agent_id=agent,
                            reason=str(event.payload.get("error") or "tool returned failure"),
                            iteration=event.payload.get("iteration"),
                        )
                    )
                created_files.update(str(path) for path in event.payload.get("created_files", []))
                modified_files.update(str(path) for path in event.payload.get("modified_files", []))
            elif event.type == EventType.ATTEMPT_FAILED:
                attempts.append(AttemptSummary.model_validate(event.payload))
            elif event.type == EventType.EVIDENCE_CREATED:
                evidence.append(dict(event.payload.get("evidence", {})))
            elif event.type == EventType.DIAGNOSIS_CREATED:
                diagnoses.append(dict(event.payload.get("diagnosis", {})))
            elif event.type == EventType.PATCH_CREATED:
                patches.append(dict(event.payload.get("fixer_output", {})))
            elif event.type in {
                EventType.MEMORY_RETRIEVED,
                EventType.MEMORY_SELECTED,
                EventType.MEMORY_USED,
            }:
                memory_ids[event.type].update(
                    str(value) for value in event.payload.get("memory_ids", [])
                )
            elif event.type in {
                EventType.SKILL_RETRIEVED,
                EventType.SKILL_SELECTED,
                EventType.SKILL_USED,
            }:
                raw_refs = event.payload.get("skills", [])
                for raw in raw_refs:
                    if isinstance(raw, dict):
                        skill_id = str(raw["skill_id"])
                        version = int(raw["version"])
                        skill_refs[event.type].add((skill_id, version))
                        if event.type == EventType.SKILL_USED:
                            skill_use_traces.append(
                                SkillUseTrace(
                                    skill_id=skill_id,
                                    version=version,
                                    agent_id=agent,
                                    invocation_id=event.invocation_id,
                                    resource=event.payload.get("resource"),
                                    execution_success=event.payload.get("success"),
                                )
                            )
                if event.type == EventType.SKILL_USED and "skill_id" in event.payload:
                    skill_id = str(event.payload["skill_id"])
                    version = int(event.payload["version"])
                    skill_refs[event.type].add((skill_id, version))
                    skill_use_traces.append(
                        SkillUseTrace(
                            skill_id=skill_id,
                            version=version,
                            agent_id=agent,
                            invocation_id=event.invocation_id,
                            resource=event.payload.get("resource"),
                            execution_success=event.payload.get("success"),
                        )
                    )
        agents = [
            AgentTrace(
                agent_id=agent_id,
                model_calls=counts["model"],
                tool_calls=counts["tool"],
                failed_tool_calls=counts["failed"],
            )
            for agent_id, counts in sorted(per_agent.items())
        ]

        def refs(event_type: EventType) -> list[SkillRef]:
            return [
                SkillRef(skill_id=skill_id, version=version)
                for skill_id, version in sorted(skill_refs[event_type])
            ]

        return TrajectoryView(
            run_id=run_id,
            agents=agents,
            tool_calls=list(calls.values()),
            failed_attempts=attempts,
            evidence=evidence,
            diagnosis_history=diagnoses,
            patch_history=patches,
            created_files=sorted(created_files),
            modified_files=sorted(modified_files),
            verification_history=verification_history,
            skills_retrieved=refs(EventType.SKILL_RETRIEVED),
            skills_selected=refs(EventType.SKILL_SELECTED),
            skills_used=refs(EventType.SKILL_USED),
            skill_use_traces=skill_use_traces,
            memories_retrieved=sorted(memory_ids[EventType.MEMORY_RETRIEVED]),
            memories_selected=sorted(memory_ids[EventType.MEMORY_SELECTED]),
            memories_used=sorted(memory_ids[EventType.MEMORY_USED]),
            final_status=final_status,
            failure_reason=failure_reason,
        )
