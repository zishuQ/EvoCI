"""Deterministic, bounded learning input assembled by the harness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evoci.domain.models import (
    FailedCandidateRef,
    SupervisorDecision,
    VerificationResult,
    WorkerExecutionResult,
)
from evoci.runtime.trajectory import ToolTrace, TrajectoryView

_FIELD_LIMIT = 2_000
_LIST_LIMIT = 20
_ITEM_LIMIT = 400
_TOOL_LIMIT = 40


def clip_text(value: str | None, limit: int = _FIELD_LIMIT) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "\n[truncated]"


def clip_list(
    values: Sequence[str], *, limit: int = _LIST_LIMIT, item_limit: int = _ITEM_LIMIT
) -> list[str]:
    clipped: list[str] = []
    for item in values[:limit]:
        text = clip_text(str(item), item_limit)
        if text:
            clipped.append(text)
    return clipped


class LearningInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    repo: str
    task_family: str
    failure_summary: str
    plan_revisions: list[str] = Field(default_factory=list)
    worker_summaries: list[str] = Field(default_factory=list)
    change_summaries: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    key_verification: dict[str, Any] = Field(default_factory=dict)
    knowledge_usage: dict[str, Any] = Field(default_factory=dict)
    outcome: str
    root_cause: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    used_skill_ids: list[str] = Field(default_factory=list)
    used_memory_ids: list[str] = Field(default_factory=list)
    recommended_skill_ids: list[str] = Field(default_factory=list)
    payload_chars: int = 0


def compact_tool_traces(traces: Sequence[ToolTrace]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for trace in traces[:_TOOL_LIMIT]:
        compact.append(
            {
                "tool_name": trace.tool_name,
                "success": trace.success,
                "exit_code": trace.exit_code,
                "error": clip_text(trace.error, 240),
                "agent_id": trace.agent_id,
            }
        )
    return compact


def compact_verification(verification: VerificationResult | None) -> dict[str, Any]:
    if verification is None:
        return {}
    failures = [
        {
            "command": command.command,
            "exit_code": command.exit_code,
            "stderr": clip_text(command.stderr, 400),
            "timed_out": command.timed_out,
            "source": command.source,
        }
        for command in verification.commands
        if command.executed and (command.exit_code != 0 or command.timed_out)
    ]
    return {
        "passed": verification.passed,
        "status": verification.status,
        "failures": failures[:_LIST_LIMIT],
        "expected_count": verification.expected_count,
        "executed_count": verification.executed_count,
    }


def build_learning_input(
    *,
    run_id: str,
    repo: str,
    task_family: str,
    failure_summary: str,
    decisions: Sequence[SupervisorDecision],
    worker_results: Sequence[WorkerExecutionResult],
    verification: VerificationResult | None,
    trajectory: TrajectoryView,
    success: bool,
    failure_reason: str | None,
    failed_candidates: Sequence[FailedCandidateRef] = (),
    recommended_skill_ids: Sequence[str] = (),
) -> LearningInput:
    normalized_decisions = [
        item if hasattr(item, "action") else SupervisorDecision.model_validate(item)
        for item in decisions
    ]
    plan_revisions = clip_list(
        [f"{item.action}: {item.reasoning_summary}" for item in normalized_decisions]
    )
    worker_summaries = clip_list(
        [f"{item.task_id}:{item.status}:{item.summary}" for item in worker_results]
    )
    change_summaries = clip_list(
        [
            f"{item.task_id}: {', '.join(item.changed_files) or 'no files'}"
            for item in worker_results
            if item.changed_files
        ]
    )
    changed_files = clip_list(
        list(dict.fromkeys(path for item in worker_results for path in item.changed_files))
    )
    used_skill_ids = list(dict.fromkeys(ref.skill_id for ref in trajectory.skills_used))
    used_memory_ids = list(trajectory.memories_used)
    payload = LearningInput(
        run_id=run_id,
        repo=repo,
        task_family=task_family,
        failure_summary=clip_text(failure_summary) or "",
        plan_revisions=plan_revisions,
        worker_summaries=worker_summaries,
        change_summaries=change_summaries,
        changed_files=changed_files,
        key_verification=compact_verification(verification),
        knowledge_usage={
            "used_skill_ids": used_skill_ids,
            "used_memory_ids": used_memory_ids,
            "recommended_skill_ids": list(recommended_skill_ids),
            "failed_candidates": [item.candidate_id for item in failed_candidates],
            "tool_summaries": compact_tool_traces(trajectory.tool_calls),
            "outcome": "success" if success else "failed",
            "failure_reason": clip_text(failure_reason),
        },
        outcome="success" if success else "failed",
        root_cause=clip_text(
            normalized_decisions[-1].reasoning_summary if normalized_decisions else None
        ),
        evidence_ids=[item.get("id", "") for item in trajectory.evidence[:10] if item.get("id")],
        used_skill_ids=used_skill_ids,
        used_memory_ids=used_memory_ids,
        recommended_skill_ids=list(recommended_skill_ids),
        payload_chars=0,
    )
    encoded = payload.model_dump_json()
    return payload.model_copy(update={"payload_chars": len(encoded)})
