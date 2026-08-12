"""Structured-output model implementations for each agent role."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from evoci.agents.base import AgentContext
from evoci.agents.tool_loop import BoundedToolAgent
from evoci.capability.registry import CapabilityRegistry
from evoci.domain.models import (
    Diagnosis,
    EvidenceItem,
    FixerOutput,
    InvestigationPlan,
    InvestigationTask,
    ReviewResult,
    SkillRef,
    VerificationResult,
    WorkerResult,
)
from evoci.model.gateway import ModelGateway, ToolLoopGateway
from evoci.runtime.budget import RunBudgetManager
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.isolation import copy_workspace_with_independent_git
from evoci.tools.policy import (
    FIXER_CAPABILITIES,
    REVIEWER_CAPABILITIES,
    WorkerCapabilities,
)
from evoci.tools.registry import create_worker_registry


def _context_payload(context: AgentContext) -> dict[str, object]:
    return {
        "repo": context.repo.model_dump(),
        "failure": context.failure.model_dump(),
        "memories": [memory.model_dump() for memory in context.memories],
        "skills": [skill.model_dump() for skill in context.skills],
        "previous_review_blockers": list(context.previous_review_blockers),
        "previous_attempt_summary": context.previous_attempt_summary,
    }


def _validated_usage[OutputT: BaseModel](output: OutputT, context: AgentContext) -> OutputT:
    updates: dict[str, object] = {}
    if "used_memory_ids" in type(output).model_fields:
        available_memories = {memory.memory_id for memory in context.memories}
        updates["used_memory_ids"] = sorted(
            available_memories.intersection(getattr(output, "used_memory_ids", []))
        )
    if "used_skill_refs" in type(output).model_fields:
        available_skills = {(skill.skill_id, skill.version) for skill in context.skills}
        claimed = {
            (ref.skill_id, ref.version)
            for ref in cast(list[SkillRef], getattr(output, "used_skill_refs", []))
        }
        updates["used_skill_refs"] = [
            SkillRef(skill_id=skill_id, version=version)
            for skill_id, version in sorted(available_skills.intersection(claimed))
        ]
    return output.model_copy(update=updates) if updates else output


def _tool_context_prompt(context: AgentContext) -> str:
    skills = [
        {
            "skill_id": skill.skill_id,
            "version": skill.version,
            "name": skill.name,
            "description": skill.description,
            "procedure": skill.skill_md,
            "resources": skill.resources,
        }
        for skill in context.skills
    ]
    return (
        "Available reusable capabilities are included in the task JSON under `skills`. "
        "Follow an applicable SKILL.md procedure. If it exposes a bundled script, call "
        "run_skill_script instead of recreating it. Report only memory IDs and skill "
        "versions that materially influenced the final answer.\n\n"
        f"Selected capability details:\n{json.dumps(skills, default=str)}"
    )


class ModelCoordinator:
    def __init__(
        self,
        gateway: ModelGateway,
        recorder: TrajectoryRecorder | None = None,
        budget_manager: RunBudgetManager | None = None,
    ) -> None:
        self.gateway = gateway
        self.recorder = recorder
        self.budget_manager = budget_manager

    async def plan(
        self,
        *,
        context: AgentContext,
        evidence: list[EvidenceItem],
        round_number: int,
        remaining_task_budget: int,
    ) -> InvestigationPlan:
        payload = _context_payload(context) | {
            "round": round_number,
            "remaining_task_budget": remaining_task_budget,
            "existing_evidence": [item.model_dump() for item in evidence],
        }
        if self.budget_manager is not None:
            self.budget_manager.for_run(context.run_id).consume_model_call()
        if self.recorder is not None:
            self.recorder.emit(
                run_id=context.run_id,
                event_type=EventType.MODEL_CALL,
                agent_id="coordinator",
                invocation_id=context.invocation_id,
                event_key=f"round:{round_number}",
                payload={"phase": "structured"},
            )
        return await self.gateway.complete(
            system_prompt=(
                "Plan bounded, independent CI investigation tasks. Do not edit files. "
                "Prefer tasks that can run in parallel and request structured evidence."
            ),
            user_prompt=json.dumps(payload, default=str),
            response_model=InvestigationPlan,
            agent_id="coordinator",
        )


class ModelInvestigator:
    def __init__(
        self,
        gateway: ToolLoopGateway,
        recorder: TrajectoryRecorder,
        *,
        capability_registry: CapabilityRegistry | None = None,
        max_iterations: int = 8,
        max_tool_calls: int = 16,
        timeout: float = 120.0,
        max_chars: int = 32_000,
        budget_manager: RunBudgetManager | None = None,
    ) -> None:
        self.loop = BoundedToolAgent(
            gateway,
            recorder,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            budget_manager=budget_manager,
        )
        self.capability_registry = capability_registry
        self.timeout = timeout
        self.max_chars = max_chars

    async def run(
        self,
        *,
        task: InvestigationTask,
        context: AgentContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerResult:
        payload = _context_payload(context) | {
            "task": task.model_dump(),
            "capabilities": asdict(capabilities),
        }
        registry = create_worker_registry(
            capabilities,
            Path(context.workspace_path),
            timeout=self.timeout,
            max_chars=self.max_chars,
            capability_registry=self.capability_registry,
            allowed_skill_refs={(skill.skill_id, skill.version) for skill in context.skills},
        )
        try:
            result = await self.loop.run(
                run_id=context.run_id,
                agent_id=f"investigator:{task.task_id}",
                invocation_id=context.invocation_id,
                system_prompt=(
                    "Investigate only the assigned objective. Do not propose unsupported "
                    "claims and "
                    "do not write files. Use tools to inspect real repository state and return "
                    f"source-linked evidence. {_tool_context_prompt(context)}"
                ),
                task_prompt=json.dumps(payload, default=str),
                tools=registry,
                output_schema=WorkerResult,
            )
        finally:
            registry.close()
        return _validated_usage(result, context)


class ModelDiagnoser:
    def __init__(
        self,
        gateway: ModelGateway,
        recorder: TrajectoryRecorder | None = None,
        budget_manager: RunBudgetManager | None = None,
    ) -> None:
        self.gateway = gateway
        self.recorder = recorder
        self.budget_manager = budget_manager

    async def diagnose(self, *, context: AgentContext, evidence: list[EvidenceItem]) -> Diagnosis:
        payload = _context_payload(context) | {"evidence": [item.model_dump() for item in evidence]}
        if self.budget_manager is not None:
            self.budget_manager.for_run(context.run_id).consume_model_call()
        if self.recorder is not None:
            self.recorder.emit(
                run_id=context.run_id,
                event_type=EventType.MODEL_CALL,
                agent_id="diagnoser",
                invocation_id=context.invocation_id,
                event_key=f"evidence:{len(evidence)}",
                payload={"phase": "structured"},
            )
        return await self.gateway.complete(
            system_prompt=(
                "Diagnose the CI failure using evidence IDs. Set needs_more_evidence when the "
                "primary hypothesis is not adequately supported."
            ),
            user_prompt=json.dumps(payload, default=str),
            response_model=Diagnosis,
            agent_id="diagnoser",
        )


class ModelFixer:
    def __init__(
        self,
        gateway: ToolLoopGateway,
        recorder: TrajectoryRecorder,
        *,
        capability_registry: CapabilityRegistry | None = None,
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        timeout: float = 120.0,
        max_chars: int = 32_000,
        budget_manager: RunBudgetManager | None = None,
    ) -> None:
        self.loop = BoundedToolAgent(
            gateway,
            recorder,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            budget_manager=budget_manager,
        )
        self.capability_registry = capability_registry
        self.timeout = timeout
        self.max_chars = max_chars

    async def propose(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        evidence: list[EvidenceItem],
        previous_verification: VerificationResult | None,
    ) -> FixerOutput:
        payload = _context_payload(context) | {
            "diagnosis": diagnosis.model_dump(),
            "evidence": [item.model_dump() for item in evidence],
            "previous_verification": (
                previous_verification.model_dump() if previous_verification else None
            ),
        }
        source = Path(context.workspace_path).resolve()
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".evoci-fixer-", dir=source.parent) as temporary:
            staging = Path(temporary) / "workspace"
            copy_workspace_with_independent_git(source, staging)
            registry = create_worker_registry(
                FIXER_CAPABILITIES,
                staging,
                timeout=self.timeout,
                max_chars=self.max_chars,
                capability_registry=self.capability_registry,
                allowed_skill_refs={(skill.skill_id, skill.version) for skill in context.skills},
            )
            try:
                result = await self.loop.run(
                    run_id=context.run_id,
                    agent_id="fixer",
                    invocation_id=context.invocation_id,
                    system_prompt=(
                        "Propose the smallest evidence-backed repair. Tool writes occur only in a "
                        "private staging copy; LangGraph remains the authority that applies "
                        "the final "
                        "structured edits after its risk gate. Return complete replacement content "
                        "for every changed file and deterministic verification argv. Never weaken "
                        f"tests or CI. {_tool_context_prompt(context)}"
                    ),
                    task_prompt=json.dumps(payload, default=str),
                    tools=registry,
                    output_schema=FixerOutput,
                )
            finally:
                registry.close()
        return _validated_usage(result, context)


class ModelReviewer:
    def __init__(
        self,
        gateway: ToolLoopGateway,
        recorder: TrajectoryRecorder,
        *,
        capability_registry: CapabilityRegistry | None = None,
        max_iterations: int = 6,
        max_tool_calls: int = 12,
        timeout: float = 120.0,
        max_chars: int = 32_000,
        budget_manager: RunBudgetManager | None = None,
    ) -> None:
        self.loop = BoundedToolAgent(
            gateway,
            recorder,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            budget_manager=budget_manager,
        )
        self.capability_registry = capability_registry
        self.timeout = timeout
        self.max_chars = max_chars

    async def review(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        patch: FixerOutput,
        verification: VerificationResult,
    ) -> ReviewResult:
        payload = _context_payload(context) | {
            "diagnosis": diagnosis.model_dump(),
            "patch": patch.model_dump(),
            "verification": verification.model_dump(),
        }
        registry = create_worker_registry(
            REVIEWER_CAPABILITIES,
            Path(context.workspace_path),
            timeout=self.timeout,
            max_chars=self.max_chars,
            capability_registry=self.capability_registry,
            allowed_skill_refs={(skill.skill_id, skill.version) for skill in context.skills},
            tool_allowlist={
                "read_file",
                "search_code",
                "git_diff",
                "run_test",
                "run_skill_script",
            },
        )
        try:
            result = await self.loop.run(
                run_id=context.run_id,
                agent_id="reviewer",
                invocation_id=context.invocation_id,
                system_prompt=(
                    "Independently review scope, safety, test integrity, hard-coded workarounds, "
                    "and CI bypasses. You cannot write files. Inspect the real diff and rerun a "
                    f"targeted check when useful. {_tool_context_prompt(context)}"
                ),
                task_prompt=json.dumps(payload, default=str),
                tools=registry,
                output_schema=ReviewResult,
            )
        finally:
            registry.close()
        return _validated_usage(result, context)
