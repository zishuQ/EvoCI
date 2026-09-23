"""Structured-output Supervisor and Worker implementations."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from evoci.agents.base import SupervisorContext, WorkerContext, WorkerRun
from evoci.agents.tool_loop import BoundedToolAgent
from evoci.capability.registry import CapabilityRegistry
from evoci.domain.models import (
    EvidenceItem,
    FileEdit,
    SkillRef,
    SupervisorDecision,
    VerificationCommandSpec,
    WorkerExecutionResult,
)
from evoci.graph.integration import (
    IntegrationError,
    candidate_prompt_summary,
    edits_from_artifact,
    integrate_edits,
    load_patch_artifact,
)
from evoci.memory.store import MemoryStore
from evoci.model.gateway import ModelGatewayError, ToolLoopGateway
from evoci.runtime.budget import (
    FINALIZATION_MODEL_CALLS,
    RepairBudgetExhausted,
    RunBudgetManager,
    RunRepairBudget,
    action_call_capacity,
)
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.isolation import copy_workspace_with_independent_git, workspace_snapshot_id
from evoci.tools.patch import PatchConflict, PatchError, collect_staged_edits
from evoci.tools.policy import (
    SUPERVISOR_CAPABILITIES,
    WORKER_INVESTIGATE_CAPABILITIES,
    WORKER_REPAIR_CAPABILITIES,
    PolicyViolation,
    WorkerCapabilities,
)
from evoci.tools.registry import create_worker_registry
from evoci.tools.scope import assert_write_path_allowed


class WorkerReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["completed", "blocked", "budget_exhausted"] = "completed"
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    used_memory_ids: list[str] = Field(default_factory=list)
    used_skill_refs: list[SkillRef] = Field(default_factory=list)
    risk: Literal["low", "medium", "high"] = "low"


StagedFixerPlan = WorkerReport


def _validated_usage[OutputT: BaseModel](
    output: OutputT, memory_ids: set[str], skill_ids: set[str]
) -> OutputT:
    updates: dict[str, object] = {}
    if "used_memory_ids" in type(output).model_fields:
        updates["used_memory_ids"] = sorted(
            memory_ids.intersection(getattr(output, "used_memory_ids", []))
        )
    if "used_skill_refs" in type(output).model_fields:
        claimed = {
            ref.skill_id
            for ref in cast(list[SkillRef], getattr(output, "used_skill_refs", []))
        }
        updates["used_skill_refs"] = [
            SkillRef(skill_id=skill_id) for skill_id in sorted(skill_ids.intersection(claimed))
        ]
    return output.model_copy(update=updates) if updates else output


def _tool_context_prompt() -> str:
    return (
        "Memory entries in the task context are brief previews. If a memory "
        "is relevant, call read_memory with its memory_id for full evidence. "
        "Failure Episodes are counterevidence, and unverified hypotheses are not facts. "
        "Reading a memory does not mean it was used in the final decision. "
        "Available skill summaries are listed in the task context. "
        "If a skill directly applies, call load_skill with its skill_id before using "
        "its procedure, resources, or scripts. You may choose to load no skill. "
        "FAILURE usage memory is counterevidence, not a recommended procedure. "
        "Current repository evidence takes priority over skill memory. "
        "After loading, call run_skill_script for bundled scripts and "
        "read_skill_resource for declared references or templates. "
        "Do not use read_file for skill package files. "
        "Report only memory IDs and skill IDs that materially influenced the final answer. "
        "Recommended skills are suggestions, not proof of use."
    )


class ModelSupervisor:
    def __init__(
        self,
        gateway: ToolLoopGateway,
        recorder: TrajectoryRecorder | None = None,
        *,
        capability_registry: CapabilityRegistry | None = None,
        memory_store: MemoryStore | None = None,
        max_iterations: int = 8,
        max_tool_calls: int = 16,
        timeout: float = 120.0,
        max_chars: int = 32_000,
        budget_manager: RunBudgetManager | None = None,
    ) -> None:
        self.gateway = gateway
        self.loop = BoundedToolAgent(
            gateway,
            recorder or TrajectoryRecorder(),
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            budget_manager=budget_manager,
        )
        self.recorder = recorder
        self.capability_registry = capability_registry
        self.memory_store = memory_store
        self.timeout = timeout
        self.max_chars = max_chars
        self.budget_manager = budget_manager

    async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
        payload = {
            "repo": context.repo.model_dump(),
            "failure": context.failure.model_dump(),
            "memories": [memory.model_dump() for memory in context.memories],
            "skill_catalog": [
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "description": skill.description,
                }
                for skill in context.skills
            ],
            "previous_decisions": list(context.previous_decisions),
            "worker_results": list(context.worker_result_summaries),
            "last_verification": (
                context.last_verification.model_dump() if context.last_verification else None
            ),
            "verification_snapshot_id": context.verification_snapshot_id,
            "remaining_batches": context.remaining_batches,
            "remaining_model_calls": context.remaining_model_calls,
            "remaining_tool_calls": context.remaining_tool_calls,
            "failed_candidates": [item.model_dump() for item in context.failed_candidates],
            "usage_complete": context.usage_complete,
        }
        registry = create_worker_registry(
            SUPERVISOR_CAPABILITIES,
            Path(context.workspace_path),
            timeout=self.timeout,
            max_chars=self.max_chars,
            capability_registry=self.capability_registry,
            allowed_skill_refs={skill.skill_id for skill in context.skills},
            memory_store=self.memory_store,
            allowed_memory_refs={memory.memory_id: memory.namespace for memory in context.memories},
            memory_repository=context.repo.full_name,
            tool_allowlist={
                "read_file",
                "read_memory",
                "list_files",
                "search_code",
                "git_status",
                "git_log",
                "git_show",
                "git_diff",
                "load_skill",
                "read_skill_resource",
            },
        )
        try:
            decision = await self.loop.run(
                run_id=context.run_id,
                agent_id="supervisor",
                invocation_id=context.invocation_id,
                system_prompt=(
                    "You are the Supervisor. Plan investigation or repair work, then stop "
                    "when formal verification already passed on the current snapshot. "
                    "You cannot write files or run tests. Dispatch exactly one task. "
                    "Repair tasks must list exact relative write_scope "
                    "paths; unknown scope requires an investigate task first. "
                    "Hypotheses are unverified. You may question earlier assumptions when "
                    "worker evidence contradicts them. Do not claim success unless the "
                    "harness already recorded a passing formal verification for the current "
                    f"code. {_tool_context_prompt()}"
                ),
                task_prompt=json.dumps(payload, default=str),
                tools=registry,
                output_schema=SupervisorDecision,
            )
        finally:
            registry.close()
        return decision


class ModelWorker:
    def __init__(
        self,
        gateway: ToolLoopGateway,
        recorder: TrajectoryRecorder,
        *,
        capability_registry: CapabilityRegistry | None = None,
        memory_store: MemoryStore | None = None,
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
        self.memory_store = memory_store
        self.timeout = timeout
        self.max_chars = max_chars
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.budget_manager = budget_manager

    def _task_budget(self, context: WorkerContext) -> RunRepairBudget:
        remaining_run_model = self.max_iterations + FINALIZATION_MODEL_CALLS
        remaining_tool = self.max_tool_calls
        if self.budget_manager is not None:
            snapshot = self.budget_manager.for_run(context.run_id).snapshot()
            remaining_run_model = snapshot.max_model_calls - snapshot.model_calls
            remaining_tool = min(remaining_tool, snapshot.max_tool_calls - snapshot.tool_calls)
        task_total = None
        if context.task.budget is not None:
            task_total = context.task.budget.max_model_calls
            remaining_tool = min(remaining_tool, context.task.budget.max_tool_calls)
        actions = action_call_capacity(
            max_actions=self.max_iterations,
            remaining_model_calls=remaining_run_model,
            task_model_calls=task_total,
        )
        total_calls = actions + FINALIZATION_MODEL_CALLS if actions else 0
        return RunRepairBudget(
            max_model_calls=total_calls,
            max_tool_calls=max(0, remaining_tool),
            exhausted_message="task-level model-call budget exhausted",
        )

    def _trace_from_events(
        self, run_id: str, agent_id: str, invocation_id: str
    ) -> tuple[list[str], list[EvidenceItem], list[SkillRef], list[VerificationCommandSpec]]:
        recorder = getattr(self.loop, "recorder", None)
        if recorder is None:
            return [], [], [], []
        commands: list[str] = []
        verification_plan: list[VerificationCommandSpec] = []
        evidence: list[EvidenceItem] = []
        skills: list[SkillRef] = []
        seen_skills: set[str] = set()
        seen_plan: set[tuple[tuple[str, ...], str]] = set()
        pending: dict[str, dict[str, object]] = {}
        for event in recorder.events(run_id):
            if event.agent_id != agent_id or event.invocation_id != invocation_id:
                continue
            payload = event.payload or {}
            if event.type == EventType.TOOL_CALL:
                name = str(payload.get("tool_name") or "")
                arguments = payload.get("arguments") or {}
                argv = arguments.get("argv") if isinstance(arguments, dict) else None
                if name in {"run_test", "run_command"} and isinstance(argv, list):
                    cwd = str(arguments.get("cwd") or ".")
                    try:
                        spec = VerificationCommandSpec(
                            argv=[str(part) for part in argv], cwd=cwd
                        )
                    except ValueError:
                        continue
                    rendered = shlex.join(spec.argv)
                    if spec.cwd != ".":
                        rendered = f"{rendered} (cwd={spec.cwd})"
                    commands.append(rendered)
                    pending[str(payload.get("call_id") or "")] = {
                        "spec": spec,
                        "name": name,
                        "network": bool(arguments.get("network")),
                    }
            elif event.type == EventType.TOOL_RESULT:
                name = str(payload.get("tool_name") or "")
                if name in {"run_test", "run_command"}:
                    call_id = str(payload.get("call_id") or "")
                    info = pending.get(call_id) or {}
                    raw_spec = info.get("spec")
                    command_spec = (
                        raw_spec if isinstance(raw_spec, VerificationCommandSpec) else None
                    )
                    success = bool(payload.get("success"))
                    exit_code = payload.get("exit_code")
                    command = shlex.join(command_spec.argv) if command_spec is not None else name
                    evidence.append(
                        EvidenceItem(
                            source_agent=agent_id,
                            kind="test_result",
                            claim=(
                                f"{name} {'passed' if success else 'failed'} "
                                f"with exit code {exit_code}"
                            ),
                            command=command,
                            confidence=1.0 if success else 0.4,
                        )
                    )
                    if (
                        success
                        and name == "run_test"
                        and not bool(info.get("network"))
                        and command_spec is not None
                    ):
                        key = (tuple(command_spec.argv), command_spec.cwd)
                        if key not in seen_plan:
                            seen_plan.add(key)
                            verification_plan.append(command_spec)
            elif event.type == EventType.SKILL_USED:
                skill_id = payload.get("skill_id")
                if not skill_id:
                    for raw in payload.get("skills") or []:
                        if isinstance(raw, dict) and raw.get("skill_id"):
                            skill_id = raw["skill_id"]
                            break
                if isinstance(skill_id, str) and skill_id and skill_id not in seen_skills:
                    seen_skills.add(skill_id)
                    skills.append(SkillRef(skill_id=skill_id))
        return commands, evidence, skills, verification_plan

    def _harness_facts(
        self,
        run_id: str,
        agent_id: str,
        invocation_id: str,
        available_skills: set[str],
    ) -> tuple[list[str], list[EvidenceItem], list[SkillRef], list[VerificationCommandSpec]]:
        commands, evidence, skills, plan = self._trace_from_events(
            run_id, agent_id, invocation_id
        )
        skills = [ref for ref in skills if ref.skill_id in available_skills]
        return commands, evidence, skills, plan

    def _collect_repair_edits(
        self,
        *,
        source: Path,
        staging: Path,
        registry: object,
        restored_paths: list[str],
        write_scope: list[str] | None,
    ) -> list[FileEdit]:
        changed = list(getattr(registry, "changed_paths", lambda: [])())
        collect_paths = sorted(set(changed) | set(restored_paths))
        edits = collect_staged_edits(source, staging, collect_paths)
        if write_scope is not None:
            allowed = tuple(write_scope)
            for edit in edits:
                assert_write_path_allowed(edit.path, allowed)
        return edits

    def _recover_repair(
        self,
        context: WorkerContext,
        *,
        source: Path,
        staging: Path,
        registry: object,
        restored_paths: list[str],
        write_scope: list[str] | None,
        reason: str,
    ) -> WorkerRun | None:
        snapshot = context.baseline_snapshot_id or workspace_snapshot_id(source)
        agent_id = f"worker:{context.task.task_id}"
        available_skills = {skill.skill_id for skill in context.recommended_skills}
        commands, evidence, skills, plan = self._harness_facts(
            context.run_id, agent_id, context.invocation_id, available_skills
        )
        try:
            edits = self._collect_repair_edits(
                source=source,
                staging=staging,
                registry=registry,
                restored_paths=restored_paths,
                write_scope=write_scope,
            )
        except (PolicyViolation, PatchConflict, PatchError, IntegrationError, ValueError) as exc:
            return self._blocked(context, str(exc))
        if not edits:
            return None
        if "exhausted" in reason.lower() or "budget" in reason.lower():
            summary = (
                "Worker modified files but structured finalization exhausted its budget."
            )
        else:
            summary = "Worker modified files but structured finalization failed."
        return WorkerRun(
            result=WorkerExecutionResult(
                task_id=context.task.task_id,
                status="completed",
                summary=summary,
                evidence=evidence,
                unresolved_questions=[reason],
                base_revision=snapshot,
                snapshot_id=snapshot,
                changed_files=[edit.path for edit in edits],
                used_skill_refs=skills,
            ),
            edits=edits,
            commands_run=commands,
            verification_plan=plan,
            risk="low" if len(edits) <= 4 else "medium",
        )

    def _blocked(
        self, context: WorkerContext, summary: str, *, questions: list[str] | None = None
    ) -> WorkerRun:
        snapshot = context.baseline_snapshot_id
        return WorkerRun(
            result=WorkerExecutionResult(
                task_id=context.task.task_id,
                status="blocked",
                summary=summary,
                unresolved_questions=questions or [summary],
                base_revision=snapshot,
                snapshot_id=snapshot,
            )
        )

    def _exhausted(self, context: WorkerContext, summary: str) -> WorkerRun:
        snapshot = context.baseline_snapshot_id
        return WorkerRun(
            result=WorkerExecutionResult(
                task_id=context.task.task_id,
                status="budget_exhausted",
                summary=summary,
                base_revision=snapshot,
                snapshot_id=snapshot,
            )
        )

    async def execute(
        self,
        *,
        context: WorkerContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerRun:
        task = context.task
        payload: dict[str, object] = {
            "repo": context.repo.model_dump(),
            "failure": context.failure.model_dump(),
            "task": task.model_dump(),
            "hypothesis_unverified": True,
            "memories": [memory.model_dump() for memory in context.memories],
            "recommended_skills": [
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "description": skill.description,
                }
                for skill in context.recommended_skills
            ],
            "baseline_snapshot_id": context.baseline_snapshot_id,
            "previous_attempt_summary": context.previous_attempt_summary,
            "capabilities": asdict(capabilities),
        }
        source = Path(context.workspace_path).resolve()
        staging = source
        temporary = None
        registry = None
        restored_paths: list[str] = []
        write_scope = list(task.write_scope) if task.kind == "repair" and task.write_scope else None
        expected_caps = (
            WORKER_REPAIR_CAPABILITIES if task.kind == "repair" else WORKER_INVESTIGATE_CAPABILITIES
        )
        if capabilities.write_files != expected_caps.write_files:
            return self._blocked(context, "worker capabilities do not match task kind")
        extra_budget = self._task_budget(context)
        if extra_budget.max_model_calls < 1 + FINALIZATION_MODEL_CALLS:
            return self._exhausted(context, "run-level model-call budget exhausted")
        try:
            if task.kind == "repair":
                import tempfile

                source.parent.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.TemporaryDirectory(
                    prefix=".evoci-worker-", dir=source.parent
                )
                staging = Path(temporary.name) / "workspace"
                copy_workspace_with_independent_git(source, staging)
                if task.resume_candidate_id:
                    if not context.resume_artifact_ref:
                        return self._blocked(
                            context,
                            f"unknown failed candidate: {task.resume_candidate_id}",
                        )
                    candidate_payload = load_patch_artifact(context.resume_artifact_ref)
                    if (
                        candidate_payload.get("baseline_snapshot_id")
                        != context.baseline_snapshot_id
                    ):
                        return self._blocked(
                            context,
                            "failed candidate baseline does not match current snapshot",
                        )
                    restored = edits_from_artifact(candidate_payload)
                    if write_scope is not None:
                        allowed = tuple(write_scope)
                        for edit in restored:
                            assert_write_path_allowed(edit.path, allowed)
                    integrate_edits(
                        staging,
                        restored,
                        expected_baseline=None,
                        hash_strict=True,
                    )
                    restored_paths = [edit.path for edit in restored]
                    payload["resumed_candidate"] = candidate_prompt_summary(candidate_payload)
            registry = create_worker_registry(
                capabilities,
                staging,
                timeout=self.timeout,
                max_chars=self.max_chars,
                capability_registry=self.capability_registry,
                allowed_skill_refs={skill.skill_id for skill in context.recommended_skills},
                memory_store=self.memory_store,
                allowed_memory_refs={
                    memory.memory_id: memory.namespace for memory in context.memories
                },
                memory_repository=context.repo.full_name,
                write_scope=write_scope,
            )
            system = (
                "Investigate only the assigned objective. Do not write files. Use tools to "
                "inspect real repository state. Return source-linked evidence. Challenge the "
                "supervisor hypothesis when evidence contradicts it and return blocked with "
                f"evidence. {_tool_context_prompt()}"
                if task.kind == "investigate"
                else (
                    "Repair only files in write_scope, inside a private staging copy. Prefer "
                    "replace_text for existing files. Final structured output is a short report "
                    "without file contents; the harness collects staged edits. Never weaken "
                    "tests or CI. Return blocked with evidence when the assigned scope or "
                    f"hypothesis is wrong. {_tool_context_prompt()}"
                )
            )
            report = await self.loop.run(
                run_id=context.run_id,
                agent_id=f"worker:{task.task_id}",
                invocation_id=context.invocation_id,
                system_prompt=system,
                task_prompt=json.dumps(payload, default=str),
                tools=registry,
                output_schema=WorkerReport,
                extra_budget=extra_budget,
            )
            available_memories = {memory.memory_id for memory in context.memories}
            available_skills = {skill.skill_id for skill in context.recommended_skills}
            report = _validated_usage(report, available_memories, available_skills)
            agent_id = f"worker:{task.task_id}"
            commands, tool_evidence, skills, plan = self._harness_facts(
                context.run_id, agent_id, context.invocation_id, available_skills
            )
            semantic_evidence = [
                item for item in report.evidence if item.kind != "test_result"
            ]
            edits: list[FileEdit] = []
            if task.kind == "repair":
                edits = self._collect_repair_edits(
                    source=source,
                    staging=staging,
                    registry=registry,
                    restored_paths=restored_paths,
                    write_scope=write_scope,
                )
            snapshot = context.baseline_snapshot_id or workspace_snapshot_id(source)
            result = WorkerExecutionResult(
                task_id=task.task_id,
                status=report.status,
                summary=report.summary,
                evidence=semantic_evidence + tool_evidence,
                unresolved_questions=report.unresolved_questions,
                base_revision=snapshot,
                snapshot_id=snapshot,
                changed_files=[edit.path for edit in edits],
                used_memory_ids=report.used_memory_ids,
                used_skill_refs=skills,
            )
            return WorkerRun(
                result=result,
                edits=edits,
                commands_run=commands,
                verification_plan=plan,
                risk=report.risk,
            )
        except RepairBudgetExhausted as exc:
            recovered = (
                self._recover_repair(
                    context,
                    source=source,
                    staging=staging,
                    registry=registry,
                    restored_paths=restored_paths,
                    write_scope=write_scope,
                    reason=str(exc),
                )
                if task.kind == "repair" and registry is not None and temporary is not None
                else None
            )
            return recovered if recovered is not None else self._exhausted(context, str(exc))
        except ModelGatewayError as exc:
            recovered = (
                self._recover_repair(
                    context,
                    source=source,
                    staging=staging,
                    registry=registry,
                    restored_paths=restored_paths,
                    write_scope=write_scope,
                    reason=str(exc),
                )
                if task.kind == "repair" and registry is not None and temporary is not None
                else None
            )
            if recovered is not None:
                return recovered
            raise
        except (
            PatchConflict,
            PatchError,
            PolicyViolation,
            FileNotFoundError,
            ValueError,
            IntegrationError,
        ) as exc:
            return self._blocked(context, str(exc))
        finally:
            if registry is not None:
                registry.close()
            if temporary is not None:
                temporary.cleanup()
