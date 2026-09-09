"""Bounded, durable LangGraph orchestration for CI recovery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any, Literal, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, interrupt

from evoci.agents.base import AgentContext, AgentSuite
from evoci.capability.curator import CuratorPipeline
from evoci.capability.materializer import CapabilityMaterializer
from evoci.capability.miner import ExperienceMining, LearningDecision
from evoci.capability.models import SkillVersionRef
from evoci.capability.promotion import TrialPromotionPolicy
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.capability.utility import SkillUtilityPolicy, WeightedUtilityPolicy
from evoci.capability.validator import CandidateValidator
from evoci.config import EvoCIConfig
from evoci.domain.models import (
    ReviewResult,
    SkillRef,
)
from evoci.graph.routing import (
    contains_review_bypass,
    contains_workspace_review_bypass,
    diagnosis_route,
    requires_approval,
)
from evoci.graph.state import EvoCIState
from evoci.memory.consolidation import MemoryConsolidator, commit_candidate
from evoci.memory.models import Episode, MemoryCandidate
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import MemoryStore
from evoci.model.gateway import ModelGatewayError
from evoci.runtime.budget import (
    RepairBudgetExhausted,
    RunBudgetManager,
)
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.events import EventType, RunEvent
from evoci.runtime.run_store import SQLiteRunStore
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.filesystem import FileTools
from evoci.tools.patch import (
    PatchConflict,
    PatchError,
    apply_edit,
    attempt_targets,
    precheck_edits,
    recover_attempt_writes,
    restore_attempt_writes,
    restore_edit_baseline,
    snapshot_edit_baseline,
)
from evoci.tools.policy import INVESTIGATOR_CAPABILITIES
from evoci.verification.service import VerificationService, build_verification_plan


@dataclass(frozen=True, slots=True)
class GraphRuntime:
    config: EvoCIConfig
    agents: AgentSuite
    event_store: SQLiteEventStore | None = None
    run_store: SQLiteRunStore | None = None
    memory_store: MemoryStore | None = None
    memory_retriever: MemoryRetriever | None = None
    memory_consolidator: MemoryConsolidator | None = None
    capability_registry: CapabilityRegistry | None = None
    capability_retriever: CapabilityRetriever | None = None
    capability_materializer: CapabilityMaterializer | None = None
    experience_miner: ExperienceMining | None = None
    candidate_validator: CandidateValidator | None = None
    recorder: TrajectoryRecorder | None = None
    promotion_policy: TrialPromotionPolicy | None = None
    utility_policy: SkillUtilityPolicy | None = None
    curator_pipeline: CuratorPipeline | None = None
    budget_manager: RunBudgetManager | None = None
    defer_success_learning: bool = False


def _context(state: EvoCIState, *, invocation_id: str = "graph:0") -> AgentContext:
    return AgentContext(
        run_id=state["run_id"],
        repo=state["repo"],
        failure=state["ci_failure"],
        workspace_path=state["workspace_path"],
        invocation_id=invocation_id,
        memories=tuple(state.get("retrieved_memories", [])),
        skills=tuple(state.get("retrieved_skills", [])),
        previous_review_blockers=tuple(state.get("previous_review_blockers", [])),
        previous_attempt_summary=state.get("previous_attempt_summary"),
    )


def _event(
    runtime: GraphRuntime,
    state: EvoCIState,
    event_type: EventType,
    *,
    agent_id: str | None = None,
    invocation_id: str | None = None,
    discriminator: str = "",
    payload: dict[str, Any] | None = None,
) -> RunEvent:
    if runtime.recorder is not None:
        return runtime.recorder.emit(
            run_id=state["run_id"],
            event_type=event_type,
            agent_id=agent_id,
            invocation_id=invocation_id,
            event_key=discriminator or "0",
            payload=payload,
        )
    event = RunEvent(
        event_id=(
            f"{state['run_id']}:{agent_id or 'harness'}:{invocation_id or 'default'}:"
            f"{event_type.value}:{discriminator or '0'}"
        ),
        run_id=state["run_id"],
        type=event_type,
        agent_id=agent_id,
        invocation_id=invocation_id,
        payload=payload or {},
    )
    if runtime.event_store is not None:
        runtime.event_store.append(event)
    return event


def _attribution(
    runtime: GraphRuntime,
    state: EvoCIState,
    *,
    agent_id: str,
    invocation_id: str,
    memory_ids: list[str],
    skill_refs: list[SkillRef],
    discriminator: str,
) -> tuple[list[RunEvent], list[str], list[SkillRef]]:
    available_memories = {memory.memory_id for memory in state.get("retrieved_memories", [])}
    available_skills = {
        (skill.skill_id, skill.version) for skill in state.get("retrieved_skills", [])
    }
    used_memories = sorted(available_memories.intersection(memory_ids))
    used_skills = [ref for ref in skill_refs if (ref.skill_id, ref.version) in available_skills]
    events: list[RunEvent] = []
    if used_memories:
        events.append(
            _event(
                runtime,
                state,
                EventType.MEMORY_USED,
                agent_id=agent_id,
                invocation_id=invocation_id,
                discriminator=discriminator,
                payload={"memory_ids": used_memories},
            )
        )
    if used_skills:
        events.append(
            _event(
                runtime,
                state,
                EventType.SKILL_USED,
                agent_id=agent_id,
                invocation_id=invocation_id,
                discriminator=discriminator,
                payload={"skills": [ref.model_dump() for ref in used_skills]},
            )
        )
    return events, used_memories, used_skills


_apply_edit = apply_edit
_snapshot_edit_baseline = snapshot_edit_baseline
_restore_edit_baseline = restore_edit_baseline


async def persist_run_outcome(
    runtime: GraphRuntime,
    state: EvoCIState,
    *,
    success: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    config = runtime.config
    status: Literal["success", "failed"] = "success" if success else "failed"
    terminal = _event(
        runtime,
        state,
        EventType.RUN_COMPLETED if success else EventType.RUN_FAILED,
        discriminator="terminal",
        payload={"status": status, "reason": failure_reason},
    )
    if runtime.run_store is not None:
        runtime.run_store.update_status(state["run_id"], status)
    assert runtime.recorder is not None
    trajectory = runtime.recorder.build_view(
        run_id=state["run_id"],
        verification_history=state.get("verification_history", []),
        final_status=status,
        failure_reason=failure_reason,
    )
    events: list[RunEvent] = [terminal]
    episode_id: str | None = None
    candidate_skill_id: str | None = None
    learning_decision: dict[str, object] | None = None
    learning_errors: list[dict[str, str]] = []
    diagnosis = state.get("diagnosis")
    output = state.get("fixer_output")
    verification = state.get("verification")

    def learning_error(stage: str, exc: Exception) -> None:
        payload = {
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc) or type(exc).__name__,
        }
        learning_errors.append(payload)
        events.append(
            _event(
                runtime,
                state,
                EventType.LEARNING_ERROR,
                discriminator=f"{stage}:{len(learning_errors)}",
                payload=payload,
            )
        )

    if runtime.memory_store is not None:
        try:
            verification_failures = [
                command.stderr or f"exit code {command.exit_code}"
                for history in state.get("verification_history", [])
                for command in history.commands
                if command.exit_code != 0 or command.timed_out
            ]
            hypotheses = []
            if diagnosis is not None:
                hypotheses = [
                    diagnosis.primary.root_cause,
                    *[alternative.root_cause for alternative in diagnosis.alternatives],
                ]
            episode = Episode(
                id=f"episode:{state['run_id']}",
                run_id=state["run_id"],
                repo=state["repo"].full_name,
                task_family=state["ci_failure"].task_family,
                failure_summary=state["ci_failure"].summary,
                root_cause=diagnosis.primary.root_cause if diagnosis else None,
                important_evidence=[item.id for item in state.get("evidence", [])[:10]],
                attempts=state.get("repair_attempt", 0),
                successful_fix_summary=(
                    output.proposal.summary if success and output is not None else None
                ),
                tools_used=[tool.tool_name for tool in trajectory.tool_calls],
                hypotheses_attempted=hypotheses,
                verification_failures=verification_failures,
                failure_reason=failure_reason,
                success=success,
            )
            runtime.memory_store.add_episode(episode)
            episode_id = episode.id
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.MEMORY_CREATED,
                    discriminator="episode",
                    payload={"episode_id": episode.id, "success": success},
                )
            )
        except Exception as exc:
            learning_error("episode", exc)

    if runtime.capability_registry is not None:
        try:
            promotion_policy = runtime.promotion_policy or TrialPromotionPolicy(
                min_uses=config.trial_min_uses,
                min_successes=config.trial_min_successes,
                min_success_rate=config.trial_min_success_rate,
                max_failures=config.trial_max_failures,
                max_exposures_without_use=config.trial_max_exposures_without_use,
            )
            utility_policy = runtime.utility_policy or WeightedUtilityPolicy()
            for used in trajectory.skills_used:
                ref = SkillVersionRef(skill_id=used.skill_id, version=used.version)
                record = runtime.capability_registry.get(ref.skill_id, ref.version)
                if record is None:
                    continue
                traces = [
                    trace
                    for trace in trajectory.skill_use_traces
                    if (trace.skill_id, trace.version) == (ref.skill_id, ref.version)
                ]
                explicit_failure = any(trace.execution_success is False for trace in traces)
                attributed_success = success and not explicit_failure
                runtime.capability_registry.record_use(
                    ref,
                    success=attributed_success,
                    tool_calls=trajectory.tool_call_count,
                    attempts=state.get("repair_attempt", 0),
                    patched=bool(trajectory.created_files or trajectory.modified_files),
                    operation_key=(
                        f"skill-use:{state['run_id']}:{ref.skill_id}:v{ref.version}"
                    ),
                )
                stats = runtime.capability_registry.stats(ref.skill_id, ref.version)
                runtime.capability_registry.set_utility(ref, utility_policy.score(stats))
            selected = [
                SkillVersionRef(skill_id=ref.skill_id, version=ref.version)
                for ref in trajectory.skills_selected
            ]
            for ref in selected:
                record = runtime.capability_registry.get(ref.skill_id, ref.version)
                if record is None:
                    continue
                action = promotion_policy.apply(
                    runtime.capability_registry,
                    record,
                    operation_key=(
                        f"skill-lifecycle:{state['run_id']}:{ref.skill_id}:v{ref.version}"
                    ),
                )
                if action not in {"promote", "reject"}:
                    continue
                events.append(
                    _event(
                        runtime,
                        state,
                        (
                            EventType.SKILL_PROMOTED
                            if action == "promote"
                            else EventType.SKILL_REJECTED
                        ),
                        discriminator=f"{ref.skill_id}:v{ref.version}",
                        payload=ref.model_dump(),
                    )
                )
        except Exception as exc:
            learning_error("skill_outcomes", exc)

    if (
        success
        and runtime.memory_store is not None
        and runtime.memory_consolidator is not None
        and diagnosis is not None
        and output is not None
        and verification is not None
    ):
        try:
            decision_key = f"memory-decision:{state['run_id']}"
            previous = runtime.memory_store.operation_result(decision_key)
            if previous is None:
                _event(
                    runtime,
                    state,
                    EventType.MODEL_CALL,
                    agent_id="memory-consolidator",
                    discriminator="final",
                    payload={"phase": "structured", "budget_scope": "post_run"},
                )
                candidate = await runtime.memory_consolidator.propose(
                    run_id=state["run_id"],
                    repo=state["repo"].full_name,
                    task_family=state["ci_failure"].task_family,
                    diagnosis=diagnosis,
                    patch=output,
                    verification=verification,
                )
                runtime.memory_store.record_operation(
                    decision_key, candidate.model_dump(mode="json")
                )
            else:
                candidate = MemoryCandidate.model_validate(previous)
            memory_id = commit_candidate(
                runtime.memory_store,
                candidate,
                run_id=state["run_id"],
                operation_key=f"memory-commit:{state['run_id']}",
            )
            if memory_id:
                events.append(
                    _event(
                        runtime,
                        state,
                        EventType.MEMORY_CREATED,
                        discriminator="semantic",
                        payload={"memory_id": memory_id},
                    )
                )
        except Exception as exc:
            learning_error("memory_consolidation", exc)

    should_mine = False
    if (
        success
        and runtime.experience_miner is not None
        and runtime.capability_registry is not None
    ):
        try:
            should_mine = runtime.experience_miner.should_mine(
                success=True,
                tool_calls=trajectory.tool_call_count,
                failed_attempts=len(trajectory.failed_attempts),
                reusable_script_created=trajectory.reusable_script_created,
                repeated_pattern_detected=bool(trajectory.memories_selected),
            )
        except Exception as exc:
            learning_error("experience_mining", exc)
    if should_mine:
        try:
            experience_miner = runtime.experience_miner
            capability_registry = runtime.capability_registry
            assert experience_miner is not None
            assert capability_registry is not None
            decision_key = f"learning-decision:{state['run_id']}"
            previous = capability_registry.operation_result(decision_key)
            if previous is None:
                _event(
                    runtime,
                    state,
                    EventType.MODEL_CALL,
                    agent_id="experience-miner",
                    discriminator="final",
                    payload={"phase": "structured", "budget_scope": "post_run"},
                )
                decision = await experience_miner.decide(trajectory)
                capability_registry.record_operation(
                    decision_key, decision.model_dump(mode="json")
                )
            else:
                decision = LearningDecision.model_validate(previous)
            learning_decision = decision.model_dump()
            if decision.action == "memory" and decision.candidate_memory:
                if runtime.memory_store is not None:
                    memory_id = commit_candidate(
                        runtime.memory_store,
                        decision.candidate_memory,
                        run_id=state["run_id"],
                        operation_key=f"learning-memory:{state['run_id']}",
                    )
                    if memory_id:
                        events.append(
                            _event(
                                runtime,
                                state,
                                EventType.MEMORY_CREATED,
                                discriminator="mined",
                                payload={"memory_id": memory_id},
                            )
                        )
            elif decision.action in {"new_skill", "update_skill"}:
                assert decision.candidate_skill is not None
                create_kwargs: dict[str, Any] = {}
                if decision.action == "update_skill":
                    assert decision.target_skill_id is not None
                    assert decision.target_version is not None
                    create_kwargs = {
                        "skill_id": decision.target_skill_id,
                        "parent_version": decision.target_version,
                    }
                created = capability_registry.create_candidate(
                    decision.candidate_skill,
                    operation_key=f"learning-skill:{state['run_id']}",
                    **create_kwargs,
                )
                candidate_skill_id = created.manifest.skill_id
                validation_passed: bool | None = None
                if runtime.candidate_validator is not None:
                    current = capability_registry.get(
                        created.manifest.skill_id, created.manifest.version
                    )
                    assert current is not None
                    if current.manifest.status == "candidate":
                        validation = await runtime.candidate_validator.avalidate_to_trial(
                            created.manifest.skill_id, created.manifest.version
                        )
                        validation_passed = validation.passed
                    else:
                        validation_passed = current.manifest.status == "trial"
                events.append(
                    _event(
                        runtime,
                        state,
                        EventType.SKILL_CANDIDATE_CREATED,
                        discriminator=(
                            f"{created.manifest.skill_id}:v{created.manifest.version}"
                        ),
                        payload={
                            "skill_id": created.manifest.skill_id,
                            "version": created.manifest.version,
                            "parent_version": created.manifest.parent_version,
                            "validation_passed": validation_passed,
                        },
                    )
                )
        except Exception as exc:
            learning_error("experience_mining", exc)

    if success and candidate_skill_id and runtime.curator_pipeline is not None:
        try:
            curation = await runtime.curator_pipeline.run(
                run_id=state["run_id"], recorder=runtime.recorder
            )
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.CURATOR_DECISION,
                    discriminator="pipeline",
                    payload=curation.model_dump(mode="json"),
                )
            )
        except Exception as exc:
            learning_error("curator", exc)
    return {
        "phase": "finalize" if success else "failed",
        "status": status,
        "failure_reason": failure_reason,
        "episode_id": episode_id,
        "candidate_skill_id": candidate_skill_id,
        "learning_decision": learning_decision,
        "learning_errors": learning_errors,
        "learning_deferred": False,
        "events": events,
    }


def build_graph(
    runtime: GraphRuntime,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    if runtime.recorder is None:
        runtime = replace(runtime, recorder=TrajectoryRecorder(runtime.event_store))
    if runtime.budget_manager is None:
        runtime = replace(
            runtime,
            budget_manager=RunBudgetManager(
                max_model_calls=runtime.config.max_run_model_calls,
                max_tool_calls=runtime.config.max_run_tool_calls,
                recorder=runtime.recorder,
            ),
        )
    assert runtime.budget_manager is not None
    for agent in (
        runtime.agents.coordinator,
        runtime.agents.investigator,
        runtime.agents.diagnoser,
        runtime.agents.fixer,
        runtime.agents.reviewer,
    ):
        if hasattr(agent, "budget_manager"):
            cast(Any, agent).budget_manager = runtime.budget_manager
        loop = getattr(agent, "loop", None)
        if loop is not None and hasattr(loop, "budget_manager"):
            loop.budget_manager = runtime.budget_manager
    config = runtime.config

    def consume_tool(state: EvoCIState) -> None:
        assert runtime.budget_manager is not None
        runtime.budget_manager.for_run(state["run_id"]).consume_tool_call()

    async def bootstrap(state: EvoCIState) -> dict[str, Any]:
        if runtime.run_store is not None:
            runtime.run_store.create(
                state["run_id"],
                state["task_id"],
                {
                    "repo": state["repo"].model_dump(),
                    "ci_failure": state["ci_failure"].model_dump(),
                    "workspace_path": state["workspace_path"],
                },
            )
        return {
            "phase": "bootstrap",
            "status": "running",
            "investigation_round": state.get("investigation_round", 0),
            "repair_attempt": state.get("repair_attempt", 0),
            "investigation_task_count": state.get("investigation_task_count", 0),
            "evidence": [],
            "budget_failures": [],
            "completed_task_ids": [],
            "retrieved_memories": state.get("retrieved_memories", []),
            "retrieved_skills": state.get("retrieved_skills", []),
            "selected_memory_ids": [],
            "used_memory_ids": [],
            "selected_skill_refs": [],
            "used_skill_refs": [],
            "used_skills": state.get("used_skills", []),
            "verification_history": [],
            "attempt_baseline": state.get("attempt_baseline", {}),
            "previous_review_blockers": state.get("previous_review_blockers", []),
            "previous_attempt_summary": state.get("previous_attempt_summary"),
            "learning_errors": state.get("learning_errors", []),
            "events": [
                _event(runtime, state, EventType.RUN_STARTED, payload={"task": state["task_id"]})
            ],
        }

    async def retrieve_context(state: EvoCIState) -> dict[str, Any]:
        memories = state.get("retrieved_memories", [])
        skills = state.get("retrieved_skills", [])
        selected_memory_ids = [memory.memory_id for memory in memories]
        selected_skill_refs = [
            SkillRef(skill_id=skill.skill_id, version=skill.version) for skill in skills
        ]
        retrieval_events: list[RunEvent] = []
        if runtime.memory_retriever is not None:
            retrieval = runtime.memory_retriever.retrieve(state["repo"], state["ci_failure"])
            memories = retrieval.hits
            selected_memory_ids = retrieval.telemetry.selected_ids
            retrieval_events.append(
                _event(
                    runtime,
                    state,
                    EventType.MEMORY_RETRIEVED,
                    payload={
                        **retrieval.telemetry.model_dump(),
                        "memory_ids": retrieval.telemetry.retrieved_ids,
                    },
                )
            )
            retrieval_events.append(
                _event(
                    runtime,
                    state,
                    EventType.MEMORY_SELECTED,
                    payload={"memory_ids": selected_memory_ids},
                )
            )
        if runtime.capability_retriever is not None:
            skills = runtime.capability_retriever.retrieve(
                state["repo"],
                state["ci_failure"],
                operation_key=f"skill-retrieval:{state['run_id']}",
            )
            selected_skill_refs = [
                SkillRef(skill_id=skill.skill_id, version=skill.version) for skill in skills
            ]
            if skills and runtime.capability_materializer is not None:
                runtime.capability_materializer.materialize(
                    skills,
                    run_id=state["run_id"],
                    workspace=Path(state["workspace_path"]),
                )
            if runtime.capability_registry is not None:
                runtime.capability_registry.record_retrieval(
                    [
                        SkillVersionRef(skill_id=ref.skill_id, version=ref.version)
                        for ref in selected_skill_refs
                    ],
                    selected=True,
                    operation_key=f"skill-selection:{state['run_id']}",
                )
            retrieval_events.append(
                _event(
                    runtime,
                    state,
                    EventType.SKILL_RETRIEVED,
                    payload={"skills": [ref.model_dump() for ref in selected_skill_refs]},
                )
            )
            retrieval_events.append(
                _event(
                    runtime,
                    state,
                    EventType.SKILL_SELECTED,
                    payload={"skills": [ref.model_dump() for ref in selected_skill_refs]},
                )
            )
        return {
            "phase": "retrieve_context",
            "retrieved_memories": memories,
            "retrieved_skills": skills,
            "selected_memory_ids": selected_memory_ids,
            "selected_skill_refs": selected_skill_refs,
            "events": retrieval_events,
        }

    async def coordinate(state: EvoCIState) -> dict[str, Any]:
        current_count = state.get("investigation_task_count", 0)
        remaining = config.max_investigation_tasks - current_count
        round_number = state.get("investigation_round", 0) + 1
        invocation_id = f"coordinate:{round_number}"
        if remaining <= 0 or round_number > config.max_investigation_rounds:
            return {
                "phase": "coordinate",
                "status": "failed",
                "failure_reason": "investigation budget exhausted",
                "investigation_plan": None,
            }
        try:
            plan = await runtime.agents.coordinator.plan(
                context=_context(state, invocation_id=invocation_id),
                evidence=state.get("evidence", []),
                round_number=round_number,
                remaining_task_budget=remaining,
            )
        except (RepairBudgetExhausted, ModelGatewayError) as exc:
            return {
                "phase": "coordinate",
                "status": "failed",
                "failure_reason": str(exc),
                "investigation_plan": None,
            }
        per_round_limit = config.max_initial_workers if round_number == 1 else remaining
        allowed = min(remaining, per_round_limit)
        completed = set(state.get("completed_task_ids", []))
        tasks = [task for task in plan.tasks if task.task_id not in completed][:allowed]
        bounded_plan = plan.model_copy(update={"tasks": tasks})
        if not tasks:
            return {
                "phase": "coordinate",
                "status": "failed",
                "failure_reason": "coordinator produced no new investigation tasks",
                "investigation_plan": bounded_plan,
                "investigation_round": round_number,
            }
        return {
            "phase": "coordinate",
            "investigation_plan": bounded_plan,
            "investigation_round": round_number,
            "investigation_task_count": current_count + len(tasks),
            "events": [
                _event(
                    runtime,
                    state,
                    EventType.AGENT_COMPLETED,
                    agent_id="coordinator",
                    invocation_id=invocation_id,
                    discriminator=str(round_number),
                    payload={"tasks": [task.model_dump() for task in tasks]},
                )
            ],
        }

    def dispatch(state: EvoCIState) -> Sequence[Send] | str:
        if state.get("status") == "failed":
            return "failed"
        plan = state.get("investigation_plan")
        if plan is None:
            return "failed"
        shared = {
            "run_id": state["run_id"],
            "task_id": state["task_id"],
            "repo": state["repo"],
            "ci_failure": state["ci_failure"],
            "workspace_path": state["workspace_path"],
            "retrieved_memories": state.get("retrieved_memories", []),
            "retrieved_skills": state.get("retrieved_skills", []),
        }
        return [Send("investigate", shared | {"worker_task": task}) for task in plan.tasks]

    async def investigate(state: EvoCIState) -> dict[str, Any]:
        task = state["worker_task"]
        agent_id = f"investigator:{task.task_id}"
        invocation_id = f"investigation:{task.task_id}"
        try:
            result = await runtime.agents.investigator.run(
                task=task,
                context=_context(state, invocation_id=invocation_id),
                capabilities=INVESTIGATOR_CAPABILITIES,
            )
        except (RepairBudgetExhausted, ModelGatewayError) as exc:
            return {
                "completed_task_ids": [task.task_id],
                "evidence": [],
                "budget_failures": [str(exc)],
            }
        usage_events, used_memories, used_skills = _attribution(
            runtime,
            state,
            agent_id=agent_id,
            invocation_id=invocation_id,
            memory_ids=result.used_memory_ids,
            skill_refs=result.used_skill_refs,
            discriminator=task.task_id,
        )
        return {
            "evidence": result.evidence,
            "completed_task_ids": [task.task_id],
            "used_memory_ids": used_memories,
            "used_skill_refs": used_skills,
            "events": [
                *[
                    _event(
                        runtime,
                        state,
                        EventType.EVIDENCE_CREATED,
                        agent_id=agent_id,
                        invocation_id=invocation_id,
                        discriminator=item.id,
                        payload={"evidence": item.model_dump()},
                    )
                    for item in result.evidence
                ],
                *usage_events,
                _event(
                    runtime,
                    state,
                    EventType.AGENT_COMPLETED,
                    agent_id=agent_id,
                    invocation_id=invocation_id,
                    discriminator=task.task_id,
                    payload={"evidence_count": len(result.evidence)},
                ),
            ],
        }

    async def diagnose(state: EvoCIState) -> dict[str, Any]:
        if state.get("status") == "failed" or state.get("budget_failures"):
            return {
                "phase": "diagnose",
                "diagnosis": None,
                "status": "failed",
                "failure_reason": (state.get("budget_failures", ["repair budget exhausted"])[0]),
            }
        invocation_id = f"diagnosis:{state.get('investigation_round', 0)}"
        try:
            diagnosis = await runtime.agents.diagnoser.diagnose(
                context=_context(state, invocation_id=invocation_id),
                evidence=state.get("evidence", []),
            )
        except (RepairBudgetExhausted, ModelGatewayError) as exc:
            return {
                "phase": "diagnose",
                "status": "failed",
                "failure_reason": str(exc),
                "diagnosis": None,
            }
        events = [
            _event(
                runtime,
                state,
                EventType.DIAGNOSIS_CREATED,
                agent_id="diagnoser",
                invocation_id=invocation_id,
                discriminator=str(state.get("investigation_round", 0)),
                payload={"diagnosis": diagnosis.model_dump()},
            )
        ]
        if diagnosis.needs_more_evidence:
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.ATTEMPT_FAILED,
                    agent_id="diagnoser",
                    invocation_id=invocation_id,
                    discriminator=f"round:{state.get('investigation_round', 0)}",
                    payload={
                        "kind": "diagnosis_rejected",
                        "agent_id": "diagnoser",
                        "reason": "; ".join(diagnosis.missing_evidence)
                        or "diagnosis requested more evidence",
                        "iteration": state.get("investigation_round", 0),
                    },
                )
            )
        return {
            "phase": "diagnose",
            "diagnosis": diagnosis,
            "events": events,
        }

    def after_diagnosis(state: EvoCIState) -> str:
        return diagnosis_route(state, config)

    async def repair(state: EvoCIState) -> dict[str, Any]:
        diagnosis = state.get("diagnosis")
        if diagnosis is None:
            return {"status": "failed", "failure_reason": "repair has no diagnosis"}
        attempt = state.get("repair_attempt", 0) + 1
        invocation_id = f"repair:{attempt}"
        try:
            output = await runtime.agents.fixer.propose(
                context=_context(state, invocation_id=invocation_id),
                diagnosis=diagnosis,
                evidence=state.get("evidence", []),
                previous_verification=state.get("verification"),
            )
        except (RepairBudgetExhausted, ModelGatewayError) as exc:
            return {
                "phase": "repair",
                "repair_attempt": attempt,
                "status": "failed",
                "failure_reason": str(exc),
                "fixer_output": None,
            }
        baseline = _snapshot_edit_baseline(Path(state["workspace_path"]), output.edits)
        targets = attempt_targets(output.edits)
        usage_events, used_memories, used_skills = _attribution(
            runtime,
            state,
            agent_id="fixer",
            invocation_id=invocation_id,
            memory_ids=output.used_memory_ids,
            skill_refs=output.used_skill_refs,
            discriminator=f"attempt:{attempt}",
        )
        return {
            "phase": "repair",
            "repair_attempt": attempt,
            "fixer_output": output,
            "approved": None,
            "attempt_baseline": baseline,
            "attempt_targets": targets,
            "attempt_written": {},
            "used_memory_ids": used_memories,
            "used_skill_refs": used_skills,
            "events": [
                *usage_events,
                _event(
                    runtime,
                    state,
                    EventType.PATCH_CREATED,
                    agent_id="fixer",
                    invocation_id=invocation_id,
                    discriminator=str(attempt),
                    payload={"fixer_output": output.model_dump()},
                ),
            ],
        }

    async def approval(state: EvoCIState) -> dict[str, Any]:
        output = state.get("fixer_output")
        if output is None:
            return {"status": "failed", "failure_reason": "approval has no patch"}
        if not requires_approval(output):
            return {"phase": "approval", "approved": True}
        if runtime.run_store is not None:
            runtime.run_store.update_status(state["run_id"], "waiting_approval")
        decision = interrupt(
            {
                "kind": "patch_approval",
                "summary": output.proposal.summary,
                "risk": output.proposal.risk,
                "files": output.proposal.changed_files,
            }
        )
        approved = bool(decision)
        events = [
            _event(
                runtime,
                state,
                EventType.INTERRUPT_RESOLVED,
                discriminator=str(state.get("repair_attempt", 0)),
                payload={"approved": approved},
            )
        ]
        if not approved:
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.ATTEMPT_FAILED,
                    discriminator=f"approval:{state.get('repair_attempt', 0)}",
                    payload={
                        "kind": "patch_rejected",
                        "reason": "patch rejected by human",
                        "iteration": state.get("repair_attempt", 0),
                    },
                )
            )
        return {
            "phase": "approval",
            "approved": approved,
            "status": "running" if approved else "failed",
            "failure_reason": None if approved else "patch rejected by human",
            "events": events,
        }

    def after_repair(state: EvoCIState) -> str:
        return "failed" if state.get("status") == "failed" else "approval"

    async def rollback_attempt(state: EvoCIState) -> dict[str, Any]:
        attempt = state.get("repair_attempt", 0)
        call_id = f"rollback:{attempt}"
        events = [
            _event(
                runtime,
                state,
                EventType.TOOL_CALL,
                agent_id="harness",
                discriminator=call_id,
                payload={
                    "call_id": call_id,
                    "tool_name": "rollback_attempt",
                    "budget_scope": "cleanup",
                },
            )
        ]
        try:
            written = state.get("attempt_written")
            if written is None:
                restored, removed = _restore_edit_baseline(
                    Path(state["workspace_path"]), state.get("attempt_baseline", {})
                )
            else:
                restored, removed = restore_attempt_writes(
                    Path(state["workspace_path"]),
                    state.get("attempt_baseline", {}),
                    written,
                )
        except Exception as exc:
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.TOOL_RESULT,
                    agent_id="harness",
                    discriminator=call_id,
                    payload={
                        "call_id": call_id,
                        "tool_name": "rollback_attempt",
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            return {
                "phase": "rollback_attempt",
                "status": "failed",
                "failure_reason": f"attempt rollback failed: {exc}",
                "events": events,
            }
        events.append(
            _event(
                runtime,
                state,
                EventType.TOOL_RESULT,
                agent_id="harness",
                discriminator=call_id,
                payload={
                    "call_id": call_id,
                    "tool_name": "rollback_attempt",
                    "success": True,
                    "modified_files": restored,
                    "created_files": [],
                    "removed_files": removed,
                },
            )
        )
        return {"phase": "rollback_attempt", "events": events}

    def after_approval(state: EvoCIState) -> str:
        return "apply_patch" if state.get("approved") else "failed"

    async def apply_patch(state: EvoCIState) -> dict[str, Any]:
        output = state.get("fixer_output")
        if output is None:
            return {"status": "failed", "failure_reason": "apply has no patch"}
        root = Path(state["workspace_path"])
        tools = FileTools(root, writable=True, max_chars=config.output_limit_chars)
        events: list[RunEvent] = []
        baseline = state.get("attempt_baseline", {})
        written = recover_attempt_writes(root, output.edits)

        def restore() -> None:
            restore_attempt_writes(root, baseline, written)

        def failed(reason: str) -> dict[str, Any]:
            restore()
            return {
                "phase": "apply_patch",
                "status": "failed",
                "failure_reason": reason,
                "attempt_written": written,
                "events": events,
            }

        try:
            precheck_edits(root, output.edits)
            if output.edits:
                assert runtime.budget_manager is not None
                runtime.budget_manager.for_run(state["run_id"]).ensure_tool_calls(len(output.edits))
            for index, edit in enumerate(output.edits):
                call_id = f"apply:{state.get('repair_attempt', 0)}:{index}"
                try:
                    consume_tool(state)
                except RepairBudgetExhausted as exc:
                    return failed(str(exc))
                events.append(
                    _event(
                        runtime,
                        state,
                        EventType.TOOL_CALL,
                        agent_id="harness",
                        discriminator=call_id,
                        payload={
                            "call_id": call_id,
                            "tool_name": "apply_patch",
                            "arguments": {"path": edit.path, "delete": edit.delete},
                        },
                    )
                )
                started = monotonic()
                try:
                    created, modified = _apply_edit(root, tools, edit)
                except Exception as exc:
                    events.append(
                        _event(
                            runtime,
                            state,
                            EventType.TOOL_RESULT,
                            agent_id="harness",
                            discriminator=call_id,
                            payload={
                                "call_id": call_id,
                                "tool_name": "apply_patch",
                                "success": False,
                                "error": f"{type(exc).__name__}: {exc}",
                                "duration_seconds": monotonic() - started,
                            },
                        )
                    )
                    return failed(f"patch apply failed: {exc}")
                except BaseException:
                    intended = None if edit.delete else edit.content
                    target = tools.boundary.resolve(edit.path)
                    current = (
                        target.read_text(encoding="utf-8")
                        if target.exists() and target.is_file()
                        else None
                    )
                    if current == intended:
                        written[edit.path] = intended
                    raise
                written[edit.path] = None if edit.delete else edit.content
                events.append(
                    _event(
                        runtime,
                        state,
                        EventType.TOOL_RESULT,
                        agent_id="harness",
                        discriminator=call_id,
                        payload={
                            "call_id": call_id,
                            "tool_name": "apply_patch",
                            "success": True,
                            "created_files": created,
                            "modified_files": modified,
                            "duration_seconds": monotonic() - started,
                        },
                    )
                )
            return {"phase": "apply_patch", "attempt_written": written, "events": events}
        except RepairBudgetExhausted as exc:
            return failed(str(exc))
        except (PatchConflict, PatchError) as exc:
            return failed(f"patch apply failed: {exc}")
        except BaseException:
            restore()
            raise

    async def verify(state: EvoCIState) -> dict[str, Any]:
        output = state.get("fixer_output")
        if output is None:
            return {"status": "failed", "failure_reason": "verify has no patch"}
        planned = build_verification_plan(
            state["ci_failure"].failed_commands,
            output.proposal.verification_plan,
        )
        attempt = state.get("repair_attempt", 0)
        events = [
            _event(
                runtime,
                state,
                EventType.VERIFICATION_STARTED,
                discriminator=str(attempt),
                payload={
                    "commands": [item.command for item in planned],
                    "oracle_source": (
                        "harness"
                        if any(item.source == "mandatory" for item in planned)
                        else "none"
                    ),
                },
            )
        ]
        started_at = monotonic()

        async def on_command_start(index: int, command: list[str], source: str) -> None:
            del source
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.TOOL_CALL,
                    agent_id="harness",
                    discriminator=f"verify:{attempt}:{index}",
                    payload={
                        "call_id": f"verify:{attempt}:{index}",
                        "tool_name": "run_test",
                        "arguments": {"argv": command, "cwd": ".", "network": False},
                    },
                )
            )

        async def on_command_done(index: int, result: Any) -> None:
            passed = result.executed and result.exit_code == 0 and not result.timed_out
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.TOOL_RESULT,
                    agent_id="harness",
                    discriminator=f"verify:{attempt}:{index}",
                    payload={
                        "call_id": f"verify:{attempt}:{index}",
                        "tool_name": "run_test",
                        "success": passed,
                        "exit_code": result.exit_code,
                        "error": result.stderr if not passed else None,
                        "duration_seconds": monotonic() - started_at,
                        "source": result.source,
                        "executed": result.executed,
                    },
                )
            )

        assert runtime.budget_manager is not None
        try:
            verification = await VerificationService(
                timeout=config.command_timeout_seconds,
                max_chars=config.output_limit_chars,
            ).run(
                workspace=Path(state["workspace_path"]),
                mandatory=state["ci_failure"].failed_commands,
                supplementary=output.proposal.verification_plan,
                budget=runtime.budget_manager.for_run(state["run_id"]),
                on_command_start=on_command_start,
                on_command_done=on_command_done,
            )
        except BaseException:
            restore_attempt_writes(
                Path(state["workspace_path"]),
                state.get("attempt_baseline", {}),
                state.get("attempt_written") or {},
            )
            raise
        if not verification.passed:
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.ATTEMPT_FAILED,
                    discriminator=f"verification:{attempt}",
                    payload={
                        "kind": "verification_failure",
                        "reason": verification.incomplete_reason
                        or next(
                            (
                                item.stderr or f"exit code {item.exit_code}"
                                for item in verification.commands
                                if item.executed and (item.exit_code != 0 or item.timed_out)
                            ),
                            "verification did not produce a verified success",
                        ),
                        "iteration": attempt,
                        "status": verification.status,
                    },
                )
            )
        events.append(
            _event(
                runtime,
                state,
                EventType.VERIFICATION_COMPLETED,
                discriminator=str(attempt),
                payload={"verification": verification.model_dump()},
            )
        )
        failure_reason = None
        status_update: dict[str, Any] = {}
        if verification.status == "unavailable":
            status_update["status"] = "failed"
            failure_reason = verification.incomplete_reason
        elif (
            not verification.passed and attempt >= config.max_repair_attempts
        ):
            failure_reason = (
                verification.incomplete_reason
                or "verification failed after repair budget was exhausted"
            )
        return {
            "phase": "verify",
            "verification": verification,
            "verification_history": [verification],
            "previous_attempt_summary": (
                verification.incomplete_reason
                or next(
                    (
                        item.stderr or f"exit code {item.exit_code}"
                        for item in verification.commands
                        if item.executed and (item.exit_code != 0 or item.timed_out)
                    ),
                    "verification did not produce a verified success",
                )
                if not verification.passed
                else state.get("previous_attempt_summary")
            ),
            "failure_reason": failure_reason,
            "events": events,
            **status_update,
        }

    def after_apply(state: EvoCIState) -> str:
        return "rollback_attempt" if state.get("status") == "failed" else "verify"

    def after_verify(state: EvoCIState) -> str:
        if state.get("status") == "failed":
            return "rollback_attempt"
        verification = state.get("verification")
        if verification and verification.passed:
            return "review"
        if state.get("repair_attempt", 0) < config.max_repair_attempts:
            return "rollback_attempt"
        return "rollback_attempt"

    async def review(state: EvoCIState) -> dict[str, Any]:
        output = state.get("fixer_output")
        diagnosis = state.get("diagnosis")
        verification = state.get("verification")
        if output is None or diagnosis is None or verification is None:
            return {"status": "failed", "failure_reason": "review context incomplete"}
        if not verification.passed:
            return {
                "status": "failed",
                "failure_reason": "review cannot override hard verification failure",
            }
        invocation_id = f"review:{state.get('repair_attempt', 0)}"
        deterministic_blockers = sorted(
            set(
                contains_review_bypass(output)
                + contains_workspace_review_bypass(Path(state["workspace_path"]))
            )
        )
        if deterministic_blockers:
            result = ReviewResult(
                accepted=False,
                blockers=deterministic_blockers,
                confidence=1.0,
            )
        else:
            try:
                result = await runtime.agents.reviewer.review(
                    context=_context(state, invocation_id=invocation_id),
                    diagnosis=diagnosis,
                    patch=output,
                    verification=verification,
                )
            except (RepairBudgetExhausted, ModelGatewayError) as exc:
                return {
                    "phase": "review",
                    "status": "failed",
                    "failure_reason": str(exc),
                }
        usage_events, used_memories, used_skills = _attribution(
            runtime,
            state,
            agent_id="reviewer",
            invocation_id=invocation_id,
            memory_ids=result.used_memory_ids,
            skill_refs=result.used_skill_refs,
            discriminator=f"attempt:{state.get('repair_attempt', 0)}",
        )
        events = usage_events
        if not result.accepted:
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.ATTEMPT_FAILED,
                    agent_id="reviewer",
                    invocation_id=invocation_id,
                    discriminator=f"review:{state.get('repair_attempt', 0)}",
                    payload={
                        "kind": "review_rejected",
                        "agent_id": "reviewer",
                        "reason": "; ".join(result.blockers) or "review rejected patch",
                        "iteration": state.get("repair_attempt", 0),
                    },
                )
            )
        return {
            "phase": "review",
            "review": result,
            "previous_review_blockers": result.blockers if not result.accepted else [],
            "previous_attempt_summary": (
                "; ".join(result.blockers) or "review rejected patch"
                if not result.accepted
                else None
            ),
            "failure_reason": (
                "; ".join(result.blockers) or "review rejected final repair attempt"
                if not result.accepted
                and state.get("repair_attempt", 0) >= config.max_repair_attempts
                else None
            ),
            "used_memory_ids": used_memories,
            "used_skill_refs": used_skills,
            "events": events,
        }

    def after_review(state: EvoCIState) -> str:
        if state.get("status") == "failed":
            return "rollback_attempt"
        review_result = state.get("review")
        if review_result and review_result.accepted:
            return "finalize"
        if state.get("repair_attempt", 0) < config.max_repair_attempts:
            return "rollback_attempt"
        return "rollback_attempt"

    def after_rollback(state: EvoCIState) -> str:
        if state.get("status") == "failed":
            return "failed"
        if state.get("repair_attempt", 0) < config.max_repair_attempts:
            return "repair"
        return "failed"

    async def persist_outcome(
        state: EvoCIState, *, success: bool, failure_reason: str | None
    ) -> dict[str, Any]:
        return await persist_run_outcome(
            runtime, state, success=success, failure_reason=failure_reason
        )

    async def finalize(state: EvoCIState) -> dict[str, Any]:
        if runtime.defer_success_learning:
            return {
                "phase": "finalize",
                "status": "success",
                "failure_reason": None,
                "learning_deferred": True,
            }
        return await persist_outcome(state, success=True, failure_reason=None)

    async def failed(state: EvoCIState) -> dict[str, Any]:
        reason = state.get("failure_reason") or "orchestration budget exhausted"
        return await persist_outcome(state, success=False, failure_reason=reason)

    builder = StateGraph(EvoCIState)
    builder.add_node("bootstrap", bootstrap)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("coordinate", coordinate)
    builder.add_node("investigate", investigate)
    builder.add_node("diagnose", diagnose)
    builder.add_node("repair", repair)
    builder.add_node("approval", approval)
    builder.add_node("apply_patch", apply_patch)
    builder.add_node("verify", verify)
    builder.add_node("review", review)
    builder.add_node("rollback_attempt", rollback_attempt)
    builder.add_node("finalize", finalize)
    builder.add_node("failed", failed)

    builder.add_edge(START, "bootstrap")
    builder.add_edge("bootstrap", "retrieve_context")
    builder.add_edge("retrieve_context", "coordinate")
    builder.add_conditional_edges("coordinate", dispatch, ["investigate", "failed"])
    builder.add_edge("investigate", "diagnose")
    builder.add_conditional_edges(
        "diagnose",
        after_diagnosis,
        {"repair": "repair", "coordinate": "coordinate", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "repair", after_repair, {"approval": "approval", "failed": "failed"}
    )
    builder.add_conditional_edges(
        "approval", after_approval, {"apply_patch": "apply_patch", "failed": "failed"}
    )
    builder.add_conditional_edges(
        "apply_patch",
        after_apply,
        {"verify": "verify", "rollback_attempt": "rollback_attempt"},
    )
    builder.add_conditional_edges(
        "verify",
        after_verify,
        {"review": "review", "rollback_attempt": "rollback_attempt"},
    )
    builder.add_conditional_edges(
        "review",
        after_review,
        {"finalize": "finalize", "rollback_attempt": "rollback_attempt"},
    )
    builder.add_conditional_edges(
        "rollback_attempt", after_rollback, {"repair": "repair", "failed": "failed"}
    )
    builder.add_edge("finalize", END)
    builder.add_edge("failed", END)
    return cast(CompiledStateGraph[Any, Any, Any, Any], builder.compile(checkpointer=checkpointer))
