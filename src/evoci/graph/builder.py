"""Supervisor-Worker LangGraph orchestration for CI recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from evoci.agents.base import AgentSuite, SupervisorContext, WorkerContext
from evoci.capability.materializer import CapabilityMaterializer
from evoci.capability.miner import SkillMining
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.config import EvoCIConfig
from evoci.domain.models import (
    FailedCandidateRef,
    ReviewResult,
    SkillCatalogEntry,
    SkillRef,
    SupervisorDecision,
    WorkerExecutionResult,
    WorkerTask,
)
from evoci.graph.integration import (
    IntegrationError,
    apply_plan_path,
    artifact_dir,
    fixer_output_from_edits,
    load_apply_plan,
    restore_batch_snapshot,
    save_apply_plan,
    save_batch_snapshot,
    save_patch_artifact,
    validate_edits_against_scope,
)
from evoci.graph.outcome import (
    classified_failure,
    emit_event,
    persist_run_outcome,
    resolve_failure_class,
)
from evoci.graph.routing import contains_workspace_review_bypass, requires_approval
from evoci.graph.scheduler import TaskScheduleError, validate_single_task
from evoci.graph.state import ORCHESTRATION_SCHEMA_VERSION, EvoCIState, LegacyOrchestrationError
from evoci.memory.consolidation import MemoryConsolidator
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import MemoryStore
from evoci.model.gateway import ModelGatewayError
from evoci.runtime.budget import RepairBudgetExhausted, RunBudgetManager
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.events import EventType, RunEvent
from evoci.runtime.run_store import SQLiteRunStore
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.runtime.usage import usage_complete_from_events
from evoci.tools.filesystem import FileTools
from evoci.tools.isolation import workspace_snapshot_id
from evoci.tools.patch import (
    PatchConflict,
    PatchError,
    apply_edit,
    restore_attempt_writes,
    restore_edit_baseline,
    snapshot_edit_baseline,
)
from evoci.tools.policy import (
    WORKER_INVESTIGATE_CAPABILITIES,
    WORKER_REPAIR_CAPABILITIES,
    PolicyViolation,
)
from evoci.verification.service import VerificationService, build_verification_plan

_apply_edit = apply_edit
_restore_edit_baseline = restore_edit_baseline


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
    skill_miner: SkillMining | None = None
    recorder: TrajectoryRecorder | None = None
    budget_manager: RunBudgetManager | None = None
    defer_success_learning: bool = False


def _event(
    runtime: GraphRuntime,
    state: EvoCIState,
    event_type: EventType,
    **kwargs: Any,
) -> RunEvent:
    return emit_event(runtime, state, event_type, **kwargs)


def _operation_key(state: EvoCIState, key_type: str, *components: str) -> str:
    prefix = f"{key_type}:{state['run_id']}"
    suffix = ":".join(components)
    return f"{prefix}:{suffix}" if suffix else prefix


def _coerce_skill_catalog(raw: object) -> list[SkillCatalogEntry]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise LegacyOrchestrationError("malformed checkpoint skill_catalog")
    catalog: list[SkillCatalogEntry] = []
    for item in raw:
        if isinstance(item, SkillCatalogEntry):
            catalog.append(item)
            continue
        if not isinstance(item, dict):
            raise LegacyOrchestrationError("malformed checkpoint skill_catalog")
        try:
            catalog.append(SkillCatalogEntry.model_validate(item))
        except ValueError as exc:
            raise LegacyOrchestrationError("malformed checkpoint skill_catalog") from exc
    return catalog


def _slice_memories(state: EvoCIState, task: WorkerTask) -> list[Any]:
    retrieved = state.get("retrieved_memories", [])
    if not task.fact_refs:
        return []
    allowed = set(task.fact_refs)
    return [memory for memory in retrieved if memory.memory_id in allowed]


def _slice_skills(state: EvoCIState, task: WorkerTask) -> list[Any]:
    catalog = _coerce_skill_catalog(state.get("skill_catalog"))
    allowed = {ref.skill_id for ref in task.recommended_skill_refs}
    return [skill for skill in catalog if skill.skill_id in allowed]


def _attribution(
    runtime: GraphRuntime,
    *,
    agent_id: str,
    invocation_id: str,
    memory_ids: list[str],
    skill_refs: list[SkillRef],
    available_memories: set[str],
    available_skills: set[str],
    discriminator: str,
    state: EvoCIState,
) -> tuple[list[RunEvent], list[str], list[SkillRef]]:
    used_memories = sorted(available_memories.intersection(memory_ids))
    used_skills = [ref for ref in skill_refs if ref.skill_id in available_skills]
    already_used: set[str] = set()
    if runtime.recorder is not None:
        for event in runtime.recorder.events(state["run_id"]):
            if event.type != EventType.SKILL_USED or event.invocation_id != invocation_id:
                continue
            if event.payload.get("skill_id"):
                already_used.add(str(event.payload["skill_id"]))
            for raw in event.payload.get("skills") or []:
                if isinstance(raw, dict) and raw.get("skill_id"):
                    already_used.add(str(raw["skill_id"]))
    missing_skills = [ref for ref in used_skills if ref.skill_id not in already_used]
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
    if missing_skills:
        events.append(
            _event(
                runtime,
                state,
                EventType.SKILL_USED,
                agent_id=agent_id,
                invocation_id=invocation_id,
                discriminator=discriminator,
                payload={"skills": [ref.model_dump() for ref in missing_skills]},
            )
        )
    return events, used_memories, used_skills


def _worker_summaries(state: EvoCIState) -> tuple[dict[str, object], ...]:
    summaries: list[dict[str, object]] = []
    for result in state.get("worker_results", []):
        summaries.append(
            {
                "task_id": result.task_id,
                "status": result.status,
                "summary": result.summary,
                "changed_files": result.changed_files,
                "unresolved_questions": result.unresolved_questions,
                "patch_artifact_ref": result.patch_artifact_ref,
            }
        )
    return tuple(summaries)


def _formal_success(state: EvoCIState) -> bool:
    verification = state.get("verification")
    snapshot = workspace_snapshot_id(Path(state["workspace_path"]))
    return bool(
        verification
        and verification.passed
        and verification.status == "passed"
        and state.get("verification_snapshot_id") == snapshot
    )


def _caused_by_timeout(exc: BaseException) -> bool:
    """Recognize provider timeout failures without depending on message wording."""

    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(current, TimeoutError) or "timeout" in type(current).__name__.lower():
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


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
    for agent in (runtime.agents.supervisor, runtime.agents.worker):
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
        if state.get("schema_version") not in {None, ORCHESTRATION_SCHEMA_VERSION}:
            raise LegacyOrchestrationError()
        if state.get("investigation_round") or state.get("investigation_plan") is not None:
            raise LegacyOrchestrationError()
        if runtime.run_store is not None:
            runtime.run_store.create(
                state["run_id"],
                state["task_id"],
                {
                    "repo": state["repo"].model_dump(),
                    "ci_failure": state["ci_failure"].model_dump(),
                    "workspace_path": state["workspace_path"],
                    "schema_version": ORCHESTRATION_SCHEMA_VERSION,
                },
            )
        return {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "phase": "bootstrap",
            "status": "running",
            "supervisor_batch": state.get("supervisor_batch", 0),
            "decision_history": state.get("decision_history", []),
            "evidence": [],
            "budget_failures": [],
            "completed_task_ids": [],
            "worker_results": [],
            "retrieved_memories": state.get("retrieved_memories", []),
            "retrieved_skills": [],
            "skill_catalog": _coerce_skill_catalog(state.get("skill_catalog")),
            "selected_memory_ids": [],
            "used_memory_ids": [],
            "selected_skill_refs": [],
            "used_skill_refs": [],
            "recommended_skill_ids": [],
            "used_skills": state.get("used_skills", []),
            "verification_history": [],
            "failed_candidates": state.get("failed_candidates", []),
            "learning_errors": state.get("learning_errors", []),
            "failure_class": state.get("failure_class"),
            "failure_classes": [],
            "failure_stage": state.get("failure_stage"),
            "events": [
                _event(
                    runtime,
                    state,
                    EventType.RUN_STARTED,
                    payload={
                        "task": state["task_id"],
                        "campaign_provenance": state.get("campaign_provenance"),
                        "schema_version": ORCHESTRATION_SCHEMA_VERSION,
                    },
                )
            ],
        }

    async def retrieve_context(state: EvoCIState) -> dict[str, Any]:
        memories = state.get("retrieved_memories", [])
        catalog = _coerce_skill_catalog(state.get("skill_catalog"))
        selected_memory_ids = [memory.memory_id for memory in memories]
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
                    payload={"memory_ids": selected_memory_ids, "audience": "supervisor"},
                )
            )
        if runtime.capability_retriever is not None:
            result = runtime.capability_retriever.retrieve(
                state["repo"],
                state["ci_failure"],
                operation_key=_operation_key(state, "skill-retrieval"),
            )
            catalog = list(result.entries)
            retrieval_events.append(
                _event(
                    runtime,
                    state,
                    EventType.SKILL_RETRIEVED,
                    payload={
                        "skills": [entry.model_dump() for entry in catalog],
                        "omitted_count": result.omitted_count,
                        "context_chars": result.context_chars,
                        "audience": "supervisor",
                    },
                )
            )
        return {
            "phase": "retrieve_context",
            "retrieved_memories": memories,
            "retrieved_skills": [],
            "skill_catalog": catalog,
            "selected_memory_ids": selected_memory_ids,
            "selected_skill_refs": [],
            "events": retrieval_events,
        }

    async def supervise(state: EvoCIState) -> dict[str, Any]:
        if state.get("status") == "failed":
            return {"phase": "supervise"}
        batch = state.get("supervisor_batch", 0) + 1
        if batch > config.max_supervisor_batches:
            return {
                "phase": "supervise",
                "status": "failed",
                "failure_class": "budget",
                "failure_stage": "supervise",
                "failure_reason": "supervisor batch budget exhausted",
            }
        invocation_id = f"supervise:{batch}"
        assert runtime.budget_manager is not None
        budget = runtime.budget_manager.for_run(state["run_id"])
        snapshot = budget.snapshot()
        usage_complete = True
        if runtime.recorder is not None:
            usage_complete = usage_complete_from_events(runtime.recorder.events(state["run_id"]))
        context = SupervisorContext(
            run_id=state["run_id"],
            repo=state["repo"],
            failure=state["ci_failure"],
            workspace_path=state["workspace_path"],
            invocation_id=invocation_id,
            memories=tuple(state.get("retrieved_memories", [])),
            skills=tuple(_coerce_skill_catalog(state.get("skill_catalog"))),
            previous_decisions=tuple(state.get("decision_history", [])),
            worker_result_summaries=_worker_summaries(state),
            last_verification=state.get("verification"),
            verification_snapshot_id=state.get("verification_snapshot_id"),
            remaining_batches=config.max_supervisor_batches - batch + 1,
            remaining_model_calls=snapshot.max_model_calls - snapshot.model_calls,
            remaining_tool_calls=snapshot.max_tool_calls - snapshot.tool_calls,
            failed_candidates=tuple(state.get("failed_candidates", [])),
            usage_complete=usage_complete,
        )
        completed_invocation_id = invocation_id
        try:
            decision = await runtime.agents.supervisor.decide(context=context)
        except RepairBudgetExhausted as exc:
            return classified_failure(exc, phase="supervise", supervisor_batch=batch)
        except ModelGatewayError as exc:
            if not _caused_by_timeout(exc):
                return classified_failure(exc, phase="supervise", supervisor_batch=batch)
            # The gateway exhausted its internal transport retries. Give the
            # Supervisor one fresh invocation without spending another batch.
            completed_invocation_id = f"{invocation_id}:timeout-retry"
            retry_context = replace(context, invocation_id=completed_invocation_id)
            try:
                decision = await runtime.agents.supervisor.decide(context=retry_context)
            except (RepairBudgetExhausted, ModelGatewayError) as retry_exc:
                return classified_failure(
                    retry_exc,
                    phase="supervise",
                    supervisor_batch=batch,
                )
        history = list(state.get("decision_history", []))
        history.append(f"{decision.action}: {decision.reasoning_summary}")
        recommended = [
            ref.skill_id for task in decision.tasks for ref in task.recommended_skill_refs
        ]
        return {
            "phase": "supervise",
            "supervisor_batch": batch,
            "decision": decision,
            "decision_history": history,
            "recommended_skill_ids": recommended,
            "batch_conflict": None,
            "events": [
                _event(
                    runtime,
                    state,
                    EventType.AGENT_COMPLETED,
                    agent_id="supervisor",
                    invocation_id=completed_invocation_id,
                    discriminator=str(batch),
                    payload={
                        "action": decision.action,
                        "tasks": [task.model_dump() for task in decision.tasks],
                        "stop_reason": decision.stop_reason,
                        "reasoning_summary": decision.reasoning_summary,
                    },
                )
            ],
        }


    def _coerce_task(raw: WorkerTask | dict[str, Any] | None) -> WorkerTask:
        if isinstance(raw, WorkerTask):
            return raw
        if isinstance(raw, dict):
            return WorkerTask.model_validate(raw)
        raise TypeError("worker_task is required")

    def _coerce_decision(
        raw: SupervisorDecision | dict[str, Any] | None,
    ) -> SupervisorDecision | None:
        if raw is None or isinstance(raw, SupervisorDecision):
            return raw
        return SupervisorDecision.model_validate(raw)

    def _latest_result(state: EvoCIState) -> WorkerExecutionResult | None:
        results = list(state.get("worker_results", []))
        if not results:
            return None
        item: Any = results[-1]
        if isinstance(item, WorkerExecutionResult):
            return item
        return WorkerExecutionResult.model_validate(item)

    def _file_text(root: Path, relative: str) -> str | None:
        target = root / relative
        if target.exists() and target.is_file():
            return target.read_text(encoding="utf-8")
        return None

    def _baseline_content(baseline: dict[str, Any], path: str) -> str | None:
        value = baseline.get(path)
        if value is None:
            return None
        if isinstance(value, dict):
            content = value.get("content")
            return content if isinstance(content, str) or content is None else None
        if isinstance(value, str):
            return value
        return None

    def after_supervise(state: EvoCIState) -> str:
        if state.get("status") == "failed":
            return "failed"
        decision = _coerce_decision(state.get("decision"))
        if decision is None:
            return "failed"
        if decision.action == "stop":
            return "finalize" if _formal_success(state) else "failed"
        return "prepare_task"

    async def prepare_task(state: EvoCIState) -> dict[str, Any]:
        decision = _coerce_decision(state.get("decision"))
        if decision is None or decision.action != "dispatch":
            return {
                "status": "failed",
                "failure_class": "repair",
                "failure_stage": "prepare_task",
                "failure_reason": "supervisor did not dispatch a task",
            }
        assert runtime.budget_manager is not None
        budget = runtime.budget_manager.for_run(state["run_id"]).snapshot()
        remaining_model = budget.max_model_calls - budget.model_calls
        remaining_tool = budget.max_tool_calls - budget.tool_calls
        if remaining_model < 2:
            return {
                "status": "failed",
                "failure_class": "budget",
                "failure_stage": "worker",
                "failure_reason": "run-level model-call budget exhausted",
                "failure_classes": ["budget"],
                "budget_failures": ["run-level model-call budget exhausted"],
            }
        try:
            task = validate_single_task(
                decision.tasks,
                known_task_ids=set(state.get("completed_task_ids", [])),
                remaining_model_calls=remaining_model,
                remaining_tool_calls=remaining_tool,
            )
        except TaskScheduleError as exc:
            remaining = config.max_supervisor_batches - state.get("supervisor_batch", 0)
            history = list(state.get("decision_history", []))
            history.append(f"harness rejected dispatch: {exc}")
            if remaining > 0:
                return {
                    "phase": "prepare_task",
                    "decision_history": history,
                    "batch_conflict": str(exc),
                    "worker_task": None,
                }
            return {
                "status": "failed",
                "failure_class": "repair",
                "failure_stage": "prepare_task",
                "failure_reason": str(exc),
            }
        workspace = Path(state["workspace_path"]).resolve()
        snapshot_path = (
            workspace.parent
            / f".evoci-snapshots-{workspace.name}-{state['run_id']}"
            / f"batch-{state.get('supervisor_batch', 0)}"
        )
        if snapshot_path.exists():
            snapshot_id = str(
                state.get("batch_snapshot_id") or workspace_snapshot_id(snapshot_path)
            )
        else:
            try:
                snapshot_id = save_batch_snapshot(workspace, snapshot_path)
            except Exception as exc:
                return {
                    "status": "failed",
                    "failure_class": "infrastructure",
                    "failure_stage": "prepare_task",
                    "failure_reason": f"parent snapshot failed: {exc}",
                }
        resume_ref = None
        if task.resume_candidate_id:
            for candidate in state.get("failed_candidates", []):
                if candidate.candidate_id == task.resume_candidate_id:
                    resume_ref = candidate.artifact_ref
                    break
        return {
            "phase": "prepare_task",
            "worker_task": task,
            "worker_memories": _slice_memories(state, task),
            "recommended_skills": _slice_skills(state, task),
            "worker_baseline_id": snapshot_id,
            "batch_snapshot_path": str(snapshot_path),
            "batch_snapshot_id": snapshot_id,
            "resume_artifact_ref": resume_ref,
            "batch_conflict": None,
            "approved": None,
        }

    def after_prepare(state: EvoCIState) -> str:
        if state.get("status") == "failed":
            return "failed"
        if state.get("batch_conflict"):
            if state.get("supervisor_batch", 0) < config.max_supervisor_batches:
                return "supervise"
            return "failed"
        if state.get("worker_task") is None:
            return "failed"
        return "worker"

    async def worker(state: EvoCIState) -> dict[str, Any]:
        task = _coerce_task(state.get("worker_task"))
        agent_id = f"worker:{task.task_id}"
        invocation_id = f"worker:{state.get('supervisor_batch', 0)}:{task.task_id}"
        capabilities = (
            WORKER_REPAIR_CAPABILITIES if task.kind == "repair" else WORKER_INVESTIGATE_CAPABILITIES
        )
        memories = list(state.get("worker_memories", []))
        skills = list(state.get("recommended_skills", []))
        context = WorkerContext(
            run_id=state["run_id"],
            repo=state["repo"],
            failure=state["ci_failure"],
            workspace_path=state["workspace_path"],
            invocation_id=invocation_id,
            task=task,
            memories=tuple(memories),
            recommended_skills=tuple(skills),
            baseline_snapshot_id=str(state.get("worker_baseline_id") or ""),
            previous_attempt_summary=state.get("previous_attempt_summary"),
            resume_artifact_ref=state.get("resume_artifact_ref"),
        )
        try:
            run = await runtime.agents.worker.execute(context=context, capabilities=capabilities)
        except RepairBudgetExhausted as exc:
            return {
                "completed_task_ids": [task.task_id],
                "worker_results": [
                    WorkerExecutionResult(
                        task_id=task.task_id,
                        status="budget_exhausted",
                        summary=str(exc),
                    )
                ],
                "budget_failures": [str(exc)],
                "failure_classes": ["budget"],
            }
        except ModelGatewayError as exc:
            return {
                "completed_task_ids": [task.task_id],
                "worker_results": [
                    WorkerExecutionResult(
                        task_id=task.task_id,
                        status="blocked",
                        summary=str(exc),
                    )
                ],
                "budget_failures": [str(exc)],
                "failure_classes": ["model"],
                "status": "failed",
                "failure_class": "model",
                "failure_stage": "worker",
                "failure_reason": str(exc),
            }
        artifact_ref = None
        fixer_output = None
        if run.edits:
            directory = artifact_dir(
                config.state_dir,
                state["run_id"],
                int(state.get("supervisor_batch", 0)),
                task.task_id,
            )
            artifact_ref = save_patch_artifact(
                directory,
                task=task,
                edits=run.edits,
                baseline_snapshot_id=context.baseline_snapshot_id,
                summary=run.result.summary,
                commands_run=run.commands_run,
                verification_plan=run.verification_plan,
            )
            fixer_output = fixer_output_from_edits(
                summary=run.result.summary,
                edits=run.edits,
                commands_run=run.commands_run,
                verification_plan=run.verification_plan,
                risk=run.risk,
            )
        result = run.result.model_copy(
            update={
                "patch_artifact_ref": artifact_ref,
                "changed_files": [edit.path for edit in run.edits] or run.result.changed_files,
                "base_revision": run.result.base_revision or context.baseline_snapshot_id,
                "snapshot_id": run.result.snapshot_id or context.baseline_snapshot_id,
            }
        )
        budget_update: dict[str, Any] = {}
        if result.status == "budget_exhausted" and not run.edits:
            budget_update = {
                "status": "failed",
                "failure_class": "budget",
                "failure_stage": "worker",
                "failure_reason": result.summary,
                "failure_classes": ["budget"],
                "budget_failures": [result.summary],
            }
        usage_events, used_memories, used_skills = _attribution(
            runtime,
            agent_id=agent_id,
            invocation_id=invocation_id,
            memory_ids=result.used_memory_ids,
            skill_refs=result.used_skill_refs,
            available_memories={memory.memory_id for memory in memories},
            available_skills={skill.skill_id for skill in skills},
            discriminator=task.task_id,
            state=state,
        )
        return {
            "evidence": result.evidence,
            "completed_task_ids": [task.task_id],
            "worker_results": [result],
            "fixer_output": fixer_output or state.get("fixer_output"),
            "used_memory_ids": used_memories,
            "used_skill_refs": used_skills,
            **budget_update,
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
                    payload={
                        "status": result.status,
                        "kind": task.kind,
                        "evidence_count": len(result.evidence),
                        "changed_files": result.changed_files,
                    },
                ),
            ],
        }

    def after_worker(state: EvoCIState) -> str:
        if state.get("status") == "failed":
            return "failed"
        classes = state.get("failure_classes") or []
        if classes and classes[-1] in {"model", "infrastructure"}:
            return "failed"
        result = _latest_result(state)
        task = state.get("worker_task")
        kind = None
        if task is not None:
            kind = _coerce_task(task).kind
        if result is None:
            return "failed"
        if result.status == "budget_exhausted" or state.get("budget_failures"):
            return "failed"
        if result.status == "blocked":
            return "supervise"
        if kind == "investigate":
            return "supervise"
        if kind == "repair":
            return "approval"
        return "supervise"

    async def approval(state: EvoCIState) -> dict[str, Any]:
        output = state.get("fixer_output")
        if output is None or not output.edits:
            return {"phase": "approval", "approved": True}
        if not requires_approval(output):
            return {"phase": "approval", "approved": True, "fixer_output": output}
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
                discriminator=str(state.get("supervisor_batch", 0)),
                payload={"approved": approved},
            )
        ]
        return {
            "phase": "approval",
            "approved": approved,
            "fixer_output": output,
            "status": "running" if approved else "failed",
            "failure_class": None if approved else "policy",
            "failure_stage": None if approved else "approval",
            "failure_reason": None if approved else "patch rejected by human",
            "events": events,
        }

    def after_approval(state: EvoCIState) -> str:
        if state.get("status") == "failed" or state.get("approved") is False:
            return "failed"
        return "apply_candidate"

    async def apply_candidate(state: EvoCIState) -> dict[str, Any]:
        output = state.get("fixer_output")
        task = _coerce_task(state.get("worker_task")) if state.get("worker_task") else None
        root = Path(state["workspace_path"])
        events: list[RunEvent] = []
        if output is None or not output.edits:
            return {
                "phase": "apply_candidate",
                "integrated_snapshot_id": workspace_snapshot_id(root),
            }
        if task is not None:
            try:
                validate_edits_against_scope(output.edits, task.write_scope)
            except PolicyViolation as exc:
                return {
                    "phase": "apply_candidate",
                    "batch_conflict": str(exc),
                    "failure_classes": ["repair"],
                    "failure_reason": str(exc),
                    "failure_stage": "apply_candidate",
                }
        expected = state.get("worker_baseline_id")
        operation_id = f"{state.get('supervisor_batch', 0)}:{task.task_id if task else 'repair'}"
        plan_file = apply_plan_path(config.state_dir, state["run_id"], operation_id)
        plan = load_apply_plan(plan_file)
        if plan is None:
            snapshot_path = state.get("batch_snapshot_path")
            snapshot_root = Path(snapshot_path) if snapshot_path else None
            if snapshot_root is not None and snapshot_root.exists():
                baseline_root = snapshot_root
                parent_id = str(
                    state.get("batch_snapshot_id") or workspace_snapshot_id(baseline_root)
                )
            else:
                current = workspace_snapshot_id(root)
                if expected and current != expected:
                    return {
                        "phase": "apply_candidate",
                        "batch_conflict": (
                            f"workspace snapshot {current} does not match baseline {expected}"
                        ),
                        "failure_classes": ["repair"],
                        "failure_reason": "workspace changed before apply",
                        "failure_stage": "apply_candidate",
                    }
                baseline_root = root
                parent_id = str(expected or current)
            baseline = snapshot_edit_baseline(baseline_root, output.edits)
            latest = _latest_result(state)
            save_apply_plan(
                plan_file,
                parent_snapshot_id=parent_id,
                artifact_ref=latest.patch_artifact_ref if latest else None,
                edits=output.edits,
                baseline=baseline,
            )
            plan = load_apply_plan(plan_file)
        assert plan is not None
        baseline = dict(plan.get("baseline") or {})
        written: dict[str, str | None] = {}
        try:
            tools = FileTools(root, writable=True)
            for edit in output.edits:
                intended = None if edit.delete else edit.content
                current_text = _file_text(root, edit.path)
                parent_text = _baseline_content(baseline, edit.path)
                if current_text == intended:
                    written[edit.path] = intended
                    continue
                if current_text != parent_text:
                    raise PatchConflict(f"file changed since proposal: {edit.path}")
                consume_tool(state)
                events.append(
                    _event(
                        runtime,
                        state,
                        EventType.TOOL_CALL,
                        agent_id=f"worker:{task.task_id if task else 'repair'}",
                        discriminator=f"apply:{operation_id}:{edit.path}",
                        payload={
                            "call_id": f"apply:{operation_id}:{edit.path}",
                            "tool_name": "apply_edit",
                            "arguments": {"path": edit.path},
                        },
                    )
                )
                written[edit.path] = intended
                _apply_edit(root, tools, edit, hash_strict=config.patch_hash_strict)
            latest = _latest_result(state)
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.PATCH_CREATED,
                    agent_id=f"worker:{task.task_id if task else 'repair'}",
                    discriminator=operation_id,
                    payload={
                        "changed_files": [edit.path for edit in output.edits],
                        "artifact_ref": latest.patch_artifact_ref if latest else None,
                    },
                )
            )
        except (
            PatchConflict,
            PatchError,
            IntegrationError,
            RepairBudgetExhausted,
            PolicyViolation,
            OSError,
        ) as exc:
            restore_attempt_writes(root, baseline, written)
            return {
                "phase": "apply_candidate",
                "attempt_baseline": baseline,
                "attempt_written": written,
                "batch_conflict": str(exc),
                "failure_classes": [
                    "budget" if isinstance(exc, RepairBudgetExhausted) else "repair"
                ],
                "failure_reason": str(exc),
                "failure_stage": "apply_candidate",
                "events": events,
            }
        except BaseException:
            restore_attempt_writes(root, baseline, written)
            raise
        return {
            "phase": "apply_candidate",
            "fixer_output": output,
            "attempt_baseline": baseline,
            "attempt_written": written,
            "integrated_snapshot_id": workspace_snapshot_id(root),
            "events": events,
        }

    def after_apply(state: EvoCIState) -> str:
        if state.get("status") == "failed":
            return "failed"
        if state.get("batch_conflict"):
            return "rollback"
        return "verify"

    async def verify(state: EvoCIState) -> dict[str, Any]:
        output = state.get("fixer_output")
        supplementary = output.proposal.verification_plan if output is not None else []
        planned = build_verification_plan(state["ci_failure"].failed_commands, supplementary)
        batch = state.get("supervisor_batch", 0)
        events = [
            _event(
                runtime,
                state,
                EventType.VERIFICATION_STARTED,
                discriminator=str(batch),
                payload={
                    "commands": [
                        {"argv": item.command, "cwd": item.cwd} for item in planned
                    ],
                    "oracle_source": (
                        "harness" if any(item.source == "mandatory" for item in planned) else "none"
                    ),
                    "snapshot_id": workspace_snapshot_id(Path(state["workspace_path"])),
                },
            )
        ]
        started_at = monotonic()

        async def on_command_start(index: int, command: list[str], source: str) -> None:
            del source
            cwd = planned[index].cwd if index < len(planned) else "."
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.TOOL_CALL,
                    agent_id="harness",
                    discriminator=f"verify:{batch}:{index}",
                    payload={
                        "call_id": f"verify:{batch}:{index}",
                        "tool_name": "run_test",
                        "arguments": {"argv": command, "cwd": cwd, "network": False},
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
                    discriminator=f"verify:{batch}:{index}",
                    payload={
                        "call_id": f"verify:{batch}:{index}",
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
        snapshot_id = workspace_snapshot_id(Path(state["workspace_path"]))
        try:
            verification = await VerificationService(
                timeout=config.command_timeout_seconds,
                max_chars=config.output_limit_chars,
            ).run(
                workspace=Path(state["workspace_path"]),
                mandatory=state["ci_failure"].failed_commands,
                supplementary=supplementary,
                budget=runtime.budget_manager.for_run(state["run_id"]),
                on_command_start=on_command_start,
                on_command_done=on_command_done,
            )
        except BaseException:
            snapshot = state.get("batch_snapshot_path")
            if snapshot:
                restore_batch_snapshot(Path(snapshot), Path(state["workspace_path"]))
            elif state.get("attempt_written") is not None:
                restore_attempt_writes(
                    Path(state["workspace_path"]),
                    state.get("attempt_baseline", {}),
                    state.get("attempt_written") or {},
                )
            raise
        review = None
        if verification.passed:
            # FileEdit.content is a complete replacement and can contain legitimate
            # pre-existing skip/noqa directives. Scan only additions in the final diff.
            blockers = contains_workspace_review_bypass(Path(state["workspace_path"]))
            if blockers:
                events.append(
                    _event(
                        runtime,
                        state,
                        EventType.ATTEMPT_FAILED,
                        discriminator=f"governance:{batch}",
                        payload={
                            "kind": "review_rejected",
                            "reason": "; ".join(blockers),
                            "iteration": batch,
                        },
                    )
                )
                return {
                    "phase": "verify",
                    "verification": verification,
                    "verification_history": [verification],
                    "verification_snapshot_id": snapshot_id,
                    "review": ReviewResult(accepted=False, blockers=blockers, confidence=1.0),
                    "batch_conflict": "; ".join(blockers),
                    "previous_attempt_summary": "; ".join(blockers),
                    "failure_reason": "; ".join(blockers),
                    "failure_stage": "verify",
                    "events": events,
                }
            review = ReviewResult(accepted=True, confidence=1.0)
        if not verification.passed and verification.status not in {"infra_error", "inconclusive"}:
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.ATTEMPT_FAILED,
                    discriminator=f"verification:{batch}",
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
                        "iteration": batch,
                        "status": verification.status,
                    },
                )
            )
        events.append(
            _event(
                runtime,
                state,
                EventType.VERIFICATION_COMPLETED,
                discriminator=str(batch),
                payload={"verification": verification.model_dump(), "snapshot_id": snapshot_id},
            )
        )
        failure_reason = None
        status_update: dict[str, Any] = {}
        if (
            verification.status == "incomplete"
            and verification.incomplete_cause == "budget"
        ):
            reason = verification.incomplete_reason or "run-level tool-call budget exhausted"
            status_update["status"] = "failed"
            status_update["failure_class"] = "budget"
            status_update["failure_stage"] = "verify"
            status_update["failure_classes"] = ["budget"]
            status_update["budget_failures"] = [reason]
            failure_reason = reason
        elif (
            verification.status == "incomplete"
            and verification.incomplete_cause == "execution"
        ):
            reason = verification.incomplete_reason or "verification command could not start"
            status_update["status"] = "failed"
            status_update["failure_class"] = "infrastructure"
            status_update["failure_stage"] = "verify"
            failure_reason = reason
        elif verification.status in {"infra_error", "inconclusive"}:
            status_update["status"] = "failed"
            status_update["failure_class"] = "infrastructure"
            status_update["failure_stage"] = "verify"
            failure_reason = verification.incomplete_reason or verification.status
        elif verification.status == "unavailable":
            status_update["status"] = "failed"
            status_update["failure_class"] = "repair"
            status_update["failure_stage"] = "verify"
            failure_reason = verification.incomplete_reason
        elif not verification.passed:
            failure_reason = verification.incomplete_reason or "verification failed"
            status_update["failure_stage"] = "verify"
        return {
            "phase": "verify",
            "verification": verification,
            "verification_history": [verification],
            "verification_snapshot_id": snapshot_id,
            "review": review,
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

    def after_verify(state: EvoCIState) -> str:
        verification = state.get("verification")
        if verification and verification.status in {"infra_error", "inconclusive"}:
            return "failed"
        if state.get("batch_conflict"):
            return "rollback"
        if verification and verification.passed:
            return "finalize"
        return "rollback"

    async def rollback(state: EvoCIState) -> dict[str, Any]:
        snapshot = state.get("batch_snapshot_path")
        events = [
            _event(
                runtime,
                state,
                EventType.TOOL_CALL,
                agent_id="harness",
                discriminator=f"rollback:{state.get('supervisor_batch', 0)}",
                payload={
                    "call_id": f"rollback:{state.get('supervisor_batch', 0)}",
                    "tool_name": "rollback",
                    "budget_scope": "cleanup",
                },
            )
        ]
        failed_candidates = list(state.get("failed_candidates", []))
        result = _latest_result(state)
        if result is not None and result.patch_artifact_ref:
            candidate_id = (
                f"{state['run_id']}:{result.task_id}:{state.get('supervisor_batch', 0)}"
            )
            if all(item.candidate_id != candidate_id for item in failed_candidates):
                failed_candidates.append(
                    FailedCandidateRef(
                        candidate_id=candidate_id,
                        task_id=result.task_id,
                        batch=int(state.get("supervisor_batch", 0)),
                        baseline_snapshot_id=result.base_revision or "",
                        artifact_ref=result.patch_artifact_ref,
                        changed_files=result.changed_files,
                    )
                )
        original_reason = state.get("failure_reason") or state.get("batch_conflict")
        original_stage = state.get("failure_stage") or state.get("phase") or "verify"
        verification_update: dict[str, Any] = {}
        if state.get("batch_conflict"):
            # A passing verification rejected by governance belongs to the candidate
            # being rolled back. Keep verification_history, but do not expose this
            # stale pass as current truth to the next Supervisor invocation.
            verification_update = {
                "verification": None,
                "verification_snapshot_id": None,
                "review": None,
            }
        apply_failed = state.get("phase") == "apply_candidate" and not state.get("verification")
        try:
            if apply_failed and state.get("attempt_written") is not None:
                restore_attempt_writes(
                    Path(state["workspace_path"]),
                    state.get("attempt_baseline", {}),
                    state.get("attempt_written") or {},
                )
            elif snapshot:
                restore_batch_snapshot(Path(snapshot), Path(state["workspace_path"]))
            elif state.get("attempt_written") is not None:
                restore_attempt_writes(
                    Path(state["workspace_path"]),
                    state.get("attempt_baseline", {}),
                    state.get("attempt_written") or {},
                )
        except Exception as exc:
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.TOOL_RESULT,
                    agent_id="harness",
                    discriminator=f"rollback:{state.get('supervisor_batch', 0)}",
                    payload={
                        "call_id": f"rollback:{state.get('supervisor_batch', 0)}",
                        "tool_name": "rollback",
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            return {
                "phase": "rollback",
                "status": "failed",
                "failure_class": "infrastructure",
                "failure_stage": "rollback",
                "failure_reason": f"rollback failed: {exc}",
                "events": events,
            }
        events.append(
            _event(
                runtime,
                state,
                EventType.TOOL_RESULT,
                agent_id="harness",
                discriminator=f"rollback:{state.get('supervisor_batch', 0)}",
                payload={
                    "call_id": f"rollback:{state.get('supervisor_batch', 0)}",
                    "tool_name": "rollback",
                    "success": True,
                },
            )
        )
        return {
            "phase": "rollback",
            "failed_candidates": failed_candidates,
            "integrated_snapshot_id": workspace_snapshot_id(Path(state["workspace_path"])),
            **verification_update,
            "failure_reason": original_reason,
            "failure_stage": original_stage,
            "batch_conflict": None,
            "events": events,
        }

    def after_rollback(state: EvoCIState) -> str:
        if state.get("status") == "failed":
            return "failed"
        classes = state.get("failure_classes") or []
        if classes and classes[-1] in {"model", "infrastructure"}:
            return "failed"
        if state.get("supervisor_batch", 0) < config.max_supervisor_batches:
            return "supervise"
        return "failed"

    async def finalize(state: EvoCIState) -> dict[str, Any]:
        decision = _coerce_decision(state.get("decision"))
        if decision is not None and decision.action == "stop" and not _formal_success(state):
            return await persist_run_outcome(
                runtime,
                state,
                success=False,
                failure_reason=(
                    decision.stop_reason or "supervisor stopped without verified success"
                ),
                failure_class="repair",
            )
        if runtime.defer_success_learning:
            return {
                "phase": "finalize",
                "status": "success",
                "failure_reason": None,
                "learning_deferred": True,
            }
        return await persist_run_outcome(runtime, state, success=True, failure_reason=None)

    async def failed(state: EvoCIState) -> dict[str, Any]:
        reason = (
            state.get("failure_reason")
            or state.get("batch_conflict")
            or "orchestration budget exhausted"
        )
        decision = _coerce_decision(state.get("decision"))
        if decision is not None and decision.action == "stop":
            reason = decision.stop_reason or reason
        failure_class = resolve_failure_class(state, success=False)
        if runtime.defer_success_learning and failure_class == "infrastructure":
            return {
                "phase": "failed",
                "status": "failed",
                "failure_reason": reason,
                "failure_class": failure_class,
                "failure_stage": state.get("failure_stage") or state.get("phase"),
                "learning_deferred": True,
            }
        return await persist_run_outcome(
            runtime,
            state,
            success=False,
            failure_reason=reason,
            failure_class=failure_class,
        )

    builder = StateGraph(EvoCIState)
    builder.add_node("bootstrap", bootstrap)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("supervise", supervise)
    builder.add_node("prepare_task", prepare_task)
    builder.add_node("worker", worker)
    builder.add_node("approval", approval)
    builder.add_node("apply_candidate", apply_candidate)
    builder.add_node("verify", verify)
    builder.add_node("rollback", rollback)
    builder.add_node("finalize", finalize)
    builder.add_node("failed", failed)

    builder.add_edge(START, "bootstrap")
    builder.add_edge("bootstrap", "retrieve_context")
    builder.add_edge("retrieve_context", "supervise")
    builder.add_conditional_edges(
        "supervise",
        after_supervise,
        {"prepare_task": "prepare_task", "finalize": "finalize", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "prepare_task",
        after_prepare,
        {"worker": "worker", "supervise": "supervise", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "worker",
        after_worker,
        {
            "supervise": "supervise",
            "approval": "approval",
            "failed": "failed",
        },
    )
    builder.add_conditional_edges(
        "approval",
        after_approval,
        {"apply_candidate": "apply_candidate", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "apply_candidate",
        after_apply,
        {"verify": "verify", "rollback": "rollback", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "verify",
        after_verify,
        {"finalize": "finalize", "rollback": "rollback", "failed": "failed"},
    )
    builder.add_conditional_edges(
        "rollback",
        after_rollback,
        {"supervise": "supervise", "failed": "failed"},
    )
    builder.add_edge("finalize", END)
    builder.add_edge("failed", END)
    return cast(CompiledStateGraph[Any, Any, Any, Any], builder.compile(checkpointer=checkpointer))
