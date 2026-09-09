"""Command-line interface for runs, memory, skills, benchmarks, and demos."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Annotated, Any, Literal, cast
from uuid import uuid4

import typer
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from rich.console import Console
from rich.table import Table

from evoci.agents.base import AgentContext, AgentSuite
from evoci.agents.model_agents import (
    ModelCoordinator,
    ModelDiagnoser,
    ModelFixer,
    ModelInvestigator,
    ModelReviewer,
)
from evoci.benchmark.adapters import CIRepairBenchAdapter
from evoci.benchmark.execution import (
    FailedCommandReplayVerifier,
    collect_metrics,
    prepare_workspace,
)
from evoci.benchmark.models import (
    BenchmarkManifestEntry,
    BenchmarkResult,
    BenchmarkVariant,
    RunMetrics,
)
from evoci.benchmark.runner import BenchmarkRunner
from evoci.benchmark.variants import VariantFeatures, variant_features
from evoci.capability.curator import CuratorPipeline, ModelCurator
from evoci.capability.materializer import (
    CapabilityMaterializer,
    resolve_capability_runtime_root,
)
from evoci.capability.miner import ExperienceMiner
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.capability.validator import CandidateValidator
from evoci.config import EvoCIConfig
from evoci.demo import run_repair_demo
from evoci.domain.models import (
    CIFailure,
    Diagnosis,
    Hypothesis,
    RepoSpec,
    ReviewResult,
    VerificationCommandResult,
    VerificationResult,
)
from evoci.graph.builder import GraphRuntime, build_graph, persist_run_outcome
from evoci.graph.routing import contains_review_bypass, contains_workspace_review_bypass
from evoci.memory.consolidation import ModelMemoryConsolidator
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import SQLiteMemoryStore
from evoci.model.gateway import ToolLoopGateway
from evoci.model.openai_compatible import OpenAICompatibleGateway
from evoci.runtime.budget import RepairBudgetExhausted, RunBudgetManager
from evoci.runtime.checkpoints import AsyncSQLiteCheckpointHandle, create_async_sqlite_checkpointer
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.events import EventType
from evoci.runtime.run_store import SQLiteRunStore
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.filesystem import FileTools
from evoci.tools.patch import (
    apply_edit,
    precheck_edits,
    restore_edit_baseline,
    snapshot_edit_baseline,
)
from evoci.tools.policy import PolicyViolation
from evoci.verification.service import VerificationService

app = typer.Typer(help="Durable, self-improving multi-agent CI recovery")
runs_app = typer.Typer(help="Inspect durable runs")
memory_app = typer.Typer(help="Search long-term memory")
skills_app = typer.Typer(help="Inspect and curate capability packages")
app.add_typer(runs_app, name="runs")
app.add_typer(memory_app, name="memory")
app.add_typer(skills_app, name="skills")
console = Console()


@dataclass(slots=True)
class LiveResources:
    runtime: GraphRuntime
    checkpoint: AsyncSQLiteCheckpointHandle
    event_store: SQLiteEventStore
    run_store: SQLiteRunStore
    memory_store: SQLiteMemoryStore
    registry: CapabilityRegistry
    gateway: ToolLoopGateway
    recorder: TrajectoryRecorder

    async def close(self) -> None:
        await self.checkpoint.close()
        self.registry.close()
        self.memory_store.close()
        self.run_store.close()
        self.event_store.close()


async def _live_resources(
    config: EvoCIConfig,
    features: VariantFeatures | None = None,
    *,
    defer_success_learning: bool = False,
) -> LiveResources:
    enabled = features or variant_features("evo")
    config.ensure_directories()
    strong_gateway = OpenAICompatibleGateway(
        config, model_name=config.strong_model_name or config.model_name
    )
    fast_gateway = OpenAICompatibleGateway(
        config, model_name=config.fast_model_name or config.model_name
    )
    aux_gateway = OpenAICompatibleGateway(
        config,
        model_name=config.aux_model_name or config.fast_model_name or config.model_name,
    )
    event_store = SQLiteEventStore(config.state_dir / "events.sqlite")
    recorder = TrajectoryRecorder(event_store)
    budget_manager = RunBudgetManager(
        max_model_calls=config.max_run_model_calls,
        max_tool_calls=config.max_run_tool_calls,
        recorder=recorder,
    )
    run_store = SQLiteRunStore(config.state_dir / "runs.sqlite")
    memory_store = SQLiteMemoryStore(config.state_dir / "memory.sqlite")
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    validator = CandidateValidator(registry)
    curator_pipeline = None
    if enabled.curator:
        curator_pipeline = CuratorPipeline(
            registry,
            validator,
            model_curator=ModelCurator(registry, aux_gateway),
            max_exposures_without_use=config.trial_max_exposures_without_use,
        )
    agent_registry = registry if enabled.capabilities else None
    checkpoint = await create_async_sqlite_checkpointer(config.state_dir / "checkpoints.sqlite")
    runtime = GraphRuntime(
        config=config,
        agents=AgentSuite(
            coordinator=ModelCoordinator(fast_gateway, recorder, budget_manager),
            investigator=ModelInvestigator(
                fast_gateway,
                recorder,
                capability_registry=agent_registry,
                max_iterations=config.max_leaf_iterations,
                max_tool_calls=config.max_leaf_tool_calls,
                timeout=config.command_timeout_seconds,
                max_chars=config.output_limit_chars,
                budget_manager=budget_manager,
            ),
            diagnoser=ModelDiagnoser(strong_gateway, recorder, budget_manager),
            fixer=ModelFixer(
                strong_gateway,
                recorder,
                capability_registry=agent_registry,
                max_iterations=config.max_leaf_iterations,
                max_tool_calls=config.max_leaf_tool_calls,
                timeout=config.command_timeout_seconds,
                max_chars=config.output_limit_chars,
                budget_manager=budget_manager,
            ),
            reviewer=ModelReviewer(
                strong_gateway,
                recorder,
                capability_registry=agent_registry,
                max_iterations=config.max_leaf_iterations,
                max_tool_calls=config.max_leaf_tool_calls,
                timeout=config.command_timeout_seconds,
                max_chars=config.output_limit_chars,
                budget_manager=budget_manager,
            ),
        ),
        event_store=event_store,
        run_store=run_store,
        memory_store=memory_store if enabled.long_term_memory else None,
        memory_retriever=(
            MemoryRetriever(memory_store, context_limit_chars=config.memory_context_limit_chars)
            if enabled.long_term_memory
            else None
        ),
        memory_consolidator=(
            ModelMemoryConsolidator(aux_gateway) if enabled.long_term_memory else None
        ),
        capability_registry=registry if enabled.capabilities else None,
        capability_retriever=(
            CapabilityRetriever(
                registry,
                top_k=config.capability_retrieval_top_k,
                trial_slots=config.trial_retrieval_slots,
            )
            if enabled.capabilities
            else None
        ),
        capability_materializer=(
            CapabilityMaterializer(
                registry,
                config.runtime_dir,
                allow_relocate=not config.runtime_dir_explicit,
            )
            if enabled.capabilities
            else None
        ),
        experience_miner=ExperienceMiner(aux_gateway) if enabled.capabilities else None,
        candidate_validator=validator if enabled.capabilities else None,
        recorder=recorder,
        curator_pipeline=curator_pipeline,
        budget_manager=budget_manager,
        defer_success_learning=defer_success_learning,
    )
    return LiveResources(
        runtime,
        checkpoint,
        event_store,
        run_store,
        memory_store,
        registry,
        strong_gateway,
        recorder,
    )


def _load_task(path: Path, task_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    workspace = Path(str(payload["workspace_path"])).resolve()
    if not workspace.is_dir():
        raise typer.BadParameter(f"workspace does not exist: {workspace}")
    return {
        # A task file describes input, not an existing execution. Reusing a persisted
        # ID here would merge checkpoints, events, and budgets from an earlier run.
        "run_id": f"run-{uuid4().hex[:12]}",
        "task_id": task_id,
        "repo": RepoSpec.model_validate(payload["repo"]),
        "ci_failure": CIFailure.model_validate(payload["ci_failure"]),
        "workspace_path": str(workspace),
    }


def _print_run_summary(result: dict[str, Any], *, task_id: str) -> None:
    """Show the evidence, patch, verification, and review behind the terminal status."""

    status = str(result.get("status", "unknown"))
    console.print(f"\nEvoCI run {task_id}: [bold]{status}[/bold]")
    diagnosis = result.get("diagnosis")
    if diagnosis is not None and hasattr(diagnosis, "primary"):
        console.print(f"Root cause         {diagnosis.primary.root_cause}")
    evidence = result.get("evidence", [])
    console.print(f"Evidence           {len(evidence)} item(s)")
    patch = result.get("fixer_output")
    if patch is not None and hasattr(patch, "proposal"):
        files = patch.proposal.changed_files
        console.print(f"Patch              {patch.proposal.summary}")
        console.print(f"Changed files      {', '.join(files) if files else 'none'}")
    verification = result.get("verification")
    if verification is not None:
        checks = [
            f"{'PASS' if command.exit_code == 0 and not command.timed_out else 'FAIL'} "
            f"{' '.join(command.command)}"
            for command in verification.commands
        ]
        console.print(f"Verification       {'; '.join(checks) if checks else 'none'}")
    review = result.get("review")
    if review is not None:
        blockers = "; ".join(review.blockers)
        console.print(
            f"Review             {'PASS' if review.accepted else 'FAIL'}"
            + (f" ({blockers})" if blockers else "")
        )
    failure_reason = result.get("failure_reason")
    if failure_reason:
        console.print(f"Failure reason     {failure_reason}")
    learning_errors = result.get("learning_errors", [])
    if learning_errors:
        stages = ", ".join(str(item.get("stage", "unknown")) for item in learning_errors)
        console.print(f"Post-run warnings   {stages}")
    console.print(f"Full report         evoci runs report {task_id}")


def _print_run_report(run_id: str, events: list[Any]) -> None:
    """Render the durable event trail as a concise, user-facing investigation report."""

    console.print(f"\nEvoCI investigation report: [bold]{run_id}[/bold]")
    plan_event = next(
        (
            event
            for event in events
            if event.type == EventType.AGENT_COMPLETED and event.agent_id == "coordinator"
        ),
        None,
    )
    if plan_event is not None:
        plan = Table("Check", "Role", "Objective")
        for task in plan_event.payload.get("tasks", []):
            if isinstance(task, dict):
                plan.add_row(
                    str(task.get("task_id", "unknown")),
                    str(task.get("role", "unknown")),
                    str(task.get("objective", "not provided")),
                )
            else:
                plan.add_row(str(task), "unknown", "Legacy task event")
        console.print(plan)

    failures: dict[str, tuple[str, str, int]] = {}
    for event in events:
        if event.type != EventType.TOOL_RESULT or event.payload.get("success", True):
            continue
        result = event.payload.get("result")
        command = ""
        if isinstance(result, dict) and isinstance(result.get("argv"), list):
            command = " ".join(str(part) for part in result["argv"])
        command = command or str(event.payload.get("tool_name", "command"))
        reason = _error_summary(str(event.payload.get("error") or "command failed"))
        key = f"{command}:{event.payload.get('exit_code')}:{reason}"
        if key in failures:
            old_command, old_reason, count = failures[key]
            failures[key] = old_command, old_reason, count + 1
        else:
            failures[key] = command, reason, 1
    if failures:
        failed_commands = Table("Failed command", "Reason", "Confirmations")
        for command, reason, count in failures.values():
            failed_commands.add_row(command, reason, str(count))
        console.print(failed_commands)

    evidence_table = Table("Check", "Kind", "Location", "Evidence", "Command")
    evidence_count = 0
    for event in events:
        if event.type != EventType.EVIDENCE_CREATED:
            continue
        item = event.payload.get("evidence")
        if not isinstance(item, dict):
            continue
        evidence_count += 1
        evidence_table.add_row(
            str(event.agent_id or "unknown").removeprefix("investigator:"),
            str(item.get("kind", "unknown")),
            str(item.get("file_path") or "-"),
            str(item.get("claim", "not provided")),
            str(item.get("command") or "-"),
        )
    if evidence_count:
        console.print(evidence_table)


def _benchmark_runtime_config(
    base: EvoCIConfig,
    *,
    output_dir: Path,
    variant: BenchmarkVariant,
    session_id: str,
    task_index: int | None,
) -> EvoCIConfig:
    """Keep benchmark lifecycle stores explicit and isolated by evaluation mode."""

    suffix = f"{variant}-{session_id}"
    if task_index is not None:
        suffix = f"{suffix}-{task_index}"
    state_dir = output_dir / "state" / suffix
    return base.model_copy(
        update={
            "state_dir": state_dir,
            "capability_dir": state_dir / "skills",
            "runtime_dir": output_dir / "runtime" / suffix,
            "workspace_dir": output_dir / "worktrees",
            "repo_cache_dir": output_dir / "repo-cache",
        }
    )


async def _drive_graph(
    resources: LiveResources,
    initial: dict[str, object] | None,
    run_id: str,
) -> dict[str, Any]:
    graph = build_graph(resources.runtime, checkpointer=resources.checkpoint.saver)
    invocation_config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    progress = asyncio.create_task(_stream_run_progress(resources, run_id))
    try:
        result = cast(dict[str, Any], await graph.ainvoke(cast(Any, initial), invocation_config))
        while "__interrupt__" in result:
            interrupt_info = result["__interrupt__"][0].value
            console.print_json(data=interrupt_info)
            approved = typer.confirm("Approve this change?", default=False)
            result = cast(
                dict[str, Any],
                await graph.ainvoke(cast(Any, Command(resume=approved)), invocation_config),
            )
        return result
    finally:
        progress.cancel()
        with suppress(asyncio.CancelledError):
            await progress


def _progress_line(
    event_type: EventType,
    agent_id: str | None,
    payload: dict[str, Any],
    announced: set[str] | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> str | None:
    agent = agent_id or "runtime"
    announced = announced if announced is not None else set()
    if event_type == EventType.MODEL_CALL:
        role = agent.split(":", 1)[0]
        if role in announced:
            return None
        announced.add(role)
        messages = {
            "coordinator": "正在制定调查计划...",
            "investigator": "正在检查代码、测试和失败现场...",
            "diagnoser": "正在综合证据并判断根因...",
            "fixer": "正在准备最小修复...",
            "reviewer": "正在审查修复和验证结果...",
            "memory-consolidator": "正在整理本次运行的可复用经验...",
            "experience-miner": "正在判断是否产生新的可复用技能...",
        }
        return messages.get(role, "正在执行模型步骤...")
    if event_type == EventType.TOOL_RESULT and not payload.get("success", True):
        return _failed_command_progress(payload, announced)
    if event_type == EventType.AGENT_COMPLETED:
        if agent == "coordinator":
            tasks = payload.get("tasks", [])
            return _investigation_plan_progress(tasks)
        if agent.startswith("investigator:"):
            count = payload.get("evidence_count", 0)
            return _investigation_evidence_progress(
                agent.split(":", 1)[1], count, evidence or []
            )
    if event_type == EventType.DIAGNOSIS_CREATED:
        primary = payload.get("diagnosis", {}).get("primary", {})
        root_cause = str(primary.get("root_cause", "根因已确定"))
        return f"根因判断: {root_cause}"
    if event_type == EventType.PATCH_CREATED:
        files = payload.get("fixer_output", {}).get("proposal", {}).get("changed_files", [])
        summary = (
            payload.get("fixer_output", {}).get("proposal", {}).get("summary", "")
            or "未提供说明"
        )
        return f"修复方案: {summary} (文件: {', '.join(files) or '无'})"
    if event_type == EventType.VERIFICATION_STARTED:
        return "正在运行修复后的验证命令..."
    if event_type == EventType.VERIFICATION_COMPLETED:
        passed = payload.get("verification", {}).get("passed", False)
        return f"验证结果: {'通过' if passed else '失败'}"
    if event_type == EventType.ATTEMPT_FAILED:
        return f"本次修复尝试未通过: {payload.get('reason', '原因未知')}"
    if event_type == EventType.RUN_COMPLETED:
        return "运行完成。"
    if event_type == EventType.RUN_FAILED:
        return f"运行失败: {payload.get('reason', '原因未知')}"
    return None


def _investigation_plan_progress(tasks: list[Any]) -> str:
    lines = [f"调查计划已生成: {len(tasks)} 个独立检查。"]
    for index, task in enumerate(tasks, start=1):
        if isinstance(task, dict):
            role = str(task.get("role", "调查"))
            objective = _compact_progress_text(str(task.get("objective", "未说明目标")), 180)
            lines.append(f"  {index}. [{role}] {objective}")
        else:
            lines.append(f"  {index}. {task}")
    return "\n".join(lines)


def _failed_command_progress(payload: dict[str, Any], announced: set[str]) -> str | None:
    result = payload.get("result")
    command: list[str] = []
    if isinstance(result, dict) and isinstance(result.get("argv"), list):
        command = [str(part) for part in result["argv"]]
    exit_code = payload.get("exit_code")
    error = _error_summary(str(payload.get("error") or "命令未成功"))
    command_text = " ".join(command) or str(payload.get("tool_name", "命令"))
    fingerprint = f"failed-command:{command_text}:{exit_code}:{error}"
    if fingerprint in announced:
        return None
    announced.add(fingerprint)
    exit_detail = f", 退出码 {exit_code}" if exit_code is not None else ""
    return f"复现命令失败: `{command_text}`{exit_detail}。{error}"


def _error_summary(error: str) -> str:
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    for line in lines:
        if "Error" in line or "Exception" in line:
            return _compact_progress_text(line, 180)
    for line in reversed(lines):
        if "FAILED" in line:
            return _compact_progress_text(line, 180)
    return _compact_progress_text(lines[-1] if lines else error, 180)


def _investigation_evidence_progress(
    task_id: str, count: int, evidence: list[dict[str, Any]]
) -> str:
    lines = [f"调查完成 ({task_id}): 收集到 {count} 条证据。"]
    ranked = sorted(
        evidence,
        key=lambda item: (
            {"test_result": 0, "runtime": 1, "ci_log": 2, "source_code": 3}.get(
                str(item.get("kind")), 4
            ),
            -float(item.get("confidence", 0)),
        ),
    )
    for item in ranked[:2]:
        claim = _compact_progress_text(str(item.get("claim", "未提供摘要")), 200)
        location = str(item.get("file_path") or item.get("kind") or "证据")
        lines.append(f"  - {location}: {claim}")
    remaining = max(0, count - len(ranked[:2]))
    if remaining:
        lines.append(f"  - 另有 {remaining} 条支持证据已记录。")
    return "\n".join(lines)


def _compact_progress_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


async def _stream_run_progress(resources: LiveResources, run_id: str) -> None:
    seen = {event.event_id for event in resources.event_store.list(run_id)}
    announced: set[str] = set()
    evidence_by_agent: dict[str, list[dict[str, Any]]] = {}
    console.print(f"Run {run_id} started")
    while True:
        await asyncio.sleep(0.25)
        for event in resources.event_store.list(run_id):
            if event.event_id in seen:
                continue
            seen.add(event.event_id)
            if event.type == EventType.EVIDENCE_CREATED and event.agent_id:
                item = event.payload.get("evidence")
                if isinstance(item, dict):
                    evidence_by_agent.setdefault(event.agent_id, []).append(item)
            line = _progress_line(
                event.type,
                event.agent_id,
                event.payload,
                announced,
                evidence_by_agent.get(event.agent_id or "", []),
            )
            if line:
                console.print(line)


async def _drive_graph_automated(
    resources: LiveResources,
    initial: dict[str, object],
    run_id: str,
) -> dict[str, Any]:
    """Drive benchmark interrupts deterministically without interactive input."""

    graph = build_graph(resources.runtime, checkpointer=resources.checkpoint.saver)
    invocation_config: RunnableConfig = {"configurable": {"thread_id": run_id}}
    result = cast(dict[str, Any], await graph.ainvoke(cast(Any, initial), invocation_config))
    while "__interrupt__" in result:
        result = cast(
            dict[str, Any],
            await graph.ainvoke(cast(Any, Command(resume=True)), invocation_config),
        )
    return result


async def _drive_single(
    resources: LiveResources,
    initial: dict[str, object],
) -> dict[str, Any]:
    """Run one tool-using coding leaf with no graph, memory, or capability retrieval."""

    run_id = str(initial["run_id"])
    repo = cast(RepoSpec, initial["repo"])
    failure = cast(CIFailure, initial["ci_failure"])
    workspace = Path(str(initial["workspace_path"]))
    resources.recorder.emit(
        run_id=run_id,
        event_type=EventType.RUN_STARTED,
        event_key="single",
        payload={"variant": "single"},
    )
    diagnosis = Diagnosis(
        primary=Hypothesis(
            root_cause="Determine the root cause directly with repository and test tools.",
            evidence_ids=[],
            confidence=0,
            affected_files=[],
            proposed_action="Inspect, repair, and verify the failing task.",
        ),
        needs_more_evidence=False,
    )
    files = FileTools(
        workspace,
        writable=True,
        max_chars=resources.runtime.config.output_limit_chars,
    )
    assert resources.runtime.budget_manager is not None
    budget = resources.runtime.budget_manager.for_run(run_id)
    previous_verification: VerificationResult | None = None
    previous_blockers: tuple[str, ...] = ()
    previous_summary: str | None = None
    verification_history: list[VerificationResult] = []
    output = None
    verification = None
    review = None
    status: Literal["success", "failed"] = "failed"
    failure_reason: str | None = "single-agent repair budget exhausted"
    attempt = 0

    verifier = VerificationService(
        timeout=resources.runtime.config.command_timeout_seconds,
        max_chars=resources.runtime.config.output_limit_chars,
    )

    for attempt in range(1, resources.runtime.config.max_repair_attempts + 1):
        try:
            output = await resources.runtime.agents.fixer.propose(
                context=AgentContext(
                    run_id=run_id,
                    repo=repo,
                    failure=failure,
                    workspace_path=str(workspace),
                    invocation_id=f"repair:{attempt}",
                    previous_review_blockers=previous_blockers,
                    previous_attempt_summary=previous_summary,
                ),
                diagnosis=diagnosis,
                evidence=[],
                previous_verification=previous_verification,
            )
        except RepairBudgetExhausted as exc:
            failure_reason = str(exc)
            break

        blockers = contains_review_bypass(output)
        if blockers:
            previous_blockers = tuple(blockers)
            previous_summary = "; ".join(blockers)
            failure_reason = previous_summary
            continue

        baseline = snapshot_edit_baseline(workspace, output.edits)
        apply_failed = False
        try:
            precheck_edits(workspace, output.edits)
            if output.edits:
                budget.ensure_tool_calls(len(output.edits))
            for index, edit in enumerate(output.edits):
                call_id = f"single-apply:{attempt}:{index}"
                try:
                    budget.consume_tool_call()
                except RepairBudgetExhausted as exc:
                    failure_reason = str(exc)
                    apply_failed = True
                    break
                resources.recorder.emit(
                    run_id=run_id,
                    event_type=EventType.TOOL_CALL,
                    agent_id="single",
                    invocation_id=f"repair:{attempt}",
                    event_key=call_id,
                    payload={
                        "call_id": call_id,
                        "tool_name": "apply_patch",
                        "arguments": {"path": edit.path, "delete": edit.delete},
                    },
                )
                started = monotonic()
                try:
                    created, modified = apply_edit(workspace, files, edit)
                except Exception as exc:
                    failure_reason = f"patch apply failed: {exc}"
                    apply_failed = True
                    resources.recorder.emit(
                        run_id=run_id,
                        event_type=EventType.TOOL_RESULT,
                        agent_id="single",
                        invocation_id=f"repair:{attempt}",
                        event_key=call_id,
                        payload={
                            "call_id": call_id,
                            "tool_name": "apply_patch",
                            "success": False,
                            "error": failure_reason,
                            "duration_seconds": monotonic() - started,
                        },
                    )
                    break
                resources.recorder.emit(
                    run_id=run_id,
                    event_type=EventType.TOOL_RESULT,
                    agent_id="single",
                    invocation_id=f"repair:{attempt}",
                    event_key=call_id,
                    payload={
                        "call_id": call_id,
                        "tool_name": "apply_patch",
                        "success": True,
                        "created_files": created,
                        "modified_files": modified,
                        "duration_seconds": monotonic() - started,
                    },
                )
        except BaseException:
            restore_edit_baseline(workspace, baseline)
            raise
        if apply_failed:
            restore_edit_baseline(workspace, baseline)
            previous_summary = failure_reason
            continue

        final_blockers = contains_workspace_review_bypass(workspace)
        if final_blockers:
            restore_edit_baseline(workspace, baseline)
            previous_blockers = tuple(final_blockers)
            previous_summary = "; ".join(final_blockers)
            failure_reason = previous_summary
            continue

        current_attempt = attempt

        async def on_command_start(
            index: int, command: list[str], source: str, *, attempt: int = current_attempt
        ) -> None:
            del source
            resources.recorder.emit(
                run_id=run_id,
                event_type=EventType.TOOL_CALL,
                agent_id="single",
                invocation_id=f"repair:{attempt}",
                event_key=f"single-verify:{attempt}:{index}",
                payload={
                    "call_id": f"single-verify:{attempt}:{index}",
                    "tool_name": "run_test",
                    "arguments": {"argv": command},
                },
            )

        async def on_command_done(
            index: int,
            result: VerificationCommandResult,
            *,
            attempt: int = current_attempt,
        ) -> None:
            passed = result.executed and result.exit_code == 0 and not result.timed_out
            resources.recorder.emit(
                run_id=run_id,
                event_type=EventType.TOOL_RESULT,
                agent_id="single",
                invocation_id=f"repair:{attempt}",
                event_key=f"single-verify:{attempt}:{index}",
                payload={
                    "call_id": f"single-verify:{attempt}:{index}",
                    "tool_name": "run_test",
                    "success": passed,
                    "exit_code": result.exit_code,
                    "error": result.stderr if not passed else None,
                    "executed": result.executed,
                    "source": result.source,
                },
            )

        verification = await verifier.run(
            workspace=workspace,
            mandatory=failure.failed_commands,
            supplementary=output.proposal.verification_plan,
            budget=budget,
            on_command_start=on_command_start,
            on_command_done=on_command_done,
        )
        verification_history.append(verification)
        previous_verification = verification
        if verification.passed:
            review = ReviewResult(accepted=False, performed=False, confidence=0.0)
            status = "success"
            failure_reason = None
            break
        restore_edit_baseline(workspace, baseline)
        previous_summary = verification.incomplete_reason or next(
            (
                result.stderr or f"exit code {result.exit_code}"
                for result in verification.commands
                if result.executed and (result.exit_code != 0 or result.timed_out)
            ),
            failure_reason,
        )
        failure_reason = previous_summary

    if review is None:
        review = ReviewResult(
            accepted=False,
            performed=False,
            blockers=[failure_reason or "single-agent repair failed"],
            confidence=1.0,
        )
    resources.recorder.emit(
        run_id=run_id,
        event_type=EventType.RUN_COMPLETED if status == "success" else EventType.RUN_FAILED,
        event_key="terminal",
        payload={"status": status},
    )
    return {
        "status": status,
        "failure_reason": failure_reason,
        "repair_attempt": attempt,
        "investigation_round": 0,
        "evidence": [],
        "fixer_output": output,
        "verification": verification,
        "verification_history": verification_history,
        "review": review,
    }


@app.command()
def doctor(
    live: Annotated[bool, typer.Option("--live", help="Run live model capability probes")] = False,
) -> None:
    """Check local runtime prerequisites, optionally including live model calls."""
    config = EvoCIConfig.from_env()
    fts_connection = sqlite3.connect(":memory:")
    try:
        fts_connection.execute("CREATE VIRTUAL TABLE check_fts USING fts5(content)")
        fts5 = True
    except sqlite3.OperationalError:
        fts5 = False
    finally:
        fts_connection.close()
    table = Table(title="EvoCI doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Python", "3.12+")
    table.add_row("Git", shutil.which("git") or "missing")
    table.add_row("SQLite FTS5", "available" if fts5 else "missing")
    table.add_row("Model", config.model_name or "not configured (offline demo still works)")
    table.add_row("State", str(config.state_dir))
    console.print(table)
    _print_model_routing(config)
    if live:
        asyncio.run(_run_model_check(config))


def _print_model_routing(config: EvoCIConfig) -> None:
    table = Table(title="Model routing")
    table.add_column("Tier")
    table.add_column("Model")
    table.add_column("Agents")
    routes = (
        ("FAST", config.fast_model_name or config.model_name, config.fast_model_name is None),
        ("STRONG", config.strong_model_name or config.model_name, config.strong_model_name is None),
        (
            "AUX",
            config.aux_model_name or config.fast_model_name or config.model_name,
            config.aux_model_name is None,
        ),
    )
    agents = {
        "FAST": "Coordinator, Investigator",
        "STRONG": "Diagnoser, Fixer, Reviewer",
        "AUX": "Memory Consolidator, Experience Miner, Curator",
    }
    for tier, model, fallback in routes:
        table.add_row(
            tier,
            f"{model or 'not configured'}{' (fallback)' if fallback else ''}",
            agents[tier],
        )
    console.print(table)


async def _run_model_check(config: EvoCIConfig) -> None:
    console.print("\nModel connectivity")
    console.print(f"Endpoint           {config.model_base_url}")
    console.print(f"Model              {config.model_name or 'not configured'}")
    results: dict[str, tuple[bool, str | None]]
    try:
        gateway = OpenAICompatibleGateway(config, model_name=config.model_name)
        results = await gateway.smoke_test()
    except Exception as exc:
        failure_reason = str(exc)[:160]
        results = {
            name: (False, failure_reason)
            for name in ("basic_chat", "structured_output", "tool_calling")
        }
    labels = (
        ("basic_chat", "Basic chat"),
        ("structured_output", "Structured output"),
        ("tool_calling", "Tool calling"),
    )
    passed = True
    for key, label in labels:
        ok, reason = results[key]
        passed = passed and ok
        console.print(f"{label:<19}{'PASS' if ok else 'FAIL'}")
        if not ok and reason:
            console.print(f"  {reason}")
    console.print(f"\nEvoCI compatible    {'YES' if passed else 'NO'}")
    if not results["structured_output"][0]:
        console.print("Structured Output failure: structured Agent output is unavailable.")
    if not results["tool_calling"][0]:
        console.print(
            "Tool Calling failure: Investigator/Fixer/Reviewer leaf tool loops are unavailable."
        )


@app.command("model-check")
def model_check() -> None:
    """Run live Basic Chat, Structured Output, and Tool Calling probes."""
    asyncio.run(_run_model_check(EvoCIConfig.from_env()))


@app.command()
def demo() -> None:
    """Run an offline failure→fan-out→repair→verify→review demonstration."""
    result = asyncio.run(run_repair_demo(Path.cwd()))
    console.print(f"status: [bold]{result['status']}[/bold]")
    console.print(f"evidence: {len(result.get('evidence', []))}")
    console.print(f"investigation rounds: {result.get('investigation_round', 0)}")
    console.print(f"repair attempts: {result.get('repair_attempt', 0)}")
    console.print(f"episode: {result.get('episode_id')}")


@app.command("run")
def run_task(
    task_id: str,
    task_file: Annotated[Path | None, typer.Option(help="JSON task specification")] = None,
) -> None:
    """Start a live model-backed run."""
    config = EvoCIConfig.from_env()
    path = task_file or config.state_dir.parent / "tasks" / f"{task_id}.json"
    if not path.is_file():
        raise typer.BadParameter(f"task file not found: {path}")
    initial = _load_task(path, task_id)
    try:
        resolve_capability_runtime_root(
            config.runtime_dir,
            Path(str(initial["workspace_path"])),
            allow_relocate=not config.runtime_dir_explicit,
        )
    except PolicyViolation as exc:
        raise typer.BadParameter(str(exc)) from exc
    run_id = str(initial["run_id"])

    async def execute() -> dict[str, Any]:
        resources = await _live_resources(config)
        try:
            return await _drive_graph(resources, initial, run_id)
        finally:
            await resources.close()

    result = asyncio.run(execute())
    _print_run_summary(result, task_id=run_id)
    console.print_json(data={"run_id": run_id, "status": result.get("status")})


@app.command()
def resume(run_id: str) -> None:
    """Resume a checkpointed run after a crash or approval interrupt."""
    config = EvoCIConfig.from_env()

    async def execute() -> dict[str, Any]:
        resources = await _live_resources(config)
        try:
            record = resources.run_store.get(run_id)
            if record is None:
                raise typer.BadParameter(f"unknown run: {run_id}")
            return await _drive_graph(resources, None, run_id)
        finally:
            await resources.close()

    result = asyncio.run(execute())
    _print_run_summary(result, task_id=run_id)
    console.print_json(data={"run_id": run_id, "status": result.get("status")})


@runs_app.command("list")
def list_runs() -> None:
    config = EvoCIConfig.from_env()
    store = SQLiteRunStore(config.state_dir / "runs.sqlite")
    try:
        table = Table("Run", "Task", "Status", "Updated")
        for record in store.list():
            table.add_row(
                record.run_id, record.task_id, record.status, record.updated_at.isoformat()
            )
        console.print(table)
    finally:
        store.close()


@runs_app.command("inspect")
def inspect_run(run_id: str) -> None:
    config = EvoCIConfig.from_env()
    run_store = SQLiteRunStore(config.state_dir / "runs.sqlite")
    event_store = SQLiteEventStore(config.state_dir / "events.sqlite")
    try:
        record = run_store.get(run_id)
        if record is None:
            raise typer.BadParameter(f"unknown run: {run_id}")
        console.print_json(data=record.model_dump(mode="json"))
        console.print_json(
            data={"events": [event.model_dump(mode="json") for event in event_store.list(run_id)]}
        )
    finally:
        event_store.close()
        run_store.close()


@runs_app.command("report")
def report_run(run_id: str) -> None:
    """Show the investigation plan, failed commands, and all recorded evidence."""
    config = EvoCIConfig.from_env()
    run_store = SQLiteRunStore(config.state_dir / "runs.sqlite")
    event_store = SQLiteEventStore(config.state_dir / "events.sqlite")
    try:
        if run_store.get(run_id) is None:
            raise typer.BadParameter(f"unknown run: {run_id}")
        _print_run_report(run_id, event_store.list(run_id))
    finally:
        event_store.close()
        run_store.close()


@memory_app.command("search")
def search_memory(
    query: str,
    namespace: Annotated[str, typer.Option(help="Exact memory namespace")] = "global:ci",
) -> None:
    config = EvoCIConfig.from_env()
    store = SQLiteMemoryStore(config.state_dir / "memory.sqlite")
    try:
        hits = store.search_semantic(query, namespaces=[namespace], limit=10)
        console.print_json(data={"hits": [hit.model_dump() for hit in hits]})
    finally:
        store.close()


@skills_app.command("list")
def list_skills() -> None:
    config = EvoCIConfig.from_env()
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        table = Table("Skill", "Version", "Status", "Description")
        for record in registry.list():
            manifest = record.manifest
            table.add_row(
                manifest.skill_id,
                str(manifest.version),
                manifest.status,
                manifest.description,
            )
        console.print(table)
    finally:
        registry.close()


@skills_app.command("show")
def show_skill(skill_id: str, version: int | None = None) -> None:
    config = EvoCIConfig.from_env()
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        record = registry.get(skill_id, version)
        if record is None:
            raise typer.BadParameter(f"unknown skill: {skill_id}")
        console.print_json(data=record.manifest.model_dump(mode="json"))
        console.print((Path(record.package_path) / "SKILL.md").read_text())
    finally:
        registry.close()


@skills_app.command("curate")
def curate_skills() -> None:
    config = EvoCIConfig.from_env()
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        model_curator = None
        if config.model_name and config.model_api_key:
            gateway = OpenAICompatibleGateway(
                config,
                model_name=config.aux_model_name or config.fast_model_name or config.model_name,
            )
            model_curator = ModelCurator(registry, gateway)
        result = asyncio.run(
            CuratorPipeline(
                registry,
                CandidateValidator(registry),
                model_curator=model_curator,
                max_exposures_without_use=config.trial_max_exposures_without_use,
            ).run()
        )
        console.print_json(data=result.model_dump(mode="json"))
    finally:
        registry.close()


@app.command()
def benchmark(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    variant: Annotated[BenchmarkVariant, typer.Option()] = "multi",
    output_dir: Annotated[Path, typer.Option()] = Path("results"),
    continual: Annotated[bool, typer.Option(help="Preserve learned state across tasks")] = False,
) -> None:
    """Execute real benchmark workspaces through the selected runtime ablation."""
    entries = BenchmarkRunner.load_manifest(manifest)
    adapter = CIRepairBenchAdapter(dataset)
    session_id = uuid4().hex[:10]

    async def execute_benchmark() -> list[BenchmarkResult]:
        base_config = EvoCIConfig.from_env()
        features = variant_features(variant)
        shared: LiveResources | None = None
        task_index = 0
        if continual:
            shared_config = _benchmark_runtime_config(
                base_config,
                output_dir=output_dir,
                variant=variant,
                session_id=session_id,
                task_index=None,
            )
            shared = await _live_resources(
                shared_config, features, defer_success_learning=True
            )

        async def execute_entry(
            entry: BenchmarkManifestEntry, selected_variant: BenchmarkVariant
        ) -> RunMetrics:
            nonlocal task_index
            task_index += 1
            run_id = f"bench-{task_index}-{uuid4().hex[:8]}"
            active_config = _benchmark_runtime_config(
                base_config,
                output_dir=output_dir,
                variant=selected_variant,
                session_id=session_id,
                task_index=None if continual else task_index,
            )
            resources = shared or await _live_resources(
                active_config, features, defer_success_learning=True
            )
            try:
                verifier = FailedCommandReplayVerifier(
                    timeout=active_config.command_timeout_seconds,
                    max_chars=active_config.output_limit_chars,
                )
                preflight_workspace, preflight_task = prepare_workspace(
                    adapter,
                    entry,
                    worktrees_dir=active_config.workspace_dir,
                    repo_cache_dir=active_config.repo_cache_dir,
                    run_id=f"{run_id}-preflight",
                )
                benchmark_preflight = await verifier.preflight(preflight_task, preflight_workspace)
                workspace, prepared = prepare_workspace(
                    adapter,
                    entry,
                    worktrees_dir=active_config.workspace_dir,
                    repo_cache_dir=active_config.repo_cache_dir,
                    run_id=run_id,
                )
                view = prepared.agent_view
                failure = view.ci_failure
                initial: dict[str, object] = {
                    "run_id": run_id,
                    "task_id": view.task_id,
                    "repo": RepoSpec(
                        owner=view.repo_owner,
                        name=view.repo_name,
                        base_commit=view.sha_fail,
                    ),
                    "ci_failure": CIFailure(
                        summary=f"{entry.category}: {failure.log_text[:500]}",
                        log_excerpt=failure.log_text,
                        failed_commands=failure.candidate_failed_commands,
                        workflow_yaml=failure.workflow_yaml,
                        workflow_path=view.workflow_path,
                        task_family=entry.category,
                    ),
                    "workspace_path": str(workspace),
                }
                started = monotonic()
                if selected_variant == "single":
                    result = await _drive_single(resources, initial)
                else:
                    result = await _drive_graph_automated(resources, initial, run_id)
                elapsed = monotonic() - started
                truth = adapter.ground_truth(entry.task_id)
                registry_size = len(resources.registry.list()) if features.capabilities else 0
                active_count = (
                    len(resources.registry.list({"active"})) if features.capabilities else 0
                )
                metrics = await collect_metrics(
                    result=result,
                    recorder=resources.recorder,
                    run_id=run_id,
                    task=prepared,
                    workspace=workspace,
                    truth=truth,
                    wall_time=elapsed,
                    verifier=verifier,
                    benchmark_preflight=benchmark_preflight,
                    skill_registry_size=registry_size,
                    active_skill_count=active_count,
                )
                if result.get("learning_deferred") and result.get("status") == "success":
                    independent_ok = metrics.benchmark_resolved
                    learned = await persist_run_outcome(
                        resources.runtime,
                        cast(Any, result),
                        success=independent_ok,
                        failure_reason=(
                            None
                            if independent_ok
                            else (
                                "independent benchmark verification "
                                f"{metrics.benchmark_verification_status}: "
                                f"{metrics.benchmark_verification.details}"
                            )
                        ),
                    )
                    result.update(learned)
                return metrics
            finally:
                if shared is None:
                    await resources.close()

        try:
            return await BenchmarkRunner(execute_entry).run(
                entries, variant=variant, output_dir=output_dir
            )
        finally:
            if shared is not None:
                await shared.close()

    results = asyncio.run(execute_benchmark())
    console.print(f"wrote {len(results)} results to {output_dir}")


if __name__ == "__main__":
    app()
