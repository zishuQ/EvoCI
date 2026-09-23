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
from typer._click.core import Command as ClickCommand
from typer._click.core import Context as ClickContext

from evoci.agents.base import AgentSuite, WorkerContext
from evoci.agents.model_agents import ModelSupervisor, ModelWorker
from evoci.benchmark.adapters import CIRepairBenchAdapter
from evoci.benchmark.campaign import CampaignError, CampaignManager
from evoci.benchmark.docker import DockerError, DockerImageManager, DockerReplayVerifier
from evoci.benchmark.execution import (
    FailedCommandReplayVerifier,
    attach_learning_metrics,
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
from evoci.benchmark.validate import validate_reference
from evoci.benchmark.variants import VariantFeatures, variant_features
from evoci.capability.materializer import (
    CapabilityMaterializer,
    resolve_capability_runtime_root,
)
from evoci.capability.miner import SkillMiner
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.config import EvoCIConfig
from evoci.demo import run_repair_demo
from evoci.domain.models import (
    CIFailure,
    FailureClass,
    RepoSpec,
    ReviewResult,
    VerificationCommandResult,
    VerificationResult,
    WorkerTask,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.graph.integration import fixer_output_from_edits
from evoci.graph.outcome import persist_run_outcome
from evoci.graph.routing import contains_workspace_review_bypass
from evoci.local_task import inspect_repository, parse_verification_command, prepare_local_task
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
    PatchConflict,
    PatchError,
    apply_edit,
    precheck_edits,
    recover_attempt_writes,
    restore_attempt_writes,
    snapshot_edit_baseline,
)
from evoci.tools.policy import WORKER_REPAIR_CAPABILITIES, PolicyViolation
from evoci.tools.shell import reset_container_executor, set_container_executor
from evoci.verification.service import VerificationService

OfficialLearningVerdict = Literal["success", "repair", "infrastructure", "skip"]


def _config_from_env(*, cwd: Path | None = None) -> EvoCIConfig:
    try:
        return EvoCIConfig.from_env(cwd=cwd)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def official_learning_verdict(
    *,
    learning_deferred: bool,
    benchmark_verification_status: str,
    benchmark_resolved: bool = False,
) -> OfficialLearningVerdict:
    """Benchmark official evaluator is the learning authority when learning was deferred."""

    del benchmark_resolved
    if not learning_deferred:
        return "skip"
    if benchmark_verification_status == "passed":
        return "success"
    if benchmark_verification_status == "failed":
        return "repair"
    return "infrastructure"


app = typer.Typer(help="Durable, self-improving multi-agent CI recovery")
runs_app = typer.Typer(help="Inspect durable runs")
memory_app = typer.Typer(help="Search long-term memory")
skills_app = typer.Typer(help="Inspect capability packages")


class _BenchmarkGroup(typer.core.TyperGroup):
    """`evoci benchmark --manifest ...` still runs; `validate` is a subcommand."""

    def resolve_command(
        self, ctx: ClickContext, args: list[str]
    ) -> tuple[str | None, ClickCommand | None, list[str]]:
        if args and args[0] in self.commands:
            return super().resolve_command(ctx, args)
        return super().resolve_command(ctx, ["run", *args])


benchmark_app = typer.Typer(
    cls=_BenchmarkGroup,
    help="Execute and validate benchmark workspaces",
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
app.add_typer(runs_app, name="runs")
app.add_typer(memory_app, name="memory")
app.add_typer(skills_app, name="skills")
app.add_typer(benchmark_app, name="benchmark")
console = Console()


@dataclass(slots=True)
class LiveResources:
    runtime: GraphRuntime
    checkpoint: AsyncSQLiteCheckpointHandle
    event_store: SQLiteEventStore
    run_store: SQLiteRunStore
    memory_store: SQLiteMemoryStore
    registry: CapabilityRegistry
    supervisor_gateway: ToolLoopGateway
    worker_gateway: ToolLoopGateway
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
    supervisor_gateway = OpenAICompatibleGateway(
        config,
        role=config.supervisor_runtime(),
    )
    worker_gateway = OpenAICompatibleGateway(
        config,
        role=config.worker_runtime(),
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
    agent_registry = registry if enabled.capabilities else None
    checkpoint = await create_async_sqlite_checkpointer(config.state_dir / "checkpoints.sqlite")
    runtime = GraphRuntime(
        config=config,
        agents=AgentSuite(
            supervisor=ModelSupervisor(
                supervisor_gateway,
                recorder,
                capability_registry=agent_registry,
                memory_store=memory_store if enabled.long_term_memory else None,
                max_iterations=config.max_leaf_iterations,
                max_tool_calls=config.max_leaf_tool_calls,
                timeout=config.command_timeout_seconds,
                max_chars=config.output_limit_chars,
                budget_manager=budget_manager,
            ),
            worker=ModelWorker(
                worker_gateway,
                recorder,
                capability_registry=agent_registry,
                memory_store=memory_store if enabled.long_term_memory else None,
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
            ModelMemoryConsolidator(worker_gateway) if enabled.long_term_memory else None
        ),
        capability_registry=registry if enabled.capabilities else None,
        capability_retriever=(
            CapabilityRetriever(
                registry,
                top_k=config.capability_retrieval_top_k,
                catalog_limit_chars=config.skill_catalog_limit_chars,
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
        skill_miner=SkillMiner(worker_gateway) if enabled.capabilities else None,
        recorder=recorder,
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
        supervisor_gateway,
        worker_gateway,
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
    _print_learning_summary(result)
    console.print(f"Full report         evoci runs report {task_id}")


def _print_learning_summary(result: dict[str, Any]) -> None:
    summary = result.get("learning_summary")
    if not isinstance(summary, dict):
        return
    facts = summary.get("long_term_facts") or 0
    fact_line = f"{facts} added" if facts else "none"
    console.print("Learning:")
    console.print(f"  Episode: {summary.get('episode', 'none')}")
    console.print(f"  Long-term facts: {fact_line}")
    console.print(f"  Skill: {summary.get('skill', 'none')}")
    console.print(f"  Skill memory: {summary.get('skill_memory', 'none')}")


def _print_run_report(run_id: str, events: list[Any]) -> None:
    """Render the durable event trail as a concise, user-facing investigation report."""

    console.print(f"\nEvoCI investigation report: [bold]{run_id}[/bold]")
    plan_event = next(
        (
            event
            for event in events
            if event.type == EventType.AGENT_COMPLETED and event.agent_id == "supervisor"
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
            str(event.agent_id or "unknown").removeprefix("worker:"),
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


def _task_completion_line(task_id: str, metrics: RunMetrics) -> str:
    return (
        f"Task {task_id}: {metrics.benchmark_verification_status}, tokens: {metrics.total_tokens:,}"
    )


def _round_summary_line(results: list[BenchmarkResult], round_number: int | None) -> str:
    passed = sum(
        result.metrics is not None and result.metrics.benchmark_resolved for result in results
    )
    failed = sum(
        result.metrics is not None and result.metrics.benchmark_verification_status == "failed"
        for result in results
    )
    other = len(results) - passed - failed
    total_tokens = sum(
        result.metrics.total_tokens if result.metrics is not None else 0 for result in results
    )
    round_label = f"Round {round_number}" if round_number is not None else "Benchmark"
    summary = (
        f"{round_label} summary: passed={passed}, failed={failed}, total_tokens={total_tokens:,}"
    )
    if other:
        summary += f", other={other}"
    return summary


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
            "supervisor": "正在规划调查或修复任务...",
            "worker": "正在执行授权范围内的调查或修复...",
            "memory-consolidator": "正在整理本次运行的可复用经验...",
            "skill-miner": "正在判断是否产生新的可复用技能...",
        }
        return messages.get(role, "正在执行模型步骤...")
    if event_type == EventType.TOOL_RESULT and not payload.get("success", True):
        return _failed_command_progress(payload, announced)
    if event_type == EventType.AGENT_COMPLETED:
        if agent == "supervisor":
            tasks = payload.get("tasks", [])
            if payload.get("action") == "stop":
                return f"主管结束: {payload.get('stop_reason') or payload.get('reasoning_summary')}"
            return _investigation_plan_progress(tasks)
        if agent.startswith("worker:"):
            count = payload.get("evidence_count", 0)
            return _investigation_evidence_progress(agent.split(":", 1)[1], count, evidence or [])
    if event_type == EventType.DIAGNOSIS_CREATED:
        primary = payload.get("diagnosis", {}).get("primary", {})
        root_cause = str(primary.get("root_cause", "根因已确定"))
        return f"根因判断: {root_cause}"
    if event_type == EventType.PATCH_CREATED:
        files = payload.get("fixer_output", {}).get("proposal", {}).get("changed_files", [])
        summary = (
            payload.get("fixer_output", {}).get("proposal", {}).get("summary", "") or "未提供说明"
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
    files = FileTools(
        workspace,
        writable=True,
        max_chars=resources.runtime.config.output_limit_chars,
    )
    assert resources.runtime.budget_manager is not None
    budget = resources.runtime.budget_manager.for_run(run_id)
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
            run = await resources.runtime.agents.worker.execute(
                context=WorkerContext(
                    run_id=run_id,
                    repo=repo,
                    failure=failure,
                    workspace_path=str(workspace),
                    invocation_id=f"repair:{attempt}",
                    task=WorkerTask(
                        task_id=f"repair-{attempt}",
                        kind="repair",
                        objective=failure.summary,
                        acceptance_criteria=["formal verification passes"],
                        write_scope=["calculator.py", "app.py", "calc.py"],
                    ),
                    previous_attempt_summary=previous_summary,
                ),
                capabilities=WORKER_REPAIR_CAPABILITIES,
            )
            output = fixer_output_from_edits(
                summary=run.result.summary,
                edits=run.edits,
                commands_run=run.commands_run,
                verification_plan=run.verification_plan,
            )
        except RepairBudgetExhausted as exc:
            failure_reason = str(exc)
            break

        baseline = snapshot_edit_baseline(workspace, output.edits)
        written = recover_attempt_writes(workspace, output.edits)
        apply_failed = False
        hash_strict = resources.runtime.config.patch_hash_strict
        try:
            precheck_edits(workspace, output.edits, hash_strict=hash_strict)
            if output.edits:
                budget.ensure_tool_calls(len(output.edits))
        except (RepairBudgetExhausted, PatchConflict, PatchError) as exc:
            restore_attempt_writes(workspace, baseline, written)
            failure_reason = (
                str(exc) if isinstance(exc, RepairBudgetExhausted) else f"patch apply failed: {exc}"
            )
            previous_summary = failure_reason
            continue
        try:
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
                    created, modified = apply_edit(workspace, files, edit, hash_strict=hash_strict)
                    written[edit.path] = None if edit.delete else edit.content
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
            restore_attempt_writes(workspace, baseline, written)
            raise
        if apply_failed:
            restore_attempt_writes(workspace, baseline, written)
            previous_summary = failure_reason
            continue

        final_blockers = contains_workspace_review_bypass(workspace)
        if final_blockers:
            restore_attempt_writes(workspace, baseline, written)
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

        try:
            verification = await verifier.run(
                workspace=workspace,
                mandatory=failure.failed_commands,
                supplementary=output.proposal.verification_plan,
                budget=budget,
                on_command_start=on_command_start,
                on_command_done=on_command_done,
            )
        except BaseException:
            restore_attempt_writes(workspace, baseline, written)
            raise
        verification_history.append(verification)
        if verification.passed:
            review = ReviewResult(accepted=False, performed=False, confidence=0.0)
            status = "success"
            failure_reason = None
            break
        restore_attempt_writes(workspace, baseline, written)
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
    config = _config_from_env()
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
    table = Table(title="Role gateways")
    table.add_column("Role")
    table.add_column("Model")
    table.add_column("Thinking requested")
    table.add_column("Effort")
    table.add_column("Uses")
    supervisor = config.supervisor_runtime()
    worker = config.worker_runtime()
    table.add_row(
        "Supervisor",
        supervisor.model_name or "not configured",
        "yes" if supervisor.enable_thinking else "no",
        supervisor.reasoning_effort or "provider default",
        "planning / replanning",
    )
    table.add_row(
        "Worker",
        worker.model_name or "not configured",
        "yes" if worker.enable_thinking else "no",
        worker.reasoning_effort or "provider default",
        "investigate / repair / learning",
    )
    console.print(table)
    console.print(
        "Thinking requested means the client sent thinking/effort parameters. "
        "It does not prove the provider executed those settings."
    )


async def _probe_role(config: EvoCIConfig, role_name: str) -> dict[str, tuple[bool, str | None]]:
    runtime = config.supervisor_runtime() if role_name == "supervisor" else config.worker_runtime()
    gateway = OpenAICompatibleGateway(config, role=runtime)
    return await gateway.smoke_test()


async def _run_model_check(config: EvoCIConfig) -> None:
    console.print("\nModel connectivity")
    console.print(f"Endpoint           {config.model_base_url}")
    console.print(f"Model              {config.model_name or 'not configured'}")
    overall = True
    for role_name, needs_tools in (("supervisor", True), ("worker", True)):
        runtime = (
            config.supervisor_runtime() if role_name == "supervisor" else config.worker_runtime()
        )
        console.print(f"\n{role_name.title()}")
        console.print(f"  Thinking requested  {'yes' if runtime.enable_thinking else 'no'}")
        console.print(f"  Reasoning effort    {runtime.reasoning_effort or 'provider default'}")
        try:
            results = await _probe_role(config, role_name)
        except Exception as exc:
            failure_reason = str(exc)[:160]
            results = {
                name: (False, failure_reason)
                for name in ("basic_chat", "structured_output", "tool_calling")
            }
        labels = (
            ("structured_output", "Structured output"),
            ("tool_calling", "Tool calling"),
        )
        for key, label in labels:
            if key not in results:
                continue
            ok, reason = results[key]
            overall = overall and ok
            console.print(f"  {label:<19}{'PASS' if ok else 'FAIL'}")
            if not ok and reason:
                console.print(f"    {reason}")
        if not results.get("structured_output", (True, None))[0]:
            console.print("  Structured output is required for this role.")
        if needs_tools and not results.get("tool_calling", (True, None))[0]:
            console.print("  Tool calling is required for this role's bounded tool loop.")
    console.print(f"\nEvoCI compatible    {'YES' if overall else 'NO'}")
    console.print(
        "PASS means the live probe observed a valid response. "
        "It does not certify that the provider honored thinking parameters."
    )


@app.command("model-check")
def model_check() -> None:
    """Run live Basic Chat, Structured Output, and Tool Calling probes."""
    asyncio.run(_run_model_check(_config_from_env()))


@app.command()
def demo() -> None:
    """Run an offline supervisor→worker→verify demonstration."""
    result = asyncio.run(run_repair_demo(Path.cwd()))
    console.print(f"status: [bold]{result['status']}[/bold]")
    console.print(f"evidence: {len(result.get('evidence', []))}")
    console.print(f"supervisor batches: {result.get('supervisor_batch', 0)}")
    console.print(f"episode: {result.get('episode_id')}")


@app.command("fix")
def fix_repository(
    command: Annotated[str, typer.Option("--command", "-c", help="Original verification command")],
    repo: Annotated[
        Path, typer.Option(help="Repository directory (defaults to current directory)")
    ] = Path("."),
    description: Annotated[
        str | None, typer.Option(help="Optional description of the failure")
    ] = None,
    prepare_only: Annotated[
        bool, typer.Option(help="Reproduce and save a task without model calls")
    ] = False,
) -> None:
    """Reproduce a command failure and repair the current repository; no task JSON required."""
    try:
        argv = parse_verification_command(command)
        repository = inspect_repository(repo)
        config = _config_from_env(cwd=repository.root)
        resolve_capability_runtime_root(
            config.runtime_dir,
            repository.root,
            allow_relocate=not config.runtime_dir_explicit,
        )
    except (OSError, ValueError, PolicyViolation) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Repository: {repository.root}")
    if repository.dirty_status:
        console.print("检测到已有本地修改; 复现将使用包含这些修改的隔离副本。")
    console.print("正在隔离副本中复现失败命令...")

    async def execute() -> dict[str, Any]:
        prepared = await prepare_local_task(repository, argv, config, description=description)
        console.print(f"Task saved: {prepared.task_file}")
        console.print(f"Preflight report: {prepared.report_file}")
        if prepared.preflight.exit_code == 0 and not prepared.preflight.timed_out:
            console.print("命令已通过, 无需修复; 未调用模型。")
            return {"status": "already_passing"}
        if prepare_only:
            console.print("失败已复现, 任务已保存; 未调用模型。")
            return {"status": "prepared"}
        if not config.model_name or not config.model_api_key:
            console.print("任务已准备。请配置 EVO_MODEL_NAME 和 EVO_MODEL_API_KEY 后重新运行。")
            return {"status": "configuration_required"}
        run_id = str(prepared.initial["run_id"])
        console.print("失败已复现, 开始修复。通过验证的修改将保留在当前仓库。")
        resources = await _live_resources(config)
        try:
            result = await _drive_graph(resources, prepared.initial, run_id)
            _print_run_summary(result, task_id=run_id)
            return {"run_id": run_id, "status": result.get("status")}
        finally:
            await resources.close()

    try:
        outcome = asyncio.run(execute())
    except (OSError, ValueError, PolicyViolation) as exc:
        console.print(f"无法准备修复任务: {exc}")
        raise typer.Exit(2) from exc
    console.print_json(data=outcome)
    if outcome["status"] == "configuration_required":
        raise typer.Exit(2)
    if outcome["status"] not in {"success", "prepared", "already_passing"}:
        raise typer.Exit(1)


@app.command("run")
def run_task(
    task_id: str,
    task_file: Annotated[Path | None, typer.Option(help="JSON task specification")] = None,
) -> None:
    """Start a live model-backed run."""
    config = _config_from_env()
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
    config = _config_from_env()

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
    config = _config_from_env()
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
    config = _config_from_env()
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
    config = _config_from_env()
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
    repository: Annotated[str, typer.Option(help="Repository full name, e.g. owner/name")],
) -> None:
    config = _config_from_env()
    store = SQLiteMemoryStore(config.state_dir / "memory.sqlite")
    try:
        hits = store.search_long_term(query, repository=repository, limit=10)
        console.print_json(data={"hits": [hit.model_dump() for hit in hits]})
    finally:
        store.close()


@skills_app.command("list")
def list_skills() -> None:
    config = _config_from_env()
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        table = Table("Skill", "Enabled", "Description")
        for record in registry.list(enabled_only=False):
            manifest = record.manifest
            table.add_row(
                manifest.skill_id,
                "yes" if manifest.enabled else "no",
                manifest.description,
            )
        console.print(table)
    finally:
        registry.close()


@skills_app.command("show")
def show_skill(skill_id: str) -> None:
    config = _config_from_env()
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        record = registry.get(skill_id)
        if record is None:
            raise typer.BadParameter(f"unknown skill: {skill_id}")
        console.print_json(data=record.manifest.model_dump(mode="json"))
        console.print((Path(record.package_path) / "SKILL.md").read_text())
    finally:
        registry.close()


@skills_app.command("enable")
def enable_skill(skill_id: str) -> None:
    config = _config_from_env()
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        registry.enable(skill_id)
        console.print(f"Enabled {skill_id}")
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        registry.close()


@skills_app.command("disable")
def disable_skill(skill_id: str) -> None:
    config = _config_from_env()
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        registry.disable(skill_id)
        console.print(f"Disabled {skill_id}")
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        registry.close()


@skills_app.command("memory")
def show_skill_memory(skill_id: str) -> None:
    config = _config_from_env()
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        if registry.get(skill_id) is None:
            raise typer.BadParameter(f"unknown skill: {skill_id}")
        memory_path = registry.skill_root(skill_id) / "memory.md"
        text = memory_path.read_text(encoding="utf-8") if memory_path.is_file() else ""
        console.print(text or "(empty)")
    finally:
        registry.close()


@benchmark_app.command("run")
def benchmark_run(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    variant: Annotated[BenchmarkVariant, typer.Option()] = "multi",
    output_dir: Annotated[Path, typer.Option()] = Path("results"),
    continual: Annotated[bool, typer.Option(help="Preserve learned state across tasks")] = False,
    campaign_dir: Annotated[Path | None, typer.Option(help="Stable campaign directory")] = None,
    round_number: Annotated[int | None, typer.Option("--round", min=1)] = None,
    freeze_learning_within_round: Annotated[
        bool, typer.Option(help="Freeze reads to the parent Generation")
    ] = False,
    campaign_strict: Annotated[
        bool, typer.Option(help="Reject cross-round task/commit overlap")
    ] = False,
    docker_official_images: Annotated[
        bool,
        typer.Option(help="Pull, pin, and verify tasks in dataset official images"),
    ] = False,
    parallelism: Annotated[
        int, typer.Option(min=1, help="Maximum tasks to execute concurrently within a round")
    ] = 1,
    enable_thinking: Annotated[
        bool,
        typer.Option(
            "--enable-thinking",
            help="Removed; set EVO_SUPERVISOR_ENABLE_THINKING and EVO_WORKER_ENABLE_THINKING",
            hidden=True,
        ),
    ] = False,
) -> None:
    """Execute real benchmark workspaces through the selected runtime ablation."""
    entries = BenchmarkRunner.load_manifest(manifest)
    adapter = CIRepairBenchAdapter(dataset)
    session_id = uuid4().hex[:10]
    if (campaign_dir is None) != (round_number is None):
        raise typer.BadParameter("--campaign-dir and --round must be provided together")
    if campaign_dir is not None and (variant != "evo" or not freeze_learning_within_round):
        raise typer.BadParameter(
            "campaign mode requires --variant evo and --freeze-learning-within-round"
        )

    async def execute_benchmark() -> list[BenchmarkResult]:
        if enable_thinking:
            raise typer.BadParameter(
                "--enable-thinking is removed. Set EVO_SUPERVISOR_ENABLE_THINKING and "
                "EVO_WORKER_ENABLE_THINKING (and the matching EVO_*_REASONING_EFFORT values)."
            )
        base_config = _config_from_env()
        features = variant_features(variant)
        campaign = (
            CampaignManager(
                campaign_dir, variant=variant, base_config=base_config, strict=campaign_strict
            )
            if campaign_dir is not None
            else None
        )
        campaign_round_dir: Path | None = None
        if campaign is not None:
            assert round_number is not None
            campaign_round_dir, warnings = campaign.prepare_round(
                round_number, manifest, dataset, entries
            )
            for warning in warnings:
                console.print(f"[yellow]Campaign warning:[/yellow] {warning}")
        docker_specs = {
            entry.task_id: spec
            for entry in entries
            if not entry.skipped and (spec := adapter.docker_task_spec(entry.task_id)) is not None
        }
        pinned_images: dict[str, str] = {}
        if docker_official_images and docker_specs:
            docker_audit = (campaign_round_dir or output_dir) / "docker-images.json"
            console.print(
                "[cyan]Docker:[/cyan] pulling and pinning "
                f"{len(docker_specs)} official task images..."
            )
            pinned_images = DockerImageManager().prepare(
                list(docker_specs.values()), audit_path=docker_audit
            )
            console.print("[green]Docker:[/green] all official images pinned")
        shared: LiveResources | None = None
        task_index = 0
        if continual and campaign is None:
            shared_config = _benchmark_runtime_config(
                base_config,
                output_dir=output_dir,
                variant=variant,
                session_id=session_id,
                task_index=None,
            )
            shared = await _live_resources(shared_config, features, defer_success_learning=True)

        async def execute_entry(
            entry: BenchmarkManifestEntry, selected_variant: BenchmarkVariant
        ) -> RunMetrics:
            nonlocal task_index
            task_index += 1
            run_id = (
                f"campaign-{campaign.root.name}-r{round_number}-{entry.task_id}"
                if campaign is not None
                else f"bench-{task_index}-{uuid4().hex[:8]}"
            )
            active_config = (
                campaign.task_config(round_number, entry.task_id, run_id)
                if campaign is not None and round_number is not None
                else _benchmark_runtime_config(
                    base_config,
                    output_dir=output_dir,
                    variant=selected_variant,
                    session_id=session_id,
                    task_index=None if continual else task_index,
                )
            )
            resources = shared or await _live_resources(
                active_config, features, defer_success_learning=True
            )
            console.print(f"[cyan]Task {entry.task_id}:[/cyan] starting")
            verifier: FailedCommandReplayVerifier | DockerReplayVerifier
            docker_spec = docker_specs.get(entry.task_id) if docker_official_images else None
            if docker_spec is not None:
                verifier = DockerReplayVerifier(
                    docker_spec,
                    pinned_images[entry.task_id],
                    timeout=max(1800.0, active_config.command_timeout_seconds),
                    work_timeout=active_config.command_timeout_seconds,
                    max_chars=max(500_000, active_config.output_limit_chars),
                )
            else:
                verifier = FailedCommandReplayVerifier(
                    timeout=active_config.command_timeout_seconds,
                    max_chars=active_config.output_limit_chars,
                )
            executor_token = None
            try:
                workspace, prepared = prepare_workspace(
                    adapter,
                    entry,
                    worktrees_dir=active_config.workspace_dir,
                    repo_cache_dir=active_config.repo_cache_dir,
                    run_id=run_id,
                )
                if isinstance(verifier, DockerReplayVerifier):
                    verifier.protect(workspace)
                    benchmark_preflight = await verifier.preflight(prepared, workspace)
                    verifier.start_work_container(workspace)
                    executor_token = set_container_executor(verifier.execute_agent_command)
                else:
                    preflight_workspace, preflight_task = prepare_workspace(
                        adapter,
                        entry,
                        worktrees_dir=active_config.workspace_dir,
                        repo_cache_dir=active_config.repo_cache_dir,
                        run_id=f"{run_id}-preflight",
                    )
                    benchmark_preflight = await verifier.preflight(
                        preflight_task, preflight_workspace
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
                if campaign is not None and round_number is not None:
                    initial["campaign_provenance"] = {
                        "campaign_id": campaign.root.name,
                        "round": round_number,
                        "task_id": entry.task_id,
                        "run_id": run_id,
                        "read_generation": round_number - 1,
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
                    len(resources.registry.list(enabled_only=True))
                    if features.capabilities
                    else 0
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
                verdict = official_learning_verdict(
                    learning_deferred=bool(result.get("learning_deferred")),
                    benchmark_verification_status=metrics.benchmark_verification_status,
                    benchmark_resolved=metrics.benchmark_resolved,
                )
                if verdict != "skip":
                    independent_ok = verdict == "success"
                    if verdict == "infrastructure":
                        failure_class: FailureClass | None = "infrastructure"
                        failure_reason = metrics.benchmark_verification.details
                    elif independent_ok:
                        failure_class = None
                        failure_reason = None
                    else:
                        failure_class = "repair"
                        failure_reason = (
                            "independent benchmark verification "
                            f"{metrics.benchmark_verification_status}: "
                            f"{metrics.benchmark_verification.details}"
                        )
                    learned = await persist_run_outcome(
                        resources.runtime,
                        cast(Any, result),
                        success=independent_ok,
                        failure_class=failure_class,
                        failure_reason=failure_reason,
                    )
                    result.update(learned)
                    registry_size = len(resources.registry.list()) if features.capabilities else 0
                    active_count = (
                        len(resources.registry.list(enabled_only=True))
                        if features.capabilities
                        else 0
                    )
                    metrics = attach_learning_metrics(
                        metrics,
                        result=result,
                        recorder=resources.recorder,
                        run_id=run_id,
                        skill_registry_size=registry_size,
                        active_skill_count=active_count,
                    )
                console.print(_task_completion_line(entry.task_id, metrics))
                return metrics
            except Exception as exc:
                console.print(f"[red]Task {entry.task_id}:[/red] error: {exc}")
                raise
            finally:
                if executor_token is not None:
                    reset_container_executor(executor_token)
                if isinstance(verifier, DockerReplayVerifier):
                    verifier.close()
                if shared is None:
                    await resources.close()

        try:
            run_output = campaign_round_dir or output_dir
            results = await BenchmarkRunner(execute_entry).run(
                entries,
                variant=variant,
                output_dir=run_output,
                resume=campaign is not None,
                parallelism=parallelism,
            )
            if campaign is not None and round_number is not None:
                campaign.finalize_round(round_number, results)
            return results
        finally:
            if shared is not None:
                await shared.close()

    try:
        if campaign_dir is not None:
            base_config = _config_from_env()
            manager = CampaignManager(
                campaign_dir, variant=variant, base_config=base_config, strict=campaign_strict
            )
            with manager.writer_lock():
                results = asyncio.run(execute_benchmark())
        else:
            results = asyncio.run(execute_benchmark())
    except (CampaignError, DockerError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    destination = campaign_dir / "rounds" / f"round-{round_number}" if campaign_dir else output_dir
    console.print(f"[bold]{_round_summary_line(results, round_number)}[/bold]")
    console.print(f"wrote {len(results)} results to {destination}")


def benchmark(*args: object, **kwargs: object) -> None:
    """Programmatic alias used by tests; CLI entry is `evoci benchmark run`."""

    benchmark_run(*args, **kwargs)  # type: ignore[arg-type]


@benchmark_app.command("validate")
def benchmark_validate(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    docker_official_images: Annotated[
        bool,
        typer.Option(help="Pin and score tasks in dataset official images"),
    ] = False,
    apply_reference_patch: Annotated[
        bool,
        typer.Option(help="Apply the withheld reference patch after baseline"),
    ] = False,
    task_id: Annotated[
        list[str] | None,
        typer.Option(help="Limit validation to these task ids"),
    ] = None,
    output_dir: Annotated[Path, typer.Option()] = Path("results"),
    timeout: Annotated[float, typer.Option(help="Per-command eval timeout in seconds")] = 1800.0,
) -> None:
    """Gold-patch validation. Never calls a model and never writes Memory/Skill."""

    try:
        report = validate_reference(
            dataset=dataset,
            manifest=manifest,
            docker_official_images=docker_official_images,
            apply_reference_patch=apply_reference_patch,
            task_ids=task_id,
            timeout=timeout,
            audit_path=output_dir / "docker-images.json",
        )
    except (DockerError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    (output_dir / "reference-validation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    console.print_json(data=payload)
    console.print(
        f"[cyan]reference validation:[/cyan] {report.valid} valid, "
        f"{report.invalid} invalid, {report.infra_error} infra_error"
    )
    if report.invalid or report.infra_error:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
