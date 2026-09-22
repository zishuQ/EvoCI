"""Workspace preparation, independent verification, and benchmark metrics."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Protocol

from evoci.benchmark.adapters import BenchmarkAdapter
from evoci.benchmark.models import (
    BenchmarkCommandResult,
    BenchmarkManifestEntry,
    BenchmarkPreflightResult,
    BenchmarkVerificationResult,
    FinalWorkspaceChanges,
    GroundTruth,
    PreparedTask,
    RunMetrics,
)
from evoci.runtime.events import EventType, RunEvent
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.isolation import copy_workspace_with_independent_git
from evoci.tools.shell import CommandRunner
from evoci.verification.service import VerificationService
from evoci.workspace.manager import WorkspaceManager


class BenchmarkVerifier(Protocol):
    async def preflight(
        self,
        task: PreparedTask,
        workspace: Path,
    ) -> BenchmarkPreflightResult: ...

    async def verify(
        self,
        task: PreparedTask,
        workspace: Path,
        preflight: BenchmarkPreflightResult,
    ) -> BenchmarkVerificationResult: ...


class FailedCommandReplayVerifier:
    """Replay dataset-derived failed commands, independent of the agent's test plan."""

    def __init__(self, *, timeout: float = 120.0, max_chars: int = 32_000) -> None:
        self.timeout = timeout
        self.max_chars = max_chars

    async def _run(self, command: list[str], workspace: Path) -> BenchmarkCommandResult:
        service = VerificationService(timeout=self.timeout, max_chars=self.max_chars)
        async with service.isolated_workspace(workspace) as snapshot:
            completed = await CommandRunner(
                snapshot, timeout=self.timeout, max_chars=self.max_chars
            ).run(command, extra_env={"CI": "1"})
        return BenchmarkCommandResult(
            command=command,
            exit_code=completed.exit_code,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=completed.timed_out,
        )

    async def preflight(
        self,
        task: PreparedTask,
        workspace: Path,
    ) -> BenchmarkPreflightResult:
        commands = task.agent_view.ci_failure.candidate_failed_commands
        if not commands:
            return BenchmarkPreflightResult(
                status="not_available",
                details="no original failed command could be reliably extracted",
            )
        results: list[BenchmarkCommandResult] = []
        for command in commands:
            try:
                result = await self._run(command, workspace)
            except (OSError, PermissionError, TypeError, ValueError) as exc:
                return BenchmarkPreflightResult(
                    status="not_available",
                    commands=results,
                    details=(
                        f"candidate command could not be replayed before repair: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            results.append(result)
        if any(result.timed_out for result in results):
            return BenchmarkPreflightResult(
                status="not_available",
                commands=results,
                details="candidate command timed out before repair",
            )
        reproduced = [result for result in results if result.exit_code != 0]
        if reproduced:
            return BenchmarkPreflightResult(
                status="reproduced",
                commands=results,
                details=f"reproduced {len(reproduced)} failing command(s) before repair",
            )
        return BenchmarkPreflightResult(
            status="not_reproduced",
            commands=results,
            details="candidate commands passed before repair and are not valid failure oracles",
        )

    async def verify(
        self,
        task: PreparedTask,
        workspace: Path,
        preflight: BenchmarkPreflightResult,
    ) -> BenchmarkVerificationResult:
        del task
        if preflight.status != "reproduced":
            return BenchmarkVerificationResult(
                status="not_available",
                details=f"benchmark oracle unavailable: {preflight.details}",
            )
        commands = [
            result.command
            for result in preflight.commands
            if result.exit_code != 0 and not result.timed_out
        ]
        if not commands:
            return BenchmarkVerificationResult(
                status="not_available",
                details="preflight contained no reproducibly failing command",
            )
        results: list[BenchmarkCommandResult] = []
        for command in commands:
            try:
                result = await self._run(command, workspace)
            except (OSError, PermissionError, TypeError, ValueError) as exc:
                return BenchmarkVerificationResult(
                    status="not_available",
                    commands=results,
                    details=f"validated oracle could not be replayed: {type(exc).__name__}: {exc}",
                )
            results.append(result)
            if result.exit_code != 0 or result.timed_out:
                reason = "timed out" if result.timed_out else f"exit code {result.exit_code}"
                return BenchmarkVerificationResult(
                    status="failed",
                    commands=results,
                    details=f"preflight-failing command still returned {reason}",
                )
        return BenchmarkVerificationResult(
            status="passed",
            commands=results,
            details=(
                f"{len(results)} command(s) changed from FAIL before repair to PASS after repair"
            ),
        )


def prepare_workspace(
    adapter: BenchmarkAdapter,
    entry: BenchmarkManifestEntry,
    *,
    worktrees_dir: Path,
    repo_cache_dir: Path,
    run_id: str,
) -> tuple[Path, PreparedTask]:
    """Prepare only agent-visible inputs; ground truth is intentionally not read here."""

    prepared = adapter.prepare_task(entry.task_id)
    view = prepared.agent_view
    if prepared.workspace_path is not None:
        source = Path(prepared.workspace_path).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"benchmark workspace does not exist: {source}")
        # Prepared benchmark rows are already pinned to a local failing workspace.
        # Copy Git and files locally; never fall back to a network clone for them.
        destination = worktrees_dir.resolve() / run_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            # Campaign run ids are deterministic.  An interrupted/error task may
            # already have a workspace and graph checkpoint; keep both so resume
            # does not discard completed model/tool work.
            if destination.is_dir():
                return destination, prepared.model_copy(
                    update={"workspace_path": str(destination)}
                )
            raise FileExistsError(f"benchmark workspace path is not a directory: {destination}")
        copy_workspace_with_independent_git(source, destination)
        return destination, prepared.model_copy(update={"workspace_path": str(destination)})
    else:
        source_location = f"https://github.com/{view.repo_owner}/{view.repo_name}.git"
    cache = repo_cache_dir.resolve() / f"{view.repo_owner}-{view.repo_name}.git"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        completed = subprocess.run(
            ["git", "clone", "--mirror", source_location, str(cache)],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark clone failed: {completed.stderr.strip()}")
    elif prepared.workspace_path is None:
        completed = subprocess.run(
            ["git", "-C", str(cache), "fetch", "--prune", "origin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"benchmark fetch failed: {completed.stderr.strip()}")
    workspace = WorkspaceManager(worktrees_dir).create(
        repo_path=cache,
        base_commit=view.sha_fail,
        run_id=run_id,
    )
    return workspace.path, prepared.model_copy(update={"workspace_path": str(workspace.path)})


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _status_paths(output: str) -> tuple[set[str], set[str]]:
    changed: set[str] = set()
    untracked: set[str] = set()
    records = output.split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = record[:2]
        path = record[3:] if len(record) > 3 else ""
        if path:
            changed.add(path)
            if status == "??":
                untracked.add(path)
        if "R" in status or "C" in status:
            index += 1
    return changed, untracked


def _untracked_lines(workspace: Path, paths: set[str]) -> int:
    lines = 0
    root = workspace.resolve()
    for relative in paths:
        candidate = (root / relative).resolve()
        if root not in candidate.parents or not candidate.is_file():
            continue
        data = candidate.read_bytes()
        if b"\0" not in data:
            lines += len(data.splitlines())
    return lines


def collect_final_workspace_changes(workspace: Path) -> FinalWorkspaceChanges:
    """Measure the final patch only from Git state in the real workspace."""

    status = _git(workspace, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if status.returncode != 0:
        return FinalWorkspaceChanges(changed_files=[])
    changed_files, untracked = _status_paths(status.stdout)
    names = _git(workspace, "diff", "--name-only", "HEAD", "--")
    if names.returncode == 0:
        changed_files.update(line for line in names.stdout.splitlines() if line)

    lines_added = 0
    lines_deleted = 0
    numstat = _git(workspace, "diff", "--numstat", "HEAD", "--")
    if numstat.returncode == 0:
        for row in numstat.stdout.splitlines():
            parts = row.split("\t")
            if len(parts) >= 2:
                if parts[0].isdigit():
                    lines_added += int(parts[0])
                if parts[1].isdigit():
                    lines_deleted += int(parts[1])
    lines_added += _untracked_lines(workspace, untracked)
    return FinalWorkspaceChanges(
        changed_files=sorted(changed_files),
        lines_added=lines_added,
        lines_deleted=lines_deleted,
    )


def token_metrics_from_events(events: list[RunEvent]) -> dict[str, int]:
    usage_events = [event for event in events if event.type == EventType.MODEL_USAGE]
    if usage_events:
        source = usage_events
        provider_requests = len(usage_events)
    else:
        source = [event for event in events if event.type == EventType.MODEL_CALL]
        provider_requests = 0

    def scoped(kind: str) -> list[RunEvent]:
        if kind == "repair":
            return [event for event in source if event.payload.get("budget_scope") != "post_run"]
        return [event for event in source if event.payload.get("budget_scope") == "post_run"]

    def field_sum(items: list[RunEvent], field: str) -> int:
        total = 0
        for event in items:
            value = event.payload.get(field)
            if value is None:
                continue
            total += int(value)
        return total

    repair = scoped("repair")
    learning = scoped("post_run")
    input_tokens = field_sum(source, "input_tokens")
    output_tokens = field_sum(source, "output_tokens")
    repair_input = field_sum(repair, "input_tokens")
    repair_output = field_sum(repair, "output_tokens")
    learning_input = field_sum(learning, "input_tokens")
    learning_output = field_sum(learning, "output_tokens")
    return {
        "provider_requests": provider_requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "repair_input_tokens": repair_input,
        "repair_output_tokens": repair_output,
        "repair_tokens": repair_input + repair_output,
        "learning_input_tokens": learning_input,
        "learning_output_tokens": learning_output,
        "learning_tokens": learning_input + learning_output,
        "cached_input_tokens": field_sum(source, "cached_input_tokens"),
        "reasoning_tokens": field_sum(source, "reasoning_tokens"),
    }


async def collect_metrics(
    *,
    result: dict[str, Any],
    recorder: TrajectoryRecorder,
    run_id: str,
    task: PreparedTask,
    workspace: Path,
    truth: GroundTruth,
    wall_time: float,
    verifier: BenchmarkVerifier | None = None,
    benchmark_preflight: BenchmarkPreflightResult | None = None,
    skill_registry_size: int = 0,
    active_skill_count: int = 0,
) -> RunMetrics:
    """Evaluate after execution with evaluator-owned verification and workspace state."""

    status = result.get("status")
    verification = result.get("verification")
    review = result.get("review")
    targeted_verification_passed = bool(verification and verification.passed)
    review_performed = bool(review is not None and getattr(review, "performed", True))
    review_passed = bool(review_performed and review is not None and review.accepted)
    trajectory = recorder.build_view(
        run_id=run_id,
        verification_history=result.get("verification_history", []),
        final_status="success" if status == "success" else "failed",
        failure_reason=result.get("failure_reason"),
    )
    final_changes = collect_final_workspace_changes(workspace)
    attempted_files = sorted(set(trajectory.created_files + trajectory.modified_files))
    expected_paths = set(truth.changed_files)
    actual_paths = set(final_changes.changed_files)
    gold_file_overlap = (
        len(expected_paths.intersection(actual_paths)) / len(expected_paths)
        if expected_paths
        else None
    )
    if benchmark_preflight is None:
        benchmark_verification = BenchmarkVerificationResult(
            status="not_available",
            details="preflight verification was not performed on the original workspace",
        )
    else:
        benchmark_verification = await (verifier or FailedCommandReplayVerifier()).verify(
            task, workspace, benchmark_preflight
        )
    events = recorder.events(run_id)
    tokens = token_metrics_from_events(events)
    repair_model_calls = sum(
        event.type == EventType.MODEL_CALL and event.payload.get("budget_scope") != "post_run"
        for event in events
    )
    post_run_model_calls = sum(
        event.type == EventType.MODEL_CALL and event.payload.get("budget_scope") == "post_run"
        for event in events
    )
    repair_tool_calls = sum(
        event.type == EventType.TOOL_CALL
        and event.payload.get("budget_scope") not in {"post_run", "cleanup"}
        for event in events
    )
    post_run_tool_calls = sum(
        event.type == EventType.TOOL_CALL and event.payload.get("budget_scope") == "post_run"
        for event in events
    )
    investigators = {
        event.agent_id
        for event in events
        if event.type == EventType.AGENT_COMPLETED
        and event.agent_id
        and event.agent_id.startswith("worker:")
    }
    skill_candidates = [
        event for event in events if event.type == EventType.SKILL_CANDIDATE_CREATED
    ]
    decision = result.get("learning_decision") or {}
    return RunMetrics(
        agent_declared_success=status == "success",
        targeted_verification_passed=targeted_verification_passed,
        review_passed=review_passed,
        review_performed=review_performed,
        benchmark_verification=benchmark_verification,
        benchmark_verification_status=benchmark_verification.status,
        benchmark_resolved=benchmark_verification.status == "passed",
        gold_file_overlap=gold_file_overlap,
        final_workspace_changes=final_changes,
        attempted_files=attempted_files,
        wall_time=wall_time,
        llm_calls=trajectory.model_call_count,
        repair_model_calls=repair_model_calls,
        post_run_model_calls=post_run_model_calls,
        provider_requests=tokens["provider_requests"],
        input_tokens=tokens["input_tokens"],
        output_tokens=tokens["output_tokens"],
        total_tokens=tokens["total_tokens"],
        repair_input_tokens=tokens["repair_input_tokens"],
        repair_output_tokens=tokens["repair_output_tokens"],
        repair_tokens=tokens["repair_tokens"],
        learning_input_tokens=tokens["learning_input_tokens"],
        learning_output_tokens=tokens["learning_output_tokens"],
        learning_tokens=tokens["learning_tokens"],
        cached_input_tokens=tokens["cached_input_tokens"],
        reasoning_tokens=tokens["reasoning_tokens"],
        tool_calls=trajectory.tool_call_count,
        repair_tool_calls=repair_tool_calls,
        post_run_tool_calls=post_run_tool_calls,
        failed_tool_calls=sum(agent.failed_tool_calls for agent in trajectory.agents),
        repair_attempts=int(result.get("supervisor_batch", result.get("repair_attempt", 0))),
        files_changed=len(final_changes.changed_files),
        lines_changed=final_changes.lines_added + final_changes.lines_deleted,
        workers_spawned=len(investigators),
        parallel_rounds=int(result.get("supervisor_batch", result.get("investigation_round", 0))),
        investigation_rounds=int(
            result.get("supervisor_batch", result.get("investigation_round", 0))
        ),
        evidence_count=len(trajectory.evidence),
        memory_retrieval_count=len(trajectory.memories_retrieved),
        memory_selected_count=len(trajectory.memories_selected),
        memory_used_count=len(trajectory.memories_used),
        skills_retrieved=len(trajectory.skills_retrieved),
        skills_selected=len(trajectory.skills_selected),
        skills_used=len(trajectory.skills_used),
        skill_created=len(skill_candidates),
        skill_updated=int(decision.get("action") == "update_skill")
        + sum(event.type == EventType.SKILL_UPDATED for event in events),
        skills_promoted=0,
        skills_rejected=0,
        skills_superseded=0,
        skill_registry_size=skill_registry_size,
        active_skill_count=active_skill_count,
    )


def attach_learning_metrics(
    metrics: RunMetrics,
    *,
    result: dict[str, Any],
    recorder: TrajectoryRecorder,
    run_id: str,
    skill_registry_size: int,
    active_skill_count: int,
) -> RunMetrics:
    """Refresh post-learning fields without re-running the independent oracle."""

    trajectory = recorder.build_view(
        run_id=run_id,
        verification_history=result.get("verification_history", []),
        final_status="success" if result.get("status") == "success" else "failed",
        failure_reason=result.get("failure_reason"),
    )
    events = recorder.events(run_id)
    tokens = token_metrics_from_events(events)
    skill_candidates = [
        event for event in events if event.type == EventType.SKILL_CANDIDATE_CREATED
    ]
    decision = result.get("learning_decision") or {}
    return metrics.model_copy(
        update={
            "llm_calls": trajectory.model_call_count,
            "post_run_model_calls": sum(
                event.type == EventType.MODEL_CALL
                and event.payload.get("budget_scope") == "post_run"
                for event in events
            ),
            **tokens,
            "post_run_tool_calls": sum(
                event.type == EventType.TOOL_CALL
                and event.payload.get("budget_scope") == "post_run"
                for event in events
            ),
            "skills_used": len(trajectory.skills_used),
            "skills_selected": len(trajectory.skills_selected),
            "skill_created": len(skill_candidates),
            "skill_updated": int(decision.get("action") == "update_skill")
            + sum(event.type == EventType.SKILL_UPDATED for event in events),
            "skills_promoted": 0,
            "skills_rejected": 0,
            "skill_registry_size": skill_registry_size,
            "active_skill_count": active_skill_count,
        }
    )
