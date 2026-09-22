import json
import subprocess
from pathlib import Path

import pytest

from evoci.agents.base import AgentSuite
from evoci.benchmark.adapters import FixtureAdapter, normalize_ci_failure
from evoci.benchmark.execution import (
    FailedCommandReplayVerifier,
    collect_metrics,
    token_metrics_from_events,
)
from evoci.benchmark.manifest import generate_balanced_manifest
from evoci.benchmark.models import (
    AgentTaskView,
    BenchmarkManifestEntry,
    BenchmarkResult,
    BenchmarkVerificationResult,
    FinalWorkspaceChanges,
    GroundTruth,
    NormalizedCIFailure,
    PreparedTask,
    RunMetrics,
)
from evoci.benchmark.runner import BenchmarkRunner
from evoci.benchmark.variants import variant_features
from evoci.config import EvoCIConfig
from evoci.demo import (
    DemoCoordinator,
    DemoInvestigator,
)
from evoci.domain.models import (
    CIFailure,
    RepoSpec,
    ReviewResult,
    VerificationCommandResult,
    VerificationResult,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder


def test_agent_view_cannot_contain_ground_truth() -> None:
    fixture = Path(__file__).parents[1] / "fixtures/ci_tasks/tasks.jsonl"
    adapter = FixtureAdapter(fixture)
    prepared = adapter.prepare_task("fixture-test")
    serialized = prepared.agent_view.model_dump_json()

    assert "sha_success" not in serialized
    assert "changed_files" not in serialized
    assert '"diff"' not in serialized
    assert "error_type" not in serialized
    assert adapter.ground_truth("fixture-test").sha_success == "success-test"


def test_realistic_workflow_and_structured_logs_are_normalized_without_argv_leakage() -> None:
    fixture = Path(__file__).parents[1] / "fixtures/ci_tasks/tasks.jsonl"
    adapter = FixtureAdapter(fixture)
    view = adapter.prepare_task("fixture-test").agent_view
    failure = view.ci_failure

    assert view.workflow_name == "ci"
    assert failure.workflow_yaml.startswith("name: CI\n")
    assert "pull_request:" in failure.workflow_yaml
    assert failure.failed_steps[0].name == "Run tests"
    assert failure.failed_steps[0].command == ["python", "-m", "unittest", "-q"]
    assert failure.candidate_failed_commands == [["python", "-m", "unittest", "-q"]]
    assert all(command[0] != "name:" for command in failure.candidate_failed_commands)
    assert adapter.ground_truth("fixture-test").error_type == "Test Failure"

    structured = normalize_ci_failure(
        {
            "workflow": {
                "name": "CI",
                "on": {"pull_request": {}},
                "jobs": {"test": {"steps": [{"run": "pytest"}]}},
            },
            "logs": [{"step_name": "tests", "log": "AssertionError"}],
        }
    )
    assert "name: CI" in structured.workflow_yaml
    assert structured.candidate_failed_commands == []


@pytest.mark.asyncio
async def test_benchmark_runner_writes_machine_readable_results(tmp_path: Path) -> None:
    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del entry, variant
        return RunMetrics(
            agent_declared_success=True,
            targeted_verification_passed=True,
            review_passed=True,
            benchmark_verification=BenchmarkVerificationResult(
                status="passed", details="fixture replay passed"
            ),
            benchmark_verification_status="passed",
            benchmark_resolved=True,
            gold_file_overlap=1.0,
            final_workspace_changes=FinalWorkspaceChanges(changed_files=["app.py"]),
            files_changed=1,
            wall_time=0.1,
            tool_calls=3,
        )

    runner = BenchmarkRunner(run_task)  # type: ignore[arg-type]
    results = await runner.run(
        [
            BenchmarkManifestEntry(task_id="one", category="Test Failure", source="fixture"),
            BenchmarkManifestEntry(
                task_id="two",
                category="Dependency",
                skipped=True,
                skip_reason="fixture unavailable",
            ),
        ],
        variant="multi",
        output_dir=tmp_path,
    )

    assert len(results) == 2
    assert (tmp_path / "runs.jsonl").is_file()
    assert '"benchmark_success_rate": 1.0' in (tmp_path / "aggregate.json").read_text()
    assert (tmp_path / "by_error_type.csv").is_file()


@pytest.mark.asyncio
async def test_benchmark_runner_resume_retries_errors_only(tmp_path: Path) -> None:
    calls: list[str] = []

    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del variant
        calls.append(entry.task_id)
        return RunMetrics(
            agent_declared_success=False,
            targeted_verification_passed=False,
            review_passed=False,
            benchmark_verification=BenchmarkVerificationResult(
                status="not_available", details="offline"
            ),
            benchmark_verification_status="not_available",
            benchmark_resolved=False,
            final_workspace_changes=FinalWorkspaceChanges(changed_files=[]),
            wall_time=0.0,
        )

    completed = BenchmarkResult(
        task_id="done",
        variant="evo",
        status="not_evaluable",
        metrics=await run_task(BenchmarkManifestEntry(task_id="x", category="x"), "evo"),
    )
    error = BenchmarkResult(
        task_id="retry",
        variant="evo",
        status="error",
        error_type="RuntimeError",
        error_message="interrupted",
    )
    calls.clear()
    (tmp_path / "runs.jsonl").write_text(
        completed.model_dump_json() + "\n" + error.model_dump_json() + "\n"
    )
    await BenchmarkRunner(run_task).run(
        [
            BenchmarkManifestEntry(task_id="done", category="x"),
            BenchmarkManifestEntry(task_id="retry", category="x"),
        ],
        variant="evo",
        output_dir=tmp_path,
        resume=True,
    )
    assert calls == ["retry"]


@pytest.mark.asyncio
async def test_benchmark_runner_resume_retries_infra_error(tmp_path: Path) -> None:
    calls: list[str] = []

    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del variant
        calls.append(entry.task_id)
        return _benchmark_metrics("passed")

    infra = BenchmarkResult(
        task_id="retry",
        variant="evo",
        status="infra_error",
        metrics=_benchmark_metrics("infra_error"),
    )
    done = BenchmarkResult(
        task_id="done",
        variant="evo",
        status="resolved",
        metrics=_benchmark_metrics("passed"),
    )
    (tmp_path / "runs.jsonl").write_text(
        done.model_dump_json() + "\n" + infra.model_dump_json() + "\n"
    )
    await BenchmarkRunner(run_task).run(
        [
            BenchmarkManifestEntry(task_id="done", category="x"),
            BenchmarkManifestEntry(task_id="retry", category="x"),
        ],
        variant="evo",
        output_dir=tmp_path,
        resume=True,
    )
    assert calls == ["retry"]


def test_balanced_manifest_generator_uses_truth_only_for_selection(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures/ci_tasks/tasks.jsonl"
    entries = generate_balanced_manifest(
        FixtureAdapter(fixture), tmp_path / "manifest.jsonl", per_category=1
    )
    assert len(entries) == 6
    assert sum(not entry.skipped for entry in entries) == 3
    assert all(entry.skip_reason for entry in entries if entry.skipped)


def test_ablation_features_are_incremental() -> None:
    assert not variant_features("single").multi_agent
    assert variant_features("multi").multi_agent
    assert variant_features("multi-memory").long_term_memory
    assert variant_features("evo").capabilities


@pytest.mark.asyncio
async def test_fixture_benchmark_metrics_come_from_actual_graph_run(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n"
    )
    (tmp_path / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import add\n"
        "class TestCalculator(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "failing fixture"], cwd=tmp_path, check=True)
    fixture = Path(__file__).parents[1] / "fixtures/ci_tasks/tasks.jsonl"
    task_view = FixtureAdapter(fixture).prepare_task("fixture-test")
    verifier = FailedCommandReplayVerifier()
    preflight = await verifier.preflight(task_view, tmp_path)
    assert preflight.status == "reproduced"
    recorder = TrajectoryRecorder()
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=DemoCoordinator(),
            worker=DemoInvestigator(),
        ),
        recorder=recorder,
    )
    run_id = "benchmark-graph"
    result = await build_graph(runtime).ainvoke(
        {
            "run_id": run_id,
            "task_id": "fixture-test",
            "repo": RepoSpec(owner="fixture", name="calculator"),
            "ci_failure": CIFailure(
                summary="addition failed",
                log_excerpt="AssertionError: -1 != 3",
                failed_commands=[["python", "-m", "unittest", "-q"]],
                task_family="Test Failure",
            ),
            "workspace_path": str(tmp_path),
        }
    )
    metrics = await collect_metrics(
        result=result,
        recorder=recorder,
        run_id=run_id,
        task=task_view,
        workspace=tmp_path,
        truth=GroundTruth(
            task_id="fixture-test",
            sha_success="evaluator-only",
            diff="evaluator-only",
            changed_files=["calculator.py"],
            error_type="Test Failure",
        ),
        wall_time=0.1,
        verifier=verifier,
        benchmark_preflight=preflight,
    )

    assert metrics.benchmark_resolved
    assert metrics.benchmark_verification_status == "passed"
    assert metrics.agent_declared_success
    assert metrics.targeted_verification_passed
    assert metrics.review_passed
    assert metrics.tool_calls >= 2
    assert metrics.workers_spawned >= 1
    assert metrics.parallel_rounds >= 1


def _prepared_task(commands: list[list[str]]) -> PreparedTask:
    return PreparedTask(
        agent_view=AgentTaskView(
            task_id="regression",
            repo_owner="fixture",
            repo_name="regression",
            workflow_name="ci",
            workflow_path=".github/workflows/ci.yml",
            sha_fail="HEAD",
            ci_failure=NormalizedCIFailure(
                workflow_yaml="name: CI\n",
                log_text="fixture failure",
                failed_steps=[],
                candidate_failed_commands=commands,
            ),
        )
    )


def _commit_fixture(workspace: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.test"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=workspace, check=True)


def _successful_harness_result() -> dict[str, object]:
    command = VerificationCommandResult(
        command=["python", "-c", "print('ok')"],
        exit_code=0,
        stdout="ok\n",
        stderr="",
    )
    verification = VerificationResult(passed=True, commands=[command])
    return {
        "status": "success",
        "verification": verification,
        "verification_history": [verification],
        "review": ReviewResult(accepted=True, confidence=1.0),
    }


@pytest.mark.asyncio
async def test_trajectory_file_claim_cannot_replace_final_workspace_changes(
    tmp_path: Path,
) -> None:
    (tmp_path / "gold.py").write_text("VALUE = 1\n")
    _commit_fixture(tmp_path)
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="staging-leak",
        event_type=EventType.TOOL_RESULT,
        agent_id="fixer",
        invocation_id="repair:1",
        event_key="staging-write",
        payload={
            "call_id": "staging-write",
            "tool_name": "apply_patch",
            "success": True,
            "created_files": ["gold.py"],
        },
    )

    metrics = await collect_metrics(
        result=_successful_harness_result(),
        recorder=recorder,
        run_id="staging-leak",
        task=_prepared_task([]),
        workspace=tmp_path,
        truth=GroundTruth(
            task_id="regression",
            sha_success="gold",
            diff="gold",
            changed_files=["gold.py"],
            error_type="Test Failure",
        ),
        wall_time=0.1,
    )

    assert metrics.attempted_files == ["gold.py"]
    assert metrics.final_workspace_changes.changed_files == []
    assert metrics.gold_file_overlap == 0.0
    assert metrics.benchmark_verification_status == "not_available"
    assert not metrics.benchmark_resolved


@pytest.mark.asyncio
async def test_agent_verification_pass_does_not_override_failed_benchmark_replay(
    tmp_path: Path,
) -> None:
    (tmp_path / "failing_test.py").write_text("raise SystemExit(1)\n")
    _commit_fixture(tmp_path)
    task_view = _prepared_task([["python", "failing_test.py"]])
    verifier = FailedCommandReplayVerifier()
    preflight = await verifier.preflight(task_view, tmp_path)
    assert preflight.status == "reproduced"

    metrics = await collect_metrics(
        result=_successful_harness_result(),
        recorder=TrajectoryRecorder(),
        run_id="independent-verifier",
        task=task_view,
        workspace=tmp_path,
        truth=GroundTruth(
            task_id="regression",
            sha_success="gold",
            diff="gold",
            changed_files=["failing_test.py"],
            error_type="Test Failure",
        ),
        wall_time=0.1,
        verifier=verifier,
        benchmark_preflight=preflight,
    )

    assert metrics.agent_declared_success
    assert metrics.targeted_verification_passed
    assert metrics.review_passed
    assert metrics.benchmark_verification_status == "failed"
    assert metrics.benchmark_verification.commands[0].exit_code == 1
    assert not metrics.benchmark_resolved


@pytest.mark.asyncio
async def test_preflight_passing_candidate_is_not_a_benchmark_oracle(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    _commit_fixture(tmp_path)
    task_view = _prepared_task([["python", "-c", "print('ok')"]])
    verifier = FailedCommandReplayVerifier()
    preflight = await verifier.preflight(task_view, tmp_path)

    metrics = await collect_metrics(
        result={"status": "failed", "verification_history": []},
        recorder=TrajectoryRecorder(),
        run_id="invalid-oracle",
        task=task_view,
        workspace=tmp_path,
        truth=GroundTruth(
            task_id="regression",
            sha_success="gold",
            diff="gold",
            changed_files=["app.py"],
            error_type="Test Failure",
        ),
        wall_time=0.1,
        verifier=verifier,
        benchmark_preflight=preflight,
    )

    assert preflight.status == "not_reproduced"
    assert metrics.benchmark_verification_status == "not_available"
    assert not metrics.benchmark_resolved
    assert metrics.final_workspace_changes.changed_files == []


def _metrics_with_tokens(status: str, input_tokens: int, output_tokens: int = 0) -> RunMetrics:
    return _benchmark_metrics(status).model_validate(
        {
            **_benchmark_metrics(status).model_dump(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "repair_input_tokens": input_tokens,
            "repair_output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "repair_tokens": input_tokens + output_tokens,
            "learning_tokens": 0,
        }
    )


def _benchmark_metrics(status: str) -> RunMetrics:
    verification = BenchmarkVerificationResult(
        status=status,  # type: ignore[arg-type]
        details=f"fixture {status}",
    )
    return RunMetrics(
        agent_declared_success=status == "passed",
        targeted_verification_passed=status == "passed",
        review_passed=status == "passed",
        benchmark_verification=verification,
        benchmark_verification_status=status,  # type: ignore[arg-type]
        benchmark_resolved=status == "passed",
        final_workspace_changes=FinalWorkspaceChanges(changed_files=[]),
        wall_time=0.1,
    )


@pytest.mark.asyncio
async def test_not_available_is_coverage_not_agent_failure(tmp_path: Path) -> None:
    statuses = iter(["passed", "failed", "not_available"])

    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del entry, variant
        return _benchmark_metrics(next(statuses))

    results = await BenchmarkRunner(run_task).run(  # type: ignore[arg-type]
        [BenchmarkManifestEntry(task_id=str(index), category="test") for index in range(3)],
        variant="multi",
        output_dir=tmp_path,
    )
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())

    assert [result.status for result in results] == [
        "resolved",
        "unresolved",
        "not_evaluable",
    ]
    assert aggregate["evaluable"] == 2
    assert aggregate["not_available"] == 1
    assert aggregate["evaluation_coverage"] == pytest.approx(2 / 3)
    assert aggregate["benchmark_success_rate"] == 0.5


@pytest.mark.asyncio
async def test_infra_error_is_not_counted_as_model_failure(tmp_path: Path) -> None:
    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del entry, variant
        return _benchmark_metrics("infra_error")

    results = await BenchmarkRunner(run_task).run(  # type: ignore[arg-type]
        [BenchmarkManifestEntry(task_id="one", category="test")],
        variant="multi",
        output_dir=tmp_path,
    )
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert results[0].status == "infra_error"
    assert aggregate["infra_errors"] == 1
    assert aggregate["evaluable"] == 0
    assert aggregate["benchmark_success_rate"] is None


@pytest.mark.asyncio
async def test_benchmark_task_error_isolated_and_reported(tmp_path: Path) -> None:
    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del variant
        if entry.task_id == "broken":
            raise ValueError("fixture task failed")
        return _benchmark_metrics("passed")

    results = await BenchmarkRunner(run_task).run(  # type: ignore[arg-type]
        [
            BenchmarkManifestEntry(task_id="broken", category="test"),
            BenchmarkManifestEntry(task_id="healthy", category="test"),
        ],
        variant="multi",
        output_dir=tmp_path,
    )

    assert results[0].status == "error"
    assert results[0].error_type == "ValueError"
    assert results[1].status == "resolved"
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 2


@pytest.mark.asyncio
async def test_benchmark_results_are_incremental_before_process_abort(
    tmp_path: Path,
) -> None:
    class AbortRun(BaseException):
        pass

    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del variant
        if entry.task_id == "abort":
            raise AbortRun()
        return _benchmark_metrics("passed")

    with pytest.raises(AbortRun):
        await BenchmarkRunner(run_task).run(  # type: ignore[arg-type]
            [
                BenchmarkManifestEntry(task_id="first", category="test"),
                BenchmarkManifestEntry(task_id="abort", category="test"),
            ],
            variant="multi",
            output_dir=tmp_path,
        )

    persisted = (tmp_path / "runs.jsonl").read_text().splitlines()
    assert len(persisted) == 1
    assert json.loads(persisted[0])["task_id"] == "first"


def _usage_event(
    recorder: TrajectoryRecorder,
    *,
    run_id: str,
    key: str,
    input_tokens: int,
    output_tokens: int,
    budget_scope: str = "repair",
    cached: int | None = None,
    reasoning: int | None = None,
    agent_id: str = "fixer",
) -> None:
    recorder.emit(
        run_id=run_id,
        event_type=EventType.MODEL_USAGE,
        agent_id=agent_id,
        invocation_id="repair:1",
        event_key=key,
        payload={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached,
            "reasoning_tokens": reasoning,
            "request_kind": "structured",
            "budget_scope": budget_scope,
        },
    )


def test_run_metrics_sum_all_model_usage_events() -> None:
    recorder = TrajectoryRecorder()
    _usage_event(recorder, run_id="r", key="u1", input_tokens=10, output_tokens=2)
    _usage_event(recorder, run_id="r", key="u2", input_tokens=20, output_tokens=4)
    metrics = token_metrics_from_events(recorder.events("r"))
    assert metrics["input_tokens"] == 30
    assert metrics["output_tokens"] == 6
    assert metrics["total_tokens"] == 36
    assert metrics["provider_requests"] == 2


def test_run_metrics_separate_repair_and_learning_tokens() -> None:
    recorder = TrajectoryRecorder()
    _usage_event(recorder, run_id="r", key="repair", input_tokens=100, output_tokens=10)
    _usage_event(
        recorder,
        run_id="r",
        key="learn",
        input_tokens=40,
        output_tokens=5,
        budget_scope="post_run",
        agent_id="skill-miner",
    )
    metrics = token_metrics_from_events(recorder.events("r"))
    assert metrics["repair_tokens"] == 110
    assert metrics["learning_tokens"] == 45
    assert metrics["total_tokens"] == 155


def test_run_metrics_fall_back_to_legacy_events_only_when_usage_events_absent() -> None:
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="legacy",
        event_type=EventType.MODEL_CALL,
        agent_id="fixer",
        event_key="tool-turn:1",
        payload={"input_tokens": 12, "output_tokens": 3},
    )
    recorder.emit(
        run_id="legacy",
        event_type=EventType.MODEL_CALL,
        agent_id="skill-miner",
        event_key="final",
        payload={"input_tokens": 8, "output_tokens": 2, "budget_scope": "post_run"},
    )
    fallback = token_metrics_from_events(recorder.events("legacy"))
    assert fallback["input_tokens"] == 20
    assert fallback["repair_tokens"] == 15
    assert fallback["learning_tokens"] == 10
    assert fallback["provider_requests"] == 0

    _usage_event(recorder, run_id="legacy", key="usage-1", input_tokens=100, output_tokens=1)
    usage_only = token_metrics_from_events(recorder.events("legacy"))
    assert usage_only["input_tokens"] == 100
    assert usage_only["total_tokens"] == 101
    assert usage_only["provider_requests"] == 1


def test_total_tokens_equal_repair_plus_learning() -> None:
    metrics = RunMetrics(
        agent_declared_success=True,
        targeted_verification_passed=True,
        review_passed=True,
        benchmark_verification=BenchmarkVerificationResult(status="passed", details="ok"),
        benchmark_verification_status="passed",
        benchmark_resolved=True,
        final_workspace_changes=FinalWorkspaceChanges(changed_files=[]),
        wall_time=0.1,
        input_tokens=150,
        output_tokens=20,
        repair_input_tokens=100,
        repair_output_tokens=10,
        learning_input_tokens=50,
        learning_output_tokens=10,
    )
    assert metrics.total_tokens == 170
    assert metrics.repair_tokens + metrics.learning_tokens == metrics.total_tokens


@pytest.mark.asyncio
async def test_tokens_per_resolved_task_includes_failed_task_spend(tmp_path: Path) -> None:
    payloads = iter(
        [
            _metrics_with_tokens("passed", 80, 20),
            _metrics_with_tokens("failed", 40, 10),
        ]
    )

    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del entry, variant
        return next(payloads)

    await BenchmarkRunner(run_task).run(  # type: ignore[arg-type]
        [
            BenchmarkManifestEntry(task_id="ok", category="test"),
            BenchmarkManifestEntry(task_id="bad", category="test"),
        ],
        variant="multi",
        output_dir=tmp_path,
    )
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["total_tokens"] == 150
    assert aggregate["benchmark_resolved"] == 1
    assert aggregate["tokens_per_resolved_task"] == 150


@pytest.mark.asyncio
async def test_median_tokens_per_task(tmp_path: Path) -> None:
    totals = iter([10, 30, 20])

    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del entry, variant
        total = next(totals)
        return _metrics_with_tokens("passed", total)

    await BenchmarkRunner(run_task).run(  # type: ignore[arg-type]
        [BenchmarkManifestEntry(task_id=str(i), category="test") for i in range(3)],
        variant="multi",
        output_dir=tmp_path,
    )
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["median_tokens_per_task"] == 20


@pytest.mark.asyncio
async def test_median_tokens_per_resolved_task_uses_only_official_passes(
    tmp_path: Path,
) -> None:
    payloads = iter(
        [
            ("passed", 10),
            ("failed", 999),
            ("passed", 30),
        ]
    )

    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del entry, variant
        status, total = next(payloads)
        return _metrics_with_tokens(status, total)

    await BenchmarkRunner(run_task).run(  # type: ignore[arg-type]
        [BenchmarkManifestEntry(task_id=str(i), category="test") for i in range(3)],
        variant="multi",
        output_dir=tmp_path,
    )
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["median_tokens_per_resolved_task"] == 20
    assert aggregate["median_tokens_per_task"] == 30


@pytest.mark.asyncio
async def test_zero_resolved_tasks_produce_null_efficiency_metrics(tmp_path: Path) -> None:
    async def run_task(entry: BenchmarkManifestEntry, variant: str) -> RunMetrics:
        del entry, variant
        return _metrics_with_tokens("failed", 40, 10)

    await BenchmarkRunner(run_task).run(  # type: ignore[arg-type]
        [BenchmarkManifestEntry(task_id="one", category="test")],
        variant="multi",
        output_dir=tmp_path,
    )
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())
    assert aggregate["tokens_per_resolved_task"] is None
    assert aggregate["median_tokens_per_resolved_task"] is None
    assert aggregate["total_tokens"] == 50
