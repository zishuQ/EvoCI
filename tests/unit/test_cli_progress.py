import json
from pathlib import Path

from evoci.benchmark.models import (
    BenchmarkResult,
    BenchmarkVerificationResult,
    FinalWorkspaceChanges,
    RunMetrics,
)
from evoci.cli import _load_task, _progress_line, _round_summary_line, _task_completion_line
from evoci.runtime.events import EventType


def test_progress_line_reports_model_and_tool_activity() -> None:
    announced: set[str] = set()
    assert _progress_line(EventType.MODEL_CALL, "supervisor", {}, announced) == (
        "正在规划调查或修复任务..."
    )
    assert _progress_line(EventType.MODEL_CALL, "supervisor", {}, announced) is None
    assert _progress_line(
        EventType.TOOL_RESULT, "investigator:tests", {"success": False}, announced
    ) == "复现命令失败: `命令`。命令未成功"


def test_progress_line_reports_patch_and_verification() -> None:
    assert _progress_line(
        EventType.PATCH_CREATED,
        "fixer",
        {"fixer_output": {"proposal": {"changed_files": ["calculator.py"]}}},
    ) == "修复方案: 未提供说明 (文件: calculator.py)"
    assert _progress_line(
        EventType.VERIFICATION_COMPLETED,
        None,
        {"verification": {"passed": True}},
    ) == "验证结果: 通过"


def test_progress_line_shows_plan_command_failure_and_key_evidence() -> None:
    announced: set[str] = set()
    plan = _progress_line(
        EventType.AGENT_COMPLETED,
        "supervisor",
        {
            "tasks": [
                {
                    "role": "test",
                    "objective": "Reproduce the reported unittest failure.",
                },
                {"role": "repository", "objective": "Compare code and test contracts."},
            ]
        },
        announced,
    )
    assert plan is not None
    assert "[test] Reproduce the reported unittest failure." in plan
    assert "[repository] Compare code and test contracts." in plan

    failure_payload = {
        "success": False,
        "exit_code": 1,
        "error": "AssertionError: 0.25 != 25.0\nFAILED (failures=1)",
        "result": {"argv": ["python", "-m", "unittest", "-q"]},
    }
    failure = _progress_line(
        EventType.TOOL_RESULT, "investigator:tests", failure_payload, announced
    )
    assert failure == (
        "复现命令失败: `python -m unittest -q`, 退出码 1。"
        "AssertionError: 0.25 != 25.0"
    )
    assert (
        _progress_line(EventType.TOOL_RESULT, "investigator:other", failure_payload, announced)
        is None
    )

    completion = _progress_line(
        EventType.AGENT_COMPLETED,
        "worker:tests",
        {"evidence_count": 3},
        evidence=[
            {
                "kind": "source_code",
                "confidence": 1.0,
                "file_path": "calculator.py",
                "claim": "The implementation returns value / total without multiplying by 100.",
            },
            {
                "kind": "test_result",
                "confidence": 1.0,
                "file_path": "test_calculator.py",
                "claim": "The test fails because 0.25 is not equal to 25.0.",
            },
            {"kind": "runtime", "confidence": 1.0, "claim": "Runtime returns 0.25."},
        ],
    )
    assert completion is not None
    assert "test_calculator.py: The test fails because 0.25 is not equal to 25.0." in completion
    assert "另有 1 条支持证据已记录。" in completion


def test_loading_the_same_task_twice_creates_independent_runs(tmp_path: Path) -> None:
    task_file = tmp_path / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "run_id": "legacy-run-id-that-must-not-be-reused",
                "workspace_path": str(tmp_path),
                "repo": {"name": "fixture"},
                "ci_failure": {"summary": "test failed", "log_excerpt": "AssertionError"},
            }
        )
    )

    first = _load_task(task_file, "same-task")
    second = _load_task(task_file, "same-task")

    assert first["run_id"] != second["run_id"]
    assert first["run_id"] != "legacy-run-id-that-must-not-be-reused"
    assert second["run_id"] != "legacy-run-id-that-must-not-be-reused"


def _cli_metrics(status: str, total_tokens: int) -> RunMetrics:
    verification = BenchmarkVerificationResult(status=status, details=status)  # type: ignore[arg-type]
    return RunMetrics(
        agent_declared_success=status == "passed",
        targeted_verification_passed=status == "passed",
        review_passed=status == "passed",
        benchmark_verification=verification,
        benchmark_verification_status=status,  # type: ignore[arg-type]
        benchmark_resolved=status == "passed",
        final_workspace_changes=FinalWorkspaceChanges(changed_files=[]),
        wall_time=0.1,
        input_tokens=total_tokens,
        output_tokens=0,
        total_tokens=total_tokens,
        repair_input_tokens=total_tokens,
        repair_output_tokens=0,
        repair_tokens=total_tokens,
    )


def _cli_result(
    task_id: str,
    *,
    status: str | None = None,
    total_tokens: int = 0,
    skipped: bool = False,
) -> BenchmarkResult:
    if skipped:
        return BenchmarkResult(
            task_id=task_id,
            variant="evo",
            status="skipped",
            skipped=True,
            skip_reason="fixture",
        )
    assert status is not None
    metrics = _cli_metrics(status, total_tokens)
    task_status = {
        "passed": "resolved",
        "failed": "unresolved",
        "not_available": "not_evaluable",
        "infra_error": "infra_error",
    }[status]
    return BenchmarkResult(
        task_id=task_id,
        variant="evo",
        status=task_status,  # type: ignore[arg-type]
        metrics=metrics,
    )


def test_task_completion_line_includes_status_and_formatted_tokens() -> None:
    line = _task_completion_line("quixbugs-rpn_eval", _cli_metrics("passed", 18426))
    assert line == "Task quixbugs-rpn_eval: passed, tokens: 18,426"
    failed = _task_completion_line("django__django-15789", _cli_metrics("failed", 54103))
    assert failed == "Task django__django-15789: failed, tokens: 54,103"


def test_round_summary_line_counts_passed_failed_other_and_tokens() -> None:
    results = [
        _cli_result("ok-a", status="passed", total_tokens=10000),
        _cli_result("ok-b", status="passed", total_tokens=8426),
        _cli_result("bad-a", status="failed", total_tokens=20000),
        _cli_result("bad-b", status="failed", total_tokens=30000),
        _cli_result("infra", status="infra_error", total_tokens=1234),
        _cli_result("missing", status="not_available", total_tokens=100),
        _cli_result("skipped", skipped=True),
    ]
    assert _round_summary_line(results, 1) == (
        "Round 1 summary: passed=2, failed=2, total_tokens=69,760, other=3"
    )
    assert _round_summary_line(results[:4], None) == (
        "Benchmark summary: passed=2, failed=2, total_tokens=68,426"
    )
