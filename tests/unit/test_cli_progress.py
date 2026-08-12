import json
from pathlib import Path

from evoci.cli import _load_task, _progress_line
from evoci.runtime.events import EventType


def test_progress_line_reports_model_and_tool_activity() -> None:
    announced: set[str] = set()
    assert _progress_line(EventType.MODEL_CALL, "coordinator", {}, announced) == (
        "正在制定调查计划..."
    )
    assert _progress_line(EventType.MODEL_CALL, "coordinator", {}, announced) is None
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
        "coordinator",
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
        "investigator:tests",
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
