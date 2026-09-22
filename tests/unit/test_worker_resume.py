from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoci.agents.base import WorkerContext
from evoci.agents.model_agents import ModelWorker, WorkerReport
from evoci.domain.models import CIFailure, FileEdit, RepoSpec, TaskBudget, WorkerTask
from evoci.graph.integration import save_patch_artifact
from evoci.model.gateway import ModelGatewayError
from evoci.runtime.budget import RepairBudgetExhausted
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.isolation import workspace_snapshot_id
from evoci.tools.policy import WORKER_REPAIR_CAPABILITIES


def _context(
    workspace: Path,
    *,
    resume_id: str | None = None,
    resume_ref: str | None = None,
    write_scope: list[str] | None = None,
    budget: TaskBudget | None = None,
) -> WorkerContext:
    task = WorkerTask(
        task_id="repair",
        kind="repair",
        objective="resume candidate",
        write_scope=write_scope or ["a.py", "b.py"],
        resume_candidate_id=resume_id,
        budget=budget,
    )
    return WorkerContext(
        run_id="run-resume",
        repo=RepoSpec(name="fixture"),
        failure=CIFailure(summary="fail", log_excerpt="err"),
        workspace_path=str(workspace),
        invocation_id="worker:1:repair",
        task=task,
        baseline_snapshot_id=workspace_snapshot_id(workspace),
        resume_artifact_ref=resume_ref,
    )


def _worker(loop: object) -> ModelWorker:
    worker = ModelWorker(object(), TrajectoryRecorder())  # type: ignore[arg-type]
    worker.loop = loop  # type: ignore[assignment]
    return worker


@pytest.mark.asyncio
async def test_resume_keeps_task_prompt_and_unions_restored_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")
    (workspace / "b.py").write_text("old-b\n")
    art = tmp_path / "art"
    art.mkdir()
    artifact = save_patch_artifact(
        art,
        task=_context(workspace).task,
        edits=[FileEdit(path="a.py", content="new-a\n")],
        baseline_snapshot_id=workspace_snapshot_id(workspace),
        summary="previous a",
        commands_run=[],
        verification_plan=[],
    )
    captured: dict[str, object] = {}

    class Loop:
        async def run(self, **kwargs: object) -> WorkerReport:
            captured["prompt"] = kwargs["task_prompt"]
            tools = kwargs["tools"]
            tools.invoke("apply_patch", files={"b.py": "new-b\n"})
            return WorkerReport(summary="union")

    run = await _worker(Loop()).execute(
        context=_context(workspace, resume_id="cand-1", resume_ref=artifact),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    payload = json.loads(str(captured["prompt"]))
    assert payload["task"]["task_id"] == "repair"
    assert payload["resumed_candidate"]["changed_files"] == ["a.py"]
    assert "new-a" not in str(captured["prompt"])
    assert "edits" not in payload["resumed_candidate"]
    assert sorted(run.result.changed_files) == ["a.py", "b.py"]
    by_path = {edit.path: edit.content for edit in run.edits}
    assert by_path == {"a.py": "new-a\n", "b.py": "new-b\n"}


@pytest.mark.asyncio
async def test_resume_skips_restored_file_reverted_to_parent(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")
    art = tmp_path / "art"
    art.mkdir()
    artifact = save_patch_artifact(
        art,
        task=_context(workspace, write_scope=["a.py"]).task,
        edits=[FileEdit(path="a.py", content="new-a\n")],
        baseline_snapshot_id=workspace_snapshot_id(workspace),
        summary="previous a",
        commands_run=[],
        verification_plan=[],
    )

    class Loop:
        async def run(self, **kwargs: object) -> WorkerReport:
            tools = kwargs["tools"]
            tools.invoke("apply_patch", files={"a.py": "old-a\n"})
            return WorkerReport(summary="reverted")

    run = await _worker(Loop()).execute(
        context=_context(
            workspace, resume_id="cand-1", resume_ref=artifact, write_scope=["a.py"]
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.edits == []
    assert run.result.changed_files == []


@pytest.mark.asyncio
async def test_unknown_resume_candidate_is_blocked(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")
    run = await _worker(object()).execute(
        context=_context(workspace, resume_id="missing"),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.status == "blocked"
    assert "unknown failed candidate" in run.result.summary


@pytest.mark.asyncio
async def test_task_budget_is_passed_into_the_tool_loop(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")
    seen: dict[str, int] = {}

    class Loop:
        async def run(self, **kwargs: object) -> WorkerReport:
            extra = kwargs["extra_budget"]
            seen["max_model_calls"] = extra.max_model_calls
            seen["max_tool_calls"] = extra.max_tool_calls
            raise RepairBudgetExhausted("task-level model-call budget exhausted")

    run = await _worker(Loop()).execute(
        context=_context(
            workspace,
            write_scope=["a.py"],
            budget=TaskBudget(max_model_calls=5, max_tool_calls=2),
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert seen == {"max_model_calls": 5, "max_tool_calls": 2}
    assert run.result.status == "budget_exhausted"


@pytest.mark.asyncio
async def test_task_budget_of_one_call_does_not_start_the_tool_loop(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")
    called = False

    class Loop:
        async def run(self, **kwargs: object) -> WorkerReport:
            del kwargs
            nonlocal called
            called = True
            raise AssertionError("tool loop must not start")

    run = await _worker(Loop()).execute(
        context=_context(
            workspace,
            write_scope=["a.py"],
            budget=TaskBudget(max_model_calls=1, max_tool_calls=2),
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert called is False
    assert run.result.status == "budget_exhausted"


@pytest.mark.asyncio
async def test_worker_collects_edits_when_finalization_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")

    class Loop:
        async def run(self, **kwargs: object) -> WorkerReport:
            tools = kwargs["tools"]
            tools.invoke("apply_patch", files={"a.py": "new-a\n"})
            raise RepairBudgetExhausted("task-level model-call budget exhausted")

    run = await _worker(Loop()).execute(
        context=_context(workspace, write_scope=["a.py"]),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.status == "completed"
    assert run.edits[0].path == "a.py"
    assert run.edits[0].content == "new-a\n"
    assert (workspace / "a.py").read_text() == "old-a\n"


@pytest.mark.asyncio
async def test_worker_collects_edits_when_structured_output_is_invalid(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")

    class Loop:
        async def run(self, **kwargs: object) -> WorkerReport:
            tools = kwargs["tools"]
            tools.invoke("apply_patch", files={"a.py": "new-a\n"})
            raise ModelGatewayError(
                "worker structured output failed after bounded correction: ValidationError"
            )

    run = await _worker(Loop()).execute(
        context=_context(workspace, write_scope=["a.py"]),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.status == "completed"
    assert "structured finalization failed" in run.result.summary
    assert run.edits[0].content == "new-a\n"
    assert (workspace / "a.py").read_text() == "old-a\n"


@pytest.mark.asyncio
async def test_failed_finalization_without_edits_is_budget_exhausted(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")

    class Loop:
        async def run(self, **kwargs: object) -> WorkerReport:
            del kwargs
            raise RepairBudgetExhausted("task-level model-call budget exhausted")

    run = await _worker(Loop()).execute(
        context=_context(workspace, write_scope=["a.py"]),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.status == "budget_exhausted"
    assert run.edits == []


@pytest.mark.asyncio
async def test_out_of_scope_edit_is_not_recovered(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")

    class Loop:
        async def run(self, **kwargs: object) -> WorkerReport:
            tools = kwargs["tools"]
            staging = tools._tools["read_file"].function.__self__.boundary.root
            (staging / "secret.py").write_text("leaked\n")
            tools._changed_paths.add("secret.py")
            raise RepairBudgetExhausted("task-level model-call budget exhausted")

    run = await _worker(Loop()).execute(
        context=_context(workspace, write_scope=["a.py"]),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.status == "blocked"
    assert run.edits == []
    assert not (workspace / "secret.py").exists()


@pytest.mark.asyncio
async def test_trace_filters_by_invocation_id(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("old-a\n")
    recorder = TrajectoryRecorder()
    worker = ModelWorker(object(), recorder)  # type: ignore[arg-type]
    recorder.emit(
        run_id="run-resume",
        event_type=EventType.TOOL_CALL,
        agent_id="worker:repair",
        invocation_id="worker:1:repair",
        event_key="a",
        payload={
            "tool_name": "run_test",
            "call_id": "a",
            "arguments": {"argv": ["python", "-c", "print('a')"]},
        },
    )
    recorder.emit(
        run_id="run-resume",
        event_type=EventType.TOOL_RESULT,
        agent_id="worker:repair",
        invocation_id="worker:1:repair",
        event_key="a",
        payload={"tool_name": "run_test", "call_id": "a", "success": True, "exit_code": 0},
    )
    recorder.emit(
        run_id="run-resume",
        event_type=EventType.TOOL_CALL,
        agent_id="worker:repair",
        invocation_id="worker:2:repair",
        event_key="b",
        payload={
            "tool_name": "run_test",
            "call_id": "b",
            "arguments": {"argv": ["python", "-c", "print('b')"]},
        },
    )
    recorder.emit(
        run_id="run-resume",
        event_type=EventType.TOOL_RESULT,
        agent_id="worker:repair",
        invocation_id="worker:2:repair",
        event_key="b",
        payload={"tool_name": "run_test", "call_id": "b", "success": True, "exit_code": 0},
    )
    commands, _, _, plan = worker._trace_from_events(
        "run-resume", "worker:repair", "worker:1:repair"
    )
    assert any("print" in item and "a" in item for item in commands)
    assert not any("print" in item and "b" in item for item in commands)
    assert [spec.argv[-1] for spec in plan] == ["print('a')"]
