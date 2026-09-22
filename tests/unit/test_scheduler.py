import pytest

from evoci.domain.models import SupervisorDecision, TaskBudget, WorkerTask
from evoci.graph.scheduler import TaskScheduleError, validate_single_task


def _task(
    task_id: str,
    *,
    kind: str = "investigate",
    write_scope: list[str] | None = None,
    budget: TaskBudget | None = None,
) -> WorkerTask:
    return WorkerTask(
        task_id=task_id,
        kind=kind,  # type: ignore[arg-type]
        objective=task_id,
        write_scope=write_scope or [],
        budget=budget,
    )


def test_validate_single_task_rejects_multiple_tasks() -> None:
    with pytest.raises(TaskScheduleError, match="exactly one task"):
        validate_single_task(
            [_task("a"), _task("b")],
            known_task_ids=set(),
            remaining_model_calls=10,
            remaining_tool_calls=10,
        )


def test_validate_single_task_rejects_empty_repair_scope() -> None:
    with pytest.raises(TaskScheduleError, match="write_scope"):
        validate_single_task(
            [_task("a", kind="repair")],
            known_task_ids=set(),
            remaining_model_calls=10,
            remaining_tool_calls=10,
        )


def test_validate_single_task_rejects_duplicate_id() -> None:
    with pytest.raises(TaskScheduleError, match="already used"):
        validate_single_task(
            [_task("a")],
            known_task_ids={"a"},
            remaining_model_calls=10,
            remaining_tool_calls=10,
        )


def test_validate_single_task_rejects_budget_above_remaining() -> None:
    with pytest.raises(TaskScheduleError, match="model-call budget"):
        validate_single_task(
            [_task("a", budget=TaskBudget(max_model_calls=5, max_tool_calls=1))],
            known_task_ids=set(),
            remaining_model_calls=2,
            remaining_tool_calls=10,
        )


def test_supervisor_decision_rejects_multi_task_dispatch() -> None:
    with pytest.raises(ValueError, match="exactly one task"):
        SupervisorDecision(
            action="dispatch",
            reasoning_summary="too many",
            tasks=[_task("a"), _task("b")],
        )


def test_worker_task_rejects_depends_on() -> None:
    with pytest.raises(ValueError, match="depends_on"):
        WorkerTask(
            task_id="a",
            kind="investigate",
            objective="a",
            depends_on=["b"],
        )
