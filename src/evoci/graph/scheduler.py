"""Harness checks for a single dispatched Worker task."""

from __future__ import annotations

from evoci.domain.models import WorkerTask
from evoci.tools.scope import normalize_write_scope


class TaskScheduleError(ValueError):
    """A supervisor dispatch violated a harness scheduling constraint."""


def normalize_task(task: WorkerTask) -> WorkerTask:
    write_scope = list(normalize_write_scope(task.write_scope))
    read_scope = list(normalize_write_scope(task.read_scope)) if task.read_scope else []
    return task.model_copy(update={"write_scope": write_scope, "read_scope": read_scope})


def validate_single_task(
    tasks: list[WorkerTask],
    *,
    known_task_ids: set[str],
    remaining_model_calls: int,
    remaining_tool_calls: int,
) -> WorkerTask:
    if len(tasks) != 1:
        raise TaskScheduleError("dispatch requires exactly one task")
    task = normalize_task(tasks[0])
    if task.task_id in known_task_ids:
        raise TaskScheduleError(f"task_id already used in this run: {task.task_id}")
    if task.depends_on:
        raise TaskScheduleError("depends_on is not supported")
    if task.kind == "repair" and not task.write_scope:
        raise TaskScheduleError(f"repair task {task.task_id} must declare exact write_scope paths")
    if task.kind == "investigate" and task.write_scope:
        raise TaskScheduleError(f"investigate task {task.task_id} cannot write")
    budget = task.budget
    if budget is not None:
        if budget.max_model_calls > remaining_model_calls:
            raise TaskScheduleError(f"task {task.task_id} exceeds remaining model-call budget")
        if budget.max_tool_calls > remaining_tool_calls:
            raise TaskScheduleError(f"task {task.task_id} exceeds remaining tool-call budget")
    return task
