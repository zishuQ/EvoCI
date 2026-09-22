"""Atomic run-level repair budgets shared by sequential supervisor and worker calls."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder


class RepairBudgetExhausted(RuntimeError):
    pass


FINALIZATION_MODEL_CALLS = 1


def action_call_capacity(
    *,
    max_actions: int,
    remaining_model_calls: int,
    task_model_calls: int | None = None,
) -> int:
    """How many next_action turns fit after reserving one structured finalize call."""

    if remaining_model_calls < 1 + FINALIZATION_MODEL_CALLS:
        return 0
    capacity = min(max_actions, remaining_model_calls - FINALIZATION_MODEL_CALLS)
    if task_model_calls is not None:
        capacity = min(capacity, max(0, task_model_calls - FINALIZATION_MODEL_CALLS))
    return max(0, capacity)


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    model_calls: int
    tool_calls: int
    max_model_calls: int
    max_tool_calls: int


class RunRepairBudget:
    def __init__(
        self,
        *,
        max_model_calls: int,
        max_tool_calls: int,
        model_calls: int = 0,
        tool_calls: int = 0,
        exhausted_message: str = "run-level model-call budget exhausted",
    ) -> None:
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self._model_calls = model_calls
        self._tool_calls = tool_calls
        self._model_message = exhausted_message
        self._tool_message = exhausted_message.replace("model-call", "tool-call")
        self._lock = Lock()

    def consume_model_call(self) -> None:
        with self._lock:
            if self._model_calls >= self.max_model_calls:
                raise RepairBudgetExhausted(self._model_message)
            self._model_calls += 1

    def consume_tool_call(self) -> None:
        with self._lock:
            if self._tool_calls >= self.max_tool_calls:
                raise RepairBudgetExhausted(self._tool_message)
            self._tool_calls += 1

    def remaining_model_calls(self) -> int:
        with self._lock:
            return max(0, self.max_model_calls - self._model_calls)

    def remaining_tool_calls(self) -> int:
        with self._lock:
            return max(0, self.max_tool_calls - self._tool_calls)

    def ensure_tool_calls(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            if self._tool_calls + count > self.max_tool_calls:
                raise RepairBudgetExhausted(self._tool_message)

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                model_calls=self._model_calls,
                tool_calls=self._tool_calls,
                max_model_calls=self.max_model_calls,
                max_tool_calls=self.max_tool_calls,
            )


class CombinedBudget:
    """Consume run-level and task-level call budgets together."""

    def __init__(self, *budgets: RunRepairBudget) -> None:
        self._budgets = budgets

    def consume_model_call(self) -> None:
        for budget in self._budgets:
            budget.consume_model_call()

    def consume_tool_call(self) -> None:
        for budget in self._budgets:
            budget.consume_tool_call()


class RunBudgetManager:
    def __init__(
        self,
        *,
        max_model_calls: int,
        max_tool_calls: int,
        recorder: TrajectoryRecorder | None = None,
    ) -> None:
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self.recorder = recorder
        self._budgets: dict[str, RunRepairBudget] = {}
        self._lock = Lock()

    def for_run(self, run_id: str) -> RunRepairBudget:
        with self._lock:
            existing = self._budgets.get(run_id)
            if existing is not None:
                return existing
            model_calls = 0
            tool_calls = 0
            if self.recorder is not None:
                events = self.recorder.events(run_id)
                model_calls = sum(
                    event.type == EventType.MODEL_CALL
                    and event.payload.get("budget_scope") != "post_run"
                    for event in events
                )
                tool_calls = sum(
                    event.type == EventType.TOOL_CALL
                    and event.payload.get("budget_scope") not in {"post_run", "cleanup"}
                    for event in events
                )
            budget = RunRepairBudget(
                max_model_calls=self.max_model_calls,
                max_tool_calls=self.max_tool_calls,
                model_calls=model_calls,
                tool_calls=tool_calls,
            )
            self._budgets[run_id] = budget
            return budget
