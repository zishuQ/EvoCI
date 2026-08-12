"""Atomic run-level repair budgets shared by all current-task workers."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder


class RepairBudgetExhausted(RuntimeError):
    pass


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
    ) -> None:
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls
        self._model_calls = model_calls
        self._tool_calls = tool_calls
        self._lock = Lock()

    def consume_model_call(self) -> None:
        with self._lock:
            if self._model_calls >= self.max_model_calls:
                raise RepairBudgetExhausted("run-level model-call budget exhausted")
            self._model_calls += 1

    def consume_tool_call(self) -> None:
        with self._lock:
            if self._tool_calls >= self.max_tool_calls:
                raise RepairBudgetExhausted("run-level tool-call budget exhausted")
            self._tool_calls += 1

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                model_calls=self._model_calls,
                tool_calls=self._tool_calls,
                max_model_calls=self.max_model_calls,
                max_tool_calls=self.max_tool_calls,
            )


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
