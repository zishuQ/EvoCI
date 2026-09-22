from concurrent.futures import ThreadPoolExecutor

from evoci.runtime.budget import (
    RepairBudgetExhausted,
    RunBudgetManager,
    RunRepairBudget,
    action_call_capacity,
)


def test_parallel_workers_atomically_share_one_tool_budget() -> None:
    budget = RunRepairBudget(max_model_calls=10, max_tool_calls=5)

    def consume() -> bool:
        try:
            budget.consume_tool_call()
            return True
        except RepairBudgetExhausted:
            return False

    with ThreadPoolExecutor(max_workers=20) as executor:
        outcomes = list(executor.map(lambda _: consume(), range(40)))

    assert sum(outcomes) == 5
    assert budget.snapshot().tool_calls == 5


def test_run_budget_manager_does_not_track_token_hard_limits() -> None:
    manager = RunBudgetManager(max_model_calls=8, max_tool_calls=8)
    assert not hasattr(manager, "tokens_for_run")
    assert not hasattr(manager, "max_tokens")
    budget = manager.for_run("run")
    budget.consume_model_call()
    assert budget.remaining_model_calls() == 7


def test_action_call_capacity_reserves_one_finalization_call() -> None:
    assert action_call_capacity(max_actions=15, remaining_model_calls=10) == 9
    assert action_call_capacity(max_actions=15, remaining_model_calls=1) == 0
    assert action_call_capacity(
        max_actions=15, remaining_model_calls=100, task_model_calls=5
    ) == 4
