from concurrent.futures import ThreadPoolExecutor

from evoci.runtime.budget import RepairBudgetExhausted, RunRepairBudget


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
