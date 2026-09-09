"""Distinguish a wall-clock threshold failure from missing parallel execution."""

import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from evoci.graph.builder import build_graph
from tests.integration.test_graph import (
    FakeCoordinator,
    FakeDiagnoser,
    FakeInvestigator,
    initial_state,
    make_runtime,
    task,
)


@pytest.mark.asyncio
async def test_three_investigators_overlap_and_join(tmp_path):
    started = set()
    barrier = asyncio.Event()

    class BarrierInvestigator(FakeInvestigator):
        async def run(self, **kwargs):
            started.add(kwargs["task"].task_id)
            if len(started) == 3:
                barrier.set()
            # Serial execution cannot release this barrier.
            await asyncio.wait_for(barrier.wait(), timeout=3)
            return await super().run(**kwargs)

    diagnoser = FakeDiagnoser()
    runtime = make_runtime(
        tmp_path,
        FakeCoordinator([[task("a"), task("b"), task("c")]]),
        BarrierInvestigator(),
        diagnoser,
    )
    graph = build_graph(runtime, checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        initial_state(tmp_path), {"configurable": {"thread_id": "parallel-barrier"}}
    )
    assert started == {"a", "b", "c"}
    assert result["status"] == "success"
    assert diagnoser.evidence_counts == [3]
