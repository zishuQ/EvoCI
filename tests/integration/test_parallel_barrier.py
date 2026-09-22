"""Serial Worker execution: the previous parallel barrier is no longer a production path."""

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
async def test_two_queued_workers_never_overlap(tmp_path):
    started: list[str] = []
    inflight = 0
    peaks: list[int] = []

    class SerialInvestigator(FakeInvestigator):
        async def execute(self, **kwargs):
            nonlocal inflight
            started.append(kwargs["context"].task.task_id)
            inflight += 1
            peaks.append(inflight)
            await asyncio.sleep(0.02)
            inflight -= 1
            return await super().execute(**kwargs)

    runtime = make_runtime(
        tmp_path,
        FakeCoordinator([[task("a"), task("b")]]),
        SerialInvestigator(),
        FakeDiagnoser(),
    )
    graph = build_graph(runtime, checkpointer=InMemorySaver())
    result = await graph.ainvoke(
        initial_state(tmp_path), {"configurable": {"thread_id": "serial-barrier"}}
    )
    assert started[:2] == ["a", "b"]
    assert max(peaks) == 1
    assert result["status"] == "success"
