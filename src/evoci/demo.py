"""Deterministic offline demonstration of the supervisor-worker repair graph."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from evoci.agents.base import AgentSuite, SupervisorContext, WorkerContext, WorkerRun
from evoci.config import EvoCIConfig
from evoci.domain.models import (
    CIFailure,
    EvidenceItem,
    FileEdit,
    RepoSpec,
    SupervisorDecision,
    WorkerExecutionResult,
    WorkerTask,
    parse_verification_plan,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import SQLiteMemoryStore
from evoci.runtime.checkpoints import create_async_sqlite_checkpointer
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.run_store import SQLiteRunStore
from evoci.tools.policy import WorkerCapabilities


class DemoSupervisor:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
        del context
        self.calls += 1
        if self.calls == 1:
            return SupervisorDecision(
                action="dispatch",
                reasoning_summary="Inspect the calculator implementation and failing test.",
                tasks=[
                    WorkerTask(
                        task_id="repository",
                        kind="investigate",
                        objective="inspect the calculator implementation",
                        acceptance_criteria=["locate the incorrect operator"],
                    )
                ],
            )
        if self.calls == 2:
            return SupervisorDecision(
                action="dispatch",
                reasoning_summary="Repair the addition operator.",
                tasks=[
                    WorkerTask(
                        task_id="repair-add",
                        kind="repair",
                        objective="fix addition",
                        acceptance_criteria=["python -c add works"],
                        write_scope=["calc.py", "calculator.py"],
                    )
                ],
            )
        return SupervisorDecision(
            action="stop",
            reasoning_summary="No further work after the serial investigate/repair pair.",
            stop_reason="demo completed its serial queue",
        )


class DemoWorker:
    async def execute(
        self,
        *,
        context: WorkerContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerRun:
        task = context.task
        if task.kind == "investigate":
            assert capabilities.write_files is False
            evidence = EvidenceItem(
                source_agent="repository",
                kind="source_code",
                claim="calc.add multiplies instead of adding",
                file_path="calc.py",
                confidence=0.95,
            )
            return WorkerRun(
                result=WorkerExecutionResult(
                    task_id=task.task_id,
                    status="completed",
                    summary="addition is implemented as multiplication",
                    evidence=[evidence],
                    base_revision=context.baseline_snapshot_id,
                    snapshot_id=context.baseline_snapshot_id,
                )
            )
        assert capabilities.write_files is True
        root = Path(context.workspace_path)
        target = "calc.py" if (root / "calc.py").exists() else "calculator.py"
        source = (root / target).read_text(encoding="utf-8")
        repaired = (
            source.replace("return a * b", "return a + b")
            .replace("return left - right", "return left + right")
        )
        return WorkerRun(
            result=WorkerExecutionResult(
                task_id=task.task_id,
                status="completed",
                summary="replace the incorrect arithmetic operator",
                changed_files=[target],
                base_revision=context.baseline_snapshot_id,
                snapshot_id=context.baseline_snapshot_id,
            ),
            edits=[FileEdit(path=target, content=repaired)],
            verification_plan=parse_verification_plan([["python", "-m", "unittest", "-q"]]),
        )


async def run_repair_demo(root: Path) -> dict[str, Any]:
    workspace = root / ".evoci" / "demo-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "calc.py").write_text("def add(a, b):\n    return a * b\n", encoding="utf-8")
    config = EvoCIConfig.from_env(cwd=root)
    event_store = SQLiteEventStore(config.state_dir / "events.sqlite")
    run_store = SQLiteRunStore(config.state_dir / "runs.sqlite")
    memory_store = SQLiteMemoryStore(config.state_dir / "memory.sqlite")
    checkpoint = await create_async_sqlite_checkpointer(config.state_dir / "checkpoints.sqlite")
    runtime = GraphRuntime(
        config=config,
        agents=AgentSuite(supervisor=DemoSupervisor(), worker=DemoWorker()),
        event_store=event_store,
        run_store=run_store,
        memory_store=memory_store,
        memory_retriever=MemoryRetriever(memory_store),
    )
    graph = build_graph(runtime, checkpointer=checkpoint.saver)
    initial = {
        "run_id": f"demo-{uuid4().hex[:8]}",
        "task_id": "demo-add",
        "repo": RepoSpec(name="demo"),
        "ci_failure": CIFailure(
            summary="add returns 6 for 2+3",
            log_excerpt="AssertionError",
            failed_commands=[["python", "-c", "from calc import add; assert add(2, 3) == 5"]],
        ),
        "workspace_path": str(workspace),
    }
    try:
        return await graph.ainvoke(initial, {"configurable": {"thread_id": initial["run_id"]}})
    finally:
        await checkpoint.close()
        memory_store.close()
        run_store.close()
        event_store.close()


def main() -> None:
    result = asyncio.run(run_repair_demo(Path.cwd()))
    print(result.get("status"))


if __name__ == "__main__":
    main()


DemoCoordinator = DemoSupervisor
DemoInvestigator = DemoWorker
DemoFixer = DemoWorker
DemoDiagnoser = DemoSupervisor
DemoReviewer = DemoSupervisor
