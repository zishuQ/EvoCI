"""Deterministic offline demonstration of the full repair graph."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

from evoci.agents.base import AgentContext, AgentSuite
from evoci.config import EvoCIConfig
from evoci.domain.models import (
    CIFailure,
    Diagnosis,
    EvidenceItem,
    FileEdit,
    FixerOutput,
    Hypothesis,
    InvestigationPlan,
    InvestigationTask,
    PatchProposal,
    RepoSpec,
    ReviewResult,
    VerificationResult,
    WorkerResult,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import SQLiteMemoryStore
from evoci.runtime.checkpoints import create_async_sqlite_checkpointer
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.run_store import SQLiteRunStore
from evoci.tools.policy import WorkerCapabilities


class DemoCoordinator:
    async def plan(
        self,
        *,
        context: AgentContext,
        evidence: list[EvidenceItem],
        round_number: int,
        remaining_task_budget: int,
    ) -> InvestigationPlan:
        del context, evidence, round_number, remaining_task_budget
        return InvestigationPlan(
            tasks=[
                InvestigationTask(
                    task_id="logs",
                    role="log",
                    objective="identify the failed assertion",
                    expected_evidence=["assertion"],
                    priority=1,
                ),
                InvestigationTask(
                    task_id="repository",
                    role="repository",
                    objective="inspect the calculator implementation",
                    expected_evidence=["source"],
                    priority=1,
                ),
                InvestigationTask(
                    task_id="test",
                    role="test",
                    objective="confirm the intended addition behavior",
                    expected_evidence=["test"],
                    priority=1,
                ),
            ],
            reasoning_summary="logs, implementation, and tests are independent evidence sources",
        )


class DemoInvestigator:
    async def run(
        self,
        *,
        task: InvestigationTask,
        context: AgentContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerResult:
        assert not capabilities.write_files
        await asyncio.sleep(0.02)
        if task.role == "repository":
            source = (Path(context.workspace_path) / "calculator.py").read_text()
            claim = "add subtracts its second operand"
            excerpt = source
            kind = "source_code"
        elif task.role == "test":
            claim = "the test expects add(1, 2) to equal 3"
            excerpt = (Path(context.workspace_path) / "test_calculator.py").read_text()
            kind = "test_result"
        else:
            claim = "CI reports that -1 did not equal 3"
            excerpt = context.failure.log_excerpt
            kind = "ci_log"
        evidence = EvidenceItem(
            id=f"demo-{task.task_id}",
            source_agent=task.task_id,
            kind=kind,  # type: ignore[arg-type]
            claim=claim,
            excerpt=excerpt,
            confidence=0.98,
        )
        return WorkerResult(task_id=task.task_id, summary=claim, evidence=[evidence])


class DemoDiagnoser:
    async def diagnose(self, *, context: AgentContext, evidence: list[EvidenceItem]) -> Diagnosis:
        del context
        return Diagnosis(
            primary=Hypothesis(
                root_cause="calculator.add uses subtraction instead of addition",
                evidence_ids=[item.id for item in evidence],
                confidence=0.99,
                affected_files=["calculator.py"],
                proposed_action="replace subtraction with addition",
            ),
            needs_more_evidence=False,
        )


class DemoFixer:
    async def propose(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        evidence: list[EvidenceItem],
        previous_verification: VerificationResult | None,
    ) -> FixerOutput:
        del context, diagnosis, evidence, previous_verification
        return FixerOutput(
            proposal=PatchProposal(
                summary="correct calculator addition",
                changed_files=["calculator.py"],
                risk="low",
                verification_plan=[["python", "-m", "unittest", "-q"]],
            ),
            edits=[
                FileEdit(
                    path="calculator.py",
                    content=("def add(left: int, right: int) -> int:\n    return left + right\n"),
                )
            ],
        )


class DemoReviewer:
    async def review(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        patch: FixerOutput,
        verification: VerificationResult,
    ) -> ReviewResult:
        del context, diagnosis, patch
        return ReviewResult(
            accepted=verification.passed,
            blockers=[] if verification.passed else ["verification failed"],
            confidence=1.0,
        )


async def run_repair_demo(project_root: Path) -> dict[str, Any]:
    config = EvoCIConfig.from_env(cwd=project_root)
    config.ensure_directories()
    workspace = project_root / ".evoci/demo-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n"
    )
    (workspace / "test_calculator.py").write_text(
        "import unittest\n\n"
        "from calculator import add\n\n\n"
        "class CalculatorTest(unittest.TestCase):\n"
        "    def test_adds_operands(self) -> None:\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )
    state_dir = config.state_dir
    event_store = SQLiteEventStore(state_dir / "events.sqlite")
    run_store = SQLiteRunStore(state_dir / "runs.sqlite")
    memory_store = SQLiteMemoryStore(state_dir / "memory.sqlite")
    checkpoint = await create_async_sqlite_checkpointer(state_dir / "checkpoints.sqlite")
    run_id = f"demo-{uuid4().hex[:10]}"
    runtime = GraphRuntime(
        config=config,
        agents=AgentSuite(
            coordinator=DemoCoordinator(),
            investigator=DemoInvestigator(),
            diagnoser=DemoDiagnoser(),
            fixer=DemoFixer(),
            reviewer=DemoReviewer(),
        ),
        event_store=event_store,
        run_store=run_store,
        memory_store=memory_store,
        memory_retriever=MemoryRetriever(
            memory_store, context_limit_chars=config.memory_context_limit_chars
        ),
    )
    graph = build_graph(runtime, checkpointer=checkpoint.saver)
    try:
        result = await graph.ainvoke(
            {
                "run_id": run_id,
                "task_id": "demo-calculator",
                "repo": RepoSpec(owner="evoci", name="demo"),
                "ci_failure": CIFailure(
                    summary="calculator addition test failed",
                    log_excerpt="AssertionError: -1 != 3",
                    failed_commands=[["python", "-m", "unittest", "-q"]],
                    task_family="test",
                ),
                "workspace_path": str(workspace),
            },
            {"configurable": {"thread_id": run_id}},
        )
        return result
    finally:
        await checkpoint.close()
        memory_store.close()
        run_store.close()
        event_store.close()
