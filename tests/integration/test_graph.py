from __future__ import annotations

import asyncio
import hashlib
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import evoci.graph.builder as graph_builder_module
from evoci.agents.base import AgentSuite, SupervisorContext, WorkerContext, WorkerRun
from evoci.agents.model_agents import ModelWorker, StagedFixerPlan
from evoci.capability.miner import SkillLearningDecision, SkillMiner
from evoci.capability.models import GeneratedFile, SkillCandidate, SkillPermissions
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.cli import official_learning_verdict
from evoci.config import EvoCIConfig
from evoci.domain.models import (
    CIFailure,
    Diagnosis,
    EvidenceItem,
    FileEdit,
    FixerOutput,
    Hypothesis,
    PatchProposal,
    RepoSpec,
    ReviewResult,
    SkillCatalogEntry,
    SkillRef,
    SupervisorDecision,
    VerificationResult,
    WorkerExecutionResult,
    WorkerTask,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.graph.outcome import persist_run_outcome
from evoci.memory.consolidation import ModelMemoryConsolidator
from evoci.memory.fingerprint import failure_fingerprint
from evoci.memory.models import LongTermFactCandidate
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import SQLiteMemoryStore
from evoci.model.gateway import ModelGatewayError, ModelUsage
from evoci.runtime.budget import RunBudgetManager
from evoci.runtime.checkpoints import create_async_sqlite_checkpointer
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.events import EventType
from evoci.runtime.run_store import SQLiteRunStore
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.policy import WorkerCapabilities
from tests.integration.test_tool_loop import CountingGateway, _tool_turn


def task(
    task_id: str, *, kind: str = "investigate", write_scope: list[str] | None = None
) -> WorkerTask:
    return WorkerTask(
        task_id=task_id,
        kind=kind,  # type: ignore[arg-type]
        objective=f"inspect {task_id}" if kind == "investigate" else f"repair {task_id}",
        acceptance_criteria=(
            ["collect evidence"] if kind == "investigate" else ["verification passes"]
        ),
        write_scope=write_scope or [],
    )


def _as_worker_task(item: WorkerTask | object) -> WorkerTask:
    if isinstance(item, WorkerTask):
        return item
    task_id = getattr(item, "task_id", str(item))
    return task(str(task_id))


class FakeCoordinator:
    def __init__(self, plans: list[list[object]]) -> None:
        self.plans = plans
        self.calls = 0
        self.seen_skill_counts: list[int] = []
        self.repair_task: WorkerTask | None = None
        self._dispatched_repair = False
        self._queue: list[WorkerTask] = []
        self._queued = False

    def _annotate(self, item: WorkerTask, context: SupervisorContext) -> WorkerTask:
        skill_refs = [SkillRef(skill_id=skill.skill_id) for skill in context.skills]
        fact_refs = [memory.memory_id for memory in context.memories]
        return item.model_copy(
            update={
                "recommended_skill_refs": list(item.recommended_skill_refs) or skill_refs,
                "fact_refs": list(item.fact_refs) or fact_refs,
            }
        )

    async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
        self.seen_skill_counts.append(len(context.skills))
        if context.last_verification is not None and context.last_verification.passed:
            return SupervisorDecision(
                action="stop",
                reasoning_summary="formal verification already passed",
                stop_reason="verified success",
            )
        self.calls += 1
        if not self._queued:
            for plan in self.plans:
                self._queue.extend(_as_worker_task(item) for item in plan)
            if self.repair_task is not None:
                self._queue.append(
                    self.repair_task.model_copy(update={"task_id": f"repair-{self.calls}"}),
                )
                self._dispatched_repair = True
            elif any(self.plans):
                self._queue.append(
                    task("repair", kind="repair", write_scope=["app.py", "a.py", "b.py"])
                )
                self._dispatched_repair = True
            self._queued = True
        nxt: WorkerTask | None = None
        if self._queue:
            nxt = self._queue.pop(0)
        if nxt is None:
            return SupervisorDecision(
                action="stop",
                reasoning_summary="coordinator produced no new investigation tasks",
                stop_reason="coordinator produced no new investigation tasks",
            )
        return SupervisorDecision(
            action="dispatch",
            reasoning_summary="fixture plan",
            tasks=[self._annotate(nxt, context)],
        )


class FakeInvestigator:
    def __init__(self, *, delay: float = 0, fail_once: str | None = None) -> None:
        self.delay = delay
        self.fail_once = fail_once
        self.calls: Counter[str] = Counter()

    async def execute(
        self,
        *,
        context: WorkerContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerRun:
        task = context.task
        if task.kind == "repair":
            return await FakeFixer().execute(context=context, capabilities=capabilities)
        assert capabilities.write_files is False
        self.calls[task.task_id] += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_once == task.task_id and self.calls[task.task_id] == 1:
            raise RuntimeError("simulated worker crash")
        evidence = EvidenceItem(
            id=f"evidence-{task.task_id}",
            source_agent=task.task_id,
            kind="source_code",
            claim=f"claim from {task.task_id}",
            confidence=0.9,
        )
        return WorkerRun(
            result=WorkerExecutionResult(
                task_id=task.task_id,
                status="completed",
                summary="done",
                evidence=[evidence],
                used_memory_ids=[memory.memory_id for memory in context.memories],
                used_skill_refs=[
                    SkillRef(skill_id=skill.skill_id) for skill in context.recommended_skills
                ],
                base_revision=context.baseline_snapshot_id,
                snapshot_id=context.baseline_snapshot_id,
            )
        )

    async def run(self, **kwargs: object) -> WorkerRun:
        return await self.execute(
            context=kwargs["context"],  # type: ignore[arg-type]
            capabilities=kwargs["capabilities"],  # type: ignore[arg-type]
        )


class FailingModelInvestigator(FakeInvestigator):
    async def execute(
        self,
        *,
        context: WorkerContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerRun:
        del context, capabilities
        raise ModelGatewayError("model call failed after 3 attempts: connection reset")


class SkillUsingInvestigator(FakeInvestigator):
    pass


class MemoryUsingInvestigator(FakeInvestigator):
    pass


class FakeMemoryConsolidator:
    async def propose(self, **kwargs: object) -> LongTermFactCandidate:
        del kwargs
        return LongTermFactCandidate(
            type="fact",
            content="test failed AssertionError fixture repair",
            confidence=0.9,
        )


class FakeDiagnoser:
    def __init__(self, *, low_first: bool = False) -> None:
        self.low_first = low_first
        self.calls = 0
        self.evidence_counts: list[int] = []


class FakeFixer:
    def __init__(self, *, edit: FileEdit | None = None, risk: str = "low") -> None:
        self.edit = edit
        self.risk = risk
        self.calls = 0

    async def execute(
        self,
        *,
        context: WorkerContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerRun:
        del capabilities
        self.calls += 1
        edits = [self.edit] if self.edit else []
        return WorkerRun(
            result=WorkerExecutionResult(
                task_id=context.task.task_id,
                status="completed",
                summary="fixture repair",
                changed_files=[edit.path for edit in edits],
                base_revision=context.baseline_snapshot_id,
                snapshot_id=context.baseline_snapshot_id,
            ),
            edits=edits,
            risk=self.risk,
            verification_plan=[["python", "-c", "raise SystemExit(0)"]],
        )


class CombinedWorker:
    def __init__(self, investigator: FakeInvestigator, fixer: FakeFixer) -> None:
        self.investigator = investigator
        self.fixer = fixer

    async def execute(
        self,
        *,
        context: WorkerContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerRun:
        if context.task.kind == "repair":
            execute = getattr(self.fixer, "execute", None)
            if execute is not None:
                return await execute(context=context, capabilities=capabilities)
            output = await self.fixer.propose(  # type: ignore[misc]
                context=context,
                diagnosis=None,
                evidence=[],
                previous_verification=None,
            )
            return WorkerRun(
                result=WorkerExecutionResult(
                    task_id=context.task.task_id,
                    status="completed",
                    summary=output.proposal.summary,
                    changed_files=output.proposal.changed_files,
                    base_revision=context.baseline_snapshot_id,
                    snapshot_id=context.baseline_snapshot_id,
                ),
                edits=list(output.edits),
                commands_run=list(output.proposal.commands_run),
                verification_plan=list(output.proposal.verification_plan),
                risk=output.proposal.risk,
            )
        return await self.investigator.execute(context=context, capabilities=capabilities)


def initial_state(workspace: Path, *, run_id: str = "run-1") -> dict[str, object]:
    return {
        "run_id": run_id,
        "task_id": "task-1",
        "repo": RepoSpec(name="fixture"),
        "ci_failure": CIFailure(
            summary="test failed",
            log_excerpt="AssertionError",
            failed_commands=[["python", "-c", "raise SystemExit(0)"]],
        ),
        "workspace_path": str(workspace),
    }


def make_runtime(
    tmp_path: Path,
    coordinator: FakeCoordinator,
    investigator: FakeInvestigator,
    diagnoser: FakeDiagnoser | None = None,
    fixer: FakeFixer | None = None,
) -> GraphRuntime:
    del diagnoser
    resolved_fixer = fixer or FakeFixer()
    if (
        hasattr(coordinator, "repair_task")
        and coordinator.repair_task is None
        and getattr(resolved_fixer, "edit", None) is not None
    ):
        write_scope = [resolved_fixer.edit.path, "a.py", "b.py", "app.py", "notes.txt"]
        coordinator.repair_task = WorkerTask(
            task_id="repair",
            kind="repair",
            objective="apply fixture repair",
            write_scope=list(dict.fromkeys(write_scope)),
        )
    return GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=coordinator,
            worker=CombinedWorker(investigator, resolved_fixer),
        ),
        recorder=TrajectoryRecorder(),
    )


@pytest.mark.asyncio
async def test_supervisor_timeout_gets_one_fresh_invocation(tmp_path: Path) -> None:
    class TimeoutThenStop:
        def __init__(self) -> None:
            self.invocation_ids: list[str] = []

        async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
            self.invocation_ids.append(context.invocation_id)
            if len(self.invocation_ids) == 1:
                try:
                    raise TimeoutError
                except TimeoutError as exc:
                    raise ModelGatewayError("model call failed") from exc
            return SupervisorDecision(
                action="stop",
                reasoning_summary="retry completed",
                stop_reason="no verified repair",
            )

    coordinator = TimeoutThenStop()
    runtime = make_runtime(
        tmp_path,
        coordinator,  # type: ignore[arg-type]
        FakeInvestigator(),
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="supervisor-timeout-retry")
    )

    assert result["status"] == "failed"
    assert result["supervisor_batch"] == 1
    assert coordinator.invocation_ids == [
        "supervise:1",
        "supervise:1:timeout-retry",
    ]


@pytest.mark.asyncio
async def test_workers_run_one_task_at_a_time(tmp_path: Path) -> None:
    inflight = 0
    peaks: list[int] = []
    order: list[str] = []

    class SerialInvestigator(FakeInvestigator):
        async def execute(self, **kwargs: object) -> WorkerRun:
            nonlocal inflight
            context = kwargs["context"]
            assert isinstance(context, WorkerContext)
            inflight += 1
            peaks.append(inflight)
            order.append(context.task.task_id)
            await asyncio.sleep(0.01)
            inflight -= 1
            return await super().execute(**kwargs)

    coordinator = FakeCoordinator([[task("a"), task("b")]])
    investigator = SerialInvestigator()
    graph = build_graph(
        make_runtime(tmp_path, coordinator, investigator, FakeDiagnoser()),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        initial_state(tmp_path), {"configurable": {"thread_id": "serial"}}
    )

    assert order[:2] == ["a", "b"]
    assert max(peaks) == 1
    assert result["status"] == "success"
    assert len(result["evidence"]) == 2


@pytest.mark.asyncio
async def test_multi_task_dispatch_is_rejected_and_replanned(tmp_path: Path) -> None:
    class MultiThenRepair:
        def __init__(self) -> None:
            self.calls = 0

        async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
            del context
            self.calls += 1
            if self.calls == 1:
                return SupervisorDecision.model_construct(
                    action="dispatch",
                    reasoning_summary="illegal batch",
                    tasks=[task("a"), task("b")],
                )
            return SupervisorDecision(
                action="dispatch",
                reasoning_summary="one repair",
                tasks=[
                    task("repair", kind="repair", write_scope=["app.py"]),
                ],
            )

    graph = build_graph(
        make_runtime(tmp_path, MultiThenRepair(), FakeInvestigator(), FakeDiagnoser()),
        checkpointer=InMemorySaver(),
    )
    result = await graph.ainvoke(
        initial_state(tmp_path), {"configurable": {"thread_id": "reject-batch"}}
    )
    assert result["status"] == "success"
    assert any("exactly one task" in item for item in result.get("decision_history", []))


@pytest.mark.asyncio
async def test_investigate_returns_to_supervisor_without_formal_ci(tmp_path: Path) -> None:
    class InvestigateOnly:
        async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
            if context.worker_result_summaries:
                return SupervisorDecision(
                    action="stop",
                    reasoning_summary="investigation complete",
                    stop_reason="need a repair task next",
                )
            return SupervisorDecision(
                action="dispatch",
                reasoning_summary="inspect only",
                tasks=[task("inspect")],
            )

    result = await build_graph(
        make_runtime(tmp_path, InvestigateOnly(), FakeInvestigator(), FakeDiagnoser())
    ).ainvoke(initial_state(tmp_path), {"configurable": {"thread_id": "inspect-only"}})
    assert result["status"] == "failed"
    assert result.get("verification") is None
    assert result["worker_results"][0].status == "completed"


@pytest.mark.asyncio
async def test_successful_parallel_writes_survive_worker_crash(tmp_path: Path) -> None:
    coordinator = FakeCoordinator([[task("a"), task("b")]])
    investigator = FakeInvestigator(fail_once="b")
    diagnoser = FakeDiagnoser()
    saver = InMemorySaver()
    graph = build_graph(
        make_runtime(tmp_path, coordinator, investigator, diagnoser), checkpointer=saver
    )
    invocation_config = {"configurable": {"thread_id": "crash"}}

    with pytest.raises(RuntimeError, match="simulated worker crash"):
        await graph.ainvoke(initial_state(tmp_path), invocation_config)

    result = await graph.ainvoke(None, invocation_config)
    assert result["status"] == "success"
    assert investigator.calls == Counter({"b": 2, "a": 1})


@pytest.mark.asyncio
async def test_model_gateway_failure_ends_run_without_graph_traceback(tmp_path: Path) -> None:
    runtime = make_runtime(
        tmp_path,
        FakeCoordinator([[task("connection-failure")]]),
        FailingModelInvestigator(),
        FakeDiagnoser(),
    )

    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    runtime = replace(runtime, memory_store=memory_store)
    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="model-failure"))

    assert result["status"] == "failed"
    assert result["failure_class"] == "model"
    assert "model call failed after 3 attempts" in str(result["failure_reason"])
    assert memory_store.get_episode("model-failure") is None
    memory_store.close()


@pytest.mark.asyncio
async def test_sqlite_checkpoint_resumes_after_runtime_rebuild(tmp_path: Path) -> None:
    coordinator = FakeCoordinator([[task("a"), task("b")]])
    investigator = FakeInvestigator(fail_once="b")
    diagnoser = FakeDiagnoser()
    runtime = make_runtime(tmp_path, coordinator, investigator, diagnoser)
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    first_handle = await create_async_sqlite_checkpointer(checkpoint_path)
    graph = build_graph(runtime, checkpointer=first_handle.saver)
    invocation_config = {"configurable": {"thread_id": "disk-crash"}}
    state = initial_state(tmp_path, run_id="disk-run")
    state["skill_catalog"] = [
        SkillCatalogEntry(
            skill_id="checkpoint-skill",
            name="Checkpoint skill",
            description="checkpoint replay fixture",
        )
    ]

    with pytest.raises(RuntimeError, match="simulated worker crash"):
        await graph.ainvoke(state, invocation_config)
    await first_handle.close()

    second_handle = await create_async_sqlite_checkpointer(checkpoint_path)
    rebuilt_graph = build_graph(runtime, checkpointer=second_handle.saver)
    result = await rebuilt_graph.ainvoke(None, invocation_config)
    await second_handle.close()

    assert result["status"] == "success"
    assert investigator.calls == Counter({"b": 2, "a": 1})
    assert isinstance(result["skill_catalog"][0], SkillCatalogEntry)
    assert result["skill_catalog"][0].skill_id == "checkpoint-skill"


@pytest.mark.asyncio
async def test_low_confidence_routes_to_additional_investigation(tmp_path: Path) -> None:
    coordinator = FakeCoordinator([[task("a")], [task("b")]])
    investigator = FakeInvestigator()
    diagnoser = FakeDiagnoser(low_first=True)
    graph = build_graph(make_runtime(tmp_path, coordinator, investigator, diagnoser))
    state = initial_state(tmp_path)
    state["ci_failure"] = CIFailure(
        summary="test failed",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "raise SystemExit(1)"]],
    )

    result = await graph.ainvoke(state)

    assert result["supervisor_batch"] >= 2
    assert investigator.calls["a"] >= 1
    assert investigator.calls["b"] >= 1
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_high_risk_patch_interrupts_before_write_and_resumes(tmp_path: Path) -> None:
    workflow = Path(".github/workflows/test.yml")
    edit = FileEdit(path=str(workflow), content="name: safe\n")
    coordinator = FakeCoordinator([[task("a")]])
    investigator = FakeInvestigator()
    diagnoser = FakeDiagnoser()
    runtime = make_runtime(
        tmp_path,
        coordinator,
        investigator,
        diagnoser,
        FakeFixer(edit=edit, risk="low"),
    )
    graph = build_graph(runtime, checkpointer=InMemorySaver())
    invocation_config = {"configurable": {"thread_id": "approval"}}

    interrupted = await graph.ainvoke(initial_state(tmp_path), invocation_config)
    assert "__interrupt__" in interrupted
    assert not (tmp_path / workflow).exists()

    result = await graph.ainvoke(Command(resume=True), invocation_config)
    assert result["status"] == "success"
    assert (tmp_path / workflow).read_text() == "name: safe\n"


class FakeSkillMiner:
    def should_mine(
        self,
        *,
        success: bool,
        failure_class: str | None = None,
        skills_used: object = None,
        tool_calls: int = 0,
        failed_attempts: int = 0,
        reusable_script_created: bool = False,
        repeated_pattern_detected: bool = False,
    ) -> bool:
        del (
            failure_class,
            skills_used,
            tool_calls,
            failed_attempts,
            reusable_script_created,
            repeated_pattern_detected,
        )
        return success

    async def decide(
        self,
        trajectory_summary: object,
        *,
        existing_skills: object = None,
        used_skill_ids: object = None,
        **kwargs: object,
    ) -> SkillLearningDecision:
        del trajectory_summary, existing_skills, used_skill_ids, kwargs
        skill_md = """---
name: assertion-repair
description: Repair arithmetic assertion failures
version: 1
---

# Purpose
Repair evidence-backed arithmetic assertion failures.
# When to Use
Use for test failed AssertionError reports.
# Procedure
Inspect implementation and expected behavior before applying the smallest fix.
# Pitfalls
Do not change or skip the test.
# Verification
Re-run the targeted test.
# Bundled Resources
No bundled files are required.
"""
        return SkillLearningDecision(
            action="new_skill",
            rationale="the repair procedure generalizes",
            candidate_skill=SkillCandidate(
                name="assertion-repair",
                description="test failed AssertionError arithmetic repair",
                triggers=["test failed", "AssertionError"],
                task_families=["unknown"],
                skill_md=skill_md,
                scripts=[
                    GeneratedFile(
                        path="scripts/inspect_assertion.py",
                        content="print('inspect assertion operands')\n",
                    )
                ],
                tests=[
                    GeneratedFile(
                        path="tests/test_inspector.py",
                        content=(
                            "import unittest\n"
                            "class TestInspector(unittest.TestCase):\n"
                            "    def test_available(self) -> None:\n"
                            "        self.assertTrue(True)\n"
                        ),
                    )
                ],
                source_run_ids=["learning-run"],
                confidence=0.9,
                permissions=SkillPermissions(execute=True),
            ),
        )


class FakeUpdateMiner(FakeSkillMiner):
    async def decide(
        self,
        trajectory_summary: object,
        *,
        existing_skills: object = None,
        used_skill_ids: object = None,
        **kwargs: object,
    ) -> SkillLearningDecision:
        del trajectory_summary, existing_skills, used_skill_ids, kwargs
        skill_md = """---
name: assertion-repair
description: Improved arithmetic assertion repair
version: 2
---

# Purpose
Repair arithmetic assertion failures.
# When to Use
Use for test failed AssertionError reports.
# Procedure
Inspect both implementation and edge cases before applying the smallest fix.
# Pitfalls
Do not change or skip tests.
# Verification
Run targeted and repository tests.
# Bundled Resources
No bundled files are required.
"""
        return SkillLearningDecision(
            action="update_skill",
            rationale="the existing procedure needs stronger verification",
            target_skill_id="assertion-repair",
            candidate_skill=SkillCandidate(
                name="assertion-repair-improved",
                description="Improved arithmetic assertion repair",
                triggers=["test failed", "AssertionError"],
                task_families=["unknown"],
                skill_md=skill_md,
                source_run_ids=["update-run"],
                confidence=0.9,
            ),
        )


@pytest.mark.asyncio
async def test_successful_run_mines_trial_skill_and_related_run_retrieves_it(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    coordinator_a = FakeCoordinator([[task("a")]])
    base_a = make_runtime(tmp_path, coordinator_a, FakeInvestigator(), FakeDiagnoser(), FakeFixer())
    runtime_a = replace(
        base_a,
        capability_registry=registry,
        skill_miner=FakeSkillMiner(),
    )
    result_a = await build_graph(runtime_a).ainvoke(initial_state(tmp_path, run_id="learning-run"))

    learned = registry.get("assertion-repair")
    assert result_a["candidate_skill_id"] == "assertion-repair"
    assert learned is not None
    assert learned.manifest.enabled

    coordinator_b = FakeCoordinator([[task("b")]])
    base_b = make_runtime(tmp_path, coordinator_b, FakeInvestigator(), FakeDiagnoser(), FakeFixer())
    runtime_b = replace(
        base_b,
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
    )
    result_b = await build_graph(runtime_b).ainvoke(initial_state(tmp_path, run_id="reuse-run"))

    assert result_b["status"] == "success"
    assert coordinator_b.seen_skill_counts
    assert coordinator_b.seen_skill_counts[0] == 1
    assert registry.stats("assertion-repair").retrieval_count == 1
    registry.close()


@pytest.mark.asyncio
async def test_failed_run_persists_negative_episode_with_reason(tmp_path: Path) -> None:
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[]]),
            FakeInvestigator(),
            FakeDiagnoser(),
        ),
        memory_store=memory_store,
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="failed-episode-run")
    )

    assert result["status"] == "failed"
    episode = memory_store.get_episode("failed-episode-run")
    assert episode is not None
    assert episode.success is False
    assert episode.failure_reason == "coordinator produced no new investigation tasks"
    assert episode.successful_fix_summary is None
    memory_store.close()


@pytest.mark.asyncio
async def test_selected_trial_skill_use_is_attributed_and_promoted(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    skill_md = """---
name: assertion-repair
description: Repair arithmetic assertion failures
version: 1
---

# Purpose
Repair arithmetic assertion failures.
# When to Use
Use for test failed AssertionError reports.
# Procedure
Inspect the implementation and apply the smallest fix.
# Pitfalls
Do not weaken tests.
# Verification
Run the failed test.
# Bundled Resources
No bundled resources.
"""
    created = registry.create_skill(
        SkillCandidate(
            name="assertion-repair",
            description="test failed AssertionError arithmetic repair",
            triggers=["test failed", "AssertionError"],
            task_families=["unknown"],
            skill_md=skill_md,
            source_run_ids=["source"],
            confidence=0.9,
        )
    )
    base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("uses-skill")]]),
        SkillUsingInvestigator(),
        FakeDiagnoser(),
    )
    runtime = replace(
        base,
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
    )

    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="promotion-run"))

    assert result["status"] == "success"
    stats = registry.stats(created.manifest.skill_id)
    assert stats.retrieval_count == 1
    assert stats.selected_count == 0
    assert stats.use_count == 1
    assert stats.success_count == 1
    current = registry.get(created.manifest.skill_id)
    assert current is not None and current.manifest.enabled
    registry.close()


@pytest.mark.asyncio
async def test_learning_update_creates_same_id_child_without_early_supersede(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    original_decision = await FakeSkillMiner().decide({})
    assert original_decision.candidate_skill is not None
    original = registry.create_skill(original_decision.candidate_skill)
    base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("update")]]),
        FakeInvestigator(),
        FakeDiagnoser(),
    )
    runtime = replace(
        base,
        capability_registry=registry,
        skill_miner=FakeUpdateMiner(),
    )

    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="update-run"))

    assert result["candidate_skill_id"] == original.manifest.skill_id
    current = registry.get(original.manifest.skill_id)
    assert current is not None
    assert "edge cases" in (Path(current.package_path) / "SKILL.md").read_text(encoding="utf-8")
    previous = Path(current.package_path).parent / "previous" / "SKILL.md"
    assert previous.is_file()
    registry.close()


@pytest.mark.asyncio
async def test_cross_run_memory_is_retrieved_selected_and_used(tmp_path: Path) -> None:
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    first_base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("learn-memory")]]),
        FakeInvestigator(),
        FakeDiagnoser(),
    )
    first = replace(
        first_base,
        memory_store=memory_store,
        memory_consolidator=FakeMemoryConsolidator(),
    )
    result_a = await build_graph(first).ainvoke(initial_state(tmp_path, run_id="memory-source-run"))
    assert result_a["status"] == "success"
    assert memory_store.get_episode("memory-source-run") is not None

    recorder = TrajectoryRecorder()
    second_base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("reuse-memory")]]),
        MemoryUsingInvestigator(),
        FakeDiagnoser(),
    )
    second = replace(
        second_base,
        memory_store=memory_store,
        memory_retriever=MemoryRetriever(memory_store),
        recorder=recorder,
    )
    result_b = await build_graph(second).ainvoke(initial_state(tmp_path, run_id="memory-reuse-run"))

    assert result_b["status"] == "success"
    selected = {
        memory_id
        for event in recorder.events("memory-reuse-run")
        if event.type == EventType.MEMORY_SELECTED
        for memory_id in event.payload["memory_ids"]
    }
    used = {
        memory_id
        for event in recorder.events("memory-reuse-run")
        if event.type == EventType.MEMORY_USED
        for memory_id in event.payload["memory_ids"]
    }
    assert selected
    assert used == selected
    memory_store.close()


class RetryFeedbackFixer:
    def __init__(self) -> None:
        self.contexts: list[object] = []
        self.previous_verifications: list[VerificationResult | None] = []

    async def propose(
        self,
        *,
        context: object,
        diagnosis: Diagnosis,
        evidence: list[EvidenceItem],
        previous_verification: VerificationResult | None,
    ) -> FixerOutput:
        del diagnosis, evidence
        self.contexts.append(context)
        self.previous_verifications.append(previous_verification)
        first = len(self.contexts) == 1
        path = "bad.py" if first else "good.py"
        content = "import pytest\npytest.skip('blocked')\n" if first else "ATTEMPT = 2\n"
        return FixerOutput(
            proposal=PatchProposal(
                summary=f"attempt {len(self.contexts) + 1}",
                changed_files=[path],
                risk="low",
                verification_plan=[["python", "-c", "raise SystemExit(0)"]],
            ),
            edits=[FileEdit(path=path, content=content)],
        )


class RejectFirstReviewer:
    def __init__(self) -> None:
        self.calls = 0

    async def review(
        self,
        *,
        context: object,
        diagnosis: Diagnosis,
        patch: FixerOutput,
        verification: VerificationResult,
    ) -> ReviewResult:
        del context, diagnosis, patch, verification
        self.calls += 1
        if self.calls == 1:
            return ReviewResult(
                accepted=False,
                blockers=["attempt A violates review policy"],
                confidence=1.0,
            )
        return ReviewResult(accepted=True, confidence=1.0)


@pytest.mark.asyncio
async def test_reviewer_rejection_rolls_back_and_feeds_next_fixer(tmp_path: Path) -> None:
    class RetryCoordinator:
        def __init__(self) -> None:
            self.contexts: list[SupervisorContext] = []

        async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
            self.contexts.append(context)
            return SupervisorDecision(
                action="dispatch",
                reasoning_summary="retry rejected candidate",
                tasks=[
                    task(
                        f"repair-{len(self.contexts)}",
                        kind="repair",
                        write_scope=["bad.py", "good.py"],
                    )
                ],
            )

    (tmp_path / "base.txt").write_text("baseline\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "test"],
        ["git", "config", "user.email", "test@example.invalid"],
        ["git", "add", "base.txt"],
        ["git", "commit", "-qm", "baseline"],
    ):
        subprocess.run(command, cwd=tmp_path, check=True)

    fixer = RetryFeedbackFixer()
    coordinator = RetryCoordinator()
    runtime = make_runtime(
        tmp_path,
        coordinator,  # type: ignore[arg-type]
        FakeInvestigator(),
        FakeDiagnoser(),
    )
    runtime = replace(
        runtime,
        agents=replace(
            runtime.agents,
            worker=CombinedWorker(FakeInvestigator(), fixer),  # type: ignore[arg-type]
        ),
    )

    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="review-retry"))

    assert (
        result["status"],
        result.get("failure_class"),
        result.get("failure_stage"),
        result.get("failure_reason"),
        len(coordinator.contexts),
        len(fixer.contexts),
    ) == ("success", None, None, None, 2, 2)
    assert len(coordinator.contexts) == 2
    assert coordinator.contexts[1].last_verification is None
    assert not (tmp_path / "bad.py").exists()
    assert (tmp_path / "good.py").read_text() == "ATTEMPT = 2\n"


class TwoFileCrashFixer:
    def __init__(self, expected_sha256: str) -> None:
        self.expected_sha256 = expected_sha256

    async def propose(
        self,
        *,
        context: object,
        diagnosis: Diagnosis,
        evidence: list[EvidenceItem],
        previous_verification: VerificationResult | None,
    ) -> FixerOutput:
        del context, diagnosis, evidence, previous_verification
        edits = [
            FileEdit(
                path=path,
                content="VALUE = 'desired'\n",
                expected_sha256=self.expected_sha256,
            )
            for path in ("a.py", "b.py")
        ]
        return FixerOutput(
            proposal=PatchProposal(
                summary="two-file crash fixture",
                changed_files=[edit.path for edit in edits],
                risk="low",
                verification_plan=[["python", "-c", "raise SystemExit(0)"]],
            ),
            edits=edits,
        )


@pytest.mark.asyncio
async def test_patch_apply_crash_resumes_after_partial_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = "VALUE = 'original'\n"
    for path in ("a.py", "b.py"):
        (tmp_path / path).write_text(original)
    fixer = TwoFileCrashFixer(hashlib.sha256(original.encode()).hexdigest())
    runtime = make_runtime(
        tmp_path,
        FakeCoordinator([[task("inspect")]]),
        FakeInvestigator(),
        FakeDiagnoser(),
    )
    runtime = replace(
        runtime,
        agents=replace(
            runtime.agents,
            worker=CombinedWorker(FakeInvestigator(), fixer),  # type: ignore[arg-type]
        ),
    )
    saver = InMemorySaver()
    graph = build_graph(runtime, checkpointer=saver)
    invocation_config = {"configurable": {"thread_id": "patch-crash"}}
    real_apply = graph_builder_module._apply_edit
    calls = 0

    class CrashAfterFirstEdit(BaseException):
        pass

    def crash_once(*args: object, **kwargs: object) -> object:
        nonlocal calls
        result = real_apply(*args, **kwargs)  # type: ignore[arg-type]
        calls += 1
        if calls == 1:
            raise CrashAfterFirstEdit()
        return result

    monkeypatch.setattr(graph_builder_module, "_apply_edit", crash_once)
    with pytest.raises(CrashAfterFirstEdit):
        await graph.ainvoke(initial_state(tmp_path, run_id="patch-crash"), invocation_config)
    assert (tmp_path / "a.py").read_text() == original
    assert (tmp_path / "b.py").read_text() == original

    result = await graph.ainvoke(None, invocation_config)

    assert result["status"] == "success"
    assert (tmp_path / "a.py").read_text() == "VALUE = 'desired'\n"
    assert (tmp_path / "b.py").read_text() == "VALUE = 'desired'\n"


class RaisingMemoryConsolidator:
    async def propose(self, **kwargs: object) -> LongTermFactCandidate:
        del kwargs
        raise RuntimeError("memory consolidation exploded")


class RaisingSkillMiner:
    def should_mine(self, **kwargs: object) -> bool:
        del kwargs
        return True

    async def decide(
        self,
        trajectory: object,
        *,
        existing_skills: object = None,
        used_skill_ids: object = None,
        **kwargs: object,
    ) -> SkillLearningDecision:
        del trajectory, existing_skills, used_skill_ids, kwargs
        raise RuntimeError("experience mining exploded")


@pytest.mark.asyncio
async def test_memory_learning_failure_cannot_flip_core_success(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("inspect")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
        ),
        memory_store=store,
        memory_consolidator=RaisingMemoryConsolidator(),
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="memory-learning-error")
    )

    assert result["status"] == "success"
    assert result["learning_errors"][0]["stage"] == "memory_consolidation"
    store.close()


@pytest.mark.asyncio
async def test_experience_mining_failure_cannot_flip_core_success(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("inspect")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
        ),
        capability_registry=registry,
        skill_miner=RaisingSkillMiner(),
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="experience-learning-error")
    )

    assert result["status"] == "success"
    assert result["learning_errors"][0]["stage"] == "skill_mining"
    registry.close()


class ExplicitlyFailingSkillInvestigator(FakeInvestigator):
    def __init__(self, recorder: TrajectoryRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    async def execute(
        self,
        *,
        context: WorkerContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerRun:
        result = await super().execute(context=context, capabilities=capabilities)
        skill = context.recommended_skills[0]
        self.recorder.emit(
            run_id=context.run_id,
            event_type=EventType.SKILL_USED,
            agent_id=f"worker:{context.task.task_id}",
            invocation_id=context.invocation_id,
            event_key="failing-script",
            payload={
                "skill_id": skill.skill_id,
                "resource": "scripts/failing.py",
                "success": False,
            },
        )
        return WorkerRun(
            result=result.result.model_copy(
                update={"used_skill_refs": [SkillRef(skill_id=skill.skill_id)]}
            ),
            edits=result.edits,
            commands_run=result.commands_run,
            verification_plan=result.verification_plan,
        )


@pytest.mark.asyncio
async def test_explicit_skill_failure_cannot_receive_success_credit(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    decision = await FakeSkillMiner().decide({})
    assert decision.candidate_skill is not None
    created = registry.create_skill(decision.candidate_skill)
    recorder = TrajectoryRecorder()
    base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("uses-failing-skill")]]),
        ExplicitlyFailingSkillInvestigator(recorder),
        FakeDiagnoser(),
    )
    runtime = replace(
        base,
        recorder=recorder,
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="failed-skill-successful-run")
    )

    assert result["status"] == "success"
    stats = registry.stats(created.manifest.skill_id)
    assert stats.success_count == 0
    assert stats.failure_count == 1
    current = registry.get(created.manifest.skill_id)
    assert current is not None and current.manifest.enabled
    traces = recorder.build_view(
        run_id="failed-skill-successful-run",
        verification_history=result["verification_history"],
        final_status="success",
        failure_reason=None,
    ).skill_use_traces
    assert any(trace.execution_success is False for trace in traces)
    registry.close()


@pytest.mark.asyncio
async def test_new_skill_collision_is_learning_conflict_not_orphan_version(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    decision = await FakeSkillMiner().decide({})
    assert decision.candidate_skill is not None
    registry.create_skill(decision.candidate_skill)
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("inspect")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
        ),
        capability_registry=registry,
        skill_miner=FakeSkillMiner(),
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="new-skill-collision")
    )

    assert result["status"] == "success"
    assert result["candidate_skill_id"] is None
    assert result["learning_errors"][0]["stage"] == "skill_candidate_creation"
    assert len(registry.list()) == 1
    registry.close()


def _campaign_state(round_number: int, task_id: str) -> dict[str, object]:
    run_id = f"campaign-demo-r{round_number}-{task_id}"
    return {
        "run_id": run_id,
        "task_id": task_id,
        "campaign_provenance": {
            "campaign_id": "demo",
            "round": round_number,
            "task_id": task_id,
            "run_id": run_id,
            "read_generation": round_number - 1,
        },
    }


def test_operation_key_is_scoped_to_run_id_across_campaign_rounds() -> None:
    round1_task_a = _campaign_state(1, "django-123")
    round1_task_b = _campaign_state(1, "flask-456")
    round2_task_a = _campaign_state(2, "django-123")

    first = graph_builder_module._operation_key(round1_task_a, "learning-memory")
    replayed = graph_builder_module._operation_key(round1_task_a, "learning-memory")
    assert first == replayed == "learning-memory:campaign-demo-r1-django-123"

    round2 = graph_builder_module._operation_key(round2_task_a, "learning-memory")
    assert round2 == "learning-memory:campaign-demo-r2-django-123"
    assert first != round2

    other_task = graph_builder_module._operation_key(round1_task_b, "learning-memory")
    assert other_task == "learning-memory:campaign-demo-r1-flask-456"
    assert first != other_task

    skill_v1 = graph_builder_module._operation_key(
        round1_task_a, "skill-use", "fix-django", "v1"
    )
    skill_v2 = graph_builder_module._operation_key(
        round1_task_a, "skill-use", "fix-django", "v2"
    )
    assert skill_v1 == "skill-use:campaign-demo-r1-django-123:fix-django:v1"
    assert skill_v2 == "skill-use:campaign-demo-r1-django-123:fix-django:v2"
    assert skill_v1 != skill_v2

    normal = {"run_id": "run-abc123", "task_id": "task-1"}
    normal_key = graph_builder_module._operation_key(normal, "memory-commit")
    normal_replay = graph_builder_module._operation_key(normal, "memory-commit")
    other_run = graph_builder_module._operation_key(
        {"run_id": "run-def456", "task_id": "task-1"}, "memory-commit"
    )
    assert normal_key == normal_replay == "memory-commit:run-abc123"
    assert normal_key != other_run


@pytest.mark.asyncio
async def test_unknown_runtime_bug_is_not_swallowed_as_model_failure(tmp_path: Path) -> None:
    graph = build_graph(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("a")]]),
            FakeInvestigator(fail_once="a"),
            FakeDiagnoser(),
        )
    )
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        await graph.ainvoke(initial_state(tmp_path, run_id="bug-run"))


@pytest.mark.asyncio
async def test_verified_patch_failure_writes_rich_episode(tmp_path: Path) -> None:
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 1}),
        memory_store=memory_store,
        recorder=TrajectoryRecorder(),
    )
    state = initial_state(tmp_path, run_id="rich-fail")
    state["repo"] = RepoSpec(owner="org", name="example")
    state["ci_failure"] = CIFailure(
        summary="tests/test_add.py::test_add failed AssertionError",
        log_excerpt="tests/test_add.py::test_add FAILED\nAssertionError: 1 != 2\n",
        failed_commands=[["python", "-c", "raise SystemExit(1)"]],
        task_family="test",
    )
    result = await build_graph(runtime).ainvoke(state)
    assert result["status"] == "failed"
    assert result["failure_class"] == "repair"
    episode = memory_store.get_episode("rich-fail")
    assert episode is not None
    assert episode.failure_fingerprint
    assert episode.attempted_files
    assert episode.verification_failures
    assert "VALUE = 2" not in str(episode.model_dump())
    content = memory_store.search_episodes("AssertionError", repo="org/example", limit=1)[0].content
    assert "OUTCOME=failed" in content
    assert "HYPOTHESIS (unverified)" in content
    assert "ATTEMPTED_FIXES" in content
    memory_store.close()


@pytest.mark.asyncio
async def test_baseline_environment_error_does_not_rollback_as_repair_failure(
    tmp_path: Path,
) -> None:
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 1}),
        memory_store=memory_store,
        recorder=TrajectoryRecorder(),
    )
    state = initial_state(tmp_path, run_id="infra-verify")
    state["ci_failure"] = CIFailure(
        summary="collection failed",
        log_excerpt="ERROR collecting tests",
        failed_commands=[
            ["python", "-c", "print('ERROR collecting tests'); raise SystemExit(2)"]
        ],
        task_family="test",
    )
    result = await build_graph(runtime).ainvoke(state)
    assert result["status"] == "failed"
    assert result["failure_class"] == "infrastructure"
    assert result["verification"].status == "infra_error"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert memory_store.get_episode("infra-verify") is None
    memory_store.close()


def _trial_skill(tmp_path: Path) -> tuple[CapabilityRegistry, str]:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    created = registry.create_skill(
        SkillCandidate(
            name="assertion-repair",
            description="test failed AssertionError arithmetic repair",
            triggers=["test failed", "AssertionError"],
            task_families=["unknown"],
            skill_md="""---
name: assertion-repair
description: Repair arithmetic assertion failures
version: 1
---

# Purpose
Repair arithmetic assertion failures.
# When to Use
Use for test failed AssertionError reports.
# Procedure
Inspect the implementation and apply the smallest fix.
# Pitfalls
Do not weaken tests.
# Verification
Run the failed test.
# Bundled Resources
No bundled resources.
""",
            source_run_ids=["source"],
            confidence=0.9,
        )
    )
    return registry, created.manifest.skill_id


def _infra_state(tmp_path: Path, run_id: str) -> dict[str, object]:
    state = initial_state(tmp_path, run_id=run_id)
    state["ci_failure"] = CIFailure(
        summary="test failed AssertionError collection failed",
        log_excerpt="ERROR collecting tests\nAssertionError",
        failed_commands=[
            ["python", "-c", "print('ERROR collecting tests'); raise SystemExit(2)"]
        ],
        task_family="unknown",
    )
    return state


@pytest.mark.asyncio
async def test_infrastructure_failure_does_not_update_skill_statistics(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    registry, skill_id = _trial_skill(tmp_path)
    before = registry.stats(skill_id)
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("uses-skill")]]),
            SkillUsingInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 2}),
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
        recorder=TrajectoryRecorder(),
    )
    result = await build_graph(runtime).ainvoke(_infra_state(tmp_path, "infra-skill-stats"))
    assert result["status"] == "failed"
    assert result["failure_class"] == "infrastructure"
    after = registry.stats(skill_id)
    assert after.retrieval_count >= 1
    assert after.selected_count == 0
    assert after.use_count == before.use_count == 0
    assert after.success_count == before.success_count == 0
    assert after.failure_count == before.failure_count == 0
    registry.close()


@pytest.mark.asyncio
async def test_model_failure_does_not_update_skill_statistics(tmp_path: Path) -> None:
    registry, skill_id = _trial_skill(tmp_path)
    before = registry.stats(skill_id)
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="model-skill-stats",
        event_type=EventType.SKILL_USED,
        payload={"skill_id": skill_id, "success": True},
    )
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("uses-skill")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
        ),
        capability_registry=registry,
        recorder=recorder,
    )
    state = initial_state(tmp_path, run_id="model-skill-stats")
    state["failure_class"] = "model"
    await persist_run_outcome(
        runtime,
        state,
        success=False,
        failure_reason="model call failed after 3 attempts",
        failure_class="model",
    )
    after = registry.stats(skill_id)
    assert after.use_count == before.use_count == 0
    assert after.success_count == before.success_count == 0
    assert after.failure_count == before.failure_count == 0
    memory = (tmp_path / "skills" / skill_id / "memory.md").read_text(encoding="utf-8")
    assert "model-skill-stats" not in memory
    registry.close()


@pytest.mark.asyncio
async def test_final_evaluator_pass_overrides_internal_inconclusive_for_learning(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    registry, skill_id = _trial_skill(tmp_path)
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("uses-skill")]]),
            SkillUsingInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 2}),
        memory_store=memory,
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
        recorder=TrajectoryRecorder(),
        defer_success_learning=True,
    )
    result = await build_graph(runtime).ainvoke(_infra_state(tmp_path, "infra-then-gold"))
    assert result["status"] == "failed"
    assert result["learning_deferred"] is True
    assert memory.get_episode("infra-then-gold") is None
    assert registry.stats(skill_id).use_count == 0
    assert official_learning_verdict(
        learning_deferred=True,
        benchmark_verification_status="passed",
        benchmark_resolved=True,
    ) == "success"
    learned = await persist_run_outcome(
        runtime, result, success=True, failure_reason=None, failure_class=None
    )
    assert learned["status"] == "success"
    episode = memory.get_episode("infra-then-gold")
    assert episode is not None
    assert episode.success is True
    stats = registry.stats(skill_id)
    assert stats.use_count == 1
    assert stats.success_count == 1
    assert stats.failure_count == 0
    memory.close()
    registry.close()


@pytest.mark.asyncio
async def test_final_evaluator_infra_error_finalizes_run_without_learning(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    run_store = SQLiteRunStore(tmp_path / "runs.sqlite")
    event_store = SQLiteEventStore(tmp_path / "events.sqlite")
    recorder = TrajectoryRecorder(event_store)
    registry, skill_id = _trial_skill(tmp_path)
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("uses-skill")]]),
            SkillUsingInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 2}),
        memory_store=memory,
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
        recorder=recorder,
        run_store=run_store,
        event_store=event_store,
        defer_success_learning=True,
    )
    result = await build_graph(runtime).ainvoke(_infra_state(tmp_path, "infra-then-infra"))
    assert result["learning_deferred"] is True
    assert official_learning_verdict(
        learning_deferred=True,
        benchmark_verification_status="infra_error",
    ) == "infrastructure"
    learned = await persist_run_outcome(
        runtime,
        result,
        success=False,
        failure_class="infrastructure",
        failure_reason="official evaluator infra_error",
    )
    assert learned["status"] == "failed"
    assert learned["failure_class"] == "infrastructure"
    record = run_store.get("infra-then-infra")
    assert record is not None
    assert record.status == "failed"
    events = recorder.events("infra-then-infra")
    assert any(event.type == EventType.RUN_FAILED for event in events)
    assert memory.get_episode("infra-then-infra") is None
    stats = registry.stats(skill_id)
    assert stats.use_count == 0
    assert stats.failure_count == 0
    memory.close()
    registry.close()
    run_store.close()
    event_store.close()


@pytest.mark.asyncio
async def test_final_evaluator_not_available_does_not_count_as_repair(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    run_store = SQLiteRunStore(tmp_path / "runs.sqlite")
    registry, skill_id = _trial_skill(tmp_path)
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("uses-skill")]]),
            SkillUsingInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 2}),
        memory_store=memory,
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
        recorder=TrajectoryRecorder(),
        run_store=run_store,
        defer_success_learning=True,
    )
    result = await build_graph(runtime).ainvoke(_infra_state(tmp_path, "infra-then-unavailable"))
    assert official_learning_verdict(
        learning_deferred=True,
        benchmark_verification_status="not_available",
    ) == "infrastructure"
    assert official_learning_verdict(
        learning_deferred=True,
        benchmark_verification_status="failed",
    ) == "repair"
    learned = await persist_run_outcome(
        runtime,
        result,
        success=False,
        failure_class="infrastructure",
        failure_reason="official evaluator not_available",
    )
    assert learned["status"] == "failed"
    assert learned["failure_class"] == "infrastructure"
    record = run_store.get("infra-then-unavailable")
    assert record is not None
    assert record.status == "failed"
    assert memory.get_episode("infra-then-unavailable") is None
    stats = registry.stats(skill_id)
    assert stats.use_count == 0
    assert stats.failure_count == 0
    memory.close()
    registry.close()
    run_store.close()


@pytest.mark.asyncio
async def test_verification_failure_preserves_stage_after_rollback(tmp_path: Path) -> None:
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 1}),
        memory_store=memory_store,
        recorder=TrajectoryRecorder(),
    )
    state = initial_state(tmp_path, run_id="stage-verify")
    state["ci_failure"] = CIFailure(
        summary="assertion failed",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "raise SystemExit(1)"]],
        task_family="test",
    )
    result = await build_graph(runtime).ainvoke(state)
    assert result["status"] == "failed"
    assert result["failure_stage"] == "verify"
    episode = memory_store.get_episode("stage-verify")
    assert episode is not None
    assert episode.failure_stage == "verify"
    memory_store.close()


@pytest.mark.asyncio
async def test_rollback_failure_records_infrastructure_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> tuple[list[str], list[str]]:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(graph_builder_module, "restore_batch_snapshot", boom)
    monkeypatch.setattr(graph_builder_module, "_restore_edit_baseline", boom)
    monkeypatch.setattr(graph_builder_module, "restore_attempt_writes", boom)
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 1}),
        memory_store=memory_store,
        recorder=TrajectoryRecorder(),
    )
    state = initial_state(tmp_path, run_id="stage-rollback")
    state["ci_failure"] = CIFailure(
        summary="assertion failed",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "raise SystemExit(1)"]],
        task_family="test",
    )
    result = await build_graph(runtime).ainvoke(state)
    assert result["status"] == "failed"
    assert result["failure_class"] == "infrastructure"
    assert result["failure_stage"] == "rollback"
    assert memory_store.get_episode("stage-rollback") is None
    memory_store.close()


@pytest.mark.asyncio
async def test_success_episode_has_no_failure_stage(tmp_path: Path) -> None:
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("a")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
        ),
        memory_store=memory_store,
    )
    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="success-stage"))
    assert result["status"] == "success"
    assert result.get("failure_stage") is None
    episode = memory_store.get_episode("success-stage")
    assert episode is not None
    assert episode.failure_stage is None
    memory_store.close()


@pytest.mark.asyncio
async def test_repeated_ci_failure_is_retrieved_by_fingerprint_not_task_id(
    tmp_path: Path,
) -> None:
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    repo = RepoSpec(owner="org", name="example")
    first_failure = CIFailure(
        summary="tests/test_add.py::test_add failed AssertionError",
        log_excerpt="tests/test_add.py::test_add FAILED\nAssertionError: 1 != 2\n",
        failed_commands=[["python", "-c", "raise SystemExit(1)"]],
        task_family="test",
    )
    first = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 1}),
        memory_store=memory_store,
        recorder=TrajectoryRecorder(),
    )
    state_a = initial_state(tmp_path, run_id="ci-run-1")
    state_a["task_id"] = "first-task"
    state_a["repo"] = repo
    state_a["ci_failure"] = first_failure
    result_a = await build_graph(first).ainvoke(state_a)
    assert result_a["status"] == "failed"
    episode = memory_store.get_episode("ci-run-1")
    assert episode is not None

    noisy = CIFailure(
        summary="tests/test_add.py::test_add failed AssertionError",
        log_excerpt=(
            "2026-09-20T08:11:00Z tests/test_add.py::test_add FAILED\n"
            "AssertionError: 1 != 2\n"
            "File /tmp/pytest-of-ci/pytest-99/workspace/src/unused.py, line 88, in add\n"
            "uuid=11111111-2222-3333-4444-555555555555\n"
        ),
        failed_commands=[["python", "-c", "raise SystemExit(1)"]],
        task_family="test",
    )

    assert failure_fingerprint(repo, first_failure) == failure_fingerprint(repo, noisy)
    second = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("repo")]]),
            MemoryUsingInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_supervisor_batches": 1}),
        memory_store=memory_store,
        memory_retriever=MemoryRetriever(memory_store),
        recorder=TrajectoryRecorder(),
    )
    state_b = initial_state(tmp_path, run_id="ci-run-2")
    state_b["task_id"] = "completely-different-task"
    state_b["repo"] = repo
    state_b["ci_failure"] = noisy
    result_b = await build_graph(second).ainvoke(state_b)
    contents = [hit.content for hit in result_b["retrieved_memories"]]
    assert any("OUTCOME=failed" in content for content in contents)
    assert any("counterevidence" not in content for content in contents)
    assert episode.failure_fingerprint == failure_fingerprint(repo, noisy)
    memory_store.close()


class _UsageCompleteGateway:
    async def complete(
        self,
        *,
        response_model: type[object],
        usage_observer: object | None = None,
        **kwargs: object,
    ) -> object:
        del kwargs
        if callable(usage_observer):
            usage_observer(
                ModelUsage(input_tokens=4, output_tokens=1, request_kind="structured")
            )
        if response_model is LongTermFactCandidate:
            return LongTermFactCandidate(type="none", confidence=0.1)
        if response_model is SkillLearningDecision:
            return SkillLearningDecision(action="none", rationale="nothing reusable")
        raise AssertionError(response_model)


def _learning_state(tmp_path: Path, run_id: str) -> dict[str, object]:
    state = initial_state(tmp_path, run_id=run_id)
    state["diagnosis"] = Diagnosis(
        primary=Hypothesis(
            root_cause="fixture cause",
            evidence_ids=[],
            confidence=0.95,
            affected_files=["app.py"],
            proposed_action="apply fixture repair",
        ),
        needs_more_evidence=False,
    )
    state["fixer_output"] = FixerOutput(
        proposal=PatchProposal(
            summary="fixture repair",
            changed_files=[],
            risk="low",
            verification_plan=[["python", "-c", "raise SystemExit(0)"]],
        ),
        edits=[],
    )
    state["verification"] = VerificationResult(passed=True, commands=[])
    return state


@pytest.mark.asyncio
async def test_skill_miner_usage_has_post_run_scope(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="miner-usage",
        event_type=EventType.TOOL_CALL,
        agent_id="fixer",
        payload={"tool_name": "read_file", "call_id": "c1"},
    )
    runtime = replace(
        make_runtime(tmp_path, FakeCoordinator([[]]), FakeInvestigator(), FakeDiagnoser()),
        recorder=recorder,
        capability_registry=CapabilityRegistry(tmp_path / "skills", tmp_path / "cap.sqlite"),
        skill_miner=SkillMiner(_UsageCompleteGateway()),  # type: ignore[arg-type]
    )
    await persist_run_outcome(
        runtime,
        _learning_state(tmp_path, "miner-usage"),
        success=True,
        failure_reason=None,
    )
    usages = [
        event for event in recorder.events("miner-usage") if event.type == EventType.MODEL_USAGE
    ]
    assert usages
    assert all(event.payload.get("budget_scope") == "post_run" for event in usages)
    assert all(event.agent_id == "skill-miner" for event in usages)
    runtime.capability_registry.close()


@pytest.mark.asyncio
async def test_memory_consolidator_usage_has_post_run_scope(tmp_path: Path) -> None:
    recorder = TrajectoryRecorder()
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    runtime = replace(
        make_runtime(tmp_path, FakeCoordinator([[]]), FakeInvestigator(), FakeDiagnoser()),
        recorder=recorder,
        memory_store=memory,
        memory_consolidator=ModelMemoryConsolidator(_UsageCompleteGateway()),  # type: ignore[arg-type]
    )
    await persist_run_outcome(
        runtime,
        _learning_state(tmp_path, "memory-usage"),
        success=True,
        failure_reason=None,
    )
    usages = [
        event for event in recorder.events("memory-usage") if event.type == EventType.MODEL_USAGE
    ]
    assert usages
    assert all(event.payload.get("budget_scope") == "post_run" for event in usages)
    assert all(event.agent_id == "memory-consolidator" for event in usages)
    memory.close()


def _repair_runtime(
    tmp_path: Path,
    gateway: CountingGateway,
    *,
    max_iterations: int = 5,
) -> GraphRuntime:
    recorder = TrajectoryRecorder()
    manager = RunBudgetManager(max_model_calls=32, max_tool_calls=32, recorder=recorder)
    worker = ModelWorker(
        gateway,
        recorder,
        max_iterations=max_iterations,
        max_tool_calls=16,
        budget_manager=manager,
    )
    coordinator = FakeCoordinator([])
    coordinator.repair_task = task("repair", kind="repair", write_scope=["app.py"])
    return GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(
            update={"max_supervisor_batches": 2}
        ),
        agents=AgentSuite(supervisor=coordinator, worker=worker),
        recorder=recorder,
        budget_manager=manager,
        memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
    )


@pytest.mark.asyncio
async def test_verified_recovered_candidate_can_finalize_success(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    gateway = CountingGateway(
        [
            _tool_turn("read", path="app.py"),
            _tool_turn("read2", path="app.py"),
            _tool_turn("read3", path="app.py"),
            _tool_turn("read4", path="app.py"),
            _tool_turn("patch", "apply_patch", files={"app.py": "VALUE = 1\n"}),
        ],
        StagedFixerPlan(summary="fixed"),
        fail_finalize=True,
    )
    runtime = _repair_runtime(tmp_path, gateway, max_iterations=5)
    state = initial_state(tmp_path, run_id="recovered-success")
    state["ci_failure"] = CIFailure(
        summary="VALUE is wrong",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "import app; assert app.VALUE == 1"]],
    )
    result = await build_graph(runtime).ainvoke(
        state, {"configurable": {"thread_id": "recovered-success"}}
    )
    assert gateway.next_action_calls == 5
    assert gateway.finalize_calls == 1
    assert result["status"] == "success"
    assert result["verification"].passed is True
    assert (tmp_path / "app.py").read_text() == "VALUE = 1\n"
    runtime.memory_store.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_invalid_structured_output_recovers_patch_for_formal_verification(
    tmp_path: Path,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    gateway = CountingGateway(
        [_tool_turn("patch", "apply_patch", files={"app.py": "VALUE = 1\n"})],
        StagedFixerPlan(summary="unused"),
        finalize_error=ModelGatewayError(
            "worker structured output failed after bounded correction: ValidationError"
        ),
    )
    runtime = _repair_runtime(tmp_path, gateway, max_iterations=1)
    state = initial_state(tmp_path, run_id="format-recover")
    state["ci_failure"] = CIFailure(
        summary="VALUE is wrong",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "import app; assert app.VALUE == 1"]],
    )
    result = await build_graph(runtime).ainvoke(
        state, {"configurable": {"thread_id": "format-recover"}}
    )
    assert gateway.finalize_calls == 1
    assert result["status"] == "success"
    assert result["verification"].passed is True
    assert (tmp_path / "app.py").read_text() == "VALUE = 1\n"
    runtime.memory_store.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_recovered_edit_still_requires_formal_verification(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    gateway = CountingGateway(
        [_tool_turn("patch", "apply_patch", files={"app.py": "VALUE = 1\n"})],
        StagedFixerPlan(summary="fixed"),
        fail_finalize=True,
    )
    runtime = _repair_runtime(tmp_path, gateway, max_iterations=1)
    state = initial_state(tmp_path, run_id="recovered-verify")
    state["ci_failure"] = CIFailure(
        summary="VALUE is wrong",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "import app; assert app.VALUE == 2"]],
    )
    result = await build_graph(runtime).ainvoke(
        state, {"configurable": {"thread_id": "recovered-verify"}}
    )
    assert result["status"] == "failed"
    assert result["verification"].passed is False
    assert (tmp_path / "app.py").read_text() == "VALUE = 0\n"
    runtime.memory_store.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_budget_exhaustion_uses_budget_failure_class(tmp_path: Path) -> None:
    class ExhaustedWorker:
        async def execute(self, **kwargs: object) -> WorkerRun:
            del kwargs
            return WorkerRun(
                result=WorkerExecutionResult(
                    task_id="repair",
                    status="budget_exhausted",
                    summary="task-level model-call budget exhausted",
                )
            )

    coordinator = FakeCoordinator([])
    coordinator.repair_task = task("repair", kind="repair", write_scope=["app.py"])
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(supervisor=coordinator, worker=ExhaustedWorker()),  # type: ignore[arg-type]
        recorder=TrajectoryRecorder(),
        memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
    )
    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="budget-class"))
    assert result["status"] == "failed"
    assert result["failure_class"] == "budget"
    assert result["failure_stage"] == "worker"
    runtime.memory_store.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_budget_failure_does_not_poison_skill_or_long_term_memory(
    tmp_path: Path,
) -> None:
    registry, skill_id = _trial_skill(tmp_path)
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    class ExhaustedWorker:
        async def execute(self, **kwargs: object) -> WorkerRun:
            del kwargs
            return WorkerRun(
                result=WorkerExecutionResult(
                    task_id="repair",
                    status="budget_exhausted",
                    summary="task-level model-call budget exhausted",
                    used_skill_refs=[SkillRef(skill_id=skill_id)],
                )
            )

    coordinator = FakeCoordinator([])
    coordinator.repair_task = task("repair", kind="repair", write_scope=["app.py"])
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(supervisor=coordinator, worker=ExhaustedWorker()),  # type: ignore[arg-type]
        recorder=TrajectoryRecorder(),
        memory_store=memory,
        memory_consolidator=FakeMemoryConsolidator(),
        capability_registry=registry,
        skill_miner=FakeSkillMiner(),
    )
    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="budget-nolearn"))
    assert result["status"] == "failed"
    assert result["failure_class"] == "budget"
    assert memory.list_long_term(repository="local/fixture") == []
    assert registry.stats(skill_id).use_count == 0
    assert registry.stats(skill_id).success_count == 0
    memory.close()
    registry.close()


@pytest.mark.asyncio
async def test_verify_budget_exhaustion_is_budget_not_repair(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    registry, skill_id = _trial_skill(tmp_path)
    memory_path = tmp_path / "skills" / skill_id / "memory.md"
    before_md = memory_path.read_bytes() if memory_path.is_file() else b""
    before_stats = registry.stats(skill_id)
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    recorder = TrajectoryRecorder()
    manager = RunBudgetManager(max_model_calls=64, max_tool_calls=1, recorder=recorder)
    coordinator = FakeCoordinator([])
    coordinator.repair_task = task("repair", kind="repair", write_scope=["app.py"])
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=coordinator,
            worker=CombinedWorker(
                FakeInvestigator(),
                FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 1\n")),
            ),
        ),
        recorder=recorder,
        budget_manager=manager,
        memory_store=memory,
        capability_registry=registry,
        skill_miner=FakeSkillMiner(),
        memory_consolidator=FakeMemoryConsolidator(),
    )
    state = initial_state(tmp_path, run_id="verify-budget")
    state["ci_failure"] = CIFailure(
        summary="VALUE is wrong",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "import app; assert app.VALUE == 1"]],
    )
    result = await build_graph(runtime).ainvoke(state)
    assert result["status"] == "failed"
    assert result["failure_class"] == "budget"
    assert result["failure_stage"] == "verify"
    assert result["verification"].status == "incomplete"
    assert result["verification"].incomplete_cause == "budget"
    assert result["verification"].commands[0].executed is False
    assert (tmp_path / "app.py").read_text() == "VALUE = 0\n"
    episode = memory.get_episode("verify-budget")
    assert episode is not None
    assert episode.success is False
    assert episode.failure_class == "budget"
    assert episode.failure_stage == "verify"
    assert memory.list_long_term(repository="local/fixture") == []
    after = registry.stats(skill_id)
    assert (
        after.use_count,
        after.success_count,
        after.failure_count,
        after.patch_count,
    ) == (
        before_stats.use_count,
        before_stats.success_count,
        before_stats.failure_count,
        before_stats.patch_count,
    )
    after_md = memory_path.read_bytes() if memory_path.is_file() else b""
    assert after_md == before_md
    assert result.get("fixer_output") is not None
    memory.close()
    registry.close()


@pytest.mark.asyncio
async def test_verify_second_command_budget_is_still_budget(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    recorder = TrajectoryRecorder()
    manager = RunBudgetManager(max_model_calls=64, max_tool_calls=2, recorder=recorder)
    coordinator = FakeCoordinator([])
    coordinator.repair_task = task("repair", kind="repair", write_scope=["app.py"])
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=coordinator,
            worker=CombinedWorker(
                FakeInvestigator(),
                FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 1\n")),
            ),
        ),
        recorder=recorder,
        budget_manager=manager,
        memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
    )
    state = initial_state(tmp_path, run_id="verify-budget-2")
    state["ci_failure"] = CIFailure(
        summary="VALUE is wrong",
        log_excerpt="AssertionError",
        failed_commands=[
            ["python", "-c", "import app; assert app.VALUE == 1"],
            ["python", "-c", "raise SystemExit(0)"],
        ],
    )
    result = await build_graph(runtime).ainvoke(state)
    assert result["failure_class"] == "budget"
    assert result["failure_stage"] == "verify"
    assert result["verification"].commands[0].executed is True
    assert result["verification"].commands[1].executed is False
    runtime.memory_store.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_verify_real_failure_stays_repair(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    coordinator = FakeCoordinator([])
    coordinator.repair_task = task("repair", kind="repair", write_scope=["app.py"])
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=coordinator,
            worker=CombinedWorker(
                FakeInvestigator(),
                FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 1\n")),
            ),
        ),
        recorder=TrajectoryRecorder(),
        budget_manager=RunBudgetManager(max_model_calls=64, max_tool_calls=32),
        memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
    )
    state = initial_state(tmp_path, run_id="verify-repair")
    state["ci_failure"] = CIFailure(
        summary="VALUE is wrong",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "import app; assert app.VALUE == 2"]],
    )
    result = await build_graph(runtime).ainvoke(state)
    assert result["status"] == "failed"
    assert result["failure_class"] == "repair"
    assert result["failure_stage"] == "verify"
    runtime.memory_store.close()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_procedure_and_script_use_count_once_and_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    registry, skill_id = _trial_skill(tmp_path)
    before = registry.stats(skill_id)
    recorder = TrajectoryRecorder()
    for kind, resource, success, key in (
        ("script", "inspect.py", True, "script"),
        ("procedure", None, None, "method-use"),
    ):
        recorder.emit(
            run_id="dual-use",
            event_type=EventType.SKILL_USED,
            agent_id="worker:repair",
            invocation_id="worker:1:repair",
            event_key=key,
            payload={
                "skill_id": skill_id,
                "usage_kind": kind,
                "resource": resource,
                "success": success,
            },
        )
    state = _learning_state(tmp_path, "dual-use")
    state["used_skill_refs"] = [SkillRef(skill_id=skill_id)]
    runtime = replace(
        make_runtime(tmp_path, FakeCoordinator([[]]), FakeInvestigator(), FakeDiagnoser()),
        capability_registry=registry,
        recorder=recorder,
        memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
    )
    await persist_run_outcome(runtime, state, success=True, failure_reason=None)
    mid = registry.stats(skill_id)
    assert mid.use_count == before.use_count + 1
    await persist_run_outcome(runtime, state, success=True, failure_reason=None)
    after = registry.stats(skill_id)
    assert after.use_count == mid.use_count
    traces = recorder.build_view(
        run_id="dual-use",
        verification_history=[],
        final_status="success",
        failure_reason=None,
    ).skill_use_traces
    assert {trace.usage_kind for trace in traces} == {"script", "procedure"}
    runtime.memory_store.close()  # type: ignore[union-attr]
    registry.close()


@pytest.mark.asyncio
async def test_script_failure_on_successful_run_writes_failure_memory(
    tmp_path: Path,
) -> None:
    registry, skill_id = _trial_skill(tmp_path)
    before = registry.stats(skill_id)
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="script-fail-success",
        event_type=EventType.SKILL_USED,
        agent_id="worker:repair",
        invocation_id="worker:1:repair",
        event_key="script",
        payload={
            "skill_id": skill_id,
            "usage_kind": "script",
            "resource": "inspect.py",
            "success": False,
        },
    )
    state = _learning_state(tmp_path, "script-fail-success")
    runtime = replace(
        make_runtime(tmp_path, FakeCoordinator([[]]), FakeInvestigator(), FakeDiagnoser()),
        capability_registry=registry,
        recorder=recorder,
        memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
    )
    await persist_run_outcome(runtime, state, success=True, failure_reason=None)
    stats = registry.stats(skill_id)
    assert stats.use_count == before.use_count + 1
    assert stats.failure_count == before.failure_count + 1
    assert stats.success_count == before.success_count
    memory_path = tmp_path / "skills" / skill_id / "memory.md"
    text = memory_path.read_text(encoding="utf-8")
    assert "Outcome: failure" in text
    assert "This skill was used during a successful repair." not in text
    assert "even though the overall repair succeeded" in text
    await persist_run_outcome(runtime, state, success=True, failure_reason=None)
    after = registry.stats(skill_id)
    assert after.use_count == stats.use_count
    assert after.failure_count == stats.failure_count
    assert memory_path.read_text(encoding="utf-8").count("Outcome: failure") == text.count(
        "Outcome: failure"
    )
    runtime.memory_store.close()  # type: ignore[union-attr]
    registry.close()


@pytest.mark.asyncio
async def test_repeated_script_executions_keep_each_result(tmp_path: Path) -> None:
    registry, skill_id = _trial_skill(tmp_path)
    recorder = TrajectoryRecorder()
    for key, success in (("call-1", True), ("call-2", False)):
        recorder.emit(
            run_id="repeat-script",
            event_type=EventType.SKILL_USED,
            agent_id="worker:repair",
            invocation_id="worker:1:repair",
            event_key=key,
            payload={
                "skill_id": skill_id,
                "usage_kind": "script",
                "resource": "inspect.py",
                "success": success,
            },
        )
    view = recorder.build_view(
        run_id="repeat-script",
        verification_history=[],
        final_status="success",
        failure_reason=None,
    )
    traces = [trace for trace in view.skill_use_traces if trace.skill_id == skill_id]
    assert len(traces) == 2
    assert {trace.execution_success for trace in traces} == {True, False}
    state = _learning_state(tmp_path, "repeat-script")
    before = registry.stats(skill_id)
    runtime = replace(
        make_runtime(tmp_path, FakeCoordinator([[]]), FakeInvestigator(), FakeDiagnoser()),
        capability_registry=registry,
        recorder=recorder,
        memory_store=SQLiteMemoryStore(tmp_path / "memory.sqlite"),
    )
    await persist_run_outcome(runtime, state, success=True, failure_reason=None)
    after = registry.stats(skill_id)
    assert after.use_count == before.use_count + 1
    assert after.failure_count == before.failure_count + 1
    assert after.success_count == before.success_count
    runtime.memory_store.close()  # type: ignore[union-attr]
    registry.close()


@pytest.mark.asyncio
async def test_verify_command_start_error_is_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    registry, skill_id = _trial_skill(tmp_path)
    memory_path = tmp_path / "skills" / skill_id / "memory.md"
    before_md = memory_path.read_bytes() if memory_path.is_file() else b""
    before_stats = registry.stats(skill_id)
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")

    async def boom(self: object, argv: object, **kwargs: object) -> object:
        del self, argv, kwargs
        raise OSError("verification runner exploded")

    monkeypatch.setattr("evoci.verification.service.CommandRunner.run", boom)
    coordinator = FakeCoordinator([])
    coordinator.repair_task = task("repair", kind="repair", write_scope=["app.py"])
    fixer = FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 1\n"))
    supervisor_calls = {"n": 0}
    original_decide = coordinator.decide

    async def counting_decide(*, context: SupervisorContext):
        supervisor_calls["n"] += 1
        return await original_decide(context=context)

    coordinator.decide = counting_decide  # type: ignore[method-assign]
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=coordinator,
            worker=CombinedWorker(FakeInvestigator(), fixer),
        ),
        recorder=TrajectoryRecorder(),
        budget_manager=RunBudgetManager(max_model_calls=64, max_tool_calls=32),
        memory_store=memory,
        capability_registry=registry,
        skill_miner=FakeSkillMiner(),
        memory_consolidator=FakeMemoryConsolidator(),
    )
    state = initial_state(tmp_path, run_id="verify-exec")
    state["ci_failure"] = CIFailure(
        summary="VALUE is wrong",
        log_excerpt="AssertionError",
        failed_commands=[["python", "-c", "import app; assert app.VALUE == 1"]],
    )
    result = await build_graph(runtime).ainvoke(state)
    assert result["status"] == "failed"
    assert result["failure_class"] == "infrastructure"
    assert result["failure_stage"] == "verify"
    assert result["verification"].status == "incomplete"
    assert result["verification"].incomplete_cause == "execution"
    assert result["verification"].commands[0].executed is False
    assert (tmp_path / "app.py").read_text() == "VALUE = 0\n"
    assert memory.get_episode("verify-exec") is None
    assert memory.list_long_term(repository="local/fixture") == []
    after = registry.stats(skill_id)
    assert (
        after.use_count,
        after.success_count,
        after.failure_count,
        after.patch_count,
    ) == (
        before_stats.use_count,
        before_stats.success_count,
        before_stats.failure_count,
        before_stats.patch_count,
    )
    after_md = memory_path.read_bytes() if memory_path.is_file() else b""
    assert after_md == before_md
    assert supervisor_calls["n"] == 1
    assert fixer.calls == 1
    assert result.get("fixer_output") is not None
    memory.close()
    registry.close()
