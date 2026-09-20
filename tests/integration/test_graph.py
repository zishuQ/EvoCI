from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import evoci.graph.builder as graph_builder_module
from evoci.agents.base import AgentContext, AgentSuite
from evoci.capability.miner import LearningDecision
from evoci.capability.models import GeneratedFile, SkillCandidate, SkillPermissions
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.capability.validator import CandidateValidator
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
    SkillRef,
    VerificationResult,
    WorkerResult,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.memory.fingerprint import failure_fingerprint
from evoci.memory.models import MemoryCandidate
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import SQLiteMemoryStore
from evoci.model.gateway import ModelGatewayError
from evoci.runtime.checkpoints import create_async_sqlite_checkpointer
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.policy import WorkerCapabilities


def task(task_id: str) -> InvestigationTask:
    return InvestigationTask(
        task_id=task_id,
        role="repository",
        objective=f"inspect {task_id}",
        expected_evidence=["source"],
        priority=1,
    )


class FakeCoordinator:
    def __init__(self, plans: list[list[InvestigationTask]]) -> None:
        self.plans = plans
        self.calls = 0
        self.seen_skill_counts: list[int] = []

    async def plan(
        self,
        *,
        context: AgentContext,
        evidence: list[EvidenceItem],
        round_number: int,
        remaining_task_budget: int,
    ) -> InvestigationPlan:
        self.seen_skill_counts.append(len(context.skills))
        del evidence, round_number, remaining_task_budget
        index = min(self.calls, len(self.plans) - 1)
        self.calls += 1
        return InvestigationPlan(tasks=self.plans[index], reasoning_summary="fixture plan")


class FakeInvestigator:
    def __init__(self, *, delay: float = 0, fail_once: str | None = None) -> None:
        self.delay = delay
        self.fail_once = fail_once
        self.calls: Counter[str] = Counter()

    async def run(
        self,
        *,
        task: InvestigationTask,
        context: AgentContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerResult:
        del context
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
        return WorkerResult(task_id=task.task_id, summary="done", evidence=[evidence])


class FailingModelInvestigator(FakeInvestigator):
    async def run(
        self,
        *,
        task: InvestigationTask,
        context: AgentContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerResult:
        del task, context, capabilities
        raise ModelGatewayError("model call failed after 3 attempts: connection reset")


class SkillUsingInvestigator(FakeInvestigator):
    async def run(
        self,
        *,
        task: InvestigationTask,
        context: AgentContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerResult:
        base = await super().run(task=task, context=context, capabilities=capabilities)
        return base.model_copy(
            update={
                "used_skill_refs": [
                    SkillRef(skill_id=skill.skill_id, version=skill.version)
                    for skill in context.skills
                ]
            }
        )


class MemoryUsingInvestigator(FakeInvestigator):
    async def run(
        self,
        *,
        task: InvestigationTask,
        context: AgentContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerResult:
        base = await super().run(task=task, context=context, capabilities=capabilities)
        return base.model_copy(
            update={"used_memory_ids": [memory.memory_id for memory in context.memories]}
        )


class FakeMemoryConsolidator:
    async def propose(self, **kwargs: object) -> MemoryCandidate:
        del kwargs
        return MemoryCandidate(
            type="semantic",
            content="test failed AssertionError fixture repair",
            namespace="repo:local/fixture",
            confidence=0.9,
        )


class FakeDiagnoser:
    def __init__(self, *, low_first: bool = False) -> None:
        self.low_first = low_first
        self.calls = 0
        self.evidence_counts: list[int] = []

    async def diagnose(self, *, context: AgentContext, evidence: list[EvidenceItem]) -> Diagnosis:
        del context
        self.calls += 1
        self.evidence_counts.append(len(evidence))
        low = self.low_first and self.calls == 1
        return Diagnosis(
            primary=Hypothesis(
                root_cause="fixture cause",
                evidence_ids=[item.id for item in evidence],
                confidence=0.5 if low else 0.95,
                affected_files=["app.py"],
                proposed_action="apply fixture repair",
            ),
            needs_more_evidence=low,
            missing_evidence=["more"] if low else [],
        )


class FakeFixer:
    def __init__(self, *, edit: FileEdit | None = None, risk: str = "low") -> None:
        self.edit = edit
        self.risk = risk
        self.calls = 0

    async def propose(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        evidence: list[EvidenceItem],
        previous_verification: VerificationResult | None,
    ) -> FixerOutput:
        del context, diagnosis, evidence, previous_verification
        self.calls += 1
        edits = [self.edit] if self.edit else []
        paths = [self.edit.path] if self.edit else []
        return FixerOutput(
            proposal=PatchProposal(
                summary="fixture repair",
                changed_files=paths,
                risk=self.risk,  # type: ignore[arg-type]
                verification_plan=[["python", "-c", "raise SystemExit(0)"]],
            ),
            edits=edits,
        )


class FakeReviewer:
    async def review(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        patch: FixerOutput,
        verification: VerificationResult,
    ) -> ReviewResult:
        del context, diagnosis, patch
        assert verification.passed
        return ReviewResult(accepted=True, confidence=0.95)


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
    diagnoser: FakeDiagnoser,
    fixer: FakeFixer | None = None,
) -> GraphRuntime:
    return GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            coordinator=coordinator,
            investigator=investigator,
            diagnoser=diagnoser,
            fixer=fixer or FakeFixer(),
            reviewer=FakeReviewer(),
        ),
    )


@pytest.mark.asyncio
async def test_send_fans_out_in_parallel_and_fans_in_before_diagnosis(tmp_path: Path) -> None:
    started: set[str] = set()
    barrier = asyncio.Event()

    class BarrierInvestigator(FakeInvestigator):
        async def run(self, **kwargs: object) -> WorkerResult:
            task = kwargs["task"]
            assert isinstance(task, InvestigationTask)
            started.add(task.task_id)
            if len(started) == 3:
                barrier.set()
            # Serial execution cannot release this barrier before the hang timeout.
            await asyncio.wait_for(barrier.wait(), timeout=3)
            return await super().run(**kwargs)

    coordinator = FakeCoordinator([[task("a"), task("b"), task("c")]])
    investigator = BarrierInvestigator()
    diagnoser = FakeDiagnoser()
    graph = build_graph(
        make_runtime(tmp_path, coordinator, investigator, diagnoser),
        checkpointer=InMemorySaver(),
    )

    result = await graph.ainvoke(
        initial_state(tmp_path), {"configurable": {"thread_id": "parallel"}}
    )

    assert started == {"a", "b", "c"}
    assert result["status"] == "success"
    assert len(result["evidence"]) == 3
    assert diagnoser.evidence_counts == [3]


@pytest.mark.asyncio
async def test_successful_parallel_writes_survive_worker_crash(tmp_path: Path) -> None:
    coordinator = FakeCoordinator([[task("a"), task("b"), task("c")]])
    investigator = FakeInvestigator(fail_once="c")
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
    assert investigator.calls == Counter({"c": 2, "a": 1, "b": 1})


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
    coordinator = FakeCoordinator([[task("a"), task("b"), task("c")]])
    investigator = FakeInvestigator(fail_once="c")
    diagnoser = FakeDiagnoser()
    runtime = make_runtime(tmp_path, coordinator, investigator, diagnoser)
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    first_handle = await create_async_sqlite_checkpointer(checkpoint_path)
    graph = build_graph(runtime, checkpointer=first_handle.saver)
    invocation_config = {"configurable": {"thread_id": "disk-crash"}}

    with pytest.raises(RuntimeError, match="simulated worker crash"):
        await graph.ainvoke(initial_state(tmp_path, run_id="disk-run"), invocation_config)
    await first_handle.close()

    second_handle = await create_async_sqlite_checkpointer(checkpoint_path)
    rebuilt_graph = build_graph(runtime, checkpointer=second_handle.saver)
    result = await rebuilt_graph.ainvoke(None, invocation_config)
    await second_handle.close()

    assert result["status"] == "success"
    assert investigator.calls == Counter({"c": 2, "a": 1, "b": 1})


@pytest.mark.asyncio
async def test_low_confidence_routes_to_additional_investigation(tmp_path: Path) -> None:
    coordinator = FakeCoordinator([[task("a")], [task("b")]])
    investigator = FakeInvestigator()
    diagnoser = FakeDiagnoser(low_first=True)
    graph = build_graph(make_runtime(tmp_path, coordinator, investigator, diagnoser))

    result = await graph.ainvoke(initial_state(tmp_path))

    assert result["status"] == "success"
    assert result["investigation_round"] == 2
    assert diagnoser.evidence_counts == [1, 2]


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


class FakeExperienceMiner:
    def should_mine(
        self,
        *,
        success: bool,
        tool_calls: int,
        failed_attempts: int,
        reusable_script_created: bool,
        repeated_pattern_detected: bool,
    ) -> bool:
        del tool_calls, failed_attempts, reusable_script_created, repeated_pattern_detected
        return success

    async def decide(
        self,
        trajectory_summary: dict[str, object],
        *,
        existing_skills: object = None,
    ) -> LearningDecision:
        del trajectory_summary, existing_skills
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
        return LearningDecision(
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


class FakeUpdateMiner(FakeExperienceMiner):
    async def decide(
        self,
        trajectory_summary: object,
        *,
        existing_skills: object = None,
    ) -> LearningDecision:
        del trajectory_summary, existing_skills
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
        return LearningDecision(
            action="update_skill",
            rationale="the existing procedure needs stronger verification",
            target_skill_id="assertion-repair",
            target_version=1,
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
        experience_miner=FakeExperienceMiner(),
        candidate_validator=CandidateValidator(registry),
    )
    result_a = await build_graph(runtime_a).ainvoke(initial_state(tmp_path, run_id="learning-run"))

    learned = registry.get("assertion-repair", 1)
    assert result_a["candidate_skill_id"] == "assertion-repair"
    assert learned is not None
    assert learned.manifest.status == "trial"

    coordinator_b = FakeCoordinator([[task("b")]])
    base_b = make_runtime(tmp_path, coordinator_b, FakeInvestigator(), FakeDiagnoser(), FakeFixer())
    runtime_b = replace(
        base_b,
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
    )
    result_b = await build_graph(runtime_b).ainvoke(initial_state(tmp_path, run_id="reuse-run"))

    assert result_b["status"] == "success"
    assert coordinator_b.seen_skill_counts == [1]
    assert registry.stats("assertion-repair", 1).retrieval_count == 1
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
    created = registry.create_candidate(
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
    assert (
        CandidateValidator(registry)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )
    base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("uses-skill")]]),
        SkillUsingInvestigator(),
        FakeDiagnoser(),
    )
    runtime = replace(
        base,
        config=base.config.model_copy(
            update={
                "trial_min_uses": 1,
                "trial_min_successes": 1,
                "trial_min_success_rate": 1.0,
            }
        ),
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
    )

    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="promotion-run"))

    assert result["status"] == "success"
    stats = registry.stats(created.manifest.skill_id, created.manifest.version)
    assert stats.retrieval_count == 1
    assert stats.selected_count == 1
    assert stats.use_count == 1
    assert stats.success_count == 1
    promoted = registry.get(created.manifest.skill_id, created.manifest.version)
    assert promoted is not None and promoted.manifest.status == "active"
    registry.close()


@pytest.mark.asyncio
async def test_learning_update_creates_same_id_child_without_early_supersede(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    original_decision = await FakeExperienceMiner().decide({})
    assert original_decision.candidate_skill is not None
    original = registry.create_candidate(original_decision.candidate_skill)
    validator = CandidateValidator(registry)
    assert validator.validate_to_trial(original.manifest.skill_id, original.manifest.version).passed
    registry.transition(original.manifest.skill_id, original.manifest.version, "active")
    base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("update")]]),
        FakeInvestigator(),
        FakeDiagnoser(),
    )
    runtime = replace(
        base,
        capability_registry=registry,
        experience_miner=FakeUpdateMiner(),
        candidate_validator=validator,
    )

    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="update-run"))

    assert result["candidate_skill_id"] == original.manifest.skill_id
    parent = registry.get(original.manifest.skill_id, 1)
    child = registry.get(original.manifest.skill_id, 2)
    assert parent is not None and parent.manifest.status == "active"
    assert child is not None and child.manifest.status == "trial"
    assert child.manifest.parent_version == 1
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
        self.contexts: list[AgentContext] = []
        self.previous_verifications: list[VerificationResult | None] = []

    async def propose(
        self,
        *,
        context: AgentContext,
        diagnosis: Diagnosis,
        evidence: list[EvidenceItem],
        previous_verification: VerificationResult | None,
    ) -> FixerOutput:
        del diagnosis, evidence
        self.contexts.append(context)
        self.previous_verifications.append(previous_verification)
        path = "bad.py" if len(self.contexts) == 1 else "good.py"
        return FixerOutput(
            proposal=PatchProposal(
                summary=f"attempt {len(self.contexts)}",
                changed_files=[path],
                risk="low",
                verification_plan=[["python", "-c", "raise SystemExit(0)"]],
            ),
            edits=[FileEdit(path=path, content=f"ATTEMPT = {len(self.contexts)}\n")],
        )


class RejectFirstReviewer:
    def __init__(self) -> None:
        self.calls = 0

    async def review(
        self,
        *,
        context: AgentContext,
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
    fixer = RetryFeedbackFixer()
    reviewer = RejectFirstReviewer()
    runtime = make_runtime(
        tmp_path,
        FakeCoordinator([[task("inspect")]]),
        FakeInvestigator(),
        FakeDiagnoser(),
    )
    runtime = replace(
        runtime,
        agents=replace(runtime.agents, fixer=fixer, reviewer=reviewer),
    )

    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="review-retry"))

    assert result["status"] == "success"
    assert not (tmp_path / "bad.py").exists()
    assert (tmp_path / "good.py").read_text() == "ATTEMPT = 2\n"
    assert fixer.contexts[1].previous_review_blockers == ("attempt A violates review policy",)
    assert fixer.previous_verifications[1] is not None
    assert fixer.previous_verifications[1].passed


class TwoFileCrashFixer:
    def __init__(self, expected_sha256: str) -> None:
        self.expected_sha256 = expected_sha256

    async def propose(
        self,
        *,
        context: AgentContext,
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
    runtime = replace(runtime, agents=replace(runtime.agents, fixer=fixer))
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
    async def propose(self, **kwargs: object) -> MemoryCandidate:
        del kwargs
        raise RuntimeError("memory consolidation exploded")


class RaisingExperienceMiner:
    def should_mine(self, **kwargs: object) -> bool:
        del kwargs
        return True

    async def decide(
        self,
        trajectory: object,
        *,
        existing_skills: object = None,
    ) -> LearningDecision:
        del trajectory, existing_skills
        raise RuntimeError("experience mining exploded")


class RaisingCurator:
    async def run(self, **kwargs: object) -> object:
        del kwargs
        raise RuntimeError("curator exploded")


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
        experience_miner=RaisingExperienceMiner(),
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="experience-learning-error")
    )

    assert result["status"] == "success"
    assert result["learning_errors"][0]["stage"] == "experience_mining"
    registry.close()


@pytest.mark.asyncio
async def test_curator_failure_cannot_flip_core_success(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("inspect")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
        ),
        capability_registry=registry,
        experience_miner=FakeExperienceMiner(),
        candidate_validator=CandidateValidator(registry),
        curator_pipeline=RaisingCurator(),  # type: ignore[arg-type]
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="curator-learning-error")
    )

    assert result["status"] == "success"
    assert result["candidate_skill_id"] == "assertion-repair"
    assert result["learning_errors"][0]["stage"] == "curator"
    registry.close()


class ExplicitlyFailingSkillInvestigator(FakeInvestigator):
    def __init__(self, recorder: TrajectoryRecorder) -> None:
        super().__init__()
        self.recorder = recorder

    async def run(
        self,
        *,
        task: InvestigationTask,
        context: AgentContext,
        capabilities: WorkerCapabilities,
    ) -> WorkerResult:
        result = await super().run(task=task, context=context, capabilities=capabilities)
        skill = context.skills[0]
        self.recorder.emit(
            run_id=context.run_id,
            event_type=EventType.SKILL_USED,
            agent_id=f"investigator:{task.task_id}",
            invocation_id=context.invocation_id,
            event_key="failing-script",
            payload={
                "skill_id": skill.skill_id,
                "version": skill.version,
                "resource": "scripts/failing.py",
                "success": False,
            },
        )
        return result.model_copy(
            update={"used_skill_refs": [SkillRef(skill_id=skill.skill_id, version=skill.version)]}
        )


@pytest.mark.asyncio
async def test_explicit_skill_failure_cannot_receive_success_credit(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    decision = await FakeExperienceMiner().decide({})
    assert decision.candidate_skill is not None
    created = registry.create_candidate(decision.candidate_skill)
    assert (
        CandidateValidator(registry)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )
    recorder = TrajectoryRecorder()
    base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("uses-failing-skill")]]),
        ExplicitlyFailingSkillInvestigator(recorder),
        FakeDiagnoser(),
    )
    runtime = replace(
        base,
        config=base.config.model_copy(update={"trial_min_uses": 1, "trial_min_successes": 1}),
        recorder=recorder,
        capability_registry=registry,
        capability_retriever=CapabilityRetriever(registry),
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="failed-skill-successful-run")
    )

    assert result["status"] == "success"
    stats = registry.stats(created.manifest.skill_id, created.manifest.version)
    assert stats.success_count == 0
    assert stats.failure_count == 1
    current = registry.get(created.manifest.skill_id, created.manifest.version)
    assert current is not None and current.manifest.status == "trial"
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
    decision = await FakeExperienceMiner().decide({})
    assert decision.candidate_skill is not None
    original = registry.create_candidate(decision.candidate_skill)
    assert (
        CandidateValidator(registry)
        .validate_to_trial(original.manifest.skill_id, original.manifest.version)
        .passed
    )
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("inspect")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
        ),
        capability_registry=registry,
        experience_miner=FakeExperienceMiner(),
        candidate_validator=CandidateValidator(registry),
    )

    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="new-skill-collision")
    )

    assert result["status"] == "success"
    assert result["candidate_skill_id"] is None
    assert result["learning_errors"][0]["stage"] == "skill_candidate_creation"
    assert registry.get(original.manifest.skill_id, 2) is None
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
            FakeCoordinator([[task("repo")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_repair_attempts": 1}),
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
async def test_verification_failure_preserves_stage_after_rollback(tmp_path: Path) -> None:
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("repo")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_repair_attempts": 1}),
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

    monkeypatch.setattr(graph_builder_module, "_restore_edit_baseline", boom)
    monkeypatch.setattr(graph_builder_module, "restore_attempt_writes", boom)
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("repo")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_repair_attempts": 1}),
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
    assert result["failure_stage"] == "rollback_attempt"
    episode = memory_store.get_episode("stage-rollback")
    assert episode is not None
    assert episode.failure_stage == "rollback_attempt"
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
            FakeCoordinator([[task("repo")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 2\n")),
        ),
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_repair_attempts": 1}),
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
        config=EvoCIConfig.from_env(cwd=tmp_path).model_copy(update={"max_repair_attempts": 1}),
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
