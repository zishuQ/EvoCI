from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from evoci.capability.registry import CapabilityRegistry
from evoci.capability.validator import CandidateValidator
from evoci.config import EvoCIConfig
from evoci.domain.models import (
    CIFailure,
    FileEdit,
    FixerOutput,
    PatchProposal,
    ReviewResult,
    VerificationResult,
)
from evoci.graph.builder import GraphRuntime, build_graph, persist_run_outcome
from evoci.memory.store import SQLiteMemoryStore
from evoci.runtime.budget import RunBudgetManager
from evoci.runtime.trajectory import TrajectoryRecorder
from tests.integration.test_graph import (
    FakeCoordinator,
    FakeDiagnoser,
    FakeExperienceMiner,
    FakeFixer,
    FakeInvestigator,
    FakeReviewer,
    initial_state,
    make_runtime,
    task,
)

PASSING = ["python", "-c", "raise SystemExit(0)"]
VALUE_ORACLE = ["python", "-c", "import app; assert app.VALUE == 1"]


class PlanReplacingFixer:
    async def propose(
        self,
        *,
        context: object,
        diagnosis: object,
        evidence: object,
        previous_verification: VerificationResult | None,
    ) -> FixerOutput:
        del context, diagnosis, evidence, previous_verification
        return FixerOutput(
            proposal=PatchProposal(
                summary="noop with passing self-check",
                changed_files=[],
                risk="low",
                verification_plan=[PASSING],
            ),
            edits=[],
        )


class RealValueFixer:
    async def propose(
        self,
        *,
        context: object,
        diagnosis: object,
        evidence: object,
        previous_verification: VerificationResult | None,
    ) -> FixerOutput:
        del context, diagnosis, evidence, previous_verification
        return FixerOutput(
            proposal=PatchProposal(
                summary="set VALUE to 1",
                changed_files=["app.py"],
                risk="low",
                verification_plan=[PASSING],
            ),
            edits=[FileEdit(path="app.py", content="VALUE = 1\n")],
        )


class AcceptingReviewer:
    async def review(self, **kwargs: object) -> ReviewResult:
        del kwargs
        return ReviewResult(accepted=True, confidence=1.0)


class TwoCommandFixer:
    def __init__(self, commands: list[list[str]]) -> None:
        self.commands = commands

    async def propose(self, **kwargs: object) -> FixerOutput:
        del kwargs
        return FixerOutput(
            proposal=PatchProposal(
                summary="no file edits",
                changed_files=[],
                risk="low",
                verification_plan=self.commands,
            ),
            edits=[],
        )


class HashMismatchFixer:
    def __init__(self, digest_a: str) -> None:
        self.digest_a = digest_a

    async def propose(self, **kwargs: object) -> FixerOutput:
        del kwargs
        return FixerOutput(
            proposal=PatchProposal(
                summary="two-file patch",
                changed_files=["a.txt", "b.txt"],
                risk="low",
                verification_plan=[PASSING],
            ),
            edits=[
                FileEdit(path="a.txt", content="new-a\n", expected_sha256=self.digest_a),
                FileEdit(path="b.txt", content="new-b\n", expected_sha256="0" * 64),
            ],
        )


def _runtime(
    tmp_path: Path,
    fixer: object,
    *,
    max_repair_attempts: int = 1,
    max_tool_calls: int = 256,
    reviewer: object | None = None,
) -> GraphRuntime:
    config = EvoCIConfig.from_env(cwd=tmp_path).model_copy(
        update={
            "max_repair_attempts": max_repair_attempts,
            "max_run_tool_calls": max_tool_calls,
        }
    )
    base = make_runtime(
        tmp_path,
        FakeCoordinator([[task("inspect")]]),
        FakeInvestigator(),
        FakeDiagnoser(),
        FakeFixer(),
    )
    return replace(
        base,
        config=config,
        agents=replace(
            base.agents,
            fixer=fixer,
            reviewer=reviewer or AcceptingReviewer(),
        ),
        budget_manager=RunBudgetManager(
            max_model_calls=config.max_run_model_calls,
            max_tool_calls=max_tool_calls,
        ),
    )


def _failure(workspace: Path, commands: list[list[str]]) -> dict[str, object]:
    state = initial_state(workspace)
    state["ci_failure"] = CIFailure(
        summary="VALUE is wrong",
        log_excerpt="AssertionError",
        failed_commands=commands,
    )
    return state


@pytest.mark.asyncio
async def test_a01_agent_cannot_replace_failed_oracle(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    runtime = replace(
        _runtime(tmp_path, PlanReplacingFixer(), reviewer=AcceptingReviewer()),
        memory_store=memory,
    )
    result = await build_graph(runtime).ainvoke(
        _failure(tmp_path, [VALUE_ORACLE]),
        {"configurable": {"thread_id": "a01-1"}},
    )
    assert result["status"] == "failed"
    assert result["verification"].passed is False
    assert result["verification"].status in {"failed", "incomplete"}
    assert any(item.source == "mandatory" for item in result["verification"].commands)
    episode = memory.get_episode("run-1")
    assert episode is not None
    assert episode.success is False
    assert (tmp_path / "app.py").read_text() == "VALUE = 0\n"
    memory.close()


@pytest.mark.asyncio
async def test_a01_true_fix_runs_mandatory_and_supplementary_once(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "caps.sqlite")
    runtime = replace(
        _runtime(tmp_path, RealValueFixer()),
        memory_store=memory,
        experience_miner=FakeExperienceMiner(),
        capability_registry=registry,
        candidate_validator=CandidateValidator(registry),
    )
    result = await build_graph(runtime).ainvoke(
        _failure(tmp_path, [VALUE_ORACLE]),
        {"configurable": {"thread_id": "a01-3"}},
    )
    assert result["status"] == "success"
    verification = result["verification"]
    assert verification.passed is True
    sources = [item.source for item in verification.commands if item.executed]
    assert "mandatory" in sources
    assert "supplementary" in sources
    episode = memory.get_episode("run-1")
    assert episode is not None
    assert episode.success is True
    memory.close()
    registry.close()


@pytest.mark.asyncio
async def test_a01_no_oracle_is_unavailable(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, PlanReplacingFixer())
    result = await build_graph(runtime).ainvoke(
        _failure(tmp_path, []),
        {"configurable": {"thread_id": "a01-5"}},
    )
    assert result["status"] == "failed"
    assert result["verification"].status == "unavailable"
    assert result["verification"].passed is False


@pytest.mark.asyncio
async def test_a01_deferred_learning_waits_for_independent_verdict(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "caps.sqlite")
    recorder = TrajectoryRecorder()
    runtime = replace(
        _runtime(tmp_path, RealValueFixer()),
        memory_store=memory,
        capability_registry=registry,
        experience_miner=FakeExperienceMiner(),
        recorder=recorder,
        defer_success_learning=True,
    )
    result = await build_graph(runtime).ainvoke(
        _failure(tmp_path, [VALUE_ORACLE]),
        {"configurable": {"thread_id": "a01-4"}},
    )
    assert result["status"] == "success"
    assert result["learning_deferred"] is True
    assert memory.get_episode("run-1") is None
    learned = await persist_run_outcome(
        runtime,
        result,
        success=False,
        failure_reason="independent benchmark verification failed",
    )
    assert learned["status"] == "failed"
    episode = memory.get_episode("run-1")
    assert episode is not None
    assert episode.success is False
    memory.close()
    registry.close()


@pytest.mark.asyncio
async def test_a02_partial_hash_failure_restores_applied_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("keep-a\n")
    (tmp_path / "b.txt").write_text("keep-b\n")
    (tmp_path / "user.txt").write_text("dirty\n")
    fixer = HashMismatchFixer(hashlib.sha256(b"keep-a\n").hexdigest())
    result = await build_graph(_runtime(tmp_path, fixer)).ainvoke(
        initial_state(tmp_path, run_id="a02-1"),
        {"configurable": {"thread_id": "a02-1"}},
    )
    assert result["status"] == "failed"
    assert (tmp_path / "a.txt").read_text() == "keep-a\n"
    assert (tmp_path / "b.txt").read_text() == "keep-b\n"
    assert (tmp_path / "user.txt").read_text() == "dirty\n"


@pytest.mark.asyncio
async def test_a02_budget_precheck_does_not_leave_partial_patch(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("keep-a\n")
    (tmp_path / "b.txt").write_text("keep-b\n")
    digest = hashlib.sha256(b"keep-a\n").hexdigest()
    digest_b = hashlib.sha256(b"keep-b\n").hexdigest()

    class TwoEditFixer:
        async def propose(self, **kwargs: object) -> FixerOutput:
            del kwargs
            return FixerOutput(
                proposal=PatchProposal(
                    summary="two edits",
                    changed_files=["a.txt", "b.txt"],
                    risk="low",
                    verification_plan=[PASSING],
                ),
                edits=[
                    FileEdit(path="a.txt", content="new-a\n", expected_sha256=digest),
                    FileEdit(path="b.txt", content="new-b\n", expected_sha256=digest_b),
                ],
            )

    runtime = _runtime(tmp_path, TwoEditFixer(), max_tool_calls=1)
    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="a02-2"),
        {"configurable": {"thread_id": "a02-2"}},
    )
    assert result["status"] == "failed"
    assert (tmp_path / "a.txt").read_text() == "keep-a\n"
    assert (tmp_path / "b.txt").read_text() == "keep-b\n"


@pytest.mark.asyncio
async def test_a03_verification_side_effects_stay_in_snapshot(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("baseline\n")
    leak = [
        "python",
        "-c",
        (
            "from pathlib import Path\n"
            "Path('undeclared.txt').write_text('leak')\n"
            "Path('notes.txt').write_text('mutated')\n"
            "raise SystemExit(1)"
        ),
    ]
    result = await build_graph(_runtime(tmp_path, PlanReplacingFixer())).ainvoke(
        _failure(tmp_path, [leak]),
        {"configurable": {"thread_id": "a03-1"}},
    )
    assert result["status"] == "failed"
    assert result["verification"].passed is False
    assert not (tmp_path / "undeclared.txt").exists()
    assert (tmp_path / "notes.txt").read_text() == "baseline\n"


@pytest.mark.asyncio
async def test_a03_passing_verification_does_not_write_back_mutations(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("baseline\n")
    mutate = [
        "python",
        "-c",
        "from pathlib import Path; Path('notes.txt').write_text('mutated'); raise SystemExit(0)",
    ]
    result = await build_graph(_runtime(tmp_path, PlanReplacingFixer())).ainvoke(
        _failure(tmp_path, [mutate]),
        {"configurable": {"thread_id": "a03-2"}},
    )
    assert result["status"] == "success"
    assert (tmp_path / "notes.txt").read_text() == "baseline\n"


@pytest.mark.asyncio
async def test_a04_single_incomplete_verification_is_not_success(tmp_path: Path) -> None:
    from evoci.agents.base import AgentSuite
    from evoci.cli import _drive_single

    recorder = TrajectoryRecorder()
    config = EvoCIConfig.from_env(cwd=tmp_path).model_copy(
        update={"max_repair_attempts": 1, "max_run_tool_calls": 1}
    )
    budget = RunBudgetManager(max_model_calls=20, max_tool_calls=1, recorder=recorder)
    runtime = GraphRuntime(
        config=config,
        agents=AgentSuite(
            coordinator=FakeCoordinator([[task("x")]]),
            investigator=FakeInvestigator(),
            diagnoser=FakeDiagnoser(),
            fixer=TwoCommandFixer(
                [
                    ["python", "-c", "raise SystemExit(0)"],
                    ["python", "-c", "raise SystemExit(1)"],
                ]
            ),
            reviewer=FakeReviewer(),
        ),
        recorder=recorder,
        budget_manager=budget,
    )
    resources = SimpleNamespace(recorder=recorder, runtime=runtime)
    result = await _drive_single(
        resources,  # type: ignore[arg-type]
        {
            "run_id": "single-a04",
            "task_id": "task",
            "repo": initial_state(tmp_path)["repo"],
            "ci_failure": CIFailure(
                summary="two checks",
                log_excerpt="fail",
                failed_commands=[
                    ["python", "-c", "raise SystemExit(0)"],
                    ["python", "-c", "raise SystemExit(1)"],
                ],
            ),
            "workspace_path": str(tmp_path),
        },
    )
    assert result["status"] != "success"
    assert result["verification"].passed is False
    assert result["verification"].status == "incomplete"
    assert result["review"].performed is False
    assert result["review"].status == "not_performed"


@pytest.mark.asyncio
async def test_a04_single_second_failure_and_full_pass(tmp_path: Path) -> None:
    from evoci.agents.base import AgentSuite
    from evoci.cli import _drive_single

    async def run_case(commands: list[list[str]], tool_calls: int) -> dict[str, object]:
        recorder = TrajectoryRecorder()
        config = EvoCIConfig.from_env(cwd=tmp_path).model_copy(
            update={"max_repair_attempts": 1, "max_run_tool_calls": tool_calls}
        )
        budget = RunBudgetManager(
            max_model_calls=20, max_tool_calls=tool_calls, recorder=recorder
        )
        runtime = GraphRuntime(
            config=config,
            agents=AgentSuite(
                coordinator=FakeCoordinator([[task("x")]]),
                investigator=FakeInvestigator(),
                diagnoser=FakeDiagnoser(),
                fixer=TwoCommandFixer(commands),
                reviewer=FakeReviewer(),
            ),
            recorder=recorder,
            budget_manager=budget,
        )
        return await _drive_single(
            SimpleNamespace(recorder=recorder, runtime=runtime),  # type: ignore[arg-type]
            {
                "run_id": f"single-{tool_calls}-{commands[-1][-1]}",
                "task_id": "task",
                "repo": initial_state(tmp_path)["repo"],
                "ci_failure": CIFailure(
                    summary="checks",
                    log_excerpt="fail",
                    failed_commands=commands,
                ),
                "workspace_path": str(tmp_path),
            },
        )

    failed = await run_case(
        [["python", "-c", "raise SystemExit(0)"], ["python", "-c", "raise SystemExit(1)"]],
        8,
    )
    assert failed["status"] == "failed"
    assert failed["verification"].status == "failed"
    passed = await run_case(
        [["python", "-c", "raise SystemExit(0)"], ["python", "-c", "raise SystemExit(0)"]],
        8,
    )
    assert passed["status"] == "success"
    assert passed["verification"].passed is True
    assert passed["review"].performed is False
    assert passed["review"].accepted is False
    assert passed["review"].status == "not_performed"
