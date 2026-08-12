from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.types import Command
from pydantic import BaseModel

import evoci.cli as cli
from evoci.benchmark.models import BenchmarkVariant
from evoci.benchmark.variants import variant_features
from evoci.capability.curator import CuratorReview
from evoci.capability.miner import LearningDecision
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
    WorkerResult,
)
from evoci.memory.models import MemoryCandidate
from evoci.model.gateway import (
    ResponseT,
    ToolCallRequest,
    ToolDefinition,
    ToolLoopMessage,
    ToolModelResponse,
)
from evoci.runtime.events import EventType


class OfflineBenchmarkGateway:
    def __init__(self, *_: object, **__: object) -> None:
        self.turns: dict[str, int] = {}

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
        agent_id: str,
    ) -> ResponseT:
        del system_prompt, user_prompt, agent_id
        output: BaseModel
        if response_model is InvestigationPlan:
            output = InvestigationPlan(
                tasks=[
                    InvestigationTask(
                        task_id="repository",
                        role="repository",
                        objective="inspect calculator implementation",
                    ),
                    InvestigationTask(
                        task_id="test",
                        role="test",
                        objective="inspect expected behavior",
                    ),
                ],
                reasoning_summary="inspect source and test independently",
            )
        elif response_model is Diagnosis:
            output = Diagnosis(
                primary=Hypothesis(
                    root_cause="calculator subtracts instead of adding",
                    evidence_ids=[],
                    confidence=0.99,
                    affected_files=["calculator.py"],
                    proposed_action="replace subtraction with addition",
                ),
                needs_more_evidence=False,
            )
        elif response_model is MemoryCandidate:
            output = MemoryCandidate(
                type="semantic",
                content="calculator addition failures can come from a subtraction operator",
                namespace="repo:fixture/calculator",
                confidence=0.9,
            )
        elif response_model is LearningDecision:
            output = LearningDecision(action="none", rationale="fixture adds no new capability")
        elif response_model is CuratorReview:
            output = CuratorReview(action="none", rationale="no duplicate")
        else:
            raise AssertionError(response_model)
        return response_model.model_validate(output.model_dump())

    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
    ) -> ToolModelResponse:
        del messages, tools
        count = self.turns.get(agent_id, 0)
        self.turns[agent_id] = count + 1
        if agent_id.startswith("investigator:") and count == 0:
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id=f"{agent_id}:read",
                        name="read_file",
                        arguments={"path": "calculator.py"},
                    )
                ]
            )
        if agent_id == "fixer" and count == 0:
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="fixer:patch",
                        name="apply_patch",
                        arguments={
                            "files": {
                                "calculator.py": (
                                    "def add(left: int, right: int) -> int:\n"
                                    "    return left + right\n"
                                )
                            }
                        },
                    )
                ]
            )
        if agent_id == "reviewer" and count == 0:
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="reviewer:test",
                        name="run_test",
                        arguments={"argv": ["python", "-m", "unittest", "-q"]},
                    )
                ]
            )
        return ToolModelResponse(content="finalize")

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
    ) -> ResponseT:
        del messages
        output: BaseModel
        if response_model is WorkerResult:
            output = WorkerResult(
                task_id=agent_id.split(":", 1)[-1],
                summary="repository inspected",
                evidence=[
                    EvidenceItem(
                        source_agent=agent_id,
                        kind="source_code",
                        claim="calculator uses subtraction",
                        file_path="calculator.py",
                        confidence=0.95,
                    )
                ],
            )
        elif response_model is FixerOutput:
            output = FixerOutput(
                proposal=PatchProposal(
                    summary="correct calculator addition",
                    changed_files=["calculator.py"],
                    commands_run=["python -m unittest -q"],
                    risk="low",
                    verification_plan=[["python", "-m", "unittest", "-q"]],
                ),
                edits=[
                    FileEdit(
                        path="calculator.py",
                        content=(
                            "def add(left: int, right: int) -> int:\n    return left + right\n"
                        ),
                    )
                ],
            )
        elif response_model is ReviewResult:
            output = ReviewResult(accepted=True, confidence=0.99)
        else:
            raise AssertionError(response_model)
        return response_model.model_validate(output.model_dump())


def make_workspace(path: Path) -> None:
    path.mkdir()
    (path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n"
    )
    (path / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import add\n"
        "class TestCalculator(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["single", "multi", "multi-memory", "evo"])
async def test_all_benchmark_variants_execute_their_real_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: BenchmarkVariant,
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleGateway", OfflineBenchmarkGateway)
    workspace = tmp_path / f"workspace-{variant}"
    make_workspace(workspace)
    config = EvoCIConfig.from_env(cwd=tmp_path).model_copy(
        update={
            "state_dir": tmp_path / f"state-{variant}",
            "workspace_dir": tmp_path / f"worktrees-{variant}",
            "repo_cache_dir": tmp_path / f"repos-{variant}",
            "max_run_model_calls": 20,
            "max_run_tool_calls": 10,
        }
    )
    features = variant_features(variant)
    resources = await cli._live_resources(config, features)
    run_id = f"variant-{variant}"
    initial: dict[str, object] = {
        "run_id": run_id,
        "task_id": "fixture",
        "repo": RepoSpec(owner="fixture", name="calculator"),
        "ci_failure": CIFailure(
            summary="addition test failed",
            log_excerpt="AssertionError: -1 != 3",
            failed_commands=[["python", "-m", "unittest", "-q"]],
            task_family="test",
        ),
        "workspace_path": str(workspace),
    }
    try:
        if variant == "single":
            result = await cli._drive_single(resources, initial)
        else:
            result = await cli._drive_graph_automated(resources, initial, run_id)
        assert result["status"] == "success"
        assert any(event.type == EventType.TOOL_CALL for event in resources.recorder.events(run_id))
        assert (resources.runtime.memory_store is not None) is features.long_term_memory
        assert (resources.runtime.capability_registry is not None) is features.capabilities
        assert (resources.runtime.curator_pipeline is not None) is features.curator
        assert resources.runtime.budget_manager is not None
        budget = resources.runtime.budget_manager.for_run(run_id).snapshot()
        assert budget.model_calls <= config.max_run_model_calls
        assert budget.tool_calls <= config.max_run_tool_calls
        events = resources.recorder.events(run_id)
        assert budget.model_calls == sum(
            event.type == EventType.MODEL_CALL and event.payload.get("budget_scope") != "post_run"
            for event in events
        )
        assert budget.tool_calls == sum(
            event.type == EventType.TOOL_CALL
            and event.payload.get("budget_scope") not in {"post_run", "cleanup"}
            for event in events
        )
    finally:
        await resources.close()


class RetrySingleGateway(OfflineBenchmarkGateway):
    def __init__(self, *_: object, **__: object) -> None:
        super().__init__()
        self.fixer_finalizations = 0

    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
    ) -> ToolModelResponse:
        del messages, tools, agent_id
        return ToolModelResponse(content="finalize")

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
    ) -> ResponseT:
        del messages, agent_id
        if response_model is not FixerOutput:
            raise AssertionError(response_model)
        self.fixer_finalizations += 1
        succeeds = self.fixer_finalizations > 1
        output = FixerOutput(
            proposal=PatchProposal(
                summary="retry fixture",
                changed_files=["calculator.py"],
                risk="low",
                verification_plan=[
                    [
                        "python",
                        "-c",
                        f"raise SystemExit({0 if succeeds else 1})",
                    ]
                ],
            ),
            edits=[
                FileEdit(
                    path="calculator.py",
                    content=(
                        "def add(left: int, right: int) -> int:\n"
                        f"    return left {'+' if succeeds else '-'} right\n"
                    ),
                )
            ],
        )
        return response_model.model_validate(output.model_dump())


@pytest.mark.asyncio
async def test_single_variant_uses_feedback_and_retries_with_shared_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleGateway", RetrySingleGateway)
    workspace = tmp_path / "single-retry"
    make_workspace(workspace)
    config = EvoCIConfig.from_env(cwd=tmp_path).model_copy(
        update={
            "state_dir": tmp_path / "single-retry-state",
            "capability_dir": tmp_path / "single-retry-skills",
            "runtime_dir": tmp_path / "single-retry-runtime",
            "max_repair_attempts": 3,
            "max_run_model_calls": 6,
            "max_run_tool_calls": 6,
        }
    )
    resources = await cli._live_resources(config, variant_features("single"))
    initial: dict[str, object] = {
        "run_id": "single-retry",
        "task_id": "fixture",
        "repo": RepoSpec(owner="fixture", name="calculator"),
        "ci_failure": CIFailure(
            summary="addition test failed",
            log_excerpt="AssertionError",
            task_family="test",
        ),
        "workspace_path": str(workspace),
    }
    try:
        result = await cli._drive_single(resources, initial)
        assert result["status"] == "success"
        assert result["repair_attempt"] == 2
        assert len(result["verification_history"]) == 2
        assert (workspace / "calculator.py").read_text().endswith("return left + right\n")
        assert resources.runtime.budget_manager is not None
        budget = resources.runtime.budget_manager.for_run("single-retry").snapshot()
        assert budget.model_calls <= 6
        assert budget.tool_calls <= 6
    finally:
        await resources.close()


@pytest.mark.asyncio
async def test_benchmark_driver_approves_every_hitl_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptingGraph:
        def __init__(self) -> None:
            self.inputs: list[object] = []

        async def ainvoke(self, value: object, config: object) -> dict[str, object]:
            del config
            self.inputs.append(value)
            if len(self.inputs) < 3:
                return {"__interrupt__": [object()]}
            return {"status": "success"}

    graph = InterruptingGraph()
    monkeypatch.setattr(cli, "build_graph", lambda *args, **kwargs: graph)
    resources = SimpleNamespace(runtime=object(), checkpoint=SimpleNamespace(saver=object()))

    result = await cli._drive_graph_automated(  # type: ignore[arg-type]
        resources, {"run_id": "hitl"}, "hitl"
    )

    assert result["status"] == "success"
    resumes = graph.inputs[1:]
    assert all(isinstance(value, Command) and value.resume is True for value in resumes)


@pytest.mark.asyncio
async def test_parallel_budget_exhaustion_is_a_clean_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleGateway", OfflineBenchmarkGateway)
    workspace = tmp_path / "budget-exhaustion"
    make_workspace(workspace)
    config = EvoCIConfig.from_env(cwd=tmp_path).model_copy(
        update={
            "state_dir": tmp_path / "budget-state",
            "capability_dir": tmp_path / "budget-skills",
            "runtime_dir": tmp_path / "budget-runtime",
            "max_run_model_calls": 2,
            "max_run_tool_calls": 2,
        }
    )
    resources = await cli._live_resources(config, variant_features("multi"))
    initial: dict[str, object] = {
        "run_id": "budget-exhaustion",
        "task_id": "fixture",
        "repo": RepoSpec(owner="fixture", name="calculator"),
        "ci_failure": CIFailure(
            summary="addition failed",
            log_excerpt="AssertionError",
            task_family="test",
        ),
        "workspace_path": str(workspace),
    }
    try:
        result = await cli._drive_graph_automated(resources, initial, "budget-exhaustion")
        assert result["status"] == "failed"
        assert "budget exhausted" in result["failure_reason"]
        assert resources.runtime.budget_manager is not None
        snapshot = resources.runtime.budget_manager.for_run("budget-exhaustion").snapshot()
        assert snapshot.model_calls == 2
        assert snapshot.tool_calls <= 2
    finally:
        await resources.close()
