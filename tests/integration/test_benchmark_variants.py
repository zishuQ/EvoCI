from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.types import Command
from pydantic import BaseModel

import evoci.cli as cli
from evoci.agents.model_agents import WorkerReport
from evoci.benchmark.models import BenchmarkVariant
from evoci.benchmark.variants import variant_features
from evoci.capability.miner import SkillLearningDecision
from evoci.config import EvoCIConfig
from evoci.domain.models import (
    CIFailure,
    EvidenceItem,
    RepoSpec,
    SupervisorDecision,
    WorkerTask,
)
from evoci.memory.models import LongTermFactCandidate
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
        usage_observer: object | None = None,
    ) -> ResponseT:
        del system_prompt, user_prompt, agent_id, usage_observer
        output: BaseModel
        if response_model is LongTermFactCandidate:
            output = LongTermFactCandidate(
                type="fact",
                content="calculator addition failures can come from a subtraction operator",
                confidence=0.9,
            )
        elif response_model is SkillLearningDecision:
            output = SkillLearningDecision(
                action="none", rationale="fixture adds no new capability"
            )
        else:
            raise AssertionError(response_model)
        return response_model.model_validate(output.model_dump())

    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
        usage_observer: object | None = None,
    ) -> ToolModelResponse:
        del messages, usage_observer
        count = self.turns.get(agent_id, 0)
        self.turns[agent_id] = count + 1
        writable = any(item.name == "apply_patch" for item in tools)
        if agent_id == "supervisor" and count == 0:
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="supervisor:read",
                        name="read_file",
                        arguments={"path": "calculator.py"},
                    )
                ]
            )
        if agent_id.startswith("worker:") and count == 0 and writable:
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id=f"{agent_id}:patch",
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
        if agent_id.startswith("worker:") and count == 0:
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id=f"{agent_id}:read",
                        name="read_file",
                        arguments={"path": "calculator.py"},
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
        usage_observer: object | None = None,
    ) -> ResponseT:
        del messages, usage_observer
        output: BaseModel
        if response_model is SupervisorDecision:
            count = self.turns.get("supervisor-final", 0)
            self.turns["supervisor-final"] = count + 1
            if count == 0:
                output = SupervisorDecision(
                    action="dispatch",
                    reasoning_summary="repair the calculator",
                    tasks=[
                        WorkerTask(
                            task_id="repair-add",
                            kind="repair",
                            objective="fix addition",
                            write_scope=["calculator.py"],
                        )
                    ],
                )
            else:
                output = SupervisorDecision(
                    action="stop",
                    reasoning_summary="formal verification already passed",
                    stop_reason="verified success",
                )
        elif response_model is WorkerReport:
            output = WorkerReport(
                summary="correct calculator addition",
                evidence=[
                    EvidenceItem(
                        source_agent=agent_id,
                        kind="source_code",
                        claim="calculator used subtraction",
                        file_path="calculator.py",
                        confidence=0.95,
                    )
                ],
                commands_run=["python -m unittest -q"],
                verification_plan=[["python", "-m", "unittest", "-q"]],
            )
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
        self._wrote_this_attempt = False

    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
        usage_observer: object | None = None,
    ) -> ToolModelResponse:
        del messages, tools, agent_id, usage_observer
        if not self._wrote_this_attempt:
            self._wrote_this_attempt = True
            operator = "+" if self.fixer_finalizations >= 1 else "-"
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id=f"fixer:patch:{self.fixer_finalizations}",
                        name="apply_patch",
                        arguments={
                            "files": {
                                "calculator.py": (
                                    "def add(left: int, right: int) -> int:\n"
                                    f"    return left {operator} right\n"
                                )
                            }
                        },
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
        usage_observer: object | None = None,
    ) -> ResponseT:
        del messages, agent_id, usage_observer
        if response_model is not WorkerReport:
            raise AssertionError(response_model)
        self.fixer_finalizations += 1
        self._wrote_this_attempt = False
        succeeds = self.fixer_finalizations > 1
        output = WorkerReport(
            summary="retry fixture",
            risk="low",
            verification_plan=[
                [
                    "python",
                    "-c",
                    f"raise SystemExit({0 if succeeds else 1})",
                ]
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
            "max_supervisor_batches": 3,
            "max_run_model_calls": 10,
            "max_run_tool_calls": 10,
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
            failed_commands=[["python", "-m", "unittest", "-q"]],
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
        assert budget.model_calls <= 10
        assert budget.tool_calls <= 10
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
