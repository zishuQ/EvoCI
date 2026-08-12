from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from evoci.agents.base import AgentSuite
from evoci.agents.model_agents import ModelFixer, ModelInvestigator, ModelReviewer
from evoci.agents.tool_loop import BoundedToolAgent
from evoci.capability.models import GeneratedFile, SkillCandidate, SkillPermissions
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.validator import CandidateValidator
from evoci.config import EvoCIConfig
from evoci.demo import DemoCoordinator, DemoDiagnoser, DemoFixer, DemoReviewer
from evoci.domain.models import (
    CIFailure,
    EvidenceItem,
    FileEdit,
    FixerOutput,
    PatchProposal,
    RepoSpec,
    ReviewResult,
    SkillRef,
    WorkerResult,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.model.gateway import (
    ResponseT,
    ToolCallRequest,
    ToolDefinition,
    ToolLoopMessage,
    ToolModelResponse,
)
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.policy import FIXER_CAPABILITIES, INVESTIGATOR_CAPABILITIES
from evoci.tools.registry import create_worker_registry


class ScriptedToolGateway:
    def __init__(self, turns: list[ToolModelResponse], final: BaseModel) -> None:
        self.turns = turns
        self.final = final
        self.messages_seen: list[list[ToolLoopMessage]] = []

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
        agent_id: str,
    ) -> ResponseT:
        del system_prompt, user_prompt, response_model, agent_id
        raise AssertionError("one-shot completion is not used by a leaf tool loop")

    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
    ) -> ToolModelResponse:
        del tools, agent_id
        self.messages_seen.append(list(messages))
        return self.turns.pop(0)

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
    ) -> ResponseT:
        del messages, agent_id
        return response_model.model_validate(self.final.model_dump())


def worker_result(*, used: list[SkillRef] | None = None) -> WorkerResult:
    return WorkerResult(
        task_id="inspect",
        summary="inspected real workspace content",
        evidence=[
            EvidenceItem(
                source_agent="investigator:inspect",
                kind="source_code",
                claim="the fixture contains VALUE = 1",
                file_path="app.py",
                excerpt="VALUE = 1",
                confidence=1.0,
            )
        ],
        used_skill_refs=used or [],
    )


@pytest.mark.asyncio
async def test_leaf_tool_loop_executes_real_tool_and_projects_events(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="read-1",
                        name="read_file",
                        arguments={"path": "app.py"},
                    )
                ]
            ),
            ToolModelResponse(content="enough evidence"),
        ],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    loop = BoundedToolAgent(gateway, recorder, max_iterations=4, max_tool_calls=4)

    result = await loop.run(
        run_id="tool-run",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="Inspect the fixture.",
        task_prompt="Read app.py.",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )

    assert result.evidence[0].file_path == "app.py"
    assert "VALUE = 1" in gateway.messages_seen[1][-1].content
    events = recorder.events("tool-run")
    assert sum(event.type == EventType.MODEL_CALL for event in events) == 3
    assert sum(event.type == EventType.TOOL_CALL for event in events) == 1
    assert sum(event.type == EventType.TOOL_RESULT for event in events) == 1
    view = recorder.build_view(
        run_id="tool-run",
        verification_history=[],
        final_status="success",
        failure_reason=None,
    )
    assert view.tool_call_count == 1
    assert view.tool_calls[0].success is True


@pytest.mark.asyncio
async def test_leaf_invocation_ids_preserve_repeated_fixer_and_reviewer_turns(
    tmp_path: Path,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite")
    recorder = TrajectoryRecorder(store)
    gateway = ScriptedToolGateway(
        [ToolModelResponse(content="done") for _ in range(4)],
        worker_result(),
    )
    loop = BoundedToolAgent(gateway, recorder, max_iterations=2, max_tool_calls=2)
    tools = create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path)

    for agent_id, invocation_ids in {
        "fixer": ["repair:1", "repair:2"],
        "reviewer": ["review:1", "review:2"],
    }.items():
        for invocation_id in invocation_ids:
            await loop.run(
                run_id="repeated-leaf-run",
                agent_id=agent_id,
                invocation_id=invocation_id,
                system_prompt="Return a result.",
                task_prompt="Finish without tools.",
                tools=tools,
                output_schema=WorkerResult,
            )

    turn_events = [
        event
        for event in store.list("repeated-leaf-run")
        if event.type == EventType.MODEL_CALL
        and event.payload.get("phase") == "tool_loop"
        and event.payload.get("iteration") == 1
    ]
    assert len([event for event in turn_events if event.agent_id == "fixer"]) == 2
    assert len([event for event in turn_events if event.agent_id == "reviewer"]) == 2
    assert {event.invocation_id for event in turn_events} == {
        "repair:1",
        "repair:2",
        "review:1",
        "review:2",
    }
    store.close()


def skill_candidate() -> SkillCandidate:
    return SkillCandidate(
        name="fixture-inspector",
        description="Inspect fixture values",
        triggers=["fixture value"],
        task_families=["test"],
        skill_md="""---
name: fixture-inspector
description: Inspect fixture values
version: 1
---

# Purpose
Inspect fixture values.
# When to Use
Use for fixture value failures.
# Procedure
Run the bundled inspector.
# Pitfalls
Do not modify the fixture.
# Verification
Read the printed value.
# Bundled Resources
Use `scripts/inspect.py`.
""",
        scripts=[GeneratedFile(path="scripts/inspect.py", content="print('skill-ok')\n")],
        source_run_ids=["source-run"],
        confidence=0.9,
        permissions=SkillPermissions(execute=True),
    )


@pytest.mark.asyncio
async def test_run_skill_script_is_formal_tool_and_emits_skill_used(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_candidate(skill_candidate())
    assert (
        CandidateValidator(registry)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )
    ref = SkillRef(skill_id=created.manifest.skill_id, version=created.manifest.version)
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="skill-1",
                        name="run_skill_script",
                        arguments={
                            "skill_id": ref.skill_id,
                            "version": ref.version,
                            "script_name": "inspect.py",
                            "args": [],
                        },
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=registry,
        allowed_skill_refs={(ref.skill_id, ref.version)},
    )

    result = await BoundedToolAgent(gateway, recorder, max_iterations=4, max_tool_calls=4).run(
        run_id="skill-run",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="Use the selected capability.",
        task_prompt="Inspect.",
        tools=tools,
        output_schema=WorkerResult,
    )

    assert result.used_skill_refs == [ref]
    skill_events = [
        event for event in recorder.events("skill-run") if event.type == EventType.SKILL_USED
    ]
    assert skill_events[0].payload["resource"] == "inspect.py"
    assert skill_events[0].payload["success"] is True
    registry.close()


@pytest.mark.asyncio
async def test_trajectory_detects_reusable_helper_created_by_real_tool(tmp_path: Path) -> None:
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="helper-1",
                        name="apply_patch",
                        arguments={
                            "files": {"tools/diagnostic_helper.py": "print('diagnostic')\n"}
                        },
                    )
                ]
            ),
            ToolModelResponse(content="helper created"),
        ],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    await BoundedToolAgent(gateway, recorder, max_iterations=4, max_tool_calls=4).run(
        run_id="helper-run",
        agent_id="fixer",
        invocation_id="repair:1",
        system_prompt="Create a reusable diagnostic helper.",
        task_prompt="Inspect the failure.",
        tools=create_worker_registry(FIXER_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )

    view = recorder.build_view(
        run_id="helper-run",
        verification_history=[],
        final_status="success",
        failure_reason=None,
    )
    assert view.created_files == ["tools/diagnostic_helper.py"]
    assert view.reusable_script_created


class ParallelInvestigatorGateway(ScriptedToolGateway):
    def __init__(self) -> None:
        super().__init__([], worker_result())
        self.calls: dict[str, int] = {}

    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
    ) -> ToolModelResponse:
        del messages, tools
        count = self.calls.get(agent_id, 0)
        self.calls[agent_id] = count + 1
        if count == 0:
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id=f"{agent_id}:read",
                        name="read_file",
                        arguments={"path": "calculator.py"},
                    )
                ]
            )
        return ToolModelResponse(content="evidence gathered")

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
    ) -> ResponseT:
        del messages
        task_id = agent_id.split(":", 1)[-1]
        return response_model.model_validate(
            WorkerResult(
                task_id=task_id,
                summary="read repository",
                evidence=[
                    EvidenceItem(
                        source_agent=agent_id,
                        kind="source_code",
                        claim="calculator currently subtracts",
                        file_path="calculator.py",
                        confidence=0.9,
                    )
                ],
            ).model_dump()
        )


@pytest.mark.asyncio
async def test_langgraph_fanout_workers_each_use_real_leaf_tools(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n"
    )
    (tmp_path / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import add\n"
        "class TestCalculator(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )
    recorder = TrajectoryRecorder()
    gateway = ParallelInvestigatorGateway()
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            coordinator=DemoCoordinator(),
            investigator=ModelInvestigator(gateway, recorder),
            diagnoser=DemoDiagnoser(),
            fixer=DemoFixer(),
            reviewer=DemoReviewer(),
        ),
        recorder=recorder,
    )

    result = await build_graph(runtime).ainvoke(
        {
            "run_id": "multi-tool-run",
            "task_id": "multi-tool",
            "repo": RepoSpec(owner="fixture", name="calculator"),
            "ci_failure": CIFailure(
                summary="addition failed",
                log_excerpt="AssertionError: -1 != 3",
                failed_commands=[["python", "-m", "unittest", "-q"]],
                task_family="test",
            ),
            "workspace_path": str(tmp_path),
        }
    )

    assert result["status"] == "success"
    tool_agents = {
        event.agent_id
        for event in recorder.events("multi-tool-run")
        if event.type == EventType.TOOL_CALL and event.agent_id != "harness"
    }
    assert tool_agents == {
        "investigator:logs",
        "investigator:repository",
        "investigator:test",
    }
    assert len(result["evidence"]) == 3


class FullLeafGateway(ParallelInvestigatorGateway):
    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
    ) -> ToolModelResponse:
        del messages, tools
        count = self.calls.get(agent_id, 0)
        self.calls[agent_id] = count + 1
        if agent_id.startswith("investigator:"):
            if count == 0:
                return ToolModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            call_id=f"{agent_id}:read",
                            name="read_file",
                            arguments={"path": "calculator.py"},
                        )
                    ]
                )
        elif agent_id == "fixer":
            if count == 0:
                return ToolModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            call_id="fixer:read",
                            name="read_file",
                            arguments={"path": "calculator.py"},
                        )
                    ]
                )
            if count == 1:
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
            if count == 2:
                return ToolModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            call_id="fixer:test",
                            name="run_test",
                            arguments={"argv": ["python", "-m", "unittest", "-q"]},
                        )
                    ]
                )
        elif agent_id == "reviewer" and count == 0:
            return ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="reviewer:test",
                        name="run_test",
                        arguments={"argv": ["python", "-m", "unittest", "-q"]},
                    )
                ]
            )
        return ToolModelResponse(content="ready for structured output")

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
    ) -> ResponseT:
        del messages
        if response_model is WorkerResult:
            task_id = agent_id.split(":", 1)[-1]
            output: BaseModel = WorkerResult(
                task_id=task_id,
                summary="repository inspected",
                evidence=[
                    EvidenceItem(
                        source_agent=agent_id,
                        kind="source_code",
                        claim="calculator subtracts instead of adding",
                        file_path="calculator.py",
                        confidence=0.95,
                    )
                ],
            )
        elif response_model is FixerOutput:
            output = FixerOutput(
                proposal=PatchProposal(
                    summary="replace subtraction with addition",
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


@pytest.mark.asyncio
async def test_full_multi_agent_path_uses_tools_in_every_leaf_role(tmp_path: Path) -> None:
    (tmp_path / "calculator.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left - right\n"
    )
    (tmp_path / "test_calculator.py").write_text(
        "import unittest\n"
        "from calculator import add\n"
        "class TestCalculator(unittest.TestCase):\n"
        "    def test_add(self) -> None:\n"
        "        self.assertEqual(add(1, 2), 3)\n"
    )
    recorder = TrajectoryRecorder()
    gateway = FullLeafGateway()
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            coordinator=DemoCoordinator(),
            investigator=ModelInvestigator(gateway, recorder),
            diagnoser=DemoDiagnoser(),
            fixer=ModelFixer(gateway, recorder),
            reviewer=ModelReviewer(gateway, recorder),
        ),
        recorder=recorder,
    )

    result = await build_graph(runtime).ainvoke(
        {
            "run_id": "full-leaf-run",
            "task_id": "full-leaf",
            "repo": RepoSpec(owner="fixture", name="calculator"),
            "ci_failure": CIFailure(
                summary="addition failed",
                log_excerpt="AssertionError: -1 != 3",
                failed_commands=[["python", "-m", "unittest", "-q"]],
                task_family="test",
            ),
            "workspace_path": str(tmp_path),
        }
    )

    assert result["status"] == "success"
    assert (tmp_path / "calculator.py").read_text().endswith("return left + right\n")
    events = recorder.events("full-leaf-run")
    assert any(event.type == EventType.MODEL_CALL for event in events)
    assert any(event.type == EventType.TOOL_RESULT for event in events)
    assert sum(event.type == EventType.EVIDENCE_CREATED for event in events) == 3
    leaf_tool_agents = {
        event.agent_id
        for event in events
        if event.type == EventType.TOOL_CALL and event.agent_id != "harness"
    }
    assert "fixer" in leaf_tool_agents
    assert "reviewer" in leaf_tool_agents
    assert len([agent for agent in leaf_tool_agents if agent.startswith("investigator:")]) == 3
    view = recorder.build_view(
        run_id="full-leaf-run",
        verification_history=result["verification_history"],
        final_status="success",
        failure_reason=None,
    )
    assert view.tool_call_count == sum(event.type == EventType.TOOL_CALL for event in events)
