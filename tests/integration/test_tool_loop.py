from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel

from evoci.agents.base import AgentSuite, WorkerContext
from evoci.agents.model_agents import (
    ModelWorker,
    StagedFixerPlan,
)
from evoci.agents.tool_loop import BoundedToolAgent
from evoci.benchmark.execution import token_metrics_from_events
from evoci.capability.models import GeneratedFile, SkillCandidate, SkillPermissions
from evoci.capability.registry import CapabilityRegistry
from evoci.config import EvoCIConfig
from evoci.demo import DemoCoordinator, DemoInvestigator
from evoci.domain.models import (
    CIFailure,
    EvidenceItem,
    RepoSpec,
    ReviewResult,
    SkillCatalogEntry,
    SkillRef,
    VerificationCommandSpec,
    WorkerResult,
    WorkerTask,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.model.gateway import (
    ModelUsage,
    ResponseT,
    ToolCallRequest,
    ToolDefinition,
    ToolLoopMessage,
    ToolModelResponse,
)
from evoci.runtime.budget import RepairBudgetExhausted, RunBudgetManager, RunRepairBudget
from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.policy import (
    FIXER_CAPABILITIES,
    INVESTIGATOR_CAPABILITIES,
    WORKER_REPAIR_CAPABILITIES,
)
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
        usage_observer: object | None = None,
    ) -> ResponseT:
        del system_prompt, user_prompt, response_model, agent_id, usage_observer
        raise AssertionError("one-shot completion is not used by a leaf tool loop")

    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
        usage_observer: object | None = None,
    ) -> ToolModelResponse:
        del tools, agent_id, usage_observer
        self.messages_seen.append(list(messages))
        return self.turns.pop(0)

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
        usage_observer: object | None = None,
    ) -> ResponseT:
        del messages, agent_id, usage_observer
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
                reasoning_content="the fixture file should contain VALUE",
                tool_calls=[
                    ToolCallRequest(
                        call_id="read-1",
                        name="read_file",
                        arguments={"path": "app.py"},
                    )
                ],
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
    assistant = next(message for message in gateway.messages_seen[1] if message.role == "assistant")
    assert assistant.reasoning_content == "the fixture file should contain VALUE"
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
async def test_large_tool_result_is_projected_and_can_be_read_on_demand(
    tmp_path: Path,
) -> None:
    content = "".join(str(index % 10) for index in range(25_000))
    (tmp_path / "large.txt").write_text(content)
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="read-1",
                        name="read_file",
                        arguments={"path": "large.txt"},
                    )
                ]
            ),
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="read-more-1",
                        name="read_tool_result",
                        arguments={"result_id": "read-1", "offset": 12_000, "limit": 200},
                    )
                ]
            ),
            ToolModelResponse(content="enough evidence"),
        ],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    loop = BoundedToolAgent(gateway, recorder, max_iterations=4, max_tool_calls=4)

    await loop.run(
        run_id="large-result-run",
        agent_id="investigator:inspect",
        invocation_id="investigation:large",
        system_prompt="Inspect the fixture.",
        task_prompt="Read large.txt.",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )

    projected = json.loads(gateway.messages_seen[1][-1].content)
    assert projected["result"]["projected"] is True
    assert projected["result"]["result_id"] == "read-1"
    assert projected["result"]["total_chars"] > 25_000
    assert len(gateway.messages_seen[1][-1].content) < 13_000

    chunk = json.loads(gateway.messages_seen[2][-1].content)["result"]
    assert chunk["result_id"] == "read-1"
    assert chunk["offset"] == 12_000
    assert len(chunk["content"]) == 200
    assert chunk["next_offset"] == 12_200

    original = next(
        event
        for event in recorder.events("large-result-run")
        if event.type == EventType.TOOL_RESULT and event.payload["call_id"] == "read-1"
    )
    assert original.payload["result"] == content


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
UNIQUE_SKILL_BODY_MARKER
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
    created = registry.create_skill(skill_candidate())
    ref = SkillRef(skill_id=created.manifest.skill_id)
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-1",
                        name="load_skill",
                        arguments={"skill_id": ref.skill_id},
                    )
                ]
            ),
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="skill-1",
                        name="run_skill_script",
                        arguments={
                            "skill_id": ref.skill_id,
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
        allowed_skill_refs={ref.skill_id},
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
async def test_unselected_skill_invocation_is_rejected_not_counted(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_skill(skill_candidate())
    ref = SkillRef(skill_id=created.manifest.skill_id)
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="skill-rejected",
                        name="run_skill_script",
                        arguments={
                            "skill_id": ref.skill_id,
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
        allowed_skill_refs=set(),
    )
    result = await BoundedToolAgent(gateway, recorder, max_iterations=4, max_tool_calls=4).run(
        run_id="skill-reject",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="Use a skill.",
        task_prompt="Inspect.",
        tools=tools,
        output_schema=WorkerResult,
    )
    assert result.used_skill_refs == []
    events = recorder.events("skill-reject")
    assert any(event.type == EventType.SKILL_INVOCATION_REJECTED for event in events)
    assert not any(event.type == EventType.SKILL_USED for event in events)
    assert registry.stats(ref.skill_id).use_count == 0
    registry.close()


@pytest.mark.asyncio
async def test_failed_skill_script_is_recorded_as_use_failure(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_skill(
        skill_candidate().model_copy(
            update={
                "scripts": [
                    GeneratedFile(
                        path="scripts/inspect.py",
                        content="raise SystemExit(1)\n",
                    )
                ]
            }
        )
    )
    ref = SkillRef(skill_id=created.manifest.skill_id)
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-fail",
                        name="load_skill",
                        arguments={"skill_id": ref.skill_id},
                    )
                ]
            ),
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="skill-fail",
                        name="run_skill_script",
                        arguments={
                            "skill_id": ref.skill_id,
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
        allowed_skill_refs={ref.skill_id},
    )
    await BoundedToolAgent(gateway, recorder, max_iterations=4, max_tool_calls=4).run(
        run_id="skill-fail",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="Use the selected capability.",
        task_prompt="Inspect.",
        tools=tools,
        output_schema=WorkerResult,
    )
    used = [
        event for event in recorder.events("skill-fail") if event.type == EventType.SKILL_USED
    ]
    assert len(used) == 1
    assert used[0].payload["usage_kind"] == "script"
    assert used[0].payload["success"] is False
    registry.close()


@pytest.mark.asyncio
async def test_repeated_skill_script_calls_keep_each_trace(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_skill(
        skill_candidate().model_copy(
            update={
                "scripts": [
                    GeneratedFile(
                        path="scripts/inspect.py",
                        content=(
                            "from pathlib import Path\n"
                            "marker = Path(__file__).with_name('.ran')\n"
                            "if marker.exists():\n"
                            "    raise SystemExit(1)\n"
                            "marker.write_text('1')\n"
                            "print('ok')\n"
                        ),
                    )
                ]
            }
        )
    )
    ref = SkillRef(skill_id=created.manifest.skill_id)
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-repeat",
                        name="load_skill",
                        arguments={"skill_id": ref.skill_id},
                    )
                ]
            ),
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="skill-1",
                        name="run_skill_script",
                        arguments={
                            "skill_id": ref.skill_id,
                            "script_name": "inspect.py",
                            "args": [],
                        },
                    )
                ]
            ),
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="skill-2",
                        name="run_skill_script",
                        arguments={
                            "skill_id": ref.skill_id,
                            "script_name": "inspect.py",
                            "args": [],
                        },
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        worker_result(used=[ref]),
    )
    recorder = TrajectoryRecorder()
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=registry,
        allowed_skill_refs={ref.skill_id},
    )
    await BoundedToolAgent(gateway, recorder, max_iterations=6, max_tool_calls=6).run(
        run_id="skill-repeat",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="Use the selected capability.",
        task_prompt="Inspect twice.",
        tools=tools,
        output_schema=WorkerResult,
    )
    used = [
        event
        for event in recorder.events("skill-repeat")
        if event.type == EventType.SKILL_USED
        and event.payload.get("usage_kind") == "script"
    ]
    assert len(used) == 2
    assert [event.payload.get("success") for event in used] == [True, False]
    traces = recorder.build_view(
        run_id="skill-repeat",
        verification_history=[],
        final_status="success",
        failure_reason=None,
    ).skill_use_traces
    script_traces = [trace for trace in traces if trace.usage_kind == "script"]
    assert len(script_traces) == 2
    assert {trace.execution_success for trace in script_traces} == {True, False}
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
        usage_observer: object | None = None,
    ) -> ToolModelResponse:
        del messages, tools, usage_observer
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
        usage_observer: object | None = None,
    ) -> ResponseT:
        del messages, usage_observer
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
    _gateway = ParallelInvestigatorGateway()
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=DemoCoordinator(),
            worker=DemoInvestigator(),
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
    assert tool_agents <= {
        "worker:repository",
        "worker:test",
        "worker:repair-add",
        "supervisor",
    }
    assert len(result["evidence"]) >= 1


class FullLeafGateway(ParallelInvestigatorGateway):
    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
        usage_observer: object | None = None,
    ) -> ToolModelResponse:
        del messages, tools, usage_observer
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
        usage_observer: object | None = None,
    ) -> ResponseT:
        del messages, usage_observer
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
        elif response_model is StagedFixerPlan:
            output = StagedFixerPlan(
                summary="replace subtraction with addition",
                commands_run=["python -m unittest -q"],
                risk="low",
                verification_plan=[["python", "-m", "unittest", "-q"]],
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
    _gateway = FullLeafGateway()
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=DemoCoordinator(),
            worker=DemoInvestigator(),
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
    assert any(event.type == EventType.TOOL_RESULT for event in events)
    assert sum(event.type == EventType.EVIDENCE_CREATED for event in events) >= 1
    view = recorder.build_view(
        run_id="full-leaf-run",
        verification_history=result["verification_history"],
        final_status="success",
        failure_reason=None,
    )
    assert view.tool_call_count == sum(event.type == EventType.TOOL_CALL for event in events)


def _fixer_context(workspace: Path, write_scope: list[str] | None = None) -> WorkerContext:
    from evoci.domain.models import WorkerTask

    return WorkerContext(
        run_id="fixer-staging",
        repo=RepoSpec(name="fixture"),
        failure=CIFailure(summary="failed", log_excerpt="error"),
        workspace_path=str(workspace),
        invocation_id="repair:1",
        task=WorkerTask(
            task_id="repair",
            kind="repair",
            objective="repair the fixture",
            write_scope=write_scope or ["app.py"],
        ),
    )


def _staged_plan() -> StagedFixerPlan:
    return StagedFixerPlan(
        summary="replace the failing assignment",
        commands_run=["python -m unittest -q"],
        risk="low",
        verification_plan=[["python", "-m", "unittest", "-q"]],
    )


@pytest.mark.asyncio
async def test_model_fixer_keeps_original_workspace_unchanged(tmp_path: Path) -> None:
    original = "VALUE = 1\n"
    (tmp_path / "app.py").write_text(original, encoding="utf-8")

    class IsolationGateway(ScriptedToolGateway):
        async def next_action(
            self,
            *,
            messages: list[ToolLoopMessage],
            tools: list[ToolDefinition],
            agent_id: str,
            usage_observer: object | None = None,
        ) -> ToolModelResponse:
            del tools, agent_id, usage_observer
            if messages:
                assert (tmp_path / "app.py").read_text(encoding="utf-8") == original
            return self.turns.pop(0)

    gateway = IsolationGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="fixer:replace",
                        name="replace_text",
                        arguments={
                            "path": "app.py",
                            "old_text": "VALUE = 1",
                            "new_text": "VALUE = 2",
                        },
                    )
                ]
            ),
            ToolModelResponse(content="staged"),
        ],
        _staged_plan(),
    )
    run = await ModelWorker(gateway, TrajectoryRecorder()).execute(
        context=_fixer_context(tmp_path),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    output = run
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == original
    assert [edit.path for edit in output.edits] == ["app.py"]
    assert output.edits[0].content == "VALUE = 2\n"


@pytest.mark.asyncio
async def test_model_fixer_collects_large_file_without_model_repeating_it(
    tmp_path: Path,
) -> None:
    original = "HEADER = 'keep'\n" + ("x" * 120_000) + "\nMARKER = 'old'\n"
    (tmp_path / "app.py").write_text(original, encoding="utf-8")
    plan = StagedFixerPlan(
        summary="flip the marker",
        risk="low",
        verification_plan=[["python", "-c", "raise SystemExit(0)"]],
    )
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="fixer:replace",
                        name="replace_text",
                        arguments={
                            "path": "app.py",
                            "old_text": "MARKER = 'old'",
                            "new_text": "MARKER = 'new'",
                        },
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        plan,
    )
    run = await ModelWorker(gateway, TrajectoryRecorder()).execute(
        context=_fixer_context(tmp_path),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    output = run
    assert output.result.summary == "flip the marker"
    assert [edit.path for edit in output.edits] == ["app.py"]
    assert output.edits[0].content is not None
    assert len(output.edits[0].content) > 100_000
    assert "MARKER = 'new'" in output.edits[0].content
    assert "MARKER = 'old'" not in output.edits[0].content
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == original
    dumped_plan = json.dumps(plan.model_dump())
    assert "x" * 1000 not in dumped_plan
    assert original not in dumped_plan


@pytest.mark.asyncio
async def test_fixer_prompt_treats_failed_episodes_as_counterevidence(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    gateway = ScriptedToolGateway(
        [ToolModelResponse(content="done")],
        _staged_plan(),
    )
    await ModelWorker(gateway, TrajectoryRecorder()).execute(
        context=_fixer_context(tmp_path),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    system = gateway.messages_seen[0][0].content
    assert "write_scope" in system
    assert "file contents" in system.lower()
    assert "benchmark" not in system.lower()
    assert "task_id" not in system.lower()


class ObservingToolGateway(ScriptedToolGateway):
    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
        usage_observer: object | None = None,
    ) -> ToolModelResponse:
        response = await super().next_action(
            messages=messages, tools=tools, agent_id=agent_id, usage_observer=usage_observer
        )
        if callable(usage_observer):
            usage_observer(
                ModelUsage(input_tokens=5, output_tokens=1, request_kind="tool_action")
            )
        return response.model_copy(update={"input_tokens": 5, "output_tokens": 1})

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
        usage_observer: object | None = None,
    ) -> ResponseT:
        if callable(usage_observer):
            usage_observer(
                ModelUsage(
                    input_tokens=7, output_tokens=2, request_kind="structured_finalize"
                )
            )
        return await super().finalize(
            messages=messages,
            response_model=response_model,
            agent_id=agent_id,
            usage_observer=usage_observer,
        )


@pytest.mark.asyncio
async def test_repair_model_usage_has_repair_scope(tmp_path: Path) -> None:
    gateway = ObservingToolGateway([ToolModelResponse(content="done")], worker_result())
    recorder = TrajectoryRecorder()
    await BoundedToolAgent(gateway, recorder, max_iterations=2, max_tool_calls=2).run(
        run_id="usage-repair",
        agent_id="fixer",
        invocation_id="repair:1",
        system_prompt="Fix",
        task_prompt="Fix it",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )
    usages = [
        event for event in recorder.events("usage-repair") if event.type == EventType.MODEL_USAGE
    ]
    assert usages
    assert all(event.payload["budget_scope"] == "repair" for event in usages)
    assert {event.payload["request_kind"] for event in usages} == {
        "tool_action",
        "structured_finalize",
    }


@pytest.mark.asyncio
async def test_model_usage_events_have_unique_keys(tmp_path: Path) -> None:
    gateway = ObservingToolGateway(
        [ToolModelResponse(content="first"), ToolModelResponse(content="second")],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    await BoundedToolAgent(gateway, recorder, max_iterations=3, max_tool_calls=3).run(
        run_id="usage-unique",
        agent_id="fixer",
        invocation_id="repair:1",
        system_prompt="Fix",
        task_prompt="Fix it",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )
    usages = [
        event for event in recorder.events("usage-unique") if event.type == EventType.MODEL_USAGE
    ]
    ids = [event.event_id for event in usages]
    assert len(ids) >= 2
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
async def test_tool_response_usage_is_not_recorded_twice(tmp_path: Path) -> None:
    gateway = ObservingToolGateway([ToolModelResponse(content="done")], worker_result())
    recorder = TrajectoryRecorder()
    await BoundedToolAgent(gateway, recorder, max_iterations=2, max_tool_calls=2).run(
        run_id="usage-once",
        agent_id="fixer",
        invocation_id="repair:1",
        system_prompt="Fix",
        task_prompt="Fix it",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )
    events = recorder.events("usage-once")
    metrics = token_metrics_from_events(events)
    usage_events = [event for event in events if event.type == EventType.MODEL_USAGE]
    expected = sum(
        int(event.payload["input_tokens"]) + int(event.payload["output_tokens"])
        for event in usage_events
    )
    assert metrics["total_tokens"] == expected
    assert metrics["provider_requests"] == len(usage_events)
    model_calls = [event for event in events if event.type == EventType.MODEL_CALL]
    call_tokens = sum(int(event.payload.get("input_tokens") or 0) for event in model_calls)
    assert call_tokens > 0
    assert metrics["input_tokens"] != call_tokens + sum(
        int(event.payload["input_tokens"]) for event in usage_events
    )


SKILL_BODY_MARKER = "UNIQUE_SKILL_BODY_MARKER"


def _catalog_context(tmp_path: Path, skill_id: str, name: str, description: str) -> WorkerContext:
    from evoci.domain.models import WorkerTask

    return WorkerContext(
        run_id="prompt-run",
        repo=RepoSpec(owner="org", name="repo"),
        failure=CIFailure(summary="fixture failed", log_excerpt="failed", task_family="test"),
        workspace_path=str(tmp_path),
        invocation_id="investigation:inspect",
        task=WorkerTask(
            task_id="inspect",
            kind="investigate",
            objective="inspect",
        ),
        recommended_skills=(
            SkillCatalogEntry(skill_id=skill_id, name=name, description=description),
        ),
    )


@pytest.mark.asyncio
async def test_initial_investigator_prompt_contains_skill_metadata_only(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    gateway = ScriptedToolGateway([ToolModelResponse(content="done")], worker_result())
    investigator = ModelWorker(gateway, TrajectoryRecorder(), capability_registry=store)
    await investigator.execute(
        context=_catalog_context(
            tmp_path, created.manifest.skill_id, created.manifest.name, created.manifest.description
        ),
        capabilities=INVESTIGATOR_CAPABILITIES,
    )
    blob = json.dumps(gateway.messages_seen, default=str)
    assert created.manifest.skill_id in blob
    assert created.manifest.description in blob
    assert SKILL_BODY_MARKER not in blob
    store.close()


@pytest.mark.asyncio
async def test_initial_fixer_prompt_contains_skill_metadata_only(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    gateway = ScriptedToolGateway([ToolModelResponse(content="done")], _staged_plan())
    fixer = ModelWorker(gateway, TrajectoryRecorder(), capability_registry=store)
    context = _catalog_context(
        tmp_path, created.manifest.skill_id, created.manifest.name, created.manifest.description
    )
    context = context.__class__(
        **{
            **{field: getattr(context, field) for field in context.__dataclass_fields__},
            "task": context.task.model_copy(
                update={"kind": "repair", "write_scope": ["app.py"], "task_id": "repair"}
            ),
            "invocation_id": "repair:1",
        }
    )
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    await fixer.execute(context=context, capabilities=WORKER_REPAIR_CAPABILITIES)
    blob = json.dumps(gateway.messages_seen, default=str)
    assert created.manifest.skill_id in blob
    assert SKILL_BODY_MARKER not in blob
    store.close()


@pytest.mark.asyncio
async def test_skill_body_absent_before_load(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    gateway = ScriptedToolGateway([ToolModelResponse(content="done")], worker_result())
    await BoundedToolAgent(gateway, TrajectoryRecorder(), max_iterations=2, max_tool_calls=2).run(
        run_id="no-load",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="inspect",
        task_prompt=json.dumps({"skills": [{"skill_id": created.manifest.skill_id}]}),
        tools=tools,
        output_schema=WorkerResult,
    )
    blob = json.dumps(gateway.messages_seen, default=str)
    assert SKILL_BODY_MARKER not in blob
    tools.close()
    store.close()


@pytest.mark.asyncio
async def test_skill_body_appears_once_after_load(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-once",
                        name="load_skill",
                        arguments={"skill_id": created.manifest.skill_id},
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        worker_result(),
    )
    await BoundedToolAgent(gateway, TrajectoryRecorder(), max_iterations=3, max_tool_calls=3).run(
        run_id="load-once",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=tools,
        output_schema=WorkerResult,
    )
    blob = json.dumps(gateway.messages_seen, default=str)
    assert blob.count(SKILL_BODY_MARKER) == 1
    tools.close()
    store.close()


@pytest.mark.asyncio
async def test_skill_memory_appears_once_after_load(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from evoci.capability.models import SkillMemoryEntry

    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    lesson = "UNIQUE_SKILL_MEMORY_MARKER"
    store.append_skill_memory(
        created.manifest.skill_id,
        SkillMemoryEntry(
            run_id="mem-1",
            repository="org/repo",
            task_summary="fixture",
            outcome="success",
            lesson=lesson,
            created_at=datetime.now(UTC),
        ),
    )
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-mem",
                        name="load_skill",
                        arguments={"skill_id": created.manifest.skill_id},
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        worker_result(),
    )
    await BoundedToolAgent(gateway, TrajectoryRecorder(), max_iterations=3, max_tool_calls=3).run(
        run_id="load-mem",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=tools,
        output_schema=WorkerResult,
    )
    blob = json.dumps(gateway.messages_seen, default=str)
    assert blob.count(lesson) == 1
    tools.close()
    store.close()


@pytest.mark.asyncio
async def test_repeated_load_skill_injects_body_and_memory_once(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from evoci.capability.models import SkillMemoryEntry

    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    memory_marker = "UNIQUE_SKILL_MEMORY_MARKER"
    store.append_skill_memory(
        created.manifest.skill_id,
        SkillMemoryEntry(
            run_id="mem-repeat",
            repository="org/repo",
            task_summary="fixture",
            outcome="success",
            lesson=memory_marker,
            created_at=datetime.now(UTC),
        ),
    )
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-first",
                        name="load_skill",
                        arguments={"skill_id": created.manifest.skill_id},
                    )
                ]
            ),
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-second",
                        name="load_skill",
                        arguments={"skill_id": created.manifest.skill_id},
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        worker_result(),
    )
    await BoundedToolAgent(gateway, TrajectoryRecorder(), max_iterations=4, max_tool_calls=4).run(
        run_id="load-twice",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=tools,
        output_schema=WorkerResult,
    )
    history = json.dumps(gateway.messages_seen[-1], default=str)
    assert history.count(SKILL_BODY_MARKER) == 1
    assert history.count(memory_marker) == 1
    tool_payloads = [
        json.loads(message.content)
        for message in gateway.messages_seen[-1]
        if message.role == "tool" and message.content
    ]
    assert any(payload.get("result", {}).get("already_loaded") is True for payload in tool_payloads)
    assert sum("procedure" in (payload.get("result") or {}) for payload in tool_payloads) == 1
    tools.close()
    store.close()


@pytest.mark.asyncio
async def test_structured_context_does_not_duplicate_skill_body(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    gateway = ScriptedToolGateway([ToolModelResponse(content="done")], worker_result())
    investigator = ModelWorker(gateway, TrajectoryRecorder(), capability_registry=store)
    await investigator.execute(
        context=_catalog_context(
            tmp_path, created.manifest.skill_id, created.manifest.name, created.manifest.description
        ),
        capabilities=INVESTIGATOR_CAPABILITIES,
    )
    blob = json.dumps(gateway.messages_seen, default=str)
    assert blob.count(SKILL_BODY_MARKER) == 0
    store.close()


@pytest.mark.asyncio
async def test_visible_skill_is_retrieved_not_selected(tmp_path: Path) -> None:
    from evoci.capability.retrieval import CapabilityRetriever
    from tests.integration.test_graph import (
        FakeCoordinator,
        FakeDiagnoser,
        FakeInvestigator,
        make_runtime,
    )

    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    store.create_skill(skill_candidate())
    recorder = TrajectoryRecorder()
    runtime = replace(
        make_runtime(tmp_path, FakeCoordinator([[]]), FakeInvestigator(), FakeDiagnoser()),
        capability_registry=store,
        capability_retriever=CapabilityRetriever(store),
        recorder=recorder,
    )
    from tests.integration.test_graph import initial_state

    result = await build_graph(runtime).ainvoke(initial_state(tmp_path, run_id="catalog-only"))
    events = recorder.events("catalog-only")
    assert any(event.type == EventType.SKILL_RETRIEVED for event in events)
    assert not any(event.type == EventType.SKILL_SELECTED for event in events)
    assert result.get("skill_catalog")
    store.close()


@pytest.mark.asyncio
async def test_loaded_skill_is_selected(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    recorder = TrajectoryRecorder()
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-sel",
                        name="load_skill",
                        arguments={"skill_id": created.manifest.skill_id},
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        worker_result(),
    )
    await BoundedToolAgent(gateway, recorder, max_iterations=3, max_tool_calls=3).run(
        run_id="selected-run",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=tools,
        output_schema=WorkerResult,
    )
    events = recorder.events("selected-run")
    selected = [event for event in events if event.type == EventType.SKILL_SELECTED]
    used = [event for event in events if event.type == EventType.SKILL_USED]
    assert len(selected) == 1
    assert used == []
    tools.close()
    store.close()


@pytest.mark.asyncio
async def test_loaded_but_unclaimed_skill_is_not_used(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-unclaimed",
                        name="load_skill",
                        arguments={"skill_id": created.manifest.skill_id},
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        worker_result(),
    )
    result = await BoundedToolAgent(
        gateway, TrajectoryRecorder(), max_iterations=3, max_tool_calls=3
    ).run(
        run_id="unclaimed",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=tools,
        output_schema=WorkerResult,
    )
    assert result.used_skill_refs == []
    tools.close()
    store.close()


@pytest.mark.asyncio
async def test_unloaded_claimed_skill_is_discarded(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    claimed = worker_result(used=[SkillRef(skill_id=created.manifest.skill_id)])
    gateway = ScriptedToolGateway([ToolModelResponse(content="done")], claimed)
    result = await BoundedToolAgent(
        gateway, TrajectoryRecorder(), max_iterations=2, max_tool_calls=2
    ).run(
        run_id="claimed-unloaded",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=tools,
        output_schema=WorkerResult,
    )
    assert result.used_skill_refs == []
    tools.close()
    store.close()


@pytest.mark.asyncio
async def test_production_investigator_clears_unloaded_skill_claim(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    claimed = worker_result(used=[SkillRef(skill_id=created.manifest.skill_id)])
    gateway = ScriptedToolGateway([ToolModelResponse(content="done")], claimed)
    investigator = ModelWorker(gateway, TrajectoryRecorder(), capability_registry=store)
    result = await investigator.execute(
        context=_catalog_context(
            tmp_path, created.manifest.skill_id, created.manifest.name, created.manifest.description
        ),
        capabilities=INVESTIGATOR_CAPABILITIES,
    )
    assert result.result.used_skill_refs == []
    store.close()


@pytest.mark.asyncio
async def test_successful_skill_script_counts_as_used(tmp_path: Path) -> None:
    store = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = store.create_skill(skill_candidate())
    tools = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load-used",
                        name="load_skill",
                        arguments={"skill_id": created.manifest.skill_id},
                    )
                ]
            ),
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="script-used",
                        name="run_skill_script",
                        arguments={
                            "skill_id": created.manifest.skill_id,
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
    result = await BoundedToolAgent(
        gateway, TrajectoryRecorder(), max_iterations=4, max_tool_calls=4
    ).run(
        run_id="script-used",
        agent_id="investigator:inspect",
        invocation_id="investigation:inspect",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=tools,
        output_schema=WorkerResult,
    )
    assert result.used_skill_refs == [SkillRef(skill_id=created.manifest.skill_id)]
    tools.close()
    store.close()


def _tool_turn(call_id: str, name: str = "read_file", **arguments: object) -> ToolModelResponse:
    return ToolModelResponse(
        tool_calls=[ToolCallRequest(call_id=call_id, name=name, arguments=dict(arguments))]
    )


class CountingGateway(ScriptedToolGateway):
    def __init__(
        self,
        turns: list[ToolModelResponse],
        final: BaseModel,
        *,
        fail_finalize: bool = False,
        finalize_error: Exception | None = None,
    ) -> None:
        super().__init__(turns, final)
        self.next_action_calls = 0
        self.finalize_calls = 0
        self.fail_finalize = fail_finalize
        self.finalize_error = finalize_error

    async def next_action(self, **kwargs: object) -> ToolModelResponse:
        self.next_action_calls += 1
        usage_observer = kwargs.get("usage_observer")
        if callable(usage_observer):
            usage_observer(
                ModelUsage(input_tokens=1, output_tokens=1, request_kind="tool_action")
            )
        return await super().next_action(**kwargs)  # type: ignore[arg-type]

    async def finalize(self, **kwargs: object) -> BaseModel:
        self.finalize_calls += 1
        usage_observer = kwargs.get("usage_observer")
        if callable(usage_observer):
            usage_observer(
                ModelUsage(
                    input_tokens=2, output_tokens=2, request_kind="structured_finalize"
                )
            )
        if self.finalize_error is not None:
            raise self.finalize_error
        if self.fail_finalize:
            raise RepairBudgetExhausted("task-level model-call budget exhausted")
        return await super().finalize(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_finalization_has_reserved_model_call(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    gateway = CountingGateway(
        [_tool_turn("r1", path="app.py"), _tool_turn("r2", path="app.py")],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    result = await BoundedToolAgent(gateway, recorder, max_iterations=2, max_tool_calls=4).run(
        run_id="reserve-final",
        agent_id="worker:t",
        invocation_id="worker:1:t",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )
    assert result.summary
    assert gateway.next_action_calls == 2
    assert gateway.finalize_calls == 1


@pytest.mark.asyncio
async def test_finalization_counts_toward_run_budget(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    gateway = CountingGateway(
        [_tool_turn("r1", path="app.py"), ToolModelResponse(content="done")],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    manager = RunBudgetManager(max_model_calls=8, max_tool_calls=8, recorder=recorder)
    await BoundedToolAgent(
        gateway, recorder, max_iterations=4, max_tool_calls=4, budget_manager=manager
    ).run(
        run_id="count-final",
        agent_id="worker:t",
        invocation_id="worker:1:t",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )
    snapshot = manager.for_run("count-final").snapshot()
    assert snapshot.model_calls == 3
    phases = [
        event.payload.get("phase")
        for event in recorder.events("count-final")
        if event.type == EventType.MODEL_CALL
    ]
    assert "structured_final" in phases
    assert sum(event.type == EventType.MODEL_USAGE for event in recorder.events("count-final")) >= 1


@pytest.mark.asyncio
async def test_worker_reserves_final_call_from_task_budget(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    gateway = CountingGateway(
        [_tool_turn(f"r{index}", path="app.py") for index in range(1, 8)],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    extra = RunRepairBudget(max_model_calls=5, max_tool_calls=20)
    await BoundedToolAgent(gateway, recorder, max_iterations=15, max_tool_calls=20).run(
        run_id="task-reserve",
        agent_id="worker:t",
        invocation_id="worker:1:t",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
        extra_budget=extra,
    )
    assert gateway.next_action_calls == 4
    assert gateway.finalize_calls == 1
    assert extra.snapshot().model_calls == 5


@pytest.mark.asyncio
async def test_global_budget_is_not_exceeded_by_finalization(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    gateway = CountingGateway(
        [_tool_turn(f"r{index}", path="app.py") for index in range(1, 8)],
        worker_result(),
    )
    recorder = TrajectoryRecorder()
    manager = RunBudgetManager(max_model_calls=3, max_tool_calls=20, recorder=recorder)
    await BoundedToolAgent(
        gateway, recorder, max_iterations=15, max_tool_calls=20, budget_manager=manager
    ).run(
        run_id="global-cap",
        agent_id="worker:t",
        invocation_id="worker:1:t",
        system_prompt="inspect",
        task_prompt="inspect",
        tools=create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path),
        output_schema=WorkerResult,
    )
    assert gateway.next_action_calls == 2
    assert gateway.finalize_calls == 1
    assert manager.for_run("global-cap").snapshot().model_calls == 3


@pytest.mark.asyncio
async def test_worker_collects_last_iteration_edit(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    gateway = CountingGateway(
        [
            _tool_turn(
                "patch",
                "apply_patch",
                files={"app.py": "VALUE = 1\n"},
            )
        ],
        StagedFixerPlan(summary="fixed"),
        fail_finalize=True,
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        max_iterations=1,
        max_tool_calls=4,
        budget_manager=RunBudgetManager(max_model_calls=8, max_tool_calls=8, recorder=recorder),
    )
    run = await worker.execute(
        context=WorkerContext(
            run_id="last-edit",
            repo=RepoSpec(name="fixture"),
            failure=CIFailure(summary="fail", log_excerpt="err"),
            workspace_path=str(tmp_path),
            invocation_id="worker:1:repair",
            task=WorkerTask(
                task_id="repair",
                kind="repair",
                objective="fix",
                write_scope=["app.py"],
            ),
            baseline_snapshot_id="snap",
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert gateway.next_action_calls == 1
    assert gateway.finalize_calls == 1
    assert run.result.status == "completed"
    assert run.edits[0].content == "VALUE = 1\n"
    assert (tmp_path / "app.py").read_text() == "VALUE = 0\n"


@pytest.mark.asyncio
async def test_worker_run_uses_real_tool_events_not_model_report(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    gateway = CountingGateway(
        [
            _tool_turn("patch", "apply_patch", files={"app.py": "VALUE = 1\n"}),
            _tool_turn(
                "test",
                "run_test",
                argv=["python", "-c", "import app; assert app.VALUE == 1"],
            ),
            ToolModelResponse(content="done"),
        ],
        StagedFixerPlan(
            summary="semantic summary only",
            commands_run=["pytest imaginary.py", "rm -rf /"],
            verification_plan=[["python", "-c", "raise SystemExit(0)"]],
            evidence=[
                EvidenceItem(
                    source_agent="worker:repair",
                    kind="test_result",
                    claim="imaginary tests passed",
                    command="pytest imaginary.py",
                    confidence=1.0,
                ),
                EvidenceItem(
                    source_agent="worker:repair",
                    kind="source_code",
                    claim="VALUE was updated in app.py",
                    file_path="app.py",
                    confidence=0.8,
                ),
            ],
        ),
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        max_iterations=4,
        max_tool_calls=8,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=WorkerContext(
            run_id="harness-facts",
            repo=RepoSpec(name="fixture"),
            failure=CIFailure(summary="fail", log_excerpt="err"),
            workspace_path=str(tmp_path),
            invocation_id="worker:1:repair",
            task=WorkerTask(
                task_id="repair",
                kind="repair",
                objective="fix",
                write_scope=["app.py"],
            ),
            baseline_snapshot_id="snap",
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.summary == "semantic summary only"
    assert any("import app; assert app.VALUE == 1" in item for item in run.commands_run)
    assert [spec.argv for spec in run.verification_plan] == [
        ["python", "-c", "import app; assert app.VALUE == 1"]
    ]
    assert [spec.cwd for spec in run.verification_plan] == ["."]
    claims = [item.claim for item in run.result.evidence]
    assert "VALUE was updated in app.py" in claims
    assert "imaginary tests passed" not in claims
    assert any("run_test passed" in claim for claim in claims)
    assert "pytest imaginary.py" not in run.commands_run


def _skill_worker_context(
    tmp_path: Path, skill_id: str, *, invocation_id: str = "worker:1:repair"
) -> WorkerContext:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    return WorkerContext(
        run_id="skill-attr",
        repo=RepoSpec(name="fixture"),
        failure=CIFailure(summary="fail", log_excerpt="err"),
        workspace_path=str(tmp_path),
        invocation_id=invocation_id,
        task=WorkerTask(
            task_id="repair",
            kind="repair",
            objective="fix",
            write_scope=["app.py"],
            recommended_skill_refs=[SkillRef(skill_id=skill_id)],
        ),
        recommended_skills=(
            SkillCatalogEntry(skill_id=skill_id, name="fixture", description="fixture"),
        ),
        baseline_snapshot_id="snap",
    )


@pytest.mark.asyncio
async def test_model_worker_procedure_skill_is_used_without_script(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_skill(skill_candidate())
    skill_id = created.manifest.skill_id
    gateway = CountingGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load",
                        name="load_skill",
                        arguments={"skill_id": skill_id},
                    )
                ]
            ),
            _tool_turn("patch", "apply_patch", files={"app.py": "VALUE = 1\n"}),
            ToolModelResponse(content="done"),
        ],
        StagedFixerPlan(
            summary="applied the loaded procedure",
            used_skill_refs=[SkillRef(skill_id=skill_id)],
        ),
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        max_iterations=6,
        max_tool_calls=8,
        capability_registry=registry,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=_skill_worker_context(tmp_path, skill_id),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.used_skill_refs == [SkillRef(skill_id=skill_id)]
    used = [event for event in recorder.events("skill-attr") if event.type == EventType.SKILL_USED]
    assert any(event.payload.get("usage_kind") == "procedure" for event in used)
    assert all(event.payload.get("usage_kind") != "script" for event in used)
    registry.close()


@pytest.mark.asyncio
async def test_loaded_skill_without_declaration_is_not_used(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_skill(skill_candidate())
    skill_id = created.manifest.skill_id
    gateway = CountingGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load",
                        name="load_skill",
                        arguments={"skill_id": skill_id},
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        StagedFixerPlan(summary="did not use the skill"),
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        capability_registry=registry,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=_skill_worker_context(tmp_path, skill_id),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.used_skill_refs == []
    assert not any(
        event.type == EventType.SKILL_USED for event in recorder.events("skill-attr")
    )
    registry.close()


@pytest.mark.asyncio
async def test_unloaded_claimed_skill_is_dropped(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_skill(skill_candidate())
    skill_id = created.manifest.skill_id
    gateway = CountingGateway(
        [ToolModelResponse(content="done")],
        StagedFixerPlan(
            summary="claimed without loading",
            used_skill_refs=[SkillRef(skill_id=skill_id)],
        ),
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        capability_registry=registry,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=_skill_worker_context(tmp_path, skill_id),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.result.used_skill_refs == []
    registry.close()


@pytest.mark.asyncio
async def test_successful_run_test_in_subdir_keeps_cwd(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "probe.py").write_text("VALUE = 1\n")
    gateway = CountingGateway(
        [
            _tool_turn(
                "test",
                "run_test",
                argv=["python", "-c", "import probe; assert probe.VALUE == 1"],
                cwd="tests",
            ),
            ToolModelResponse(content="done"),
        ],
        StagedFixerPlan(summary="tested in tests/"),
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        max_iterations=4,
        max_tool_calls=8,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=WorkerContext(
            run_id="cwd-run",
            repo=RepoSpec(name="fixture"),
            failure=CIFailure(summary="fail", log_excerpt="err"),
            workspace_path=str(tmp_path),
            invocation_id="worker:1:repair",
            task=WorkerTask(
                task_id="repair",
                kind="repair",
                objective="fix",
                write_scope=["app.py"],
            ),
            baseline_snapshot_id="snap",
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.verification_plan == [
        VerificationCommandSpec(
            argv=["python", "-c", "import probe; assert probe.VALUE == 1"],
            cwd="tests",
        )
    ]
    assert any("cwd=tests" in item for item in run.commands_run)


@pytest.mark.asyncio
async def test_successful_run_command_is_audit_only(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    gateway = CountingGateway(
        [
            _tool_turn("env", "run_command", argv=["python", "-c", "print('diag')"]),
            _tool_turn(
                "test",
                "run_test",
                argv=["python", "-c", "raise SystemExit(0)"],
            ),
            ToolModelResponse(content="done"),
        ],
        StagedFixerPlan(summary="mixed commands"),
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=WorkerContext(
            run_id="audit-only",
            repo=RepoSpec(name="fixture"),
            failure=CIFailure(summary="fail", log_excerpt="err"),
            workspace_path=str(tmp_path),
            invocation_id="worker:1:repair",
            task=WorkerTask(
                task_id="repair", kind="repair", objective="fix", write_scope=["app.py"]
            ),
            baseline_snapshot_id="snap",
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert any("diag" in item for item in run.commands_run)
    assert [spec.argv for spec in run.verification_plan] == [
        ["python", "-c", "raise SystemExit(0)"]
    ]


@pytest.mark.asyncio
async def test_failed_run_test_is_not_supplementary(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    gateway = CountingGateway(
        [
            _tool_turn(
                "test",
                "run_test",
                argv=["python", "-c", "raise SystemExit(1)"],
            ),
            ToolModelResponse(content="done"),
        ],
        StagedFixerPlan(summary="failed local test"),
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=WorkerContext(
            run_id="failed-test",
            repo=RepoSpec(name="fixture"),
            failure=CIFailure(summary="fail", log_excerpt="err"),
            workspace_path=str(tmp_path),
            invocation_id="worker:1:repair",
            task=WorkerTask(
                task_id="repair", kind="repair", objective="fix", write_scope=["app.py"]
            ),
            baseline_snapshot_id="snap",
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.verification_plan == []
    assert run.commands_run


@pytest.mark.asyncio
async def test_network_run_test_is_not_supplementary(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    gateway = CountingGateway(
        [
            _tool_turn(
                "test",
                "run_test",
                argv=["python", "-c", "raise SystemExit(0)"],
                network=True,
            ),
            ToolModelResponse(content="done"),
        ],
        StagedFixerPlan(summary="network test"),
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=WorkerContext(
            run_id="net-test",
            repo=RepoSpec(name="fixture"),
            failure=CIFailure(summary="fail", log_excerpt="err"),
            workspace_path=str(tmp_path),
            invocation_id="worker:1:repair",
            task=WorkerTask(
                task_id="repair", kind="repair", objective="fix", write_scope=["app.py"]
            ),
            baseline_snapshot_id="snap",
        ),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.verification_plan == []


@pytest.mark.asyncio
async def test_finalize_failure_does_not_infer_procedure_use(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_skill(skill_candidate())
    skill_id = created.manifest.skill_id
    gateway = CountingGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="load",
                        name="load_skill",
                        arguments={"skill_id": skill_id},
                    )
                ]
            ),
            _tool_turn("patch", "apply_patch", files={"app.py": "VALUE = 1\n"}),
        ],
        StagedFixerPlan(summary="unused"),
        fail_finalize=True,
    )
    recorder = TrajectoryRecorder()
    worker = ModelWorker(
        gateway,
        recorder,
        capability_registry=registry,
        max_iterations=2,
        budget_manager=RunBudgetManager(max_model_calls=16, max_tool_calls=16, recorder=recorder),
    )
    run = await worker.execute(
        context=_skill_worker_context(tmp_path, skill_id),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert run.edits
    assert run.result.used_skill_refs == []
    assert not any(
        event.type == EventType.SKILL_USED for event in recorder.events("skill-attr")
    )
    registry.close()


@pytest.mark.asyncio
async def test_invocation_ids_do_not_mix_commands_or_skills(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "skills.sqlite")
    created = registry.create_skill(skill_candidate())
    skill_id = created.manifest.skill_id
    recorder = TrajectoryRecorder()
    manager = RunBudgetManager(max_model_calls=32, max_tool_calls=32, recorder=recorder)
    worker = ModelWorker(
        CountingGateway(
            [
                ToolModelResponse(
                    tool_calls=[
                        ToolCallRequest(
                            call_id="load-a",
                            name="load_skill",
                            arguments={"skill_id": skill_id},
                        )
                    ]
                ),
                _tool_turn("a-test", "run_test", argv=["python", "-c", "print('first')"]),
                ToolModelResponse(content="done"),
            ],
            StagedFixerPlan(
                summary="first", used_skill_refs=[SkillRef(skill_id=skill_id)]
            ),
        ),
        recorder,
        capability_registry=registry,
        budget_manager=manager,
    )
    first = await worker.execute(
        context=_skill_worker_context(tmp_path, skill_id, invocation_id="worker:1:a"),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    worker.loop.gateway = CountingGateway(  # type: ignore[assignment]
        [
            _tool_turn("b-test", "run_test", argv=["python", "-c", "print('second')"]),
            ToolModelResponse(content="done"),
        ],
        StagedFixerPlan(summary="second"),
    )
    second = await worker.execute(
        context=_skill_worker_context(tmp_path, skill_id, invocation_id="worker:2:b"),
        capabilities=WORKER_REPAIR_CAPABILITIES,
    )
    assert any("first" in item for item in first.commands_run)
    assert all("second" not in item for item in first.commands_run)
    assert any("second" in item for item in second.commands_run)
    assert all("first" not in item for item in second.commands_run)
    assert first.result.used_skill_refs == [SkillRef(skill_id=skill_id)]
    assert second.result.used_skill_refs == []
    registry.close()

