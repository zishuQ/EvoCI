"""Independent acceptance tests for residual defects at 8104149; no model API."""

import asyncio
import os
import sys
import tracemalloc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.errors import NodeCancelledError

from evoci.agents.tool_loop import BoundedToolAgent
from evoci.capability.execution import run_skill_script
from evoci.capability.validator import CandidateValidator
from evoci.cli import _drive_single
from evoci.domain.models import VerificationResult, WorkerResult
from evoci.graph.builder import build_graph
from evoci.model.gateway import ToolCallRequest, ToolModelResponse
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.policy import FIXER_CAPABILITIES
from evoci.tools.registry import create_worker_registry
from evoci.tools.shell import run_grouped_subprocess
from evoci.verification.service import VerificationService
from tests.integration.test_correctness import (
    VALUE_ORACLE,
    RealValueFixer,
    _failure,
    _runtime,
)
from tests.integration.test_graph import FakeExperienceMiner
from tests.integration.test_tool_loop import ScriptedToolGateway, worker_result
from tests.unit.test_capability import candidate, registry


@pytest.mark.asyncio
async def test_n01_partial_write_error_restores_baseline(tmp_path, monkeypatch):
    path = tmp_path / "app.py"
    path.write_text("VALUE = 0\n")
    original = Path.write_text

    def partial_write(self, content, *args, **kwargs):
        if content == "VALUE = 1\n":
            original(self, "VALUE = ", *args, **kwargs)
            raise OSError("simulated disk full after partial write")
        return original(self, content, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", partial_write)
    result = await build_graph(_runtime(tmp_path, RealValueFixer())).ainvoke(
        _failure(tmp_path, [VALUE_ORACLE])
    )
    assert result["status"] == "failed"
    assert path.read_text() == "VALUE = 0\n"
    assert not list(tmp_path.glob(".app.py.*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancel", "failure"])
async def test_n02_single_cancel_preserves_external_edit(tmp_path, monkeypatch, outcome):
    path = tmp_path / "app.py"
    path.write_text("VALUE = 0\n")

    async def external_edit_then_cancel(self, **kwargs):
        assert path.read_text() == "VALUE = 1\n"
        path.write_text("VALUE = 99  # external edit during verification\n")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return VerificationResult(passed=False, commands=[])

    monkeypatch.setattr(VerificationService, "run", external_edit_then_cancel)
    runtime = _runtime(tmp_path, RealValueFixer())
    operation = _drive_single(
        SimpleNamespace(runtime=runtime, recorder=TrajectoryRecorder()),
        _failure(tmp_path, [VALUE_ORACLE]),
    )
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        assert (await operation)["status"] == "failed"
    assert path.read_text() == "VALUE = 99  # external edit during verification\n"


@pytest.mark.asyncio
async def test_n03_validation_obeys_scheduled_cancellation(tmp_path):
    store = registry(tmp_path)
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    timers = []

    class SlowValidationMiner(FakeExperienceMiner):
        async def decide(self, trajectory_summary, *, existing_skills=None):
            del existing_skills
            decision = await super().decide(trajectory_summary)
            timers.append(
                asyncio.get_running_loop().call_later(0.05, asyncio.current_task().cancel)
            )
            return decision.model_copy(
                update={
                    "candidate_skill": decision.candidate_skill.model_copy(
                        update={
                            "tests": [],
                            "verification_commands": [
                                ["python", "-c", "import time; time.sleep(0.5)"]
                            ],
                        }
                    )
                }
            )

    runtime = replace(
        _runtime(tmp_path, RealValueFixer()),
        capability_registry=store,
        experience_miner=SlowValidationMiner(),
        candidate_validator=CandidateValidator(store),
    )
    try:
        with pytest.raises((asyncio.CancelledError, NodeCancelledError)):
            await build_graph(runtime).ainvoke(_failure(tmp_path, [VALUE_ORACLE]))
    finally:
        for timer in timers:
            timer.cancel()
        store.close()


@pytest.mark.parametrize("runner", ["validation_runner", "skill_runner"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_n04_output_limit_also_bounds_buffer_memory(tmp_path, runner, stream):
    # 8 MiB output is safe to reproduce, yet far beyond the requested 1 KiB buffer.
    script = f"import sys\nfor _ in range(128):\n    sys.{stream}.write('x' * 65536)\n"
    store = registry(tmp_path)
    record = store.create_candidate(candidate(script=script).model_copy(update={"tests": []}))
    assert CandidateValidator(store).validate_to_trial(record.manifest.skill_id, 1).passed
    tracemalloc.start()
    try:
        if runner == "validation_runner":
            result = run_grouped_subprocess(
                [sys.executable, "-c", script],
                cwd=tmp_path,
                env={"PATH": os.environ["PATH"]},
                timeout=5,
                max_chars=1024,
            )
        else:
            result = run_skill_script(
                store,
                skill_id=record.manifest.skill_id,
                version=1,
                script_name="inspect_imports.py",
                args=[],
                workspace=tmp_path,
                timeout=5,
                max_chars=1024,
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        store.close()
    assert result.exit_code == 0 and len(getattr(result, stream)) == 1024
    assert peak < 4 * 1024 * 1024, f"peak={peak} bytes for max_chars=1024"


@pytest.mark.asyncio
async def test_n05_coerced_version_execution_is_attributed(tmp_path):
    store = registry(tmp_path)
    record = store.create_candidate(candidate().model_copy(update={"tests": []}))
    assert CandidateValidator(store).validate_to_trial(record.manifest.skill_id, 1).passed
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={(record.manifest.skill_id, 1)},
    )
    recorder = TrajectoryRecorder()
    gateway = ScriptedToolGateway(
        [
            ToolModelResponse(
                tool_calls=[
                    ToolCallRequest(
                        call_id="coerced",
                        name="run_skill_script",
                        arguments={
                            "skill_id": record.manifest.skill_id,
                            "version": "1",
                            "script_name": "inspect_imports.py",
                            "args": [],
                        },
                    )
                ]
            ),
            ToolModelResponse(content="done"),
        ],
        worker_result(),
    )
    try:
        await BoundedToolAgent(gateway, recorder, max_iterations=3, max_tool_calls=3).run(
            run_id="coerced",
            agent_id="fixer",
            invocation_id="repair:1",
            system_prompt="test",
            task_prompt="test",
            tools=tools,
            output_schema=WorkerResult,
        )
        events = recorder.events("coerced")
        executions = [e for e in events if e.type == EventType.TOOL_RESULT]
        assert executions[0].payload["success"] is True
        assert "imports-ok" in executions[0].payload["result"]["stdout"]
        used = [e for e in events if e.type == EventType.SKILL_USED]
        assert len(used) == 1, "script really executed, but SKILL_USED is missing"
    finally:
        tools.close()
        store.close()
