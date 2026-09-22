"""Independent review: assert desired behavior, so failures demonstrate remaining bugs.

Run from the pinned EvoCI checkout:
  PYTHONPATH="$PWD" uv run pytest /path/to/test_review_regressions.py -q --tb=short
These tests never call a real model API and do not modify the project source.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeCancelledError
from langgraph.types import Command

from evoci.agents.tool_loop import BoundedToolAgent
from evoci.benchmark.execution import FailedCommandReplayVerifier
from evoci.capability.models import GeneratedFile, SkillPermissions
from evoci.capability.validator import CandidateValidator
from evoci.cli import _drive_single
from evoci.domain.models import FileEdit, FixerOutput, PatchProposal, WorkerResult
from evoci.graph.builder import build_graph
from evoci.model.gateway import ToolCallRequest, ToolModelResponse
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.policy import FIXER_CAPABILITIES
from evoci.tools.registry import create_worker_registry
from evoci.verification.service import VerificationService
from tests.integration.test_correctness import (
    PASSING,
    VALUE_ORACLE,
    RealValueFixer,
    _failure,
    _runtime,
)
from tests.integration.test_tool_loop import ScriptedToolGateway, worker_result
from tests.unit.test_capability import candidate, registry


@pytest.mark.asyncio
async def test_r01_async_learning_can_validate_declared_commands(tmp_path: Path) -> None:
    store = registry(tmp_path)
    try:
        record = store.create_skill(candidate().model_copy(update={"tests": []}))
        record = store.get(record.manifest.skill_id)
        assert record is not None and record.manifest.enabled
    finally:
        store.close()


@pytest.mark.asyncio
async def test_r02_benchmark_verification_side_effects_stay_isolated(tmp_path: Path) -> None:
    command = ["python", "-c", (
        "from pathlib import Path; p=Path('oracle_marker'); exists=p.exists(); "
        "p.write_text('side-effect'); raise SystemExit(0 if exists else 1)"
    )]
    task = SimpleNamespace(agent_view=SimpleNamespace(
        ci_failure=SimpleNamespace(candidate_failed_commands=[command])
    ))
    verifier = FailedCommandReplayVerifier(timeout=5)
    preflight_workspace = tmp_path / "preflight"
    workspace = tmp_path / "agent"
    preflight_workspace.mkdir()
    workspace.mkdir()
    preflight = await verifier.preflight(task, preflight_workspace)
    # Match the CLI: preflight and agent workspaces are distinct.
    verdict = await verifier.verify(task, workspace, preflight)
    observed = (preflight.status, verdict.status, (workspace / "oracle_marker").exists())
    assert observed == ("reproduced", "failed", False)


@pytest.mark.asyncio
async def test_r03_conflict_after_approval_preserves_external_edit(tmp_path: Path) -> None:
    path = tmp_path / "app.py"
    path.write_text("VALUE = 0\n")

    class HighRiskFixer:
        async def propose(self, **kwargs: object) -> FixerOutput:
            return FixerOutput(
                proposal=PatchProposal(summary="set value", changed_files=["app.py"],
                                       risk="high", verification_plan=[PASSING]),
                edits=[FileEdit(path="app.py", content="VALUE = 1\n",
                                expected_sha256=hashlib.sha256(b"VALUE = 0\n").hexdigest())],
            )

    graph = build_graph(_runtime(tmp_path, HighRiskFixer()), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "review-conflict"}}
    pending = await graph.ainvoke(_failure(tmp_path, [VALUE_ORACLE]), config)
    assert pending.get("__interrupt__")
    path.write_text("VALUE = 99  # user edit while awaiting approval\n")
    result = await graph.ainvoke(Command(resume=True), config)
    assert result["status"] == "failed"
    assert path.read_text() == "VALUE = 99  # user edit while awaiting approval\n"


def test_r04_nested_failing_skill_test_must_be_discovered(tmp_path: Path) -> None:
    store = registry(tmp_path)
    try:
        with pytest.raises(Exception, match="validation failed"):
            store.create_skill(candidate().model_copy(update={"tests": [
                GeneratedFile(path="tests/nested/test_failure.py",
                              content="def test_failure():\n    assert False\n"),
            ]}))
        assert store.list() == []
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_arguments", [
    {"skill_id": "selected", "version": 1, "script_name": "read.py", "args": "not-a-list"},
    {"skill_id": "selected", "version": "invalid-version", "script_name": "read.py", "args": []},
])
async def test_r05_invalid_skill_arguments_are_rejected_without_use(
    tmp_path: Path, bad_arguments: dict,
) -> None:
    store = registry(tmp_path)
    tools = create_worker_registry(FIXER_CAPABILITIES, tmp_path, capability_registry=store,
                                   allowed_skill_refs={"selected"})
    recorder = TrajectoryRecorder()
    gateway = ScriptedToolGateway([
        ToolModelResponse(tool_calls=[ToolCallRequest(
            call_id="invalid", name="run_skill_script", arguments=bad_arguments,
        )]), ToolModelResponse(content="done"),
    ], worker_result())
    try:
        result = await BoundedToolAgent(gateway, recorder, max_iterations=3, max_tool_calls=3).run(
            run_id="invalid-args", agent_id="fixer", invocation_id="repair:1",
            system_prompt="test", task_prompt="test", tools=tools, output_schema=WorkerResult,
        )
        used = [
            event.payload
            for event in recorder.events("invalid-args")
            if event.type == EventType.SKILL_USED
        ]
        assert not used, used
        assert not result.used_skill_refs
        assert any(e.type == EventType.SKILL_INVOCATION_REJECTED
                   for e in recorder.events("invalid-args"))
    finally:
        tools.close()
        store.close()


def test_r06_write_skill_invalidates_read_skill_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "value.txt").write_text("before")
    store = registry(tmp_path)
    records = []
    for name, script, may_write in [
        ("reader", "from pathlib import Path\nprint(Path('value.txt').read_text())\n", False),
        ("writer", "from pathlib import Path\nPath('value.txt').write_text('after')\n", True),
    ]:
        record = store.create_skill(candidate(script=script).model_copy(update={
            "name": name, "tests": [],
            "permissions": SkillPermissions(execute=True, write_workspace=may_write),
        }))
        records.append(record)
    tools = create_worker_registry(FIXER_CAPABILITIES, workspace, capability_registry=store,
        allowed_skill_refs={r.manifest.skill_id for r in records})
    def run(record):
        if record.manifest.skill_id not in tools.activated_skill_ids:
            tools.invoke("load_skill", skill_id=record.manifest.skill_id)
        return tools.invoke("run_skill_script", skill_id=record.manifest.skill_id,
                            script_name="inspect_imports.py", args=[])
    try:
        first = run(records[0])
        assert first.stdout.strip() == "before"
        assert run(records[1]).exit_code == 0
        assert (workspace / "value.txt").read_text() == "after"
        second = run(records[0])
        assert second.stdout.strip() == "after", second.model_dump()
        assert second.observed_revision != first.observed_revision
    finally:
        tools.close()
        store.close()


@pytest.mark.asyncio
async def test_r07_single_patch_budget_exhaustion_is_a_failed_result(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")

    class TwoFileFixer:
        async def propose(self, **kwargs):
            return FixerOutput(proposal=PatchProposal(summary="two changes", risk="low",
                changed_files=["app.py", "extra.py"], verification_plan=[PASSING]),
                edits=[FileEdit(path="app.py", content="VALUE = 1\n"),
                       FileEdit(path="extra.py", content="EXTRA = 1\n")])

    runtime = _runtime(tmp_path, TwoFileFixer(), max_tool_calls=1)
    resources = SimpleNamespace(runtime=runtime, recorder=TrajectoryRecorder())
    result = await _drive_single(resources, _failure(tmp_path, [VALUE_ORACLE]))
    assert result["status"] == "failed"
    assert (tmp_path / "app.py").read_text() == "VALUE = 0\n"
    assert not (tmp_path / "extra.py").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["graph", "single"])
async def test_r08_cancellation_during_verification_restores_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str,
) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    async def cancel_at_verification(self, **kwargs):
        assert (tmp_path / "app.py").read_text() == "VALUE = 1\n"
        raise asyncio.CancelledError("review simulated cancellation at verification boundary")
    monkeypatch.setattr(VerificationService, "run", cancel_at_verification)
    runtime = _runtime(tmp_path, RealValueFixer())
    state = _failure(tmp_path, [VALUE_ORACLE])
    with pytest.raises((asyncio.CancelledError, NodeCancelledError)):
        if engine == "graph":
            await build_graph(runtime).ainvoke(state)
        else:
            resources = SimpleNamespace(runtime=runtime, recorder=TrajectoryRecorder())
            await _drive_single(resources, state)
    assert (tmp_path / "app.py").read_text() == "VALUE = 0\n"


def test_r09_parent_pytest_quiet_config_cannot_reject_valid_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -q\n")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    store = registry(tmp_path)
    try:
        record = store.create_skill(candidate())
        result = CandidateValidator().validate_package(
            Path(record.package_path), record.manifest
        )
        assert result.passed, result.model_dump()
        assert result.tests_run == 1
    finally:
        store.close()


def test_r10_validator_timeout_reaps_test_child_process(tmp_path: Path) -> None:
    import time
    copied = tmp_path / "package"
    (copied / "tests").mkdir(parents=True)
    (copied / "pytest.ini").write_text("[pytest]\n")
    (copied / "tests/test_child.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "def test_child():\n"
        "    child = os.fork()\n"
        "    if child == 0:\n"
        "        os.close(1)\n        os.close(2)\n"
        "        time.sleep(3)\n"
        "        Path('survived.txt').write_text('child survived timeout')\n"
        "        os._exit(0)\n"
        "    Path('child.pid').write_text(str(child))\n"
        "    time.sleep(15)\n"
    )
    error, _ = CandidateValidator(SimpleNamespace(), timeout=2)._run_pytest(copied, 1)
    assert error == "skill tests timed out", error
    # The pid file confirms startup; a delayed marker crosses PID namespaces reliably.
    assert (copied / "child.pid").exists()
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not (copied / "survived.txt").exists():
        time.sleep(0.05)
    assert not (copied / "survived.txt").exists(), "child performed a write after validator timeout"


def test_r11_benchmark_metrics_include_deferred_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import evoci.cli as cli
    from evoci.capability.miner import SkillLearningDecision
    from tests.integration.test_benchmark_variants import OfflineBenchmarkGateway, make_workspace
    from tests.integration.test_graph import FakeSkillMiner

    class MiningGateway(OfflineBenchmarkGateway):
        async def complete(self, **kwargs):
            if kwargs["response_model"] is SkillLearningDecision:
                decision = await FakeSkillMiner().decide({})
                return decision.model_copy(update={"candidate_skill":
                    decision.candidate_skill.model_copy(update={"tests": []})})
            return await super().complete(**kwargs)

    source = tmp_path / "source"
    make_workspace(source)
    root = Path(__import__("tests").__file__).parent
    row = json.loads((root / "fixtures/ci_tasks/tasks.jsonl").read_text().splitlines()[1])
    row["workspace_path"] = str(source)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"task_id": "fixture-test", "category": "test"}) + "\n")
    monkeypatch.setattr(cli, "OpenAICompatibleGateway", MiningGateway)
    original_live = cli._live_resources
    captured = []

    async def live_with_forced_mining(*args, **kwargs):
        resources = await original_live(*args, **kwargs)
        # Ensure this small fixture exercises learning; production wiring and CLI remain real.
        resources.runtime.skill_miner.tool_call_threshold = 0
        close = resources.close
        async def capture_then_close():
            captured.extend(resources.registry.list())
            await close()
        return SimpleNamespace(runtime=resources.runtime, recorder=resources.recorder,
            checkpoint=resources.checkpoint, event_store=resources.event_store,
            run_store=resources.run_store, memory_store=resources.memory_store,
            registry=resources.registry, gateway=resources.worker_gateway, close=capture_then_close)

    monkeypatch.setattr(cli, "_live_resources", live_with_forced_mining)
    output = tmp_path / "benchmark-output"
    cli.benchmark(manifest=manifest, dataset=dataset, variant="evo", output_dir=output,
                  continual=False)
    result = json.loads((output / "runs.jsonl").read_text().splitlines()[0])
    assert result["status"] == "resolved", result
    assert len(captured) == 1, "fixture must actually create a skill before measuring metrics"
    assert result["metrics"]["skill_created"] == 1, result["metrics"]
    assert result["metrics"]["skill_registry_size"] == 1
    assert result["metrics"]["post_run_model_calls"] > 0
