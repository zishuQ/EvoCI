from pathlib import Path
from typing import cast

import pytest

from evoci.agents.base import AgentSuite, SupervisorContext
from evoci.benchmark.docker import DockerReplayVerifier
from evoci.benchmark.models import BenchmarkPreflightResult
from evoci.config import EvoCIConfig
from evoci.domain.models import (
    FileEdit,
    SupervisorDecision,
    VerificationCommandResult,
    VerificationResult,
)
from evoci.graph.builder import GraphRuntime, build_graph
from evoci.model.gateway import ModelGatewayError
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from tests.integration.test_graph import (
    CombinedWorker,
    FakeFixer,
    FakeInvestigator,
    initial_state,
    task,
)
from tests.unit.test_benchmark_docker import _spec, _task


@pytest.mark.asyncio
async def test_official_failure_reaches_next_supervisor_batch(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    seen: list[str] = []

    class Supervisor:
        async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
            if context.last_verification is not None:
                assert context.last_verification.status == "failed"
                assert context.last_verification.incomplete_reason is not None
                seen.append(context.last_verification.incomplete_reason)
            return SupervisorDecision(
                action="dispatch",
                reasoning_summary="retry with official feedback",
                tasks=[
                    task(
                        f"repair-{len(seen)}",
                        kind="repair",
                        write_scope=["app.py"],
                    )
                ],
            )

    class Official:
        spec = _spec()

        def __init__(self) -> None:
            self.calls = 0

        async def verify_candidate(self, *args: object) -> VerificationResult:
            self.calls += 1
            passed = self.calls == 2
            return VerificationResult(
                passed=passed,
                status="passed" if passed else "failed",
                commands=[
                    VerificationCommandResult(
                        command=["python", "-m", "pytest", "tests/test_x.py::test_target"],
                        exit_code=0 if passed else 1,
                        stdout="PASSED" if passed else "FAILED official target",
                        stderr="",
                    )
                ],
                incomplete_reason=None if passed else "official target assertion failed",
            )

    official = Official()
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(
            supervisor=Supervisor(),
            worker=CombinedWorker(
                FakeInvestigator(),
                FakeFixer(edit=FileEdit(path="app.py", content="VALUE = 1\n")),
            ),
        ),
        recorder=TrajectoryRecorder(),
        official_verifier=cast(DockerReplayVerifier, official),
        official_preflight=BenchmarkPreflightResult(
            status="reproduced", details="baseline failed"
        ),
        official_task=_task(),
    )
    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="official-candidate-feedback")
    )
    assert result["status"] == "success"
    assert official.calls == 2
    assert seen == ["official target assertion failed"]
    assert [item.status for item in result["verification_history"]] == ["failed", "passed"]
    started = [
        event
        for event in result["events"]
        if event.type == EventType.VERIFICATION_STARTED
    ]
    assert len(started) == 2
    assert all(event.payload["oracle_source"] == "official_image" for event in started)
    assert all(len(event.payload["commands"]) == 2 for event in started)


@pytest.mark.asyncio
async def test_http_524_gets_exactly_one_extra_supervisor_invocation(tmp_path: Path) -> None:
    class HTTP524(Exception):
        status_code = 524

    class Supervisor:
        def __init__(self) -> None:
            self.ids: list[str] = []

        async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
            self.ids.append(context.invocation_id)
            if len(self.ids) == 1:
                raise ModelGatewayError("gateway retries exhausted") from HTTP524()
            return SupervisorDecision(
                action="stop",
                reasoning_summary="recovered from 524",
                stop_reason="no repair required",
            )

    supervisor = Supervisor()
    runtime = GraphRuntime(
        config=EvoCIConfig.from_env(cwd=tmp_path),
        agents=AgentSuite(supervisor=supervisor, worker=FakeInvestigator()),
        recorder=TrajectoryRecorder(),
    )
    result = await build_graph(runtime).ainvoke(
        initial_state(tmp_path, run_id="http-524")
    )
    assert result["status"] == "failed"
    assert result["supervisor_batch"] == 1
    assert supervisor.ids == ["supervise:1", "supervise:1:timeout-retry"]


@pytest.mark.asyncio
async def test_http_524_retry_is_bounded_and_http_500_does_not_retry(tmp_path: Path) -> None:
    class HTTPError(Exception):
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    for status_code, expected_calls in ((524, 2), (500, 1)):
        class Supervisor:
            def __init__(self, code: int) -> None:
                self.code = code
                self.ids: list[str] = []

            async def decide(self, *, context: SupervisorContext) -> SupervisorDecision:
                self.ids.append(context.invocation_id)
                raise ModelGatewayError("gateway retries exhausted") from HTTPError(self.code)

        supervisor = Supervisor(status_code)
        runtime = GraphRuntime(
            config=EvoCIConfig.from_env(cwd=tmp_path),
            agents=AgentSuite(supervisor=supervisor, worker=FakeInvestigator()),
            recorder=TrajectoryRecorder(),
        )
        result = await build_graph(runtime).ainvoke(
            initial_state(tmp_path, run_id=f"http-{status_code}-always-fails")
        )
        assert result["status"] == "failed"
        assert result["failure_class"] == "model"
        assert len(supervisor.ids) == expected_calls
