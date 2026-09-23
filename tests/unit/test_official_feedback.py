import asyncio
import subprocess
import threading
from pathlib import Path

import pytest

from evoci.benchmark.docker import DockerReplayVerifier
from tests.unit.test_benchmark_docker import FakeDocker, _init_repo, _spec, _task


@pytest.mark.asyncio
async def test_candidate_feedback_runs_the_official_tests_before_final_scoring(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    assert preflight.status == "reproduced"
    (tmp_path / "src.py").write_text("VALUE = 2\n")

    fake.ftp_mode = "always_fail"
    feedback = await verifier.verify_candidate(_task(), tmp_path, preflight)
    assert feedback.status == "failed"
    assert feedback.incomplete_reason is not None
    assert "tests/test_x.py::test_target" in feedback.incomplete_reason
    assert feedback.commands[0].command[:3] == ["python", "-m", "pytest"]
    assert any(
        any("/tmp/test.patch" in arg for arg in call)
        for call in fake.calls
        if call[:1] == ["cp"]
    )

    fake.ftp_mode = "fail_until_candidate"
    repaired = await verifier.verify_candidate(_task(), tmp_path, preflight)
    assert repaired.passed
    assert repaired.commands[0].command[-1] == "tests/test_x.py::test_target"
    assert any("test_reg" in item.command[-1] for item in repaired.commands)
    verifier.close()


@pytest.mark.asyncio
async def test_separate_official_preflights_can_run_concurrently(tmp_path: Path) -> None:
    barrier = threading.Barrier(2, timeout=5)

    class ConcurrentDocker(FakeDocker):
        def run(self, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
            if args[:1] == ["run"]:
                barrier.wait()
            return super().run(args, timeout=timeout)

    verifiers = [
        DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=ConcurrentDocker())
        for _ in range(2)
    ]
    try:
        outcomes = await asyncio.gather(
            *(verifier.preflight(_task(), tmp_path) for verifier in verifiers)
        )
        assert all(outcome.status == "reproduced" for outcome in outcomes)
    finally:
        for verifier in verifiers:
            verifier.close()
