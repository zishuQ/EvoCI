from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from typer.testing import CliRunner

import evoci.cli as cli
from evoci.local_task import parse_verification_command
from tests.integration.test_correctness import RealValueFixer, _runtime


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "app.py").write_text("VALUE = 0\n")
    subprocess.run(["git", "-C", str(root), "add", "app.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )
    for key in ("EVO_MODEL_NAME", "EVO_MODEL_API_KEY", "EVO_RUNTIME_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EVO_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("EVO_CAPABILITY_DIR", str(tmp_path / "skills"))
    return root


def test_prepare_only_captures_failure_without_mutating_repo(repo: Path) -> None:
    (repo / "dirty.txt").write_text("user work")
    command = (
        "python -c \"from pathlib import Path; Path('probe').write_text('x'); "
        "print('original failure'); raise SystemExit(1)\""
    )
    result = CliRunner().invoke(
        cli.app, ["fix", "--repo", str(repo), "-c", command, "--prepare-only"]
    )
    assert result.exit_code == 0, result.output
    assert '"status": "prepared"' in result.output
    assert not (repo / "probe").exists()
    assert (repo / "dirty.txt").read_text() == "user work"
    task = json.loads(next((repo.parent / "state").rglob("task.json")).read_text())
    assert task["ci_failure"]["failed_commands"] == [parse_verification_command(command)]
    assert "original failure" in task["ci_failure"]["log_excerpt"]
    assert task["workspace_path"] == str(repo)


def test_passing_command_needs_no_model(repo: Path) -> None:
    result = CliRunner().invoke(cli.app, ["fix", "--repo", str(repo), "-c", "python -c 'pass'"])
    assert result.exit_code == 0, result.output
    assert "already_passing" in result.output


def test_missing_model_saves_task_and_returns_actionable_error(repo: Path) -> None:
    result = CliRunner().invoke(cli.app, ["fix", "--repo", str(repo), "-c", "python -c 'exit(1)'"])
    assert result.exit_code == 2, result.output
    assert "EVO_MODEL_API_KEY" in result.output
    assert list((repo.parent / "state").rglob("task.json"))


@pytest.mark.parametrize("command", ["python -V; git status", "python -V | python -V", ""])
def test_shell_operators_are_rejected_before_running(repo: Path, command: str) -> None:
    result = CliRunner().invoke(cli.app, ["fix", "--repo", str(repo), "-c", command])
    assert result.exit_code == 2
    assert not (repo.parent / "state").exists()


def test_fix_runs_real_graph_and_keeps_verified_patch(repo: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVO_MODEL_NAME", "offline-fixture")
    monkeypatch.setenv("EVO_MODEL_API_KEY", "offline-fixture")
    closed = []

    async def resources(config):
        async def close():
            closed.append(True)

        return SimpleNamespace(
            runtime=_runtime(repo, RealValueFixer()),
            checkpoint=SimpleNamespace(saver=InMemorySaver()),
            event_store=SimpleNamespace(list=lambda run_id: []),
            close=close,
        )

    monkeypatch.setattr(cli, "_live_resources", resources)
    result = CliRunner().invoke(
        cli.app,
        [
            "fix",
            "--repo",
            str(repo),
            "-c",
            "python -c 'import app; assert app.VALUE == 1'",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"status": "success"' in result.output
    assert (repo / "app.py").read_text() == "VALUE = 1\n"
    assert closed == [True]
