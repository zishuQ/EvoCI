from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evoci.tools.git import GitTools
from evoci.tools.shell import CommandRunner
from evoci.workspace.manager import WorkspaceManager


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _init_repo(path: Path) -> tuple[str, str]:
    path.mkdir()
    assert _git(path, "init", "-q").returncode == 0
    _git(path, "config", "user.email", "fixture@example.test")
    _git(path, "config", "user.name", "Fixture")
    (path / "app.py").write_text("VALUE = 0\n")
    assert _git(path, "add", "app.py").returncode == 0
    assert _git(path, "commit", "-qm", "fail").returncode == 0
    fail = _git(path, "rev-parse", "HEAD").stdout.strip()
    (path / "app.py").write_text("VALUE = 1\n")
    assert _git(path, "add", "app.py").returncode == 0
    assert _git(path, "commit", "-qm", "fix").returncode == 0
    success = _git(path, "rev-parse", "HEAD").stdout.strip()
    _git(path, "tag", "fixed")
    return fail, success


@pytest.mark.asyncio
async def test_agent_git_view_cannot_read_future_fix(tmp_path: Path) -> None:
    repo = tmp_path / "origin"
    fail, success = _init_repo(repo)
    workspace = WorkspaceManager(tmp_path / "worktrees").create(
        repo_path=repo, base_commit=fail, run_id="task-1"
    )
    assert (workspace.path / "app.py").read_text() == "VALUE = 0\n"
    git = GitTools(CommandRunner(workspace.path, timeout=5, max_chars=4000))

    current = await git.show("HEAD:app.py")
    assert "VALUE = 0" in current.stdout
    assert current.exit_code == 0

    future_sha = await git.show(f"{success}:app.py")
    assert future_sha.exit_code != 0 or "VALUE = 1" not in future_sha.stdout

    tagged = await git.show("fixed:app.py")
    assert tagged.exit_code != 0 or "VALUE = 1" not in tagged.stdout

    main = await git.show("main:app.py")
    if main.exit_code == 0:
        assert "VALUE = 1" not in main.stdout

    log = await git.log(limit=20)
    assert success[:7] not in log.stdout


@pytest.mark.asyncio
async def test_agent_can_read_failure_baseline(tmp_path: Path) -> None:
    repo = tmp_path / "origin"
    fail, _success = _init_repo(repo)
    workspace = WorkspaceManager(tmp_path / "worktrees").create(
        repo_path=repo, base_commit=fail, run_id="task-2"
    )
    git = GitTools(CommandRunner(workspace.path, timeout=5, max_chars=4000))
    shown = await git.show("HEAD")
    assert shown.exit_code == 0
    status = await git.status()
    assert status.exit_code == 0
