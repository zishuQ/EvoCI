from __future__ import annotations

import subprocess
from pathlib import Path

from evoci.tools.filesystem import FileTools
from evoci.tools.isolation import copy_workspace_with_independent_git


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_fixer_copy_has_independent_worktree_git_metadata(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    assert _git(repository, "init", "-q").returncode == 0
    _git(repository, "config", "user.email", "fixture@example.test")
    _git(repository, "config", "user.name", "Fixture")
    (repository / "app.py").write_text("VALUE = 1\n")
    assert _git(repository, "add", "app.py").returncode == 0
    assert _git(repository, "commit", "-qm", "fixture").returncode == 0

    real = tmp_path / "real-worktree"
    assert _git(repository, "worktree", "add", "-q", str(real)).returncode == 0
    assert (real / ".git").is_file()
    real_head = _git(real, "rev-parse", "HEAD").stdout
    real_index = _git(real, "diff", "--cached", "--name-only").stdout

    staging = tmp_path / "fixer-staging"
    copy_workspace_with_independent_git(real, staging)
    assert (staging / ".git").is_dir()
    assert _git(staging, "remote").stdout.strip() == ""

    (staging / "app.py").write_text("VALUE = 2\n")
    (staging / "temporary.py").write_text("TEMP = True\n")
    assert _git(staging, "add", "--all").returncode == 0
    assert _git(staging, "reset", "--hard", "HEAD").returncode == 0

    assert (real / "app.py").read_text() == "VALUE = 1\n"
    assert not (real / "temporary.py").exists()
    assert _git(real, "rev-parse", "HEAD").stdout == real_head
    assert _git(real, "diff", "--cached", "--name-only").stdout == real_index
    assert _git(real, "status", "--porcelain").stdout == ""

    FileTools(real, writable=True).write_file("app.py", "VALUE = 3\n")
    assert (real / "app.py").read_text() == "VALUE = 3\n"


def test_copy_without_git_initializes_and_commits_baseline(tmp_path: Path) -> None:
    source = tmp_path / "plain"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 1\n")
    staging = tmp_path / "staging"
    copy_workspace_with_independent_git(source, staging)
    assert (staging / ".git").is_dir()
    assert _git(staging, "rev-parse", "HEAD").returncode == 0
    assert (staging / "app.py").read_text() == "VALUE = 1\n"
    (staging / "app.py").write_text("VALUE = 2\n")
    assert _git(staging, "reset", "--hard", "HEAD").returncode == 0
    assert (staging / "app.py").read_text() == "VALUE = 1\n"
    assert (source / "app.py").read_text() == "VALUE = 1\n"
