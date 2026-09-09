"""Isolated benchmark workspaces with Git history truncated at the failure baseline."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from evoci.tools.policy import WorkspaceBoundary


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    run_id: str
    path: Path
    base_commit: str


def _git(
    *args: str, cwd: Path | None = None, timeout: float = 60
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def truncate_git_to_commit(source_repo: Path, destination: Path, base_commit: str) -> None:
    """Clone only `base_commit` and its ancestors into a standalone repository.

    Later commits, tags, and objects remain in the original mirror and are not
    copied into the agent-visible workspace. This is Git-object isolation, not
    an OS sandbox.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    cloned = _git(
        "clone",
        "--no-local",
        "--no-hardlinks",
        str(source_repo.resolve()),
        str(destination),
        timeout=120,
    )
    if cloned.returncode != 0:
        raise WorkspaceError(cloned.stderr.strip() or "git clone failed")
    checkout = _git("-C", str(destination), "checkout", "--force", "--detach", base_commit)
    if checkout.returncode != 0:
        raise WorkspaceError(checkout.stderr.strip() or "git checkout of failure baseline failed")
    refs = _git("-C", str(destination), "for-each-ref", "--format=%(refname)")
    for ref in refs.stdout.splitlines():
        if ref.startswith(("refs/heads/", "refs/tags/", "refs/remotes/")):
            _git("-C", str(destination), "update-ref", "-d", ref)
    _git("-C", str(destination), "remote", "remove", "origin")
    _git("-C", str(destination), "reflog", "expire", "--expire=now", "--all")
    gc = _git("-C", str(destination), "gc", "--prune=now", "--quiet", timeout=120)
    if gc.returncode != 0:
        raise WorkspaceError(gc.stderr.strip() or "git gc failed")


class WorkspaceManager:
    def __init__(self, worktrees_dir: Path) -> None:
        self.worktrees_dir = worktrees_dir.resolve()
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        self._boundary = WorkspaceBoundary(self.worktrees_dir)

    def create(self, *, repo_path: Path, base_commit: str, run_id: str) -> RunWorkspace:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not run_id or any(char not in allowed for char in run_id):
            raise WorkspaceError("run_id contains unsupported characters")
        destination = self._boundary.resolve(run_id)
        if destination.exists():
            raise WorkspaceError(f"workspace already exists: {run_id}")
        truncate_git_to_commit(repo_path.resolve(), destination, base_commit)
        return RunWorkspace(run_id=run_id, path=destination, base_commit=base_commit)
