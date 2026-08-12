"""Git worktree lifecycle for isolated benchmark runs."""

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
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path.resolve()),
                "worktree",
                "add",
                "--detach",
                str(destination),
                base_commit,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise WorkspaceError(completed.stderr.strip())
        return RunWorkspace(run_id=run_id, path=destination, base_commit=base_commit)
