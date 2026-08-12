"""Disposable workspace copies with independent Git metadata."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class WorkspaceCopyError(RuntimeError):
    pass


def _run_git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def copy_workspace_with_independent_git(source: Path, destination: Path) -> Path:
    """Copy current files while ensuring `.git` never points back to `source`."""

    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", ".evoci", ".evoci_runtime"),
    )
    if not (source / ".git").exists():
        return destination

    head = _run_git("-C", str(source), "rev-parse", "HEAD")
    if head.returncode != 0:
        initialized = _run_git("init", "-q", str(destination))
        if initialized.returncode != 0:
            raise WorkspaceCopyError(initialized.stderr.strip() or "git init failed")
        return destination

    metadata_clone = destination.parent / f".{destination.name}-git-metadata"
    if metadata_clone.exists():
        raise FileExistsError(metadata_clone)
    cloned = _run_git(
        "clone",
        "--no-local",
        "--no-hardlinks",
        "--no-checkout",
        str(source),
        str(metadata_clone),
    )
    if cloned.returncode != 0:
        raise WorkspaceCopyError(cloned.stderr.strip() or "git metadata clone failed")
    try:
        shutil.move(str(metadata_clone / ".git"), str(destination / ".git"))
    finally:
        shutil.rmtree(metadata_clone, ignore_errors=True)

    updated = _run_git("-C", str(destination), "update-ref", "HEAD", head.stdout.strip())
    if updated.returncode != 0:
        raise WorkspaceCopyError(updated.stderr.strip() or "git update-ref failed")
    reset = _run_git("-C", str(destination), "reset", "--mixed", "HEAD")
    if reset.returncode != 0:
        raise WorkspaceCopyError(reset.stderr.strip() or "git reset failed")
    _run_git("-C", str(destination), "remote", "remove", "origin")
    return destination
