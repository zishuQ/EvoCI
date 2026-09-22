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


def _commit_isolated_baseline(destination: Path) -> None:
    _run_git("-C", str(destination), "config", "user.name", "EvoCI isolated workspace")
    _run_git("-C", str(destination), "config", "user.email", "evoci-isolated@example.invalid")
    added = _run_git("-C", str(destination), "add", "--all")
    committed = _run_git(
        "-C", str(destination), "commit", "--allow-empty", "-qm", "isolated baseline"
    )
    if added.returncode != 0 or committed.returncode != 0:
        raise WorkspaceCopyError(committed.stderr.strip() or "git baseline commit failed")


def copy_workspace_with_independent_git(source: Path, destination: Path) -> Path:
    """Copy current files while ensuring `.git` never points back to `source`."""

    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", ".evoci", ".evoci_runtime", "__pycache__", "*.pyc", ".pytest_cache"
        ),
    )
    if not (source / ".git").exists():
        initialized = _run_git("init", "-q", str(destination))
        if initialized.returncode != 0:
            raise WorkspaceCopyError(initialized.stderr.strip() or "git init failed")
        _commit_isolated_baseline(destination)
        return destination

    head = _run_git("-C", str(source), "rev-parse", "HEAD")
    if head.returncode != 0:
        initialized = _run_git("init", "-q", str(destination))
        if initialized.returncode != 0:
            raise WorkspaceCopyError(initialized.stderr.strip() or "git init failed")
        _commit_isolated_baseline(destination)
        return destination

    # Copy the repository metadata locally.  Calling `git clone` here can ask the
    # source's origin for missing shallow objects, which is both non-deterministic
    # and invalid for prepared offline benchmark workspaces.
    git_metadata = source / ".git"
    preserve_head = git_metadata.is_dir()
    if git_metadata.is_dir():
        shutil.copytree(git_metadata, destination / ".git")
    else:
        initialized = _run_git("init", "-q", str(destination))
        if initialized.returncode != 0:
            raise WorkspaceCopyError(initialized.stderr.strip() or "git init failed")
        _commit_isolated_baseline(destination)

    if preserve_head:
        updated = _run_git("-C", str(destination), "update-ref", "HEAD", head.stdout.strip())
        if updated.returncode != 0:
            raise WorkspaceCopyError(updated.stderr.strip() or "git update-ref failed")
        reset = _run_git("-C", str(destination), "reset", "--mixed", "HEAD")
        if reset.returncode != 0:
            raise WorkspaceCopyError(reset.stderr.strip() or "git reset failed")
    _run_git("-C", str(destination), "remote", "remove", "origin")
    return destination


def workspace_snapshot_id(workspace: Path) -> str:
    """Content fingerprint of HEAD plus uncommitted and untracked files."""

    import hashlib

    root = workspace.resolve()
    head = _run_git("-C", str(root), "rev-parse", "HEAD")
    revision = head.stdout.strip() if head.returncode == 0 else "no-head"
    diff = _run_git("-C", str(root), "diff", "HEAD")
    untracked = _run_git(
        "-C",
        str(root),
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    digest = hashlib.sha256()
    digest.update(revision.encode())
    digest.update(b"\n")
    digest.update((diff.stdout or "").encode())
    names = (untracked.stdout or "").split("\0")
    for name in sorted(part for part in names if part):
        digest.update(name.encode())
        target = root / name
        if target.is_file() and not target.is_symlink():
            digest.update(target.read_bytes())
    return f"{revision}:{digest.hexdigest()[:16]}"
