"""Deterministic graph policy and risk gates."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

from evoci.domain.models import FixerOutput


def requires_approval(output: FixerOutput) -> bool:
    proposal = output.proposal
    paths = {PurePosixPath(path) for path in proposal.changed_files}
    if proposal.risk in {"medium", "high"}:
        return True
    if any(str(path).startswith(".github/workflows/") for path in paths):
        return True
    if any(edit.delete for edit in output.edits) or len(paths) > 4:
        return True
    manifest_names = {"pyproject.toml", "package.json", "cargo.toml", "go.mod"}
    lock_names = {"uv.lock", "poetry.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"}
    lowered = {path.name.lower() for path in paths}
    return bool(lowered & manifest_names and lowered & lock_names)


def contains_review_bypass(output: FixerOutput) -> list[str]:
    blockers: list[str] = []
    forbidden = (
        "pytest.skip(",
        "@pytest.mark.skip",
        "continue-on-error: true",
        "|| true",
        "# noqa",
        "eslint-disable",
    )
    for edit in output.edits:
        if edit.content and any(token in edit.content for token in forbidden):
            blockers.append(f"possible test or CI bypass in {edit.path}")
    return blockers


def contains_workspace_review_bypass(workspace: Path) -> list[str]:
    """Inspect the complete final Git diff, including untracked files."""

    forbidden = (
        "pytest.skip(",
        "@pytest.mark.skip",
        "continue-on-error: true",
        "|| true",
        "# noqa",
        "eslint-disable",
    )
    completed = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--unified=0", "HEAD", "--"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    blockers: list[str] = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                for token in forbidden:
                    if token in line:
                        blockers.append(f"possible test or CI bypass in final diff: {token}")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if status.returncode == 0:
        for row in status.stdout.splitlines():
            if not row.startswith("?? "):
                continue
            relative = row[3:]
            target = (workspace / relative).resolve()
            root = workspace.resolve()
            if root not in target.parents or not target.is_file():
                continue
            content = target.read_text(encoding="utf-8", errors="replace")
            if any(token in content for token in forbidden):
                blockers.append(f"possible test or CI bypass in untracked file {relative}")
    return sorted(set(blockers))
