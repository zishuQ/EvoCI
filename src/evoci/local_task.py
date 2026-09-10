"""Turn a repository and a verification command into a reproducible repair task."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from evoci.config import EvoCIConfig
from evoci.domain.models import CIFailure, RepoSpec
from evoci.tools.policy import validate_command
from evoci.tools.shell import CommandResult, CommandRunner
from evoci.verification.service import VerificationService


def parse_verification_command(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    argv = list(lexer)
    if any(token and set(token) <= set(";&|<>") for token in argv):
        raise ValueError("Use one command without shell pipes, redirects or chaining.")
    validate_command(argv)
    return argv


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass(frozen=True)
class LocalRepository:
    root: Path
    spec: RepoSpec
    dirty_status: str


def inspect_repository(path: Path) -> LocalRepository:
    requested = path.expanduser().resolve()
    if not requested.is_dir():
        raise ValueError(f"Repository directory does not exist: {requested}")
    root_text = _git(requested, "rev-parse", "--show-toplevel")
    if root_text is None:
        raise ValueError("evoci fix requires a Git repository; run git init first.")
    root = Path(root_text).resolve()
    commit = _git(root, "rev-parse", "HEAD") or "HEAD"
    owner, name = "local", root.name
    remote = _git(root, "remote", "get-url", "origin")
    if remote:
        remote_path = urlsplit(remote).path if "://" in remote else remote.split(":", 1)[-1]
        parts = remote_path.strip("/").removesuffix(".git").split("/")
        if len(parts) >= 2:
            owner, name = parts[-2:]
    return LocalRepository(
        root=root,
        spec=RepoSpec(owner=owner, name=name, base_commit=commit),
        dirty_status=_git(root, "status", "--porcelain=v1", "--untracked-files=normal") or "",
    )


@dataclass(frozen=True)
class PreparedLocalTask:
    initial: dict[str, object]
    task_file: Path
    report_file: Path
    preflight: CommandResult


async def prepare_local_task(
    repository: LocalRepository,
    argv: list[str],
    config: EvoCIConfig,
    *,
    description: str | None = None,
) -> PreparedLocalTask:
    """Probe a disposable copy, preserving the user's dirty files and Git index."""
    run_id = f"run-{uuid4().hex[:12]}"
    service = VerificationService(
        timeout=config.command_timeout_seconds,
        max_chars=config.output_limit_chars,
    )
    async with service.isolated_workspace(repository.root) as snapshot:
        completed = await CommandRunner(
            snapshot,
            timeout=config.command_timeout_seconds,
            max_chars=config.output_limit_chars,
        ).run(argv, extra_env={"CI": "1"})
    failure = CIFailure(
        summary=description or f"Repair failing command: {shlex.join(argv)}",
        log_excerpt=(
            f"Command: {shlex.join(argv)}\nExit code: {completed.exit_code}\n"
            f"Timed out: {completed.timed_out}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        ),
        failed_commands=[argv],
        task_family="test",
    )
    initial: dict[str, object] = {
        "run_id": run_id,
        "task_id": run_id,
        "repo": repository.spec,
        "ci_failure": failure,
        "workspace_path": str(repository.root),
    }
    directory = config.state_dir / "fix" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    task_file = directory / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "repo": repository.spec.model_dump(mode="json"),
                "ci_failure": failure.model_dump(mode="json"),
                "workspace_path": str(repository.root),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_file = directory / "preflight.json"
    report_file.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "workspace": str(repository.root),
                "command": argv,
                "base_commit": repository.spec.base_commit,
                "dirty_before": repository.dirty_status,
                "exit_code": completed.exit_code,
                "timed_out": completed.timed_out,
                "truncated": completed.truncated,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "reproduced": completed.exit_code != 0 or completed.timed_out,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedLocalTask(initial, task_file, report_file, completed)
