"""Filesystem, command, and worker permission policies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class PolicyViolation(PermissionError):
    """A requested tool operation exceeded its granted capability."""


@dataclass(frozen=True, slots=True)
class WorkerCapabilities:
    read_files: bool
    search_code: bool
    git_read: bool
    execute_tests: bool
    write_files: bool
    network: bool = False


SUPERVISOR_CAPABILITIES = WorkerCapabilities(True, True, True, False, False)
INVESTIGATOR_CAPABILITIES = WorkerCapabilities(True, True, True, True, False)
WORKER_INVESTIGATE_CAPABILITIES = INVESTIGATOR_CAPABILITIES
WORKER_REPAIR_CAPABILITIES = WorkerCapabilities(True, True, True, True, True)
FIXER_CAPABILITIES = WORKER_REPAIR_CAPABILITIES
REVIEWER_CAPABILITIES = WorkerCapabilities(True, True, True, True, False)


class WorkspaceBoundary:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str | Path, *, must_exist: bool = False) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PolicyViolation(f"path escapes workspace: {path}")
        if must_exist and not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate


SAFE_COMMANDS = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "ruff",
        "mypy",
        "git",
        "uv",
        "npm",
        "pnpm",
        "yarn",
        "node",
        "cargo",
        "go",
    }
)


def validate_command(argv: list[str], *, network: bool = False) -> None:
    if not argv:
        raise PolicyViolation("empty command")
    executable = Path(argv[0]).name
    versioned_python = re.fullmatch(r"python3(?:\.\d+)?", executable)
    if executable not in SAFE_COMMANDS and versioned_python is None:
        raise PolicyViolation(f"command is not allowlisted: {executable}")
    joined = " ".join(argv).lower()
    forbidden = ("sudo ", "rm -rf", "curl ", "wget ", "--privileged", "/etc/", "../")
    if any(token in joined for token in forbidden):
        raise PolicyViolation("command contains a forbidden pattern")
    if (
        not network
        and executable in {"npm", "pnpm", "yarn", "uv"}
        and any(token in argv for token in ("add", "install", "publish", "upload"))
    ):
        raise PolicyViolation("network-affecting package command is disabled")
