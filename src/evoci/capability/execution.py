"""Restricted execution of scripts owned by validated capabilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from evoci.capability.models import ScriptExecutionResult
from evoci.capability.registry import CapabilityRegistry
from evoci.tools.policy import PolicyViolation, WorkspaceBoundary
from evoci.tools.shell import run_grouped_subprocess


def run_skill_script(
    registry: CapabilityRegistry,
    *,
    skill_id: str,
    script_name: str,
    args: list[str],
    workspace: Path,
    timeout: float = 30.0,
    max_chars: int = 32_000,
    observed_revision: int | None = None,
) -> ScriptExecutionResult:
    record = registry.get(skill_id)
    if record is None:
        raise KeyError(f"unknown skill: {skill_id}")
    if not record.manifest.enabled:
        raise PolicyViolation("only enabled skill scripts may execute")
    if not record.manifest.permissions.execute:
        raise PolicyViolation("skill does not declare execute permission")
    package = Path(record.package_path).resolve()
    script = (package / "scripts" / script_name).resolve()
    if package not in script.parents or not script.is_file():
        raise PolicyViolation("script is not part of this skill package")
    declared = {file.path for file in record.manifest.files}
    if str(script.relative_to(package)) not in declared:
        raise PolicyViolation("script is not declared in manifest")
    if script.suffix != ".py":
        raise PolicyViolation(f"unsupported_runner: {script.name}")
    resolved_workspace = WorkspaceBoundary(workspace).resolve(".", must_exist=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
    }
    completed = run_grouped_subprocess(
        [sys.executable, str(script), *args],
        cwd=resolved_workspace,
        env=environment,
        timeout=timeout,
        max_chars=max_chars,
    )
    return ScriptExecutionResult(
        exit_code=completed.exit_code,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=completed.timed_out,
        observed_revision=observed_revision,
    )
