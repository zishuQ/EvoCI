"""Restricted execution of scripts owned by validated capabilities."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from evoci.capability.models import ScriptExecutionResult
from evoci.capability.registry import CapabilityRegistry
from evoci.tools.policy import PolicyViolation, WorkspaceBoundary


def run_skill_script(
    registry: CapabilityRegistry,
    *,
    skill_id: str,
    version: int,
    script_name: str,
    args: list[str],
    workspace: Path,
    timeout: float = 30.0,
    max_chars: int = 32_000,
) -> ScriptExecutionResult:
    record = registry.get(skill_id, version)
    if record is None:
        raise KeyError(f"unknown skill: {skill_id} v{version}")
    if record.manifest.status not in {"trial", "active"}:
        raise PolicyViolation("only trial or active skill scripts may execute")
    if not record.manifest.permissions.execute:
        raise PolicyViolation("skill does not declare execute permission")
    package = Path(record.package_path).resolve()
    script = (package / "scripts" / script_name).resolve()
    if package not in script.parents or not script.is_file():
        raise PolicyViolation("script is not part of this skill package")
    declared = {file.path for file in record.manifest.files}
    if str(script.relative_to(package)) not in declared:
        raise PolicyViolation("script is not declared in manifest")
    resolved_workspace = WorkspaceBoundary(workspace).resolve(".", must_exist=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
    }
    try:
        result = subprocess.run(
            [sys.executable, str(script), *args],
            cwd=resolved_workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ScriptExecutionResult(
            exit_code=result.returncode,
            stdout=result.stdout[-max_chars:],
            stderr=result.stderr[-max_chars:],
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ScriptExecutionResult(
            exit_code=-1,
            stdout=(exc.stdout or "")[-max_chars:] if isinstance(exc.stdout, str) else "",
            stderr=(exc.stderr or "")[-max_chars:] if isinstance(exc.stderr, str) else "",
            timed_out=True,
        )
