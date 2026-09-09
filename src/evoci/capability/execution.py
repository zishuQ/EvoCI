"""Restricted execution of scripts owned by validated capabilities."""

from __future__ import annotations

import os
import signal
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
    observed_revision: int | None = None,
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
    if script.suffix != ".py":
        raise PolicyViolation(f"unsupported_runner: {script.name}")
    resolved_workspace = WorkspaceBoundary(workspace).resolve(".", must_exist=True)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
    }
    process = subprocess.Popen(
        [sys.executable, str(script), *args],
        cwd=resolved_workspace,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        stdout_bytes, stderr_bytes = process.communicate(timeout=2)
    stdout = (stdout_bytes or b"").decode(errors="replace")[-max_chars:]
    stderr = (stderr_bytes or b"").decode(errors="replace")[-max_chars:]
    return ScriptExecutionResult(
        exit_code=process.returncode if process.returncode is not None else -1,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        observed_revision=observed_revision,
    )
