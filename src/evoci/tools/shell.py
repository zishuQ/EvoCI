"""No-shell subprocess execution with timeouts and bounded output."""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evoci.tools.policy import WorkspaceBoundary, validate_command


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


class CommandRunner:
    def __init__(self, root: Path, *, timeout: float = 120.0, max_chars: int = 32_000) -> None:
        self.boundary = WorkspaceBoundary(root)
        self.timeout = timeout
        self.max_chars = max_chars

    async def run(
        self,
        argv: list[str],
        *,
        cwd: str = ".",
        network: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> CommandResult:
        validate_command(argv, network=network)
        resolved_cwd = self.boundary.resolve(cwd, must_exist=True)
        allowed_env_keys = {"PATH", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV"}
        environment = {key: value for key, value in os.environ.items() if key in allowed_env_keys}
        if extra_env:
            invalid = set(extra_env) - {"CI", "PYTHONPATH", "PYTHONWARNINGS"}
            if invalid:
                raise ValueError(f"environment keys are not allowlisted: {sorted(invalid)}")
            environment.update(extra_env)
        with tempfile.TemporaryDirectory(prefix="evoci-pycache-") as pycache:
            environment["PYTHONPYCACHEPREFIX"] = pycache
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=resolved_cwd,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            timed_out = False
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )
            except TimeoutError:
                timed_out = True
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
        return CommandResult(
            argv=tuple(argv),
            cwd=str(resolved_cwd),
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode(errors="replace")[-self.max_chars :],
            stderr=stderr_bytes.decode(errors="replace")[-self.max_chars :],
            timed_out=timed_out,
        )
