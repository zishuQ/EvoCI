"""No-shell subprocess execution with timeouts, process groups, and bounded output."""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
from contextlib import suppress
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
    truncated: bool = False


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    pid = process.pid
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, OSError):
        pgid = pid
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            with suppress(ProcessLookupError):
                process.kill()
            break


async def _read_bounded(stream: asyncio.StreamReader | None, limit: int) -> tuple[bytes, bool]:
    if stream is None:
        return b"", False
    buf = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        if truncated:
            continue
        room = limit - len(buf)
        if room <= 0:
            truncated = True
            continue
        if len(chunk) > room:
            buf.extend(chunk[:room])
            truncated = True
        else:
            buf.extend(chunk)
    return bytes(buf), truncated


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
                start_new_session=True,
            )
            stdout_task = asyncio.create_task(_read_bounded(process.stdout, self.max_chars))
            stderr_task = asyncio.create_task(_read_bounded(process.stderr, self.max_chars))
            waiter = asyncio.create_task(process.wait())
            timed_out = False
            try:
                await asyncio.wait_for(asyncio.shield(waiter), timeout=self.timeout)
            except TimeoutError:
                timed_out = True
                _kill_process_group(process)
                try:
                    await asyncio.wait_for(waiter, timeout=2.0)
                except TimeoutError:
                    _kill_process_group(process)
            except asyncio.CancelledError:
                _kill_process_group(process)
                waiter.cancel()
                stdout_task.cancel()
                stderr_task.cancel()
                raise
            stdout_bytes, stdout_truncated = await stdout_task
            stderr_bytes, stderr_truncated = await stderr_task
        return CommandResult(
            argv=tuple(argv),
            cwd=str(resolved_cwd),
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
        )
