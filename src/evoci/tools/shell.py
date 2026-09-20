"""No-shell subprocess execution with timeouts, process groups, and bounded output."""

from __future__ import annotations

import asyncio
import os
import selectors
import signal
import subprocess
import tempfile
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

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


def _kill_process_group(process: Any) -> None:
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


_process_cancel: ContextVar[threading.Event | None] = ContextVar("process_cancel", default=None)
ContainerExecutor = Callable[[list[str], str, bool], Awaitable[CommandResult]]
_container_executor: ContextVar[ContainerExecutor | None] = ContextVar(
    "container_executor", default=None
)


def set_container_executor(executor: ContainerExecutor) -> Any:
    return _container_executor.set(executor)


def reset_container_executor(token: Any) -> None:
    _container_executor.reset(token)


async def run_cancellable[T](function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run synchronous process work off-loop; cancellation joins its cleanup first."""
    cancellation = threading.Event()
    token = _process_cancel.set(cancellation)
    try:
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    finally:
        _process_cancel.reset(token)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancellation.set()
        # Shield against repeated cancellation too: callers must not remove a workspace
        # or close its registry while the worker still owns a subprocess.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not worker.cancelled():
            worker.exception()
        raise


def run_grouped_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    max_chars: int,
) -> CommandResult:
    """Drain both pipes incrementally with bounded tails and process-group cleanup."""
    if max_chars < 1 or timeout <= 0:
        raise ValueError("timeout and max_chars must be positive")
    cancellation = _process_cancel.get()
    if cancellation is not None and cancellation.is_set():
        raise asyncio.CancelledError()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    tails = [bytearray(), bytearray()]
    truncated = False
    timed_out = False
    deadline = monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for index, stream in enumerate((process.stdout, process.stderr)):
                assert stream is not None
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, index)
            while selector.get_map() or process.poll() is None:
                if cancellation is not None and cancellation.is_set():
                    raise asyncio.CancelledError()
                remaining = deadline - monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 0.05)):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    tail = tails[key.data]
                    tail.extend(chunk)
                    if len(tail) > max_chars:
                        del tail[:-max_chars]
                        truncated = True
    finally:
        # Also reap background children after the direct child exits normally.
        _kill_process_group(process)
        process.wait(timeout=2)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    return CommandResult(
        argv=tuple(argv),
        cwd=str(cwd),
        exit_code=process.returncode,
        stdout=tails[0].decode(errors="replace"),
        stderr=tails[1].decode(errors="replace"),
        timed_out=timed_out,
        truncated=truncated,
    )


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
        executor = _container_executor.get()
        if executor is not None:
            return await executor(argv, cwd, network)
        return await run_cancellable(
            self.run_sync,
            argv,
            cwd=cwd,
            network=network,
            extra_env=extra_env,
        )

    def run_sync(
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
            return run_grouped_subprocess(
                argv,
                cwd=resolved_cwd,
                env=environment,
                timeout=self.timeout,
                max_chars=self.max_chars,
            )
