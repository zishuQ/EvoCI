from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from evoci.tools.shell import CommandRunner


def _pid_running(pid: int) -> bool:
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return False
    try:
        state = proc.joinpath("stat").read_text().split()[2]
    except OSError:
        return False
    return state not in {"Z"}


@pytest.mark.asyncio
async def test_timeout_kills_process_group_and_returns_promptly(tmp_path: Path) -> None:
    script = tmp_path / "hold.py"
    script.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.close(1)\n"
        "    os.close(2)\n"
        "    while True:\n"
        "        time.sleep(1)\n"
        "Path('pids.txt').write_text(f'{os.getpid()}\\n{child}\\n')\n"
        "time.sleep(30)\n"
    )
    runner = CommandRunner(tmp_path, timeout=0.5, max_chars=2000)
    started = time.monotonic()
    result = await runner.run([sys.executable, str(script.name)])
    elapsed = time.monotonic() - started
    assert result.timed_out is True
    assert elapsed < 3.0
    time.sleep(0.2)
    pids_file = tmp_path / "pids.txt"
    assert pids_file.exists()
    pids = [int(line) for line in pids_file.read_text().split() if line.strip().isdigit()]
    assert pids
    for pid in pids:
        assert not _pid_running(pid)


@pytest.mark.asyncio
async def test_output_is_bounded_without_loading_the_entire_stream(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, timeout=5, max_chars=64)
    result = await runner.run(
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 10_000)"]
    )
    assert len(result.stdout) <= 64
    assert result.truncated is True
    assert result.timed_out is False
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_short_command_output_stays_compatible(tmp_path: Path) -> None:
    runner = CommandRunner(tmp_path, timeout=5, max_chars=2000)
    result = await runner.run([sys.executable, "-c", "print('hello')"])
    assert result.exit_code == 0
    assert result.timed_out is False
    assert "hello" in result.stdout
    assert result.truncated is False
