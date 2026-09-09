import os
import stat
from pathlib import Path

import pytest

from evoci.tools.filesystem import FileTools


def test_atomic_write_preserves_executable_mode(tmp_path: Path) -> None:
    path = tmp_path / "script.py"
    path.write_text("old")
    path.chmod(0o751)
    FileTools(tmp_path, writable=True).write_file("script.py", "new")
    assert path.read_text() == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o751
    assert not list(tmp_path.glob(".script.py.*"))


def test_replace_failure_keeps_original_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "file.txt"
    path.write_text("original")

    def fail_replace(source: object, target: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        FileTools(tmp_path, writable=True).write_file("file.txt", "new")
    assert path.read_text() == "original"
    assert not list(tmp_path.glob(".file.txt.*"))
