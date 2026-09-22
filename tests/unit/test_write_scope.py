from pathlib import Path

import pytest

from evoci.tools.policy import FIXER_CAPABILITIES, INVESTIGATOR_CAPABILITIES, PolicyViolation
from evoci.tools.registry import create_worker_registry
from evoci.tools.scope import assert_write_path_allowed, normalize_write_path


def test_normalize_write_path_rejects_escape() -> None:
    with pytest.raises(PolicyViolation):
        normalize_write_path("../secret.py")
    with pytest.raises(PolicyViolation):
        normalize_write_path("/tmp/x.py")
    assert normalize_write_path("./app.py") == "app.py"


def test_write_scope_blocks_unlisted_paths(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("y = 1\n", encoding="utf-8")
    registry = create_worker_registry(
        FIXER_CAPABILITIES, tmp_path, write_scope=["app.py"]
    )
    try:
        registry.invoke("replace_text", path="app.py", old_text="x = 1", new_text="x = 2")
        with pytest.raises(PolicyViolation, match="write_scope"):
            registry.invoke("replace_text", path="other.py", old_text="y = 1", new_text="y = 2")
    finally:
        registry.close()


def test_investigate_registry_cannot_write(tmp_path: Path) -> None:
    registry = create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path)
    try:
        with pytest.raises(PolicyViolation):
            registry.invoke("create_file", path="app.py", content="x\n")
    finally:
        registry.close()


def test_assert_write_path_allowed_accepts_new_files(tmp_path: Path) -> None:
    assert assert_write_path_allowed("new.py", {"new.py"}, workspace=tmp_path) == "new.py"
