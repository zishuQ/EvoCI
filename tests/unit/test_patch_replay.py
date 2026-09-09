from __future__ import annotations

import hashlib
from pathlib import Path

from evoci.domain.models import FileEdit
from evoci.graph.builder import _apply_edit
from evoci.tools.filesystem import FileTools
from evoci.tools.patch import precheck_edits


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def test_same_patch_replay_is_a_successful_noop(tmp_path: Path) -> None:
    original = "VALUE = 'A'\n"
    desired = "VALUE = 'B'\n"
    (tmp_path / "app.py").write_text(original)
    edit = FileEdit(path="app.py", content=desired, expected_sha256=_hash(original))
    tools = FileTools(tmp_path, writable=True)

    assert _apply_edit(tmp_path, tools, edit) == ([], ["app.py"])
    assert _apply_edit(tmp_path, tools, edit) == ([], [])
    assert (tmp_path / "app.py").read_text() == desired


def test_partial_multi_file_patch_can_resume_after_first_file(tmp_path: Path) -> None:
    original = "VALUE = 'A'\n"
    desired = "VALUE = 'B'\n"
    for path in ("a.py", "b.py"):
        (tmp_path / path).write_text(original)
    edits = [
        FileEdit(path=path, content=desired, expected_sha256=_hash(original))
        for path in ("a.py", "b.py")
    ]
    tools = FileTools(tmp_path, writable=True)

    _apply_edit(tmp_path, tools, edits[0])
    results = [_apply_edit(tmp_path, tools, edit) for edit in edits]

    assert results == [([], []), ([], ["b.py"])]
    assert (tmp_path / "a.py").read_text() == desired
    assert (tmp_path / "b.py").read_text() == desired


def test_delete_and_create_replay_are_noops_at_desired_state(tmp_path: Path) -> None:
    tools = FileTools(tmp_path, writable=True)
    delete = FileEdit(path="gone.py", delete=True, expected_sha256=_hash("old\n"))
    create = FileEdit(path="new.py", content="new\n")

    assert _apply_edit(tmp_path, tools, delete) == ([], [])
    assert _apply_edit(tmp_path, tools, create) == (["new.py"], [])
    assert _apply_edit(tmp_path, tools, create) == ([], [])
    precheck_edits(tmp_path, [delete, create])
