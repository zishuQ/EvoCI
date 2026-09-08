from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evoci.domain.models import FileEdit
from evoci.tools.patch import (
    PatchConflict,
    apply_edits,
    precheck_edits,
    restore_edit_baseline,
    snapshot_edit_baseline,
)


def test_hash_mismatch_precheck_does_not_write(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("keep-a\n")
    (tmp_path / "b.txt").write_text("keep-b\n")
    digest_a = hashlib.sha256(b"keep-a\n").hexdigest()
    edits = [
        FileEdit(path="a.txt", content="new-a\n", expected_sha256=digest_a),
        FileEdit(path="b.txt", content="new-b\n", expected_sha256="0" * 64),
    ]
    with pytest.raises(PatchConflict, match=r"b\.txt"):
        precheck_edits(tmp_path, edits)
    assert (tmp_path / "a.txt").read_text() == "keep-a\n"
    assert (tmp_path / "b.txt").read_text() == "keep-b\n"


def test_restore_covers_create_update_and_delete(tmp_path: Path) -> None:
    (tmp_path / "update.txt").write_text("old\n")
    (tmp_path / "delete.txt").write_text("gone-later\n")
    (tmp_path / "unrelated.txt").write_text("user-dirty\n")
    edits = [
        FileEdit(path="created.txt", content="new-file\n"),
        FileEdit(path="update.txt", content="updated\n"),
        FileEdit(path="delete.txt", delete=True),
    ]
    baseline = snapshot_edit_baseline(tmp_path, edits)
    apply_edits(tmp_path, edits)
    assert (tmp_path / "created.txt").read_text() == "new-file\n"
    assert (tmp_path / "update.txt").read_text() == "updated\n"
    assert not (tmp_path / "delete.txt").exists()
    restore_edit_baseline(tmp_path, baseline)
    assert not (tmp_path / "created.txt").exists()
    assert (tmp_path / "update.txt").read_text() == "old\n"
    assert (tmp_path / "delete.txt").read_text() == "gone-later\n"
    assert (tmp_path / "unrelated.txt").read_text() == "user-dirty\n"


def test_restore_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("old\n")
    baseline = snapshot_edit_baseline(tmp_path, [FileEdit(path="a.txt", content="new\n")])
    apply_edits(tmp_path, [FileEdit(path="a.txt", content="new\n")])
    restore_edit_baseline(tmp_path, baseline)
    restore_edit_baseline(tmp_path, baseline)
    assert (tmp_path / "a.txt").read_text() == "old\n"
