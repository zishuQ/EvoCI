"""Atomic file-edit apply and attempt-baseline restore."""

from __future__ import annotations

import hashlib
from pathlib import Path

from evoci.domain.models import FileEdit
from evoci.tools.filesystem import FileTools


class PatchError(RuntimeError):
    pass


class PatchConflict(PatchError):
    pass


class PatchPrecheckError(PatchError):
    pass


def snapshot_edit_baseline(root: Path, edits: list[FileEdit]) -> dict[str, str | None]:
    tools = FileTools(root)
    baseline: dict[str, str | None] = {}
    for edit in edits:
        target = tools.boundary.resolve(edit.path)
        if target.exists() and not target.is_file():
            raise PatchPrecheckError(f"patch target is not a regular file: {edit.path}")
        baseline[edit.path] = target.read_text(encoding="utf-8") if target.exists() else None
    return baseline


def restore_edit_baseline(
    root: Path, baseline: dict[str, str | None]
) -> tuple[list[str], list[str]]:
    tools = FileTools(root, writable=True)
    restored: list[str] = []
    removed: list[str] = []
    for path, content in baseline.items():
        target = tools.boundary.resolve(path)
        if content is None:
            if target.exists():
                if not target.is_file():
                    raise PatchError(f"rollback target is not a regular file: {path}")
                target.unlink()
                removed.append(path)
            continue
        if target.exists() and target.read_text(encoding="utf-8") == content:
            continue
        tools.write_file(path, content)
        restored.append(path)
    return restored, removed


def _file_digest(target: Path) -> str:
    return hashlib.sha256(target.read_bytes()).hexdigest()


def precheck_edits(root: Path, edits: list[FileEdit]) -> None:
    tools = FileTools(root)
    seen: set[str] = set()
    for edit in edits:
        if edit.path in seen:
            raise PatchPrecheckError(f"duplicate patch target: {edit.path}")
        seen.add(edit.path)
        target = tools.boundary.resolve(edit.path)
        if target.exists() and not target.is_file():
            raise PatchPrecheckError(f"patch target is not a regular file: {edit.path}")
        if edit.expected_sha256:
            if not target.exists():
                raise PatchConflict(f"file changed since proposal: {edit.path}")
            actual = _file_digest(target)
            if actual != edit.expected_sha256:
                raise PatchConflict(f"file changed since proposal: {edit.path}")
        elif edit.delete and not target.exists():
            continue


def apply_edit(root: Path, tools: FileTools, edit: FileEdit) -> tuple[list[str], list[str]]:
    target = tools.boundary.resolve(edit.path)
    existed = target.exists()
    if edit.delete:
        if not target.exists():
            return [], []
        if edit.expected_sha256:
            actual = _file_digest(target)
            if actual != edit.expected_sha256:
                raise PatchConflict(f"file changed since proposal: {edit.path}")
        target.unlink()
        return [], [edit.path]
    assert edit.content is not None
    if target.exists() and target.read_text(encoding="utf-8") == edit.content:
        return [], []
    if edit.expected_sha256:
        if not target.exists():
            raise PatchConflict(f"file changed since proposal: {edit.path}")
        actual = _file_digest(target)
        if actual != edit.expected_sha256:
            raise PatchConflict(f"file changed since proposal: {edit.path}")
    tools.write_file(edit.path, edit.content)
    return ([edit.path], []) if not existed else ([], [edit.path])


def apply_edits(
    root: Path, edits: list[FileEdit], *, max_chars: int = 32_000
) -> tuple[list[str], list[str]]:
    tools = FileTools(root, writable=True, max_chars=max_chars)
    created: list[str] = []
    modified: list[str] = []
    for edit in edits:
        new_files, changed_files = apply_edit(root, tools, edit)
        created.extend(new_files)
        modified.extend(changed_files)
    return created, modified
