"""Atomic file-edit apply and attempt-baseline restore."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from evoci.domain.models import FileEdit
from evoci.tools.filesystem import FileTools
from evoci.tools.policy import PolicyViolation, WorkspaceBoundary


class PatchError(RuntimeError):
    pass


class PatchConflict(PatchError):
    pass


class PatchPrecheckError(PatchError):
    pass


class FileBaseline(TypedDict):
    content: str | None
    mode: int | None


def _intended_content(edit: FileEdit) -> str | None:
    return None if edit.delete else edit.content


def _read_text_or_none(target: Path) -> str | None:
    if target.exists() and target.is_file():
        return target.read_text(encoding="utf-8")
    return None


def _normalize_baseline(baseline: Mapping[str, object]) -> dict[str, FileBaseline]:
    normalized: dict[str, FileBaseline] = {}
    for path, value in baseline.items():
        if isinstance(value, str) or value is None:
            normalized[path] = {"content": value, "mode": None}
            continue
        if isinstance(value, Mapping):
            content = value.get("content")
            mode = value.get("mode")
            if content is not None and not isinstance(content, str):
                raise PatchError(f"invalid attempt baseline content for {path}")
            if mode is not None and not isinstance(mode, int):
                raise PatchError(f"invalid attempt baseline mode for {path}")
            normalized[path] = {"content": content, "mode": mode}
            continue
        raise PatchError(f"invalid attempt baseline for {path}")
    return normalized


def snapshot_edit_baseline(root: Path, edits: list[FileEdit]) -> dict[str, FileBaseline]:
    tools = FileTools(root)
    baseline: dict[str, FileBaseline] = {}
    for edit in edits:
        target = tools.boundary.resolve(edit.path)
        if target.exists() and not target.is_file():
            raise PatchPrecheckError(f"patch target is not a regular file: {edit.path}")
        if target.exists():
            baseline[edit.path] = {
                "content": target.read_text(encoding="utf-8"),
                "mode": stat.S_IMODE(target.stat().st_mode),
            }
        else:
            baseline[edit.path] = {"content": None, "mode": None}
    return baseline


def attempt_targets(edits: list[FileEdit]) -> dict[str, str | None]:
    return {edit.path: _intended_content(edit) for edit in edits}


def recover_attempt_writes(root: Path, edits: list[FileEdit]) -> dict[str, str | None]:
    """Reconstruct writes that already match this attempt's desired state."""

    tools = FileTools(root)
    written: dict[str, str | None] = {}
    for edit in edits:
        intended = _intended_content(edit)
        target = tools.boundary.resolve(edit.path)
        if _read_text_or_none(target) == intended:
            written[edit.path] = intended
    return written


def restore_attempt_writes(
    root: Path,
    baseline: Mapping[str, object],
    written: dict[str, str | None],
) -> tuple[list[str], list[str]]:
    """Restore only files this attempt wrote, and only if they still match that write."""

    normalized = _normalize_baseline(baseline)
    subset = {path: normalized.get(path, {"content": None, "mode": None}) for path in written}
    tools = FileTools(root, writable=True)
    eligible: dict[str, FileBaseline] = {}
    for path, applied in written.items():
        target = tools.boundary.resolve(path)
        if _read_text_or_none(target) != applied:
            continue
        eligible[path] = subset[path]
    return restore_edit_baseline(root, eligible)


def restore_edit_baseline(
    root: Path, baseline: Mapping[str, object]
) -> tuple[list[str], list[str]]:
    tools = FileTools(root, writable=True)
    restored: list[str] = []
    removed: list[str] = []
    for path, snapshot in _normalize_baseline(baseline).items():
        content = snapshot["content"]
        mode = snapshot["mode"]
        target = tools.boundary.resolve(path)
        if content is None:
            if target.exists():
                if not target.is_file():
                    raise PatchError(f"rollback target is not a regular file: {path}")
                target.unlink()
                removed.append(path)
            continue
        current = _read_text_or_none(target)
        current_mode = (
            stat.S_IMODE(target.stat().st_mode) if target.exists() and target.is_file() else None
        )
        if current == content:
            if mode is not None and current_mode != mode:
                target.chmod(mode)
                restored.append(path)
            continue
        tools.write_file(path, content, mode=mode)
        restored.append(path)
    return restored, removed


def _file_digest(target: Path) -> str:
    return hashlib.sha256(target.read_bytes()).hexdigest()


def precheck_edits(root: Path, edits: list[FileEdit], *, hash_strict: bool = True) -> None:
    tools = FileTools(root)
    seen: set[str] = set()
    for edit in edits:
        if edit.path in seen:
            raise PatchPrecheckError(f"duplicate patch target: {edit.path}")
        seen.add(edit.path)
        target = tools.boundary.resolve(edit.path)
        if target.exists() and not target.is_file():
            raise PatchPrecheckError(f"patch target is not a regular file: {edit.path}")
        # Desired-state replay: an already-applied create/update/delete is not a conflict.
        if _read_text_or_none(target) == _intended_content(edit):
            continue
        if edit.expected_sha256 and hash_strict:
            if not target.exists():
                raise PatchConflict(f"file changed since proposal: {edit.path}")
            actual = _file_digest(target)
            if actual != edit.expected_sha256:
                raise PatchConflict(f"file changed since proposal: {edit.path}")
        elif edit.delete and not target.exists():
            continue


def apply_edit(
    root: Path, tools: FileTools, edit: FileEdit, *, hash_strict: bool = True
) -> tuple[list[str], list[str]]:
    target = tools.boundary.resolve(edit.path)
    existed = target.exists()
    if edit.delete:
        if not target.exists():
            return [], []
        if edit.expected_sha256 and hash_strict:
            actual = _file_digest(target)
            if actual != edit.expected_sha256:
                raise PatchConflict(f"file changed since proposal: {edit.path}")
        target.unlink()
        return [], [edit.path]
    assert edit.content is not None
    if target.exists() and target.read_text(encoding="utf-8") == edit.content:
        return [], []
    if edit.expected_sha256 and hash_strict:
        if not target.exists():
            raise PatchConflict(f"file changed since proposal: {edit.path}")
        actual = _file_digest(target)
        if actual != edit.expected_sha256:
            raise PatchConflict(f"file changed since proposal: {edit.path}")
    tools.write_file(edit.path, edit.content)
    return ([edit.path], []) if not existed else ([], [edit.path])


def apply_edits(
    root: Path,
    edits: list[FileEdit],
    *,
    max_chars: int = 32_000,
    hash_strict: bool = True,
) -> tuple[list[str], list[str]]:
    tools = FileTools(root, writable=True, max_chars=max_chars)
    created: list[str] = []
    modified: list[str] = []
    for edit in edits:
        new_files, changed_files = apply_edit(root, tools, edit, hash_strict=hash_strict)
        created.extend(new_files)
        modified.extend(changed_files)
    return created, modified


def _resolve_staged_path(boundary: WorkspaceBoundary, path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise PolicyViolation(f"path escapes workspace: {path}")
    raw = boundary.root / relative
    if raw.is_symlink():
        raise PolicyViolation(f"refusing to operate on symlink: {path}")
    target = boundary.resolve(path)
    if target.is_symlink():
        raise PolicyViolation(f"refusing to operate on symlink: {path}")
    if target.exists() and not stat.S_ISREG(target.stat().st_mode):
        raise PolicyViolation(f"target is not a regular file: {path}")
    return target


def _read_utf8_text(target: Path, path: str) -> str:
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PolicyViolation(f"staged file is not valid UTF-8 text: {path}") from exc


def collect_staged_edits(
    source: Path,
    staging: Path,
    changed_paths: Sequence[str],
    *,
    max_files: int = 20,
    max_total_bytes: int = 4 * 1024 * 1024,
) -> list[FileEdit]:
    unique_paths = sorted({path for path in changed_paths if path})
    if len(unique_paths) > max_files:
        raise PolicyViolation(f"staged changes include too many files: {len(unique_paths)}")
    source_boundary = WorkspaceBoundary(source)
    staging_boundary = WorkspaceBoundary(staging)
    edits: list[FileEdit] = []
    total_bytes = 0
    for path in unique_paths:
        source_target = _resolve_staged_path(source_boundary, path)
        staging_target = _resolve_staged_path(staging_boundary, path)
        source_exists = source_target.exists()
        staging_exists = staging_target.exists()
        if source_exists and not staging_exists:
            edits.append(
                FileEdit(
                    path=path,
                    delete=True,
                    expected_sha256=_file_digest(source_target),
                )
            )
            continue
        if not source_exists and not staging_exists:
            continue
        staging_content = _read_utf8_text(staging_target, path)
        if source_exists and source_target.read_bytes() == staging_content.encode("utf-8"):
            continue
        size = len(staging_content.encode("utf-8"))
        if total_bytes + size > max_total_bytes:
            raise PolicyViolation("staged changes exceed the total byte limit")
        total_bytes += size
        edits.append(
            FileEdit(
                path=path,
                content=staging_content,
                expected_sha256=_file_digest(source_target) if source_exists else None,
            )
        )
    return edits
