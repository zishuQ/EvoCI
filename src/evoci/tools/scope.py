"""Exact relative-path write authorization without glob matching."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from evoci.tools.policy import PolicyViolation, WorkspaceBoundary


def normalize_write_path(path: str) -> str:
    """Return a workspace-relative POSIX path. New files may be declared."""

    raw = path.strip()
    if not raw or raw in {".", "./"}:
        raise PolicyViolation("write path must be a relative file path")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or Path(raw).is_absolute():
        raise PolicyViolation(f"path escapes workspace: {path}")
    parts = [part for part in candidate.parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise PolicyViolation(f"path escapes workspace: {path}")
    return str(PurePosixPath(*parts))


def normalize_write_scope(paths: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    unique = [normalize_write_path(path) for path in paths]
    return tuple(dict.fromkeys(unique))


def assert_write_path_allowed(
    path: str,
    write_scope: set[str] | frozenset[str] | tuple[str, ...],
    *,
    workspace: Path | None = None,
) -> str:
    normalized = normalize_write_path(path)
    allowed = set(write_scope)
    if allowed and normalized not in allowed:
        raise PolicyViolation(f"write outside task write_scope: {path}")
    if workspace is not None:
        boundary = WorkspaceBoundary(workspace)
        target = boundary.resolve(normalized)
        raw = (boundary.root / normalized)
        if raw.is_symlink() or target.is_symlink():
            raise PolicyViolation(f"refusing to operate on symlink: {path}")
    return normalized
