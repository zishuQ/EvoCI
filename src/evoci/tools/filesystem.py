"""Read/search/write tools with strict path validation."""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evoci.tools.policy import PolicyViolation, WorkspaceBoundary


class PatchApplicationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    deleted_files: list[str] = Field(default_factory=list)


class FileTools:
    def __init__(self, root: Path, *, writable: bool = False, max_chars: int = 32_000) -> None:
        self.boundary = WorkspaceBoundary(root)
        self.writable = writable
        self.max_chars = max_chars

    def _require_write(self) -> None:
        if not self.writable:
            raise PolicyViolation("worker does not have write permission")

    def _resolve_regular_file(self, path: str, *, must_exist: bool) -> Path:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise PolicyViolation(f"path escapes workspace: {path}")
        raw = self.boundary.root / relative
        if raw.is_symlink():
            raise PolicyViolation(f"refusing to operate on symlink: {path}")
        target = self.boundary.resolve(path, must_exist=must_exist)
        if target.is_symlink():
            raise PolicyViolation(f"refusing to operate on symlink: {path}")
        if target.exists() and not stat.S_ISREG(target.stat().st_mode):
            raise PolicyViolation(f"target is not a regular file: {path}")
        return target

    def _read_utf8_text(self, target: Path, path: str) -> str:
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise PolicyViolation(f"file is not valid UTF-8 text: {path}") from exc

    def read_file(self, path: str, *, offset: int = 0, limit: int | None = None) -> str:
        if offset < 0 or (limit is not None and limit < 1):
            raise ValueError("offset must be non-negative and limit must be positive")
        target = self.boundary.resolve(path, must_exist=True)
        if not target.is_file():
            raise IsADirectoryError(target)
        content = target.read_text(encoding="utf-8", errors="replace")
        window = min(limit or self.max_chars, self.max_chars)
        return content[offset : offset + window]

    def list_files(self, path: str = ".", *, limit: int = 500) -> list[str]:
        target = self.boundary.resolve(path, must_exist=True)
        files = target.rglob("*") if target.is_dir() else (target,)
        result: list[str] = []
        for item in files:
            if item.is_file() and ".git" not in item.parts:
                result.append(str(item.relative_to(self.boundary.root)))
                if len(result) >= limit:
                    break
        return sorted(result)

    def search_code(self, pattern: str, path: str = ".", *, limit: int = 100) -> list[str]:
        expression = re.compile(pattern)
        matches: list[str] = []
        for relative in self.list_files(path):
            target = self.boundary.resolve(relative, must_exist=True)
            try:
                lines = target.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for line_number, line in enumerate(lines, 1):
                if expression.search(line):
                    matches.append(f"{relative}:{line_number}:{line[:300]}")
                    if len(matches) >= limit:
                        return matches
        return matches

    def write_file(self, path: str, content: str, *, mode: int | None = None) -> None:
        if not self.writable:
            raise PolicyViolation("worker does not have write permission")
        target = self.boundary.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the destination so a failed/partial write cannot truncate it.
        if mode is not None:
            file_mode = stat.S_IMODE(mode)
        elif target.exists():
            file_mode = stat.S_IMODE(target.stat().st_mode)
        else:
            file_mode = 0o600
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        staged = Path(temporary)
        os.close(fd)
        try:
            staged.write_text(content, encoding="utf-8")
            staged.chmod(file_mode)
            os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)

    def apply_patch(self, files: dict[str, str]) -> PatchApplicationResult:
        """Apply a bounded set of full-file replacements through the writer boundary."""
        self._require_write()
        if len(files) > 20:
            raise PolicyViolation("patch changes too many files")
        created: list[str] = []
        modified: list[str] = []
        for path, content in files.items():
            target = self.boundary.resolve(path)
            existed = target.exists()
            self.write_file(path, content)
            (modified if existed else created).append(path)
        return PatchApplicationResult(created_files=created, modified_files=modified)

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> PatchApplicationResult:
        self._require_write()
        if not old_text:
            raise PolicyViolation("old_text must not be empty")
        target = self._resolve_regular_file(path, must_exist=True)
        content = self._read_utf8_text(target, path)
        matches = content.count(old_text)
        if matches != expected_replacements:
            raise PolicyViolation(
                f"replace_text expected {expected_replacements} occurrence(s) of old_text "
                f"in {path}, found {matches}"
            )
        self.write_file(path, content.replace(old_text, new_text, expected_replacements))
        return PatchApplicationResult(modified_files=[path])

    def create_file(self, path: str, content: str) -> PatchApplicationResult:
        self._require_write()
        target = self._resolve_regular_file(path, must_exist=False)
        if target.exists():
            raise PolicyViolation(f"file already exists: {path}")
        self.write_file(path, content)
        return PatchApplicationResult(created_files=[path])

    def delete_file(self, path: str) -> PatchApplicationResult:
        self._require_write()
        target = self._resolve_regular_file(path, must_exist=True)
        target.unlink()
        return PatchApplicationResult(deleted_files=[path])
