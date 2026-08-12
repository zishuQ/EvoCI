"""Read/search/write tools with strict path validation."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evoci.tools.policy import PolicyViolation, WorkspaceBoundary


class PatchApplicationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)


class FileTools:
    def __init__(self, root: Path, *, writable: bool = False, max_chars: int = 32_000) -> None:
        self.boundary = WorkspaceBoundary(root)
        self.writable = writable
        self.max_chars = max_chars

    def read_file(self, path: str) -> str:
        target = self.boundary.resolve(path, must_exist=True)
        if not target.is_file():
            raise IsADirectoryError(target)
        return target.read_text(encoding="utf-8", errors="replace")[: self.max_chars]

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

    def write_file(self, path: str, content: str) -> None:
        if not self.writable:
            raise PolicyViolation("worker does not have write permission")
        target = self.boundary.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def apply_patch(self, files: dict[str, str]) -> PatchApplicationResult:
        """Apply a bounded set of full-file replacements through the writer boundary."""
        if not self.writable:
            raise PolicyViolation("worker does not have write permission")
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
