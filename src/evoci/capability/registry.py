"""Single-package skill storage with mutable SQLite metadata."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

from evoci.capability.models import (
    GeneratedFile,
    RegisteredSkill,
    SkillCandidate,
    SkillFile,
    SkillManifest,
    SkillMemoryEntry,
    SkillStats,
)
from evoci.capability.skill_memory import (
    format_skill_memory_block,
    memory_marker,
    parse_skill_memory_entries,
)
from evoci.learning_state import LegacyLearningStateError
from evoci.tools.policy import PolicyViolation, WorkspaceBoundary

_MEMORY_LOCKS: dict[str, Lock] = {}
_MEMORY_LOCKS_GUARD = Lock()


def _memory_thread_lock(path: Path) -> Lock:
    key = str(path.resolve())
    with _MEMORY_LOCKS_GUARD:
        return _MEMORY_LOCKS.setdefault(key, Lock())


class CapabilityRegistryError(RuntimeError):
    pass


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise CapabilityRegistryError("skill name cannot produce an empty ID")
    return slug[:80]


def _safe_relative(path: str, expected_prefix: str | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise PolicyViolation(f"unsafe skill path: {path}")
    if expected_prefix and candidate.parts[0] != expected_prefix:
        raise PolicyViolation(f"{path} must be under {expected_prefix}/")
    return candidate


def _writable_rmtree(path: Path) -> None:
    if not path.exists():
        return
    for item in path.rglob("*"):
        if item.is_file() or item.is_symlink():
            with contextlib.suppress(OSError):
                item.chmod(item.stat().st_mode | 0o200)
    shutil.rmtree(path, ignore_errors=True)


def _make_package_readonly(package: Path) -> None:
    for item in package.rglob("*"):
        if item.is_file():
            item.chmod(item.stat().st_mode & ~0o222)


def _fts_content(candidate: SkillCandidate) -> str:
    return " ".join(
        [
            candidate.name,
            candidate.description,
            *candidate.triggers,
            *candidate.task_families,
        ]
    )


class CapabilityRegistry:
    def __init__(
        self,
        root: Path,
        database_path: Path,
        *,
        validation_timeout: float = 10.0,
    ) -> None:
        self.root = root.resolve()
        self._validation_timeout = validation_timeout
        self.root.mkdir(parents=True, exist_ok=True)
        self._boundary = WorkspaceBoundary(self.root)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._assert_fresh_schema()
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL,
                manifest_json TEXT NOT NULL,
                package_path TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(
                skill_id UNINDEXED,
                content
            );
            CREATE TABLE IF NOT EXISTS skill_stats (
                skill_id TEXT PRIMARY KEY,
                retrieval_count INTEGER NOT NULL DEFAULT 0,
                selected_count INTEGER NOT NULL DEFAULT 0,
                use_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                patch_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT
            );
            CREATE TABLE IF NOT EXISTS applied_operations (
                operation_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def _assert_fresh_schema(self) -> None:
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "skills" not in tables:
            return
        columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(skills)").fetchall()
        }
        if "version" in columns or "status" in columns:
            raise LegacyLearningStateError()

    def create_skill(
        self,
        candidate: SkillCandidate,
        *,
        skill_id: str | None = None,
        operation_key: str | None = None,
    ) -> RegisteredSkill:
        if operation_key:
            prior = self.operation_result(operation_key)
            if prior is not None:
                prior_skill_id = prior.get("skill_id")
                if not isinstance(prior_skill_id, str):
                    raise CapabilityRegistryError(
                        f"operation {operation_key} has an invalid stored result"
                    )
                existing = self.get(prior_skill_id)
                if existing is None:
                    raise CapabilityRegistryError(
                        f"operation {operation_key} references a missing skill"
                    )
                return existing
        resolved_id = _slug(skill_id or candidate.name)
        if self.get(resolved_id) is not None:
            raise CapabilityRegistryError(f"skill already exists: {resolved_id}")
        staging, manifest = self._write_staging(candidate, resolved_id)
        try:
            from evoci.capability.validator import CandidateValidator

            validation = CandidateValidator(timeout=self._validation_timeout).validate_package(
                staging, manifest
            )
            if not validation.passed:
                raise CapabilityRegistryError(
                    "skill validation failed: " + "; ".join(validation.errors)
                )
            return self._install_new(
                resolved_id, staging, manifest, candidate, operation_key=operation_key
            )
        except Exception:
            _writable_rmtree(staging)
            raise

    def update_skill(
        self,
        skill_id: str,
        candidate: SkillCandidate,
        *,
        operation_key: str | None = None,
    ) -> RegisteredSkill:
        if operation_key:
            prior = self.operation_result(operation_key)
            if prior is not None:
                prior_skill_id = prior.get("skill_id")
                if not isinstance(prior_skill_id, str):
                    raise CapabilityRegistryError(
                        f"operation {operation_key} has an invalid stored result"
                    )
                existing = self.get(prior_skill_id)
                if existing is None:
                    raise CapabilityRegistryError(
                        f"operation {operation_key} references a missing skill"
                    )
                return existing
        current = self.get(skill_id)
        if current is None:
            raise CapabilityRegistryError(f"unknown skill: {skill_id}")
        staging, manifest = self._write_staging(
            candidate,
            skill_id,
            created_at=current.manifest.created_at,
            source_run_ids=list(
                dict.fromkeys([*current.manifest.source_run_ids, *candidate.source_run_ids])
            ),
            enabled=current.manifest.enabled,
        )
        try:
            from evoci.capability.validator import CandidateValidator

            validation = CandidateValidator(timeout=self._validation_timeout).validate_package(
                staging, manifest
            )
            if not validation.passed:
                raise CapabilityRegistryError(
                    "skill validation failed: " + "; ".join(validation.errors)
                )
            return self._install_update(
                skill_id, staging, manifest, candidate, operation_key=operation_key
            )
        except Exception:
            _writable_rmtree(staging)
            raise

    def _write_staging(
        self,
        candidate: SkillCandidate,
        skill_id: str,
        *,
        created_at: datetime | None = None,
        source_run_ids: list[str] | None = None,
        enabled: bool = True,
    ) -> tuple[Path, SkillManifest]:
        temporary = self._boundary.resolve(f".staging-{skill_id}-{uuid4().hex}")
        temporary.mkdir(parents=True)
        files_to_write: list[GeneratedFile] = [
            GeneratedFile(path="SKILL.md", content=candidate.skill_md),
            *self._categorized(candidate.scripts, "scripts"),
            *self._categorized(candidate.references, "references"),
            *self._categorized(candidate.templates, "templates"),
            *self._categorized(candidate.tests, "tests"),
        ]
        try:
            skill_files: list[SkillFile] = []
            for generated in files_to_write:
                relative = _safe_relative(generated.path)
                destination = (temporary / relative).resolve()
                if temporary not in destination.parents:
                    raise PolicyViolation(f"skill path escapes package: {generated.path}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(generated.content, encoding="utf-8")
                if generated.executable:
                    destination.chmod(destination.stat().st_mode | 0o100)
                skill_files.append(
                    SkillFile(
                        path=generated.path,
                        sha256=hashlib.sha256(generated.content.encode()).hexdigest(),
                        executable=generated.executable,
                    )
                )
            now = datetime.now(UTC)
            manifest = SkillManifest(
                skill_id=skill_id,
                name=candidate.name,
                description=candidate.description,
                triggers=candidate.triggers,
                task_families=candidate.task_families,
                permissions=candidate.permissions,
                source_run_ids=source_run_ids or list(candidate.source_run_ids),
                enabled=enabled,
                files=skill_files,
                created_at=created_at or now,
                updated_at=now,
            )
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            return temporary, manifest
        except Exception:
            _writable_rmtree(temporary)
            raise

    def _install_new(
        self,
        skill_id: str,
        staging: Path,
        manifest: SkillManifest,
        candidate: SkillCandidate,
        *,
        operation_key: str | None,
    ) -> RegisteredSkill:
        skill_root = self._boundary.resolve(skill_id)
        package = skill_root / "package"
        if package.exists():
            raise CapabilityRegistryError(f"skill package already exists: {package}")
        skill_root_existed = skill_root.exists()
        skill_root.mkdir(parents=True, exist_ok=True)
        memory_path = skill_root / "memory.md"
        memory_existed = memory_path.exists()
        if not memory_existed:
            memory_path.write_text("", encoding="utf-8")
        _make_package_readonly(staging)
        try:
            staging.rename(package)
            self._upsert_rows(manifest, package, candidate, operation_key=operation_key)
        except Exception:
            _writable_rmtree(package)
            if not memory_existed:
                memory_path.unlink(missing_ok=True)
            if not skill_root_existed:
                _writable_rmtree(skill_root)
            _writable_rmtree(staging)
            raise
        return RegisteredSkill(manifest=manifest, package_path=str(package))

    def _install_update(
        self,
        skill_id: str,
        staging: Path,
        manifest: SkillManifest,
        candidate: SkillCandidate,
        *,
        operation_key: str | None,
    ) -> RegisteredSkill:
        skill_root = self._boundary.resolve(skill_id)
        current = skill_root / "package"
        previous = skill_root / "previous"
        old_previous = skill_root / f".old-previous-{uuid4().hex}"
        if not current.is_dir():
            _writable_rmtree(staging)
            raise CapabilityRegistryError(f"current package missing for {skill_id}")
        _make_package_readonly(staging)
        moved_current = False
        try:
            if previous.exists():
                previous.rename(old_previous)
            current.rename(previous)
            moved_current = True
            staging.rename(current)
            self._upsert_rows(manifest, current, candidate, operation_key=operation_key)
        except Exception:
            if moved_current and current.exists():
                _writable_rmtree(current)
            if not current.exists() and previous.exists():
                previous.rename(current)
            if old_previous.exists() and not previous.exists():
                old_previous.rename(previous)
            _writable_rmtree(staging)
            raise
        _writable_rmtree(old_previous)
        return RegisteredSkill(manifest=manifest, package_path=str(current))

    def _upsert_rows(
        self,
        manifest: SkillManifest,
        package: Path,
        candidate: SkillCandidate,
        *,
        operation_key: str | None,
    ) -> None:
        manifest_json = manifest.model_dump_json()
        now = datetime.now(UTC).isoformat()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                """
                INSERT INTO skills(skill_id, enabled, manifest_json, package_path)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(skill_id) DO UPDATE SET
                    enabled=excluded.enabled,
                    manifest_json=excluded.manifest_json,
                    package_path=excluded.package_path
                """,
                (manifest.skill_id, int(manifest.enabled), manifest_json, str(package)),
            )
            self._connection.execute(
                "DELETE FROM skill_fts WHERE skill_id = ?", (manifest.skill_id,)
            )
            self._connection.execute(
                "INSERT INTO skill_fts(skill_id, content) VALUES (?, ?)",
                (manifest.skill_id, _fts_content(candidate)),
            )
            self._connection.execute(
                """
                INSERT INTO skill_stats(skill_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(skill_id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (manifest.skill_id, now, now),
            )
            if operation_key:
                self._connection.execute(
                    "INSERT OR IGNORE INTO applied_operations VALUES (?, ?, ?)",
                    (
                        operation_key,
                        json.dumps({"skill_id": manifest.skill_id}),
                        now,
                    ),
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    @staticmethod
    def _categorized(files: Iterable[GeneratedFile], category: str) -> list[GeneratedFile]:
        result: list[GeneratedFile] = []
        for generated in files:
            _safe_relative(generated.path, category)
            result.append(generated)
        return result

    def get(self, skill_id: str) -> RegisteredSkill | None:
        row = self._connection.execute(
            "SELECT * FROM skills WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        if row is None:
            return None
        manifest = SkillManifest.model_validate_json(str(row["manifest_json"]))
        manifest = manifest.model_copy(update={"enabled": bool(row["enabled"])})
        return RegisteredSkill(manifest=manifest, package_path=str(row["package_path"]))

    def list(self, *, enabled_only: bool = True) -> list[RegisteredSkill]:
        query = "SELECT skill_id FROM skills"
        params: tuple[object, ...] = ()
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY skill_id"
        rows = self._connection.execute(query, params).fetchall()
        records = [self.get(str(row["skill_id"])) for row in rows]
        return [record for record in records if record is not None]

    def enable(self, skill_id: str) -> None:
        self._set_enabled(skill_id, True)

    def disable(self, skill_id: str) -> None:
        self._set_enabled(skill_id, False)

    def _set_enabled(self, skill_id: str, enabled: bool) -> None:
        record = self.get(skill_id)
        if record is None:
            raise KeyError(f"unknown skill: {skill_id}")
        manifest = record.manifest.model_copy(update={"enabled": enabled})
        self._connection.execute(
            "UPDATE skills SET enabled = ?, manifest_json = ? WHERE skill_id = ?",
            (int(enabled), manifest.model_dump_json(), skill_id),
        )
        self._connection.commit()

    def stats(self, skill_id: str) -> SkillStats:
        row = self._connection.execute(
            "SELECT * FROM skill_stats WHERE skill_id = ?", (skill_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown skill stats: {skill_id}")
        return SkillStats.model_validate(dict(row))

    def skill_root(self, skill_id: str) -> Path:
        record = self.get(skill_id)
        if record is None:
            raise KeyError(f"unknown skill: {skill_id}")
        return Path(record.package_path).resolve().parent

    def operation_result(self, operation_key: str) -> dict[str, object] | None:
        row = self._connection.execute(
            "SELECT result_json FROM applied_operations WHERE operation_key = ?",
            (operation_key,),
        ).fetchone()
        return json.loads(str(row["result_json"])) if row is not None else None

    def record_operation(self, operation_key: str, result: dict[str, object]) -> bool:
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO applied_operations VALUES (?, ?, ?)",
            (
                operation_key,
                json.dumps(result, sort_keys=True),
                datetime.now(UTC).isoformat(),
            ),
        )
        self._connection.commit()
        return cursor.rowcount == 1

    def record_retrieval(
        self,
        skill_ids: Sequence[str],
        *,
        selected: bool = False,
        operation_key: str | None = None,
    ) -> bool:
        column = "selected_count" if selected else "retrieval_count"
        now = datetime.now(UTC).isoformat()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            if operation_key and self.operation_result(operation_key) is not None:
                self._connection.rollback()
                return False
            self._connection.executemany(
                f"UPDATE skill_stats SET {column} = {column} + 1, updated_at = ? "
                "WHERE skill_id = ?",
                [(now, skill_id) for skill_id in skill_ids],
            )
            if operation_key:
                self._connection.execute(
                    "INSERT INTO applied_operations VALUES (?, ?, ?)",
                    (
                        operation_key,
                        json.dumps({"count": len(skill_ids), "selected": selected}),
                        now,
                    ),
                )
            self._connection.commit()
            return True
        except Exception as exc:
            self._connection.rollback()
            raise RuntimeError(
                f"Failed to record retrieval for {len(skill_ids)} skills: {exc}"
            ) from exc

    def record_use(
        self,
        skill_id: str,
        *,
        success: bool,
        patched: bool,
        operation_key: str | None = None,
    ) -> bool:
        now = datetime.now(UTC).isoformat()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            if operation_key and self.operation_result(operation_key) is not None:
                self._connection.rollback()
                return False
            self._connection.execute(
                """
            UPDATE skill_stats SET
                use_count = use_count + 1,
                success_count = success_count + ?,
                failure_count = failure_count + ?,
                patch_count = patch_count + ?,
                last_used_at = ?,
                updated_at = ?
            WHERE skill_id = ?
            """,
                (
                    int(success),
                    int(not success),
                    int(patched),
                    now,
                    now,
                    skill_id,
                ),
            )
            if operation_key:
                self._connection.execute(
                    "INSERT INTO applied_operations VALUES (?, ?, ?)",
                    (
                        operation_key,
                        json.dumps({"recorded": True}),
                        now,
                    ),
                )
            self._connection.commit()
            return True
        except Exception:
            self._connection.rollback()
            raise

    def append_skill_memory(self, skill_id: str, entry: SkillMemoryEntry) -> bool:
        skill_root = self.root / skill_id
        if not (skill_root / "package").is_dir():
            return False
        memory_path = skill_root / "memory.md"
        lock_path = skill_root / ".memory.lock"
        skill_root.mkdir(parents=True, exist_ok=True)
        with _memory_thread_lock(memory_path), lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                with memory_path.open("a+", encoding="utf-8") as handle:
                    handle.seek(0)
                    existing = handle.read()
                    marker = memory_marker(entry.run_id, skill_id)
                    if marker in existing:
                        return False
                    block = format_skill_memory_block(skill_id, entry)
                    prefix = ""
                    if existing and not existing.endswith("\n"):
                        prefix = "\n"
                    if existing.strip():
                        prefix += "\n"
                    handle.seek(0, os.SEEK_END)
                    handle.write(prefix + block)
                    handle.flush()
                    os.fsync(handle.fileno())
                return True
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def read_skill_memory(
        self,
        skill_id: str,
        *,
        limit: int = 8,
        max_chars: int = 4_000,
    ) -> Sequence[SkillMemoryEntry]:
        try:
            skill_root = self.skill_root(skill_id)
        except KeyError:
            return []
        memory_path = skill_root / "memory.md"
        if not memory_path.is_file():
            return []
        entries = parse_skill_memory_entries(memory_path.read_text(encoding="utf-8"))
        recent = entries[-limit:] if limit >= 0 else entries
        selected: list[SkillMemoryEntry] = []
        used = 0
        for entry in reversed(recent):
            size = len(entry.lesson) + len(entry.task_summary) + 32
            if selected and used + size > max_chars:
                break
            selected.append(entry)
            used += size
        selected.reverse()
        return selected

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        self._connection.close()
