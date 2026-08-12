"""Immutable on-disk packages with mutable SQLite lifecycle metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from evoci.capability.models import (
    GeneratedFile,
    RegisteredSkill,
    SkillCandidate,
    SkillFile,
    SkillManifest,
    SkillStats,
    SkillStatus,
    SkillVersionRef,
)
from evoci.tools.policy import PolicyViolation, WorkspaceBoundary


class CapabilityRegistryError(RuntimeError):
    pass


ALLOWED_TRANSITIONS: dict[SkillStatus, set[SkillStatus]] = {
    "candidate": {"trial", "rejected"},
    "trial": {"active", "rejected", "stale"},
    "active": {"stale", "superseded"},
    "stale": {"active", "archived", "superseded"},
    "archived": set(),
    "rejected": set(),
    "superseded": set(),
}


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


class CapabilityRegistry:
    def __init__(self, root: Path, database_path: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._boundary = WorkspaceBoundary(self.root)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                package_path TEXT NOT NULL,
                PRIMARY KEY(skill_id, version)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS skill_fts USING fts5(
                skill_key UNINDEXED, content
            );
            CREATE TABLE IF NOT EXISTS skill_stats (
                skill_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                retrieval_count INTEGER NOT NULL DEFAULT 0,
                selected_count INTEGER NOT NULL DEFAULT 0,
                use_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                patch_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                last_modified_at TEXT,
                avg_tool_calls_when_used REAL,
                avg_attempts_when_used REAL,
                utility_score REAL,
                PRIMARY KEY(skill_id, version)
            );
            CREATE TABLE IF NOT EXISTS applied_operations (
                operation_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    def create_candidate(
        self,
        candidate: SkillCandidate,
        *,
        skill_id: str | None = None,
        parent_version: int | None = None,
        supersedes: list[SkillVersionRef] | None = None,
        operation_key: str | None = None,
    ) -> RegisteredSkill:
        if operation_key:
            prior = self.operation_result(operation_key)
            if prior is not None:
                prior_skill_id = prior.get("skill_id")
                prior_version = prior.get("version")
                if not isinstance(prior_skill_id, str) or not isinstance(prior_version, int):
                    raise CapabilityRegistryError(
                        f"operation {operation_key} has an invalid stored result"
                    )
                existing = self.get(prior_skill_id, prior_version)
                if existing is None:
                    raise CapabilityRegistryError(
                        f"operation {operation_key} references a missing skill"
                    )
                return existing
        resolved_id = _slug(skill_id or candidate.name)
        latest = self.get(resolved_id)
        if latest is not None and parent_version is None:
            raise CapabilityRegistryError(
                f"new skill slug already exists: {resolved_id}; use update_skill with lineage"
            )
        if parent_version is not None and self.get(resolved_id, parent_version) is None:
            raise CapabilityRegistryError(
                f"update parent does not exist: {resolved_id} v{parent_version}"
            )
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS next FROM skills WHERE skill_id = ?",
            (resolved_id,),
        ).fetchone()
        version = int(row["next"])
        package = self._boundary.resolve(f"{resolved_id}/v{version}")
        if package.exists():
            manifest_path = package / "manifest.json"
            if operation_key and manifest_path.is_file():
                recovered = SkillManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                if recovered.operation_key == operation_key:
                    self._insert_candidate_rows(
                        recovered, package, candidate, operation_key=operation_key
                    )
                    record = self.get(recovered.skill_id, recovered.version)
                    assert record is not None
                    return record
            raise CapabilityRegistryError(f"immutable package already exists: {package}")
        temporary = self._boundary.resolve(f".staging-{resolved_id}-{uuid4().hex}")
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
            manifest = SkillManifest(
                skill_id=resolved_id,
                version=version,
                name=candidate.name,
                description=candidate.description,
                status="candidate",
                triggers=candidate.triggers,
                task_families=candidate.task_families,
                permissions=candidate.permissions,
                source_run_ids=candidate.source_run_ids,
                parent_version=parent_version,
                supersedes=supersedes or [],
                files=skill_files,
                verification_commands=candidate.verification_commands,
                operation_key=operation_key,
            )
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            os.chmod(manifest_path, 0o444)
            for item in temporary.rglob("*"):
                if item.is_file():
                    item.chmod(item.stat().st_mode & ~0o222)
            package.parent.mkdir(parents=True, exist_ok=True)
            temporary.rename(package)
        except Exception:
            if temporary.exists():
                for item in temporary.rglob("*"):
                    if item.is_file():
                        item.chmod(item.stat().st_mode | 0o200)
                import shutil

                shutil.rmtree(temporary)
            raise
        self._insert_candidate_rows(manifest, package, candidate, operation_key=operation_key)
        return RegisteredSkill(manifest=manifest, package_path=str(package))

    def _insert_candidate_rows(
        self,
        manifest: SkillManifest,
        package: Path,
        candidate: SkillCandidate,
        *,
        operation_key: str | None,
    ) -> None:
        manifest_json = manifest.model_dump_json()
        resolved_id = manifest.skill_id
        version = manifest.version
        key = f"{resolved_id}:{version}"
        now = datetime.now(UTC).isoformat()
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                "INSERT OR IGNORE INTO skills VALUES (?, ?, 'candidate', ?, ?)",
                (resolved_id, version, manifest_json, str(package)),
            )
            fts_exists = self._connection.execute(
                "SELECT 1 FROM skill_fts WHERE skill_key = ?", (key,)
            ).fetchone()
            if fts_exists is None:
                self._connection.execute(
                    "INSERT INTO skill_fts(skill_key, content) VALUES (?, ?)",
                    (
                        key,
                        " ".join(
                            [
                                candidate.name,
                                candidate.description,
                                *candidate.triggers,
                                *candidate.task_families,
                            ]
                        ),
                    ),
                )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO skill_stats(skill_id, version, created_at)
                VALUES (?, ?, ?)
                """,
                (resolved_id, version, now),
            )
            if operation_key:
                self._connection.execute(
                    "INSERT OR IGNORE INTO applied_operations VALUES (?, ?, ?)",
                    (
                        operation_key,
                        json.dumps({"skill_id": resolved_id, "version": version}),
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

    def get(self, skill_id: str, version: int | None = None) -> RegisteredSkill | None:
        if version is None:
            row = self._connection.execute(
                "SELECT * FROM skills WHERE skill_id = ? ORDER BY version DESC LIMIT 1",
                (skill_id,),
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT * FROM skills WHERE skill_id = ? AND version = ?",
                (skill_id, version),
            ).fetchone()
        if row is None:
            return None
        manifest = SkillManifest.model_validate_json(str(row["manifest_json"]))
        manifest = manifest.model_copy(update={"status": str(row["status"])})
        return RegisteredSkill(manifest=manifest, package_path=str(row["package_path"]))

    def list(self, statuses: set[SkillStatus] | None = None) -> list[RegisteredSkill]:
        rows = self._connection.execute(
            "SELECT skill_id, version FROM skills ORDER BY skill_id, version"
        ).fetchall()
        records = [
            record
            for row in rows
            if (record := self.get(str(row["skill_id"]), int(row["version"]))) is not None
        ]
        return [
            record for record in records if statuses is None or record.manifest.status in statuses
        ]

    def transition(self, skill_id: str, version: int, target: SkillStatus) -> None:
        record = self.get(skill_id, version)
        if record is None:
            raise KeyError(f"unknown skill: {skill_id} v{version}")
        current = record.manifest.status
        if target not in ALLOWED_TRANSITIONS[current]:
            raise CapabilityRegistryError(f"invalid lifecycle transition: {current} -> {target}")
        self._connection.execute(
            "UPDATE skills SET status = ? WHERE skill_id = ? AND version = ?",
            (target, skill_id, version),
        )
        self._connection.execute(
            "UPDATE skill_stats SET last_modified_at = ? WHERE skill_id = ? AND version = ?",
            (datetime.now(UTC).isoformat(), skill_id, version),
        )
        self._connection.commit()

    def stats(self, skill_id: str, version: int) -> SkillStats:
        row = self._connection.execute(
            "SELECT * FROM skill_stats WHERE skill_id = ? AND version = ?", (skill_id, version)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown skill stats: {skill_id} v{version}")
        return SkillStats.model_validate(dict(row))

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
        refs: Sequence[SkillVersionRef],
        *,
        selected: bool = False,
        operation_key: str | None = None,
    ) -> bool:
        column = "selected_count" if selected else "retrieval_count"
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            if operation_key and self.operation_result(operation_key) is not None:
                self._connection.rollback()
                return False
            self._connection.executemany(
                f"UPDATE skill_stats SET {column} = {column} + 1 "
                "WHERE skill_id = ? AND version = ?",
                [(ref.skill_id, ref.version) for ref in refs],
            )
            if operation_key:
                self._connection.execute(
                    "INSERT INTO applied_operations VALUES (?, ?, ?)",
                    (
                        operation_key,
                        json.dumps({"count": len(refs), "selected": selected}),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            self._connection.commit()
            return True
        except Exception:
            self._connection.rollback()
            raise

    def record_use(
        self,
        ref: SkillVersionRef,
        *,
        success: bool,
        tool_calls: int,
        attempts: int,
        patched: bool,
        operation_key: str | None = None,
    ) -> bool:
        if operation_key and self.operation_result(operation_key) is not None:
            return False
        stats = self.stats(ref.skill_id, ref.version)
        uses = stats.use_count + 1

        def average(previous: float | None, value: int) -> float:
            return ((previous or 0.0) * stats.use_count + value) / uses

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
                avg_tool_calls_when_used = ?,
                avg_attempts_when_used = ?
            WHERE skill_id = ? AND version = ?
            """,
                (
                    int(success),
                    int(not success),
                    int(patched),
                    datetime.now(UTC).isoformat(),
                    average(stats.avg_tool_calls_when_used, tool_calls),
                    average(stats.avg_attempts_when_used, attempts),
                    ref.skill_id,
                    ref.version,
                ),
            )
            if operation_key:
                self._connection.execute(
                    "INSERT INTO applied_operations VALUES (?, ?, ?)",
                    (
                        operation_key,
                        json.dumps({"recorded": True}),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            self._connection.commit()
            return True
        except Exception:
            self._connection.rollback()
            raise

    def set_utility(self, ref: SkillVersionRef, score: float) -> None:
        self._connection.execute(
            "UPDATE skill_stats SET utility_score = ? WHERE skill_id = ? AND version = ?",
            (score, ref.skill_id, ref.version),
        )
        self._connection.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        self._connection.close()
