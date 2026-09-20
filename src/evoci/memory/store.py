"""SQLite + FTS5 implementation of long-term memory."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from evoci.domain.models import MemoryHit
from evoci.memory.models import Episode, SemanticMemory


class MemoryStore(Protocol):
    def add_episode(self, episode: Episode) -> None: ...

    def get_episode(self, run_id: str) -> Episode | None: ...

    def add_semantic(self, memory: SemanticMemory, *, operation_key: str | None = None) -> bool: ...

    def operation_result(self, operation_key: str) -> dict[str, object] | None: ...

    def record_operation(self, operation_key: str, result: dict[str, object]) -> bool: ...

    def search_semantic(
        self, query: str, *, namespaces: Sequence[str], limit: int
    ) -> list[MemoryHit]: ...

    def search_episodes(
        self, query: str, *, repo: str, limit: int
    ) -> list[MemoryHit]: ...

    def search_episodes_by_fingerprint(
        self, *, repo: str, fingerprint: str, limit: int = 2
    ) -> list[MemoryHit]: ...

    def archive(self, memory_id: str) -> None: ...


def _fts_query(text: str) -> str:
    ascii_tokens = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())
    unicode_tokens = re.findall(r"[\u0080-\uffff]{2,}", text)
    tokens = list(dict.fromkeys([*ascii_tokens, *unicode_tokens]))[:24]
    if not tokens:
        tokens = re.findall(r"[A-Za-z0-9\u0080-\uffff]", text.lower())[:24]
    return " OR ".join(f'"{token}"' for token in tokens)


def _join(items: list[str] | None) -> str:
    return "; ".join(str(item) for item in items or [] if item)


def format_episode_content(
    *,
    success: bool,
    failure_summary: str,
    root_cause: str | None = None,
    successful_fix: str | None = None,
    failure_reason: str | None = None,
    hypotheses: list[str] | None = None,
    verification_failures: list[str] | None = None,
    failure_class: str | None = None,
    failure_stage: str | None = None,
    attempted_fixes: list[str] | None = None,
    attempted_files: list[str] | None = None,
    external_failure_details: list[str] | None = None,
) -> str:
    """Keep outcome and counterevidence visible; mark unverified causes as hypotheses."""

    parts = [
        f"OUTCOME={'success' if success else 'failed'}",
        failure_summary,
    ]
    if success:
        if root_cause:
            parts.append(f"ROOT_CAUSE={root_cause}")
        if successful_fix:
            parts.append(f"SUCCESSFUL_FIX={successful_fix}")
        return "\n".join(part for part in parts if part)

    parts.insert(1, f"FAILURE_CLASS={failure_class or 'repair'}")
    if failure_stage:
        parts.append(f"FAILURE_STAGE={failure_stage}")
    if failure_reason:
        parts.append(f"FAILURE_REASON={failure_reason}")
    if attempted_fixes:
        marked = [
            item
            if "failed" in item.lower() or "unconfirmed" in item.lower()
            else f"{item} (failed/unconfirmed)"
            for item in attempted_fixes
        ]
        parts.append(f"ATTEMPTED_FIXES={_join(marked)}")
    if attempted_files:
        parts.append(f"ATTEMPTED_FILES={_join(attempted_files)}")
    if verification_failures:
        parts.append(f"VERIFICATION_FAILURES={_join(verification_failures)}")
    if external_failure_details:
        parts.append(f"EXTERNAL_FAILURE_DETAILS={_join(external_failure_details)}")
    if root_cause:
        parts.append(f"HYPOTHESIS (unverified): {root_cause}")
    if hypotheses:
        parts.append(f"HYPOTHESES_ATTEMPTED (unverified): {_join(hypotheses)}")
    return "\n".join(part for part in parts if part)


_JSON_LIST_FIELDS = (
    "important_evidence",
    "tools_used",
    "hypotheses_attempted",
    "verification_failures",
    "attempted_fix_summaries",
    "attempted_files",
    "external_failure_details",
)


def _episode_from_row(row: sqlite3.Row) -> Episode:
    payload = dict(zip(row.keys(), tuple(row), strict=True))
    payload.pop("rank", None)
    for field in _JSON_LIST_FIELDS:
        raw = payload.get(field)
        payload[field] = json.loads(str(raw or "[]")) if not isinstance(raw, list) else raw
    payload["success"] = bool(payload.get("success"))
    return Episode.model_validate(payload)


def _episode_content(episode: Episode) -> str:
    return format_episode_content(
        success=episode.success,
        failure_summary=episode.failure_summary,
        root_cause=episode.root_cause,
        successful_fix=episode.successful_fix_summary,
        failure_reason=episode.failure_reason,
        hypotheses=list(episode.hypotheses_attempted),
        verification_failures=list(episode.verification_failures),
        failure_class=episode.failure_class,
        failure_stage=episode.failure_stage,
        attempted_fixes=list(episode.attempted_fix_summaries),
        attempted_files=list(episode.attempted_files),
        external_failure_details=list(episode.external_failure_details),
    )


class SQLiteMemoryStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL UNIQUE,
                repo TEXT NOT NULL,
                task_family TEXT NOT NULL,
                failure_summary TEXT NOT NULL,
                root_cause TEXT,
                important_evidence TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                successful_fix_summary TEXT,
                tools_used TEXT NOT NULL,
                hypotheses_attempted TEXT NOT NULL DEFAULT '[]',
                verification_failures TEXT NOT NULL DEFAULT '[]',
                failure_reason TEXT,
                success INTEGER NOT NULL,
                failure_fingerprint TEXT NOT NULL DEFAULT '',
                repo_revision TEXT,
                failure_class TEXT NOT NULL DEFAULT 'repair',
                failure_stage TEXT,
                attempted_fix_summaries TEXT NOT NULL DEFAULT '[]',
                attempted_files TEXT NOT NULL DEFAULT '[]',
                external_failure_details TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                episode_id UNINDEXED, content
            );
            CREATE TABLE IF NOT EXISTS semantic_memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL,
                confidence REAL NOT NULL,
                source_run_ids TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memory_namespace
                ON semantic_memories(namespace, archived);
            CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
                memory_id UNINDEXED, content
            );
            CREATE TABLE IF NOT EXISTS applied_operations (
                operation_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        episode_columns = {
            str(row["name"])
            for row in self._connection.execute("PRAGMA table_info(episodes)").fetchall()
        }
        for name, declaration in (
            ("hypotheses_attempted", "TEXT NOT NULL DEFAULT '[]'"),
            ("verification_failures", "TEXT NOT NULL DEFAULT '[]'"),
            ("failure_reason", "TEXT"),
            ("failure_fingerprint", "TEXT NOT NULL DEFAULT ''"),
            ("repo_revision", "TEXT"),
            ("failure_class", "TEXT NOT NULL DEFAULT 'repair'"),
            ("failure_stage", "TEXT"),
            ("attempted_fix_summaries", "TEXT NOT NULL DEFAULT '[]'"),
            ("attempted_files", "TEXT NOT NULL DEFAULT '[]'"),
            ("external_failure_details", "TEXT NOT NULL DEFAULT '[]'"),
        ):
            if name not in episode_columns:
                self._connection.execute(f"ALTER TABLE episodes ADD COLUMN {name} {declaration}")
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_episode_fingerprint
            ON episodes(failure_fingerprint, created_at)
            """
        )
        self._connection.commit()

    def add_episode(self, episode: Episode) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO episodes (
                id, run_id, repo, task_family, failure_summary, root_cause,
                important_evidence, attempts, successful_fix_summary, tools_used,
                hypotheses_attempted, verification_failures, failure_reason, success,
                failure_fingerprint, repo_revision, failure_class, failure_stage,
                attempted_fix_summaries, attempted_files, external_failure_details, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.id,
                episode.run_id,
                episode.repo,
                episode.task_family,
                episode.failure_summary,
                episode.root_cause,
                json.dumps(episode.important_evidence),
                episode.attempts,
                episode.successful_fix_summary,
                json.dumps(episode.tools_used),
                json.dumps(episode.hypotheses_attempted),
                json.dumps(episode.verification_failures),
                episode.failure_reason,
                int(episode.success),
                episode.failure_fingerprint,
                episode.repo_revision,
                episode.failure_class,
                episode.failure_stage,
                json.dumps(episode.attempted_fix_summaries),
                json.dumps(episode.attempted_files),
                json.dumps(episode.external_failure_details),
                episode.created_at.isoformat(),
            ),
        )
        if self._connection.execute("SELECT changes()").fetchone()[0]:
            self._connection.execute(
                "INSERT INTO episodes_fts(episode_id, content) VALUES (?, ?)",
                (episode.id, _episode_content(episode)),
            )
        self._connection.commit()

    def get_episode(self, run_id: str) -> Episode | None:
        row = self._connection.execute(
            "SELECT * FROM episodes WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return _episode_from_row(row)

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

    def add_semantic(self, memory: SemanticMemory, *, operation_key: str | None = None) -> bool:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            if operation_key and self.operation_result(operation_key) is not None:
                self._connection.rollback()
                return False
            self._connection.execute(
                """
            INSERT INTO semantic_memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content,
                importance=excluded.importance,
                confidence=excluded.confidence,
                source_run_ids=excluded.source_run_ids,
                updated_at=excluded.updated_at,
                archived=0
            """,
                (
                    memory.id,
                    memory.namespace,
                    memory.content,
                    memory.importance,
                    memory.confidence,
                    json.dumps(memory.source_run_ids),
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
                ),
            )
            self._connection.execute("DELETE FROM semantic_fts WHERE memory_id = ?", (memory.id,))
            self._connection.execute(
                "INSERT INTO semantic_fts(memory_id, content) VALUES (?, ?)",
                (memory.id, memory.content),
            )
            if operation_key:
                self._connection.execute(
                    "INSERT INTO applied_operations VALUES (?, ?, ?)",
                    (
                        operation_key,
                        json.dumps({"memory_id": memory.id}),
                        datetime.now(UTC).isoformat(),
                    ),
                )
            self._connection.commit()
            return True
        except Exception as exc:
            self._connection.rollback()
            raise RuntimeError(f"Failed to add semantic memory {memory.id}: {exc}") from exc

    def search_semantic(
        self, query: str, *, namespaces: Sequence[str], limit: int
    ) -> list[MemoryHit]:
        expression = _fts_query(query)
        if not expression or not namespaces or limit <= 0:
            return []
        placeholders = ",".join("?" for _ in namespaces)
        rows = self._connection.execute(
            f"""
            SELECT m.id, m.namespace, m.content, bm25(semantic_fts) AS rank
            FROM semantic_fts
            JOIN semantic_memories m ON m.id = semantic_fts.memory_id
            WHERE semantic_fts MATCH ?
              AND m.namespace IN ({placeholders})
              AND m.archived = 0
            ORDER BY rank, m.importance DESC, m.confidence DESC
            LIMIT ?
            """,
            (expression, *namespaces, limit),
        ).fetchall()
        now = datetime.now(UTC).isoformat()
        ids = [str(row["id"]) for row in rows]
        self._connection.executemany(
            "UPDATE semantic_memories SET last_accessed_at = ? WHERE id = ?",
            [(now, memory_id) for memory_id in ids],
        )
        self._connection.commit()
        return [
            MemoryHit(
                memory_id=str(row["id"]),
                namespace=str(row["namespace"]),
                content=str(row["content"]),
                score=1.0 / (1.0 + abs(float(row["rank"]))),
            )
            for row in rows
        ]

    def _hit_from_episode_row(self, row: sqlite3.Row, *, rank: float | None = None) -> MemoryHit:
        episode = _episode_from_row(row)
        score = 1.0 if rank is None else 1.0 / (1.0 + abs(rank))
        return MemoryHit(
            memory_id=episode.id,
            namespace=f"episode:{episode.repo}",
            content=_episode_content(episode),
            score=score,
        )

    def search_episodes(
        self,
        query: str,
        *,
        repo: str,
        limit: int,
    ) -> list[MemoryHit]:
        expression = _fts_query(query)
        if not expression or limit <= 0:
            return []
        rows = self._connection.execute(
            """
            SELECT e.*, bm25(episodes_fts) AS rank
            FROM episodes_fts
            JOIN episodes e ON e.id = episodes_fts.episode_id
            WHERE episodes_fts MATCH ?
            ORDER BY
                (e.repo = ?) DESC,
                e.success DESC,
                rank,
                e.created_at DESC
            LIMIT ?
            """,
            (expression, repo, limit),
        ).fetchall()
        return [self._hit_from_episode_row(row, rank=float(row["rank"])) for row in rows]

    def search_episodes_by_fingerprint(
        self, *, repo: str, fingerprint: str, limit: int = 2
    ) -> list[MemoryHit]:
        if not fingerprint or limit <= 0:
            return []
        rows = self._connection.execute(
            """
            SELECT * FROM episodes
            WHERE repo = ? AND failure_fingerprint = ? AND failure_fingerprint != ''
            ORDER BY created_at DESC
            """,
            (repo, fingerprint),
        ).fetchall()
        latest_failed: sqlite3.Row | None = None
        latest_success: sqlite3.Row | None = None
        for row in rows:
            if bool(row["success"]):
                if latest_success is None:
                    latest_success = row
            elif latest_failed is None:
                latest_failed = row
            if latest_failed is not None and latest_success is not None:
                break
        selected = [row for row in (latest_failed, latest_success) if row is not None][:limit]
        return [self._hit_from_episode_row(row) for row in selected]

    def archive(self, memory_id: str) -> None:
        self._connection.execute(
            "UPDATE semantic_memories SET archived = 1 WHERE id = ?", (memory_id,)
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
