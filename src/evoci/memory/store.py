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

    def search_episodes(self, query: str, *, repo: str, limit: int) -> list[MemoryHit]: ...

    def archive(self, memory_id: str) -> None: ...


def _fts_query(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())[:24]
    return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))


def format_episode_content(
    *,
    success: bool,
    failure_summary: str,
    root_cause: str | None = None,
    successful_fix: str | None = None,
    failure_reason: str | None = None,
    hypotheses: list[str] | None = None,
    verification_failures: list[str] | None = None,
) -> str:
    """Keep outcome and counterevidence visible; mark unverified causes as hypotheses."""

    hypotheses = hypotheses or []
    verification_failures = verification_failures or []
    parts = [
        f"OUTCOME={'success' if success else 'failed'}",
        failure_summary,
    ]
    if failure_reason:
        parts.append(f"FAILURE_REASON={failure_reason}")
    if verification_failures:
        joined = "; ".join(str(item) for item in verification_failures)
        parts.append(f"VERIFICATION_FAILURES={joined}")
    if success:
        if root_cause:
            parts.append(f"ROOT_CAUSE={root_cause}")
        if successful_fix:
            parts.append(f"SUCCESSFUL_FIX={successful_fix}")
    else:
        if root_cause:
            parts.append(f"HYPOTHESIS (unverified): {root_cause}")
        if hypotheses:
            parts.append(
                "HYPOTHESES_ATTEMPTED (unverified): " + "; ".join(str(item) for item in hypotheses)
            )
        if successful_fix:
            parts.append(f"UNCONFIRMED_FIX_SUMMARY={successful_fix}")
    return " | ".join(part for part in parts if part)


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
        ):
            if name not in episode_columns:
                self._connection.execute(f"ALTER TABLE episodes ADD COLUMN {name} {declaration}")
        self._connection.commit()

    def add_episode(self, episode: Episode) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO episodes (
                id, run_id, repo, task_family, failure_summary, root_cause,
                important_evidence, attempts, successful_fix_summary, tools_used,
                hypotheses_attempted, verification_failures, failure_reason, success, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                episode.created_at.isoformat(),
            ),
        )
        if self._connection.execute("SELECT changes()").fetchone()[0]:
            content = format_episode_content(
                success=episode.success,
                failure_summary=episode.failure_summary,
                root_cause=episode.root_cause,
                successful_fix=episode.successful_fix_summary,
                failure_reason=episode.failure_reason,
                hypotheses=list(episode.hypotheses_attempted),
                verification_failures=list(episode.verification_failures),
            )
            self._connection.execute(
                "INSERT INTO episodes_fts(episode_id, content) VALUES (?, ?)",
                (episode.id, content),
            )
        self._connection.commit()

    def get_episode(self, run_id: str) -> Episode | None:
        row = self._connection.execute(
            "SELECT * FROM episodes WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        for field in (
            "important_evidence",
            "tools_used",
            "hypotheses_attempted",
            "verification_failures",
        ):
            payload[field] = json.loads(str(payload[field]))
        payload["success"] = bool(payload["success"])
        return Episode.model_validate(payload)

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
        except Exception:
            self._connection.rollback()
            raise

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

    def search_episodes(self, query: str, *, repo: str, limit: int) -> list[MemoryHit]:
        expression = _fts_query(query)
        if not expression or limit <= 0:
            return []
        rows = self._connection.execute(
            """
            SELECT e.id, e.failure_summary, e.root_cause, e.successful_fix_summary,
                   e.hypotheses_attempted, e.verification_failures, e.failure_reason,
                   e.success, bm25(episodes_fts) AS rank
            FROM episodes_fts
            JOIN episodes e ON e.id = episodes_fts.episode_id
            WHERE episodes_fts MATCH ? AND e.repo = ?
            ORDER BY rank, e.created_at DESC
            LIMIT ?
            """,
            (expression, repo, limit),
        ).fetchall()
        hits: list[MemoryHit] = []
        for row in rows:
            payload = dict(row)
            for field in ("hypotheses_attempted", "verification_failures"):
                raw = payload.get(field) or "[]"
                payload[field] = json.loads(str(raw)) if not isinstance(raw, list) else raw
            payload["success"] = bool(payload["success"])
            hits.append(
                MemoryHit(
                    memory_id=str(row["id"]),
                    namespace=f"episode:{repo}",
                    content=format_episode_content(
                success=bool(payload["success"]),
                failure_summary=str(payload.get("failure_summary") or ""),
                root_cause=(
                    None if not payload.get("root_cause") else str(payload.get("root_cause"))
                ),
                successful_fix=(
                    None
                    if payload.get("successful_fix_summary") is None
                    else str(payload.get("successful_fix_summary"))
                ),
                failure_reason=(
                    None
                    if payload.get("failure_reason") is None
                    else str(payload.get("failure_reason"))
                ),
                hypotheses=[str(item) for item in payload.get("hypotheses_attempted") or []],
                verification_failures=[
                    str(item) for item in payload.get("verification_failures") or []
                ],
            ),
                    score=1.0 / (1.0 + abs(float(row["rank"]))),
                )
            )
        return hits

    def archive(self, memory_id: str) -> None:
        self._connection.execute(
            "UPDATE semantic_memories SET archived = 1 WHERE id = ?", (memory_id,)
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
