"""SQLite-backed append-only trajectory store."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from evoci.runtime.events import RunEvent


class SQLiteEventStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                agent_id TEXT,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_time ON events(run_id, timestamp)"
        )
        self._connection.commit()

    def append(self, event: RunEvent) -> None:
        payload = dict(event.payload)
        if event.invocation_id is not None:
            payload["invocation_id"] = event.invocation_id
        self._connection.execute(
            "INSERT OR IGNORE INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.run_id,
                event.timestamp.isoformat(),
                event.type.value,
                event.agent_id,
                json.dumps(payload, sort_keys=True),
            ),
        )
        self._connection.commit()

    def list(self, run_id: str) -> list[RunEvent]:
        rows = self._connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY timestamp, event_id", (run_id,)
        ).fetchall()
        events: list[RunEvent] = []
        for row in rows:
            payload = json.loads(row["payload"])
            events.append(
                RunEvent.model_validate(
                    {
                        "event_id": row["event_id"],
                        "run_id": row["run_id"],
                        "timestamp": row["timestamp"],
                        "type": row["type"],
                        "agent_id": row["agent_id"],
                        "invocation_id": payload.get("invocation_id"),
                        "payload": payload,
                    }
                )
            )
        return events

    def iter_all(self) -> Iterator[RunEvent]:
        rows = self._connection.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id")
        for row in rows:
            yield from self.list(str(row["run_id"]))

    def count(
        self,
        run_id: str,
        *,
        event_type: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        clauses = ["run_id = ?"]
        arguments: list[str] = [run_id]
        if event_type is not None:
            clauses.append("type = ?")
            arguments.append(event_type)
        if agent_id is not None:
            clauses.append("agent_id = ?")
            arguments.append(agent_id)
        row = self._connection.execute(
            f"SELECT COUNT(*) AS count FROM events WHERE {' AND '.join(clauses)}",
            arguments,
        ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        self._connection.close()
