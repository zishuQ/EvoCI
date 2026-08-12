"""Run metadata persistence independent from graph checkpoints."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

RunStatus = Literal["running", "waiting_approval", "success", "failed"]


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    task_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any]


class SQLiteRunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def create(self, run_id: str, task_id: str, metadata: dict[str, Any] | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        self._connection.execute(
            "INSERT OR IGNORE INTO runs VALUES (?, ?, 'running', ?, ?, ?)",
            (run_id, task_id, now, now, json.dumps(metadata or {}, sort_keys=True)),
        )
        self._connection.commit()

    def update_status(self, run_id: str, status: RunStatus) -> None:
        self._connection.execute(
            "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
            (status, datetime.now(UTC).isoformat(), run_id),
        )
        self._connection.commit()

    def get(self, run_id: str) -> RunRecord | None:
        row = self._connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return RunRecord.model_validate(
            {
                "run_id": row["run_id"],
                "task_id": row["task_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "metadata": json.loads(row["metadata"]),
            }
        )

    def list(self) -> list[RunRecord]:
        rows = self._connection.execute("SELECT run_id FROM runs ORDER BY created_at DESC")
        return [record for row in rows if (record := self.get(str(row["run_id"]))) is not None]

    def close(self) -> None:
        self._connection.close()
