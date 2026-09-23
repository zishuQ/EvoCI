"""Durable local LangGraph checkpoint construction."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from evoci.domain.models import (
    CIFailure,
    Diagnosis,
    EvidenceItem,
    FailedCandidateRef,
    FileEdit,
    FixerOutput,
    InvestigationPlan,
    InvestigationTask,
    MemoryHit,
    PatchProposal,
    RepoSpec,
    ReviewResult,
    SkillCatalogEntry,
    SkillHit,
    SkillRef,
    SkillUsage,
    SupervisorDecision,
    TaskBudget,
    VerificationCommandResult,
    VerificationResult,
    WorkerExecutionResult,
    WorkerTask,
)
from evoci.runtime.events import EventType, RunEvent

_SERIALIZED_TYPES = (
    CIFailure,
    Diagnosis,
    EvidenceItem,
    EventType,
    FailedCandidateRef,
    FileEdit,
    FixerOutput,
    InvestigationPlan,
    InvestigationTask,
    MemoryHit,
    PatchProposal,
    RepoSpec,
    ReviewResult,
    RunEvent,
    SkillCatalogEntry,
    SkillHit,
    SkillRef,
    SkillUsage,
    SupervisorDecision,
    TaskBudget,
    VerificationCommandResult,
    VerificationResult,
    WorkerExecutionResult,
    WorkerTask,
)


def _serializer() -> JsonPlusSerializer:
    allowed = [(cls.__module__, cls.__name__) for cls in _SERIALIZED_TYPES]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


@dataclass(slots=True)
class SQLiteCheckpointHandle:
    saver: SqliteSaver
    connection: sqlite3.Connection

    def close(self) -> None:
        self.connection.close()


def create_sqlite_checkpointer(path: Path) -> SQLiteCheckpointHandle:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection, serde=_serializer())
    saver.setup()
    return SQLiteCheckpointHandle(saver=saver, connection=connection)


@dataclass(slots=True)
class AsyncSQLiteCheckpointHandle:
    saver: AsyncSqliteSaver
    connection: aiosqlite.Connection

    async def close(self) -> None:
        await self.connection.close()


async def create_async_sqlite_checkpointer(path: Path) -> AsyncSQLiteCheckpointHandle:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(connection, serde=_serializer())
    await saver.setup()
    return AsyncSQLiteCheckpointHandle(saver=saver, connection=connection)
