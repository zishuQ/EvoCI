"""Transactional benchmark campaigns with immutable generation snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from evoci.benchmark.models import BenchmarkManifestEntry, BenchmarkResult, BenchmarkVariant
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.skill_memory import merge_memory_texts
from evoci.config import EvoCIConfig
from evoci.memory.store import SQLiteMemoryStore, format_episode_content


class CampaignError(RuntimeError):
    """A campaign invariant was violated."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(*roots: Path) -> str:
    digest = hashlib.sha256()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and not item.name.endswith(("-wal", "-shm", "-journal"))
        ):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_sqlite_state(state: Path) -> None:
    """Flush committed WAL contents before hashing an immutable generation."""
    for database in sorted((*state.glob("*.sqlite"), *state.glob("*.db"))):
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(database)
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            # State directories may contain non-SQLite files with a database-like suffix.
            continue
        finally:
            if connection is not None:
                connection.close()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_json(path: Path, default: object) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


class CampaignManager:
    """Own campaign lineage, frozen reads, checkpoints, and atomic generation commits."""

    schema_version = 1

    def __init__(
        self,
        root: Path,
        *,
        variant: BenchmarkVariant,
        base_config: EvoCIConfig,
        strict: bool = False,
    ) -> None:
        if variant != "evo":
            raise CampaignError("benchmark campaigns currently require --variant evo")
        self.root = root.resolve()
        self.variant = variant
        self.base_config = base_config
        self.strict = strict
        self.metadata_path = self.root / "campaign.json"
        self.generations = self.root / "generations"
        self.rounds = self.root / "rounds"

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        lock = self.root / ".writer.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            owner = _safe_json(lock, {})
            pid = owner.get("pid") if isinstance(owner, dict) else None
            alive = False
            if isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
            if alive:
                raise CampaignError(f"campaign already has a writer: {owner}") from exc
            lock.unlink(missing_ok=True)
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as retry_exc:
                raise CampaignError("campaign writer lock was concurrently acquired") from retry_exc
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "started_at": _now()}).encode())
            os.close(fd)
            yield
        finally:
            lock.unlink(missing_ok=True)

    def initialize(self) -> dict[str, Any]:
        if self.metadata_path.is_file():
            metadata = cast(dict[str, Any], _safe_json(self.metadata_path, {}))
            if metadata.get("variant") != self.variant:
                raise CampaignError("campaign variant does not match this invocation")
            return metadata
        self.generations.mkdir(parents=True, exist_ok=True)
        self.rounds.mkdir(parents=True, exist_ok=True)
        generation = self.generations / "generation-0"
        temporary = self.generations / f".generation-0-{uuid4().hex}.tmp"
        state = temporary / "state"
        skills = temporary / "skills"
        state.mkdir(parents=True)
        skills.mkdir()
        memory = SQLiteMemoryStore(state / "memory.sqlite")
        memory.close()
        registry = CapabilityRegistry(skills, state / "capabilities.sqlite")
        registry.close()
        checkpoint_sqlite_state(state)
        _write_json(
            temporary / "metadata.json",
            {
                "schema_version": self.schema_version,
                "generation": 0,
                "parent_generation": None,
                "status": "complete",
                "created_at": _now(),
                "source_round": None,
                "content_sha256": tree_hash(state, skills),
            },
        )
        os.replace(temporary, generation)
        metadata = {
            "schema_version": self.schema_version,
            "campaign_id": self.root.name or uuid4().hex,
            "created_at": _now(),
            "variant": self.variant,
            "orchestration_schema": 3,
            "model": self._model_summary(),
            "config": self._config_summary(),
            "current_generation": 0,
            "lineage": [{"generation": 0, "parent": None}],
            "rounds": {},
            "freeze_learning_within_round": True,
        }
        _write_json(self.metadata_path, metadata)
        return metadata

    def _config_summary(self) -> dict[str, Any]:
        names = (
            "max_supervisor_batches",
            "max_run_model_calls",
            "max_run_tool_calls",
            "capability_retrieval_top_k",
        )
        return {name: getattr(self.base_config, name) for name in names}

    def _model_summary(self) -> dict[str, Any]:
        supervisor = self.base_config.supervisor_runtime()
        worker = self.base_config.worker_runtime()
        return {
            "base_url": self.base_config.model_base_url,
            "model_name": self.base_config.model_name,
            "orchestration_schema": 3,
            "supervisor": {
                "enable_thinking": supervisor.enable_thinking,
                "reasoning_effort": supervisor.reasoning_effort,
            },
            "worker": {
                "enable_thinking": worker.enable_thinking,
                "reasoning_effort": worker.reasoning_effort,
            },
        }

    def prepare_round(
        self,
        round_number: int,
        manifest: Path,
        dataset: Path,
        entries: list[BenchmarkManifestEntry],
    ) -> tuple[Path, list[str]]:
        if round_number < 1:
            raise CampaignError("round must be >= 1")
        for entry in entries:
            task_path = Path(entry.task_id)
            if (
                not entry.task_id
                or task_path.is_absolute()
                or ".." in task_path.parts
                or "/" in entry.task_id
                or "\\" in entry.task_id
            ):
                raise CampaignError(f"unsafe task_id for campaign staging: {entry.task_id!r}")
        metadata = self.initialize()
        parent = self.generations / f"generation-{round_number - 1}"
        parent_meta = _safe_json(parent / "metadata.json", {})
        if parent_meta.get("status") != "complete":
            raise CampaignError(f"parent Generation {round_number - 1} is missing or incomplete")
        actual_parent_hash = tree_hash(parent / "state", parent / "skills")
        if parent_meta.get("content_sha256") != actual_parent_hash:
            raise CampaignError(f"parent Generation {round_number - 1} was modified after commit")
        if metadata.get("current_generation", 0) < round_number - 1:
            raise CampaignError("campaign lineage does not contain the required parent generation")
        round_dir = self.rounds / f"round-{round_number}"
        hashes = {
            "manifest_sha256": content_hash(manifest),
            "dataset_sha256": content_hash(dataset),
        }
        existing = _safe_json(round_dir / "metadata.json", None)
        if existing is not None:
            if any(existing.get(key) != value for key, value in hashes.items()):
                raise CampaignError("manifest or dataset content changed while resuming the round")
            if existing.get("parent_generation") != round_number - 1:
                raise CampaignError("round parent generation does not match campaign lineage")
            warnings = list(existing.get("warnings", []))
            if (
                existing.get("config") != self._config_summary()
                or existing.get("model") != self._model_summary()
            ):
                warning = "critical model budget or campaign configuration changed on resume"
                if warning not in warnings:
                    warnings.append(warning)
                    existing["warnings"] = warnings
                    _write_json(round_dir / "metadata.json", existing)
            return round_dir, warnings
        if metadata.get("current_generation", 0) >= round_number:
            raise CampaignError(f"Round {round_number} is already finalized")
        warnings = self._task_overlap_warnings(metadata, entries, dataset)
        if warnings and self.strict:
            raise CampaignError("strict task isolation failed: " + "; ".join(warnings))
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "learning-delta").mkdir(exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "round": round_number,
            "parent_generation": round_number - 1,
            "status": "running",
            "created_at": _now(),
            **hashes,
            "manifest": str(manifest.resolve()),
            "dataset": str(dataset.resolve()),
            "task_ids": [entry.task_id for entry in entries],
            "warnings": warnings,
            "config": self._config_summary(),
            "model": self._model_summary(),
        }
        _write_json(round_dir / "metadata.json", payload)
        _write_json(round_dir / "status.json", {"status": "running", "completed_tasks": []})
        metadata["rounds"][str(round_number)] = {
            **hashes,
            "status": "running",
            "task_ids": payload["task_ids"],
        }
        _write_json(self.metadata_path, metadata)
        return round_dir, warnings

    def _task_overlap_warnings(
        self, metadata: dict[str, Any], entries: list[BenchmarkManifestEntry], dataset: Path
    ) -> list[str]:
        # Duplicate task ids inside one manifest make resume/finalization ambiguous
        # and can silently collapse two experiments into one run record.
        current_ids = [entry.task_id for entry in entries]
        duplicates = sorted({task for task in current_ids if current_ids.count(task) > 1})
        if duplicates:
            raise CampaignError(
                "manifest contains duplicate task_id values: " + ", ".join(duplicates)
            )
        previous_ids = {
            task
            for item in metadata.get("rounds", {}).values()
            for task in item.get("task_ids", [])
        }
        warnings = [
            f"duplicate task_id across rounds: {task}"
            for task in sorted(previous_ids & set(current_ids))
        ]
        # Dataset schemas vary; inspect common repository/failing-commit identity fields.
        prior_pairs = {
            tuple(pair)
            for item in metadata.get("rounds", {}).values()
            for pair in item.get("repo_fail_pairs", [])
        }
        pairs: set[tuple[str, str]] = set()
        wanted = set(current_ids)
        for line in dataset.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or row.get("id") or "")
            if task_id not in wanted:
                continue
            repo_value = row.get("repo") or row.get("repository") or {}
            if isinstance(repo_value, dict):
                repo = "/".join(
                    str(repo_value.get(key, "")).strip("/")
                    for key in ("owner", "name")
                    if repo_value.get(key)
                )
            else:
                repo = str(repo_value)
            fail = str(row.get("sha_fail") or row.get("failing_commit") or "")
            if repo and fail:
                pairs.add((repo, fail))
        warnings.extend(
            f"duplicate repository + failing commit: {repo}@{sha}"
            for repo, sha in sorted(prior_pairs & pairs)
        )
        # Reusing an existing task or exact repository/failing commit invalidates
        # the intended A/B/C task isolation.  Strict mode is the opt-in escape
        # hatch for legacy campaigns; normal campaigns retain a visible warning.
        # Store only non-secret task identity fields for later comparisons.
        metadata.setdefault("_pending_repo_fail_pairs", [list(pair) for pair in sorted(pairs)])
        return warnings

    def task_config(self, round_number: int, task_id: str, run_id: str) -> EvoCIConfig:
        task_path = Path(task_id)
        if (
            not task_id
            or task_path.is_absolute()
            or ".." in task_path.parts
            or "/" in task_id
            or "\\" in task_id
        ):
            raise CampaignError(f"unsafe task_id for campaign staging: {task_id!r}")
        parent = self.generations / f"generation-{round_number - 1}"
        branch = self.rounds / f"round-{round_number}" / "learning-delta" / task_id
        state = branch / "state"
        if not state.exists():
            temporary = branch.with_name(f".{branch.name}-{uuid4().hex}.tmp")
            shutil.copytree(parent / "state", temporary / "state")
            shutil.copytree(parent / "skills", temporary / "skills")
            _write_json(
                temporary / "provenance.json",
                {
                    "campaign_id": _safe_json(self.metadata_path, {})["campaign_id"],
                    "round": round_number,
                    "task_id": task_id,
                    "run_id": run_id,
                    "read_generation": round_number - 1,
                },
            )
            os.replace(temporary, branch)
            self._relocate_skill_paths(
                branch / "state",
                stored_skill_root=branch / "skills",
                validation_skill_root=branch / "skills",
            )
        return self.base_config.model_copy(
            update={
                "state_dir": state,
                "capability_dir": branch / "skills",
                "runtime_dir": branch / "runtime",
                "workspace_dir": self.root / "worktrees",
                "repo_cache_dir": self.root / "repo-cache",
            }
        )

    def completed_results(self, round_number: int) -> dict[str, BenchmarkResult]:
        path = self.rounds / f"round-{round_number}" / "runs.jsonl"
        if not path.is_file():
            return {}
        results: dict[str, BenchmarkResult] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result = BenchmarkResult.model_validate_json(line)
                results[result.task_id] = result
        return results

    def finalize_round(self, round_number: int, results: list[BenchmarkResult]) -> Path:
        round_dir = self.rounds / f"round-{round_number}"
        metadata = _safe_json(round_dir / "metadata.json", {})
        expected = set(metadata.get("task_ids", []))
        if set(result.task_id for result in results) != expected:
            raise CampaignError("round is incomplete; refusing to commit a generation")
        # Allow rounds with errors to complete - these represent agent capability limits,
        # not retryable infrastructure failures. Learning data from successful tasks
        # should still be preserved.
        errors = [result.task_id for result in results if result.status == "error"]
        if errors:
            # Log warning but don't fail - record in metadata
            print(
                f"Warning: Round {round_number} completed with {len(errors)} error(s): "
                f"{', '.join(sorted(errors))}"
            )
            metadata.setdefault("warnings", []).append(
                f"completed with {len(errors)} task error(s): {', '.join(sorted(errors))}"
            )
            _write_json(round_dir / "metadata.json", metadata)
        destination = self.generations / f"generation-{round_number}"
        if destination.is_dir():
            complete = _safe_json(destination / "metadata.json", {}).get("status") == "complete"
            if complete:
                return destination
            raise CampaignError("destination generation exists but is incomplete")
        temporary = self.generations / f".generation-{round_number}-{uuid4().hex}.tmp"
        parent = self.generations / f"generation-{round_number - 1}"
        shutil.copytree(parent / "state", temporary / "state")
        shutil.copytree(parent / "skills", temporary / "skills")
        self._merge_branches(round_number, temporary)
        self._relocate_skill_paths(
            temporary / "state",
            stored_skill_root=destination / "skills",
            validation_skill_root=temporary / "skills",
        )
        checkpoint_sqlite_state(temporary / "state")
        generation_hash = tree_hash(temporary / "state", temporary / "skills")
        _write_json(
            temporary / "metadata.json",
            {
                "schema_version": self.schema_version,
                "generation": round_number,
                "parent_generation": round_number - 1,
                "source_round": round_number,
                "status": "complete",
                "created_at": _now(),
                "manifest_sha256": metadata["manifest_sha256"],
                "dataset_sha256": metadata["dataset_sha256"],
                "merged_task_ids": sorted(expected),
                "content_sha256": generation_hash,
            },
        )
        os.replace(temporary, destination)
        metadata["status"] = "complete"
        metadata["completed_at"] = _now()
        _write_json(round_dir / "metadata.json", metadata)
        _write_json(
            round_dir / "status.json", {"status": "complete", "completed_tasks": sorted(expected)}
        )
        campaign = _safe_json(self.metadata_path, {})
        campaign["current_generation"] = round_number
        campaign["lineage"].append(
            {"generation": round_number, "parent": round_number - 1, "round": round_number}
        )
        campaign["rounds"][str(round_number)]["status"] = "complete"
        pending = campaign.pop("_pending_repo_fail_pairs", [])
        campaign["rounds"][str(round_number)]["repo_fail_pairs"] = pending
        campaign["rounds"][str(round_number)]["skills"] = self._skill_inventory(destination)
        aggregate_path = round_dir / "aggregate.json"
        if not aggregate_path.is_file():
            evaluable = [
                result
                for result in results
                if result.metrics is not None
                and result.metrics.benchmark_verification_status in {"passed", "failed"}
            ]
            resolved = sum(
                result.metrics is not None
                and result.metrics.benchmark_verification_status == "passed"
                for result in evaluable
            )
            _write_json(
                aggregate_path,
                {
                    "selected_tasks": len(results),
                    "completed_tasks": len(results),
                    "evaluable_tasks": len(evaluable),
                    "evaluation_coverage": (len(evaluable) / len(results) if results else None),
                    "benchmark_resolved": resolved,
                    "benchmark_success_rate": (resolved / len(evaluable) if evaluable else None),
                },
            )
        aggregate = _safe_json(aggregate_path, {})
        inventory = self._skill_inventory(destination)
        parent_inventory = self._skill_inventory(parent)
        parent_ids = {item["skill_id"] for item in parent_inventory}
        aggregate.update(
            {
                "registry_size": len(inventory),
                "enabled_skill_count": sum(item["enabled"] for item in inventory),
                "skills_created": sum(item["skill_id"] not in parent_ids for item in inventory),
            }
        )
        _write_json(aggregate_path, aggregate)
        _write_json(self.metadata_path, campaign)
        self.write_summary()
        return destination

    def _merge_branches(self, round_number: int, target: Path) -> None:
        delta = self.rounds / f"round-{round_number}" / "learning-delta"
        parent = self.generations / f"generation-{round_number - 1}"
        parent_db = parent / "state" / "capabilities.sqlite"
        metadata = _safe_json(self.rounds / f"round-{round_number}" / "metadata.json", {})
        task_ids = list(metadata.get("task_ids", []))
        if not task_ids:
            task_ids = sorted(path.name for path in delta.iterdir() if path.is_dir())
        claimed_updates: set[str] = set()
        for task_id in task_ids:
            branch = delta / task_id
            if not branch.is_dir():
                continue
            self._merge_memory(
                branch / "state" / "memory.sqlite", target / "state" / "memory.sqlite"
            )
            self._merge_skills(branch, target, parent, parent_db, claimed_updates)

    @staticmethod
    def _merge_memory(source: Path, target: Path) -> None:
        if not source.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        source_store = SQLiteMemoryStore(source)
        source_store.close()
        target_store = SQLiteMemoryStore(target)
        target_store.close()
        connection = sqlite3.connect(target)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("ATTACH DATABASE ? AS delta", (str(source),))
            columns = [
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(episodes)").fetchall()
            ]
            column_sql = ", ".join(columns)
            connection.execute(
                f"INSERT OR IGNORE INTO main.episodes ({column_sql}) "
                f"SELECT {column_sql} FROM delta.episodes"
            )
            memory_rows = connection.execute("SELECT * FROM delta.long_term_memories").fetchall()
            for row in memory_rows:
                connection.execute(
                    """
                    INSERT INTO main.long_term_memories
                        (id, repository, content, confidence, source_run_ids,
                         created_at, updated_at, last_accessed_at, archived)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        repository=excluded.repository, content=excluded.content,
                        confidence=excluded.confidence,
                        source_run_ids=excluded.source_run_ids, updated_at=excluded.updated_at,
                        last_accessed_at=excluded.last_accessed_at, archived=excluded.archived
                    """,
                    tuple(row),
                )
            connection.execute(
                "INSERT OR IGNORE INTO main.applied_operations "
                "SELECT * FROM delta.applied_operations"
            )
            # Bug fix #4: Wrap FTS rebuild in a transaction with savepoint
            # If rebuild fails midway, the entire operation rolls back
            connection.execute("SAVEPOINT fts_rebuild")
            try:
                connection.execute("DELETE FROM main.episodes_fts")
                connection.execute("DELETE FROM main.long_term_memory_fts")
                rows = connection.execute("SELECT * FROM main.episodes").fetchall()
                for row in rows:
                    payload = dict(row)
                    content = format_episode_content(
                        success=bool(payload["success"]),
                        failure_summary=str(payload["failure_summary"]),
                        root_cause=payload.get("root_cause"),
                        successful_fix=payload.get("successful_fix_summary"),
                        failure_reason=payload.get("failure_reason"),
                        hypotheses=json.loads(str(payload.get("hypotheses_attempted") or "[]")),
                        verification_failures=json.loads(
                            str(payload.get("verification_failures") or "[]")
                        ),
                        failure_class=payload.get("failure_class"),
                        failure_stage=payload.get("failure_stage"),
                        attempted_fixes=json.loads(
                            str(payload.get("attempted_fix_summaries") or "[]")
                        ),
                        attempted_files=json.loads(str(payload.get("attempted_files") or "[]")),
                        external_failure_details=json.loads(
                            str(payload.get("external_failure_details") or "[]")
                        ),
                    )
                    connection.execute(
                        "INSERT INTO main.episodes_fts VALUES (?, ?)", (payload["id"], content)
                    )
                connection.execute(
                    "INSERT INTO main.long_term_memory_fts "
                    "SELECT id, content FROM main.long_term_memories"
                )
                connection.execute("RELEASE SAVEPOINT fts_rebuild")
            except Exception:
                connection.execute("ROLLBACK TO SAVEPOINT fts_rebuild")
                raise
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _merge_sqlite(source: Path, target: Path, tables: tuple[str, ...]) -> None:
        if not source.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(target)
        try:
            connection.execute("ATTACH DATABASE ? AS delta", (str(source),))
            for table in tables:
                exists = connection.execute(
                    "SELECT 1 FROM delta.sqlite_master WHERE name=?", (table,)
                ).fetchone()
                target_exists = connection.execute(
                    "SELECT 1 FROM main.sqlite_master WHERE name=?", (table,)
                ).fetchone()
                if exists and target_exists:
                    connection.execute(
                        f"INSERT OR IGNORE INTO main.{table} SELECT * FROM delta.{table}"
                    )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _merge_skills(
        branch: Path,
        target: Path,
        parent: Path,
        parent_db: Path,
        claimed_updates: set[str],
    ) -> None:
        source_db = branch / "state" / "capabilities.sqlite"
        target_db = target / "state" / "capabilities.sqlite"
        if not source_db.is_file():
            return
        accepted = CampaignManager._merge_skill_packages(
            branch, target, parent, claimed_updates
        )
        CampaignManager._merge_skill_rows(source_db, target_db, target, accepted)
        CampaignManager._merge_existing_skill_stat_deltas(source_db, target_db, parent_db)
        target_connection = sqlite3.connect(target_db)
        try:
            target_connection.execute("DELETE FROM skill_fts")
            for row in target_connection.execute("SELECT skill_id,manifest_json FROM skills"):
                manifest = json.loads(str(row[1]))
                content = " ".join(
                    [
                        manifest["name"],
                        manifest["description"],
                        *manifest["triggers"],
                        *manifest["task_families"],
                    ]
                )
                target_connection.execute(
                    "INSERT INTO skill_fts(skill_id,content) VALUES (?,?)",
                    (row[0], content),
                )
            target_connection.commit()
        finally:
            target_connection.close()
        CampaignManager._relocate_skill_paths(
            target / "state",
            stored_skill_root=target / "skills",
            validation_skill_root=target / "skills",
        )

    @staticmethod
    def _merge_skill_packages(
        branch: Path,
        target: Path,
        parent: Path,
        claimed_updates: set[str],
    ) -> set[str]:
        source_root = branch / "skills"
        target_root = target / "skills"
        parent_root = parent / "skills"
        accepted: set[str] = set()
        if not source_root.is_dir():
            return accepted
        target_root.mkdir(parents=True, exist_ok=True)
        for skill_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            skill_id = skill_dir.name
            if skill_id.startswith("."):
                continue
            destination = target_root / skill_id
            parent_skill = parent_root / skill_id
            if not destination.exists():
                shutil.copytree(skill_dir, destination)
                claimed_updates.add(skill_id)
                accepted.add(skill_id)
                continue
            CampaignManager._merge_skill_memory(
                destination / "memory.md", skill_dir / "memory.md"
            )
            source_package = skill_dir / "package"
            parent_package = parent_skill / "package"
            dest_package = destination / "package"
            if not source_package.is_dir():
                continue
            package_changed = (
                not parent_package.is_dir()
                or tree_hash(source_package) != tree_hash(parent_package)
            )
            if package_changed and skill_id not in claimed_updates:
                dest_previous = destination / "previous"
                if dest_previous.exists():
                    shutil.rmtree(dest_previous)
                if dest_package.exists():
                    dest_package.rename(dest_previous)
                shutil.copytree(source_package, dest_package)
                claimed_updates.add(skill_id)
                accepted.add(skill_id)
        return accepted

    @staticmethod
    def _merge_skill_memory(destination: Path, source: Path) -> None:
        texts: list[str] = []
        if destination.is_file():
            texts.append(destination.read_text(encoding="utf-8"))
        if source.is_file():
            texts.append(source.read_text(encoding="utf-8"))
        merged = merge_memory_texts(texts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(merged, encoding="utf-8")

    @staticmethod
    def _merge_skill_rows(
        source: Path,
        target: Path,
        target_root: Path,
        accepted_ids: set[str],
    ) -> None:
        connection = sqlite3.connect(target)
        connection.row_factory = sqlite3.Row
        source_connection = sqlite3.connect(source)
        source_connection.row_factory = sqlite3.Row
        try:
            for row in source_connection.execute("SELECT * FROM skills"):
                skill_id = str(row["skill_id"])
                package_path = str(target_root / "skills" / skill_id / "package")
                existing = connection.execute(
                    "SELECT skill_id FROM skills WHERE skill_id=?",
                    (skill_id,),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO skills "
                        "(skill_id,enabled,manifest_json,package_path) VALUES (?,?,?,?)",
                        (
                            skill_id,
                            row["enabled"],
                            row["manifest_json"],
                            package_path,
                        ),
                    )
                elif skill_id in accepted_ids:
                    connection.execute(
                        """
                        UPDATE skills SET enabled=?, manifest_json=?, package_path=?
                        WHERE skill_id=?
                        """,
                        (row["enabled"], row["manifest_json"], package_path, skill_id),
                    )
                stats = source_connection.execute(
                    "SELECT * FROM skill_stats WHERE skill_id=?",
                    (skill_id,),
                ).fetchone()
                columns = [
                    item[1]
                    for item in source_connection.execute("PRAGMA table_info(skill_stats)")
                ]
                if stats is not None:
                    connection.execute(
                        f"INSERT OR IGNORE INTO skill_stats ({','.join(columns)}) "
                        f"VALUES ({','.join('?' for _ in columns)})",
                        tuple(stats),
                    )
                else:
                    now = datetime.now(UTC).isoformat()
                    connection.execute(
                        "INSERT OR IGNORE INTO skill_stats"
                        "(skill_id, created_at, updated_at) VALUES (?, ?, ?)",
                        (skill_id, now, now),
                    )
                operation = source_connection.execute(
                    "SELECT operation_key,result_json,created_at "
                    "FROM applied_operations WHERE result_json LIKE ?",
                    (f'%"skill_id": "{skill_id}"%',),
                ).fetchall()
                for item in operation:
                    connection.execute(
                        "INSERT OR IGNORE INTO applied_operations VALUES (?,?,?)", tuple(item)
                    )
            connection.commit()
        finally:
            source_connection.close()
            connection.close()

    @staticmethod
    def _merge_existing_skill_stat_deltas(source: Path, target: Path, parent: Path) -> None:
        if not parent.is_file():
            return
        connection = sqlite3.connect(target)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("ATTACH DATABASE ? AS delta", (str(source),))
            connection.execute("ATTACH DATABASE ? AS parent", (str(parent),))
            rows = connection.execute(
                """
                SELECT d.skill_id,
                       d.retrieval_count - p.retrieval_count AS retrieval_delta,
                       d.selected_count - p.selected_count AS selected_delta,
                       d.use_count - p.use_count AS use_delta,
                       d.success_count - p.success_count AS success_delta,
                       d.failure_count - p.failure_count AS failure_delta,
                       d.patch_count - p.patch_count AS patch_delta
                FROM delta.skill_stats d
                JOIN parent.skill_stats p USING (skill_id)
                """
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE main.skill_stats SET
                        retrieval_count = retrieval_count + ?,
                        selected_count = selected_count + ?,
                        use_count = use_count + ?,
                        success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        patch_count = patch_count + ?
                    WHERE skill_id = ?
                    """,
                    (
                        max(0, row["retrieval_delta"]),
                        max(0, row["selected_delta"]),
                        max(0, row["use_delta"]),
                        max(0, row["success_delta"]),
                        max(0, row["failure_delta"]),
                        max(0, row["patch_delta"]),
                        row["skill_id"],
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _relocate_skill_paths(
        state: Path,
        stored_skill_root: Path,
        validation_skill_root: Path | None = None,
    ) -> None:
        database = state / "capabilities.sqlite"
        if not database.is_file():
            return
        validation_root = validation_skill_root or stored_skill_root
        connection = sqlite3.connect(database)
        try:
            rows = connection.execute("SELECT skill_id FROM skills").fetchall()
            updates = []
            missing = []
            for (skill_id,) in rows:
                validated = validation_root / str(skill_id) / "package"
                stored = stored_skill_root / str(skill_id) / "package"
                if validated.is_dir():
                    updates.append((str(stored), skill_id))
                else:
                    missing.append((skill_id, str(validated)))

            if missing:
                details = ", ".join(f"{skill_id} at {path}" for skill_id, path in missing)
                raise CampaignError(f"skill package missing during generation commit: {details}")

            for package_path, skill_id in updates:
                connection.execute(
                    "UPDATE skills SET package_path=? WHERE skill_id=?",
                    (package_path, skill_id),
                )
            connection.commit()
        finally:
            connection.close()

    def write_summary(self) -> None:
        rounds: list[dict[str, Any]] = []
        for path in sorted(self.rounds.glob("round-*")):
            aggregate = _safe_json(path / "aggregate.json", None)
            if aggregate is not None:
                number = int(path.name.split("-")[1])
                rounds.append(
                    {
                        "round": number,
                        **aggregate,
                        "skills": self._skill_inventory(self.generations / f"generation-{number}"),
                    }
                )
        comparisons = []
        for before, after in pairwise(rounds):
            comparison: dict[str, Any] = {"from_round": before["round"], "to_round": after["round"]}
            for key, delta_name in (
                ("benchmark_success_rate", "success_rate_delta"),
                ("benchmark_success_rate", "benchmark_success_rate_delta"),
                ("evaluation_coverage", "evaluation_coverage_delta"),
                ("total_tokens", "total_tokens_delta"),
                ("repair_tokens", "repair_tokens_delta"),
                ("learning_tokens", "learning_tokens_delta"),
                ("tokens_per_resolved_task", "tokens_per_resolved_task_delta"),
                ("provider_requests", "provider_requests_delta"),
                ("tool_calls_per_resolved_task", "tool_calls_per_resolved_task_delta"),
                ("attempts_per_resolved_task", "attempts_per_resolved_task_delta"),
            ):
                left, right = before.get(key), after.get(key)
                comparison[delta_name] = (
                    right - left if left is not None and right is not None else None
                )
            comparisons.append(comparison)
        summary = {
            "campaign_id": _safe_json(self.metadata_path, {}).get("campaign_id"),
            "rounds": rounds,
            "comparisons": comparisons,
            "interpretation": (
                "Round differences are observational and are not, by themselves, "
                "strict causal estimates."
            ),
        }
        _write_json(self.root / "campaign-summary.json", summary)
        lines = ["# EvoCI Benchmark Campaign", "", summary["interpretation"], ""]
        for item in rounds:
            lines.extend(
                [
                    f"## Round {item['round']}",
                    "",
                    f"- Success rate: {item.get('benchmark_success_rate')}",
                    f"- Evaluation coverage: {item.get('evaluation_coverage')}",
                    f"- Resolved: {item.get('benchmark_resolved')}",
                    "",
                ]
            )
        for item in comparisons:
            lines.extend(
                [
                    f"## Round {item['from_round']} → Round {item['to_round']}",
                    "",
                    *[f"- {key}: {value}" for key, value in item.items() if key.endswith("_delta")],
                    "",
                ]
            )
        (self.root / "campaign-summary.md").write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _skill_inventory(generation: Path) -> list[dict[str, Any]]:
        database = generation / "state" / "capabilities.sqlite"
        if not database.is_file():
            return []
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            inventory = []
            rows = connection.execute(
                """
                SELECT s.skill_id, s.enabled, s.manifest_json,
                       st.retrieval_count, st.selected_count, st.use_count,
                       st.success_count, st.failure_count
                FROM skills s JOIN skill_stats st USING (skill_id)
                ORDER BY s.skill_id
                """
            ).fetchall()
            for row in rows:
                manifest = json.loads(str(row["manifest_json"]))
                inventory.append(
                    {
                        "skill_id": row["skill_id"],
                        "enabled": bool(row["enabled"]),
                        "source_run_ids": manifest.get("source_run_ids", []),
                        "retrieval_count": row["retrieval_count"],
                        "selected_count": row["selected_count"],
                        "use_count": row["use_count"],
                        "success_count": row["success_count"],
                        "failure_count": row["failure_count"],
                    }
                )
            return inventory
        finally:
            connection.close()
