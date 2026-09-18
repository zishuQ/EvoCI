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
from evoci.capability.curator import DeterministicCurator
from evoci.capability.promotion import TrialPromotionPolicy
from evoci.capability.registry import CapabilityRegistry
from evoci.config import EvoCIConfig
from evoci.memory.store import SQLiteMemoryStore


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
            "model": {
                "base_url": self.base_config.model_base_url,
                "model_name": self.base_config.model_name,
                "fast_model_name": self.base_config.fast_model_name,
                "strong_model_name": self.base_config.strong_model_name,
                "aux_model_name": self.base_config.aux_model_name,
            },
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
            "max_repair_attempts",
            "max_run_model_calls",
            "max_run_tool_calls",
            "trial_min_uses",
            "trial_min_successes",
            "trial_min_success_rate",
            "trial_max_failures",
            "capability_retrieval_top_k",
        )
        return {name: getattr(self.base_config, name) for name in names}

    def prepare_round(
        self,
        round_number: int,
        manifest: Path,
        dataset: Path,
        entries: list[BenchmarkManifestEntry],
    ) -> tuple[Path, list[str]]:
        if round_number < 1:
            raise CampaignError("round must be >= 1")
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
            if existing.get("config") != self._config_summary():
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
        previous_ids = {
            task
            for item in metadata.get("rounds", {}).values()
            for task in item.get("task_ids", [])
        }
        current_ids = [entry.task_id for entry in entries]
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
            repo = str(row.get("repo") or row.get("repository") or "")
            fail = str(row.get("sha_fail") or row.get("failing_commit") or "")
            if repo and fail:
                pairs.add((repo, fail))
        warnings.extend(
            f"duplicate repository + failing commit: {repo}@{sha}"
            for repo, sha in sorted(prior_pairs & pairs)
        )
        # Store only non-secret task identity fields for later comparisons.
        metadata.setdefault("_pending_repo_fail_pairs", [list(pair) for pair in sorted(pairs)])
        return warnings

    def task_config(self, round_number: int, task_id: str, run_id: str) -> EvoCIConfig:
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
            self._relocate_skill_paths(branch / "state", branch / "skills")
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
        errors = [result.task_id for result in results if result.status == "error"]
        if errors:
            raise CampaignError(
                "round has retryable task errors; refusing to commit a generation: "
                + ", ".join(sorted(errors))
            )
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
        self._relocate_skill_paths(temporary / "state", destination / "skills")
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
                result for result in results
                if result.metrics is not None
                and result.metrics.benchmark_verification_status in {"passed", "failed"}
            ]
            resolved = sum(
                result.metrics is not None
                and result.metrics.benchmark_verification_status == "passed"
                for result in evaluable
            )
            _write_json(aggregate_path, {
                "selected_tasks": len(results),
                "completed_tasks": len(results),
                "evaluable_tasks": len(evaluable),
                "evaluation_coverage": (
                    len(evaluable) / len(results) if results else None
                ),
                "benchmark_resolved": resolved,
                "benchmark_success_rate": (
                    resolved / len(evaluable) if evaluable else None
                ),
            })
        _write_json(self.metadata_path, campaign)
        self.write_summary()
        return destination

    def _merge_branches(self, round_number: int, target: Path) -> None:
        delta = self.rounds / f"round-{round_number}" / "learning-delta"
        parent_db = (
            self.generations / f"generation-{round_number - 1}" / "state" / "capabilities.sqlite"
        )
        for branch in sorted(path for path in delta.iterdir() if path.is_dir()):
            self._merge_sqlite(
                branch / "state" / "memory.sqlite",
                target / "state" / "memory.sqlite",
                (
                    "episodes",
                    "episodes_fts",
                    "semantic_memories",
                    "semantic_fts",
                    "applied_operations",
                ),
            )
            self._merge_skills(branch, target, parent_db)
        self._finalize_skill_lifecycle(round_number, target)

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
    def _merge_skills(branch: Path, target: Path, parent_db: Path) -> None:
        source_db = branch / "state" / "capabilities.sqlite"
        target_db = target / "state" / "capabilities.sqlite"
        if not source_db.is_file():
            return
        # Registry operation keys make independently mined candidates stable. Merge rows whose
        # version key is free; colliding equivalents are deliberately deduplicated.
        CampaignManager._merge_sqlite(
            source_db, target_db, ("skills", "skill_fts", "skill_stats", "applied_operations")
        )
        CampaignManager._merge_existing_skill_stat_deltas(source_db, target_db, parent_db)
        for package in sorted((branch / "skills").glob("*/v*")):
            relative = package.relative_to(branch / "skills")
            destination = target / "skills" / relative
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(package, destination)
        CampaignManager._relocate_skill_paths(target / "state", target / "skills")

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
                SELECT d.skill_id, d.version,
                       d.retrieval_count - p.retrieval_count AS retrieval_delta,
                       d.selected_count - p.selected_count AS selected_delta,
                       d.use_count - p.use_count AS use_delta,
                       d.success_count - p.success_count AS success_delta,
                       d.failure_count - p.failure_count AS failure_delta,
                       d.patch_count - p.patch_count AS patch_delta
                FROM delta.skill_stats d
                JOIN parent.skill_stats p USING (skill_id, version)
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
                    WHERE skill_id = ? AND version = ?
                    """,
                    (
                        max(0, row["retrieval_delta"]),
                        max(0, row["selected_delta"]),
                        max(0, row["use_delta"]),
                        max(0, row["success_delta"]),
                        max(0, row["failure_delta"]),
                        max(0, row["patch_delta"]),
                        row["skill_id"],
                        row["version"],
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def _finalize_skill_lifecycle(self, round_number: int, target: Path) -> None:
        registry = CapabilityRegistry(target / "skills", target / "state" / "capabilities.sqlite")
        try:
            policy = TrialPromotionPolicy(
                min_uses=self.base_config.trial_min_uses,
                min_successes=self.base_config.trial_min_successes,
                min_success_rate=self.base_config.trial_min_success_rate,
                max_failures=self.base_config.trial_max_failures,
                max_exposures_without_use=self.base_config.trial_max_exposures_without_use,
            )
            for record in registry.list({"trial"}):
                manifest = record.manifest
                policy.apply(
                    registry,
                    record,
                    operation_key=(
                        f"campaign-round:{round_number}:lifecycle:"
                        f"{manifest.skill_id}:v{manifest.version}"
                    ),
                )
            DeterministicCurator(
                registry,
                max_exposures_without_use=(self.base_config.trial_max_exposures_without_use),
            ).run(operation_prefix=f"campaign-round:{round_number}:curator")
        finally:
            registry.close()

    @staticmethod
    def _relocate_skill_paths(state: Path, skill_root: Path) -> None:
        database = state / "capabilities.sqlite"
        if not database.is_file():
            return
        connection = sqlite3.connect(database)
        try:
            rows = connection.execute("SELECT skill_id, version FROM skills").fetchall()
            for skill_id, version in rows:
                package = skill_root / str(skill_id) / f"v{version}"
                connection.execute(
                    "UPDATE skills SET package_path=? WHERE skill_id=? AND version=?",
                    (str(package), skill_id, version),
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
            for key in (
                "benchmark_success_rate",
                "evaluation_coverage",
                "tokens_per_resolved_task",
                "tool_calls_per_resolved_task",
                "attempts_per_resolved_task",
            ):
                left, right = before.get(key), after.get(key)
                comparison[f"{key}_delta"] = (
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
                SELECT s.skill_id, s.version, s.status, s.manifest_json,
                       st.retrieval_count, st.selected_count, st.use_count,
                       st.success_count, st.failure_count
                FROM skills s JOIN skill_stats st USING (skill_id, version)
                ORDER BY s.skill_id, s.version
                """
            ).fetchall()
            for row in rows:
                manifest = json.loads(str(row["manifest_json"]))
                inventory.append(
                    {
                        "skill_id": row["skill_id"],
                        "version": row["version"],
                        "status": row["status"],
                        "source_run_ids": manifest.get("source_run_ids", []),
                        "parent_version": manifest.get("parent_version"),
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
