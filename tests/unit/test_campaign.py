import json
from pathlib import Path

import pytest

from evoci.benchmark.campaign import CampaignError, CampaignManager
from evoci.benchmark.models import (
    BenchmarkManifestEntry,
    BenchmarkResult,
    BenchmarkVerificationResult,
    FinalWorkspaceChanges,
    RunMetrics,
)
from evoci.capability.models import SkillCandidate, SkillPermissions, SkillVersionRef
from evoci.capability.registry import CapabilityRegistry
from evoci.config import EvoCIConfig
from evoci.memory.models import Episode
from evoci.memory.store import SQLiteMemoryStore


def _files(root: Path, task_ids: list[str]) -> tuple[Path, Path]:
    manifest = root / "manifest.jsonl"
    dataset = root / "dataset.jsonl"
    manifest.write_text(
        "".join(json.dumps({"task_id": task, "category": "test"}) + "\n" for task in task_ids)
    )
    dataset.write_text(
        "".join(
            json.dumps({"task_id": task, "repo": f"org/{task}", "sha_fail": task}) + "\n"
            for task in task_ids
        )
    )
    return manifest, dataset


def _entries(task_ids: list[str]) -> list[BenchmarkManifestEntry]:
    return [BenchmarkManifestEntry(task_id=task, category="test") for task in task_ids]


def _result(task_id: str) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        variant="evo",
        status="resolved",
        category="test",
        metrics=RunMetrics(
            agent_declared_success=True,
            targeted_verification_passed=True,
            review_passed=True,
            benchmark_verification=BenchmarkVerificationResult(
                status="passed", details="offline oracle passed"
            ),
            benchmark_verification_status="passed",
            benchmark_resolved=True,
            final_workspace_changes=FinalWorkspaceChanges(changed_files=[]),
            wall_time=0.01,
        ),
    )


def _manager(root: Path) -> CampaignManager:
    return CampaignManager(
        root / "campaign", variant="evo", base_config=EvoCIConfig.from_env(cwd=root)
    )


def _add_episode(config: EvoCIConfig, run_id: str, task: str, success: bool = True) -> None:
    store = SQLiteMemoryStore(config.state_dir / "memory.sqlite")
    try:
        store.add_episode(
            Episode(
                id=f"episode:{run_id}",
                run_id=run_id,
                repo=f"org/{task}",
                task_family="test",
                failure_summary=f"failure {task}",
                attempts=1,
                successful_fix_summary="fixed" if success else None,
                success=success,
                failure_reason=None if success else "oracle failed",
            )
        )
    finally:
        store.close()


def _add_skill(config: EvoCIConfig, run_id: str) -> None:
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        registry.create_candidate(
            SkillCandidate(
                name="Offline repair",
                description="Reusable offline repair",
                triggers=["offline"],
                task_families=["test"],
                skill_md=(
                    "---\nname: Offline repair\ndescription: reusable\n---\n"
                    "# Purpose\nrepair\n# When to Use\noffline\n# Procedure\nfix\n"
                    "# Pitfalls\nnone\n# Verification\ntest\n# Bundled Resources\nnone\n"
                ),
                source_run_ids=[run_id],
                confidence=0.9,
                permissions=SkillPermissions(),
            ),
            operation_key=f"campaign:{run_id}:skill",
        )
    finally:
        registry.close()


def test_campaign_freezes_round_and_inherits_across_manager_instances(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a", "b"])
    manager.prepare_round(1, manifest, dataset, _entries(["a", "b"]))
    a = manager.task_config(1, "a", "run-a")
    _add_episode(a, "run-a", "a")
    _add_skill(a, "run-a")
    b = manager.task_config(1, "b", "run-b")
    b_store = SQLiteMemoryStore(b.state_dir / "memory.sqlite")
    b_registry = CapabilityRegistry(b.capability_dir, b.state_dir / "capabilities.sqlite")
    try:
        assert b_store.get_episode("run-a") is None
        assert b_registry.list() == []
    finally:
        b_store.close()
        b_registry.close()
    generation = manager.finalize_round(1, [_result("a"), _result("b")])
    assert generation.name == "generation-1"

    # A fresh manager represents a new CLI process and must resolve Generation 1.
    second = _manager(tmp_path)
    manifest2, dataset2 = _files(tmp_path, ["c"])
    second.prepare_round(2, manifest2, dataset2, _entries(["c"]))
    c = second.task_config(2, "c", "run-c")
    store = SQLiteMemoryStore(c.state_dir / "memory.sqlite")
    registry = CapabilityRegistry(c.capability_dir, c.state_dir / "capabilities.sqlite")
    try:
        assert store.get_episode("run-a") is not None
        skill = registry.list()[0]
        assert skill.manifest.source_run_ids == ["run-a"]
        assert Path(skill.package_path).is_relative_to(c.capability_dir)
    finally:
        store.close()
        registry.close()
    _add_episode(c, "run-c", "c")
    second.finalize_round(2, [_result("c")])
    assert (second.generations / "generation-2" / "metadata.json").is_file()
    assert (second.root / "campaign-summary.json").is_file()


def test_campaign_recovery_hash_lock_and_idempotent_finalize(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    assert not (manager.generations / "generation-1").exists()
    with (
        manager.writer_lock(),
        pytest.raises(CampaignError, match="already has a writer"),
        manager.writer_lock(),
    ):
        pass
    manifest.write_text(json.dumps({"task_id": "changed", "category": "test"}) + "\n")
    with pytest.raises(CampaignError, match="content changed"):
        manager.prepare_round(1, manifest, dataset, _entries(["a"]))

    manifest, dataset = _files(tmp_path, ["a"])
    first = manager.finalize_round(1, [_result("a")])
    second = manager.finalize_round(1, [_result("a")])
    assert first == second


def test_campaign_refuses_missing_parent_and_strict_overlap(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    with pytest.raises(CampaignError, match="parent Generation 1"):
        manager.prepare_round(2, manifest, dataset, _entries(["a"]))

    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    manager.finalize_round(1, [_result("a")])
    strict = CampaignManager(
        manager.root, variant="evo", base_config=manager.base_config, strict=True
    )
    manifest2 = tmp_path / "manifest-2.jsonl"
    dataset2 = tmp_path / "dataset-2.jsonl"
    manifest2.write_text(manifest.read_text())
    dataset2.write_text(dataset.read_text())
    with pytest.raises(CampaignError, match="strict task isolation"):
        strict.prepare_round(2, manifest2, dataset2, _entries(["a"]))


def test_campaign_metadata_has_no_ground_truth(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    serialized = "".join(path.read_text(errors="ignore") for path in manager.root.rglob("*.json"))
    for forbidden in ("sha_success", "ground-truth patch", "reference changed files"):
        assert forbidden not in serialized


def test_generation_hash_ignores_sqlite_sidecars_but_detects_mutation(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    generation = manager.finalize_round(1, [_result("a")])
    (generation / "state" / "memory.sqlite-shm").write_bytes(b"ephemeral")
    manifest2, dataset2 = _files(tmp_path, ["b"])
    manager.prepare_round(2, manifest2, dataset2, _entries(["b"]))

    database = generation / "state" / "memory.sqlite"
    database.write_bytes(database.read_bytes() + b"tampered")
    with pytest.raises(CampaignError, match="modified after commit"):
        manager.prepare_round(2, manifest2, dataset2, _entries(["b"]))


def test_campaign_aggregates_frozen_branch_skill_usage(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["seed"])
    manager.prepare_round(1, manifest, dataset, _entries(["seed"]))
    seed = manager.task_config(1, "seed", "run-seed")
    _add_skill(seed, "run-seed")
    seed_registry = CapabilityRegistry(seed.capability_dir, seed.state_dir / "capabilities.sqlite")
    try:
        seed_registry.transition("offline-repair", 1, "trial")
    finally:
        seed_registry.close()
    manager.finalize_round(1, [_result("seed")])

    manifest2, dataset2 = _files(tmp_path, ["left", "right"])
    manager.prepare_round(2, manifest2, dataset2, _entries(["left", "right"]))
    ref = SkillVersionRef(skill_id="offline-repair", version=1)
    for task in ("left", "right"):
        config = manager.task_config(2, task, f"run-{task}")
        registry = CapabilityRegistry(
            config.capability_dir, config.state_dir / "capabilities.sqlite"
        )
        try:
            registry.record_retrieval([ref], operation_key=f"retrieve:{task}")
            registry.record_retrieval([ref], selected=True, operation_key=f"select:{task}")
            registry.record_use(
                ref,
                success=True,
                tool_calls=1,
                attempts=1,
                patched=True,
                operation_key=f"use:{task}",
            )
        finally:
            registry.close()
    generation = manager.finalize_round(2, [_result("left"), _result("right")])
    registry = CapabilityRegistry(
        generation / "skills", generation / "state" / "capabilities.sqlite"
    )
    try:
        stats = registry.stats(ref.skill_id, ref.version)
        assert stats.retrieval_count == 2
        assert stats.selected_count == 2
        assert stats.use_count == 2
        assert stats.success_count == 2
        assert registry.get(ref.skill_id, ref.version).manifest.status == "active"  # type: ignore[union-attr]
    finally:
        registry.close()
