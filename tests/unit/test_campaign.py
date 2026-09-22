import json
import os
import sqlite3
from pathlib import Path

import pytest

from evoci.benchmark.campaign import CampaignError, CampaignManager, tree_hash
from evoci.benchmark.models import (
    BenchmarkManifestEntry,
    BenchmarkResult,
    BenchmarkVerificationResult,
    FinalWorkspaceChanges,
    RunMetrics,
)
from evoci.capability.models import (
    GeneratedFile,
    SkillCandidate,
    SkillMemoryEntry,
    SkillPermissions,
)
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


def test_campaign_records_reasoning_effort_and_warns_if_it_changes(tmp_path: Path) -> None:
    config = EvoCIConfig(
        worker_enable_thinking=True,
        worker_reasoning_effort="medium",
        supervisor_enable_thinking=True,
        supervisor_reasoning_effort="xhigh",
    )
    manager = CampaignManager(tmp_path / "campaign", variant="evo", base_config=config)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))

    campaign = json.loads((manager.root / "campaign.json").read_text(encoding="utf-8"))
    assert campaign["model"]["worker"]["reasoning_effort"] == "medium"
    assert campaign["model"]["supervisor"]["reasoning_effort"] == "xhigh"

    changed = CampaignManager(
        tmp_path / "campaign",
        variant="evo",
        base_config=config.model_copy(update={"worker_reasoning_effort": "xhigh"}),
    )
    _, warnings = changed.prepare_round(1, manifest, dataset, _entries(["a"]))
    assert "critical model budget or campaign configuration changed on resume" in warnings


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
                failure_fingerprint=f"fp-{task}",
                attempted_fix_summaries=["try operator"] if not success else [],
                attempted_files=["app.py"] if not success else [],
            )
        )
    finally:
        store.close()


def _add_skill(config: EvoCIConfig, run_id: str) -> None:
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        registry.create_skill(
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
        merged = store.get_episode("run-a")
        assert merged is not None
        assert merged.failure_fingerprint == "fp-a"
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


def test_campaign_rejects_duplicate_task_ids_in_manifest(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    with pytest.raises(CampaignError, match="duplicate task_id"):
        manager.prepare_round(1, manifest, dataset, _entries(["a", "a"]))


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


def test_generation_hash_remains_stable_after_sqlite_wal_reopen(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    config = manager.task_config(1, "a", "run-a")
    _add_skill(config, "run-a")
    generation = manager.finalize_round(1, [_result("a")])
    expected = json.loads((generation / "metadata.json").read_text())["content_sha256"]

    registry = CapabilityRegistry(
        generation / "skills", generation / "state" / "capabilities.sqlite"
    )
    registry.close()

    assert tree_hash(generation / "state", generation / "skills") == expected
    manifest2, dataset2 = _files(tmp_path, ["b"])
    manager.prepare_round(2, manifest2, dataset2, _entries(["b"]))


def test_campaign_aggregates_frozen_branch_skill_usage(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["seed"])
    manager.prepare_round(1, manifest, dataset, _entries(["seed"]))
    seed = manager.task_config(1, "seed", "run-seed")
    _add_skill(seed, "run-seed")
    manager.finalize_round(1, [_result("seed")])

    manifest2, dataset2 = _files(tmp_path, ["left", "right"])
    manager.prepare_round(2, manifest2, dataset2, _entries(["left", "right"]))
    for task in ("left", "right"):
        config = manager.task_config(2, task, f"run-{task}")
        registry = CapabilityRegistry(
            config.capability_dir, config.state_dir / "capabilities.sqlite"
        )
        try:
            registry.record_retrieval(["offline-repair"], operation_key=f"retrieve:{task}")
            registry.record_retrieval(
                ["offline-repair"], selected=True, operation_key=f"select:{task}"
            )
            registry.record_use(
                "offline-repair",
                success=True,
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
        stats = registry.stats("offline-repair")
        assert stats.retrieval_count == 2
        assert stats.selected_count == 2
        assert stats.use_count == 2
        assert stats.success_count == 2
        record = registry.get("offline-repair")
        assert record is not None and record.manifest.enabled
    finally:
        registry.close()


def test_campaign_skills_are_not_written_to_default_evoci_state(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    config = manager.task_config(1, "a", "run-a")
    _add_skill(config, "run-a")
    manager.finalize_round(1, [_result("a")])
    assert not (tmp_path / ".evoci" / "skills").exists()
    assert list((manager.generations / "generation-1" / "skills").glob("*/package"))


def test_round_reads_only_parent_generation(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a", "b"])
    manager.prepare_round(1, manifest, dataset, _entries(["a", "b"]))
    a = manager.task_config(1, "a", "run-a")
    _add_skill(a, "run-a")
    b = manager.task_config(1, "b", "run-b")
    b_registry = CapabilityRegistry(b.capability_dir, b.state_dir / "capabilities.sqlite")
    try:
        assert b_registry.list() == []
    finally:
        b_registry.close()
    manager.finalize_round(1, [_result("a"), _result("b")])


def test_round_merges_new_skill(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a", "b"])
    manager.prepare_round(1, manifest, dataset, _entries(["a", "b"]))
    _add_skill(manager.task_config(1, "a", "run-a"), "run-a")
    generation = manager.finalize_round(1, [_result("a"), _result("b")])
    registry = CapabilityRegistry(
        generation / "skills", generation / "state" / "capabilities.sqlite"
    )
    try:
        record = registry.get("offline-repair")
        assert record is not None
        package = Path(record.package_path)
        assert package.is_dir()
        assert package == (generation / "skills" / "offline-repair" / "package").resolve()
        assert ".tmp" not in record.package_path
        assert (generation / "skills" / "offline-repair" / "package").is_dir()
        assert not list((generation / "skills").glob("*-variant-*"))
    finally:
        registry.close()


def test_round_merges_skill_memory_by_run_marker(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["seed"])
    manager.prepare_round(1, manifest, dataset, _entries(["seed"]))
    seed = manager.task_config(1, "seed", "run-seed")
    _add_skill(seed, "run-seed")
    manager.finalize_round(1, [_result("seed")])

    manifest2, dataset2 = _files(tmp_path, ["left", "right"])
    manager.prepare_round(2, manifest2, dataset2, _entries(["left", "right"]))
    from datetime import UTC, datetime

    for task_id in ("left", "right"):
        config = manager.task_config(2, task_id, f"run-{task_id}")
        registry = CapabilityRegistry(
            config.capability_dir, config.state_dir / "capabilities.sqlite"
        )
        try:
            registry.append_skill_memory(
                "offline-repair",
                SkillMemoryEntry(
                    run_id=f"run-{task_id}",
                    repository=f"org/{task_id}",
                    task_summary=f"task {task_id}",
                    outcome="success",
                    lesson=f"lesson from {task_id}",
                    created_at=datetime.now(UTC),
                ),
            )
        finally:
            registry.close()
    generation = manager.finalize_round(2, [_result("left"), _result("right")])
    memory = (generation / "skills" / "offline-repair" / "memory.md").read_text(encoding="utf-8")
    assert "<!-- evoci-skill-memory:run-left:offline-repair -->" in memory
    assert "<!-- evoci-skill-memory:run-right:offline-repair -->" in memory


def test_same_round_skill_updates_use_stable_first_writer(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["seed"])
    manager.prepare_round(1, manifest, dataset, _entries(["seed"]))
    seed = manager.task_config(1, "seed", "run-seed")
    _add_skill(seed, "run-seed")
    manager.finalize_round(1, [_result("seed")])

    manifest2, dataset2 = _files(tmp_path, ["left", "right"])
    manager.prepare_round(2, manifest2, dataset2, _entries(["left", "right"]))
    left = manager.task_config(2, "left", "run-left")
    right = manager.task_config(2, "right", "run-right")
    left_registry = CapabilityRegistry(left.capability_dir, left.state_dir / "capabilities.sqlite")
    right_registry = CapabilityRegistry(
        right.capability_dir, right.state_dir / "capabilities.sqlite"
    )
    skill_md = (
        "---\nname: Offline repair\ndescription: reusable\n---\n"
        "# Purpose\nrepair\n# When to Use\noffline\n# Procedure\nfix\n"
        "# Pitfalls\nnone\n# Verification\ntest\n# Bundled Resources\nnone\n"
    )
    try:
        left_registry.update_skill(
            "offline-repair",
            SkillCandidate(
                name="Offline repair",
                description="left update",
                triggers=["offline"],
                task_families=["test"],
                skill_md=skill_md.replace("fix", "left procedure"),
                references=[
                    GeneratedFile(path="references/notes.md", content="left-only resource\n")
                ],
                source_run_ids=["run-left"],
                confidence=0.9,
            ),
        )
        right_registry.update_skill(
            "offline-repair",
            SkillCandidate(
                name="Offline repair",
                description="right update",
                triggers=["offline"],
                task_families=["test"],
                skill_md=skill_md.replace("fix", "right procedure"),
                source_run_ids=["run-right"],
                confidence=0.9,
            ),
        )
    finally:
        left_registry.close()
        right_registry.close()
    generation = manager.finalize_round(2, [_result("left"), _result("right")])
    package_dir = generation / "skills" / "offline-repair" / "package"
    package = (package_dir / "SKILL.md").read_text(encoding="utf-8")
    assert "left procedure" in package
    assert "right procedure" not in package
    registry = CapabilityRegistry(
        generation / "skills", generation / "state" / "capabilities.sqlite"
    )
    try:
        record = registry.get("offline-repair")
        assert record is not None
        assert record.manifest.description == "left update"
        assert "run-left" in record.manifest.source_run_ids
        assert "run-right" not in record.manifest.source_run_ids
        on_disk = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
        assert json.loads(record.manifest.model_dump_json()) == on_disk
    finally:
        registry.close()
    manifest3, dataset3 = _files(tmp_path, ["next"])
    manager.prepare_round(3, manifest3, dataset3, _entries(["next"]))
    nxt = manager.task_config(3, "next", "run-next")
    notes = nxt.capability_dir / "offline-repair" / "package" / "references" / "notes.md"
    assert notes.read_text(encoding="utf-8") == "left-only resource\n"


def test_campaign_merge_does_not_create_variant_skill(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a", "b"])
    manager.prepare_round(1, manifest, dataset, _entries(["a", "b"]))
    _add_skill(manager.task_config(1, "a", "run-a"), "run-a")
    _add_skill(manager.task_config(1, "b", "run-b"), "run-b")
    generation = manager.finalize_round(1, [_result("a"), _result("b")])
    names = [path.name for path in (generation / "skills").iterdir() if path.is_dir()]
    assert names == ["offline-repair"]
    assert not any("variant" in name for name in names)


def test_generation_commit_records_final_paths_before_atomic_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    _add_skill(manager.task_config(1, "a", "run-a"), "run-a")
    original_replace = os.replace
    seen: list[tuple[Path, Path, str]] = []

    def spy_replace(source: str | os.PathLike[str], dest: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        dest_path = Path(dest)
        if dest_path.name == "generation-1":
            database = source_path / "state" / "capabilities.sqlite"
            row = sqlite3.connect(database).execute(
                "SELECT package_path FROM skills WHERE skill_id=?", ("offline-repair",)
            ).fetchone()
            assert row is not None
            expected = dest_path / "skills" / "offline-repair" / "package"
            assert row[0] == str(expected)
            assert (source_path / "skills" / "offline-repair" / "package").is_dir()
            assert not dest_path.exists()
            seen.append((source_path, dest_path, row[0]))
        original_replace(source, dest)

    monkeypatch.setattr(os, "replace", spy_replace)
    generation = manager.finalize_round(1, [_result("a")])
    assert seen
    assert Path(seen[0][2]).is_dir()
    assert Path(seen[0][2]) == generation / "skills" / "offline-repair" / "package"


def test_failed_generation_rename_leaves_no_incomplete_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    _add_skill(manager.task_config(1, "a", "run-a"), "run-a")
    original_replace = os.replace

    def boom(source: str | os.PathLike[str], dest: str | os.PathLike[str]) -> None:
        dest_path = Path(dest)
        if dest_path.name == "generation-1":
            raise OSError("simulated generation rename failure")
        original_replace(source, dest)

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated generation rename failure"):
        manager.finalize_round(1, [_result("a")])
    assert not (manager.generations / "generation-1").exists()
    assert not list(manager.generations.glob("generation-1"))


def test_failed_generation_commit_preserves_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    parent = manager.generations / "generation-0"
    parent_hash = json.loads((parent / "metadata.json").read_text())["content_sha256"]
    parent_tree = tree_hash(parent / "state", parent / "skills")
    original_replace = os.replace

    def boom(source: str | os.PathLike[str], dest: str | os.PathLike[str]) -> None:
        dest_path = Path(dest)
        if dest_path.name == "generation-1":
            raise OSError("simulated generation rename failure")
        original_replace(source, dest)

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated generation rename failure"):
        manager.finalize_round(1, [_result("a")])
    assert json.loads((parent / "metadata.json").read_text())["content_sha256"] == parent_hash
    assert tree_hash(parent / "state", parent / "skills") == parent_tree
    assert not (manager.generations / "generation-1").exists()


def test_campaign_summary_reports_token_deltas(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manifest, dataset = _files(tmp_path, ["a"])
    manager.prepare_round(1, manifest, dataset, _entries(["a"]))
    manager.finalize_round(1, [_result("a")])
    round1 = manager.rounds / "round-1" / "aggregate.json"
    round1.write_text(
        json.dumps(
            {
                "benchmark_success_rate": 0.5,
                "total_tokens": 100,
                "repair_tokens": 80,
                "learning_tokens": 20,
                "tokens_per_resolved_task": 200,
                "provider_requests": 10,
            }
        )
    )
    manifest2, dataset2 = _files(tmp_path, ["b"])
    manager.prepare_round(2, manifest2, dataset2, _entries(["b"]))
    manager.finalize_round(2, [_result("b")])
    (manager.rounds / "round-2" / "aggregate.json").write_text(
        json.dumps(
            {
                "benchmark_success_rate": 0.75,
                "total_tokens": 130,
                "repair_tokens": 90,
                "learning_tokens": 40,
                "tokens_per_resolved_task": 173.33,
                "provider_requests": 12,
            }
        )
    )
    manager.write_summary()
    summary = json.loads((manager.root / "campaign-summary.json").read_text())
    delta = summary["comparisons"][0]
    assert delta["success_rate_delta"] == pytest.approx(0.25)
    assert delta["total_tokens_delta"] == 30
    assert delta["repair_tokens_delta"] == 10
    assert delta["learning_tokens_delta"] == 20
    assert delta["tokens_per_resolved_task_delta"] == pytest.approx(-26.67)
    assert delta["provider_requests_delta"] == 2
