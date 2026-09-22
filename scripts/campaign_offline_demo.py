"""Run a deterministic three-process EvoCI campaign without model or network access."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from evoci.benchmark.campaign import CampaignManager
from evoci.benchmark.models import (
    BenchmarkManifestEntry,
    BenchmarkResult,
    BenchmarkVerificationResult,
    FinalWorkspaceChanges,
    RunMetrics,
)
from evoci.capability.models import SkillCandidate
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.config import EvoCIConfig
from evoci.domain.models import CIFailure, RepoSpec
from evoci.memory.models import Episode
from evoci.memory.store import SQLiteMemoryStore


def _write_inputs(root: Path, round_number: int) -> tuple[Path, Path, list[str]]:
    task_ids = [f"round-{round_number}-task-a", f"round-{round_number}-task-b"]
    manifest = root / f"round-{round_number}.jsonl"
    dataset = root / f"dataset-{round_number}.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"task_id": task_id, "category": "offline-repair"}) + "\n"
            for task_id in task_ids
        ),
        encoding="utf-8",
    )
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "repo": f"offline/{task_id}",
                    "sha_fail": f"failing-{round_number}-{index}",
                }
            )
            + "\n"
            for index, task_id in enumerate(task_ids, 1)
        ),
        encoding="utf-8",
    )
    return manifest, dataset, task_ids


def _count_rows(database: Path, table: str) -> int:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


def _result(task_id: str) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=task_id,
        variant="evo",
        status="resolved",
        category="offline-repair",
        metrics=RunMetrics(
            agent_declared_success=True,
            targeted_verification_passed=True,
            review_passed=True,
            benchmark_verification=BenchmarkVerificationResult(
                status="passed", details="deterministic offline oracle passed"
            ),
            benchmark_verification_status="passed",
            benchmark_resolved=True,
            final_workspace_changes=FinalWorkspaceChanges(changed_files=[]),
            wall_time=0.0,
        ),
    )


def _seed_skill(config: EvoCIConfig, run_id: str) -> None:
    registry = CapabilityRegistry(config.capability_dir, config.state_dir / "capabilities.sqlite")
    try:
        created = registry.create_skill(
            SkillCandidate(
                name="Offline repair procedure",
                description="Reusable offline repair procedure",
                triggers=["offline", "repair"],
                task_families=["offline-repair"],
                skill_md=(
                    "---\nname: Offline repair procedure\n"
                    "description: Reusable offline repair procedure\n---\n"
                    "# Purpose\nRepair deterministic offline failures.\n"
                    "# When to Use\nUse for offline repair tasks.\n"
                    "# Procedure\nInspect, patch, and verify.\n"
                    "# Pitfalls\nDo not use evaluator-private data.\n"
                    "# Verification\nRun the independent oracle.\n"
                    "# Bundled Resources\nNo bundled files are required.\n"
                ),
                source_run_ids=[run_id],
                confidence=0.95,
            ),
            operation_key=f"offline-demo:{run_id}:skill",
        )
        del created
    finally:
        registry.close()


def run_round(root: Path, round_number: int) -> dict[str, object]:
    campaign_dir = root / "campaign"
    config = EvoCIConfig.from_env(cwd=root)
    manager = CampaignManager(campaign_dir, variant="evo", base_config=config)
    manifest, dataset, task_ids = _write_inputs(root, round_number)
    entries = [
        BenchmarkManifestEntry(task_id=task_id, category="offline-repair") for task_id in task_ids
    ]
    round_dir, _ = manager.prepare_round(round_number, manifest, dataset, entries)
    observations: list[dict[str, object]] = []
    for index, task_id in enumerate(task_ids):
        run_id = f"offline-round-{round_number}-{task_id}"
        branch = manager.task_config(round_number, task_id, run_id)
        episodes_before = _count_rows(branch.state_dir / "memory.sqlite", "episodes")
        skills_before = _count_rows(branch.state_dir / "capabilities.sqlite", "skills")
        retrieved_skills: list[str] = []
        if round_number >= 2:
            registry = CapabilityRegistry(
                branch.capability_dir, branch.state_dir / "capabilities.sqlite"
            )
            try:
                catalog = CapabilityRetriever(registry).retrieve(
                    RepoSpec(owner="offline", name=task_id),
                    CIFailure(
                        summary="offline repair failure",
                        log_excerpt="offline repair assertion",
                        task_family="offline-repair",
                    ),
                    operation_key=f"offline-demo:retrieve:{run_id}",
                )
                retrieved_skills = [entry.skill_id for entry in catalog.entries]
            finally:
                registry.close()
        store = SQLiteMemoryStore(branch.state_dir / "memory.sqlite")
        try:
            store.add_episode(
                Episode(
                    id=f"episode:{run_id}",
                    run_id=run_id,
                    repo=f"offline/{task_id}",
                    task_family="offline-repair",
                    failure_summary="deterministic offline failure",
                    attempts=1,
                    successful_fix_summary="deterministic repair",
                    success=True,
                )
            )
        finally:
            store.close()
        if round_number == 1 and index == 0:
            _seed_skill(branch, run_id)
        observations.append(
            {
                "task_id": task_id,
                "read_generation": round_number - 1,
                "episodes_before_learning": episodes_before,
                "skills_before_learning": skills_before,
                "retrieved_skills": retrieved_skills,
            }
        )
    generation = manager.finalize_round(round_number, [_result(task_id) for task_id in task_ids])
    report = {
        "round": round_number,
        "process_id": __import__("os").getpid(),
        "parent_generation": round_number - 1,
        "committed_generation": round_number,
        "generation_path": str(generation),
        "observations": observations,
    }
    (round_dir / "offline-demo.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def orchestrate(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    for round_number in (1, 2, 3):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(root),
                "--worker",
                str(round_number),
            ],
            check=True,
        )
        report_path = root / "campaign" / "rounds" / f"round-{round_number}" / "offline-demo.json"
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    summary = json.loads((root / "campaign" / "campaign-summary.json").read_text(encoding="utf-8"))
    result = {
        "campaign_dir": str(root / "campaign"),
        "worker_process_ids": [report["process_id"] for report in reports],
        "rounds": reports,
        "summary_rounds": [item["round"] for item in summary["rounds"]],
        "summary_comparisons": len(summary["comparisons"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--worker", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    if args.worker is not None:
        run_round(args.output_dir.resolve(), args.worker)
    else:
        orchestrate(args.output_dir.resolve())


if __name__ == "__main__":
    main()
