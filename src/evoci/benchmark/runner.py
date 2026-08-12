"""Fair ablation execution with machine-readable output artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from evoci.benchmark.models import (
    BenchmarkManifestEntry,
    BenchmarkResult,
    BenchmarkTaskStatus,
    BenchmarkVariant,
    RunMetrics,
)

RunTask = Callable[[BenchmarkManifestEntry, BenchmarkVariant], Awaitable[RunMetrics]]


class BenchmarkRunner:
    def __init__(self, run_task: RunTask) -> None:
        self.run_task = run_task

    async def run(
        self,
        entries: list[BenchmarkManifestEntry],
        *,
        variant: BenchmarkVariant,
        output_dir: Path,
    ) -> list[BenchmarkResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        runs_path = output_dir / "runs.jsonl"
        runs_path.write_text("", encoding="utf-8")
        results: list[BenchmarkResult] = []
        for entry in entries:
            if entry.skipped:
                result = BenchmarkResult(
                    task_id=entry.task_id,
                    variant=variant,
                    status="skipped",
                    category=entry.category,
                    skipped=True,
                    skip_reason=entry.skip_reason,
                )
            else:
                try:
                    metrics = await self.run_task(entry, variant)
                    task_status: BenchmarkTaskStatus
                    if metrics.benchmark_verification_status == "passed":
                        task_status = "resolved"
                    elif metrics.benchmark_verification_status == "failed":
                        task_status = "unresolved"
                    else:
                        task_status = "not_evaluable"
                    result = BenchmarkResult(
                        task_id=entry.task_id,
                        variant=variant,
                        status=task_status,
                        category=entry.category,
                        metrics=metrics,
                    )
                except Exception as exc:
                    result = BenchmarkResult(
                        task_id=entry.task_id,
                        variant=variant,
                        status="error",
                        category=entry.category,
                        error_type=type(exc).__name__,
                        error_message=str(exc) or type(exc).__name__,
                    )
            results.append(result)
            with runs_path.open("a", encoding="utf-8") as handle:
                handle.write(result.model_dump_json() + "\n")
        self._write_outputs(results, output_dir)
        return results

    @staticmethod
    def load_manifest(path: Path) -> list[BenchmarkManifestEntry]:
        return [
            BenchmarkManifestEntry.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    @staticmethod
    def _write_outputs(results: list[BenchmarkResult], output_dir: Path) -> None:
        (output_dir / "runs.jsonl").write_text(
            "".join(result.model_dump_json() + "\n" for result in results), encoding="utf-8"
        )
        completed = [result for result in results if result.metrics is not None]
        evaluable = [
            result
            for result in completed
            if result.metrics
            and result.metrics.benchmark_verification_status in {"passed", "failed"}
        ]
        resolved = sum(
            result.metrics is not None and result.metrics.benchmark_verification_status == "passed"
            for result in evaluable
        )
        not_available = sum(
            result.metrics is not None
            and result.metrics.benchmark_verification_status == "not_available"
            for result in completed
        )
        aggregate = {
            "tasks": len(results),
            "completed": len(completed),
            "skipped": sum(result.status == "skipped" for result in results),
            "errors": sum(result.status == "error" for result in results),
            "evaluable": len(evaluable),
            "not_available": not_available,
            "evaluation_coverage": (len(evaluable) / len(completed) if completed else None),
            "benchmark_resolved": resolved,
            "benchmark_success_rate": resolved / len(evaluable) if evaluable else None,
        }
        (output_dir / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (output_dir / "by_error_type.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "task_id",
                    "error_type",
                    "variant",
                    "benchmark_verification_status",
                    "benchmark_resolved",
                    "gold_file_overlap",
                    "tool_calls",
                ]
            )
            for result in completed:
                assert result.metrics is not None
                writer.writerow(
                    [
                        result.task_id,
                        result.category,
                        result.variant,
                        result.metrics.benchmark_verification_status,
                        result.metrics.benchmark_resolved,
                        result.metrics.gold_file_overlap,
                        result.metrics.tool_calls,
                    ]
                )
        with (output_dir / "continual_learning.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "task_index",
                    "task_family",
                    "benchmark_verification_status",
                    "benchmark_resolved",
                    "tool_calls",
                    "repair_attempts",
                    "memory_hits",
                    "skills_used",
                    "skill_registry_size",
                    "active_skill_count",
                ]
            )
            for task_index, result in enumerate(completed, 1):
                assert result.metrics is not None
                writer.writerow(
                    [
                        task_index,
                        result.category,
                        result.metrics.benchmark_verification_status,
                        result.metrics.benchmark_resolved,
                        result.metrics.tool_calls,
                        result.metrics.repair_attempts,
                        result.metrics.memory_retrieval_count,
                        result.metrics.skills_used,
                        result.metrics.skill_registry_size,
                        result.metrics.active_skill_count,
                    ]
                )
