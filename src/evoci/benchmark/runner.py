"""Fair ablation execution with machine-readable output artifacts."""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

from evoci.benchmark.models import (
    BenchmarkManifestEntry,
    BenchmarkResult,
    BenchmarkTaskStatus,
    BenchmarkVariant,
    RunMetrics,
)

RunTask = Callable[[BenchmarkManifestEntry, BenchmarkVariant], Awaitable[RunMetrics]]


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return (values[middle - 1] + values[middle]) / 2


class BenchmarkRunner:
    def __init__(self, run_task: RunTask) -> None:
        self.run_task = run_task

    async def run(
        self,
        entries: list[BenchmarkManifestEntry],
        *,
        variant: BenchmarkVariant,
        output_dir: Path,
        resume: bool = False,
        parallelism: int = 1,
    ) -> list[BenchmarkResult]:
        if parallelism < 1:
            raise ValueError("parallelism must be >= 1")
        output_dir.mkdir(parents=True, exist_ok=True)
        runs_path = output_dir / "runs.jsonl"
        prior: dict[str, BenchmarkResult] = {}
        if resume and runs_path.is_file():
            for line in runs_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    result = BenchmarkResult.model_validate_json(line)
                    if result.status not in {"error", "infra_error"}:
                        prior[result.task_id] = result
        else:
            runs_path.write_text("", encoding="utf-8")
        results_by_id: dict[str, BenchmarkResult] = {}
        pending: list[BenchmarkManifestEntry] = []
        for entry in entries:
            if entry.task_id in prior:
                results_by_id[entry.task_id] = prior[entry.task_id]
            elif entry.skipped:
                results_by_id[entry.task_id] = BenchmarkResult(
                    task_id=entry.task_id,
                    variant=variant,
                    status="skipped",
                    category=entry.category,
                    skipped=True,
                    skip_reason=entry.skip_reason,
                )
            else:
                pending.append(entry)

        semaphore = asyncio.Semaphore(parallelism)

        async def execute(entry: BenchmarkManifestEntry) -> BenchmarkResult:
            async with semaphore:
                try:
                    metrics = await self.run_task(entry, variant)
                    status = cast(
                        BenchmarkTaskStatus,
                        {
                            "passed": "resolved",
                            "failed": "unresolved",
                            "not_available": "not_evaluable",
                            "infra_error": "infra_error",
                        }[metrics.benchmark_verification_status],
                    )
                    return BenchmarkResult(
                        task_id=entry.task_id,
                        variant=variant,
                        status=status,
                        category=entry.category,
                        metrics=metrics,
                    )
                except Exception as exc:
                    return BenchmarkResult(
                        task_id=entry.task_id,
                        variant=variant,
                        status="error",
                        category=entry.category,
                        error_type=type(exc).__name__,
                        error_message=str(exc) or type(exc).__name__,
                    )

        tasks = [asyncio.create_task(execute(entry)) for entry in pending]
        for task in asyncio.as_completed(tasks):
            result = await task
            results_by_id[result.task_id] = result
            with runs_path.open("a", encoding="utf-8") as handle:
                handle.write(result.model_dump_json() + "\n")
        results = [results_by_id[entry.task_id] for entry in entries]
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
        infra_errors = sum(
            result.metrics is not None
            and result.metrics.benchmark_verification_status == "infra_error"
            for result in completed
        )
        aggregate = {
            "tasks": len(results),
            "completed": len(completed),
            "skipped": sum(result.status == "skipped" for result in results),
            "errors": sum(result.status == "error" for result in results),
            "evaluable": len(evaluable),
            "not_available": not_available,
            "infra_errors": infra_errors,
            "evaluation_coverage": (len(evaluable) / len(completed) if completed else None),
            "benchmark_resolved": resolved,
            "benchmark_success_rate": resolved / len(evaluable) if evaluable else None,
        }

        def total(field: str) -> int:
            return sum(
                int(getattr(result.metrics, field)) for result in completed if result.metrics
            )

        def median(field: str) -> float | None:
            values = sorted(
                int(getattr(result.metrics, field)) for result in completed if result.metrics
            )
            return _median(values)

        resolved_token_values = sorted(
            int(result.metrics.total_tokens)
            for result in completed
            if (
                result.metrics is not None
                and result.metrics.benchmark_verification_status == "passed"
            )
        )
        total_tokens = total("total_tokens")
        learning_tokens = total("learning_tokens")
        aggregate.update(
            {
                "selected_tasks": len(results),
                "completed_tasks": len(completed),
                "evaluable_tasks": len(evaluable),
                "total_model_calls": total("llm_calls"),
                "median_model_calls": median("llm_calls"),
                "total_tool_calls": total("tool_calls"),
                "median_tool_calls": median("tool_calls"),
                "total_attempts": total("repair_attempts"),
                "median_attempts": median("repair_attempts"),
                "provider_requests": total("provider_requests"),
                "input_tokens": total("input_tokens"),
                "output_tokens": total("output_tokens"),
                "total_tokens": total_tokens,
                "repair_tokens": total("repair_tokens"),
                "learning_tokens": learning_tokens,
                "learning_overhead_ratio": (
                    learning_tokens / total_tokens if total_tokens else None
                ),
                "median_tokens_per_task": median("total_tokens"),
                "median_tokens_per_resolved_task": _median(resolved_token_values),
                "wall_time": sum(
                    result.metrics.wall_time for result in completed if result.metrics
                ),
                "memory_selected": total("memory_selected_count"),
                "memory_used": total("memory_used_count"),
                "skills_retrieved": total("skills_retrieved"),
                "skills_selected": total("skills_selected"),
                "skills_used": total("skills_used"),
                "skills_created": total("skill_created"),
                "skills_updated": total("skill_updated"),
                "skills_promoted": total("skills_promoted"),
                "skills_rejected": total("skills_rejected"),
                "skills_superseded": total("skills_superseded"),
                "registry_size": max(
                    (result.metrics.skill_registry_size for result in completed if result.metrics),
                    default=0,
                ),
                "active_skill_count": max(
                    (result.metrics.active_skill_count for result in completed if result.metrics),
                    default=0,
                ),
                "tokens_per_resolved_task": (
                    total_tokens / resolved if resolved else None
                ),
                "tool_calls_per_resolved_task": total("tool_calls") / resolved
                if resolved
                else None,
                "attempts_per_resolved_task": total("repair_attempts") / resolved
                if resolved
                else None,
            }
        )
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
