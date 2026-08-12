"""Balanced, reproducibility-aware MVP manifest generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from evoci.benchmark.adapters import BenchmarkAdapter
from evoci.benchmark.models import BenchmarkManifestEntry

DEFAULT_CATEGORIES = (
    "Formatting",
    "Linting",
    "Type Checking",
    "Test Failure",
    "Dependency",
    "Configuration",
)


def generate_balanced_manifest(
    adapter: BenchmarkAdapter,
    output_path: Path,
    *,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    per_category: int = 5,
) -> list[BenchmarkManifestEntry]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for meta in adapter.list_tasks():
        error_type = adapter.ground_truth(meta.task_id).error_type
        if error_type in categories:
            buckets[error_type].append(meta.task_id)
    entries: list[BenchmarkManifestEntry] = []
    for category in categories:
        selected = buckets[category][:per_category]
        entries.extend(
            BenchmarkManifestEntry(task_id=task_id, category=category) for task_id in selected
        )
        for slot in range(len(selected) + 1, per_category + 1):
            entries.append(
                BenchmarkManifestEntry(
                    task_id=f"select-{category.lower().replace(' ', '-')}-{slot:02d}",
                    category=category,
                    skipped=True,
                    skip_reason=(
                        "No additional reproducible task of this category exists in the local "
                        "export."
                    ),
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(entry.model_dump_json() + "\n" for entry in entries), encoding="utf-8"
    )
    return entries
