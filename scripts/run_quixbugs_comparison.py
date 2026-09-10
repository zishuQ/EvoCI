"""Run equal-task online comparisons with real configured models, never reference patches."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evoci.config import EvoCIConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    config = EvoCIConfig.from_env()
    if not config.model_name or not config.model_api_key:
        raise SystemExit("Configure EVO_MODEL_NAME and EVO_MODEL_API_KEY before model evaluation.")
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.output.exists():
        raise SystemExit("Use a fresh --output directory to keep evaluation runs independent.")
    prepared = args.prepared.resolve()
    entries = [json.loads(line) for line in (prepared / "manifest.jsonl").read_text().splitlines()]
    entries = entries[: args.limit]
    if not entries:
        raise SystemExit("Manifest contains no tasks.")
    args.output.mkdir(parents=True)
    manifest = args.output.resolve() / "comparison-manifest.jsonl"
    manifest.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    summary = {
        "protocol": "online_continual_same_order_fresh_state_per_variant",
        "model": config.model_name,
        "model_fast": config.fast_model_name,
        "model_strong": config.strong_model_name,
        "model_aux": config.aux_model_name,
        "max_run_model_calls": config.max_run_model_calls,
        "max_run_tool_calls": config.max_run_tool_calls,
        "task_ids": [entry["task_id"] for entry in entries],
        "variants": {},
    }
    for variant in ("multi", "multi-memory", "evo"):
        output = args.output.resolve() / variant
        invocation = [
            sys.executable,
            "-m",
            "evoci.cli",
            "benchmark",
            "--manifest",
            str(manifest),
            "--dataset",
            str(prepared / "dataset.jsonl"),
            "--variant",
            variant,
            "--continual",
            "--output-dir",
            str(output),
        ]
        print(f"Running {variant}: {len(entries)} tasks", flush=True)
        result = subprocess.run(invocation, check=False)
        if result.returncode:
            summary["variants"][variant] = {"process_exit_code": result.returncode}
        else:
            aggregate = json.loads((output / "aggregate.json").read_text())
            runs = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
            metrics = [row["metrics"] for row in runs if row.get("metrics")]
            aggregate.update(
                {
                    "resolved_over_all_selected": aggregate["benchmark_resolved"] / len(entries),
                    "tool_calls_total": sum(row.get("tool_calls", 0) for row in metrics),
                    "llm_calls_total": sum(row.get("llm_calls", 0) for row in metrics),
                    "skill_uses_total": sum(row.get("skills_used", 0) for row in metrics),
                    "skill_candidates_created": sum(row.get("skill_created", 0) for row in metrics),
                    "token_cost": None,
                }
            )
            summary["variants"][variant] = aggregate
        (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
