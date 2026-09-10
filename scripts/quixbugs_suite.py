"""Prepare or validate the bundled QuixBugs subset; never feed reference fixes to agents."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic

from evoci.tools.shell import run_grouped_subprocess


def read_tasks(data: Path) -> list[dict]:
    return json.loads((data / "tasks.json").read_text())


def command(task: dict) -> list[str]:
    return ["python", "-m", "pytest", f"python_testcases/test_{task['name']}.py", "-q"]


def prepare(data: Path, output: Path, split: str) -> None:
    if output.exists():
        raise SystemExit(f"Output already exists: {output}; choose a fresh directory.")
    output.mkdir(parents=True)
    rows = []
    entries = []
    for task in read_tasks(data):
        if split != "all" and task["split"] != split:
            continue
        name = task["name"]
        workspace = output / "workspaces" / name
        shutil.copytree(data / "tasks" / name, workspace)
        for args in (
            ["init", "-q"],
            ["add", "."],
            [
                "-c",
                "user.name=EvoCI benchmark",
                "-c",
                "user.email=benchmark@example.invalid",
                "commit",
                "-qm",
                "QuixBugs defective baseline",
            ],
        ):
            subprocess.run(["git", "-C", str(workspace), *args], check=True, capture_output=True)
        sha = subprocess.check_output(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True
        ).strip()
        task_id = f"quixbugs-{name}"
        rows.append(
            {
                "task_id": task_id,
                "repo_owner": "quixbugs",
                "repo_name": f"QuixBugs-{name}",
                "sha_fail": sha,
                "sha_success": "reference-outside-agent-workspace",
                "workflow_name": "QuixBugs pytest",
                "workflow_path": ".github/workflows/tests.yml",
                "workflow": "",
                "workspace_path": str(workspace.resolve()),
                "logs": [
                    {
                        "step_name": "pytest",
                        "command": command(task),
                        "log": f"Repair {name}. Run the original tests.",
                    }
                ],
                "error_type": task["family"],
                "changed_files": [f"python_programs/{name}.py"],
                "diff": "",
                "source_dataset": "QuixBugs",
                "split": task["split"],
            }
        )
        entries.append({"task_id": task_id, "category": task["family"], "source": "QuixBugs"})
    for filename, items in (("dataset.jsonl", rows), ("manifest.jsonl", entries)):
        (output / filename).write_text("".join(json.dumps(x) + "\n" for x in items))
    print(json.dumps({"prepared": len(rows), "output": str(output.resolve())}))


def verify_one(data: Path, task: dict, mode: str, timeout: float) -> dict:
    started = monotonic()
    with tempfile.TemporaryDirectory(prefix="evoci-quixbugs-") as directory:
        workspace = Path(directory) / "workspace"
        shutil.copytree(data / "tasks" / task["name"], workspace)
        if mode == "reference":
            shutil.copy2(
                data / "reference" / f"{task['name']}.py",
                workspace / "python_programs" / f"{task['name']}.py",
            )
        argv = command(task)
        argv[0] = sys.executable
        result = run_grouped_subprocess(
            argv,
            cwd=workspace,
            timeout=timeout,
            max_chars=16000,
            env={
                "PATH": os.environ["PATH"],
                "PYTHONPATH": str(workspace),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        return {
            "task_id": f"quixbugs-{task['name']}",
            "mode": mode,
            "family": task["family"],
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "passed": result.exit_code == 0 and not result.timed_out,
            "seconds": round(monotonic() - started, 4),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def verify(data: Path, output: Path, timeout: float, jobs: int, split: str) -> None:
    if output.exists():
        raise SystemExit(f"Output already exists: {output}; choose a fresh directory.")
    output.mkdir(parents=True)
    tasks = [task for task in read_tasks(data) if split == "all" or task["split"] == split]
    work = [(task, mode) for mode in ("baseline", "reference") for task in tasks]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda item: verify_one(data, *item, timeout), work))
    (output / "runs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in results))
    summary = {
        "tasks": len(tasks),
        "split": split,
        "evaluation": "dataset_validation_not_agent_repair",
        "timeout_seconds": timeout,
        "real_model_calls": 0,
        "agent_success_rate": None,
        "self_evolution_gain": None,
    }
    for mode in ("baseline", "reference"):
        rows = [x for x in results if x["mode"] == mode]
        summary[mode] = {
            "tasks": len(rows),
            "passed": sum(x["passed"] for x in rows),
            "timed_out": sum(x["timed_out"] for x in rows),
            "failures": sum(not x["passed"] for x in rows),
            "pass_rate": sum(x["passed"] for x in rows) / len(rows),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify"])
    parser.add_argument(
        "--data", type=Path, required=True, help="Bundled datasets/quixbugs directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["all", "dev", "evaluation"], default="all")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.timeout <= 0 or args.jobs < 1:
        parser.error("--timeout and --jobs must be positive")
    if args.action == "prepare":
        prepare(args.data.resolve(), args.output.resolve(), args.split)
    else:
        verify(args.data.resolve(), args.output.resolve(), args.timeout, args.jobs, args.split)


if __name__ == "__main__":
    main()
