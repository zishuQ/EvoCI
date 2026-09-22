"""Clone Campaign-13 tasks and prepare isolated environments."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path("campaign13").resolve()
DATASET = ROOT / "prepared" / "dataset.jsonl"
WORKSPACES = ROOT / "workspaces"
ENVS = ROOT / "environments"
REPORT = ROOT / "environments.json"


def run(args: list[str], *, cwd: Path | None = None, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/evoci-uv-cache")
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)


def python_install(repo: Path, interpreter: Path) -> tuple[str, str]:
    commands: list[list[str]] = []
    if (repo / "uv.lock").is_file() or (repo / "pyproject.toml").is_file():
        commands.append(["uv", "sync", "--active"])
    if (repo / "requirements.txt").is_file():
        commands.append(["uv", "pip", "install", "--python", str(interpreter), "-r", "requirements.txt"])
    if (repo / "setup.py").is_file() or (repo / "setup.cfg").is_file():
        commands.append(["uv", "pip", "install", "--python", str(interpreter), "-e", "."])
    if not commands:
        return "no-dependency-file", "no pyproject/requirements/setup file found"
    for command in commands:
        if command[:2] == ["uv", "sync"]:
            command = ["uv", "sync", "--project", str(repo), "--python", str(interpreter)]
        result = run(command, cwd=repo, timeout=1800)
        if result.returncode != 0:
            return "install-failed", (result.stderr or result.stdout)[-2000:]
    return "installed", ""


def main() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
    report: dict[str, object] = {"root": str(ROOT), "tasks": []}
    for row in rows:
        task_id = str(row["task_id"])
        task_dir = WORKSPACES / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = {"task_id": task_id, "repo": row.get("repo_name")}
        if task_id == "quixbugs-rpn_eval":
            record.update({"status": "installed", "environment": str(ENVS / task_id / ".venv")})
            report["tasks"].append(record)
            continue
        if row.get("repo_name") == "composer":
            record.update({"status": "blocked", "reason": "php/composer runtime is not installed"})
            report["tasks"].append(record)
            continue
        repo = task_dir / "repo"
        if not (repo / ".git").is_dir():
            clone = run(["git", "clone", f"https://github.com/{row['repo_owner']}/{row['repo_name']}.git", str(repo)], timeout=1800)
            if clone.returncode != 0:
                record.update({"status": "clone-failed", "reason": (clone.stderr or clone.stdout)[-2000:]})
                report["tasks"].append(record)
                continue
        checkout = run(["git", "checkout", "--force", str(row["sha_fail"])], cwd=repo)
        if checkout.returncode != 0:
            record.update({"status": "checkout-failed", "reason": (checkout.stderr or checkout.stdout)[-2000:]})
            report["tasks"].append(record)
            continue
        interpreter = ENVS / task_id / ".venv" / "bin" / "python"
        if not interpreter.exists():
            venv = run(["uv", "venv", str(interpreter.parent.parent), "--python", "3.12"], cwd=ROOT)
            if venv.returncode != 0:
                record.update({"status": "venv-failed", "reason": (venv.stderr or venv.stdout)[-2000:]})
                report["tasks"].append(record)
                continue
        status, reason = python_install(repo, interpreter)
        record.update({"status": status, "environment": str(interpreter), "reason": reason})
        report["tasks"].append(record)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
