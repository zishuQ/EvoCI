import json
from pathlib import Path

root = Path("campaign13")
tasks = [
    json.loads(line)["task_id"]
    for line in (root / "prepared/dataset.jsonl").read_text().splitlines()
]
(root / "environments").mkdir(parents=True, exist_ok=True)
rows = []
for task in tasks:
    directory = root / "environments" / task
    directory.mkdir(parents=True, exist_ok=True)
    rows.append({"task_id": task, "environment": str(directory / ".venv"), "status": "pending-install"})
(root / "environments.json").write_text(json.dumps({"uv_cache_dir": "/tmp/evoci-uv-cache", "python": "3.12", "tasks": rows}, indent=2) + "\n")
print(json.dumps({"tasks": len(tasks), "manifest": "campaign13/environments.json"}))
