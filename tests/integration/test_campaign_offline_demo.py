import json
import subprocess
import sys
from pathlib import Path


def test_offline_campaign_runs_three_independent_processes(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "scripts" / "campaign_offline_demo.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "demo")],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert len(set(report["worker_process_ids"])) == 3
    assert report["summary_rounds"] == [1, 2, 3]
    assert report["summary_comparisons"] == 2

    round_one = report["rounds"][0]
    assert all(
        item["read_generation"] == 0
        and item["episodes_before_learning"] == 0
        and item["skills_before_learning"] == 0
        for item in round_one["observations"]
    )
    round_two = report["rounds"][1]
    assert all(item["read_generation"] == 1 for item in round_two["observations"])
    assert all(item["episodes_before_learning"] == 2 for item in round_two["observations"])
    assert all(item["retrieved_skills"] for item in round_two["observations"])
    round_three = report["rounds"][2]
    assert all(item["read_generation"] == 2 for item in round_three["observations"])
    assert all(item["episodes_before_learning"] == 4 for item in round_three["observations"])
