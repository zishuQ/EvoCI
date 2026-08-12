from pathlib import Path

from evoci.runtime.run_store import SQLiteRunStore


def test_run_store_tracks_status(tmp_path: Path) -> None:
    store = SQLiteRunStore(tmp_path / "runs.sqlite")
    store.create("run-1", "task-1", {"source": "fixture"})
    store.update_status("run-1", "waiting_approval")
    record = store.get("run-1")
    assert record is not None
    assert record.status == "waiting_approval"
    assert record.metadata == {"source": "fixture"}
    store.close()
