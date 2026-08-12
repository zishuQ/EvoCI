from pathlib import Path

from evoci.runtime.event_store import SQLiteEventStore
from evoci.runtime.events import EventType, RunEvent
from evoci.runtime.trajectory import TrajectoryRecorder


def test_events_are_append_only_and_idempotent(tmp_path: Path) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite")
    event = RunEvent(run_id="run-1", type=EventType.RUN_STARTED, payload={"task": "x"})
    store.append(event)
    store.append(event)
    assert store.list("run-1") == [event]
    store.close()


def test_repeated_fixer_and_reviewer_invocations_keep_same_iteration_events(
    tmp_path: Path,
) -> None:
    store = SQLiteEventStore(tmp_path / "events.sqlite")
    recorder = TrajectoryRecorder(store)

    for agent_id, invocation_ids in {
        "fixer": ["repair:1", "repair:2"],
        "reviewer": ["review:1", "review:2"],
    }.items():
        for invocation_id in invocation_ids:
            recorder.emit(
                run_id="repeat-run",
                event_type=EventType.MODEL_CALL,
                agent_id=agent_id,
                invocation_id=invocation_id,
                event_key="tool-turn:1",
                payload={"iteration": 1},
            )

    events = [event for event in store.list("repeat-run") if event.type == EventType.MODEL_CALL]
    assert len([event for event in events if event.agent_id == "fixer"]) == 2
    assert len([event for event in events if event.agent_id == "reviewer"]) == 2
    assert len({event.event_id for event in events}) == 4
    assert any(":repair:1:ModelCall:tool-turn:1" in event.event_id for event in events)
    assert any(":repair:2:ModelCall:tool-turn:1" in event.event_id for event in events)
    assert any(":review:1:ModelCall:tool-turn:1" in event.event_id for event in events)
    assert any(":review:2:ModelCall:tool-turn:1" in event.event_id for event in events)
    store.close()
