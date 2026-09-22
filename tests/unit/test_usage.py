from evoci.model.gateway import ModelUsage, UsageObserver
from evoci.runtime.events import EventType, RunEvent
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.runtime.usage import make_usage_observer, usage_complete_from_events


def _observer(recorder: TrajectoryRecorder) -> UsageObserver:
    return make_usage_observer(
        recorder=recorder,
        run_id="run-replay",
        agent_id="fixer",
        invocation_id="repair:1",
        budget_scope="repair",
        call_key="tool-turn:1",
    )


def _usage(*, request_id: str | None, input_tokens: int = 3) -> ModelUsage:
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=1,
        provider_request_id=request_id,
        request_kind="tool_action",
    )


def _usage_events(recorder: TrajectoryRecorder) -> list[RunEvent]:
    return [event for event in recorder.events("run-replay") if event.type == EventType.MODEL_USAGE]


def test_replayed_observers_keep_distinct_provider_request_ids() -> None:
    recorder = TrajectoryRecorder()
    first = _observer(recorder)
    second = _observer(recorder)
    first(_usage(request_id="req-a", input_tokens=10))
    second(_usage(request_id="req-b", input_tokens=20))
    events = _usage_events(recorder)
    assert len(events) == 2
    assert {event.payload["provider_request_id"] for event in events} == {"req-a", "req-b"}
    assert len({event.event_id for event in events}) == 2


def test_duplicate_provider_request_id_is_recorded_once() -> None:
    recorder = TrajectoryRecorder()
    first = _observer(recorder)
    second = _observer(recorder)
    first(_usage(request_id="req-same", input_tokens=11))
    second(_usage(request_id="req-same", input_tokens=99))
    events = _usage_events(recorder)
    assert len(events) == 1
    assert events[0].payload["provider_request_id"] == "req-same"
    assert events[0].payload["input_tokens"] == 11


def test_missing_provider_request_id_keeps_events_from_separate_observers() -> None:
    recorder = TrajectoryRecorder()
    first = _observer(recorder)
    second = _observer(recorder)
    first(_usage(request_id=None, input_tokens=4))
    second(_usage(request_id=None, input_tokens=5))
    events = _usage_events(recorder)
    assert len(events) == 2
    assert len({event.event_id for event in events}) == 2
    assert {event.payload["input_tokens"] for event in events} == {4, 5}


def test_usage_complete_is_false_without_usage_events() -> None:
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="run-replay",
        event_type=EventType.MODEL_CALL,
        agent_id="worker",
        invocation_id="worker:1",
        event_key="tool-turn:1",
        payload={"phase": "tool_loop"},
    )
    assert usage_complete_from_events(recorder.events("run-replay")) is False


def test_usage_complete_rejects_zero_token_usage() -> None:
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="run-replay",
        event_type=EventType.MODEL_USAGE,
        agent_id="worker",
        invocation_id="worker:1",
        event_key="usage:1",
        payload={"input_tokens": 0, "output_tokens": 0, "usage_complete": False},
    )
    assert usage_complete_from_events(recorder.events("run-replay")) is False
