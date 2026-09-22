"""Record provider-level model usage without changing repair or learning logic."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
from uuid import uuid4

from evoci.model.gateway import ModelUsage, UsageObserver
from evoci.runtime.events import EventType, RunEvent
from evoci.runtime.trajectory import TrajectoryRecorder

BudgetScope = Literal["repair", "post_run"]


def usage_complete_from_events(events: Sequence[RunEvent]) -> bool:
    """True only when every repair model call has non-zero reported usage."""

    model_calls = [
        event
        for event in events
        if event.type == EventType.MODEL_CALL and event.payload.get("budget_scope") != "post_run"
    ]
    usage_events = [
        event
        for event in events
        if event.type == EventType.MODEL_USAGE and event.payload.get("budget_scope") != "post_run"
    ]
    if model_calls and not usage_events:
        return False
    for event in usage_events:
        if event.payload.get("usage_complete") is False:
            return False
        if not (event.payload.get("input_tokens") or event.payload.get("output_tokens")):
            return False
    return True


def make_usage_observer(
    *,
    recorder: TrajectoryRecorder | None,
    run_id: str,
    agent_id: str,
    invocation_id: str | None,
    budget_scope: BudgetScope,
    call_key: str,
    extra: dict[str, int | str | None] | None = None,
) -> UsageObserver:
    sequence = 0
    observer_nonce = uuid4().hex

    def observe(usage: ModelUsage) -> None:
        nonlocal sequence
        sequence += 1
        if recorder is None:
            return
        request_id = usage.provider_request_id.strip() if usage.provider_request_id else ""
        request_key = (
            f"request:{request_id}" if request_id else f"observer:{observer_nonce}:{sequence}"
        )
        payload = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "model_name": usage.model_name,
            "provider_request_id": usage.provider_request_id,
            "request_kind": usage.request_kind,
            "budget_scope": budget_scope,
            "usage_complete": bool(
                usage.input_tokens or usage.output_tokens or usage.total_tokens
            ),
        }
        if extra:
            payload.update(extra)
        try:
            recorder.emit(
                run_id=run_id,
                event_type=EventType.MODEL_USAGE,
                agent_id=agent_id,
                invocation_id=invocation_id,
                event_key=f"usage:{call_key}:{request_key}",
                payload=payload,
            )
        except Exception:
            return

    return observe
