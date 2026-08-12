"""Bounded, event-producing tool loop shared by live leaf agents."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from time import monotonic
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from evoci.domain.models import SkillRef
from evoci.model.gateway import ToolLoopGateway, ToolLoopMessage
from evoci.runtime.budget import RunBudgetManager
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.registry import ToolRegistry

OutputT = TypeVar("OutputT", bound=BaseModel)


def _serializable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _result_success(result: Any) -> tuple[bool, int | None, str | None]:
    serialized = _serializable(result)
    if not isinstance(serialized, dict):
        return True, None, None
    exit_code = serialized.get("exit_code")
    timed_out = bool(serialized.get("timed_out", False))
    success = (exit_code in {None, 0}) and not timed_out
    error = None
    if not success:
        error = str(serialized.get("stderr") or "tool returned a non-zero exit status")
    return success, cast(int | None, exit_code), error


class BoundedToolAgent:
    """Keep leaf reasoning local while LangGraph retains orchestration control."""

    def __init__(
        self,
        gateway: ToolLoopGateway,
        recorder: TrajectoryRecorder,
        *,
        max_iterations: int,
        max_tool_calls: int,
        budget_manager: RunBudgetManager | None = None,
    ) -> None:
        if max_iterations < 1 or max_tool_calls < 1:
            raise ValueError("tool-loop budgets must be positive")
        self.gateway = gateway
        self.recorder = recorder
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.budget_manager = budget_manager

    async def run(
        self,
        *,
        run_id: str,
        agent_id: str,
        invocation_id: str,
        system_prompt: str,
        task_prompt: str,
        tools: ToolRegistry,
        output_schema: type[OutputT],
    ) -> OutputT:
        if not invocation_id.strip():
            raise ValueError("invocation_id must be a stable non-empty identifier")
        messages = [
            ToolLoopMessage(role="system", content=system_prompt),
            ToolLoopMessage(role="user", content=task_prompt),
        ]
        tool_count = 0
        used_skills: set[tuple[str, int]] = set()
        should_finalize = False
        budget = self.budget_manager.for_run(run_id) if self.budget_manager else None
        for iteration in range(1, self.max_iterations + 1):
            if budget is not None:
                budget.consume_model_call()
            try:
                response = await self.gateway.next_action(
                    messages=messages,
                    tools=tools.definitions(),
                    agent_id=agent_id,
                )
            except Exception as exc:
                self.recorder.emit(
                    run_id=run_id,
                    event_type=EventType.MODEL_CALL,
                    agent_id=agent_id,
                    invocation_id=invocation_id,
                    event_key=f"tool-turn:{iteration}",
                    payload={
                        "phase": "tool_loop",
                        "iteration": iteration,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
            self.recorder.emit(
                run_id=run_id,
                event_type=EventType.MODEL_CALL,
                agent_id=agent_id,
                invocation_id=invocation_id,
                event_key=f"tool-turn:{iteration}",
                payload={
                    "phase": "tool_loop",
                    "iteration": iteration,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
            messages.append(
                ToolLoopMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                should_finalize = True
                break
            for call in response.tool_calls:
                tool_count += 1
                if tool_count > self.max_tool_calls:
                    messages.append(
                        ToolLoopMessage(
                            role="tool",
                            tool_call_id=call.call_id,
                            content=json.dumps(
                                {"error": "leaf tool-call budget exhausted"}, sort_keys=True
                            ),
                        )
                    )
                    self.recorder.emit(
                        run_id=run_id,
                        event_type=EventType.ATTEMPT_FAILED,
                        agent_id=agent_id,
                        invocation_id=invocation_id,
                        event_key=f"tool-budget:{iteration}",
                        payload={
                            "kind": "tool_failure",
                            "agent_id": agent_id,
                            "reason": "leaf tool-call budget exhausted",
                            "iteration": iteration,
                        },
                    )
                    should_finalize = True
                    break
                if budget is not None:
                    budget.consume_tool_call()
                self.recorder.emit(
                    run_id=run_id,
                    event_type=EventType.TOOL_CALL,
                    agent_id=agent_id,
                    invocation_id=invocation_id,
                    event_key=call.call_id,
                    payload={
                        "call_id": call.call_id,
                        "tool_name": call.name,
                        "arguments": call.arguments,
                        "iteration": iteration,
                    },
                )
                started = monotonic()
                success = False
                exit_code: int | None = None
                error: str | None = None
                result: Any = None
                try:
                    result = await tools.ainvoke(call.name, **call.arguments)
                    success, exit_code, error = _result_success(result)
                except (KeyError, ValueError, TypeError, RuntimeError, OSError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                serialized = _serializable(result)
                payload: dict[str, Any] = {
                    "call_id": call.call_id,
                    "tool_name": call.name,
                    "success": success,
                    "exit_code": exit_code,
                    "error": error,
                    "duration_seconds": monotonic() - started,
                    "iteration": iteration,
                    "result": serialized,
                }
                if isinstance(serialized, dict):
                    payload["created_files"] = serialized.get("created_files", [])
                    payload["modified_files"] = serialized.get("modified_files", [])
                self.recorder.emit(
                    run_id=run_id,
                    event_type=EventType.TOOL_RESULT,
                    agent_id=agent_id,
                    invocation_id=invocation_id,
                    event_key=call.call_id,
                    payload=payload,
                )
                if call.name == "run_skill_script":
                    skill_id = str(call.arguments.get("skill_id", ""))
                    version = int(call.arguments.get("version", 0))
                    if skill_id and version > 0:
                        used_skills.add((skill_id, version))
                        self.recorder.emit(
                            run_id=run_id,
                            event_type=EventType.SKILL_USED,
                            agent_id=agent_id,
                            invocation_id=invocation_id,
                            event_key=call.call_id,
                            payload={
                                "skill_id": skill_id,
                                "version": version,
                                "resource": call.arguments.get("script_name"),
                                "success": success,
                            },
                        )
                tool_content = {"ok": success, "result": serialized, "error": error}
                messages.append(
                    ToolLoopMessage(
                        role="tool",
                        tool_call_id=call.call_id,
                        content=json.dumps(tool_content, default=str, sort_keys=True),
                    )
                )
            if should_finalize:
                break
        if not should_finalize:
            messages.append(
                ToolLoopMessage(
                    role="user",
                    content=(
                        "The leaf iteration budget is exhausted. Synthesize the best "
                        "supported final output."
                    ),
                )
            )
        try:
            if budget is not None:
                budget.consume_model_call()
            result = await self.gateway.finalize(
                messages=messages,
                response_model=output_schema,
                agent_id=agent_id,
            )
        except Exception as exc:
            self.recorder.emit(
                run_id=run_id,
                event_type=EventType.MODEL_CALL,
                agent_id=agent_id,
                invocation_id=invocation_id,
                event_key="structured-final",
                payload={
                    "phase": "structured_final",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        self.recorder.emit(
            run_id=run_id,
            event_type=EventType.MODEL_CALL,
            agent_id=agent_id,
            invocation_id=invocation_id,
            event_key="structured-final",
            payload={"phase": "structured_final"},
        )
        if "used_skill_refs" in type(result).model_fields:
            claimed = {
                (ref.skill_id, ref.version)
                for ref in cast(list[SkillRef], getattr(result, "used_skill_refs", []))
            }
            refs = [
                SkillRef(skill_id=skill_id, version=version)
                for skill_id, version in sorted(claimed | used_skills)
            ]
            result = cast(OutputT, result.model_copy(update={"used_skill_refs": refs}))
        return cast(OutputT, result)
