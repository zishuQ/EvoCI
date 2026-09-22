"""Bounded, event-producing tool loop shared by live leaf agents."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from time import monotonic
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from evoci.domain.models import SkillRef
from evoci.model.gateway import ToolLoopGateway, ToolLoopMessage
from evoci.runtime.budget import (
    FINALIZATION_MODEL_CALLS,
    CombinedBudget,
    RepairBudgetExhausted,
    RunBudgetManager,
    RunRepairBudget,
)
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.runtime.usage import make_usage_observer
from evoci.tools.policy import PolicyViolation
from evoci.tools.registry import LoadSkillArgs, RunSkillScriptArgs, ToolRegistry

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


def _skill_ref_from_arguments(arguments: dict[str, Any]) -> str | None:
    try:
        validated = RunSkillScriptArgs.model_validate(arguments)
    except ValidationError:
        return None
    return validated.skill_id


def _load_skill_id(arguments: dict[str, Any]) -> str | None:
    try:
        return LoadSkillArgs.model_validate(arguments).skill_id
    except ValidationError:
        return None


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
        extra_budget: RunRepairBudget | None = None,
    ) -> OutputT:
        if not invocation_id.strip():
            raise ValueError("invocation_id must be a stable non-empty identifier")
        messages = [
            ToolLoopMessage(role="system", content=system_prompt),
            ToolLoopMessage(role="user", content=task_prompt),
        ]
        tool_count = 0
        used_skills: set[str] = set()
        selected_skills: set[str] = set()
        should_finalize = False
        budgets: list[RunRepairBudget] = []
        if self.budget_manager is not None:
            budgets.append(self.budget_manager.for_run(run_id))
        if extra_budget is not None:
            budgets.append(extra_budget)
        budget = CombinedBudget(*budgets) if budgets else None
        action_limit = self.max_iterations
        max_tools = self.max_tool_calls
        if extra_budget is not None:
            action_limit = min(
                action_limit, max(0, extra_budget.max_model_calls - FINALIZATION_MODEL_CALLS)
            )
            max_tools = min(max_tools, extra_budget.max_tool_calls)
        if self.budget_manager is not None:
            remaining = self.budget_manager.for_run(run_id).remaining_model_calls()
            if remaining < 1 + FINALIZATION_MODEL_CALLS:
                raise RepairBudgetExhausted("run-level model-call budget exhausted")
            action_limit = min(action_limit, remaining - FINALIZATION_MODEL_CALLS)
        if action_limit < 1 or max_tools < 1:
            raise RepairBudgetExhausted("task-level model-call budget exhausted")
        for iteration in range(1, action_limit + 1):
            if budget is not None:
                budget.consume_model_call()
            try:
                response = await self.gateway.next_action(
                    messages=messages,
                    tools=tools.definitions(),
                    agent_id=agent_id,
                    usage_observer=make_usage_observer(
                        recorder=self.recorder,
                        run_id=run_id,
                        agent_id=agent_id,
                        invocation_id=invocation_id,
                        budget_scope="repair",
                        call_key=f"tool-turn:{iteration}",
                    ),
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
                    reasoning_content=response.reasoning_content,
                    tool_calls=response.tool_calls,
                )
            )
            if not response.tool_calls:
                should_finalize = True
                break
            budget_exhausted = False
            for idx, call in enumerate(response.tool_calls):
                tool_count += 1
                if tool_count > max_tools:
                    # Add error responses for all remaining tool_calls to satisfy API protocol
                    budget_exhausted = True
                    for remaining_call in response.tool_calls[idx:]:
                        messages.append(
                            ToolLoopMessage(
                                role="tool",
                                tool_call_id=remaining_call.call_id,
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
                if budget_exhausted:
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
                rejected = False
                try:
                    result = await tools.ainvoke(call.name, **call.arguments)
                    success, exit_code, error = _result_success(result)
                except (PolicyViolation, ValidationError) as exc:
                    rejected = True
                    error = f"{type(exc).__name__}: {exc}"
                except (KeyError, ValueError, TypeError, RuntimeError, OSError, re.error) as exc:
                    rejected = call.name in {
                        "run_skill_script",
                        "read_skill_resource",
                        "load_skill",
                    }
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
                if call.name == "load_skill":
                    skill_id = _load_skill_id(call.arguments)
                    if rejected or not success:
                        self.recorder.emit(
                            run_id=run_id,
                            event_type=EventType.SKILL_INVOCATION_REJECTED,
                            agent_id=agent_id,
                            invocation_id=invocation_id,
                            event_key=call.call_id,
                            payload={
                                "skill_id": skill_id,
                                "reason": error,
                            },
                        )
                    elif skill_id and skill_id not in selected_skills:
                        selected_skills.add(skill_id)
                        self.recorder.emit(
                            run_id=run_id,
                            event_type=EventType.SKILL_SELECTED,
                            agent_id=agent_id,
                            invocation_id=invocation_id,
                            event_key=f"select:{skill_id}",
                            payload={"skills": [{"skill_id": skill_id}]},
                        )
                if call.name == "run_skill_script":
                    ref = _skill_ref_from_arguments(call.arguments)
                    resource = call.arguments.get("script_name")
                    if rejected or result is None:
                        self.recorder.emit(
                            run_id=run_id,
                            event_type=EventType.SKILL_INVOCATION_REJECTED,
                            agent_id=agent_id,
                            invocation_id=invocation_id,
                            event_key=call.call_id,
                            payload={
                                "skill_id": ref,
                                "resource": resource,
                                "reason": error,
                            },
                        )
                    elif ref is not None:
                        used_skills.add(ref)
                        self.recorder.emit(
                            run_id=run_id,
                            event_type=EventType.SKILL_USED,
                            agent_id=agent_id,
                            invocation_id=invocation_id,
                            event_key=call.call_id,
                            payload={
                                "skill_id": ref,
                                "usage_kind": "script",
                                "resource": resource,
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
                usage_observer=make_usage_observer(
                    recorder=self.recorder,
                    run_id=run_id,
                    agent_id=agent_id,
                    invocation_id=invocation_id,
                    budget_scope="repair",
                    call_key="structured-final",
                ),
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
                ref.skill_id
                for ref in cast(list[SkillRef], getattr(result, "used_skill_refs", []))
            }
            activated = set(tools.activated_skill_ids) | selected_skills
            valid_claimed = claimed & activated
            # Script use preferentially represents this Skill. If a script already
            # produced SKILL_USED, do not also emit a procedure use for the same
            # skill_id. The production loop therefore does not currently emit both
            # procedure and script traces for one Skill.
            for skill_id in sorted(valid_claimed - used_skills):
                self.recorder.emit(
                    run_id=run_id,
                    event_type=EventType.SKILL_USED,
                    agent_id=agent_id,
                    invocation_id=invocation_id,
                    event_key=f"method-use:{skill_id}",
                    payload={
                        "skill_id": skill_id,
                        "usage_kind": "procedure",
                        "resource": None,
                        "success": None,
                    },
                )
            refs = [
                SkillRef(skill_id=skill_id)
                for skill_id in sorted(valid_claimed | used_skills)
            ]
            result = cast(OutputT, result.model_copy(update={"used_skill_refs": refs}))
        return cast(OutputT, result)
