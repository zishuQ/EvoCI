"""OpenAI-compatible model gateway implementation."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, ValidationError

from evoci.config import EvoCIConfig, ReasoningEffort, RoleRuntimeConfig
from evoci.model.gateway import (
    ModelGatewayError,
    ModelUsage,
    RequestKind,
    ResponseT,
    ToolCallRequest,
    ToolDefinition,
    ToolLoopMessage,
    ToolModelResponse,
    UsageObserver,
)

ResultT = TypeVar("ResultT")

_MAX_STRUCTURED_CORRECTIONS = 2
_UNCATCHABLE = (KeyboardInterrupt, SystemExit)
_TRANSIENT_ERRORS = (
    TimeoutError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
_STRUCTURED_CONTENT_ERRORS = (
    ValidationError,
    json.JSONDecodeError,
    OutputParserException,
    TypeError,
    ValueError,
)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _as_optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _extract_usage(
    message: AIMessage,
    *,
    request_kind: RequestKind,
    model_name: str | None = None,
) -> ModelUsage:
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    resolved_model = model_name
    request_id = message.id if isinstance(message.id, str) and message.id else None

    raw_usage = message.usage_metadata
    usage_meta: dict[str, Any] = dict(raw_usage) if isinstance(raw_usage, dict) else {}
    if "input_tokens" in usage_meta:
        input_tokens = _as_int(usage_meta.get("input_tokens"))
    if "output_tokens" in usage_meta:
        output_tokens = _as_int(usage_meta.get("output_tokens"))
    input_details = usage_meta.get("input_token_details")
    if isinstance(input_details, dict):
        cached_input_tokens = _as_optional_int(
            input_details.get("cache_read", input_details.get("cached_tokens"))
        )
    output_details = usage_meta.get("output_token_details")
    if isinstance(output_details, dict):
        reasoning_tokens = _as_optional_int(
            output_details.get("reasoning", output_details.get("reasoning_tokens"))
        )

    response_meta = message.response_metadata if isinstance(message.response_metadata, dict) else {}
    token_usage = response_meta.get("token_usage") or response_meta.get("usage")
    if isinstance(token_usage, dict):
        if input_tokens == 0:
            input_tokens = _as_int(
                token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
            )
        if output_tokens == 0:
            output_tokens = _as_int(
                token_usage.get("completion_tokens", token_usage.get("output_tokens"))
            )
        prompt_details = token_usage.get("prompt_tokens_details")
        if cached_input_tokens is None and isinstance(prompt_details, dict):
            cached_input_tokens = _as_optional_int(
                prompt_details.get("cached_tokens", prompt_details.get("cache_read"))
            )
        completion_details = token_usage.get("completion_tokens_details")
        if reasoning_tokens is None and isinstance(completion_details, dict):
            reasoning_tokens = _as_optional_int(
                completion_details.get("reasoning_tokens", completion_details.get("reasoning"))
            )
        if cached_input_tokens is None:
            cached_input_tokens = _as_optional_int(token_usage.get("cached_tokens"))
        if reasoning_tokens is None:
            reasoning_tokens = _as_optional_int(token_usage.get("reasoning_tokens"))
    if resolved_model is None:
        reported = response_meta.get("model_name") or response_meta.get("model")
        if isinstance(reported, str):
            resolved_model = reported
    if request_id is None:
        reported_id = response_meta.get("id") or response_meta.get("response_id")
        if isinstance(reported_id, str):
            request_id = reported_id

    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        reasoning_tokens=reasoning_tokens,
        model_name=resolved_model,
        provider_request_id=request_id,
        request_kind=request_kind,
    )


def _raw_message_from_exception(exc: BaseException) -> AIMessage | None:
    for attr in ("llm_output", "generation", "response"):
        value = getattr(exc, attr, None)
        if isinstance(value, AIMessage):
            return value
        if isinstance(value, dict):
            message = value.get("message") or value.get("raw")
            if isinstance(message, AIMessage):
                return message
    return None


def _observe_usage(
    observer: UsageObserver | None,
    message: object,
    *,
    request_kind: RequestKind,
    model_name: str | None,
) -> None:
    if observer is None or not isinstance(message, AIMessage):
        return
    usage = _extract_usage(message, request_kind=request_kind, model_name=model_name)
    try:
        observer(usage)
    except Exception:
        return


def _reasoning_from_completion(response: Any) -> str | None:
    """Read DeepSeek-style reasoning_content from a raw chat completion."""
    message: Any = None
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
    else:
        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
    if message is None:
        return None
    if isinstance(message, dict):
        reasoning = message.get("reasoning_content")
    else:
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is None:
            extra = getattr(message, "model_extra", None)
            if isinstance(extra, dict):
                reasoning = extra.get("reasoning_content")
    return reasoning if isinstance(reasoning, str) else None


def _inject_reasoning_into_payload(
    payload: dict[str, Any], messages: list[BaseMessage]
) -> dict[str, Any]:
    """Copy assistant reasoning_content onto serialized chat-completion messages."""
    serialized = payload.get("messages")
    if not isinstance(serialized, list):
        return payload
    for original, item in zip(messages, serialized, strict=False):
        if not isinstance(original, AIMessage) or not isinstance(item, dict):
            continue
        reasoning = original.additional_kwargs.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            item["reasoning_content"] = reasoning
    return payload


class _ReasoningChatOpenAI(ChatOpenAI):
    """Preserve provider reasoning_content that stock ChatOpenAI drops."""

    def _create_chat_result(
        self,
        response: dict[str, Any] | Any,
        generation_info: dict[str, Any] | None = None,
    ) -> Any:
        result = super()._create_chat_result(response, generation_info)
        reasoning = _reasoning_from_completion(response)
        if not reasoning:
            return result
        generations = getattr(result, "generations", None) or []
        if not generations:
            return result
        message = generations[0].message
        if isinstance(message, AIMessage):
            message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return _inject_reasoning_into_payload(payload, self._convert_input(input_).to_messages())


class StructuredProbe(BaseModel):
    answer: str
    ok: bool


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract one complete JSON object from noisy model output."""
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    raise json.JSONDecodeError("model response did not contain a complete JSON object", text, 0)


class OpenAICompatibleGateway:
    """Call a user-configured OpenAI-compatible endpoint with structured output."""

    def __init__(
        self,
        config: EvoCIConfig,
        *,
        model_name: str | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        role: RoleRuntimeConfig | None = None,
        max_attempts: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if role is not None:
            resolved_model = model_name or role.model_name or config.model_name
            thinking = role.enable_thinking if enable_thinking is None else enable_thinking
            effort = role.reasoning_effort if reasoning_effort is None else reasoning_effort
            self.role: str | None = role.role
        else:
            resolved_model = model_name or config.model_name
            thinking = bool(enable_thinking)
            effort = reasoning_effort
            self.role = None
        if not resolved_model:
            raise ValueError("EVO_MODEL_NAME is required for live model calls")
        if not config.model_api_key:
            raise ValueError("EVO_MODEL_API_KEY is required for live model calls")
        self._timeout = config.model_timeout_seconds
        configured_attempts = max_attempts or int(os.environ.get("EVO_MODEL_RETRY_ATTEMPTS", "6"))
        self._max_attempts = max(1, configured_attempts)
        self.model_name = resolved_model
        self.enable_thinking = thinking
        self.reasoning_effort = effort
        self.max_output_tokens = max_output_tokens
        client_kwargs: dict[str, Any] = {}
        if thinking:
            if effort is not None:
                client_kwargs["reasoning_effort"] = effort
            client_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        self._client = _ReasoningChatOpenAI(
            model=resolved_model,
            base_url=config.model_base_url,
            api_key=config.model_api_key,
            timeout=self._timeout,
            max_retries=0,
            **client_kwargs,
        )

    def requested_runtime(self) -> dict[str, Any]:
        """Request parameters sent to the provider. This is not proof of execution."""

        return {
            "role": self.role,
            "model_name": self.model_name,
            "thinking_requested": self.enable_thinking,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
        }

    def _bind_output_limit(self, runnable: Any, max_output_tokens: int | None) -> Any:
        limit = (
            max_output_tokens
            if max_output_tokens is not None
            else getattr(self, "max_output_tokens", None)
        )
        if limit is None:
            return runnable
        return runnable.bind(max_tokens=limit)

    async def smoke_test(self) -> dict[str, tuple[bool, str | None]]:
        """Exercise the same ChatOpenAI client features used by the runtime."""

        results: dict[str, tuple[bool, str | None]] = {}
        try:
            response = await self._retry(
                lambda: self._client.ainvoke([("human", "Reply with exactly: hello")])
            )
            raw_content = response.content if isinstance(response, AIMessage) else str(response)
            content = raw_content if isinstance(raw_content, str) else str(raw_content)
            results["basic_chat"] = (
                content.strip() == "hello",
                None if content.strip() == "hello" else f"unexpected response: {content[:120]}",
            )
        except Exception as exc:
            results["basic_chat"] = (False, str(exc)[:160])

        try:
            result = await self.complete(
                system_prompt="Return structured JSON.",
                user_prompt='Return answer="hello" and ok=true.',
                response_model=StructuredProbe,
                agent_id="model-check",
            )
            valid = (
                isinstance(result, StructuredProbe)
                and result.answer == "hello"
                and result.ok is True
            )
            results["structured_output"] = (
                valid,
                None if valid else "response did not match answer=hello, ok=true",
            )
        except Exception as exc:
            results["structured_output"] = (False, str(exc)[:160])

        try:
            definition = {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo text.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            }
            runnable = self._client.bind_tools(
                [definition], tool_choice="auto", parallel_tool_calls=False
            )
            response = await self._retry(
                lambda: runnable.ainvoke([("human", 'Call echo with text="hello".')])
            )
            calls = getattr(response, "tool_calls", [])
            call = next((item for item in calls if item.get("name") == "echo"), None)
            valid = call is not None and call.get("args", {}).get("text") == "hello"
            results["tool_calling"] = (
                valid,
                None if valid else "response did not contain echo(text=hello)",
            )
        except Exception as exc:
            results["tool_calling"] = (False, str(exc)[:160])
        return results

    async def _retry(self, operation: Callable[[], Awaitable[ResultT]]) -> ResultT:
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return await asyncio.wait_for(operation(), timeout=self._timeout)
            except _UNCATCHABLE:
                raise
            except asyncio.CancelledError:
                raise
            except _TRANSIENT_ERRORS as exc:
                last_error = exc
                if attempt < self._max_attempts:
                    await asyncio.sleep(2.0 * (2 ** (attempt - 1)))
                    continue
            except ModelGatewayError:
                raise
        raise ModelGatewayError(
            f"model call failed after {self._max_attempts} attempts: {last_error}"
        ) from last_error

    async def _plain_json(
        self,
        messages: list[Any],
        response_model: type[ResponseT],
        correction: str | None = None,
        *,
        usage_observer: UsageObserver | None = None,
    ) -> ResponseT:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        prompt = (
            "Return ONLY one valid JSON object, with no Markdown, explanation, or preamble. "
            f"It must validate against this JSON schema: {schema}"
        )
        if correction:
            prompt += (
                f" Previous output failed validation: {correction}. Correct every issue. "
                "Return one complete valid JSON object and do not truncate it."
            )
        json_client = self._client.bind(response_format={"type": "json_object"})
        response = await self._retry(lambda: json_client.ainvoke([*messages, ("human", prompt)]))
        _observe_usage(
            usage_observer,
            response,
            request_kind="plain_json_correction",
            model_name=getattr(self, "model_name", None),
        )
        content = response.content if isinstance(response, AIMessage) else str(response)
        text = content if isinstance(content, str) else str(content)
        payload = _extract_json_object(text)
        if isinstance(payload, dict) and "tasks" in payload and "reasoning_summary" not in payload:
            payload["reasoning_summary"] = "Model returned an investigation task plan."
        return response_model.model_validate(payload)

    async def _structured_or_correct(
        self,
        *,
        invoke: Callable[[], Awaitable[Any]],
        messages: list[Any],
        response_model: type[ResponseT],
        agent_id: str,
        request_kind: RequestKind,
        usage_observer: UsageObserver | None = None,
    ) -> ResponseT:
        last_error: Exception | None = None
        try:
            result = await self._retry(invoke)
            if isinstance(result, dict) and "raw" in result:
                raw = result.get("raw")
                parsed = result.get("parsed")
                parsing_error = result.get("parsing_error")
                _observe_usage(
                    usage_observer,
                    raw,
                    request_kind=request_kind,
                    model_name=getattr(self, "model_name", None),
                )
                try:
                    if isinstance(parsed, response_model):
                        return parsed
                    if parsed is not None:
                        return response_model.model_validate(cast(Any, parsed))
                except _STRUCTURED_CONTENT_ERRORS as exc:
                    last_error = exc
                else:
                    if isinstance(parsing_error, Exception):
                        last_error = parsing_error
                    else:
                        last_error = ValueError("structured output returned no parsed result")
            elif isinstance(result, response_model):
                return result
            else:
                return response_model.model_validate(cast(Any, result))
        except _UNCATCHABLE:
            raise
        except asyncio.CancelledError:
            raise
        except ModelGatewayError:
            raise
        except _STRUCTURED_CONTENT_ERRORS as exc:
            last_error = exc
            raw_message = _raw_message_from_exception(exc)
            if raw_message is not None:
                _observe_usage(
                    usage_observer,
                    raw_message,
                    request_kind=request_kind,
                    model_name=getattr(self, "model_name", None),
                )
        if last_error is None:
            raise ModelGatewayError(f"{agent_id} structured output failed without an error")
        correction = str(last_error)
        for _ in range(_MAX_STRUCTURED_CORRECTIONS):
            try:
                return await self._plain_json(
                    messages,
                    response_model,
                    correction=correction,
                    usage_observer=usage_observer,
                )
            except _UNCATCHABLE:
                raise
            except asyncio.CancelledError:
                raise
            except ModelGatewayError:
                raise
            except _STRUCTURED_CONTENT_ERRORS as exc:
                last_error = exc
                correction = str(exc)
        assert last_error is not None
        raise ModelGatewayError(
            f"{agent_id} structured output failed after bounded correction: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
        agent_id: str,
        usage_observer: UsageObserver | None = None,
        max_output_tokens: int | None = None,
    ) -> ResponseT:
        runnable = self._bind_output_limit(
            self._client.with_structured_output(response_model, include_raw=True),
            max_output_tokens,
        )
        messages = [
            ("system", system_prompt),
            ("human", f"Agent: {agent_id}\n\n{user_prompt}"),
        ]
        return await self._structured_or_correct(
            invoke=lambda: runnable.ainvoke(messages),
            messages=messages,
            response_model=response_model,
            agent_id=agent_id,
            request_kind="structured",
            usage_observer=usage_observer,
        )

    @staticmethod
    def _messages(messages: list[ToolLoopMessage]) -> list[BaseMessage]:
        converted: list[BaseMessage] = []
        for message in messages:
            if message.role == "system":
                converted.append(SystemMessage(content=message.content))
            elif message.role == "user":
                converted.append(HumanMessage(content=message.content))
            elif message.role == "assistant":
                converted.append(
                    AIMessage(
                        content=message.content,
                        tool_calls=[
                            {
                                "name": call.name,
                                "args": call.arguments,
                                "id": call.call_id,
                                "type": "tool_call",
                            }
                            for call in message.tool_calls
                        ],
                        additional_kwargs=(
                            {"reasoning_content": message.reasoning_content}
                            if message.reasoning_content
                            else {}
                        ),
                    )
                )
            else:
                if message.tool_call_id is None:
                    raise ValueError("tool messages require tool_call_id")
                converted.append(
                    ToolMessage(content=message.content, tool_call_id=message.tool_call_id)
                )
        return converted

    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
        usage_observer: UsageObserver | None = None,
        max_output_tokens: int | None = None,
    ) -> ToolModelResponse:
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
        runnable = self._bind_output_limit(
            self._client.bind_tools(definitions, tool_choice="auto", parallel_tool_calls=False),
            max_output_tokens,
        )
        try:
            raw = await self._retry(lambda: runnable.ainvoke(self._messages(messages)))
        except _UNCATCHABLE:
            raise
        except asyncio.CancelledError:
            raise
        except ModelGatewayError:
            raise
        if not isinstance(raw, AIMessage):
            raise ModelGatewayError("tool-bound model returned a non-assistant message")
        _observe_usage(
            usage_observer,
            raw,
            request_kind="tool_action",
            model_name=getattr(self, "model_name", None),
        )
        try:
            usage: dict[str, Any] = dict(raw.usage_metadata or {})
            calls = [
                ToolCallRequest(
                    call_id=str(call["id"]),
                    name=str(call["name"]),
                    arguments=dict(call.get("args", {})),
                )
                for call in raw.tool_calls
            ]
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            reasoning_content = raw.additional_kwargs.get("reasoning_content")
            if not isinstance(reasoning_content, str):
                reasoning_content = None
            return ToolModelResponse(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=calls,
                input_tokens=cast(int | None, usage.get("input_tokens")),
                output_tokens=cast(int | None, usage.get("output_tokens")),
            )
        except _UNCATCHABLE:
            raise
        except asyncio.CancelledError:
            raise
        except RuntimeError:
            raise
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise ModelGatewayError(
                f"{agent_id} returned invalid tool-call response: {type(exc).__name__}: {exc}"
            ) from exc

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
        usage_observer: UsageObserver | None = None,
        max_output_tokens: int | None = None,
    ) -> ResponseT:
        runnable = self._bind_output_limit(
            self._client.with_structured_output(response_model, include_raw=True),
            max_output_tokens,
        )
        final_messages = [
            *self._messages(messages),
            HumanMessage(
                content=(
                    f"Agent {agent_id}: return your final answer now as the required "
                    f"{response_model.__name__} structured object. Only claim memories and "
                    "skills that materially influenced the answer."
                )
            ),
        ]
        return await self._structured_or_correct(
            invoke=lambda: runnable.ainvoke(final_messages),
            messages=final_messages,
            response_model=response_model,
            agent_id=agent_id,
            request_kind="structured_finalize",
            usage_observer=usage_observer,
        )
