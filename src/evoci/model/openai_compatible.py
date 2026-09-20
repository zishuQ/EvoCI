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

from evoci.config import EvoCIConfig
from evoci.model.gateway import (
    ModelGatewayError,
    ResponseT,
    ToolCallRequest,
    ToolDefinition,
    ToolLoopMessage,
    ToolModelResponse,
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
        max_attempts: int | None = None,
    ) -> None:
        resolved_model = model_name or config.model_name
        if not resolved_model:
            raise ValueError("EVO_MODEL_NAME is required for live model calls")
        if not config.model_api_key:
            raise ValueError("EVO_MODEL_API_KEY is required for live model calls")
        self._timeout = config.model_timeout_seconds
        configured_attempts = max_attempts or int(os.environ.get("EVO_MODEL_RETRY_ATTEMPTS", "6"))
        self._max_attempts = max(1, configured_attempts)
        self.model_name = resolved_model
        self._enable_thinking = config.enable_thinking
        client_kwargs: dict[str, Any] = {}
        if self._enable_thinking:
            client_kwargs["reasoning_effort"] = "high"
            client_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        self._client = _ReasoningChatOpenAI(
            model=resolved_model,
            base_url=config.model_base_url,
            api_key=config.model_api_key,
            timeout=self._timeout,
            max_retries=0,
            **client_kwargs,
        )

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
    ) -> ResponseT:
        last_error: Exception | None = None
        try:
            result = await self._retry(invoke)
            if isinstance(result, response_model):
                return result
            return response_model.model_validate(cast(Any, result))
        except _UNCATCHABLE:
            raise
        except asyncio.CancelledError:
            raise
        except ModelGatewayError:
            raise
        except _STRUCTURED_CONTENT_ERRORS as exc:
            last_error = exc
        correction = str(last_error)
        for _ in range(_MAX_STRUCTURED_CORRECTIONS):
            try:
                return await self._plain_json(messages, response_model, correction=correction)
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
    ) -> ResponseT:
        runnable = self._client.with_structured_output(response_model)
        messages = [
            ("system", system_prompt),
            ("human", f"Agent: {agent_id}\n\n{user_prompt}"),
        ]
        return await self._structured_or_correct(
            invoke=lambda: runnable.ainvoke(messages),
            messages=messages,
            response_model=response_model,
            agent_id=agent_id,
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
        runnable = self._client.bind_tools(
            definitions, tool_choice="auto", parallel_tool_calls=False
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
    ) -> ResponseT:
        runnable = self._client.with_structured_output(response_model)
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
        )
