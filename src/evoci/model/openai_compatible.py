"""OpenAI-compatible model gateway implementation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel

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


class StructuredProbe(BaseModel):
    answer: str
    ok: bool


class OpenAICompatibleGateway:
    """Call a user-configured OpenAI-compatible endpoint with structured output."""

    def __init__(
        self,
        config: EvoCIConfig,
        *,
        model_name: str | None = None,
        max_attempts: int = 6,
    ) -> None:
        resolved_model = model_name or config.model_name
        if not resolved_model:
            raise ValueError("EVO_MODEL_NAME is required for live model calls")
        if not config.model_api_key:
            raise ValueError("EVO_MODEL_API_KEY is required for live model calls")
        self._timeout = config.model_timeout_seconds
        self._max_attempts = max_attempts
        self.model_name = resolved_model
        self._client = ChatOpenAI(
            model=resolved_model,
            base_url=config.model_base_url,
            api_key=config.model_api_key,
            timeout=self._timeout,
            max_retries=0,
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
            runnable = self._client.with_structured_output(StructuredProbe)
            result = await self._retry(
                lambda: runnable.ainvoke(
                    [("human", 'Return answer="hello" and ok=true.')]
                )
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
            except (
                TimeoutError,
                ValueError,
                TypeError,
                RuntimeError,
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            ) as exc:
                last_error = exc
                if attempt < self._max_attempts:
                    await asyncio.sleep(2.0 * (2 ** (attempt - 1)))
        raise ModelGatewayError(
            f"model call failed after {self._max_attempts} attempts: {last_error}"
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
        result = await self._retry(
            lambda: runnable.ainvoke(
                [
                    ("system", system_prompt),
                    ("human", f"Agent: {agent_id}\n\n{user_prompt}"),
                ]
            )
        )
        if isinstance(result, response_model):
            return result
        return response_model.model_validate(cast(Any, result))

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
        del agent_id
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
        raw = await self._retry(lambda: runnable.ainvoke(self._messages(messages)))
        if not isinstance(raw, AIMessage):
            raise ModelGatewayError("tool-bound model returned a non-assistant message")
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
        return ToolModelResponse(
            content=content,
            tool_calls=calls,
            input_tokens=cast(int | None, usage.get("input_tokens")),
            output_tokens=cast(int | None, usage.get("output_tokens")),
        )

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
        result = await self._retry(lambda: runnable.ainvoke(final_messages))
        if isinstance(result, response_model):
            return result
        return response_model.model_validate(cast(Any, result))
