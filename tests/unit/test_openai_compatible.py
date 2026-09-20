from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from openai import APIConnectionError
from pydantic import BaseModel, SecretStr, ValidationError

from evoci.config import EvoCIConfig
from evoci.model.gateway import ModelGatewayError, ToolCallRequest, ToolDefinition, ToolLoopMessage
from evoci.model.openai_compatible import (
    _MAX_STRUCTURED_CORRECTIONS,
    OpenAICompatibleGateway,
    _extract_json_object,
    _inject_reasoning_into_payload,
    _reasoning_from_completion,
)


def test_extract_json_object_ignores_trailing_json() -> None:
    assert _extract_json_object('{"answer": "first"}\n{"answer": "second"}') == {"answer": "first"}


def test_extract_json_object_supports_fenced_json() -> None:
    assert _extract_json_object('Result:\n```json\n{"ok": true}\n```') == {"ok": True}


def test_extract_json_object_rejects_truncated_json() -> None:
    with pytest.raises(ValueError, match="complete JSON object"):
        _extract_json_object('{"answer": "unfinished"')


@pytest.mark.asyncio
async def test_retry_recovers_from_openai_connection_error(monkeypatch) -> None:
    gateway = object.__new__(OpenAICompatibleGateway)
    gateway._timeout = 1.0
    gateway._max_attempts = 2
    attempts = 0

    async def no_wait(_: float) -> None:
        return None

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise APIConnectionError(request=httpx.Request("POST", "https://example.test"))
        return "recovered"

    monkeypatch.setattr("evoci.model.openai_compatible.asyncio.sleep", no_wait)
    retry_operation: Callable[[], Awaitable[str]] = operation
    assert await gateway._retry(retry_operation) == "recovered"
    assert attempts == 2


def _config(*, enable_thinking: bool) -> EvoCIConfig:
    return EvoCIConfig(
        model_name="deepseek-chat",
        model_api_key=SecretStr("sk-test"),
        enable_thinking=enable_thinking,
    )


def _echo_tool() -> ToolDefinition:
    return ToolDefinition(
        name="echo",
        description="Echo text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


class _FakeRunnable:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.seen: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(list(messages))
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.runnable = _FakeRunnable(responses)

    def bind_tools(self, *args: object, **kwargs: object) -> _FakeRunnable:
        del args, kwargs
        return self.runnable


def _gateway_with_client(client: _FakeClient) -> OpenAICompatibleGateway:
    gateway = object.__new__(OpenAICompatibleGateway)
    gateway._timeout = 5.0
    gateway._max_attempts = 1
    gateway._client = client
    return gateway


def test_thinking_enabled_sets_client_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("evoci.model.openai_compatible._ReasoningChatOpenAI", FakeChatOpenAI)
    OpenAICompatibleGateway(_config(enable_thinking=True))
    assert captured["reasoning_effort"] == "high"
    assert captured["extra_body"] == {"thinking": {"type": "enabled"}}


def test_thinking_disabled_omits_client_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("evoci.model.openai_compatible._ReasoningChatOpenAI", FakeChatOpenAI)
    OpenAICompatibleGateway(_config(enable_thinking=False))
    assert "reasoning_effort" not in captured
    assert "extra_body" not in captured


def test_reasoning_from_completion_reads_provider_field() -> None:
    assert (
        _reasoning_from_completion(
            {"choices": [{"message": {"role": "assistant", "reasoning_content": "step by step"}}]}
        )
        == "step by step"
    )
    assert _reasoning_from_completion({"choices": [{"message": {"role": "assistant"}}]}) is None


def test_inject_reasoning_into_payload_copies_assistant_field() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "calling tool"},
        ]
    }
    messages: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(
            content="calling tool",
            additional_kwargs={"reasoning_content": "need the tool"},
        ),
    ]
    injected = _inject_reasoning_into_payload(payload, messages)
    assert injected["messages"][1]["reasoning_content"] == "need the tool"


@pytest.mark.asyncio
async def test_next_action_extracts_reasoning_content() -> None:
    client = _FakeClient(
        [
            AIMessage(
                content="calling echo",
                additional_kwargs={"reasoning_content": "I should echo first"},
                tool_calls=[
                    {
                        "name": "echo",
                        "args": {"text": "hello"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    gateway = _gateway_with_client(client)
    response = await gateway.next_action(
        messages=[ToolLoopMessage(role="user", content="hello")],
        tools=[_echo_tool()],
        agent_id="tester",
    )
    assert response.content == "calling echo"
    assert response.reasoning_content == "I should echo first"
    assert response.tool_calls[0].call_id == "call-1"


@pytest.mark.asyncio
async def test_next_action_drops_non_string_reasoning_content() -> None:
    client = _FakeClient(
        [AIMessage(content="done", additional_kwargs={"reasoning_content": {"nested": True}})]
    )
    gateway = _gateway_with_client(client)
    response = await gateway.next_action(
        messages=[ToolLoopMessage(role="user", content="hello")],
        tools=[_echo_tool()],
        agent_id="tester",
    )
    assert response.reasoning_content is None


@pytest.mark.asyncio
async def test_second_turn_sends_first_turn_reasoning() -> None:
    client = _FakeClient(
        [
            AIMessage(
                content="calling echo",
                additional_kwargs={"reasoning_content": "inspect then echo"},
                tool_calls=[
                    {
                        "name": "echo",
                        "args": {"text": "hello"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    gateway = _gateway_with_client(client)
    first = await gateway.next_action(
        messages=[ToolLoopMessage(role="user", content="hello")],
        tools=[_echo_tool()],
        agent_id="tester",
    )
    await gateway.next_action(
        messages=[
            ToolLoopMessage(role="user", content="hello"),
            ToolLoopMessage(
                role="assistant",
                content=first.content,
                reasoning_content=first.reasoning_content,
                tool_calls=first.tool_calls,
            ),
            ToolLoopMessage(role="tool", tool_call_id="call-1", content="hello"),
        ],
        tools=[_echo_tool()],
        agent_id="tester",
    )
    second_request = client.runnable.seen[1]
    assistant = next(message for message in second_request if isinstance(message, AIMessage))
    assert assistant.additional_kwargs["reasoning_content"] == "inspect then echo"


def test_messages_omits_reasoning_when_absent() -> None:
    converted = OpenAICompatibleGateway._messages(
        [
            ToolLoopMessage(
                role="assistant",
                content="calling echo",
                tool_calls=[
                    ToolCallRequest(call_id="call-1", name="echo", arguments={"text": "hi"})
                ],
            )
        ]
    )
    assert isinstance(converted[0], AIMessage)
    assert converted[0].additional_kwargs == {}


class _Probe(BaseModel):
    answer: str


class _StructuredThenJsonClient:
    def __init__(self, structured_error: Exception, json_payloads: list[str]) -> None:
        self.structured_error = structured_error
        self.json_payloads = list(json_payloads)
        self.structured_calls = 0
        self.json_calls = 0

    def with_structured_output(self, model: type[BaseModel]) -> _StructuredThenJsonClient:
        del model
        return self

    def bind(self, **kwargs: object) -> _StructuredThenJsonClient:
        del kwargs
        return self

    async def ainvoke(self, messages: object) -> AIMessage | BaseModel:
        del messages
        if self.json_calls == 0 and self.structured_calls == 0:
            self.structured_calls += 1
            raise self.structured_error
        self.json_calls += 1
        return AIMessage(content=self.json_payloads.pop(0))


def _gateway_for_structured(client: _StructuredThenJsonClient) -> OpenAICompatibleGateway:
    gateway = object.__new__(OpenAICompatibleGateway)
    gateway._timeout = 5.0
    gateway._max_attempts = 1
    gateway._client = client
    return gateway


@pytest.mark.asyncio
async def test_complete_turns_final_validation_error_into_gateway_error() -> None:
    client = _StructuredThenJsonClient(
        ValidationError.from_exception_data("Probe", []),
        ['{"wrong": true}', '{"still": "wrong"}'],
    )
    gateway = _gateway_for_structured(client)
    with pytest.raises(ModelGatewayError, match="bounded correction"):
        await gateway.complete(
            system_prompt="sys",
            user_prompt="user",
            response_model=_Probe,
            agent_id="diagnoser",
        )
    assert client.json_calls == _MAX_STRUCTURED_CORRECTIONS


@pytest.mark.asyncio
async def test_finalize_turns_final_validation_error_into_gateway_error() -> None:
    client = _StructuredThenJsonClient(
        ValidationError.from_exception_data("Probe", []),
        ['{"wrong": true}', '{"still": "wrong"}'],
    )
    gateway = _gateway_for_structured(client)
    with pytest.raises(ModelGatewayError, match="bounded correction"):
        await gateway.finalize(
            messages=[ToolLoopMessage(role="user", content="done")],
            response_model=_Probe,
            agent_id="fixer",
        )


@pytest.mark.asyncio
async def test_length_finish_error_becomes_gateway_error() -> None:
    client = _StructuredThenJsonClient(
        ValueError("Could not parse response content as the length limit was reached"),
        ['{"answer": "trun', '{"answer":'],
    )
    gateway = _gateway_for_structured(client)
    with pytest.raises(ModelGatewayError, match="bounded correction"):
        await gateway.complete(
            system_prompt="sys",
            user_prompt="user",
            response_model=_Probe,
            agent_id="fixer",
        )


@pytest.mark.asyncio
async def test_truncated_json_becomes_gateway_error() -> None:
    client = _StructuredThenJsonClient(
        ValueError("truncated"),
        ['{"answer": "unfinished"', '{"answer":'],
    )
    gateway = _gateway_for_structured(client)
    with pytest.raises(ModelGatewayError, match=r"JSONDecodeError|bounded correction"):
        await gateway.complete(
            system_prompt="sys",
            user_prompt="user",
            response_model=_Probe,
            agent_id="fixer",
        )


@pytest.mark.asyncio
async def test_structured_corrections_are_strictly_bounded() -> None:
    client = _StructuredThenJsonClient(
        ValidationError.from_exception_data("Probe", []),
        ['{"nope": 1}', '{"nope": 2}', '{"nope": 3}'],
    )
    gateway = _gateway_for_structured(client)
    with pytest.raises(ModelGatewayError):
        await gateway.complete(
            system_prompt="sys",
            user_prompt="user",
            response_model=_Probe,
            agent_id="coordinator",
        )
    assert client.json_calls == _MAX_STRUCTURED_CORRECTIONS
    assert client.json_payloads == ['{"nope": 3}']


@pytest.mark.asyncio
async def test_cancelled_error_is_not_wrapped() -> None:
    client = _StructuredThenJsonClient(asyncio.CancelledError(), [])
    gateway = _gateway_for_structured(client)
    with pytest.raises(asyncio.CancelledError):
        await gateway.complete(
            system_prompt="sys",
            user_prompt="user",
            response_model=_Probe,
            agent_id="diagnoser",
        )


@pytest.mark.asyncio
async def test_complete_does_not_swallow_unexpected_runtime_error() -> None:
    client = _StructuredThenJsonClient(RuntimeError("internal parser crash"), ['{"answer": "ok"}'])
    gateway = _gateway_for_structured(client)
    with pytest.raises(RuntimeError, match="internal parser crash"):
        await gateway.complete(
            system_prompt="sys",
            user_prompt="user",
            response_model=_Probe,
            agent_id="diagnoser",
        )
    assert client.json_calls == 0


@pytest.mark.asyncio
async def test_correction_model_gateway_error_is_not_retried() -> None:
    client = _StructuredThenJsonClient(
        ValidationError.from_exception_data("Probe", []),
        [],
    )

    async def first_then_fail(messages: object) -> AIMessage:
        del messages
        if client.structured_calls == 0:
            client.structured_calls += 1
            raise ValidationError.from_exception_data("Probe", [])
        client.json_calls += 1
        raise APIConnectionError(request=httpx.Request("POST", "https://example.test"))

    client.ainvoke = first_then_fail  # type: ignore[method-assign]
    gateway = _gateway_for_structured(client)
    with pytest.raises(ModelGatewayError, match="model call failed after"):
        await gateway.complete(
            system_prompt="sys",
            user_prompt="user",
            response_model=_Probe,
            agent_id="fixer",
        )
    assert client.json_calls == 1


def _assistant_with_tool_calls(tool_calls: list[object]) -> AIMessage:
    message = AIMessage(content="calling echo")
    object.__setattr__(message, "tool_calls", tool_calls)
    return message


@pytest.mark.asyncio
async def test_malformed_tool_call_becomes_gateway_error() -> None:
    client = _FakeClient(
        [_assistant_with_tool_calls([{"name": "echo", "args": {"text": "hello"}}])]
    )
    gateway = _gateway_with_client(client)
    with pytest.raises(ModelGatewayError, match="invalid tool-call response"):
        await gateway.next_action(
            messages=[ToolLoopMessage(role="user", content="hello")],
            tools=[_echo_tool()],
            agent_id="fixer",
        )


@pytest.mark.asyncio
async def test_invalid_tool_arguments_become_gateway_error() -> None:
    client = _FakeClient(
        [
            _assistant_with_tool_calls(
                [{"id": "call-1", "name": "echo", "args": "not-a-mapping"}]
            )
        ]
    )
    gateway = _gateway_with_client(client)
    with pytest.raises(ModelGatewayError, match="invalid tool-call response"):
        await gateway.next_action(
            messages=[ToolLoopMessage(role="user", content="hello")],
            tools=[_echo_tool()],
            agent_id="fixer",
        )


@pytest.mark.asyncio
async def test_retry_does_not_repeat_schema_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0

    async def boom() -> str:
        nonlocal attempts
        attempts += 1
        raise ValidationError.from_exception_data("Probe", [])

    gateway = object.__new__(OpenAICompatibleGateway)
    gateway._timeout = 1.0
    gateway._max_attempts = 6
    monkeypatch.setattr("evoci.model.openai_compatible.asyncio.sleep", lambda _: None)
    with pytest.raises(ValidationError):
        await gateway._retry(boom)
    assert attempts == 1
