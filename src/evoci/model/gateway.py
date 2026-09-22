"""Typed dependency boundary for model calls."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

ResponseT = TypeVar("ResponseT", bound=BaseModel)

RequestKind = Literal[
    "structured",
    "plain_json_correction",
    "tool_action",
    "structured_finalize",
]


class ModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    model_name: str | None = None
    provider_request_id: str | None = None
    request_kind: RequestKind

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


UsageObserver = Callable[[ModelUsage], None]


class ModelGatewayError(RuntimeError):
    """A typed model-call failure after retry exhaustion."""


@dataclass(frozen=True, slots=True)
class ModelCall:
    system_prompt: str
    user_prompt: str
    response_model: type[BaseModel]
    agent_id: str


class ToolDefinition(BaseModel):
    """Provider-neutral tool definition exposed to a leaf model."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolLoopMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    reasoning_content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)


class ToolModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str = ""
    reasoning_content: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelGateway(Protocol):
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
        """Return a validated structured response."""
        ...


class ToolLoopGateway(ModelGateway, Protocol):
    async def next_action(
        self,
        *,
        messages: list[ToolLoopMessage],
        tools: list[ToolDefinition],
        agent_id: str,
        usage_observer: UsageObserver | None = None,
        max_output_tokens: int | None = None,
    ) -> ToolModelResponse:
        """Return a provider-neutral assistant turn and any requested tool calls."""
        ...

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
        usage_observer: UsageObserver | None = None,
        max_output_tokens: int | None = None,
    ) -> ResponseT:
        """Produce the existing validated leaf-agent output schema."""
        ...
