"""Typed dependency boundary for model calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

ResponseT = TypeVar("ResponseT", bound=BaseModel)


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
    ) -> ToolModelResponse:
        """Return a provider-neutral assistant turn and any requested tool calls."""
        ...

    async def finalize(
        self,
        *,
        messages: list[ToolLoopMessage],
        response_model: type[ResponseT],
        agent_id: str,
    ) -> ResponseT:
        """Produce the existing validated leaf-agent output schema."""
        ...
