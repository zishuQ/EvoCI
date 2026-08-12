from __future__ import annotations

from collections.abc import Awaitable, Callable

import httpx
import pytest
from openai import APIConnectionError

from evoci.model.openai_compatible import OpenAICompatibleGateway


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
