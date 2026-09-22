"""Model-assisted repository fact extraction with harness-owned commits."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from evoci.memory.models import LongTermFactCandidate, LongTermMemory
from evoci.memory.store import MemoryStore
from evoci.model.gateway import ModelGateway, UsageObserver
from evoci.runtime.learning_payload import LearningInput


def normalize_fact_content(content: str) -> str:
    text = re.sub(r"\s+", " ", content.strip().lower())
    return re.sub(r"[.。!！?？;；,，:：]+$", "", text)  # noqa: RUF001


def fact_memory_id(repository: str, content: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"evoci:{repository}:{normalize_fact_content(content)}"))


class MemoryConsolidator(Protocol):
    async def propose(
        self,
        *,
        run_id: str,
        repo: str,
        task_family: str,
        learning: LearningInput,
        existing_facts: Sequence[str] = (),
        usage_observer: UsageObserver | None = None,
    ) -> LongTermFactCandidate: ...


class ModelMemoryConsolidator:
    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    async def propose(
        self,
        *,
        run_id: str,
        repo: str,
        task_family: str,
        learning: LearningInput,
        existing_facts: Sequence[str] = (),
        usage_observer: UsageObserver | None = None,
    ) -> LongTermFactCandidate:
        payload = {
            "run_id": run_id,
            "repo": repo,
            "task_family": task_family,
            "learning": learning.model_dump(mode="json"),
            "existing_repository_facts": list(existing_facts),
        }
        return await self.gateway.complete(
            system_prompt=(
                "Extract one durable fact about this repository, or none. Facts describe "
                "what the repository is like: test runner, generated directories, legacy "
                "APIs, or similar stable conditions. Do not write workflows or procedures; "
                "those belong in Skills. Do not output a namespace or repository field; the "
                "harness binds the fact to the current repository. If existing_repository_facts "
                "already covers the useful fact, return type=none."
            ),
            user_prompt=json.dumps(payload),
            response_model=LongTermFactCandidate,
            agent_id="memory-consolidator",
            usage_observer=usage_observer,
        )


def commit_candidate(
    store: MemoryStore,
    candidate: LongTermFactCandidate,
    *,
    run_id: str,
    repository: str,
    operation_key: str | None = None,
) -> str | None:
    if operation_key:
        existing = store.operation_result(operation_key)
        if existing is not None:
            if existing.get("created") and existing.get("memory_id"):
                return str(existing["memory_id"])
            return None
    if candidate.type != "fact" or not candidate.content:
        if operation_key:
            store.record_operation(operation_key, {"memory_id": None, "created": False})
        return None
    normalized = normalize_fact_content(candidate.content)
    if not normalized:
        if operation_key:
            store.record_operation(operation_key, {"memory_id": None, "created": False})
        return None
    memory = LongTermMemory(
        id=fact_memory_id(repository, candidate.content),
        repository=repository,
        content=candidate.content.strip(),
        confidence=candidate.confidence,
        source_run_ids=[run_id],
    )
    created = store.add_long_term(memory, operation_key=operation_key)
    return memory.id if created else None
