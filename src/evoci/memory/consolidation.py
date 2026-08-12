"""Model-assisted memory extraction with harness-owned commits."""

from __future__ import annotations

import json
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from evoci.domain.models import Diagnosis, FixerOutput, VerificationResult
from evoci.memory.models import MemoryCandidate, SemanticMemory
from evoci.memory.store import MemoryStore
from evoci.model.gateway import ModelGateway


class MemoryConsolidator(Protocol):
    async def propose(
        self,
        *,
        run_id: str,
        repo: str,
        task_family: str,
        diagnosis: Diagnosis,
        patch: FixerOutput,
        verification: VerificationResult,
    ) -> MemoryCandidate: ...


class ModelMemoryConsolidator:
    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    async def propose(
        self,
        *,
        run_id: str,
        repo: str,
        task_family: str,
        diagnosis: Diagnosis,
        patch: FixerOutput,
        verification: VerificationResult,
    ) -> MemoryCandidate:
        payload = {
            "run_id": run_id,
            "repo": repo,
            "task_family": task_family,
            "diagnosis": diagnosis.model_dump(),
            "patch": patch.model_dump(),
            "verification": verification.model_dump(),
        }
        return await self.gateway.complete(
            system_prompt=(
                "Extract one durable fact useful in future CI runs, or none. Keep repo-specific "
                "facts in repo:<owner>/<repo>, family knowledge in family:<type>, and only truly "
                "universal knowledge in global:ci."
            ),
            user_prompt=json.dumps(payload),
            response_model=MemoryCandidate,
            agent_id="memory-consolidator",
        )


def commit_candidate(
    store: MemoryStore,
    candidate: MemoryCandidate,
    *,
    run_id: str,
    operation_key: str | None = None,
) -> str | None:
    if operation_key:
        existing = store.operation_result(operation_key)
        if existing is not None:
            memory_id = existing.get("memory_id")
            return str(memory_id) if memory_id else None
    if candidate.type != "semantic" or not candidate.content or not candidate.namespace:
        if operation_key:
            store.record_operation(operation_key, {"memory_id": None})
        return None
    memory = SemanticMemory(
        id=str(uuid5(NAMESPACE_URL, f"evoci:{run_id}:{candidate.namespace}:{candidate.content}")),
        namespace=candidate.namespace,
        content=candidate.content,
        importance=0.7,
        confidence=candidate.confidence,
        source_run_ids=[run_id],
    )
    store.add_semantic(memory, operation_key=operation_key)
    return memory.id
