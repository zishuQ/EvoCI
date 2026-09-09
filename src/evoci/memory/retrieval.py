"""Namespace-aware, budgeted memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from evoci.domain.models import CIFailure, MemoryHit, RepoSpec
from evoci.memory.models import RetrievalTelemetry
from evoci.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class MemoryRetrieval:
    hits: list[MemoryHit]
    telemetry: RetrievalTelemetry


class MemoryRetriever:
    def __init__(self, store: MemoryStore, *, context_limit_chars: int = 6_000) -> None:
        self.store = store
        self.context_limit_chars = context_limit_chars

    def retrieve(self, repo: RepoSpec, failure: CIFailure) -> MemoryRetrieval:
        query = f"{failure.task_family} {failure.summary} {failure.log_excerpt[:2000]}"
        namespaces = [
            f"repo:{repo.full_name}",
            f"family:{failure.task_family}",
            "global:ci",
        ]
        retrieved = self.store.search_semantic(query, namespaces=namespaces, limit=5)
        retrieved += self.store.search_episodes(query, repo=repo.full_name, limit=3)
        selected: list[MemoryHit] = []
        used_chars = 0
        for hit in retrieved:
            remaining = self.context_limit_chars - used_chars
            if remaining <= 0:
                break
            content = hit.content
            if len(content) > remaining:
                marker = "\n[truncated]"
                keep = max(0, remaining - len(marker))
                content = content[:keep] + marker
            if content != hit.content:
                hit = hit.model_copy(update={"content": content})
            selected.append(hit)
            used_chars += len(hit.content)
        return MemoryRetrieval(
            hits=selected,
            telemetry=RetrievalTelemetry(
                query=query,
                retrieved_ids=[hit.memory_id for hit in retrieved],
                selected_ids=[hit.memory_id for hit in selected],
                context_chars=used_chars,
            ),
        )
