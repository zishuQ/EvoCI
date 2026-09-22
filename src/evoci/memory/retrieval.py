"""Namespace-aware, budgeted memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from evoci.domain.models import CIFailure, MemoryHit, RepoSpec
from evoci.memory.fingerprint import failure_fingerprint, stable_error_signature
from evoci.memory.models import RetrievalTelemetry
from evoci.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class MemoryRetrieval:
    hits: list[MemoryHit]
    telemetry: RetrievalTelemetry


def _truncate_preserving_outcome(content: str, remaining: int) -> str:
    if remaining <= 0:
        return ""
    if len(content) <= remaining:
        return content
    marker = "\n[truncated]"
    keep = max(0, remaining - len(marker))
    if content.startswith("OUTCOME="):
        boundary = content.find("\n")
        if boundary == -1:
            boundary = min(len(content), 32)
        else:
            boundary += 1
        capped = remaining - len(marker) if remaining > len(marker) else keep
        keep = max(keep, min(boundary, capped))
    return content[:keep] + marker


class MemoryRetriever:
    def __init__(self, store: MemoryStore, *, context_limit_chars: int = 6_000) -> None:
        self.store = store
        self.context_limit_chars = context_limit_chars

    def retrieve(
        self,
        repo: RepoSpec,
        failure: CIFailure,
    ) -> MemoryRetrieval:
        fingerprint = failure_fingerprint(repo, failure)
        signature = stable_error_signature(failure.log_excerpt, failure.summary)
        command = " ".join(part for argv in failure.failed_commands for part in argv)
        query = (
            f"{failure.task_family} {failure.summary} {command} "
            f"{' '.join(signature.get('messages', []))} {failure.log_excerpt[:800]}"
        )
        retrieved = list(
            self.store.search_episodes_by_fingerprint(
                repo=repo.full_name,
                fingerprint=fingerprint,
                limit=2,
            )
        )
        retrieved += self.store.search_episodes(query, repo=repo.full_name, limit=4)
        retrieved += self.store.search_long_term(
            query,
            repository=repo.full_name,
            limit=4,
        )

        deduplicated: list[MemoryHit] = []
        seen_ids: set[str] = set()
        for hit in retrieved:
            if hit.memory_id in seen_ids:
                continue
            seen_ids.add(hit.memory_id)
            deduplicated.append(hit)

        selected: list[MemoryHit] = []
        used_chars = 0
        for hit in deduplicated:
            remaining = self.context_limit_chars - used_chars
            if remaining <= 0:
                break
            content = _truncate_preserving_outcome(hit.content, remaining)
            if not content:
                break
            if content != hit.content:
                hit = hit.model_copy(update={"content": content})
            selected.append(hit)
            used_chars += len(hit.content)
        return MemoryRetrieval(
            hits=selected,
            telemetry=RetrievalTelemetry(
                query=query,
                retrieved_ids=[hit.memory_id for hit in deduplicated],
                selected_ids=[hit.memory_id for hit in selected],
                context_chars=used_chars,
            ),
        )
