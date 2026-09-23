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


def _clip(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _preview(hit: MemoryHit) -> MemoryHit:
    """Keep a brief catalog entry; the complete content is read by ID."""
    if not hit.namespace.startswith("episode:"):
        return hit.model_copy(update={"content": f"FACT={_clip(hit.content, 360)}"})
    lines = hit.content.splitlines()
    outcome = lines[0] if lines else "OUTCOME=unknown"
    fields = (
        "FAILURE_CLASS=",
        "FAILURE_REASON=",
        "ROOT_CAUSE=",
        "SUCCESSFUL_FIX=",
        "ATTEMPTED_FILES=",
    )
    summary = next(
        (
            line for line in lines[1:]
            if line.strip() and not line.startswith(fields) and "=" not in line[:32]
            and not line.startswith("HYPOTHESIS ")
        ),
        "",
    )
    parts = [outcome, f"SUMMARY={_clip(summary, 180)}"]
    if outcome == "OUTCOME=failed":
        for prefix, limit in (
            ("FAILURE_CLASS=", 40),
            ("FAILURE_REASON=", 220),
            ("ATTEMPTED_FILES=", 100),
        ):
            value = next((line[len(prefix):] for line in lines if line.startswith(prefix)), "")
            if value:
                parts.append(f"{prefix}{_clip(value, limit)}")
    else:
        for prefix, limit in (("ROOT_CAUSE=", 150), ("SUCCESSFUL_FIX=", 180)):
            value = next((line[len(prefix):] for line in lines if line.startswith(prefix)), "")
            if value:
                parts.append(f"{prefix}{_clip(value, limit)}")
    return hit.model_copy(update={"content": "\n".join(parts)})


def _truncate_preserving_outcome(content: str, remaining: int) -> str:
    if remaining <= 0:
        return ""
    if len(content) <= remaining:
        return content
    marker = "\n[truncated]"
    if remaining <= len(marker):
        return content[:remaining]
    return content[: remaining - len(marker)] + marker


class MemoryRetriever:
    def __init__(self, store: MemoryStore, *, context_limit_chars: int = 10_000) -> None:
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
        # Repository facts first, then exact same-repository episodes; broad
        # matches may include episodes from another repository.
        retrieved = self.store.search_long_term(
            query, repository=repo.full_name, limit=4
        )
        retrieved += self.store.search_episodes_by_fingerprint(
            repo=repo.full_name, fingerprint=fingerprint, limit=2
        )
        retrieved += self.store.search_episodes(query, repo=repo.full_name, limit=4)

        deduplicated: list[MemoryHit] = []
        seen_ids: set[str] = set()
        for hit in retrieved:
            if hit.memory_id in seen_ids:
                continue
            seen_ids.add(hit.memory_id)
            deduplicated.append(_preview(hit))

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
