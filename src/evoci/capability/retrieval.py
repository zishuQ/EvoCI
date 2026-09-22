"""Precision-first ranking of enabled capabilities into a compact catalog."""

from __future__ import annotations

import json
import re

from evoci.capability.registry import CapabilityRegistry
from evoci.domain.models import CIFailure, RepoSpec, SkillCatalog, SkillCatalogEntry

_STOPWORDS = {
    "bug",
    "command",
    "error",
    "failed",
    "failure",
    "file",
    "fix",
    "none",
    "python",
    "pytest",
    "test",
    "tests",
}
_MAX_DESCRIPTION_CHARS = 400


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        if token not in _STOPWORDS
    }


def _query(text: str) -> str:
    tokens = list(dict.fromkeys(_tokens(text)))[:32]
    return " OR ".join(f'"{token}"' for token in tokens)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1].rstrip() + "…"


def _entry_chars(entry: SkillCatalogEntry) -> int:
    return len(entry.skill_id) + len(entry.name) + len(entry.description)


def _rank_score(
    *,
    name: str,
    description: str,
    triggers: list[str],
    task_families: list[str],
    repo: RepoSpec,
    failure: CIFailure,
) -> float:
    del repo
    query_tokens = _tokens(
        f"{failure.task_family} {failure.summary} {failure.log_excerpt[:1200]}"
    )
    skill_tokens = _tokens(" ".join([name, description, *triggers, *task_families]))
    overlap = len(query_tokens & skill_tokens)
    family_overlap = len(_tokens(failure.task_family) & skill_tokens)
    return float(overlap) + 4.0 * family_overlap


def _fit_catalog(
    entries: list[SkillCatalogEntry], limit: int
) -> tuple[list[SkillCatalogEntry], int, int]:
    fitted = [
        entry.model_copy(
            update={"description": _truncate(entry.description, _MAX_DESCRIPTION_CHARS)}
        )
        if len(entry.description) > _MAX_DESCRIPTION_CHARS
        else entry
        for entry in entries
    ]
    omitted = 0
    while fitted and sum(_entry_chars(entry) for entry in fitted) > limit:
        if len(fitted) == 1:
            only = fitted[0]
            room = limit - len(only.skill_id) - len(only.name)
            fitted = [
                only.model_copy(update={"description": _truncate(only.description, max(room, 0))})
            ]
            break
        fitted.pop()
        omitted += 1
    return fitted, omitted, sum(_entry_chars(entry) for entry in fitted)


class CapabilityRetriever:
    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        top_k: int = 2,
        catalog_limit_chars: int = 8_000,
    ) -> None:
        if top_k < 1:
            raise ValueError("invalid capability retrieval quotas")
        if catalog_limit_chars < 1:
            raise ValueError("skill catalog character budget must be positive")
        self.registry = registry
        self.top_k = top_k
        self.catalog_limit_chars = catalog_limit_chars

    def retrieve(
        self,
        repo: RepoSpec,
        failure: CIFailure,
        *,
        operation_key: str | None = None,
    ) -> SkillCatalog:
        query_text = (
            f"{repo.full_name} {failure.task_family} {failure.summary} {failure.log_excerpt[:2000]}"
        )
        expression = _query(query_text)
        fts_rank: dict[str, float] = {}
        if expression:
            rows = self.registry.connection.execute(
                """
                SELECT s.skill_id, bm25(skill_fts) AS rank
                FROM skill_fts
                JOIN skills s ON skill_fts.skill_id = s.skill_id
                WHERE skill_fts MATCH ? AND s.enabled = 1
                """,
                (expression,),
            ).fetchall()
            for row in rows:
                fts_rank[str(row["skill_id"])] = abs(float(row["rank"]))

        ranked: list[tuple[float, str, SkillCatalogEntry]] = []
        for row in self.registry.connection.execute(
            """
            SELECT skill_id, manifest_json FROM skills WHERE enabled = 1 ORDER BY skill_id
            """
        ).fetchall():
            skill_id = str(row["skill_id"])
            manifest = json.loads(str(row["manifest_json"]))
            name = str(manifest.get("name") or skill_id)
            description = str(manifest.get("description") or "")
            triggers = [str(item) for item in manifest.get("triggers") or []]
            families = [str(item) for item in manifest.get("task_families") or []]
            overlap = _rank_score(
                name=name,
                description=description,
                triggers=triggers,
                task_families=families,
                repo=repo,
                failure=failure,
            )
            fts_boost = 1.0 / (1.0 + fts_rank[skill_id]) if skill_id in fts_rank else 0.0
            ranked.append(
                (
                    overlap + fts_boost,
                    skill_id,
                    SkillCatalogEntry(skill_id=skill_id, name=name, description=description),
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        entries = [entry for _, _, entry in ranked]
        fitted, omitted, context_chars = _fit_catalog(entries, self.catalog_limit_chars)
        self.registry.record_retrieval(
            [entry.skill_id for entry in fitted], operation_key=operation_key
        )
        return SkillCatalog(entries=fitted, omitted_count=omitted, context_chars=context_chars)
