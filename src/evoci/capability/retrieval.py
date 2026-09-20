"""Precision-first retrieval restricted to trial and active capabilities."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from evoci.capability.models import RegisteredSkill, SkillVersionRef
from evoci.capability.registry import CapabilityRegistry
from evoci.domain.models import CIFailure, RepoSpec, SkillHit


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


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9_]{3,}", text.lower())
        if token not in _STOPWORDS
    }


def _query(text: str) -> str:
    tokens = list(dict.fromkeys(_tokens(text)))[:32]
    return " OR ".join(f'"{token}"' for token in tokens)


def _matches_source_task(record: RegisteredSkill, task_id: str | None) -> bool:
    return bool(task_id) and any(
        source_run_id.endswith(task_id) for source_run_id in record.manifest.source_run_ids
    )


def _relevance(
    record: RegisteredSkill,
    *,
    repo: RepoSpec,
    failure: CIFailure,
    task_id: str | None,
    status: str,
) -> float | None:
    """Apply deterministic precision gates after broad FTS candidate generation."""

    if _matches_source_task(record, task_id):
        return 100.0 + (1.0 if status == "active" else 0.0)

    manifest = record.manifest
    query_tokens = _tokens(
        f"{failure.task_family} {failure.summary} {failure.log_excerpt[:1200]}"
    )
    skill_tokens = _tokens(
        " ".join(
            [
                manifest.name,
                manifest.description,
                *manifest.triggers,
                *manifest.task_families,
            ]
        )
    )
    overlap = len(query_tokens & skill_tokens)
    family_overlap = len(_tokens(failure.task_family) & skill_tokens)
    if overlap < 2 and family_overlap == 0:
        return None

    repo_tokens = _tokens(repo.full_name)
    source_tokens = _tokens(" ".join(manifest.source_run_ids))
    return (
        float(overlap)
        + 4.0 * family_overlap
        + (1.0 if repo_tokens & source_tokens else 0.0)
        + (1.0 if status == "active" else 0.0)
    )


class CapabilityRetriever:
    def __init__(
        self,
        registry: CapabilityRegistry,
        *,
        top_k: int = 2,
        trial_slots: int = 1,
    ) -> None:
        if top_k < 1 or trial_slots < 0 or trial_slots > top_k:
            raise ValueError("invalid capability retrieval quotas")
        self.registry = registry
        self.top_k = top_k
        self.trial_slots = trial_slots

    def retrieve(
        self,
        repo: RepoSpec,
        failure: CIFailure,
        *,
        task_id: str | None = None,
        operation_key: str | None = None,
    ) -> list[SkillHit]:
        expression = _query(
            f"{repo.full_name} {failure.task_family} {failure.summary} {failure.log_excerpt[:2000]}"
        )
        if not expression:
            return []

        candidate_limit = max(20, self.top_k * 8)

        def ranked(status: str) -> list[sqlite3.Row]:
            return list(
                self.registry.connection.execute(
                    """
            SELECT s.skill_id, s.version, s.package_path, s.manifest_json,
                   s.status, bm25(skill_fts) AS rank
            FROM skill_fts
            JOIN skills s ON skill_fts.skill_key = s.skill_id || ':' || s.version
            WHERE skill_fts MATCH ? AND s.status = ?
            ORDER BY rank
            LIMIT ?
            """,
                    (expression, status, candidate_limit),
                ).fetchall()
            )

        candidate_rows = [*ranked("active"), *ranked("trial")]
        if task_id:
            candidate_rows.extend(
                self.registry.connection.execute(
                    """
                    SELECT skill_id, version, package_path, manifest_json, status, 0.0 AS rank
                    FROM skills
                    WHERE status IN ('active', 'trial') AND manifest_json LIKE ?
                    """,
                    (f"%{task_id}%",),
                ).fetchall()
            )

        ranked_rows: list[tuple[float, sqlite3.Row, RegisteredSkill]] = []
        seen_refs: set[tuple[str, int]] = set()
        for row in candidate_rows:
            ref = (str(row["skill_id"]), int(row["version"]))
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            record = self.registry.get(*ref)
            if record is None:
                continue
            score = _relevance(
                record,
                repo=repo,
                failure=failure,
                task_id=task_id,
                status=str(row["status"]),
            )
            if score is not None:
                ranked_rows.append((score, row, record))
        ranked_rows.sort(key=lambda item: item[0], reverse=True)

        selected: list[tuple[float, sqlite3.Row, RegisteredSkill]] = []
        trial_count = 0
        for item in ranked_rows:
            if str(item[1]["status"]) == "trial":
                if trial_count >= self.trial_slots:
                    continue
                trial_count += 1
            selected.append(item)
            if len(selected) >= self.top_k:
                break

        rows = [row for _, row, _ in selected]
        refs = [
            SkillVersionRef(skill_id=str(row["skill_id"]), version=int(row["version"]))
            for row in rows
        ]
        self.registry.record_retrieval(refs, operation_key=operation_key)

        hits: list[SkillHit] = []
        for score, _, record in selected:
            package = Path(record.package_path)
            hits.append(
                SkillHit(
                    skill_id=record.manifest.skill_id,
                    version=record.manifest.version,
                    name=record.manifest.name,
                    description=record.manifest.description,
                    skill_md=(package / "SKILL.md").read_text(encoding="utf-8"),
                    score=score,
                    resources=[
                        file.path for file in record.manifest.files if file.path != "SKILL.md"
                    ],
                )
            )
        return hits
