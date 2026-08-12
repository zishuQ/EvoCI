"""Top-k retrieval restricted to trial and active capabilities."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from evoci.capability.models import SkillVersionRef
from evoci.capability.registry import CapabilityRegistry
from evoci.domain.models import CIFailure, RepoSpec, SkillHit


def _query(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]{2,}", text.lower())[:24]
    return " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))


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
        operation_key: str | None = None,
    ) -> list[SkillHit]:
        expression = _query(
            f"{repo.full_name} {failure.task_family} {failure.summary} {failure.log_excerpt[:2000]}"
        )
        if not expression:
            return []

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
                    (expression, status, self.top_k),
                ).fetchall()
            )

        active = ranked("active")
        trial = ranked("trial")
        rows = trial[: self.trial_slots]
        rows.extend(active[: self.top_k - len(rows)])
        if len(rows) < self.top_k:
            rows.extend(trial[self.trial_slots : self.top_k - len(rows) + self.trial_slots])
        refs = [
            SkillVersionRef(skill_id=str(row["skill_id"]), version=int(row["version"]))
            for row in rows
        ]
        self.registry.record_retrieval(refs, operation_key=operation_key)
        hits: list[SkillHit] = []
        for row in rows:
            record = self.registry.get(str(row["skill_id"]), int(row["version"]))
            if record is None:
                continue
            package = Path(record.package_path)
            hits.append(
                SkillHit(
                    skill_id=record.manifest.skill_id,
                    version=record.manifest.version,
                    name=record.manifest.name,
                    description=record.manifest.description,
                    skill_md=(package / "SKILL.md").read_text(encoding="utf-8"),
                    score=(1.0 if str(row["status"]) == "active" else 0.75)
                    / (1.0 + abs(float(row["rank"]))),
                    resources=[
                        file.path for file in record.manifest.files if file.path != "SKILL.md"
                    ],
                )
            )
        return hits
