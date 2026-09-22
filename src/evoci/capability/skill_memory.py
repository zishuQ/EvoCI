"""Append-only skill usage memory stored beside the current package."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime

from evoci.capability.models import SkillMemoryEntry

MEMORY_MARKER_RE = re.compile(r"<!-- evoci-skill-memory:([^:]+):(\S+) -->")
_BLOCK_RE = re.compile(
    r"<!-- evoci-skill-memory:(?P<run_id>[^:]+):(?P<skill_id>\S+) -->\s*"
    r"## (?P<created_at>\S+)\s*"
    r"- Run: (?P<run>.*?)\n"
    r"- Repository: (?P<repository>.*?)\n"
    r"- Outcome: (?P<outcome>success|failure)\n"
    r"- Task: (?P<task>.*?)\n"
    r"- Lesson: (?P<lesson>.*)",
    re.DOTALL,
)


def memory_marker(run_id: str, skill_id: str) -> str:
    return f"<!-- evoci-skill-memory:{run_id}:{skill_id} -->"


def format_skill_memory_block(skill_id: str, entry: SkillMemoryEntry) -> str:
    created = entry.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    timestamp = created.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"{memory_marker(entry.run_id, skill_id)}\n\n"
        f"## {timestamp}\n\n"
        f"- Run: {entry.run_id}\n"
        f"- Repository: {entry.repository}\n"
        f"- Outcome: {entry.outcome}\n"
        f"- Task: {entry.task_summary}\n"
        f"- Lesson: {entry.lesson}\n"
    )


def split_memory_blocks(text: str) -> list[str]:
    parts = re.split(r"(?=<!-- evoci-skill-memory:)", text)
    return [
        part.strip()
        for part in parts
        if part.strip().startswith("<!-- evoci-skill-memory:")
    ]


def parse_skill_memory_entries(text: str) -> list[SkillMemoryEntry]:
    entries: list[SkillMemoryEntry] = []
    for block in split_memory_blocks(text):
        match = _BLOCK_RE.search(block)
        if match is None:
            continue
        created_raw = match.group("created_at")
        try:
            created_at = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        except ValueError:
            created_at = datetime.now(UTC)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        entries.append(
            SkillMemoryEntry(
                run_id=match.group("run_id").strip(),
                repository=match.group("repository").strip(),
                task_summary=match.group("task").strip(),
                outcome=match.group("outcome"),  # type: ignore[arg-type]
                lesson=match.group("lesson").strip(),
                created_at=created_at,
            )
        )
    return entries


def format_skill_memory_lines(entries: Sequence[SkillMemoryEntry]) -> list[str]:
    return [f"- {entry.outcome.upper()}: {entry.lesson}" for entry in entries]


def merge_memory_texts(texts: list[str]) -> str:
    seen: set[str] = set()
    blocks: list[str] = []
    for text in texts:
        for block in split_memory_blocks(text):
            match = MEMORY_MARKER_RE.search(block)
            marker = match.group(0) if match is not None else block
            if marker in seen:
                continue
            seen.add(marker)
            blocks.append(block.strip())
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"
