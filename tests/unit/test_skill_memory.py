from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evoci.capability.models import SkillMemoryEntry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.domain.models import CIFailure, RepoSpec
from evoci.graph.builder import persist_run_outcome
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from tests.integration.test_graph import (
    FakeCoordinator,
    FakeDiagnoser,
    FakeFixer,
    FakeInvestigator,
    initial_state,
    make_runtime,
    task,
)
from tests.unit.test_capability import SKILL_MD, candidate, registry


def _entry(
    run_id: str,
    *,
    outcome: str = "success",
    lesson: str = "Inspect hook signatures first.",
    created_at: datetime | None = None,
) -> SkillMemoryEntry:
    return SkillMemoryEntry(
        run_id=run_id,
        repository="pytest-dev/pytest",
        task_summary="Repair parametrized fixture collection.",
        outcome=outcome,  # type: ignore[arg-type]
        lesson=lesson,
        created_at=created_at or datetime.now(UTC),
    )


def test_used_skill_success_appends_memory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    assert store.append_skill_memory(
        created.manifest.skill_id, _entry("run-123", outcome="success")
    )
    text = (tmp_path / "skills" / created.manifest.skill_id / "memory.md").read_text(
        encoding="utf-8"
    )
    assert "<!-- evoci-skill-memory:run-123:pytest-import-debugging -->" in text
    assert "Outcome: success" in text
    store.close()


def test_used_skill_failure_appends_memory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    assert store.append_skill_memory(
        created.manifest.skill_id, _entry("run-123", outcome="failure")
    )
    entries = store.read_skill_memory(created.manifest.skill_id)
    assert entries[0].outcome == "failure"
    store.close()


def test_retrieved_but_unused_skill_does_not_append_memory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    CapabilityRetriever(store).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(
            summary="pytest ModuleNotFoundError",
            log_excerpt="pytest import failed",
            task_family="test",
        ),
    )
    memory = (tmp_path / "skills" / created.manifest.skill_id / "memory.md").read_text(
        encoding="utf-8"
    )
    assert memory == ""
    store.close()


def test_same_run_does_not_append_memory_twice(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    first = store.append_skill_memory(created.manifest.skill_id, _entry("run-123"))
    second = store.append_skill_memory(created.manifest.skill_id, _entry("run-123"))
    assert first is True
    assert second is False
    text = (tmp_path / "skills" / created.manifest.skill_id / "memory.md").read_text(
        encoding="utf-8"
    )
    assert text.count("<!-- evoci-skill-memory:run-123:pytest-import-debugging -->") == 1
    store.close()


def test_parallel_appends_do_not_interleave(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    skill_id = created.manifest.skill_id

    def write(index: int) -> bool:
        return store.append_skill_memory(
            skill_id,
            _entry(f"run-{index}", lesson=f"lesson-{index} " + ("x" * 200)),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(8)))
    assert all(results)
    text = (tmp_path / "skills" / skill_id / "memory.md").read_text(encoding="utf-8")
    for index in range(8):
        assert f"<!-- evoci-skill-memory:run-{index}:{skill_id} -->" in text
        assert f"lesson-{index}" in text
    store.close()


def test_recent_memory_is_loaded_with_skill(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    store.append_skill_memory(created.manifest.skill_id, _entry("run-1", outcome="failure"))
    store.append_skill_memory(created.manifest.skill_id, _entry("run-2", outcome="success"))
    catalog = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(
            summary="pytest ModuleNotFoundError",
            log_excerpt="pytest import failed",
            task_family="test",
        ),
    )
    assert [entry.skill_id for entry in catalog.entries] == [created.manifest.skill_id]
    from evoci.tools.policy import FIXER_CAPABILITIES
    from evoci.tools.registry import create_worker_registry

    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    loaded = tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    assert any(line.startswith("- SUCCESS:") for line in loaded["usage_memory"])
    assert any(line.startswith("- FAILURE:") for line in loaded["usage_memory"])
    tools.close()
    store.close()


def test_memory_context_is_bounded(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    for index in range(12):
        store.append_skill_memory(
            created.manifest.skill_id,
            _entry(f"run-{index}", lesson=("L" * 800) + f"-{index}"),
        )
    entries = store.read_skill_memory(created.manifest.skill_id, limit=8, max_chars=4_000)
    assert len(entries) <= 8
    assert sum(len(entry.lesson) for entry in entries) <= 4_000 + 800
    store.close()


def test_skill_update_preserves_memory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    store.append_skill_memory(created.manifest.skill_id, _entry("run-keep"))
    store.update_skill(
        created.manifest.skill_id,
        candidate().model_copy(
            update={"skill_md": SKILL_MD.replace("compare import roots", "updated procedure")}
        ),
    )
    text = (tmp_path / "skills" / created.manifest.skill_id / "memory.md").read_text(
        encoding="utf-8"
    )
    assert "run-keep" in text
    store.close()


@pytest.mark.asyncio
async def test_learning_failure_writes_fallback_memory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())

    class ExplodingMiner:
        def should_mine(self, **kwargs: object) -> bool:
            del kwargs
            return True

        async def decide(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("miner exploded")

    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="fallback-run",
        event_type=EventType.SKILL_USED,
        payload={"skill_id": created.manifest.skill_id, "success": True},
    )
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("a")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(),
        ),
        capability_registry=store,
        skill_miner=ExplodingMiner(),  # type: ignore[arg-type]
        recorder=recorder,
    )
    await persist_run_outcome(
        runtime,
        initial_state(tmp_path, run_id="fallback-run"),
        success=True,
        failure_reason=None,
    )
    entries = store.read_skill_memory(created.manifest.skill_id)
    assert entries
    assert entries[-1].run_id == "fallback-run"
    assert entries[-1].outcome == "success"
    store.close()


@pytest.mark.asyncio
async def test_infrastructure_failure_does_not_write_memory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="infra-run",
        event_type=EventType.SKILL_USED,
        payload={"skill_id": created.manifest.skill_id, "success": True},
    )
    runtime = replace(
        make_runtime(
            tmp_path,
            FakeCoordinator([[task("a")]]),
            FakeInvestigator(),
            FakeDiagnoser(),
            FakeFixer(),
        ),
        capability_registry=store,
        recorder=recorder,
    )
    state = initial_state(tmp_path, run_id="infra-run")
    state["failure_class"] = "infrastructure"
    await persist_run_outcome(
        runtime,
        state,
        success=False,
        failure_reason="docker missing",
        failure_class="infrastructure",
    )
    text = (tmp_path / "skills" / created.manifest.skill_id / "memory.md").read_text(
        encoding="utf-8"
    )
    assert "infra-run" not in text
    store.close()
