from pathlib import Path

from evoci.capability.models import GeneratedFile, SkillCandidate
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.domain.models import CIFailure, RepoSpec
from tests.unit.test_capability import SKILL_MD


def make_candidate(
    *,
    name: str,
    description: str,
    triggers: list[str],
    task_families: list[str],
    source_run_ids: list[str] | None = None,
) -> SkillCandidate:
    return SkillCandidate(
        name=name,
        description=description,
        triggers=triggers,
        task_families=task_families,
        skill_md=SKILL_MD.replace("pytest-import-debugging", name),
        tests=[
            GeneratedFile(
                path="tests/test_skill.py",
                content="def test_skill_package():\n    assert True\n",
            )
        ],
        source_run_ids=source_run_ids or ["campaign-r1-other-task"],
        confidence=0.9,
    )


def registry(tmp_path: Path) -> CapabilityRegistry:
    return CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")


def test_irrelevant_skill_is_not_injected(tmp_path: Path) -> None:
    store = registry(tmp_path)
    store.create_skill(
        make_candidate(
            name="signed-regex-parser",
            description="Repair signed regular expression parser failures",
            triggers=["signed regex parser"],
            task_families=["parser"],
        )
    )

    catalog = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="pallets", name="flask"),
        CIFailure(
            summary="constructor accepts empty import name",
            log_excerpt="ValueError when name is empty",
            task_family="api",
        ),
    )

    assert all(
        set(entry.model_dump()) == {"skill_id", "name", "description"} for entry in catalog.entries
    )
    store.close()


def test_task_id_does_not_affect_skill_ranking(tmp_path: Path) -> None:
    store = registry(tmp_path)
    generic = store.create_skill(
        make_candidate(
            name="generic-constructor-debugging",
            description="Debug constructor failures with empty values",
            triggers=["constructor empty value"],
            task_families=["api"],
        )
    )
    exact = store.create_skill(
        make_candidate(
            name="flask-empty-name-guard",
            description="Repair Flask constructor handling for empty import names",
            triggers=["empty import name", "Flask constructor"],
            task_families=["api"],
            source_run_ids=["campaign-r1-pallets__flask-5014"],
        )
    )

    retriever = CapabilityRetriever(store, top_k=2)
    catalog = retriever.retrieve(
        RepoSpec(owner="pallets", name="flask"),
        CIFailure(
            summary="constructor accepts empty import name",
            log_excerpt="ValueError when name is empty",
            task_family="api",
        ),
    )
    assert {entry.skill_id for entry in catalog.entries} <= {
        generic.manifest.skill_id,
        exact.manifest.skill_id,
    }
    store.close()


def test_source_run_id_does_not_force_match(tmp_path: Path) -> None:
    store = registry(tmp_path)
    store.create_skill(
        make_candidate(
            name="unrelated-django-orm",
            description="Repair Django ORM query compilation",
            triggers=["django orm query"],
            task_families=["orm"],
            source_run_ids=["campaign-r1-pallets__flask-5014"],
        )
    )
    catalog = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="pallets", name="flask"),
        CIFailure(
            summary="constructor accepts empty import name",
            log_excerpt="ValueError when name is empty",
            task_family="api",
        ),
    )
    assert all(
        set(entry.model_dump()) == {"skill_id", "name", "description"} for entry in catalog.entries
    )
    store.close()


def test_skill_retrieval_includes_recent_memory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(
        make_candidate(
            name="flask-empty-name-guard",
            description="Repair Flask constructor handling for empty import names",
            triggers=["empty import name", "Flask constructor"],
            task_families=["api"],
        )
    )
    from datetime import UTC, datetime

    from evoci.capability.models import SkillMemoryEntry

    store.append_skill_memory(
        created.manifest.skill_id,
        SkillMemoryEntry(
            run_id="run-memory",
            repository="pallets/flask",
            task_summary="constructor empty name",
            outcome="failure",
            lesson="This revision uses an older factory signature.",
            created_at=datetime.now(UTC),
        ),
    )
    catalog = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="pallets", name="flask"),
        CIFailure(
            summary="constructor accepts empty import name",
            log_excerpt="ValueError when name is empty",
            task_family="api",
        ),
    )
    assert catalog.entries
    assert not hasattr(catalog.entries[0], "memory") or not getattr(
        catalog.entries[0], "memory", None
    )
    store.close()


def test_failure_memory_is_presented_as_counterevidence(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(
        make_candidate(
            name="flask-empty-name-guard",
            description="Repair Flask constructor handling for empty import names",
            triggers=["empty import name", "Flask constructor"],
            task_families=["api"],
        )
    )
    from datetime import UTC, datetime

    from evoci.capability.models import SkillMemoryEntry

    store.append_skill_memory(
        created.manifest.skill_id,
        SkillMemoryEntry(
            run_id="run-fail",
            repository="pallets/flask",
            task_summary="constructor empty name",
            outcome="failure",
            lesson="Do not assume the current factory matches the previous revision.",
            created_at=datetime.now(UTC),
        ),
    )
    catalog = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="pallets", name="flask"),
        CIFailure(
            summary="constructor accepts empty import name",
            log_excerpt="ValueError when name is empty",
            task_family="api",
        ),
    )
    assert catalog.entries
    assert "memory" not in catalog.entries[0].model_dump()
    store.close()
