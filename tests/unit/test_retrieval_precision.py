from pathlib import Path

from evoci.capability.models import GeneratedFile, SkillCandidate
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.capability.validator import CandidateValidator
from evoci.domain.models import CIFailure, RepoSpec


SKILL_MD = """---
name: retrieval-test-skill
description: Test retrieval precision
version: 1
---

# Purpose
Exercise retrieval precision.

# When to Use
Use only for the described failure family.

# Procedure
Inspect the failure and apply the narrow repair.

# Pitfalls
Do not apply this to unrelated failures.

# Verification
Run the targeted tests.

# Bundled Resources
No runtime resources are required.
"""


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
        skill_md=SKILL_MD,
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
    irrelevant = store.create_candidate(
        make_candidate(
            name="signed-regex-parser",
            description="Repair signed regular expression parser failures",
            triggers=["signed regex parser"],
            task_families=["parser"],
        )
    )
    assert (
        CandidateValidator(store)
        .validate_to_trial(irrelevant.manifest.skill_id, irrelevant.manifest.version)
        .passed
    )

    hits = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="pallets", name="flask"),
        CIFailure(
            summary="constructor accepts empty import name",
            log_excerpt="ValueError when name is empty",
            task_family="api",
        ),
    )

    assert hits == []
    store.close()


def test_exact_task_provenance_beats_generic_skill(tmp_path: Path) -> None:
    store = registry(tmp_path)
    validator = CandidateValidator(store)
    generic = store.create_candidate(
        make_candidate(
            name="generic-constructor-debugging",
            description="Debug constructor failures with empty values",
            triggers=["constructor empty value"],
            task_families=["api"],
        )
    )
    exact = store.create_candidate(
        make_candidate(
            name="flask-empty-name-guard",
            description="Repair Flask constructor handling for empty import names",
            triggers=["empty import name", "Flask constructor"],
            task_families=["api"],
            source_run_ids=["campaign-r1-pallets__flask-5014"],
        )
    )
    for record in (generic, exact):
        assert validator.validate_to_trial(record.manifest.skill_id, record.manifest.version).passed
    store.transition(generic.manifest.skill_id, generic.manifest.version, "active")

    hits = CapabilityRetriever(store, top_k=1, trial_slots=1).retrieve(
        RepoSpec(owner="pallets", name="flask"),
        CIFailure(
            summary="constructor accepts empty import name",
            log_excerpt="ValueError when name is empty",
            task_family="api",
        ),
        task_id="pallets__flask-5014",
    )

    assert [hit.skill_id for hit in hits] == [exact.manifest.skill_id]
    assert hits[0].score >= 100
    store.close()
