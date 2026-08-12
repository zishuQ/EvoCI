from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from evoci.capability.curator import (
    CuratorPipeline,
    CuratorReview,
    DeterministicCurator,
    ModelCurator,
)
from evoci.capability.execution import run_skill_script
from evoci.capability.materializer import CapabilityMaterializer
from evoci.capability.models import (
    GeneratedFile,
    SkillCandidate,
    SkillPermissions,
    SkillVersionRef,
)
from evoci.capability.promotion import TrialPromotionPolicy
from evoci.capability.registry import CapabilityRegistry, CapabilityRegistryError
from evoci.capability.retrieval import CapabilityRetriever
from evoci.capability.validator import CandidateValidator
from evoci.domain.models import CIFailure, RepoSpec
from evoci.tools.policy import PolicyViolation

SKILL_MD = """---
name: pytest-import-debugging
description: Diagnose Python test import failures
version: 1
---

# Purpose
Diagnose import failures without weakening tests.

# When to Use
Use for pytest import and module discovery failures.

# Procedure
Run the bundled inspector and compare import roots.

# Pitfalls
Do not mutate the environment globally.

# Verification
Re-run the targeted pytest command.

# Bundled Resources
`scripts/inspect_imports.py` prints the current import root.
"""


def candidate(*, script: str = "print('imports-ok')\n") -> SkillCandidate:
    return SkillCandidate(
        name="pytest-import-debugging",
        description="Diagnose Python pytest import failures",
        triggers=["pytest import", "ModuleNotFoundError"],
        task_families=["test"],
        skill_md=SKILL_MD,
        scripts=[GeneratedFile(path="scripts/inspect_imports.py", content=script, executable=True)],
        tests=[
            GeneratedFile(
                path="tests/test_inspect.py",
                content=(
                    "import unittest\n"
                    "class TestInspector(unittest.TestCase):\n"
                    "    def test_fixture(self):\n"
                    "        self.assertTrue(True)\n"
                ),
            )
        ],
        source_run_ids=["run-1"],
        confidence=0.9,
        permissions=SkillPermissions(execute=True),
    )


def registry(tmp_path: Path) -> CapabilityRegistry:
    return CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")


def test_candidate_validates_trials_promotes_and_retrieves(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_candidate(candidate())
    result = CandidateValidator(store).validate_to_trial(
        created.manifest.skill_id, created.manifest.version
    )
    assert result.passed
    assert result.tests_run == 1
    store.transition(created.manifest.skill_id, created.manifest.version, "active")

    hits = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(
            summary="pytest ModuleNotFoundError",
            log_excerpt="pytest import failed",
            task_family="test",
        ),
    )
    assert [hit.skill_id for hit in hits] == ["pytest-import-debugging"]
    assert store.stats("pytest-import-debugging", 1).retrieval_count == 1
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    materialized = CapabilityMaterializer(store, tmp_path / "runtime").materialize(
        hits, run_id="materialize", workspace=workspace
    )
    script = materialized[0] / "scripts/inspect_imports.py"
    assert script.read_text() == "print('imports-ok')\n"
    assert script.stat().st_mode & 0o222 == 0
    assert all(workspace.resolve() not in path.resolve().parents for path in materialized)
    assert not (workspace / ".evoci_runtime").exists()
    with pytest.raises(PolicyViolation, match="outside target workspace"):
        CapabilityMaterializer(store, workspace / "runtime").materialize(
            hits, run_id="unsafe", workspace=workspace
        )
    store.close()


def test_candidate_rejects_path_traversal(tmp_path: Path) -> None:
    store = registry(tmp_path)
    unsafe = candidate().model_copy(
        update={"scripts": [GeneratedFile(path="../../escape.py", content="print('no')")]}
    )
    with pytest.raises(PolicyViolation):
        store.create_candidate(unsafe)
    store.close()


@pytest.mark.parametrize(
    "script",
    [
        "TOKEN = 'sk-abcdefghijklmnopqrstuvwxyz'\n",
        "from pathlib import Path\nprint(Path('.env').read_text())\n",
        "import subprocess\nsubprocess.run(['whoami'])\n",
    ],
)
def test_validator_rejects_secrets_host_reads_and_subprocess(tmp_path: Path, script: str) -> None:
    store = registry(tmp_path)
    created = store.create_candidate(candidate(script=script))
    result = CandidateValidator(store).validate_to_trial(
        created.manifest.skill_id, created.manifest.version
    )
    assert not result.passed
    rejected = store.get(created.manifest.skill_id, created.manifest.version)
    assert rejected is not None
    assert rejected.manifest.status == "rejected"
    store.close()


def test_script_runner_blocks_escape_and_kills_timeout(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_candidate(candidate(script="import time\ntime.sleep(5)\n"))
    assert (
        CandidateValidator(store)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )

    with pytest.raises(PolicyViolation):
        run_skill_script(
            store,
            skill_id=created.manifest.skill_id,
            version=created.manifest.version,
            script_name="../../escape.py",
            args=[],
            workspace=tmp_path,
        )
    result = run_skill_script(
        store,
        skill_id=created.manifest.skill_id,
        version=created.manifest.version,
        script_name="inspect_imports.py",
        args=[],
        workspace=tmp_path,
        timeout=0.05,
    )
    assert result.timed_out
    store.close()


def test_merge_target_supersedes_sources_without_overwriting_history(tmp_path: Path) -> None:
    store = registry(tmp_path)
    source_a = store.create_candidate(candidate().model_copy(update={"name": "pytest-import-a"}))
    source_b = store.create_candidate(candidate().model_copy(update={"name": "pytest-import-b"}))
    validator = CandidateValidator(store)
    for source in (source_a, source_b):
        assert validator.validate_to_trial(source.manifest.skill_id, source.manifest.version).passed
        store.transition(source.manifest.skill_id, source.manifest.version, "active")
    source_refs = [
        SkillVersionRef(skill_id=source.manifest.skill_id, version=source.manifest.version)
        for source in (source_a, source_b)
    ]
    merged = store.create_candidate(
        candidate().model_copy(update={"name": "pytest-import-merged"}),
        supersedes=source_refs,
    )
    assert validator.validate_to_trial(merged.manifest.skill_id, merged.manifest.version).passed
    store.transition(merged.manifest.skill_id, merged.manifest.version, "active")
    target = SkillVersionRef(skill_id=merged.manifest.skill_id, version=merged.manifest.version)
    DeterministicCurator(store).supersede_sources(source_refs, target)

    first = store.get(source_a.manifest.skill_id, 1)
    second = store.get(source_b.manifest.skill_id, 1)
    assert first is not None and first.manifest.status == "superseded"
    assert second is not None and second.manifest.status == "superseded"
    assert Path(source_a.package_path, "SKILL.md").read_text() == SKILL_MD
    store.close()


def test_curator_moves_unused_active_skill_to_stale_then_archive(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_candidate(candidate())
    assert (
        CandidateValidator(store)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )
    store.transition(created.manifest.skill_id, created.manifest.version, "active")
    old = datetime.now(UTC) - timedelta(days=40)
    store.connection.execute(
        "UPDATE skill_stats SET created_at = ? WHERE skill_id = ? AND version = ?",
        (old.isoformat(), created.manifest.skill_id, created.manifest.version),
    )
    store.connection.commit()
    curator = DeterministicCurator(store)
    assert curator.run(now=datetime.now(UTC))[0].action == "stale"
    later = old + timedelta(days=101)
    assert curator.run(now=later)[0].action == "archive"
    archived = store.get(created.manifest.skill_id, 1)
    assert archived is not None and archived.manifest.status == "archived"
    store.close()


def test_trial_policy_promotes_success_and_rejects_repeated_failure(tmp_path: Path) -> None:
    store = registry(tmp_path)
    validator = CandidateValidator(store)
    successful = store.create_candidate(candidate())
    failing = store.create_candidate(candidate().model_copy(update={"name": "failing-trial"}))
    for record in (successful, failing):
        assert validator.validate_to_trial(record.manifest.skill_id, record.manifest.version).passed
    policy = TrialPromotionPolicy(
        min_uses=2,
        min_successes=2,
        min_success_rate=0.75,
        max_failures=2,
    )
    success_ref = SkillVersionRef(
        skill_id=successful.manifest.skill_id, version=successful.manifest.version
    )
    store.record_use(success_ref, success=True, tool_calls=3, attempts=1, patched=True)
    assert policy.apply(store, successful) == "hold"
    store.record_use(success_ref, success=True, tool_calls=2, attempts=1, patched=True)
    assert policy.apply(store, successful) == "promote"
    assert store.get(success_ref.skill_id, success_ref.version).manifest.status == "active"  # type: ignore[union-attr]

    failure_ref = SkillVersionRef(
        skill_id=failing.manifest.skill_id, version=failing.manifest.version
    )
    for _ in range(2):
        store.record_use(failure_ref, success=False, tool_calls=5, attempts=2, patched=False)
    assert policy.apply(store, failing) == "reject"
    assert store.get(failure_ref.skill_id, failure_ref.version).manifest.status == "rejected"  # type: ignore[union-attr]
    store.close()


def test_update_version_keeps_parent_active_until_child_promotes(tmp_path: Path) -> None:
    store = registry(tmp_path)
    validator = CandidateValidator(store)
    original = store.create_candidate(candidate())
    assert validator.validate_to_trial(original.manifest.skill_id, original.manifest.version).passed
    store.transition(original.manifest.skill_id, original.manifest.version, "active")
    updated = store.create_candidate(
        candidate().model_copy(update={"description": "Improved import diagnosis"}),
        skill_id=original.manifest.skill_id,
        parent_version=original.manifest.version,
    )
    assert updated.manifest.version == original.manifest.version + 1
    assert validator.validate_to_trial(updated.manifest.skill_id, updated.manifest.version).passed
    parent = store.get(original.manifest.skill_id, original.manifest.version)
    assert parent is not None and parent.manifest.status == "active"
    child_ref = SkillVersionRef(
        skill_id=updated.manifest.skill_id, version=updated.manifest.version
    )
    store.record_use(child_ref, success=True, tool_calls=2, attempts=1, patched=True)
    assert TrialPromotionPolicy(min_uses=1, min_successes=1).apply(store, updated) == "promote"
    parent = store.get(original.manifest.skill_id, original.manifest.version)
    child = store.get(updated.manifest.skill_id, updated.manifest.version)
    assert parent is not None and parent.manifest.status == "superseded"
    assert child is not None and child.manifest.status == "active"
    assert Path(parent.package_path, "SKILL.md").is_file()
    store.close()


class MergeGateway:
    def __init__(self, merged: SkillCandidate) -> None:
        self.merged = merged
        self.payload: str | None = None

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        agent_id: str,
    ) -> Any:
        del system_prompt, response_model, agent_id
        self.payload = user_prompt
        return CuratorReview(
            action="merge",
            rationale="procedures overlap",
            merged_candidate=self.merged,
        )


@pytest.mark.asyncio
async def test_curator_reviews_whole_packages_and_creates_merge_trial(tmp_path: Path) -> None:
    store = registry(tmp_path)
    validator = CandidateValidator(store)
    first = store.create_candidate(
        candidate().model_copy(update={"name": "pytest-import-diagnosis"})
    )
    second = store.create_candidate(
        candidate().model_copy(update={"name": "pytest-import-debugger"})
    )
    for record in (first, second):
        assert validator.validate_to_trial(record.manifest.skill_id, record.manifest.version).passed
        store.transition(record.manifest.skill_id, record.manifest.version, "active")
    merged_candidate = candidate().model_copy(update={"name": "pytest-import-toolkit"})
    gateway = MergeGateway(merged_candidate)
    result = await CuratorPipeline(
        store,
        validator,
        model_curator=ModelCurator(store, gateway),  # type: ignore[arg-type]
    ).run()

    assert result.shortlisted_groups
    assert result.candidates[0].validation_passed
    merged = store.get(result.candidates[0].ref.skill_id, result.candidates[0].ref.version)
    assert merged is not None and merged.manifest.status == "trial"
    assert len(merged.manifest.supersedes) == 2
    assert gateway.payload is not None
    assert "scripts/inspect_imports.py" in gateway.payload
    assert "tests/test_inspect.py" in gateway.payload
    for source in (first, second):
        current = store.get(source.manifest.skill_id, source.manifest.version)
        assert current is not None and current.manifest.status == "active"
    merged_ref = result.candidates[0].ref
    store.record_use(merged_ref, success=True, tool_calls=2, attempts=1, patched=True)
    assert TrialPromotionPolicy(min_uses=1, min_successes=1).apply(store, merged) == "promote"
    for source in (first, second):
        current = store.get(source.manifest.skill_id, source.manifest.version)
        assert current is not None and current.manifest.status == "superseded"
    store.close()


def test_new_skill_slug_collision_is_rejected_without_implicit_version(
    tmp_path: Path,
) -> None:
    store = registry(tmp_path)
    first = store.create_candidate(candidate())

    with pytest.raises(CapabilityRegistryError, match="use update_skill with lineage"):
        store.create_candidate(
            candidate().model_copy(update={"description": "collision without lineage"})
        )

    assert store.get(first.manifest.skill_id, 2) is None
    assert len(store.list()) == 1
    store.close()


def test_capability_side_effect_operation_keys_are_exactly_once(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_candidate(candidate(), operation_key="create:run-1")
    replayed = store.create_candidate(candidate(), operation_key="create:run-1")
    assert replayed.manifest.version == created.manifest.version
    assert len(store.list()) == 1
    assert (
        CandidateValidator(store)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )
    ref = SkillVersionRef(skill_id=created.manifest.skill_id, version=created.manifest.version)

    assert store.record_retrieval([ref], operation_key="retrieve:run-1")
    assert not store.record_retrieval([ref], operation_key="retrieve:run-1")
    assert store.record_retrieval([ref], selected=True, operation_key="select:run-1")
    assert not store.record_retrieval([ref], selected=True, operation_key="select:run-1")
    assert store.record_use(
        ref,
        success=True,
        tool_calls=2,
        attempts=1,
        patched=True,
        operation_key="use:run-1",
    )
    assert not store.record_use(
        ref,
        success=True,
        tool_calls=2,
        attempts=1,
        patched=True,
        operation_key="use:run-1",
    )

    stats = store.stats(ref.skill_id, ref.version)
    assert stats.retrieval_count == 1
    assert stats.selected_count == 1
    assert stats.use_count == 1
    store.close()


def test_trial_retrieval_slot_prevents_active_skill_starvation(tmp_path: Path) -> None:
    store = registry(tmp_path)
    validator = CandidateValidator(store)
    active = store.create_candidate(candidate().model_copy(update={"name": "active-pytest-import"}))
    trial = store.create_candidate(candidate().model_copy(update={"name": "trial-pytest-import"}))
    for record in (active, trial):
        assert validator.validate_to_trial(record.manifest.skill_id, record.manifest.version).passed
    store.transition(active.manifest.skill_id, active.manifest.version, "active")

    hits = CapabilityRetriever(store, top_k=2, trial_slots=1).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(
            summary="pytest ModuleNotFoundError",
            log_excerpt="pytest import failed",
            task_family="test",
        ),
    )

    assert {hit.skill_id for hit in hits} == {
        active.manifest.skill_id,
        trial.manifest.skill_id,
    }
    store.close()


def test_unused_trial_is_bounded_by_selection_exposure(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_candidate(candidate())
    assert (
        CandidateValidator(store)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )
    ref = SkillVersionRef(skill_id=created.manifest.skill_id, version=created.manifest.version)
    for index in range(3):
        store.record_retrieval([ref], selected=True, operation_key=f"selection:{index}")

    decisions = DeterministicCurator(store, max_exposures_without_use=3).run()

    assert decisions[0].action == "reject"
    current = store.get(ref.skill_id, ref.version)
    assert current is not None and current.manifest.status == "rejected"
    store.close()
