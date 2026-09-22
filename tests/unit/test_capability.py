from __future__ import annotations

from pathlib import Path

import pytest

from evoci.capability.execution import run_skill_script
from evoci.capability.materializer import CapabilityMaterializer
from evoci.capability.models import GeneratedFile, SkillCandidate, SkillPermissions
from evoci.capability.registry import CapabilityRegistry, CapabilityRegistryError
from evoci.capability.retrieval import CapabilityRetriever
from evoci.domain.models import CIFailure, RepoSpec, SkillHit, SkillRef
from evoci.learning_state import LEGACY_LEARNING_STATE_MESSAGE, LegacyLearningStateError
from evoci.tools.policy import PolicyViolation

SKILL_MD = """---
name: pytest-import-debugging
description: Diagnose Python test import failures
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


def candidate(
    *,
    name: str = "pytest-import-debugging",
    script: str = "print('imports-ok')\n",
) -> SkillCandidate:
    return SkillCandidate(
        name=name,
        description="Diagnose Python pytest import failures",
        triggers=["pytest import", "ModuleNotFoundError"],
        task_families=["test"],
        skill_md=SKILL_MD.replace("pytest-import-debugging", name),
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


def test_skill_is_stored_without_version_directory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    skill_root = tmp_path / "skills" / created.manifest.skill_id
    assert (skill_root / "package").is_dir()
    assert (skill_root / "package" / "SKILL.md").is_file()
    assert (skill_root / "memory.md").is_file()
    assert not any(skill_root.glob("v*"))
    store.close()


def test_new_skill_creates_current_package(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    assert created.manifest.skill_id == "pytest-import-debugging"
    assert created.manifest.enabled
    assert Path(created.package_path).name == "package"
    assert (Path(created.package_path) / "SKILL.md").is_file()
    store.close()


def test_skill_update_replaces_current_package(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    updated_md = SKILL_MD.replace("compare import roots", "inspect sys.path first")
    updated = store.update_skill(
        created.manifest.skill_id,
        candidate().model_copy(update={"skill_md": updated_md, "source_run_ids": ["run-2"]}),
    )
    package = Path(updated.package_path)
    assert "inspect sys.path first" in (package / "SKILL.md").read_text(encoding="utf-8")
    previous = package.parent / "previous" / "SKILL.md"
    assert "compare import roots" in previous.read_text(encoding="utf-8")
    store.close()


def test_skill_update_keeps_one_previous_backup(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    skill_id = created.manifest.skill_id
    store.update_skill(
        skill_id,
        candidate().model_copy(
            update={"skill_md": SKILL_MD.replace("compare import roots", "second revision")}
        ),
    )
    store.update_skill(
        skill_id,
        candidate().model_copy(
            update={"skill_md": SKILL_MD.replace("compare import roots", "third revision")}
        ),
    )
    skill_root = tmp_path / "skills" / skill_id
    assert "third revision" in (skill_root / "package" / "SKILL.md").read_text(encoding="utf-8")
    assert "second revision" in (skill_root / "previous" / "SKILL.md").read_text(encoding="utf-8")
    assert not (skill_root / "previous" / "previous").exists()
    store.close()


def test_failed_update_preserves_current_skill(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    original = (Path(created.package_path) / "SKILL.md").read_text(encoding="utf-8")
    with pytest.raises(CapabilityRegistryError, match="validation failed"):
        store.update_skill(
            created.manifest.skill_id,
            candidate(script="eval('unsafe')\n"),
        )
    current = store.get(created.manifest.skill_id)
    assert current is not None
    assert (Path(current.package_path) / "SKILL.md").read_text(encoding="utf-8") == original
    store.close()


def test_skill_ref_contains_only_skill_id() -> None:
    assert set(SkillRef.model_fields) == {"skill_id"}
    assert SkillRef(skill_id="debug-pytest-collection").model_dump() == {
        "skill_id": "debug-pytest-collection"
    }


def test_disabled_skill_is_not_retrieved(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    store.disable(created.manifest.skill_id)
    catalog = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(
            summary="pytest ModuleNotFoundError",
            log_excerpt="pytest import failed",
            task_family="test",
        ),
    )
    assert catalog.entries == []
    store.enable(created.manifest.skill_id)
    catalog = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(
            summary="pytest ModuleNotFoundError",
            log_excerpt="pytest import failed",
            task_family="test",
        ),
    )
    assert [entry.skill_id for entry in catalog.entries] == ["pytest-import-debugging"]
    store.close()


def test_create_skill_validates_and_materializes(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    catalog = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(
            summary="pytest ModuleNotFoundError",
            log_excerpt="pytest import failed",
            task_family="test",
        ),
    )
    assert [entry.skill_id for entry in catalog.entries] == ["pytest-import-debugging"]
    assert store.stats("pytest-import-debugging").retrieval_count == 1
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    hits = [
        SkillHit(
            skill_id=created.manifest.skill_id,
            name=created.manifest.name,
            description=created.manifest.description,
            skill_md="",
            score=1.0,
        )
    ]
    materialized = CapabilityMaterializer(store, tmp_path / "runtime").materialize(
        hits, run_id="materialize", workspace=workspace
    )
    script = materialized[0] / "scripts/inspect_imports.py"
    assert script.read_text() == "print('imports-ok')\n"
    assert script.stat().st_mode & 0o222 == 0
    assert materialized[0].name == created.manifest.skill_id
    with pytest.raises(PolicyViolation, match="outside target workspace"):
        CapabilityMaterializer(store, workspace / "runtime").materialize(
            hits, run_id="unsafe", workspace=workspace
        )
    store.close()


def test_colliding_slug_is_rejected(tmp_path: Path) -> None:
    store = registry(tmp_path)
    store.create_skill(candidate())
    with pytest.raises(CapabilityRegistryError, match="already exists"):
        store.create_skill(candidate())
    store.close()


def test_operation_key_is_idempotent(tmp_path: Path) -> None:
    store = registry(tmp_path)
    first = store.create_skill(candidate(), operation_key="create:run-1")
    replayed = store.create_skill(candidate(), operation_key="create:run-1")
    assert first.manifest.skill_id == replayed.manifest.skill_id
    assert len(store.list()) == 1
    store.close()


def test_unsafe_script_is_rejected(tmp_path: Path) -> None:
    store = registry(tmp_path)
    with pytest.raises(CapabilityRegistryError, match="validation failed"):
        store.create_skill(candidate(script="eval('boom')\n"))
    assert store.list() == []
    store.close()


def test_timed_out_skill_tests_are_rejected(tmp_path: Path) -> None:
    store = CapabilityRegistry(
        tmp_path / "skills",
        tmp_path / "capabilities.sqlite",
        validation_timeout=0.2,
    )
    slow = candidate().model_copy(
        update={
            "tests": [
                GeneratedFile(
                    path="tests/test_inspect.py",
                    content=(
                        "import time\n"
                        "def test_sleep() -> None:\n"
                        "    time.sleep(5)\n"
                    ),
                )
            ]
        }
    )
    with pytest.raises(CapabilityRegistryError, match="validation failed"):
        store.create_skill(slow)
    store.close()


def test_run_skill_script_requires_enabled_skill(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = run_skill_script(
        store,
        skill_id=created.manifest.skill_id,
        script_name="inspect_imports.py",
        args=[],
        workspace=workspace,
    )
    assert result.exit_code == 0
    store.disable(created.manifest.skill_id)
    with pytest.raises(PolicyViolation, match="enabled"):
        run_skill_script(
            store,
            skill_id=created.manifest.skill_id,
            script_name="inspect_imports.py",
            args=[],
            workspace=workspace,
        )
    store.close()


def test_create_rolls_back_package_when_database_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = registry(tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("db commit failed")

    monkeypatch.setattr(CapabilityRegistry, "_upsert_rows", boom)
    with pytest.raises(RuntimeError, match="db commit failed"):
        store.create_skill(candidate())
    skill_root = tmp_path / "skills" / "pytest-import-debugging"
    assert not (skill_root / "package").exists()
    assert store.get("pytest-import-debugging") is None
    store.close()


def test_update_rolls_back_package_when_database_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    updated = store.update_skill(
        created.manifest.skill_id,
        candidate().model_copy(
            update={"skill_md": SKILL_MD.replace("compare import roots", "second revision")}
        ),
    )
    skill_root = Path(updated.package_path).parent
    original_package = (skill_root / "package" / "SKILL.md").read_text(encoding="utf-8")
    original_previous = (skill_root / "previous" / "SKILL.md").read_text(encoding="utf-8")
    original_description = updated.manifest.description

    def boom(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("db commit failed")

    monkeypatch.setattr(CapabilityRegistry, "_upsert_rows", boom)
    with pytest.raises(RuntimeError, match="db commit failed"):
        store.update_skill(
            created.manifest.skill_id,
            candidate().model_copy(
                update={
                    "description": "should not persist",
                    "skill_md": SKILL_MD.replace("compare import roots", "third revision"),
                }
            ),
        )
    current = store.get(created.manifest.skill_id)
    assert current is not None
    assert current.manifest.description == original_description
    assert (skill_root / "package" / "SKILL.md").read_text(encoding="utf-8") == original_package
    assert (skill_root / "previous" / "SKILL.md").read_text(encoding="utf-8") == original_previous
    store.close()


def test_legacy_skill_schema_requires_fresh_state(tmp_path: Path) -> None:
    database = tmp_path / "capabilities.sqlite"
    import sqlite3

    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE skills (skill_id TEXT, version INTEGER, status TEXT, "
        "manifest_json TEXT, package_path TEXT, PRIMARY KEY(skill_id, version))"
    )
    connection.commit()
    connection.close()
    with pytest.raises(LegacyLearningStateError, match=LEGACY_LEARNING_STATE_MESSAGE):
        CapabilityRegistry(tmp_path / "skills", database)
