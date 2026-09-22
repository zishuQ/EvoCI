from datetime import UTC, datetime
from pathlib import Path

import pytest

from evoci.capability.models import GeneratedFile, SkillMemoryEntry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.domain.models import CIFailure, RepoSpec
from evoci.tools.policy import FIXER_CAPABILITIES, PolicyViolation
from evoci.tools.registry import create_worker_registry
from tests.unit.test_capability import candidate, registry
from tests.unit.test_retrieval_precision import make_candidate


def _failure() -> CIFailure:
    return CIFailure(
        summary="pytest ModuleNotFoundError",
        log_excerpt="pytest import failed",
        task_family="test",
    )


def test_skill_catalog_contains_metadata_only(tmp_path: Path) -> None:
    store = registry(tmp_path)
    store.create_skill(candidate())
    catalog = CapabilityRetriever(store).retrieve(RepoSpec(owner="org", name="repo"), _failure())
    assert catalog.entries
    dump = catalog.entries[0].model_dump()
    assert set(dump) == {"skill_id", "name", "description"}
    assert "skill_md" not in dump
    assert "memory" not in dump
    store.close()


def test_skill_catalog_excludes_disabled_skills(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    store.disable(created.manifest.skill_id)
    catalog = CapabilityRetriever(store).retrieve(RepoSpec(owner="org", name="repo"), _failure())
    assert catalog.entries == []
    store.close()


def test_skill_catalog_respects_character_budget(tmp_path: Path) -> None:
    store = registry(tmp_path)
    for index in range(6):
        store.create_skill(
            make_candidate(
                name=f"skill-{index:02d}",
                description="x" * 500,
                triggers=["pytest import"],
                task_families=["test"],
            )
        )
    catalog = CapabilityRetriever(store, catalog_limit_chars=800).retrieve(
        RepoSpec(owner="org", name="repo"), _failure()
    )
    assert catalog.entries
    assert catalog.context_chars <= 800
    assert catalog.omitted_count >= 1
    store.close()


def test_skill_catalog_order_is_stable(tmp_path: Path) -> None:
    store = registry(tmp_path)
    store.create_skill(
        make_candidate(
            name="zeta-unrelated",
            description="Unrelated zeta parser",
            triggers=["zeta parser"],
            task_families=["parser"],
        )
    )
    store.create_skill(
        make_candidate(
            name="alpha-unrelated",
            description="Unrelated alpha parser",
            triggers=["alpha parser"],
            task_families=["parser"],
        )
    )
    first = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(summary="unrelated", log_excerpt="no overlap tokens here", task_family="other"),
    )
    second = CapabilityRetriever(store).retrieve(
        RepoSpec(owner="org", name="repo"),
        CIFailure(summary="unrelated", log_excerpt="no overlap tokens here", task_family="other"),
    )
    assert [entry.skill_id for entry in first.entries] == [
        entry.skill_id for entry in second.entries
    ]
    assert [entry.skill_id for entry in first.entries] == sorted(
        entry.skill_id for entry in first.entries
    )
    store.close()


def test_skill_retrieval_does_not_record_selection(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    CapabilityRetriever(store).retrieve(RepoSpec(owner="org", name="repo"), _failure())
    stats = store.stats(created.manifest.skill_id)
    assert stats.retrieval_count == 1
    assert stats.selected_count == 0
    store.close()


def test_skill_catalog_does_not_read_package_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    original = Path.read_text

    def guarded(self: Path, *args: object, **kwargs: object) -> str:
        if self.name in {"SKILL.md", "memory.md"} and created.manifest.skill_id in str(self):
            raise AssertionError(f"catalog read package body: {self}")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)  # type: ignore[attr-defined]
    catalog = CapabilityRetriever(store).retrieve(RepoSpec(owner="org", name="repo"), _failure())
    assert catalog.entries
    store.close()


def test_load_skill_returns_procedure_and_recent_memory(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    store.append_skill_memory(
        created.manifest.skill_id,
        SkillMemoryEntry(
            run_id="run-mem",
            repository="org/repo",
            task_summary="import failure",
            outcome="success",
            lesson="Add src to PYTHONPATH.",
            created_at=datetime.now(UTC),
        ),
    )
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    loaded = tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    assert loaded["skill_id"] == created.manifest.skill_id
    assert "Diagnose import failures" in loaded["procedure"]
    assert any("PYTHONPATH" in line for line in loaded["usage_memory"])
    assert "scripts/inspect_imports.py" in loaded["resources"]
    tools.close()
    store.close()


def test_load_skill_rejects_skill_outside_catalog(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs=set(),
    )
    with pytest.raises(PolicyViolation, match="not in the current catalog"):
        tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    tools.close()
    store.close()


def test_load_skill_rejects_disabled_skill(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    store.disable(created.manifest.skill_id)
    with pytest.raises(PolicyViolation, match="only enabled skills may be loaded"):
        tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    tools.close()
    store.close()


def test_load_skill_rejects_hash_mismatch(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    package = Path(created.package_path)
    skill_md = package / "SKILL.md"
    skill_md.chmod(skill_md.stat().st_mode | 0o200)
    skill_md.write_text(skill_md.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    with pytest.raises(PolicyViolation, match="hash mismatch"):
        tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    tools.close()
    store.close()


def _replace_with_symlink(path: Path, target: Path) -> None:
    path.chmod(path.stat().st_mode | 0o200)
    path.unlink()
    path.symlink_to(target)
    assert path.is_symlink()


def test_load_skill_rejects_in_package_symlink(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    package = Path(created.package_path)
    skill_md = package / "SKILL.md"
    _replace_with_symlink(skill_md, package / "scripts" / "inspect_imports.py")
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    with pytest.raises(PolicyViolation, match="must not be a symlink"):
        tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    tools.close()
    store.close()


def test_load_skill_rejects_out_of_package_symlink(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    outside = tmp_path / "outside-skill.md"
    outside.write_text("# escaped procedure\n", encoding="utf-8")
    _replace_with_symlink(Path(created.package_path) / "SKILL.md", outside)
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    with pytest.raises(PolicyViolation, match="must not be a symlink"):
        tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    tools.close()
    store.close()


def test_read_skill_resource_rejects_symlink(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(
        candidate().model_copy(
            update={"references": [GeneratedFile(path="references/notes.md", content="hello\n")]}
        )
    )
    package = Path(created.package_path)
    resource = package / "references" / "notes.md"
    _replace_with_symlink(resource, package / "SKILL.md")
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    with pytest.raises(PolicyViolation, match="must not be a symlink"):
        tools.invoke(
            "read_skill_resource",
            skill_id=created.manifest.skill_id,
            path="references/notes.md",
        )
    tools.close()
    store.close()


def test_repeated_load_records_one_selection(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    first = tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    second = tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    assert "Diagnose import failures" in first["procedure"]
    assert "procedure" not in second
    assert "usage_memory" not in second
    assert "resources" not in second
    assert second == {"skill_id": created.manifest.skill_id, "already_loaded": True}
    assert store.stats(created.manifest.skill_id).selected_count == 1
    tools.close()
    store.close()


def test_read_skill_resource_requires_load_skill(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(
        candidate().model_copy(
            update={
                "references": [GeneratedFile(path="references/notes.md", content="hello\n")]
            }
        )
    )
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    with pytest.raises(PolicyViolation, match="must be loaded with load_skill"):
        tools.invoke(
            "read_skill_resource",
            skill_id=created.manifest.skill_id,
            path="references/notes.md",
        )
    tools.close()
    store.close()


def test_run_skill_script_requires_load_skill(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    with pytest.raises(PolicyViolation, match="must be loaded with load_skill"):
        tools.invoke(
            "run_skill_script",
            skill_id=created.manifest.skill_id,
            script_name="inspect_imports.py",
            args=[],
        )
    tools.close()
    store.close()


def test_loaded_skill_resource_is_accessible(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(
        candidate().model_copy(
            update={"references": [GeneratedFile(path="references/notes.md", content="hello\n")]}
        )
    )
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    payload = tools.invoke(
        "read_skill_resource",
        skill_id=created.manifest.skill_id,
        path="references/notes.md",
    )
    assert payload["content"] == "hello\n"
    tools.close()
    store.close()


def test_loaded_skill_script_is_runnable(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    result = tools.invoke(
        "run_skill_script",
        skill_id=created.manifest.skill_id,
        script_name="inspect_imports.py",
        args=[],
    )
    assert result.exit_code == 0
    tools.close()
    store.close()
