from __future__ import annotations

from pathlib import Path

import pytest

from evoci.capability.materializer import CapabilityMaterializer
from evoci.capability.models import (
    GeneratedFile,
    SkillCandidate,
    SkillSpec,
)
from evoci.capability.registry import CapabilityRegistryError
from evoci.capability.render import REQUIRED_SKILL_SECTIONS, render_skill_markdown
from evoci.capability.validator import CandidateValidator
from evoci.config import EvoCIConfig
from evoci.domain.models import SkillHit
from evoci.tools.policy import FIXER_CAPABILITIES, PolicyViolation
from evoci.tools.registry import create_worker_registry
from tests.unit.test_capability import candidate, registry


def test_a05_structured_spec_renders_valid_skill_md() -> None:
    spec = SkillSpec(
        name="assertion-repair",
        description="Fix assertion mismatches",
        purpose="Repair evidence-backed assertion failures.",
        when_to_use="Use when a unit test assertion fails.",
        procedure="Inspect expected and actual values, then apply the smallest fix.",
        pitfalls="Do not change or skip the test.",
        verification="Re-run the failing test.",
    )
    rendered = render_skill_markdown(spec)
    created = SkillCandidate(
        name="assertion-repair",
        description="Fix assertion mismatches",
        triggers=["AssertionError"],
        task_families=["test"],
        spec=spec,
        source_run_ids=["run-spec"],
        confidence=0.9,
    )
    assert created.skill_md.startswith("---\n")
    assert all(section in created.skill_md for section in REQUIRED_SKILL_SECTIONS)
    assert created.skill_md == rendered


def test_a05_existing_markdown_candidates_remain_valid() -> None:
    created = candidate()
    assert created.skill_md.startswith("---\n")


def test_a06_default_runtime_relocates_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    hits = [
        SkillHit(
            skill_id=created.manifest.skill_id,
            name=created.manifest.name,
            description=created.manifest.description,
            skill_md="unused",
            score=1.0,
            resources=["scripts/inspect_imports.py"],
        )
    ]
    destinations = CapabilityMaterializer(
        store, workspace / ".evoci" / "runtime", allow_relocate=True
    ).materialize(hits, run_id="run-a06", workspace=workspace)
    assert destinations
    assert all(workspace.resolve() not in path.resolve().parents for path in destinations)
    assert not (workspace / ".evoci" / "runtime").exists()
    store.close()


def test_a06_explicit_runtime_inside_workspace_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    monkeypatch.setenv("EVO_RUNTIME_DIR", str(workspace / "runtime"))
    config = EvoCIConfig.from_env(cwd=tmp_path)
    assert config.runtime_dir_explicit is True
    store = registry(tmp_path)
    created = store.create_skill(candidate())
    hits = [
        SkillHit(
            skill_id=created.manifest.skill_id,
            name=created.manifest.name,
            description=created.manifest.description,
            skill_md="unused",
            score=1.0,
        )
    ]
    with pytest.raises(PolicyViolation, match="outside target workspace"):
        CapabilityMaterializer(
            store, config.runtime_dir, allow_relocate=not config.runtime_dir_explicit
        ).materialize(hits, run_id="run-a06-explicit", workspace=workspace)
    store.close()


def test_a07_read_skill_resource_allows_declared_and_rejects_escape(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(
        candidate().model_copy(
            update={
                "references": [
                    GeneratedFile(path="references/notes.md", content="hello-resource\n")
                ]
            }
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
    assert payload["content"] == "hello-resource\n"
    assert payload["truncated"] is False
    with pytest.raises(PolicyViolation):
        tools.invoke(
            "read_skill_resource",
            skill_id=created.manifest.skill_id,
            path="../escape.md",
        )
    with pytest.raises(PolicyViolation, match="not declared"):
        tools.invoke(
            "read_skill_resource",
            skill_id=created.manifest.skill_id,
            path="references/missing.md",
        )
    tools.close()
    denied = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs=set(),
    )
    with pytest.raises(PolicyViolation, match="not in the current catalog"):
        denied.invoke(
            "read_skill_resource",
            skill_id=created.manifest.skill_id,
            path="references/notes.md",
        )
    denied.close()
    store.close()


def test_a07_truncated_resource_includes_metadata(tmp_path: Path) -> None:
    store = registry(tmp_path)
    created = store.create_skill(
        candidate().model_copy(
            update={"references": [GeneratedFile(path="references/long.md", content="abcdefghij")]}
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
        path="references/long.md",
        offset=2,
        limit=3,
    )
    assert payload["content"] == "cde"
    assert payload["truncated"] is True
    assert payload["length"] == 10
    tools.close()
    store.close()


def test_a08_skill_snapshot_refreshes_after_patch(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("before\n")
    store = registry(tmp_path)
    created = store.create_skill(
        candidate(
            script=("from pathlib import Path\nprint(Path('target.txt').read_text(), end='')\n")
        )
    )
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=store,
        allowed_skill_refs={created.manifest.skill_id},
    )
    tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    first = tools.invoke(
        "run_skill_script",
        skill_id=created.manifest.skill_id,
        script_name="inspect_imports.py",
        args=[],
    )
    assert first.stdout == "before\n"
    first_revision = first.observed_revision
    tools.invoke("apply_patch", files={"target.txt": "after\n"})
    second = tools.invoke(
        "run_skill_script",
        skill_id=created.manifest.skill_id,
        script_name="inspect_imports.py",
        args=[],
    )
    assert second.stdout == "after\n"
    assert second.observed_revision != first_revision
    assert (tmp_path / "target.txt").read_text() == "after\n"
    tools.close()
    store.close()


def test_a09_mixed_pytest_failure_is_rejected(tmp_path: Path) -> None:
    store = registry(tmp_path)
    mixed = candidate().model_copy(
        update={
            "tests": [
                GeneratedFile(
                    path="tests/test_mixed.py",
                    content=(
                        "import unittest\n"
                        "class TestOk(unittest.TestCase):\n"
                        "    def test_ok(self) -> None:\n"
                        "        self.assertTrue(True)\n"
                        "def test_pytest_fails() -> None:\n"
                        "    assert False\n"
                    ),
                )
            ]
        }
    )
    with pytest.raises(CapabilityRegistryError, match="validation failed"):
        store.create_skill(mixed)
    assert store.list() == []
    store.close()


def test_model_skill_candidate_has_no_verification_commands() -> None:
    assert "verification_commands" not in SkillCandidate.model_fields
    from evoci.capability.models import SkillManifest

    assert "verification_commands" not in SkillManifest.model_fields
    properties = SkillCandidate.model_json_schema().get("properties", {})
    assert "verification_commands" not in properties


def test_repository_verification_is_rendered_as_skill_guidance() -> None:
    spec = SkillSpec(
        name="rpn-eval-repair",
        description="Repair RPN evaluator operator handling",
        purpose="Fix operator application in reverse-polish evaluation.",
        when_to_use="Use when rpn_eval tests fail on operator order.",
        procedure="Inspect the operator stack and apply binary operators correctly.",
        pitfalls="Do not change the public function signature.",
        verification="Re-run python -m pytest python_testcases/test_rpn_eval.py -q",
    )
    rendered = render_skill_markdown(spec)
    assert "# Verification" in rendered
    assert "python_testcases/test_rpn_eval.py" in rendered
    created = SkillCandidate(
        name="rpn-eval-repair",
        description="Repair RPN evaluator operator handling",
        triggers=["rpn_eval", "operator"],
        task_families=["expression"],
        spec=spec,
        source_run_ids=["run-guidance"],
        confidence=0.9,
    )
    assert "# Verification" in created.skill_md
    assert "python_testcases/test_rpn_eval.py" in created.skill_md


def test_skill_installation_only_runs_bundled_package_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []
    original = CandidateValidator._run_pytest

    def spy_pytest(
        self: CandidateValidator, copied: Path, file_count: int
    ) -> tuple[str | None, int]:
        launched.append(["pytest", "tests"])
        return original(self, copied, file_count)

    def refuse_repo_commands(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("skill install must not run repository verification commands")

    monkeypatch.setattr(CandidateValidator, "_run_pytest", spy_pytest)
    monkeypatch.setattr("evoci.tools.shell.CommandRunner.run", refuse_repo_commands)
    store = registry(tmp_path)
    store.create_skill(candidate())
    assert launched == [["pytest", "tests"]]
    launched.clear()
    store.create_skill(candidate(name="guidance-only").model_copy(update={"tests": []}))
    assert launched == []
    assert not hasattr(CandidateValidator, "_run_verification_commands")
    store.close()


def test_a11_shell_scripts_are_rejected_before_python_exec(tmp_path: Path) -> None:
    store = registry(tmp_path)
    with pytest.raises(CapabilityRegistryError, match="unsupported_runner"):
        store.create_skill(
            candidate().model_copy(
                update={
                    "scripts": [
                        GeneratedFile(
                            path="scripts/inspect_imports.sh",
                            content="#!/bin/sh\necho ok\n",
                            executable=True,
                        )
                    ]
                }
            )
        )
    store.close()
