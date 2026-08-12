import asyncio
from pathlib import Path

import pytest

from evoci.capability.models import GeneratedFile, SkillCandidate, SkillPermissions
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.validator import CandidateValidator
from evoci.tools.filesystem import FileTools
from evoci.tools.policy import (
    INVESTIGATOR_CAPABILITIES,
    REVIEWER_CAPABILITIES,
    PolicyViolation,
    WorkspaceBoundary,
    validate_command,
)
from evoci.tools.registry import create_worker_registry


def test_workspace_boundary_rejects_escape(tmp_path: Path) -> None:
    boundary = WorkspaceBoundary(tmp_path)
    with pytest.raises(PolicyViolation):
        boundary.resolve("../../etc/passwd")


def test_read_only_worker_cannot_write(tmp_path: Path) -> None:
    tools = FileTools(tmp_path, writable=False)
    with pytest.raises(PolicyViolation):
        tools.write_file("change.py", "pass")


def test_command_policy_rejects_unbounded_shell() -> None:
    with pytest.raises(PolicyViolation):
        validate_command(["bash", "-lc", "anything"])


def test_investigator_registry_exposes_reads_but_not_patch(tmp_path: Path) -> None:
    registry = create_worker_registry(INVESTIGATOR_CAPABILITIES, tmp_path)
    assert "read_file" in registry.available()
    assert "run_test" in registry.available()
    assert "apply_patch" not in registry.available()
    registry.close()


@pytest.mark.asyncio
async def test_read_only_command_mutates_only_disposable_copy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "tracked.py").write_text("VALUE = 1\n")
    registry = create_worker_registry(INVESTIGATOR_CAPABILITIES, workspace)
    try:
        result = await registry.ainvoke(
            "run_command",
            argv=[
                "python",
                "-c",
                "from pathlib import Path; Path('x').write_text('temporary')",
            ],
        )
        assert result.exit_code == 0
        assert not (workspace / "x").exists()
        assert registry.execution_workspace is not None
        assert (registry.execution_workspace / "x").read_text() == "temporary"
    finally:
        registry.close()


@pytest.mark.asyncio
async def test_parallel_investigators_and_reviewer_cannot_write_real_repo(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registries = [
        create_worker_registry(INVESTIGATOR_CAPABILITIES, workspace),
        create_worker_registry(INVESTIGATOR_CAPABILITIES, workspace),
        create_worker_registry(REVIEWER_CAPABILITIES, workspace),
    ]
    try:
        results = await asyncio.gather(
            *[
                registry.ainvoke(
                    "run_test",
                    argv=[
                        "python",
                        "-c",
                        (
                            "from pathlib import Path; "
                            f"Path('write-{index}').write_text('temporary')"
                        ),
                    ],
                )
                for index, registry in enumerate(registries)
            ]
        )
        assert all(result.exit_code == 0 for result in results)
        assert not any((workspace / f"write-{index}").exists() for index in range(3))
    finally:
        for registry in registries:
            registry.close()


@pytest.mark.asyncio
async def test_read_only_worker_skill_script_cannot_mutate_real_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capability_registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    skill_md = """---
name: isolated-writer
description: Exercise workspace isolation
version: 1
---

# Purpose
Exercise workspace isolation.
# When to Use
Use in isolation tests.
# Procedure
Run the bundled script.
# Pitfalls
Never trust declared permissions alone.
# Verification
Inspect the disposable output.
# Bundled Resources
The writer script is bundled.
"""
    created = capability_registry.create_candidate(
        SkillCandidate(
            name="isolated-writer",
            description="Exercise workspace isolation",
            triggers=["isolation"],
            task_families=["test"],
            skill_md=skill_md,
            scripts=[
                GeneratedFile(
                    path="scripts/write.py",
                    content=(
                        "from pathlib import Path\nPath('skill-output').write_text('temporary')\n"
                    ),
                )
            ],
            source_run_ids=["isolation"],
            confidence=1.0,
            permissions=SkillPermissions(execute=True, write_workspace=False),
        )
    )
    assert (
        CandidateValidator(capability_registry)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )
    registry = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        workspace,
        capability_registry=capability_registry,
        allowed_skill_refs={(created.manifest.skill_id, created.manifest.version)},
    )
    try:
        result = await registry.ainvoke(
            "run_skill_script",
            skill_id=created.manifest.skill_id,
            version=created.manifest.version,
            script_name="write.py",
            args=[],
        )
        assert result.exit_code == 0
        assert not (workspace / "skill-output").exists()
    finally:
        registry.close()
        capability_registry.close()


@pytest.mark.asyncio
async def test_read_only_worker_rejects_skill_that_requests_workspace_write(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capability_registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "capabilities.sqlite")
    created = capability_registry.create_candidate(
        SkillCandidate(
            name="workspace-writer",
            description="Request a workspace-writing permission",
            triggers=["write"],
            task_families=["test"],
            skill_md="""---
name: workspace-writer
description: Request a workspace-writing permission
version: 1
---

# Purpose
Exercise the permission intersection.
# When to Use
Use in permission tests.
# Procedure
Run the bundled script.
# Pitfalls
The worker role remains authoritative.
# Verification
Confirm rejection for a read-only worker.
# Bundled Resources
The writer script is bundled.
""",
            scripts=[
                GeneratedFile(
                    path="scripts/write.py",
                    content="from pathlib import Path\nPath('skill-output').write_text('bad')\n",
                )
            ],
            source_run_ids=["permission-intersection"],
            confidence=1.0,
            permissions=SkillPermissions(execute=True, write_workspace=True),
        )
    )
    assert (
        CandidateValidator(capability_registry)
        .validate_to_trial(created.manifest.skill_id, created.manifest.version)
        .passed
    )
    registry = create_worker_registry(
        INVESTIGATOR_CAPABILITIES,
        workspace,
        capability_registry=capability_registry,
        allowed_skill_refs={(created.manifest.skill_id, created.manifest.version)},
    )
    try:
        with pytest.raises(PolicyViolation, match="worker role"):
            await registry.ainvoke(
                "run_skill_script",
                skill_id=created.manifest.skill_id,
                version=created.manifest.version,
                script_name="write.py",
                args=[],
            )
        assert not (workspace / "skill-output").exists()
    finally:
        registry.close()
        capability_registry.close()
