"""Offline end-to-end skill pipeline: generate, validate, retrieve, execute, count, promote."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from evoci.capability.materializer import CapabilityMaterializer
from evoci.capability.models import GeneratedFile, SkillCandidate, SkillPermissions, SkillSpec
from evoci.capability.registry import CapabilityRegistry
from evoci.capability.retrieval import CapabilityRetriever
from evoci.domain.models import CIFailure
from evoci.graph.builder import build_graph, persist_run_outcome
from evoci.memory.store import SQLiteMemoryStore
from evoci.runtime.events import EventType
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.policy import FIXER_CAPABILITIES
from evoci.tools.registry import create_worker_registry
from tests.integration.test_correctness import VALUE_ORACLE, RealValueFixer, _failure, _runtime
from tests.integration.test_graph import (
    FakeCoordinator,
    FakeDiagnoser,
    FakeSkillMiner,
    SkillUsingInvestigator,
    initial_state,
    make_runtime,
    task,
)


class E2EMiner(FakeSkillMiner):
    async def decide(
        self,
        trajectory_summary: object,
        *,
        existing_skills: object = None,
        used_skill_ids: object = None,
        **kwargs: object,
    ) -> object:
        del existing_skills, used_skill_ids
        decision = await super().decide(trajectory_summary, **kwargs)
        assert decision.candidate_skill is not None
        skill = decision.candidate_skill.model_copy(
            update={
                "references": [
                    GeneratedFile(
                        path="references/notes.md", content="check VALUE after patch\n"
                    )
                ],
                "scripts": [
                    GeneratedFile(
                        path="scripts/inspect_assertion.py",
                        content=(
                            "from pathlib import Path\n"
                            "print(Path('app.py').read_text(), end='')\n"
                        ),
                        executable=True,
                    )
                ],
                "permissions": SkillPermissions(execute=True, read_workspace=True),
            }
        )
        return decision.model_copy(update={"candidate_skill": skill})


@pytest.mark.asyncio
async def test_offline_skill_pipeline_generate_use_promote(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "caps.sqlite")
    memory = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    runtime_a = replace(
        _runtime(tmp_path, RealValueFixer()),
        capability_registry=registry,
        skill_miner=E2EMiner(),
        memory_store=memory,
        recorder=TrajectoryRecorder(),
    )
    result_a = await build_graph(runtime_a).ainvoke(
        _failure(tmp_path, [VALUE_ORACLE]),
        {"configurable": {"thread_id": "t1"}},
    )
    assert result_a["status"] == "success"
    assert result_a["candidate_skill_id"] == "assertion-repair"
    learned = registry.get("assertion-repair")
    assert learned is not None
    assert learned.manifest.enabled
    episode = memory.get_episode("run-1")
    assert episode is not None
    assert episode.success is True

    workspace_b = tmp_path / "task-b"
    workspace_b.mkdir()
    (workspace_b / "app.py").write_text("VALUE = 0\n")
    catalog = CapabilityRetriever(registry).retrieve(
        initial_state(workspace_b)["repo"],  # type: ignore[arg-type]
        CIFailure(
            summary="test failed AssertionError",
            log_excerpt="AssertionError",
            failed_commands=[VALUE_ORACLE],
        ),
    )
    assert catalog.entries
    skill_id = catalog.entries[0].skill_id
    from evoci.domain.models import SkillHit

    CapabilityMaterializer(registry, tmp_path / "runtime", allow_relocate=True).materialize(
        [
            SkillHit(
                skill_id=skill_id,
                name=catalog.entries[0].name,
                description=catalog.entries[0].description,
                skill_md="",
                score=1.0,
            )
        ],
        run_id="t2",
        workspace=workspace_b,
    )
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        workspace_b,
        capability_registry=registry,
        allowed_skill_refs={skill_id},
    )
    tools.invoke("load_skill", skill_id=skill_id)
    resource = tools.invoke(
        "read_skill_resource",
        skill_id=skill_id,
        path="references/notes.md",
    )
    assert "VALUE" in resource["content"]
    first = tools.invoke(
        "run_skill_script",
        skill_id=skill_id,
        script_name="inspect_assertion.py",
        args=[],
    )
    assert first.stdout == "VALUE = 0\n"
    tools.invoke("apply_patch", files={"app.py": "VALUE = 1\n"})
    second = tools.invoke(
        "run_skill_script",
        skill_id=skill_id,
        script_name="inspect_assertion.py",
        args=[],
    )
    assert second.stdout == "VALUE = 1\n"
    assert second.observed_revision != first.observed_revision
    tools.close()

    def reuse_runtime(run_workspace: Path, run_id: str) -> object:
        return replace(
            make_runtime(
                run_workspace,
                FakeCoordinator([[task(run_id)]]),
                SkillUsingInvestigator(),
                FakeDiagnoser(),
                RealValueFixer(),
            ),
            capability_registry=registry,
            capability_retriever=CapabilityRetriever(registry),
            capability_materializer=CapabilityMaterializer(
                registry, tmp_path / "runtime", allow_relocate=True
            ),
            recorder=TrajectoryRecorder(),
            config=EvoCIConfig_from(run_workspace),
        )

    def EvoCIConfig_from(workspace: Path) -> object:
        from evoci.config import EvoCIConfig

        return EvoCIConfig.from_env(cwd=tmp_path)

    runtime_b = reuse_runtime(workspace_b, "t2")
    state_b = _failure(workspace_b, [VALUE_ORACLE])
    state_b["run_id"] = "run-t2"
    state_b["workspace_path"] = str(workspace_b)
    result_b = await build_graph(runtime_b).ainvoke(state_b)
    assert result_b["status"] == "success"

    runtime_c = reuse_runtime(workspace_b, "t3")
    state_c = _failure(workspace_b, [VALUE_ORACLE])
    state_c["run_id"] = "run-t3"
    state_c["workspace_path"] = str(workspace_b)
    result_c = await build_graph(runtime_c).ainvoke(state_c)
    assert result_c["status"] == "success"
    promoted = registry.get("assertion-repair")
    assert promoted is not None
    assert promoted.manifest.enabled

    uses_before = registry.stats("assertion-repair").use_count
    await persist_run_outcome(runtime_c, result_c, success=True, failure_reason=None)
    assert registry.stats("assertion-repair").use_count == uses_before
    registry.close()
    memory.close()


@pytest.mark.asyncio
async def test_offline_negative_skill_failure_does_not_promote(tmp_path: Path) -> None:
    registry = CapabilityRegistry(tmp_path / "skills", tmp_path / "caps.sqlite")
    created = registry.create_skill(
        SkillCandidate(
            name="flaky-inspector",
            description="fails on purpose",
            triggers=["test failed"],
            task_families=["unknown"],
            spec=SkillSpec(
                name="flaky-inspector",
                description="fails on purpose",
                purpose="Exercise failure attribution.",
                when_to_use="When testing skill failure accounting.",
                procedure="Run the bundled script.",
                pitfalls="Do not treat a rejected call as use.",
                verification="Script exits 1.",
            ),
            scripts=[
                GeneratedFile(
                    path="scripts/inspect_assertion.py", content="raise SystemExit(1)\n"
                )
            ],
            source_run_ids=["seed"],
            confidence=0.9,
            permissions=SkillPermissions(execute=True),
        )
    )
    tools = create_worker_registry(
        FIXER_CAPABILITIES,
        tmp_path,
        capability_registry=registry,
        allowed_skill_refs={created.manifest.skill_id},
    )
    tools.invoke("load_skill", skill_id=created.manifest.skill_id)
    executed = tools.invoke(
        "run_skill_script",
        skill_id=created.manifest.skill_id,
        script_name="inspect_assertion.py",
        args=[],
    )
    assert executed.exit_code == 1
    recorder = TrajectoryRecorder()
    recorder.emit(
        run_id="neg-run",
        event_type=EventType.SKILL_SELECTED,
        event_key="select",
        payload={
            "skills": [
                {"skill_id": created.manifest.skill_id}
            ]
        },
    )
    recorder.emit(
        run_id="neg-run",
        event_type=EventType.SKILL_USED,
        agent_id="fixer",
        event_key="skill-fail",
        payload={
            "skill_id": created.manifest.skill_id,
            "resource": "inspect_assertion.py",
            "success": False,
        },
    )
    (tmp_path / "app.py").write_text("VALUE = 0\n")
    runtime = replace(
        _runtime(tmp_path, RealValueFixer()),
        capability_registry=registry,
        recorder=recorder,
    )
    state = _failure(tmp_path, [VALUE_ORACLE])
    state["run_id"] = "neg-run"
    result = await build_graph(runtime).ainvoke(state)
    assert result["status"] == "success"
    stats = registry.stats(created.manifest.skill_id)
    assert stats.failure_count >= 1
    assert stats.success_count == 0
    current = registry.get(created.manifest.skill_id)
    assert current is not None
    assert current.manifest.enabled
    tools.close()
    registry.close()
