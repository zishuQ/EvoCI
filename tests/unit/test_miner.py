"""Skill miner prompt contract and decision constraints."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from evoci.capability.miner import SkillLearningDecision, SkillMiner, SkillUsageLesson
from evoci.capability.models import SkillCandidate, SkillSpec
from evoci.domain.models import SkillRef
from evoci.runtime.trajectory import AgentTrace, TrajectoryView


class CaptureGateway:
    def __init__(self, decisions: list[BaseModel | Exception]) -> None:
        self.decisions = decisions
        self.system_prompts: list[str] = []

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        agent_id: str,
        usage_observer: object | None = None,
    ) -> Any:
        del user_prompt, response_model, agent_id, usage_observer
        self.system_prompts.append(system_prompt)
        outcome = self.decisions.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _candidate() -> SkillCandidate:
    return SkillCandidate(
        name="assertion-repair",
        description="repair assertion failures",
        triggers=["AssertionError"],
        task_families=["test"],
        spec=SkillSpec(
            name="assertion-repair",
            description="repair assertion failures",
            purpose="reduce repeated assertion failures",
            when_to_use="pytest reports AssertionError",
            procedure="inspect implementation and edge cases",
            pitfalls="do not change tests",
            verification="run targeted pytest",
        ),
        source_run_ids=["run-miner"],
        confidence=0.9,
    )


def _trajectory(*, success: bool = True, used: list[str] | None = None) -> TrajectoryView:
    return TrajectoryView(
        run_id="run-miner",
        agents=[AgentTrace(agent_id="fixer", model_calls=2, tool_calls=6, failed_tool_calls=0)],
        tool_calls=[],
        failed_attempts=[],
        evidence=[],
        diagnosis_history=[],
        patch_history=[],
        created_files=[],
        modified_files=[],
        verification_history=[],
        skills_retrieved=[],
        skills_selected=[],
        skills_used=[SkillRef(skill_id=skill_id) for skill_id in used or []],
        memories_retrieved=[],
        memories_selected=[],
        memories_used=[],
        final_status="success" if success else "failed",
    )


def test_skill_miner_cannot_return_memory() -> None:
    with pytest.raises(ValidationError):
        SkillLearningDecision.model_validate(
            {"action": "memory", "rationale": "store a fact", "candidate_memory": {}}
        )


def test_successful_trajectory_can_create_skill() -> None:
    miner = SkillMiner(CaptureGateway([]))
    assert miner.should_mine(
        success=True,
        failure_class=None,
        skills_used=[],
        tool_calls=1,
        failed_attempts=0,
        reusable_script_created=False,
        repeated_pattern_detected=False,
    )


@pytest.mark.asyncio
async def test_used_skill_success_can_update_skill() -> None:
    decision = SkillLearningDecision(
        action="update_skill",
        rationale="improved verification",
        target_skill_id="assertion-repair",
        candidate_skill=_candidate(),
        usage_lessons=[SkillUsageLesson(skill_id="assertion-repair", lesson="worked")],
    )
    gateway = CaptureGateway([decision])
    miner = SkillMiner(gateway)
    result = await miner.decide(
        _trajectory(used=["assertion-repair"]),
        existing_skills=[{"skill_id": "assertion-repair", "name": "assertion-repair"}],
        used_skill_ids=["assertion-repair"],
    )
    assert result.action == "update_skill"
    assert result.target_skill_id == "assertion-repair"
    assert "the only valid update_skill targets" in gateway.system_prompts[0]


@pytest.mark.asyncio
async def test_used_skill_failure_produces_failure_lesson() -> None:
    decision = SkillLearningDecision(
        action="new_skill",
        rationale="should be ignored",
        candidate_skill=_candidate(),
        usage_lessons=[
            SkillUsageLesson(skill_id="assertion-repair", lesson="older pluggy interface")
        ],
    )
    miner = SkillMiner(CaptureGateway([decision]))
    result = await miner.decide(
        _trajectory(success=False, used=["assertion-repair"]),
        used_skill_ids=["assertion-repair"],
    )
    assert result.action == "none"
    assert result.candidate_skill is None
    assert result.usage_lessons[0].lesson == "older pluggy interface"


def test_failed_no_skill_run_cannot_create_skill() -> None:
    miner = SkillMiner(CaptureGateway([]))
    assert not miner.should_mine(
        success=False,
        failure_class="repair",
        skills_used=[],
        tool_calls=100,
        failed_attempts=5,
        reusable_script_created=True,
        repeated_pattern_detected=True,
    )


@pytest.mark.asyncio
async def test_empty_registry_blocks_update_skill_in_prompt() -> None:
    decision = SkillLearningDecision(
        action="new_skill", rationale="reusable", candidate_skill=_candidate()
    )
    gateway = CaptureGateway([decision])
    miner = SkillMiner(gateway)
    result = await miner.decide(_trajectory(), existing_skills=[])
    assert result.action == "new_skill"
    prompt = gateway.system_prompts[0]
    assert "The skill registry is empty" in prompt
    assert "update_skill is not available" in prompt


@pytest.mark.asyncio
async def test_new_skill_is_not_silently_converted_to_used_skill_update() -> None:
    decision = SkillLearningDecision(
        action="new_skill",
        rationale="another procedure",
        candidate_skill=_candidate(),
        usage_lessons=[SkillUsageLesson(skill_id="assertion-repair", lesson="worked")],
    )
    miner = SkillMiner(CaptureGateway([decision]))
    result = await miner.decide(
        _trajectory(used=["assertion-repair"]),
        used_skill_ids=["assertion-repair"],
    )
    assert result.action == "none"
    assert result.candidate_skill is None
    assert result.target_skill_id is None
    assert result.usage_lessons[0].lesson == "worked"


@pytest.mark.asyncio
async def test_miner_cannot_update_an_unused_skill() -> None:
    decision = SkillLearningDecision(
        action="update_skill",
        rationale="wrong target",
        target_skill_id="unrelated-skill",
        candidate_skill=_candidate(),
        usage_lessons=[SkillUsageLesson(skill_id="assertion-repair", lesson="worked")],
    )
    miner = SkillMiner(CaptureGateway([decision]))
    result = await miner.decide(
        _trajectory(used=["assertion-repair"]),
        used_skill_ids=["assertion-repair"],
    )
    assert result.action == "none"
    assert result.candidate_skill is None
    assert result.target_skill_id is None
    assert result.usage_lessons[0].skill_id == "assertion-repair"


@pytest.mark.asyncio
async def test_repair_prompt_includes_registry_context() -> None:
    class InvalidDecision(ValueError):
        pass

    fixed = SkillLearningDecision(action="none", rationale="nothing durable")
    gateway = CaptureGateway([InvalidDecision("update_skill requires target_skill_id"), fixed])
    miner = SkillMiner(gateway)
    result = await miner.decide(_trajectory(), existing_skills=[])
    assert result.action == "none"
    assert len(gateway.system_prompts) == 2
    assert "The skill registry is empty" in gateway.system_prompts[1]
