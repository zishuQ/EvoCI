"""Experience miner prompt contract: registry grounding and anti-conservatism."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from evoci.capability.miner import ExperienceMiner, LearningDecision
from evoci.capability.models import SkillCandidate, SkillSpec
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
    ) -> Any:
        del user_prompt, response_model, agent_id
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


def _trajectory() -> TrajectoryView:
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
        skills_used=[],
        memories_retrieved=[],
        memories_selected=[],
        memories_used=[],
        final_status="success",
    )


@pytest.mark.asyncio
async def test_empty_registry_blocks_update_skill_in_prompt() -> None:
    decision = LearningDecision(
        action="new_skill", rationale="reusable", candidate_skill=_candidate()
    )
    gateway = CaptureGateway([decision])
    miner = ExperienceMiner(gateway)
    result = await miner.decide(_trajectory(), existing_skills=[])
    assert result.action == "new_skill"
    prompt = gateway.system_prompts[0]
    assert "The skill registry is empty" in prompt
    assert "update_skill is not available" in prompt
    assert "prefer new_skill over none" in prompt


@pytest.mark.asyncio
async def test_populated_registry_lists_valid_update_targets() -> None:
    decision = LearningDecision(
        action="update_skill",
        rationale="improved verification",
        target_skill_id="assertion-repair",
        target_version=1,
        candidate_skill=_candidate(),
    )
    gateway = CaptureGateway([decision])
    miner = ExperienceMiner(gateway)
    skills = [
        {
            "skill_id": "assertion-repair",
            "version": 1,
            "name": "assertion-repair",
            "description": "repair assertion failures",
            "triggers": ["AssertionError"],
            "task_families": ["test"],
        }
    ]
    await miner.decide(_trajectory(), existing_skills=skills)
    prompt = gateway.system_prompts[0]
    assert "the only valid update_skill targets" in prompt
    assert "assertion-repair" in prompt
    assert "target_skill_id and target_version from this list" in prompt


@pytest.mark.asyncio
async def test_repair_prompt_includes_registry_context() -> None:
    class InvalidDecision(ValueError):
        pass

    fixed = LearningDecision(action="none", rationale="nothing durable")
    gateway = CaptureGateway([InvalidDecision("update_skill requires target_skill_id"), fixed])
    miner = ExperienceMiner(gateway)
    result = await miner.decide(_trajectory(), existing_skills=[])
    assert result.action == "none"
    assert len(gateway.system_prompts) == 2
    assert "The skill registry is empty" in gateway.system_prompts[1]
