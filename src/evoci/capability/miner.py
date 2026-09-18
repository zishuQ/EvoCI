"""Trajectory-triggered memory-versus-skill learning decisions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evoci.capability.models import SkillCandidate
from evoci.memory.models import MemoryCandidate
from evoci.model.gateway import ModelGateway
from evoci.runtime.trajectory import TrajectoryView


class LearningDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["none", "memory", "new_skill", "update_skill"]
    rationale: str
    candidate_memory: MemoryCandidate | None = None
    candidate_skill: SkillCandidate | None = None
    target_skill_id: str | None = None
    target_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def action_payload_is_complete(self) -> LearningDecision:
        if self.action in {"new_skill", "update_skill"} and self.candidate_skill is None:
            raise ValueError(f"{self.action} requires candidate_skill")
        if self.action == "update_skill" and (
            not self.target_skill_id or self.target_version is None
        ):
            raise ValueError("update_skill requires target_skill_id and target_version")
        if self.action == "new_skill" and (
            self.target_skill_id is not None or self.target_version is not None
        ):
            raise ValueError("new_skill cannot declare update lineage")
        return self


class ExperienceMining(Protocol):
    def should_mine(
        self,
        *,
        success: bool,
        tool_calls: int,
        failed_attempts: int,
        reusable_script_created: bool,
        repeated_pattern_detected: bool,
    ) -> bool: ...

    async def decide(
        self,
        trajectory: TrajectoryView,
        *,
        existing_skills: Sequence[dict[str, Any]] | None = None,
    ) -> LearningDecision: ...


class ExperienceMiner:
    def __init__(self, gateway: ModelGateway, *, tool_call_threshold: int = 5) -> None:
        self.gateway = gateway
        self.tool_call_threshold = tool_call_threshold

    def should_mine(
        self,
        *,
        success: bool,
        tool_calls: int,
        failed_attempts: int,
        reusable_script_created: bool,
        repeated_pattern_detected: bool,
    ) -> bool:
        return success and (
            tool_calls >= self.tool_call_threshold
            or failed_attempts >= 1
            or reusable_script_created
            or repeated_pattern_detected
        )

    async def _complete(
        self,
        trajectory: TrajectoryView,
        *,
        existing_skills: Sequence[dict[str, Any]] | None = None,
        repair_error: str | None = None,
    ) -> LearningDecision:
        skills = list(existing_skills or [])
        if skills:
            registry_block = (
                "Current skill registry (the only valid update_skill targets):\n"
                f"{json.dumps(skills)}\n"
                "For update_skill, set target_skill_id and target_version from this list."
            )
        else:
            registry_block = (
                "The skill registry is empty. update_skill is not available; "
                "choose none, memory, or new_skill."
            )
        if repair_error is None:
            prompt = json.dumps(trajectory.model_dump(mode="json"), default=str)
            system = (
                "Decide whether this successful trajectory teaches nothing, a durable fact, a new "
                "reusable procedure, or an update. Skills must generalize beyond the current "
                "run and preserve useful helper artifacts as package scripts when applicable. "
                "When proposing a skill, fill candidate_skill.spec (purpose, when_to_use, "
                "procedure, pitfalls, verification, bundled_resources). Do not write SKILL.md; "
                "the runtime renders it. Scripts must be Python. Proposing is low-cost: every "
                "candidate must pass an isolated pytest validator and curator review before it "
                "is registered, so prefer new_skill over none when the trajectory contains a "
                f"reusable procedure.\n{registry_block}"
            )
        else:
            prompt = (
                f"{json.dumps(trajectory.model_dump(mode='json'), default=str)}\n\n"
                f"Previous structured output failed validation: {repair_error}. "
                "Return a complete LearningDecision. If proposing a skill, include a valid spec "
                "with every required field."
            )
            system = (
                "Repair the previous invalid skill or memory decision. Fill SkillSpec "
                f"fields instead of SKILL.md. This is a content repair, not a network retry.\n"
                f"{registry_block}"
            )
        return await self.gateway.complete(
            system_prompt=system,
            user_prompt=prompt,
            response_model=LearningDecision,
            agent_id="experience-miner",
        )

    async def decide(
        self,
        trajectory: TrajectoryView,
        *,
        existing_skills: Sequence[dict[str, Any]] | None = None,
    ) -> LearningDecision:
        try:
            return await self._complete(trajectory, existing_skills=existing_skills)
        except Exception as exc:
            return await self._complete(
                trajectory, existing_skills=existing_skills, repair_error=str(exc)
            )
