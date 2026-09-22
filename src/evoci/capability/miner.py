"""Trajectory-triggered skill learning decisions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evoci.capability.models import SkillCandidate
from evoci.model.gateway import ModelGateway, UsageObserver
from evoci.runtime.learning_payload import LearningInput, compact_tool_traces
from evoci.runtime.trajectory import TrajectoryView


class SkillUsageLesson(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    lesson: str


class SkillLearningDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["none", "new_skill", "update_skill"]
    rationale: str
    candidate_skill: SkillCandidate | None = None
    target_skill_id: str | None = None
    usage_lessons: list[SkillUsageLesson] = Field(default_factory=list)

    @model_validator(mode="after")
    def action_payload_is_complete(self) -> SkillLearningDecision:
        if self.action in {"new_skill", "update_skill"} and self.candidate_skill is None:
            raise ValueError(f"{self.action} requires candidate_skill")
        if self.action == "update_skill" and not self.target_skill_id:
            raise ValueError("update_skill requires target_skill_id")
        if self.action == "new_skill" and self.target_skill_id is not None:
            raise ValueError("new_skill cannot declare update lineage")
        if self.action == "none" and self.candidate_skill is not None:
            raise ValueError("none cannot include candidate_skill")
        return self


class SkillMining(Protocol):
    def should_mine(
        self,
        *,
        success: bool,
        failure_class: str | None,
        skills_used: Sequence[str],
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
        used_skill_ids: Sequence[str] | None = None,
        learning: LearningInput | None = None,
        usage_observer: UsageObserver | None = None,
    ) -> SkillLearningDecision: ...


class SkillMiner:
    def __init__(self, gateway: ModelGateway, *, tool_call_threshold: int = 5) -> None:
        self.gateway = gateway
        self.tool_call_threshold = tool_call_threshold

    def should_mine(
        self,
        *,
        success: bool,
        failure_class: str | None,
        skills_used: Sequence[str],
        tool_calls: int,
        failed_attempts: int,
        reusable_script_created: bool,
        repeated_pattern_detected: bool,
    ) -> bool:
        del failed_attempts, repeated_pattern_detected
        if failure_class in {"infrastructure", "model"}:
            return False
        if success:
            del tool_calls, reusable_script_created
            return True
        return failure_class in {None, "repair"} and bool(skills_used)

    async def _complete(
        self,
        trajectory: TrajectoryView,
        *,
        existing_skills: Sequence[dict[str, Any]] | None = None,
        used_skill_ids: Sequence[str] | None = None,
        learning: LearningInput | None = None,
        repair_error: str | None = None,
        usage_observer: UsageObserver | None = None,
    ) -> SkillLearningDecision:
        skills = list(existing_skills or [])
        used = list(used_skill_ids or [ref.skill_id for ref in trajectory.skills_used])
        if skills:
            registry_block = (
                "Current skill registry (the only valid update_skill targets):\n"
                f"{json.dumps(skills)}\n"
                "For update_skill, set target_skill_id from this list."
            )
        else:
            registry_block = (
                "The skill registry is empty. update_skill is not available; "
                "choose none or new_skill."
            )
        used_block = (
            f"Skills actually used in this run: {json.dumps(used)}. "
            "usage_lessons may only reference these skill IDs."
            if used
            else "No skills were used in this run. usage_lessons must be empty."
        )
        failed_run = trajectory.final_status != "success"
        if failed_run:
            constraint = (
                "This repair failed. You cannot create or update a skill. "
                "Return action=none. You may add usage_lessons for actually used skills; "
                "those lessons are counterevidence, not recommended procedures."
            )
        else:
            constraint = (
                "If no skill was used, you may create a new reusable procedure. "
                "If a skill was used, prefer update_skill when the procedure should change, "
                "and always include usage_lessons for used skills. "
                "Do not write long-term memory facts; the harness extracts those separately. "
                "Prefer new_skill over none when the trajectory contains a reusable procedure."
            )
        compact: dict[str, Any] = {
            "run_id": trajectory.run_id,
            "final_status": trajectory.final_status,
            "failure_reason": trajectory.failure_reason,
            "agents": [item.model_dump() for item in trajectory.agents],
            "failed_attempts": [item.model_dump() for item in trajectory.failed_attempts],
            "evidence": trajectory.evidence[:20],
            "created_files": trajectory.created_files,
            "modified_files": trajectory.modified_files,
            "skills_used": [ref.model_dump() for ref in trajectory.skills_used],
            "tool_summaries": compact_tool_traces(trajectory.tool_calls),
        }
        if learning is not None:
            compact["learning"] = learning.model_dump(mode="json")
        if repair_error is None:
            prompt = json.dumps(compact, default=str)
            system = (
                "Decide whether this trajectory teaches nothing, a new reusable procedure, "
                "or an update to an existing skill. Skills must generalize beyond the current "
                "run and preserve useful helper artifacts as package scripts when applicable. "
                "When proposing a skill, fill candidate_skill.spec (purpose, when_to_use, "
                "procedure, pitfalls, verification, bundled_resources). spec.verification is "
                "guidance for future agents about how to verify a repair in the repository; "
                "it is not executed when the skill is installed. Do not write SKILL.md; "
                "the runtime renders it. Scripts must be Python. "
                f"{constraint}\n{used_block}\n{registry_block}"
            )
        else:
            prompt = (
                f"{json.dumps(compact, default=str)}\n\n"
                f"Previous structured output failed validation: {repair_error}. "
                "Return a complete SkillLearningDecision. If proposing a skill, include a valid "
                "spec with every required field."
            )
            system = (
                "Repair the previous invalid skill decision. Fill SkillSpec "
                f"fields instead of SKILL.md. This is a content repair, not a network retry.\n"
                f"{constraint}\n{used_block}\n{registry_block}"
            )
        return await self.gateway.complete(
            system_prompt=system,
            user_prompt=prompt,
            response_model=SkillLearningDecision,
            agent_id="skill-miner",
            usage_observer=usage_observer,
        )

    async def decide(
        self,
        trajectory: TrajectoryView,
        *,
        existing_skills: Sequence[dict[str, Any]] | None = None,
        used_skill_ids: Sequence[str] | None = None,
        learning: LearningInput | None = None,
        usage_observer: UsageObserver | None = None,
    ) -> SkillLearningDecision:
        used = list(used_skill_ids or [ref.skill_id for ref in trajectory.skills_used])
        total_tool_calls = sum(agent.tool_calls for agent in trajectory.agents)
        if (
            trajectory.final_status == "success"
            and total_tool_calls == 0
            and not trajectory.reusable_script_created
        ):
            return SkillLearningDecision(
                action="none",
                rationale="trajectory has no tool usage or reusable scripts to learn from",
            )

        try:
            decision = await self._complete(
                trajectory,
                existing_skills=existing_skills,
                used_skill_ids=used,
                learning=learning,
                usage_observer=usage_observer,
            )
        except Exception as exc:
            try:
                decision = await self._complete(
                    trajectory,
                    existing_skills=existing_skills,
                    used_skill_ids=used,
                    learning=learning,
                    repair_error=str(exc),
                    usage_observer=usage_observer,
                )
            except Exception:
                raise exc from None
        return self._constrain(decision, trajectory=trajectory, used_skill_ids=used)

    @staticmethod
    def _constrain(
        decision: SkillLearningDecision,
        *,
        trajectory: TrajectoryView,
        used_skill_ids: Sequence[str],
    ) -> SkillLearningDecision:
        allowed = set(used_skill_ids)
        lessons = [lesson for lesson in decision.usage_lessons if lesson.skill_id in allowed]
        action = decision.action
        candidate = decision.candidate_skill
        target = decision.target_skill_id
        if trajectory.final_status != "success":
            action = "none"
            candidate = None
            target = None
        elif allowed:
            if action != "update_skill" or target not in allowed:
                action = "none"
                candidate = None
                target = None
        elif action == "update_skill":
            action = "none"
            candidate = None
            target = None
        if action == "none":
            candidate = None
            target = None
        return decision.model_copy(
            update={
                "action": action,
                "candidate_skill": candidate,
                "target_skill_id": target,
                "usage_lessons": lessons,
            }
        )
