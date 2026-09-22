"""Terminal episode, memory, and skill learning owned by the harness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from evoci.capability.miner import SkillLearningDecision
from evoci.capability.models import SkillMemoryEntry
from evoci.domain.models import FailureClass, SupervisorDecision, WorkerExecutionResult
from evoci.graph.state import EvoCIState
from evoci.memory.consolidation import commit_candidate
from evoci.memory.fingerprint import failure_fingerprint
from evoci.memory.models import Episode, LongTermFactCandidate
from evoci.model.gateway import ModelGatewayError
from evoci.runtime.budget import RepairBudgetExhausted
from evoci.runtime.events import EventType, RunEvent
from evoci.runtime.learning_payload import (
    build_learning_input,
    clip_list,
    clip_text,
)
from evoci.runtime.usage import make_usage_observer
from evoci.tools.shell import run_cancellable


def _operation_key(state: EvoCIState, key_type: str, *components: str) -> str:
    prefix = f"{key_type}:{state['run_id']}"
    suffix = ":".join(components)
    return f"{prefix}:{suffix}" if suffix else prefix


def emit_event(
    runtime: Any,
    state: EvoCIState,
    event_type: EventType,
    *,
    agent_id: str | None = None,
    invocation_id: str | None = None,
    discriminator: str = "",
    payload: dict[str, Any] | None = None,
) -> RunEvent:
    if runtime.recorder is not None:
        recorded = runtime.recorder.emit(
            run_id=state["run_id"],
            event_type=event_type,
            agent_id=agent_id,
            invocation_id=invocation_id,
            event_key=discriminator or "0",
            payload=payload,
        )
        return cast(RunEvent, recorded)
    event = RunEvent(
        event_id=(
            f"{state['run_id']}:{agent_id or 'harness'}:{invocation_id or 'default'}:"
            f"{event_type.value}:{discriminator or '0'}"
        ),
        run_id=state["run_id"],
        type=event_type,
        agent_id=agent_id,
        invocation_id=invocation_id,
        payload=payload or {},
    )
    if runtime.event_store is not None:
        runtime.event_store.append(event)
    return event


def _event(runtime: Any, state: EvoCIState, event_type: EventType, **kwargs: Any) -> RunEvent:
    return emit_event(runtime, state, event_type, **kwargs)


def classified_failure(exc: Exception, *, phase: str, **extra: Any) -> dict[str, Any]:
    if isinstance(exc, RepairBudgetExhausted):
        failure_class: FailureClass = "budget"
    elif isinstance(exc, ModelGatewayError):
        failure_class = "model"
    else:
        raise exc
    payload = {
        "phase": phase,
        "status": "failed",
        "failure_class": failure_class,
        "failure_reason": str(exc),
        **extra,
    }
    payload.setdefault("failure_stage", phase)
    return payload


def has_domain_attempt(state: EvoCIState) -> bool:
    return bool(
        state.get("fixer_output")
        or state.get("verification")
        or state.get("verification_history")
        or state.get("worker_results")
        or state.get("evidence")
        or state.get("decision")
    )


def should_persist_episode(
    *,
    success: bool,
    failure_class: FailureClass | None,
    state: EvoCIState,
) -> bool:
    if success:
        return True
    has_patch = state.get("fixer_output") is not None or any(
        result.changed_files for result in state.get("worker_results", [])
    )
    has_verification = bool(state.get("verification") or state.get("verification_history"))
    if failure_class == "model":
        return has_patch and has_verification
    if failure_class == "infrastructure":
        return False
    if failure_class == "budget":
        return has_domain_attempt(state)
    if failure_class == "policy":
        return has_patch
    return True


def resolve_failure_class(state: EvoCIState, *, success: bool) -> FailureClass | None:
    if success:
        return None
    recorded = state.get("failure_class")
    if recorded is not None:
        return recorded
    verification = state.get("verification")
    if (
        verification is not None
        and getattr(verification, "status", None) == "incomplete"
    ):
        cause = getattr(verification, "incomplete_cause", None)
        if cause == "budget":
            return "budget"
        if cause == "execution":
            return "infrastructure"
    classes = state.get("failure_classes") or []
    if classes:
        last = classes[-1]
        if last in {"repair", "model", "budget", "policy", "infrastructure"}:
            return last  # type: ignore[return-value]
    if state.get("fixer_output") is not None or any(
        result.changed_files for result in state.get("worker_results", [])
    ):
        return "repair"
    reason = (state.get("failure_reason") or "").lower()
    if "exhausted" in reason or "budget" in reason:
        return "budget"
    return "repair"


async def persist_run_outcome(
    runtime: Any,
    state: EvoCIState,
    *,
    success: bool,
    failure_reason: str | None,
    failure_class: FailureClass | None = None,
    external_failure_details: Sequence[str] | None = None,
) -> dict[str, Any]:
    status: Literal["success", "failed"] = "success" if success else "failed"
    resolved_class = failure_class or resolve_failure_class(state, success=success)
    terminal = _event(
        runtime,
        state,
        EventType.RUN_COMPLETED if success else EventType.RUN_FAILED,
        discriminator="terminal",
        payload={
            "status": status,
            "reason": failure_reason,
            "failure_class": resolved_class,
        },
    )
    if runtime.run_store is not None:
        runtime.run_store.update_status(state["run_id"], status)
    assert runtime.recorder is not None
    trajectory = runtime.recorder.build_view(
        run_id=state["run_id"],
        verification_history=state.get("verification_history", []),
        final_status=status,
        failure_reason=failure_reason,
    )
    events: list[RunEvent] = [terminal]
    episode_id: str | None = None
    candidate_skill_id: str | None = None
    learning_decision: dict[str, object] | None = None
    learning_errors: list[dict[str, str]] = []
    output = state.get("fixer_output")
    verification = state.get("verification")
    decisions: list[SupervisorDecision] = []
    raw_decision = state.get("decision")
    if raw_decision is not None:
        decisions.append(
            raw_decision
            if isinstance(raw_decision, SupervisorDecision)
            else SupervisorDecision.model_validate(raw_decision)
        )
    worker_results: list[WorkerExecutionResult] = []
    for item in state.get("worker_results", []):
        worker_results.append(
            item
            if isinstance(item, WorkerExecutionResult)
            else WorkerExecutionResult.model_validate(item)
        )
    learning = build_learning_input(
        run_id=state["run_id"],
        repo=state["repo"].full_name,
        task_family=state["ci_failure"].task_family,
        failure_summary=state["ci_failure"].summary,
        decisions=decisions,
        worker_results=worker_results,
        verification=verification,
        trajectory=trajectory,
        success=success,
        failure_reason=failure_reason,
        failed_candidates=state.get("failed_candidates", []),
        recommended_skill_ids=state.get("recommended_skill_ids", []),
    )

    def learning_error(stage: str, exc: Exception) -> None:
        payload = {
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc) or type(exc).__name__,
        }
        learning_errors.append(payload)
        events.append(
            _event(
                runtime,
                state,
                EventType.LEARNING_ERROR,
                discriminator=f"{stage}:{len(learning_errors)}",
                payload=payload,
            )
        )

    persist_episode = runtime.memory_store is not None and should_persist_episode(
        success=success,
        failure_class=resolved_class,
        state=state,
    )
    if persist_episode:
        assert runtime.memory_store is not None
        try:
            verification_failures = clip_list(
                [
                    command.stderr or f"exit code {command.exit_code}"
                    for history in state.get("verification_history", [])
                    for command in history.commands
                    if command.exit_code != 0 or command.timed_out
                ]
            )
            episode = Episode(
                id=f"episode:{state['run_id']}",
                run_id=state["run_id"],
                repo=state["repo"].full_name,
                task_family=state["ci_failure"].task_family,
                failure_summary=clip_text(state["ci_failure"].summary) or "",
                root_cause=learning.root_cause,
                important_evidence=learning.evidence_ids,
                attempts=state.get("supervisor_batch", 0),
                successful_fix_summary=(
                    clip_text(output.proposal.summary)
                    if success and output is not None
                    else None
                ),
                tools_used=clip_list([tool.tool_name for tool in trajectory.tool_calls]),
                hypotheses_attempted=learning.plan_revisions,
                verification_failures=verification_failures,
                failure_reason=clip_text(failure_reason),
                success=success,
                failure_fingerprint=failure_fingerprint(state["repo"], state["ci_failure"]),
                repo_revision=state["repo"].base_commit,
                failure_class=resolved_class or "repair",
                failure_stage=(
                    None if success else (state.get("failure_stage") or state.get("phase"))
                ),
                attempted_fix_summaries=learning.change_summaries,
                attempted_files=clip_list([*trajectory.created_files, *trajectory.modified_files]),
                external_failure_details=clip_list(list(external_failure_details or ())),
            )
            runtime.memory_store.add_episode(episode)
            episode_id = episode.id
            events.append(
                _event(
                    runtime,
                    state,
                    EventType.MEMORY_CREATED,
                    discriminator="episode",
                    payload={
                        "episode_id": episode.id,
                        "success": success,
                        "failure_class": resolved_class,
                    },
                )
            )
        except Exception as exc:
            learning_error("episode", exc)
            raise RuntimeError(f"Critical: failed to persist episode: {exc}") from exc

    used_skill_ids = list(dict.fromkeys(ref.skill_id for ref in trajectory.skills_used))
    skill_attribution: dict[str, dict[str, bool]] = {}
    for skill_id in used_skill_ids:
        traces = [
            trace for trace in trajectory.skill_use_traces if trace.skill_id == skill_id
        ]
        explicit_failure = any(trace.execution_success is False for trace in traces)
        skill_attribution[skill_id] = {
            "explicit_failure": explicit_failure,
            "success": bool(success) and not explicit_failure,
        }
    skill_action: str | None = None
    long_term_facts_added = 0
    skill_memory_appended: str | None = None

    if runtime.capability_registry is not None and resolved_class not in {
        "infrastructure",
        "model",
        "budget",
    }:
        try:
            for skill_id in used_skill_ids:
                record = runtime.capability_registry.get(skill_id)
                if record is None:
                    continue
                attributed = skill_attribution[skill_id]
                try:
                    runtime.capability_registry.record_use(
                        skill_id,
                        success=attributed["success"],
                        patched=bool(trajectory.created_files or trajectory.modified_files),
                        operation_key=_operation_key(state, "skill-use", skill_id),
                    )
                except Exception as exc:
                    learning_error(f"skill_use_{skill_id}", exc)
        except Exception as exc:
            learning_error("skill_outcomes", exc)

    if (
        success
        and runtime.memory_store is not None
        and runtime.memory_consolidator is not None
        and verification is not None
    ):
        try:
            decision_key = f"memory-decision:{state['run_id']}"
            previous = runtime.memory_store.operation_result(decision_key)
            if previous is None:
                _event(
                    runtime,
                    state,
                    EventType.MODEL_CALL,
                    agent_id="memory-consolidator",
                    discriminator="final",
                    payload={"phase": "structured", "budget_scope": "post_run"},
                )
                existing_facts = [
                    fact.content
                    for fact in runtime.memory_store.list_long_term(
                        repository=state["repo"].full_name, limit=8
                    )
                ]
                candidate = await runtime.memory_consolidator.propose(
                    run_id=state["run_id"],
                    repo=state["repo"].full_name,
                    task_family=state["ci_failure"].task_family,
                    learning=learning,
                    existing_facts=existing_facts,
                    usage_observer=make_usage_observer(
                        recorder=runtime.recorder,
                        run_id=state["run_id"],
                        agent_id="memory-consolidator",
                        invocation_id="final",
                        budget_scope="post_run",
                        call_key="final",
                    ),
                )
                runtime.memory_store.record_operation(
                    decision_key, candidate.model_dump(mode="json")
                )
            else:
                candidate = LongTermFactCandidate.model_validate(previous)
            memory_id = commit_candidate(
                runtime.memory_store,
                candidate,
                run_id=state["run_id"],
                repository=state["repo"].full_name,
                operation_key=_operation_key(state, "memory-commit"),
            )
            if memory_id:
                long_term_facts_added = 1
                events.append(
                    _event(
                        runtime,
                        state,
                        EventType.MEMORY_CREATED,
                        discriminator="long-term",
                        payload={"memory_id": memory_id},
                    )
                )
        except Exception as ext:
            learning_error("memory_consolidation", ext)

    should_mine = False
    if (
        runtime.skill_miner is not None
        and runtime.capability_registry is not None
        and resolved_class not in {"infrastructure", "model", "budget"}
    ):
        try:
            should_mine = runtime.skill_miner.should_mine(
                success=success,
                failure_class=resolved_class,
                skills_used=used_skill_ids,
                tool_calls=trajectory.tool_call_count,
                failed_attempts=len(trajectory.failed_attempts),
                reusable_script_created=trajectory.reusable_script_created,
                repeated_pattern_detected=bool(trajectory.memories_selected),
            )
        except Exception as exc:
            learning_error("skill_mining", exc)
    usage_lessons: dict[str, str] = {}
    if should_mine:
        try:
            skill_miner = runtime.skill_miner
            capability_registry = runtime.capability_registry
            assert skill_miner is not None
            assert capability_registry is not None
            decision_key = f"learning-decision:{state['run_id']}"
            previous = capability_registry.operation_result(decision_key)
            if previous is None:
                _event(
                    runtime,
                    state,
                    EventType.MODEL_CALL,
                    agent_id="skill-miner",
                    discriminator="final",
                    payload={"phase": "structured", "budget_scope": "post_run"},
                )
                existing_skills = [
                    {
                        "skill_id": skill.manifest.skill_id,
                        "name": skill.manifest.name,
                        "description": skill.manifest.description,
                        "triggers": skill.manifest.triggers,
                        "task_families": skill.manifest.task_families,
                    }
                    for skill in capability_registry.list(enabled_only=True)
                ]
                decision = await skill_miner.decide(
                    trajectory,
                    existing_skills=existing_skills,
                    used_skill_ids=used_skill_ids,
                    learning=learning,
                    usage_observer=make_usage_observer(
                        recorder=runtime.recorder,
                        run_id=state["run_id"],
                        agent_id="skill-miner",
                        invocation_id="final",
                        budget_scope="post_run",
                        call_key="final",
                    ),
                )
                capability_registry.record_operation(
                    decision_key, decision.model_dump(mode="json")
                )
            else:
                decision = SkillLearningDecision.model_validate(previous)
            learning_decision = decision.model_dump()
            usage_lessons = {
                lesson.skill_id: lesson.lesson
                for lesson in decision.usage_lessons
                if lesson.skill_id in set(used_skill_ids)
            }
            if success and decision.action in {"new_skill", "update_skill"}:
                assert decision.candidate_skill is not None
                try:
                    if decision.action == "update_skill":
                        assert decision.target_skill_id is not None
                        created = await run_cancellable(
                            capability_registry.update_skill,
                            decision.target_skill_id,
                            decision.candidate_skill,
                            operation_key=_operation_key(state, "learning-skill"),
                        )
                        skill_action = "updated"
                    else:
                        created = await run_cancellable(
                            capability_registry.create_skill,
                            decision.candidate_skill,
                            operation_key=_operation_key(state, "learning-skill"),
                        )
                        skill_action = "created"
                    candidate_skill_id = created.manifest.skill_id
                    events.append(
                        _event(
                            runtime,
                            state,
                            (
                                EventType.SKILL_UPDATED
                                if skill_action == "updated"
                                else EventType.SKILL_CANDIDATE_CREATED
                            ),
                            discriminator=created.manifest.skill_id,
                            payload={"skill_id": created.manifest.skill_id, "action": skill_action},
                        )
                    )
                except Exception as exc:
                    learning_error("skill_candidate_creation", exc)
        except Exception as exc:
            learning_error("skill_mining", exc)

    write_skill_memory = (
        runtime.capability_registry is not None
        and used_skill_ids
        and resolved_class not in {"infrastructure", "model"}
        and (success or resolved_class in {None, "repair"})
    )
    if write_skill_memory:
        assert runtime.capability_registry is not None
        success_fallback = "This skill was used during a successful repair."
        failure_fallback = (
            "This skill was used but the repair failed; treat this as "
            "counterevidence, not a verified procedure."
        )
        script_failure_lesson = (
            "The skill script failed during this run even though the overall repair succeeded."
        )
        for skill_id in used_skill_ids:
            attributed = skill_attribution.get(
                skill_id, {"explicit_failure": False, "success": bool(success)}
            )
            if attributed["explicit_failure"]:
                outcome: Literal["success", "failure"] = "failure"
                lesson = script_failure_lesson if success else failure_fallback
            elif attributed["success"]:
                outcome = "success"
                lesson = usage_lessons.get(skill_id, success_fallback)
            else:
                outcome = "failure"
                lesson = usage_lessons.get(skill_id, failure_fallback)
            try:
                appended = runtime.capability_registry.append_skill_memory(
                    skill_id,
                    SkillMemoryEntry(
                        run_id=state["run_id"],
                        repository=state["repo"].full_name,
                        task_summary=state["ci_failure"].summary,
                        outcome=outcome,
                        lesson=lesson,
                    ),
                )
                if appended:
                    skill_memory_appended = (
                        "success appended" if outcome == "success" else "failure appended"
                    )
            except Exception as exc:
                learning_error(f"skill_memory_{skill_id}", exc)

    if skill_action and candidate_skill_id:
        skill_summary = f"{candidate_skill_id} {skill_action}"
    elif used_skill_ids:
        skill_summary = f"{used_skill_ids[0]} used"
    else:
        skill_summary = "none"
    learning_summary = {
        "episode": "stored" if episode_id else "none",
        "long_term_facts": long_term_facts_added,
        "skill": skill_summary,
        "skill_memory": skill_memory_appended or "none",
        "payload_chars": learning.payload_chars,
    }
    usage_complete = True
    if runtime.recorder is not None:
        from evoci.runtime.usage import usage_complete_from_events

        usage_complete = usage_complete_from_events(runtime.recorder.events(state["run_id"]))
    return {
        "phase": "finalize" if success else "failed",
        "status": status,
        "failure_reason": failure_reason,
        "failure_class": resolved_class,
        "failure_stage": None if success else (state.get("failure_stage") or state.get("phase")),
        "episode_id": episode_id,
        "candidate_skill_id": candidate_skill_id,
        "learning_decision": learning_decision,
        "learning_errors": learning_errors,
        "learning_deferred": False,
        "learning_summary": learning_summary,
        "learning_payload_chars": learning.payload_chars,
        "usage_complete": usage_complete,
        "events": events,
    }



