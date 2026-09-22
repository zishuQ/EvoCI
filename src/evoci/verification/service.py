"""Harness-owned verification that agents cannot replace or partially pass."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evoci.domain.models import (
    VerificationCommandResult,
    VerificationResult,
    parse_verification_command,
)
from evoci.runtime.budget import RepairBudgetExhausted, RunRepairBudget
from evoci.tools.isolation import copy_workspace_with_independent_git
from evoci.tools.policy import PolicyViolation, WorkspaceBoundary
from evoci.tools.shell import CommandRunner

CommandCallback = Callable[[int, list[str], str], Awaitable[None] | None]
CommandResultCallback = Callable[[int, VerificationCommandResult], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class PlannedCommand:
    command: list[str]
    source: Literal["mandatory", "supplementary"]
    cwd: str = "."


def build_verification_plan(
    mandatory: Sequence[object],
    supplementary: Sequence[object] | None = None,
) -> list[PlannedCommand]:
    """Keep harness oracles first; agent checks are extra and never replace them."""

    planned: list[PlannedCommand] = []
    seen: set[tuple[tuple[str, ...], str]] = set()
    for command in mandatory:
        spec = parse_verification_command(command)
        key = (tuple(spec.argv), spec.cwd)
        if key in seen:
            continue
        seen.add(key)
        planned.append(PlannedCommand(command=list(spec.argv), source="mandatory", cwd=spec.cwd))
    for command in supplementary or ():
        spec = parse_verification_command(command)
        key = (tuple(spec.argv), spec.cwd)
        if key in seen:
            continue
        seen.add(key)
        planned.append(
            PlannedCommand(command=list(spec.argv), source="supplementary", cwd=spec.cwd)
        )
    return planned


_ENVIRONMENT_MARKERS = (
    "ERROR collecting",
    "ModuleNotFoundError",
    "ImportError",
    "InvalidVersion",
    "fixture not found",
    "recursive fixture",
    "INTERNALERROR",
    "no tests ran",
)


def environment_failure_reason(result: VerificationCommandResult) -> str | None:
    """Classify environment/collection failures that are not assertion regressions."""

    if not result.executed:
        return None
    if result.timed_out:
        return "timeout"
    text = f"{result.stdout}\n{result.stderr}"
    for marker in _ENVIRONMENT_MARKERS:
        if marker in text:
            return marker
    return None


def evaluate_verification(
    commands: list[VerificationCommandResult],
    *,
    expected_count: int,
    has_harness_oracle: bool,
    incomplete_reason: str | None = None,
    incomplete_cause: Literal["budget", "execution"] | None = None,
    baseline_environment_reason: str | None = None,
) -> VerificationResult:
    executed = [item for item in commands if item.executed]
    if not has_harness_oracle:
        return VerificationResult(
            passed=False,
            status="unavailable",
            commands=commands,
            expected_count=expected_count,
            executed_count=len(executed),
            incomplete_reason=incomplete_reason
            or "no reliable original verification; agent checks cannot substitute",
            oracle_source="none",
        )
    failed = [item for item in executed if item.exit_code != 0 or item.timed_out]
    if failed:
        reasons = [environment_failure_reason(item) for item in failed]
        env_reason = next((reason for reason in reasons if reason), None)
        if env_reason:
            marked_baseline = any(
                "BASELINE_ENVIRONMENT_ERROR=" in f"{item.stdout}\n{item.stderr}"
                for item in failed
            )
            same_as_baseline = marked_baseline or (
                baseline_environment_reason is not None
                and baseline_environment_reason == env_reason
            )
            return VerificationResult(
                passed=False,
                status="inconclusive" if same_as_baseline else "infra_error",
                commands=commands,
                expected_count=expected_count,
                executed_count=len(executed),
                incomplete_reason=env_reason,
                oracle_source="harness",
            )
        return VerificationResult(
            passed=False,
            status="failed",
            commands=commands,
            expected_count=expected_count,
            executed_count=len(executed),
            oracle_source="harness",
        )
    if incomplete_reason or len(executed) < expected_count:
        return VerificationResult(
            passed=False,
            status="incomplete",
            commands=commands,
            expected_count=expected_count,
            executed_count=len(executed),
            incomplete_reason=incomplete_reason
            or f"executed {len(executed)} of {expected_count} required checks",
            incomplete_cause=incomplete_cause,
            oracle_source="harness",
        )
    return VerificationResult(
        passed=True,
        status="passed",
        commands=commands,
        expected_count=expected_count,
        executed_count=len(executed),
        oracle_source="harness",
    )


class VerificationService:
    def __init__(self, *, timeout: float = 120.0, max_chars: int = 32_000) -> None:
        self.timeout = timeout
        self.max_chars = max_chars

    @asynccontextmanager
    async def isolated_workspace(self, source: Path) -> AsyncIterator[Path]:
        with tempfile.TemporaryDirectory(prefix="evoci-verify-") as temporary:
            snapshot = Path(temporary) / "workspace"
            copy_workspace_with_independent_git(source, snapshot)
            yield snapshot

    async def run(
        self,
        *,
        workspace: Path,
        mandatory: Sequence[object],
        supplementary: Sequence[object] | None = None,
        budget: RunRepairBudget | None = None,
        isolate: bool = True,
        on_command_start: CommandCallback | None = None,
        on_command_done: CommandResultCallback | None = None,
    ) -> VerificationResult:
        planned = build_verification_plan(mandatory, supplementary)
        has_oracle = any(item.source == "mandatory" for item in planned)
        if not has_oracle:
            skipped = [
                VerificationCommandResult(
                    command=item.command,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    timed_out=False,
                    source=item.source,
                    executed=False,
                    skip_reason="no harness oracle",
                    cwd=item.cwd,
                )
                for item in planned
            ]
            return evaluate_verification(
                skipped,
                expected_count=0,
                has_harness_oracle=False,
            )

        results: list[VerificationCommandResult] = []
        incomplete_reason: str | None = None
        incomplete_cause: Literal["budget", "execution"] | None = None

        async def execute(root: Path) -> None:
            nonlocal incomplete_reason, incomplete_cause
            runner = CommandRunner(root, timeout=self.timeout, max_chars=self.max_chars)
            for index, item in enumerate(planned):
                WorkspaceBoundary(root).resolve(item.cwd, must_exist=True)
                if budget is not None:
                    try:
                        budget.consume_tool_call()
                    except RepairBudgetExhausted as exc:
                        incomplete_reason = str(exc)
                        incomplete_cause = "budget"
                        results.append(
                            VerificationCommandResult(
                                command=item.command,
                                exit_code=-1,
                                stdout="",
                                stderr="",
                                timed_out=False,
                                source=item.source,
                                executed=False,
                                skip_reason=str(exc),
                                cwd=item.cwd,
                            )
                        )
                        for remaining in planned[index + 1 :]:
                            results.append(
                                VerificationCommandResult(
                                    command=remaining.command,
                                    exit_code=-1,
                                    stdout="",
                                    stderr="",
                                    timed_out=False,
                                    source=remaining.source,
                                    executed=False,
                                    skip_reason=str(exc),
                                    cwd=remaining.cwd,
                                )
                            )
                        return
                if on_command_start is not None:
                    started = on_command_start(index, item.command, item.source)
                    if started is not None:
                        await started
                try:
                    completed = await runner.run(
                        item.command, cwd=item.cwd, extra_env={"CI": "1"}
                    )
                    result = VerificationCommandResult(
                        command=item.command,
                        exit_code=completed.exit_code,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        timed_out=completed.timed_out,
                        source=item.source,
                        cwd=item.cwd,
                    )
                except (OSError, PolicyViolation, TypeError, ValueError) as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    incomplete_reason = reason
                    incomplete_cause = "execution"
                    result = VerificationCommandResult(
                        command=item.command,
                        exit_code=-1,
                        stdout="",
                        stderr=reason,
                        timed_out=False,
                        source=item.source,
                        executed=False,
                        skip_reason=reason,
                        cwd=item.cwd,
                    )
                    results.append(result)
                    if on_command_done is not None:
                        done = on_command_done(index, result)
                        if done is not None:
                            await done
                    for remaining in planned[index + 1 :]:
                        results.append(
                            VerificationCommandResult(
                                command=remaining.command,
                                exit_code=-1,
                                stdout="",
                                stderr="",
                                timed_out=False,
                                source=remaining.source,
                                executed=False,
                                skip_reason=reason,
                                cwd=remaining.cwd,
                            )
                        )
                    return
                results.append(result)
                if on_command_done is not None:
                    done = on_command_done(index, result)
                    if done is not None:
                        await done
                if result.exit_code != 0 or result.timed_out:
                    for remaining in planned[index + 1 :]:
                        results.append(
                            VerificationCommandResult(
                                command=remaining.command,
                                exit_code=-1,
                                stdout="",
                                stderr="",
                                timed_out=False,
                                source=remaining.source,
                                executed=False,
                                skip_reason="earlier check failed",
                                cwd=remaining.cwd,
                            )
                        )
                    return

        try:
            if isolate:
                async with self.isolated_workspace(workspace) as snapshot:
                    await execute(snapshot)
            else:
                await execute(workspace)
        except Exception as exc:
            incomplete_reason = f"{type(exc).__name__}: {exc}"
            if incomplete_cause is None:
                incomplete_cause = "execution"
            if not results:
                results = [
                    VerificationCommandResult(
                        command=item.command,
                        exit_code=-1,
                        stdout="",
                        stderr=incomplete_reason,
                        timed_out=False,
                        source=item.source,
                        executed=False,
                        skip_reason=incomplete_reason,
                        cwd=item.cwd,
                    )
                    for item in planned
                ]

        return evaluate_verification(
            results,
            expected_count=len(planned),
            has_harness_oracle=True,
            incomplete_reason=incomplete_reason,
            incomplete_cause=incomplete_cause,
        )
