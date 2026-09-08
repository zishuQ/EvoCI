"""Harness-owned verification that agents cannot replace or partially pass."""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evoci.domain.models import VerificationCommandResult, VerificationResult
from evoci.runtime.budget import RepairBudgetExhausted, RunRepairBudget
from evoci.tools.isolation import copy_workspace_with_independent_git
from evoci.tools.policy import PolicyViolation
from evoci.tools.shell import CommandRunner

CommandCallback = Callable[[int, list[str], str], Awaitable[None] | None]
CommandResultCallback = Callable[[int, VerificationCommandResult], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class PlannedCommand:
    command: list[str]
    source: Literal["mandatory", "supplementary"]


def build_verification_plan(
    mandatory: Sequence[Sequence[str]],
    supplementary: Sequence[Sequence[str]] | None = None,
) -> list[PlannedCommand]:
    """Keep harness oracles first; agent checks are extra and never replace them."""

    planned: list[PlannedCommand] = []
    seen: set[tuple[str, ...]] = set()
    for command in mandatory:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        planned.append(PlannedCommand(command=list(command), source="mandatory"))
    for command in supplementary or ():
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        planned.append(PlannedCommand(command=list(command), source="supplementary"))
    return planned


def evaluate_verification(
    commands: list[VerificationCommandResult],
    *,
    expected_count: int,
    has_harness_oracle: bool,
    incomplete_reason: str | None = None,
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
    failed = any(item.exit_code != 0 or item.timed_out for item in executed)
    if failed:
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
        mandatory: Sequence[Sequence[str]],
        supplementary: Sequence[Sequence[str]] | None = None,
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

        async def execute(root: Path) -> None:
            nonlocal incomplete_reason
            runner = CommandRunner(root, timeout=self.timeout, max_chars=self.max_chars)
            for index, item in enumerate(planned):
                if budget is not None:
                    try:
                        budget.consume_tool_call()
                    except RepairBudgetExhausted as exc:
                        incomplete_reason = str(exc)
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
                                )
                            )
                        return
                if on_command_start is not None:
                    started = on_command_start(index, item.command, item.source)
                    if started is not None:
                        await started
                try:
                    completed = await runner.run(item.command, extra_env={"CI": "1"})
                    result = VerificationCommandResult(
                        command=item.command,
                        exit_code=completed.exit_code,
                        stdout=completed.stdout,
                        stderr=completed.stderr,
                        timed_out=completed.timed_out,
                        source=item.source,
                    )
                except (OSError, PolicyViolation, TypeError, ValueError) as exc:
                    result = VerificationCommandResult(
                        command=item.command,
                        exit_code=-1,
                        stdout="",
                        stderr=f"{type(exc).__name__}: {exc}",
                        timed_out=False,
                        source=item.source,
                    )
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
                    )
                    for item in planned
                ]

        return evaluate_verification(
            results,
            expected_count=len(planned),
            has_harness_oracle=True,
            incomplete_reason=incomplete_reason,
        )
