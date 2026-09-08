from __future__ import annotations

from pathlib import Path

import pytest

from evoci.domain.models import VerificationCommandResult
from evoci.runtime.budget import RunRepairBudget
from evoci.verification.service import (
    VerificationService,
    build_verification_plan,
    evaluate_verification,
)


def test_agent_plan_cannot_replace_mandatory_oracle() -> None:
    planned = build_verification_plan(
        [["python", "-c", "raise SystemExit(1)"]],
        [["python", "-c", "raise SystemExit(0)"]],
    )
    assert [item.source for item in planned] == ["mandatory", "supplementary"]
    assert planned[0].command == ["python", "-c", "raise SystemExit(1)"]


def test_duplicate_commands_keep_mandatory_source() -> None:
    command = ["python", "-c", "raise SystemExit(0)"]
    planned = build_verification_plan([command], [command])
    assert len(planned) == 1
    assert planned[0].source == "mandatory"


def test_missing_oracle_is_unavailable_not_success() -> None:
    result = evaluate_verification(
        [
            VerificationCommandResult(
                command=["python", "-c", "raise SystemExit(0)"],
                exit_code=0,
                stdout="",
                stderr="",
                source="supplementary",
                executed=False,
                skip_reason="no harness oracle",
            )
        ],
        expected_count=0,
        has_harness_oracle=False,
    )
    assert result.passed is False
    assert result.status == "unavailable"
    assert result.oracle_source == "none"


def test_partial_execution_is_incomplete_not_success() -> None:
    result = evaluate_verification(
        [
            VerificationCommandResult(
                command=["python", "-c", "raise SystemExit(0)"],
                exit_code=0,
                stdout="",
                stderr="",
            ),
            VerificationCommandResult(
                command=["python", "-c", "raise SystemExit(1)"],
                exit_code=-1,
                stdout="",
                stderr="",
                executed=False,
                skip_reason="budget exhausted",
            ),
        ],
        expected_count=2,
        has_harness_oracle=True,
        incomplete_reason="run-level tool-call budget exhausted",
    )
    assert result.passed is False
    assert result.status == "incomplete"
    assert result.executed_count == 1
    assert result.expected_count == 2


@pytest.mark.asyncio
async def test_isolated_verification_does_not_leak_side_effects(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("baseline\n")
    result = await VerificationService(timeout=5, max_chars=2000).run(
        workspace=tmp_path,
        mandatory=[
            [
                "python",
                "-c",
                (
                    "from pathlib import Path\n"
                    "Path('undeclared.txt').write_text('leaked')\n"
                    "Path('notes.txt').write_text('mutated')\n"
                    "raise SystemExit(1)"
                ),
            ]
        ],
    )
    assert result.passed is False
    assert result.status == "failed"
    assert not (tmp_path / "undeclared.txt").exists()
    assert (tmp_path / "notes.txt").read_text() == "baseline\n"


@pytest.mark.asyncio
async def test_isolated_verification_observes_approved_patch(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    result = await VerificationService(timeout=5, max_chars=2000).run(
        workspace=tmp_path,
        mandatory=[["python", "-c", "import app; assert app.VALUE == 1"]],
        supplementary=[["python", "-c", "raise SystemExit(0)"]],
    )
    assert result.passed is True
    assert result.status == "passed"
    assert result.executed_count == 2


@pytest.mark.asyncio
async def test_budget_exhaustion_does_not_mark_success(tmp_path: Path) -> None:
    budget = RunRepairBudget(max_model_calls=10, max_tool_calls=1)
    result = await VerificationService(timeout=5, max_chars=2000).run(
        workspace=tmp_path,
        mandatory=[
            ["python", "-c", "raise SystemExit(0)"],
            ["python", "-c", "raise SystemExit(1)"],
        ],
        budget=budget,
    )
    assert result.passed is False
    assert result.status == "incomplete"
    assert result.executed_count == 1
