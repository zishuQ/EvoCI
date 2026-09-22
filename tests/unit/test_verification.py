from __future__ import annotations

from pathlib import Path

import pytest

from evoci.domain.models import (
    FileEdit,
    VerificationCommandResult,
    VerificationCommandSpec,
    WorkerTask,
)
from evoci.graph.integration import (
    IntegrationError,
    save_patch_artifact,
    verification_plan_from_artifact,
)
from evoci.runtime.budget import RunRepairBudget
from evoci.tools.shell import CommandRunner
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


def test_collection_error_is_infra_error_not_repair_failure() -> None:
    result = evaluate_verification(
        [
            VerificationCommandResult(
                command=["python", "-m", "pytest"],
                exit_code=2,
                stdout="ERROR collecting tests\nImportError: missing helper\n",
                stderr="",
            )
        ],
        expected_count=1,
        has_harness_oracle=True,
    )
    assert result.passed is False
    assert result.status == "infra_error"


def test_same_baseline_environment_error_is_inconclusive() -> None:
    result = evaluate_verification(
        [
            VerificationCommandResult(
                command=["python", "-m", "pytest", "tests/test_x.py::test_target"],
                exit_code=2,
                stdout="ERROR collecting tests\n",
                stderr="fixture not found\nBASELINE_ENVIRONMENT_ERROR=fixture not found\n",
            )
        ],
        expected_count=1,
        has_harness_oracle=True,
    )
    assert result.passed is False
    assert result.status == "inconclusive"


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
    assert result.incomplete_cause == "budget"
    assert result.commands[1].executed is False


def test_same_argv_different_cwd_are_not_deduped() -> None:
    planned = build_verification_plan(
        [VerificationCommandSpec(argv=["python", "-m", "pytest"], cwd=".")],
        [VerificationCommandSpec(argv=["python", "-m", "pytest"], cwd="tests")],
    )
    assert len(planned) == 2
    assert [item.cwd for item in planned] == [".", "tests"]


def test_duplicate_argv_and_cwd_keep_mandatory() -> None:
    command = VerificationCommandSpec(argv=["python", "-c", "raise SystemExit(0)"], cwd=".")
    planned = build_verification_plan([command], [command])
    assert len(planned) == 1
    assert planned[0].source == "mandatory"


def test_cwd_escape_and_absolute_paths_are_rejected() -> None:
    with pytest.raises(ValueError):
        VerificationCommandSpec(argv=["python", "-c", "print(1)"], cwd="../outside")
    with pytest.raises(ValueError):
        VerificationCommandSpec(argv=["python", "-c", "print(1)"], cwd="/tmp")


def test_old_artifact_argv_lists_become_root_cwd(tmp_path: Path) -> None:
    artifact = save_patch_artifact(
        tmp_path,
        task=WorkerTask(task_id="t", kind="repair", objective="fix", write_scope=["a.py"]),
        edits=[FileEdit(path="a.py", content="x\n")],
        baseline_snapshot_id="snap",
        summary="s",
        commands_run=[],
        verification_plan=[["python", "-m", "pytest"]],
    )
    from pathlib import Path as P

    payload = __import__("json").loads(P(artifact).read_text())
    payload["verification_plan"] = [["python", "-m", "pytest"]]
    P(artifact).write_text(__import__("json").dumps(payload))
    specs = verification_plan_from_artifact(payload)
    assert specs == [VerificationCommandSpec(argv=["python", "-m", "pytest"], cwd=".")]


def test_malformed_artifact_verification_plan_raises(tmp_path: Path) -> None:
    with pytest.raises((IntegrationError, ValueError)):
        verification_plan_from_artifact({"verification_plan": "not-a-list"})


def test_new_artifact_round_trip_keeps_cwd(tmp_path: Path) -> None:
    spec = VerificationCommandSpec(argv=["python", "-m", "pytest"], cwd="tests")
    artifact = save_patch_artifact(
        tmp_path,
        task=WorkerTask(task_id="t", kind="repair", objective="fix", write_scope=["a.py"]),
        edits=[FileEdit(path="a.py", content="x\n")],
        baseline_snapshot_id="snap",
        summary="s",
        commands_run=[],
        verification_plan=[spec],
    )
    payload = __import__("json").loads(__import__("pathlib").Path(artifact).read_text())
    loaded = verification_plan_from_artifact(payload)
    assert loaded == [spec]


@pytest.mark.asyncio
async def test_verification_executes_supplementary_in_recorded_cwd(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "probe.py").write_text("VALUE = 1\n")
    result = await VerificationService(timeout=5, max_chars=2000).run(
        workspace=tmp_path,
        mandatory=[["python", "-c", "raise SystemExit(0)"]],
        supplementary=[
            VerificationCommandSpec(
                argv=["python", "-c", "import probe; assert probe.VALUE == 1"],
                cwd="tests",
            )
        ],
    )
    assert result.passed is True
    assert result.commands[1].cwd == "tests"


@pytest.mark.asyncio
async def test_first_mandatory_failure_is_not_budget(tmp_path: Path) -> None:
    result = await VerificationService(timeout=5, max_chars=2000).run(
        workspace=tmp_path,
        mandatory=[["python", "-c", "raise SystemExit(1)"]],
        budget=RunRepairBudget(max_model_calls=10, max_tool_calls=10),
    )
    assert result.status == "failed"
    assert result.incomplete_cause is None
    assert result.commands[0].executed is True


@pytest.mark.asyncio
async def test_command_start_oserror_is_incomplete_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(self: object, argv: object, **kwargs: object) -> object:
        del self, argv, kwargs
        raise OSError("cannot start command")

    monkeypatch.setattr("evoci.verification.service.CommandRunner.run", boom)
    result = await VerificationService(timeout=5, max_chars=2000).run(
        workspace=tmp_path,
        mandatory=[
            ["python", "-c", "raise SystemExit(0)"],
            ["python", "-c", "print('second')"],
        ],
    )
    assert result.passed is False
    assert result.status == "incomplete"
    assert result.incomplete_cause == "execution"
    assert result.incomplete_reason is not None
    assert "OSError" in result.incomplete_reason
    assert result.commands[0].executed is False
    assert result.commands[0].exit_code == -1
    assert result.commands[0].skip_reason is not None
    assert "OSError" in result.commands[0].skip_reason
    assert result.commands[1].executed is False
    assert result.commands[1].exit_code == -1


@pytest.mark.asyncio
async def test_command_runner_honors_relative_cwd(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "probe.py").write_text("ok = 1\n")
    runner = CommandRunner(tmp_path, timeout=5, max_chars=2000)
    result = await runner.run(
        ["python", "-c", "import probe; assert probe.ok == 1"], cwd="tests"
    )
    assert result.exit_code == 0
