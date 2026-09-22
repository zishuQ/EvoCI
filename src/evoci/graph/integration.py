"""Batch snapshot, patch harvest, conflict detection, and rollback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evoci.domain.models import (
    FileEdit,
    FixerOutput,
    PatchProposal,
    VerificationCommandSpec,
    WorkerTask,
    parse_verification_plan,
)
from evoci.tools.filesystem import FileTools
from evoci.tools.isolation import copy_workspace_with_independent_git, workspace_snapshot_id
from evoci.tools.patch import (
    PatchConflict,
    apply_edit,
    precheck_edits,
    snapshot_edit_baseline,
)
from evoci.tools.scope import assert_write_path_allowed, normalize_write_path


class IntegrationError(RuntimeError):
    pass


def artifact_dir(state_dir: Path, run_id: str, batch: int, task_id: str) -> Path:
    path = state_dir / "artifacts" / run_id / f"batch-{batch}" / task_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_patch_artifact(
    directory: Path,
    *,
    task: WorkerTask,
    edits: list[FileEdit],
    baseline_snapshot_id: str,
    summary: str,
    commands_run: list[str],
    verification_plan: list[VerificationCommandSpec] | list[list[str]],
) -> str:
    payload = {
        "task_id": task.task_id,
        "write_scope": list(task.write_scope),
        "baseline_snapshot_id": baseline_snapshot_id,
        "summary": summary,
        "commands_run": commands_run,
        "verification_plan": [
            spec.model_dump(mode="json") for spec in parse_verification_plan(verification_plan)
        ],
        "edits": [edit.model_dump(mode="json") for edit in edits],
        "changed_files": [edit.path for edit in edits],
    }
    target = directory / "patch.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(target)


def load_patch_artifact(artifact_ref: str) -> dict[str, Any]:
    path = Path(artifact_ref)
    if not path.is_file():
        raise IntegrationError(f"missing patch artifact: {artifact_ref}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IntegrationError(f"invalid patch artifact: {artifact_ref}")
    return payload


def edits_from_artifact(payload: dict[str, Any]) -> list[FileEdit]:
    return [FileEdit.model_validate(item) for item in payload.get("edits") or []]


def verification_plan_from_artifact(payload: dict[str, Any]) -> list[VerificationCommandSpec]:
    try:
        return parse_verification_plan(payload.get("verification_plan"))
    except (TypeError, ValueError) as exc:
        raise IntegrationError(f"invalid verification_plan in patch artifact: {exc}") from exc


def validate_edits_against_scope(edits: list[FileEdit], write_scope: list[str]) -> None:
    allowed = {normalize_write_path(path) for path in write_scope}
    for edit in edits:
        assert_write_path_allowed(edit.path, allowed)


def save_batch_snapshot(source: Path, destination: Path) -> str:
    import shutil

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_workspace_with_independent_git(source, destination)
    return workspace_snapshot_id(destination)


def restore_batch_snapshot(snapshot: Path, workspace: Path) -> None:
    if not snapshot.exists():
        raise IntegrationError(f"batch snapshot missing: {snapshot}")
    import shutil

    workspace = workspace.resolve()
    snapshot = snapshot.resolve()
    if workspace == snapshot or workspace in snapshot.parents:
        raise IntegrationError("batch snapshot must not live inside the workspace")
    for item in list(workspace.iterdir()):
        if item.name in {".evoci", ".git"}:
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in snapshot.iterdir():
        if item.name in {".evoci", ".git"}:
            continue
        destination = workspace / item.name
        if item.is_dir() and not item.is_symlink():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def integrate_edits(
    workspace: Path,
    edits: list[FileEdit],
    *,
    expected_baseline: str | None,
    hash_strict: bool,
) -> dict[str, str | None]:
    if expected_baseline is not None:
        current = workspace_snapshot_id(workspace)
        if current != expected_baseline:
            raise PatchConflict(
                f"workspace snapshot {current} does not match baseline {expected_baseline}"
            )
    if not edits:
        return {}
    precheck_edits(workspace, edits, hash_strict=hash_strict)
    tools = FileTools(workspace, writable=True)
    written: dict[str, str | None] = {}
    for edit in edits:
        created, modified = apply_edit(workspace, tools, edit, hash_strict=hash_strict)
        del created, modified
        written[edit.path] = None if edit.delete else edit.content
    return written


def fixer_output_from_edits(
    *,
    summary: str,
    edits: list[FileEdit],
    commands_run: list[str],
    verification_plan: list[VerificationCommandSpec] | list[list[str]],
    risk: str | None = None,
) -> FixerOutput:
    resolved_risk = risk or ("low" if len(edits) <= 4 else "medium")
    if resolved_risk not in {"low", "medium", "high"}:
        resolved_risk = "medium"
    return FixerOutput(
        proposal=PatchProposal(
            summary=summary,
            changed_files=[edit.path for edit in edits],
            commands_run=commands_run,
            risk=resolved_risk,  # type: ignore[arg-type]
            verification_plan=parse_verification_plan(verification_plan),
        ),
        edits=edits,
    )


def snapshot_for_edits(workspace: Path, edits: list[FileEdit]) -> dict[str, Any]:
    return snapshot_edit_baseline(workspace, edits)


def candidate_prompt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Task-prompt fragment for a resumed candidate. Never includes file bodies."""

    return {
        "candidate_task_id": payload.get("task_id"),
        "summary": payload.get("summary"),
        "changed_files": list(payload.get("changed_files") or []),
        "baseline_snapshot_id": payload.get("baseline_snapshot_id"),
        "commands_run": list(payload.get("commands_run") or []),
    }


def apply_plan_path(state_dir: Path, run_id: str, operation_id: str) -> Path:
    directory = state_dir / "artifacts" / run_id / "apply"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{operation_id}.json"


def save_apply_plan(
    path: Path,
    *,
    parent_snapshot_id: str,
    artifact_ref: str | None,
    edits: list[FileEdit],
    baseline: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            {
                "parent_snapshot_id": parent_snapshot_id,
                "artifact_ref": artifact_ref,
                "edits": [edit.model_dump(mode="json") for edit in edits],
                "baseline": baseline,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_apply_plan(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IntegrationError(f"invalid apply plan: {path}")
    return payload
