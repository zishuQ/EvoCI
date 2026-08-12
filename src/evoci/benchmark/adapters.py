"""Offline-first adapters for fixtures and CI-Repair-Bench exports."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml  # type: ignore[import-untyped]

from evoci.benchmark.models import (
    AgentTaskView,
    FailedStep,
    GroundTruth,
    NormalizedCIFailure,
    PreparedTask,
    TaskMeta,
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RUN_LINE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}T\S+\s+)?(?:##\[group\])?Run\s+(.+?)\s*$")
_SHELL_META = re.compile(r"(?:&&|\|\||[;|<>`]|\$\(|\$\{)")
_LOG_KEYS = ("log", "logs", "message", "output", "text")
_NAME_KEYS = ("step_name", "name", "step")
_COMMAND_KEYS = ("command", "failed_command", "run", "argv")


class BenchmarkAdapter(Protocol):
    def list_tasks(self) -> list[TaskMeta]: ...

    def prepare_task(self, task_id: str) -> PreparedTask: ...

    def ground_truth(self, task_id: str) -> GroundTruth: ...


def _required_text(value: Any, *, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise ValueError(f"{field} must be text, got {type(value).__name__}")


def _identifier_text(value: Any, *, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError(f"{field} must be a string or integer identifier")


def _optional_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float, bool)):
        return json.dumps(value)
    return None


def _normalize_workflow_yaml(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (Mapping, list, tuple)):
        rendered = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
        if isinstance(rendered, str):
            return rendered
        raise ValueError("workflow serializer did not return text")
    raise ValueError(f"workflow must be YAML text or structured data, got {type(value).__name__}")


def _normalize_log_fragment(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return "\n".join(
            fragment for item in value if (fragment := _normalize_log_fragment(item)).strip()
        )
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (int, float, bool)):
        return json.dumps(value)
    raise ValueError(f"unsupported CI log value: {type(value).__name__}")


def _parse_command(value: Any) -> list[str] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if value and all(isinstance(part, str) and part for part in value):
            return list(value)
        return None
    if isinstance(value, Mapping):
        for key in ("argv", "command", "run"):
            if key in value:
                return _parse_command(value[key])
        return None
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or "\n" in raw or len(raw) > 4096 or _SHELL_META.search(raw):
        return None
    try:
        command = shlex.split(raw)
    except ValueError:
        return None
    if not command or command[0] in {"cd", "export", "set", "source"}:
        return None
    if "=" in command[0] and "/" not in command[0]:
        return None
    return command


def _commands_from_log(log: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for raw_line in log.splitlines():
        line = _ANSI_ESCAPE.sub("", raw_line).strip()
        match = _RUN_LINE.match(line)
        if match is None:
            continue
        command = _parse_command(match.group(1))
        if command is not None and command not in commands:
            commands.append(command)
    return commands


def _step_from_mapping(value: Mapping[Any, Any]) -> FailedStep | None:
    log_key = next((key for key in _LOG_KEYS if key in value), None)
    if log_key is None:
        return None
    name = next(
        (_optional_label(value[key]) for key in _NAME_KEYS if key in value),
        None,
    )
    explicit_command = next(
        (_parse_command(value[key]) for key in _COMMAND_KEYS if key in value),
        None,
    )
    return FailedStep(
        name=name,
        log=_normalize_log_fragment(value[log_key]),
        command=explicit_command,
    )


def _normalize_failed_steps(value: Any) -> list[FailedStep]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        steps: list[FailedStep] = []
        for item in value:
            steps.extend(_normalize_failed_steps(item))
        return steps
    if isinstance(value, Mapping):
        direct = _step_from_mapping(value)
        if direct is not None:
            return [direct]
        if "steps" in value:
            return _normalize_failed_steps(value["steps"])
        steps = []
        for raw_name, raw_log in value.items():
            log = _normalize_log_fragment(raw_log)
            if log.strip():
                steps.append(FailedStep(name=_optional_label(raw_name), log=log))
        return steps
    log = _normalize_log_fragment(value)
    return [FailedStep(log=log)] if log.strip() else []


def normalize_ci_failure(row: Mapping[str, Any]) -> NormalizedCIFailure:
    """Normalize heterogeneous workflow/log values without treating YAML as argv."""

    workflow_yaml = _normalize_workflow_yaml(row.get("workflow", ""))
    steps = _normalize_failed_steps(row.get("logs"))
    candidates: list[list[str]] = []
    normalized_steps: list[FailedStep] = []
    for step in steps:
        log_commands = _commands_from_log(step.log)
        command = step.command or (log_commands[0] if len(log_commands) == 1 else None)
        normalized_steps.append(step.model_copy(update={"command": command}))
        for candidate in ([step.command] if step.command is not None else []) + log_commands:
            if candidate not in candidates:
                candidates.append(candidate)
    log_text = "\n\n".join(
        f"[{step.name}]\n{step.log}" if step.name else step.log for step in normalized_steps
    )
    return NormalizedCIFailure(
        workflow_yaml=workflow_yaml,
        log_text=log_text,
        failed_steps=normalized_steps,
        candidate_failed_commands=candidates,
    )


def normalize_error_type(value: Any) -> str:
    """Normalize dataset labels without leaking Python container reprs into reports."""

    if isinstance(value, str):
        return value.strip() or "Unknown"
    if isinstance(value, Mapping):
        for key in ("category", "error_type", "type", "name"):
            if key in value:
                return normalize_error_type(value[key])
        return "Unknown"
    if isinstance(value, (list, tuple)):
        labels: list[str] = []
        for item in value:
            label = normalize_error_type(item)
            if label != "Unknown" and label not in labels:
                labels.append(label)
        return " / ".join(labels) if labels else "Unknown"
    if value is None:
        return "Unknown"
    return "Unknown"


def _normalize_changed_files(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [path for path in value if isinstance(path, str)]
    return []


class CIRepairBenchAdapter:
    """Load a local JSON/JSONL export without exposing evaluator fields."""

    def __init__(self, dataset_path: Path) -> None:
        self._rows = self._load(dataset_path)

    @staticmethod
    def _load(path: Path) -> dict[str, dict[str, Any]]:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text())
            rows = payload if isinstance(payload, list) else payload["rows"]
        indexed: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            owner = _required_text(row["repo_owner"], field="repo_owner")
            name = _required_text(row["repo_name"], field="repo_name")
            raw_task_id = row.get("task_id") or row.get("id")
            task_id = (
                _identifier_text(raw_task_id, field="task_id")
                if raw_task_id is not None
                else f"{owner}-{name}-{index}"
            )
            indexed[task_id] = dict(row) | {"task_id": task_id}
        return indexed

    def list_tasks(self) -> list[TaskMeta]:
        return [
            TaskMeta(
                task_id=task_id,
                repo=(
                    f"{_required_text(row['repo_owner'], field='repo_owner')}/"
                    f"{_required_text(row['repo_name'], field='repo_name')}"
                ),
                task_family=normalize_error_type(row.get("error_type")),
            )
            for task_id, row in self._rows.items()
        ]

    def prepare_task(self, task_id: str) -> PreparedTask:
        row = self._rows[task_id]
        workflow_path = _required_text(row["workflow_path"], field="workflow_path")
        workflow_name = (
            _required_text(row["workflow_name"], field="workflow_name")
            if row.get("workflow_name")
            else Path(workflow_path).stem
        )
        return PreparedTask(
            agent_view=AgentTaskView(
                task_id=task_id,
                repo_owner=_required_text(row["repo_owner"], field="repo_owner"),
                repo_name=_required_text(row["repo_name"], field="repo_name"),
                workflow_name=workflow_name,
                workflow_path=workflow_path,
                sha_fail=_required_text(row["sha_fail"], field="sha_fail"),
                ci_failure=normalize_ci_failure(row),
            ),
            workspace_path=(
                _required_text(row["workspace_path"], field="workspace_path")
                if row.get("workspace_path")
                else None
            ),
        )

    def ground_truth(self, task_id: str) -> GroundTruth:
        row = self._rows[task_id]
        return GroundTruth(
            task_id=task_id,
            sha_success=_required_text(row["sha_success"], field="sha_success"),
            diff=_normalize_log_fragment(row.get("diff")),
            changed_files=_normalize_changed_files(row.get("changed_files")),
            error_type=normalize_error_type(row.get("error_type")),
        )


class FixtureAdapter(CIRepairBenchAdapter):
    """A CI-Repair-Bench-compatible local fixture adapter."""


class SyntheticAdapter:
    def __init__(
        self, tasks: list[tuple[AgentTaskView, GroundTruth]], workspace_path: str | None = None
    ) -> None:
        self._tasks = {view.task_id: (view, truth) for view, truth in tasks}
        self._workspace_path = workspace_path

    def list_tasks(self) -> list[TaskMeta]:
        return [
            TaskMeta(task_id=view.task_id, repo=f"{view.repo_owner}/{view.repo_name}")
            for view, _ in self._tasks.values()
        ]

    def prepare_task(self, task_id: str) -> PreparedTask:
        return PreparedTask(agent_view=self._tasks[task_id][0], workspace_path=self._workspace_path)

    def ground_truth(self, task_id: str) -> GroundTruth:
        return self._tasks[task_id][1]
