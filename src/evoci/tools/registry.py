"""Permission-aware tool registry with call telemetry hooks."""

from __future__ import annotations

import hashlib
import inspect
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from evoci.model.gateway import ToolDefinition
from evoci.tools.filesystem import FileTools
from evoci.tools.git import GitTools
from evoci.tools.isolation import copy_workspace_with_independent_git
from evoci.tools.policy import PolicyViolation, WorkerCapabilities
from evoci.tools.shell import CommandRunner, run_cancellable

if TYPE_CHECKING:
    from evoci.capability.registry import CapabilityRegistry


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


class ListFilesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = "."
    limit: int = Field(default=500, ge=1, le=500)


class SearchCodeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pattern: str
    path: str = "."
    limit: int = Field(default=100, ge=1, le=100)


class GitLogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=20, ge=1, le=100)


class GitShowArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str = "HEAD"


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunCommandArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    argv: list[str] = Field(min_length=1, max_length=100)
    cwd: str = "."
    network: bool = False


class ApplyPatchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    files: dict[str, str]


class ReplaceTextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    old_text: str
    new_text: str
    expected_replacements: int = 1


class CreateFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    content: str


class DeleteFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


class RunSkillScriptArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    version: int = Field(ge=1)
    script_name: str
    args: list[str] = Field(default_factory=list, max_length=100)


class ReadSkillResourceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    version: int = Field(ge=1)
    path: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8_000, ge=1, le=32_000)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    capability: str
    function: Callable[..., Any]
    description: str
    args_model: type[BaseModel] | None


_WRITE_PATH_TOOLS = frozenset(
    {"apply_patch", "replace_text", "create_file", "delete_file", "run_skill_script"}
)
_WRITE_PATH_KEYS = ("created_files", "modified_files", "deleted_files")


class ToolRegistry:
    def __init__(
        self,
        capabilities: WorkerCapabilities,
        *,
        tool_allowlist: set[str] | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.tool_allowlist = tool_allowlist
        self._tools: dict[str, ToolSpec] = {}
        self._temporary_workspaces: list[tempfile.TemporaryDirectory[str]] = []
        self.execution_workspace: Path | None = None
        self._changed_paths: set[str] = set()

    def changed_paths(self) -> list[str]:
        return sorted(self._changed_paths)

    def _record_changed_paths(self, name: str, result: Any) -> None:
        if name not in _WRITE_PATH_TOOLS:
            return
        if isinstance(result, BaseModel):
            payload = result.model_dump()
        elif isinstance(result, dict):
            payload = result
        else:
            return
        if name == "run_skill_script":
            if not self.capabilities.write_files:
                return
            if payload.get("exit_code") not in {0, None} or payload.get("timed_out"):
                return
            if not any(key in payload for key in _WRITE_PATH_KEYS):
                return
        for key in _WRITE_PATH_KEYS:
            paths = payload.get(key) or []
            if not isinstance(paths, list):
                continue
            for path in paths:
                if isinstance(path, str) and path:
                    self._changed_paths.add(path)

    def own_temporary_workspace(
        self, temporary: tempfile.TemporaryDirectory[str], workspace: Path
    ) -> None:
        self._temporary_workspaces.append(temporary)
        self.execution_workspace = workspace

    def close(self) -> None:
        while self._temporary_workspaces:
            self._temporary_workspaces.pop().cleanup()

    def register(
        self,
        name: str,
        capability: str,
        function: Callable[..., Any],
        *,
        description: str = "",
        args_model: type[BaseModel] | None = None,
    ) -> None:
        self._tools[name] = ToolSpec(
            capability=capability,
            function=function,
            description=description or name.replace("_", " "),
            args_model=args_model,
        )

    def _resolve(self, name: str, kwargs: dict[str, Any]) -> tuple[ToolSpec, dict[str, Any]]:
        try:
            spec = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc
        if self.tool_allowlist is not None and name not in self.tool_allowlist:
            raise PolicyViolation(f"tool {name} is not available to this agent role")
        if not bool(getattr(self.capabilities, spec.capability, False)):
            raise PolicyViolation(f"tool {name} requires {spec.capability}")
        if spec.args_model is None:
            return spec, kwargs
        validated = spec.args_model.model_validate(kwargs)
        return spec, validated.model_dump()

    def invoke(self, name: str, **kwargs: Any) -> Any:
        spec, validated = self._resolve(name, kwargs)
        result = spec.function(**validated)
        if inspect.isawaitable(result):
            raise RuntimeError(f"tool {name} is asynchronous; use ainvoke")
        self._record_changed_paths(name, result)
        return result

    async def ainvoke(self, name: str, **kwargs: Any) -> Any:
        spec, validated = self._resolve(name, kwargs)
        if name == "run_skill_script":
            result = await run_cancellable(spec.function, **validated)
        else:
            result = spec.function(**validated)
            if inspect.isawaitable(result):
                result = await result
        self._record_changed_paths(name, result)
        return result

    def available(self) -> list[str]:
        return sorted(
            name
            for name, spec in self._tools.items()
            if bool(getattr(self.capabilities, spec.capability, False))
            and (self.tool_allowlist is None or name in self.tool_allowlist)
        )

    def definitions(self) -> list[ToolDefinition]:
        definitions: list[ToolDefinition] = []
        for name in self.available():
            spec = self._tools[name]
            parameters = (
                spec.args_model.model_json_schema()
                if spec.args_model is not None
                else {"type": "object", "properties": {}, "additionalProperties": True}
            )
            definitions.append(
                ToolDefinition(
                    name=name,
                    description=spec.description,
                    parameters=parameters,
                )
            )
        return definitions


def create_worker_registry(
    capabilities: WorkerCapabilities,
    workspace: Path,
    *,
    timeout: float = 120.0,
    max_chars: int = 32_000,
    capability_registry: CapabilityRegistry | None = None,
    allowed_skill_refs: set[tuple[str, int]] | None = None,
    tool_allowlist: set[str] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(capabilities, tool_allowlist=tool_allowlist)
    file_tools = FileTools(workspace, writable=capabilities.write_files, max_chars=max_chars)
    inspection_runner = CommandRunner(workspace, timeout=timeout, max_chars=max_chars)
    git = GitTools(inspection_runner)
    disposable: Path | None = None
    workspace_revision = 0

    def invalidate_snapshots() -> None:
        nonlocal disposable, workspace_revision
        workspace_revision += 1
        while registry._temporary_workspaces:
            registry._temporary_workspaces.pop().cleanup()
        disposable = None
        registry.execution_workspace = None

    def disposable_workspace() -> Path:
        nonlocal disposable
        if disposable is None:
            source = workspace.resolve()
            temporary = tempfile.TemporaryDirectory(prefix=".evoci-exec-", dir=source.parent)
            disposable = Path(temporary.name) / "workspace"
            copy_workspace_with_independent_git(source, disposable)
            registry.own_temporary_workspace(temporary, disposable)
        return disposable

    command_workspace = workspace if capabilities.write_files else disposable_workspace()
    runner = CommandRunner(command_workspace, timeout=timeout, max_chars=max_chars)

    async def run_bounded_command(argv: list[str], cwd: str = ".", network: bool = False) -> Any:
        if network and not capabilities.network:
            raise PolicyViolation("worker does not have network permission")
        result = await runner.run(argv, cwd=cwd, network=network)
        if capabilities.write_files:
            invalidate_snapshots()
        return result

    registry.register(
        "read_file",
        "read_files",
        file_tools.read_file,
        description="Read a UTF-8 text file inside the workspace.",
        args_model=ReadFileArgs,
    )
    registry.register(
        "list_files",
        "read_files",
        file_tools.list_files,
        description="List files below a workspace-relative path.",
        args_model=ListFilesArgs,
    )
    registry.register(
        "search_code",
        "search_code",
        file_tools.search_code,
        description="Search workspace text files with a regular expression.",
        args_model=SearchCodeArgs,
    )
    registry.register(
        "git_status",
        "git_read",
        git.status,
        description="Show concise Git status.",
        args_model=NoArgs,
    )
    registry.register(
        "git_log",
        "git_read",
        git.log,
        description="Show recent Git commits.",
        args_model=GitLogArgs,
    )
    registry.register(
        "git_show",
        "git_read",
        git.show,
        description="Show a Git revision summary.",
        args_model=GitShowArgs,
    )
    registry.register(
        "git_diff",
        "git_read",
        git.diff,
        description="Show the current Git diff.",
        args_model=NoArgs,
    )
    registry.register(
        "run_command",
        "execute_tests",
        run_bounded_command,
        description="Run one allowlisted argv command without a shell.",
        args_model=RunCommandArgs,
    )
    registry.register(
        "run_test",
        "execute_tests",
        run_bounded_command,
        description="Run one allowlisted test argv command without a shell.",
        args_model=RunCommandArgs,
    )
    def apply_patch(files: dict[str, str]) -> Any:
        result = file_tools.apply_patch(files)
        invalidate_snapshots()
        return result

    def replace_text(
        path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> Any:
        result = file_tools.replace_text(
            path, old_text, new_text, expected_replacements=expected_replacements
        )
        invalidate_snapshots()
        return result

    def create_file(path: str, content: str) -> Any:
        result = file_tools.create_file(path, content)
        invalidate_snapshots()
        return result

    def delete_file(path: str) -> Any:
        result = file_tools.delete_file(path)
        invalidate_snapshots()
        return result

    registry.register(
        "apply_patch",
        "write_files",
        apply_patch,
        description=(
            "Apply full-file replacements for new or small files. Prefer replace_text "
            "for modifying existing large files."
        ),
        args_model=ApplyPatchArgs,
    )
    registry.register(
        "replace_text",
        "write_files",
        replace_text,
        description=(
            "Replace exact text in an existing UTF-8 workspace file. Fails unless "
            "old_text occurs expected_replacements times."
        ),
        args_model=ReplaceTextArgs,
    )
    registry.register(
        "create_file",
        "write_files",
        create_file,
        description="Create a new UTF-8 file. Fails if the path already exists.",
        args_model=CreateFileArgs,
    )
    registry.register(
        "delete_file",
        "write_files",
        delete_file,
        description="Delete an existing regular file. Fails for directories and symbolic links.",
        args_model=DeleteFileArgs,
    )
    if capability_registry is not None:
        from evoci.capability.execution import run_skill_script

        def execute_skill(skill_id: str, version: int, script_name: str, args: list[str]) -> Any:
            if allowed_skill_refs is not None and (skill_id, version) not in allowed_skill_refs:
                raise PolicyViolation("skill was not selected for this run")
            record = capability_registry.get(skill_id, version)
            if record is None:
                raise KeyError(f"unknown skill: {skill_id} v{version}")
            if record.manifest.permissions.write_workspace and not capabilities.write_files:
                raise PolicyViolation(
                    "skill requests workspace writes that this worker role does not permit"
                )
            may_write = capabilities.write_files and record.manifest.permissions.write_workspace
            script_workspace = workspace if may_write else disposable_workspace()
            try:
                return run_skill_script(
                    capability_registry,
                    skill_id=skill_id,
                    version=version,
                    script_name=script_name,
                    args=args,
                    workspace=script_workspace,
                    timeout=timeout,
                    max_chars=max_chars,
                    observed_revision=workspace_revision,
                )
            finally:
                if may_write:
                    invalidate_snapshots()

        def read_skill_resource(
            skill_id: str,
            version: int,
            path: str,
            offset: int = 0,
            limit: int = 8_000,
        ) -> dict[str, Any]:
            if allowed_skill_refs is not None and (skill_id, version) not in allowed_skill_refs:
                raise PolicyViolation("skill was not selected for this run")
            record = capability_registry.get(skill_id, version)
            if record is None:
                raise KeyError(f"unknown skill: {skill_id} v{version}")
            if record.manifest.status not in {"trial", "active"}:
                raise PolicyViolation("only trial or active skill resources may be read")
            package = Path(record.package_path).resolve()
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise PolicyViolation(f"unsafe skill resource path: {path}")
            declared = {file.path: file.sha256 for file in record.manifest.files}
            if path not in declared:
                raise PolicyViolation("resource is not declared in the skill manifest")
            target = (package / relative).resolve()
            if package not in target.parents or not target.is_file():
                raise PolicyViolation("skill resource escaped the package")
            if target.is_symlink():
                raise PolicyViolation("skill resource must not be a symlink")
            payload = target.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if digest != declared[path]:
                raise PolicyViolation(f"skill resource hash mismatch: {path}")
            text = payload.decode("utf-8", errors="replace")
            fragment = text[offset : offset + limit]
            return {
                "skill_id": skill_id,
                "version": version,
                "path": path,
                "offset": offset,
                "limit": limit,
                "length": len(text),
                "truncated": offset + limit < len(text) or offset > 0,
                "sha256": digest,
                "content": fragment,
            }

        registry.register(
            "run_skill_script",
            "execute_tests",
            execute_skill,
            description=(
                "Run a declared script from a retrieved trial or active capability package."
            ),
            args_model=RunSkillScriptArgs,
        )
        registry.register(
            "read_skill_resource",
            "read_files",
            read_skill_resource,
            description=(
                "Read a declared reference, template, or other file from a selected skill package."
            ),
            args_model=ReadSkillResourceArgs,
        )
    return registry
