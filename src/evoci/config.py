"""Environment-backed configuration isolated from personal agent settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr

ReasoningEffort = Literal["low", "medium", "high", "xhigh"]

_REMOVED_SETTINGS = {
    "EVO_MODEL_FAST": (
        "FAST/STRONG/AUX routing has been removed. Use EVO_MODEL_NAME with "
        "EVO_SUPERVISOR_* and EVO_WORKER_* role settings."
    ),
    "EVO_MODEL_STRONG": (
        "FAST/STRONG/AUX routing has been removed. Use EVO_MODEL_NAME with "
        "EVO_SUPERVISOR_* and EVO_WORKER_* role settings."
    ),
    "EVO_MODEL_AUX": (
        "FAST/STRONG/AUX routing has been removed. Learning reuses the Worker "
        "gateway. Use EVO_MODEL_NAME and EVO_WORKER_*."
    ),
    "EVO_AUX_MODEL_NAME": (
        "FAST/STRONG/AUX routing has been removed. Learning reuses the Worker "
        "gateway. Use EVO_MODEL_NAME and EVO_WORKER_*."
    ),
    "EVO_ENABLE_THINKING": (
        "EVO_ENABLE_THINKING is removed. Set EVO_SUPERVISOR_ENABLE_THINKING and "
        "EVO_WORKER_ENABLE_THINKING separately."
    ),
    "EVO_REASONING_EFFORT": (
        "EVO_REASONING_EFFORT is removed. Set EVO_SUPERVISOR_REASONING_EFFORT "
        "and EVO_WORKER_REASONING_EFFORT separately."
    ),
    "EVO_MAX_PARALLEL_WORKERS": (
        "Multi-worker dispatch has been removed. EvoCI runs one Worker at a time."
    ),
    "EVO_MAX_TASKS_PER_BATCH": (
        "Multi-task batches have been removed. Supervisor dispatch must contain "
        "exactly one task."
    ),
    "EVO_MAX_RUN_TOKENS": (
        "Token hard limits have been removed. Usage is recorded but does not stop "
        "a run. Unset EVO_MAX_RUN_TOKENS."
    ),
    "EVO_MAX_TASK_OUTPUT_TOKENS": (
        "Per-request output token caps are not sent by default. Unset "
        "EVO_MAX_TASK_OUTPUT_TOKENS."
    ),
}


def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_effort(name: str, default: str) -> ReasoningEffort:
    raw = os.environ.get(name, default).strip().lower()
    if raw not in {"low", "medium", "high", "xhigh"}:
        raise ValueError(f"{name} must be one of low, medium, high, xhigh")
    return cast(ReasoningEffort, raw)


def _reject_removed_settings() -> None:
    conflicts = [
        f"{name}: {_REMOVED_SETTINGS[name]}"
        for name in _REMOVED_SETTINGS
        if os.environ.get(name, "").strip()
    ]
    if conflicts:
        raise ValueError("Removed EvoCI settings must be migrated:\n" + "\n".join(conflicts))


class RoleRuntimeConfig(BaseModel):
    """Immutable per-role model request settings."""

    model_config = ConfigDict(frozen=True)

    role: Literal["supervisor", "worker"]
    model_name: str | None = None
    enable_thinking: bool = True
    reasoning_effort: ReasoningEffort | None = None


class EvoCIConfig(BaseModel):
    """Runtime configuration read exclusively from EVO_* variables."""

    model_config = ConfigDict(frozen=True)

    model_base_url: str = "https://api.openai.com/v1"
    model_api_key: SecretStr | None = None
    model_name: str | None = None
    state_dir: Path = Path(".evoci/state")
    workspace_dir: Path = Path(".evoci/worktrees")
    repo_cache_dir: Path = Path(".evoci/repos")
    capability_dir: Path = Path(".evoci/skills")
    runtime_dir: Path = Path(".evoci/runtime")
    runtime_dir_explicit: bool = False
    patch_hash_strict: bool = True
    model_timeout_seconds: float = Field(default=120.0, gt=0)
    command_timeout_seconds: float = Field(default=120.0, gt=0)
    output_limit_chars: int = Field(default=32_000, ge=1_000)
    memory_context_limit_chars: int = Field(default=6_000, ge=1_000)
    max_supervisor_batches: int = Field(default=3, ge=1, le=8)
    max_leaf_iterations: int = Field(default=15, ge=1, le=20)
    max_leaf_tool_calls: int = Field(default=30, ge=1, le=50)
    max_run_model_calls: int = Field(default=256, ge=1, le=1_000)
    max_run_tool_calls: int = Field(default=320, ge=1, le=2_000)
    capability_retrieval_top_k: int = Field(default=2, ge=1, le=20)
    skill_catalog_limit_chars: int = Field(default=8_000, ge=500, le=32_000)
    supervisor_enable_thinking: bool = True
    supervisor_reasoning_effort: ReasoningEffort | None = "xhigh"
    worker_enable_thinking: bool = True
    worker_reasoning_effort: ReasoningEffort | None = "medium"

    @property
    def max_repair_attempts(self) -> int:
        """Historical alias: one supervisor batch is one repair/replan cycle."""

        return self.max_supervisor_batches

    def supervisor_runtime(self) -> RoleRuntimeConfig:
        return RoleRuntimeConfig(
            role="supervisor",
            model_name=self.model_name,
            enable_thinking=self.supervisor_enable_thinking,
            reasoning_effort=self.supervisor_reasoning_effort,
        )

    def worker_runtime(self) -> RoleRuntimeConfig:
        return RoleRuntimeConfig(
            role="worker",
            model_name=self.model_name,
            enable_thinking=self.worker_enable_thinking,
            reasoning_effort=self.worker_reasoning_effort,
        )

    @classmethod
    def from_env(cls, *, cwd: Path | None = None) -> EvoCIConfig:
        root = (cwd or Path.cwd()).resolve()
        env_file = root / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
        _reject_removed_settings()

        def path_value(name: str, default: str) -> Path:
            raw = Path(os.environ.get(name, default))
            return raw if raw.is_absolute() else root / raw

        api_key = os.environ.get("EVO_MODEL_API_KEY")
        supervisor_batches = os.environ.get("EVO_MAX_SUPERVISOR_BATCHES")
        repair_attempts = os.environ.get("EVO_MAX_REPAIR_ATTEMPTS")
        if supervisor_batches and repair_attempts and supervisor_batches != repair_attempts:
            raise ValueError(
                "EVO_MAX_SUPERVISOR_BATCHES and EVO_MAX_REPAIR_ATTEMPTS disagree; "
                "set only EVO_MAX_SUPERVISOR_BATCHES"
            )
        batches = int(supervisor_batches or repair_attempts or "3")
        return cls(
            model_base_url=os.environ.get("EVO_MODEL_BASE_URL", "https://api.openai.com/v1"),
            model_api_key=SecretStr(api_key) if api_key else None,
            model_name=os.environ.get("EVO_MODEL_NAME") or None,
            state_dir=path_value("EVO_STATE_DIR", ".evoci/state"),
            workspace_dir=path_value("EVO_WORKSPACE_DIR", ".evoci/worktrees"),
            repo_cache_dir=path_value("EVO_REPO_CACHE_DIR", ".evoci/repos"),
            capability_dir=path_value("EVO_CAPABILITY_DIR", ".evoci/skills"),
            runtime_dir=path_value("EVO_RUNTIME_DIR", ".evoci/runtime"),
            runtime_dir_explicit=bool(os.environ.get("EVO_RUNTIME_DIR")),
            patch_hash_strict=os.environ.get("EVO_PATCH_HASH_STRICT", "1").strip().lower()
            not in {"0", "false", "no", "off"},
            model_timeout_seconds=float(os.environ.get("EVO_MODEL_TIMEOUT_SECONDS", "120")),
            command_timeout_seconds=float(os.environ.get("EVO_COMMAND_TIMEOUT_SECONDS", "120")),
            max_supervisor_batches=batches,
            max_leaf_iterations=int(os.environ.get("EVO_MAX_LEAF_ITERATIONS", "15")),
            max_leaf_tool_calls=int(os.environ.get("EVO_MAX_LEAF_TOOL_CALLS", "30")),
            max_run_model_calls=int(os.environ.get("EVO_MAX_RUN_MODEL_CALLS", "256")),
            max_run_tool_calls=int(os.environ.get("EVO_MAX_RUN_TOOL_CALLS", "320")),
            capability_retrieval_top_k=int(os.environ.get("EVO_CAPABILITY_RETRIEVAL_TOP_K", "2")),
            skill_catalog_limit_chars=int(
                os.environ.get("EVO_SKILL_CATALOG_LIMIT_CHARS", "8000")
            ),
            supervisor_enable_thinking=_env_bool("EVO_SUPERVISOR_ENABLE_THINKING", "1"),
            supervisor_reasoning_effort=_env_effort("EVO_SUPERVISOR_REASONING_EFFORT", "xhigh"),
            worker_enable_thinking=_env_bool("EVO_WORKER_ENABLE_THINKING", "1"),
            worker_reasoning_effort=_env_effort("EVO_WORKER_REASONING_EFFORT", "medium"),
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.state_dir,
            self.workspace_dir,
            self.repo_cache_dir,
            self.capability_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
