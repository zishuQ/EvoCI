"""Environment-backed configuration isolated from personal agent settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class EvoCIConfig(BaseModel):
    """Runtime configuration read exclusively from EVO_* variables."""

    model_config = ConfigDict(frozen=True)

    model_base_url: str = "https://api.openai.com/v1"
    model_api_key: SecretStr | None = None
    model_name: str | None = None
    fast_model_name: str | None = None
    strong_model_name: str | None = None
    aux_model_name: str | None = None
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
    max_initial_workers: int = Field(default=4, ge=1, le=4)
    max_investigation_tasks: int = Field(default=8, ge=1, le=8)
    max_investigation_rounds: int = Field(default=3, ge=1, le=3)
    max_repair_attempts: int = Field(default=3, ge=1, le=5)
    max_leaf_iterations: int = Field(default=8, ge=1, le=20)
    max_leaf_tool_calls: int = Field(default=16, ge=1, le=50)
    max_run_model_calls: int = Field(default=256, ge=1, le=1_000)
    max_run_tool_calls: int = Field(default=256, ge=1, le=2_000)
    trial_min_uses: int = Field(default=1, ge=1, le=100)
    trial_min_successes: int = Field(default=1, ge=1, le=100)
    trial_min_success_rate: float = Field(default=0.5, ge=0, le=1)
    trial_max_failures: int = Field(default=3, ge=1, le=100)
    capability_retrieval_top_k: int = Field(default=2, ge=1, le=20)
    trial_retrieval_slots: int = Field(default=1, ge=0, le=20)
    trial_max_exposures_without_use: int = Field(default=5, ge=1, le=1_000)
    enable_thinking: bool = False

    @model_validator(mode="after")
    def validate_task_limits(self) -> EvoCIConfig:
        if self.max_initial_workers > self.max_investigation_tasks:
            raise ValueError("max_initial_workers cannot exceed max_investigation_tasks")
        if self.trial_retrieval_slots > self.capability_retrieval_top_k:
            raise ValueError("trial_retrieval_slots cannot exceed capability_retrieval_top_k")
        return self

    @classmethod
    def from_env(cls, *, cwd: Path | None = None) -> EvoCIConfig:
        root = (cwd or Path.cwd()).resolve()
        env_file = root / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)

        def path_value(name: str, default: str) -> Path:
            raw = Path(os.environ.get(name, default))
            return raw if raw.is_absolute() else root / raw

        api_key = os.environ.get("EVO_MODEL_API_KEY")
        return cls(
            model_base_url=os.environ.get("EVO_MODEL_BASE_URL", "https://api.openai.com/v1"),
            model_api_key=SecretStr(api_key) if api_key else None,
            model_name=os.environ.get("EVO_MODEL_NAME") or None,
            fast_model_name=os.environ.get("EVO_MODEL_FAST") or None,
            strong_model_name=os.environ.get("EVO_MODEL_STRONG") or None,
            aux_model_name=(
                os.environ.get("EVO_MODEL_AUX") or os.environ.get("EVO_AUX_MODEL_NAME") or None
            ),
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
            max_repair_attempts=int(os.environ.get("EVO_MAX_REPAIR_ATTEMPTS", "3")),
            max_leaf_iterations=int(os.environ.get("EVO_MAX_LEAF_ITERATIONS", "8")),
            max_leaf_tool_calls=int(os.environ.get("EVO_MAX_LEAF_TOOL_CALLS", "16")),
            max_run_model_calls=int(os.environ.get("EVO_MAX_RUN_MODEL_CALLS", "256")),
            max_run_tool_calls=int(os.environ.get("EVO_MAX_RUN_TOOL_CALLS", "256")),
            trial_min_uses=int(os.environ.get("EVO_TRIAL_MIN_USES", "1")),
            trial_min_successes=int(os.environ.get("EVO_TRIAL_MIN_SUCCESSES", "1")),
            trial_min_success_rate=float(os.environ.get("EVO_TRIAL_MIN_SUCCESS_RATE", "0.5")),
            trial_max_failures=int(os.environ.get("EVO_TRIAL_MAX_FAILURES", "3")),
            capability_retrieval_top_k=int(os.environ.get("EVO_CAPABILITY_RETRIEVAL_TOP_K", "2")),
            trial_retrieval_slots=int(os.environ.get("EVO_TRIAL_RETRIEVAL_SLOTS", "1")),
            trial_max_exposures_without_use=int(
                os.environ.get("EVO_TRIAL_MAX_EXPOSURES_WITHOUT_USE", "5")
            ),
            enable_thinking=os.environ.get("EVO_ENABLE_THINKING", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.state_dir,
            self.workspace_dir,
            self.repo_cache_dir,
            self.capability_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
