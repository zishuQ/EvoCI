import os
from pathlib import Path

import pytest

from evoci.cli import _benchmark_runtime_config, _live_resources
from evoci.config import EvoCIConfig


def test_config_reads_only_evoci_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVO_MODEL_NAME", "test-model")
    monkeypatch.setenv("CODEX_HOME", "/should/not/be/read")
    config = EvoCIConfig.from_env(cwd=tmp_path)
    assert config.model_name == "test-model"
    assert config.state_dir == tmp_path / ".evoci/state"
    assert "codex" not in str(config.state_dir).lower()


def test_config_creates_runtime_directories(tmp_path: Path) -> None:
    config = EvoCIConfig.from_env(cwd=tmp_path)
    config.ensure_directories()
    assert config.state_dir.is_dir()
    assert config.workspace_dir.is_dir()
    assert config.repo_cache_dir.is_dir()
    assert config.capability_dir.is_dir()
    assert not config.runtime_dir.exists()


def test_capability_and_runtime_directories_are_explicit_and_distinct(
    tmp_path: Path,
) -> None:
    config = EvoCIConfig.from_env(cwd=tmp_path)
    assert config.capability_dir == tmp_path / ".evoci/skills"
    assert config.runtime_dir == tmp_path / ".evoci/runtime"
    assert config.capability_dir != config.state_dir.parent / "state" / "skills"


def test_benchmark_runtime_state_is_shared_only_in_continual_mode(
    tmp_path: Path,
) -> None:
    base = EvoCIConfig.from_env(cwd=tmp_path)
    first = _benchmark_runtime_config(
        base,
        output_dir=tmp_path / "results",
        variant="evo",
        session_id="session",
        task_index=1,
    )
    second = _benchmark_runtime_config(
        base,
        output_dir=tmp_path / "results",
        variant="evo",
        session_id="session",
        task_index=2,
    )
    shared = _benchmark_runtime_config(
        base,
        output_dir=tmp_path / "results",
        variant="evo",
        session_id="session",
        task_index=None,
    )

    assert first.state_dir != second.state_dir
    assert first.capability_dir != second.capability_dir
    assert shared.capability_dir == shared.state_dir / "skills"
    assert first.runtime_dir != first.workspace_dir


def test_config_automatically_loads_project_dotenv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EVO_MODEL_NAME", raising=False)
    (tmp_path / ".env").write_text("EVO_MODEL_NAME=dotenv-model\n", encoding="utf-8")
    try:
        assert EvoCIConfig.from_env(cwd=tmp_path).model_name == "dotenv-model"
    finally:
        os.environ.pop("EVO_MODEL_NAME", None)


def test_system_environment_overrides_project_dotenv(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("EVO_MODEL_NAME=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("EVO_MODEL_NAME", "system-model")
    assert EvoCIConfig.from_env(cwd=tmp_path).model_name == "system-model"


def test_missing_dotenv_uses_existing_defaults(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EVO_MODEL_NAME", raising=False)
    assert EvoCIConfig.from_env(cwd=tmp_path).model_name is None


def test_config_reads_role_reasoning_effort(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVO_SUPERVISOR_REASONING_EFFORT", "xhigh")
    monkeypatch.setenv("EVO_WORKER_REASONING_EFFORT", "medium")
    config = EvoCIConfig.from_env(cwd=tmp_path)
    assert config.supervisor_reasoning_effort == "xhigh"
    assert config.worker_reasoning_effort == "medium"


def test_config_rejects_invalid_reasoning_effort(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVO_WORKER_REASONING_EFFORT", "extreme")
    with pytest.raises(ValueError, match="EVO_WORKER_REASONING_EFFORT"):
        EvoCIConfig.from_env(cwd=tmp_path)


def test_removed_tier_settings_raise(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVO_MODEL_FAST", "fast")
    with pytest.raises(ValueError, match="Removed EvoCI settings"):
        EvoCIConfig.from_env(cwd=tmp_path)


def test_removed_thinking_switch_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVO_ENABLE_THINKING", "1")
    with pytest.raises(ValueError, match="EVO_ENABLE_THINKING"):
        EvoCIConfig.from_env(cwd=tmp_path)


def test_removed_parallel_and_token_settings_raise(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVO_MAX_RUN_TOKENS", "2000000")
    with pytest.raises(ValueError, match="EVO_MAX_RUN_TOKENS"):
        EvoCIConfig.from_env(cwd=tmp_path)
    monkeypatch.delenv("EVO_MAX_RUN_TOKENS")
    monkeypatch.setenv("EVO_MAX_PARALLEL_WORKERS", "2")
    with pytest.raises(ValueError, match="EVO_MAX_PARALLEL_WORKERS"):
        EvoCIConfig.from_env(cwd=tmp_path)


def test_default_run_budgets_cover_the_default_agent_limits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EVO_MAX_RUN_MODEL_CALLS", raising=False)
    monkeypatch.delenv("EVO_MAX_RUN_TOOL_CALLS", raising=False)
    config = EvoCIConfig.from_env(cwd=tmp_path)
    assert config.max_run_model_calls == 256
    assert config.max_run_tool_calls == 320


def test_role_runtimes_share_model_name(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVO_MODEL_NAME", "main")
    config = EvoCIConfig.from_env(cwd=tmp_path)
    assert config.supervisor_runtime().model_name == "main"
    assert config.worker_runtime().model_name == "main"
    assert config.supervisor_runtime().reasoning_effort == "xhigh"
    assert config.worker_runtime().reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_live_resources_use_independent_role_gateways(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("EVO_MODEL_API_KEY", "test-key")
    monkeypatch.setenv("EVO_MODEL_NAME", "main")
    resources = await _live_resources(EvoCIConfig.from_env(cwd=tmp_path))
    try:
        agents = resources.runtime.agents
        assert agents.supervisor.loop.gateway.model_name == "main"
        assert agents.worker.loop.gateway.model_name == "main"
        assert agents.supervisor.loop.gateway.enable_thinking is True
        assert agents.worker.loop.gateway.enable_thinking is True
        assert agents.supervisor.loop.gateway.reasoning_effort == "xhigh"
        assert agents.worker.loop.gateway.reasoning_effort == "medium"
        assert resources.runtime.memory_consolidator.gateway.reasoning_effort == "medium"
        assert resources.runtime.skill_miner.gateway.reasoning_effort == "medium"
        assert resources.supervisor_gateway is not resources.worker_gateway
    finally:
        await resources.close()
