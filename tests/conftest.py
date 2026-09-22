import pytest

_REMOVED_EVO_SETTINGS = (
    "EVO_MODEL_FAST",
    "EVO_MODEL_STRONG",
    "EVO_MODEL_AUX",
    "EVO_AUX_MODEL_NAME",
    "EVO_ENABLE_THINKING",
    "EVO_REASONING_EFFORT",
    "EVO_MAX_PARALLEL_WORKERS",
    "EVO_MAX_TASKS_PER_BATCH",
    "EVO_MAX_RUN_TOKENS",
    "EVO_MAX_TASK_OUTPUT_TOKENS",
)


@pytest.fixture(autouse=True)
def clear_removed_evoci_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _REMOVED_EVO_SETTINGS:
        monkeypatch.delenv(name, raising=False)
