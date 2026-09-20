from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evoci.agents.model_agents import StagedFixerPlan
from evoci.tools.filesystem import FileTools
from evoci.tools.isolation import copy_workspace_with_independent_git
from evoci.tools.patch import collect_staged_edits
from evoci.tools.policy import FIXER_CAPABILITIES, PolicyViolation
from evoci.tools.registry import create_worker_registry


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_replace_text_replaces_exact_match(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\nVALUE = 2\n", encoding="utf-8")
    result = FileTools(tmp_path, writable=True).replace_text(
        "app.py", "VALUE = 1", "VALUE = 9"
    )
    assert result.modified_files == ["app.py"]
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "VALUE = 9\nVALUE = 2\n"


def test_replace_text_rejects_too_few_matches(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(PolicyViolation, match="found 0"):
        FileTools(tmp_path, writable=True).replace_text("app.py", "missing", "x")
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_replace_text_rejects_too_many_matches(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\nVALUE = 1\n", encoding="utf-8")
    with pytest.raises(PolicyViolation, match="found 2"):
        FileTools(tmp_path, writable=True).replace_text("app.py", "VALUE = 1", "VALUE = 9")
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "VALUE = 1\nVALUE = 1\n"


def test_replace_text_rejects_empty_old_text(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(PolicyViolation, match="old_text must not be empty"):
        FileTools(tmp_path, writable=True).replace_text("app.py", "", "VALUE = 9")


def test_create_file_writes_new_path(tmp_path: Path) -> None:
    result = FileTools(tmp_path, writable=True).create_file("created.py", "print('ok')\n")
    assert result.created_files == ["created.py"]
    assert (tmp_path / "created.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_create_file_rejects_overwrite(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("keep\n", encoding="utf-8")
    with pytest.raises(PolicyViolation, match="already exists"):
        FileTools(tmp_path, writable=True).create_file("app.py", "new\n")
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "keep\n"


def test_delete_file_removes_regular_file(tmp_path: Path) -> None:
    (tmp_path / "gone.py").write_text("delete me\n", encoding="utf-8")
    result = FileTools(tmp_path, writable=True).delete_file("gone.py")
    assert result.deleted_files == ["gone.py"]
    assert not (tmp_path / "gone.py").exists()


def test_delete_file_rejects_directory_and_symlink(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "real.py").write_text("keep\n", encoding="utf-8")
    (tmp_path / "link.py").symlink_to(tmp_path / "real.py")
    tools = FileTools(tmp_path, writable=True)
    with pytest.raises(PolicyViolation, match="regular file"):
        tools.delete_file("nested")
    with pytest.raises(PolicyViolation, match="symlink"):
        tools.delete_file("link.py")
    assert (tmp_path / "nested").is_dir()
    assert (tmp_path / "real.py").read_text(encoding="utf-8") == "keep\n"
    assert (tmp_path / "link.py").is_symlink()


@pytest.mark.parametrize("tool_name", ["replace_text", "create_file", "delete_file"])
def test_write_tools_reject_path_escape(tmp_path: Path, tool_name: str) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    tools = FileTools(tmp_path, writable=True)
    with pytest.raises(PolicyViolation, match="escapes workspace"):
        if tool_name == "replace_text":
            tools.replace_text("../../etc/passwd", "root", "x")
        elif tool_name == "create_file":
            tools.create_file("../../tmp/evoci-escape", "x")
        else:
            tools.delete_file("../../etc/passwd")


def test_failed_write_tools_do_not_record_changed_paths(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    registry = create_worker_registry(FIXER_CAPABILITIES, tmp_path)
    try:
        with pytest.raises(PolicyViolation):
            registry.invoke("replace_text", path="app.py", old_text="missing", new_text="x")
        with pytest.raises(PolicyViolation):
            registry.invoke("create_file", path="app.py", content="nope")
        with pytest.raises(FileNotFoundError):
            registry.invoke("delete_file", path="missing.py")
        assert registry.changed_paths() == []
    finally:
        registry.close()


def test_successful_writes_are_recorded_on_registry(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "gone.py").write_text("delete\n", encoding="utf-8")
    registry = create_worker_registry(FIXER_CAPABILITIES, tmp_path)
    try:
        registry.invoke("replace_text", path="app.py", old_text="VALUE = 1", new_text="VALUE = 2")
        registry.invoke("create_file", path="created.py", content="new\n")
        registry.invoke("delete_file", path="gone.py")
        registry.invoke("apply_patch", files={"small.py": "tiny\n"})
        assert registry.changed_paths() == ["app.py", "created.py", "gone.py", "small.py"]
    finally:
        registry.close()


@pytest.mark.asyncio
async def test_run_test_cache_files_are_not_recorded(tmp_path: Path) -> None:
    registry = create_worker_registry(FIXER_CAPABILITIES, tmp_path)
    try:
        result = await registry.ainvoke(
            "run_test",
            argv=[
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    "cache = Path('.pytest_cache'); cache.mkdir(); "
                    "(cache / 'v').write_text('cache')"
                ),
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / ".pytest_cache" / "v").is_file()
        assert registry.changed_paths() == []
    finally:
        registry.close()


def test_collect_staged_edits_for_modified_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    original = "VALUE = 1\n"
    (source / "app.py").write_text(original, encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    (staging / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    edits = collect_staged_edits(source, staging, ["app.py"])
    assert len(edits) == 1
    assert edits[0].path == "app.py"
    assert edits[0].content == "VALUE = 2\n"
    assert edits[0].delete is False
    assert edits[0].expected_sha256 == _sha256(original)


def test_collect_staged_edits_for_created_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "keep.py").write_text("keep\n", encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    (staging / "created.py").write_text("new\n", encoding="utf-8")
    edits = collect_staged_edits(source, staging, ["created.py"])
    assert len(edits) == 1
    assert edits[0].path == "created.py"
    assert edits[0].content == "new\n"
    assert edits[0].expected_sha256 is None


def test_collect_staged_edits_for_deleted_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    original = "gone\n"
    (source / "gone.py").write_text(original, encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    (staging / "gone.py").unlink()
    edits = collect_staged_edits(source, staging, ["gone.py"])
    assert len(edits) == 1
    assert edits[0].path == "gone.py"
    assert edits[0].delete is True
    assert edits[0].content is None
    assert edits[0].expected_sha256 == _sha256(original)


def test_collect_staged_edits_expected_sha256_comes_from_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    original = "before\n"
    (source / "app.py").write_text(original, encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    (staging / "app.py").write_text("after\n", encoding="utf-8")
    (source / "app.py").write_text(original, encoding="utf-8")
    edits = collect_staged_edits(source, staging, ["app.py"])
    assert edits[0].expected_sha256 == hashlib.sha256(original.encode("utf-8")).hexdigest()
    assert edits[0].expected_sha256 != hashlib.sha256(b"after\n").hexdigest()


def test_collect_staged_edits_skips_restored_content(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    original = "VALUE = 1\n"
    (source / "app.py").write_text(original, encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    (staging / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (staging / "app.py").write_text(original, encoding="utf-8")
    assert collect_staged_edits(source, staging, ["app.py"]) == []


def test_collect_staged_edits_ignores_unrecorded_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    (staging / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (staging / "secret.py").write_text("leak\n", encoding="utf-8")
    edits = collect_staged_edits(source, staging, ["app.py"])
    assert [edit.path for edit in edits] == ["app.py"]


def test_collect_staged_edits_rejects_more_than_max_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "app.py").write_text("ok\n", encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    paths = [f"file-{index}.txt" for index in range(21)]
    with pytest.raises(PolicyViolation, match="too many files"):
        collect_staged_edits(source, staging, paths)


def test_collect_staged_edits_rejects_total_byte_limit(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "keep.py").write_text("keep\n", encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    (staging / "big.py").write_text("abcdefghij", encoding="utf-8")
    with pytest.raises(PolicyViolation, match="byte limit"):
        collect_staged_edits(source, staging, ["big.py"], max_total_bytes=4)


def test_collect_staged_edits_rejects_escape_and_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    staging = tmp_path / "staging"
    source.mkdir()
    (source / "app.py").write_text("ok\n", encoding="utf-8")
    copy_workspace_with_independent_git(source, staging)
    (staging / "link.py").symlink_to(staging / "app.py")
    with pytest.raises(PolicyViolation, match="escapes workspace"):
        collect_staged_edits(source, staging, ["../secret.py"])
    with pytest.raises(PolicyViolation, match="symlink"):
        collect_staged_edits(source, staging, ["link.py"])


def test_staged_fixer_plan_schema_omits_file_payloads() -> None:
    serialized = json.dumps(StagedFixerPlan.model_json_schema())
    assert '"edits"' not in serialized
    assert '"content"' not in serialized
    assert '"changed_files"' not in serialized
