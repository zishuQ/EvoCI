"""Independent review of checkpoint replay and metadata rollback at 2d51967."""

from __future__ import annotations

import hashlib
import stat
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from evoci.cli import _drive_single
from evoci.domain.models import FileEdit, FixerOutput, PatchProposal
from evoci.graph.builder import build_graph
from evoci.runtime.trajectory import TrajectoryRecorder
from evoci.tools.filesystem import FileTools
from evoci.tools.patch import apply_edit
from tests.integration.test_correctness import PASSING, _failure, _runtime


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


async def _pause_before_apply(graph, state, config: dict[str, object]):
    pending = await graph.ainvoke(state, config, interrupt_before=["apply_candidate"])
    snapshot = await graph.aget_state(config)
    while snapshot.next and snapshot.next[0] != "apply_candidate":
        if pending.get("__interrupt__") or snapshot.next[0] == "approval":
            pending = await graph.ainvoke(
                Command(resume=True), config, interrupt_before=["apply_candidate"]
            )
        else:
            pending = await graph.ainvoke(None, config, interrupt_before=["apply_candidate"])
        snapshot = await graph.aget_state(config)
    assert snapshot.next == ("apply_candidate",)
    return pending


@pytest.mark.asyncio
async def test_checkpoint_replay_accepts_already_applied_first_edit(tmp_path):
    before = "VALUE = 0\n"
    after = "VALUE = 1\n"
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(before)
    edits = [
        FileEdit(path=name, content=after, expected_sha256=_digest(before))
        for name in ("a.py", "b.py")
    ]

    class Fixer:
        async def propose(self, **kwargs):
            return FixerOutput(
                proposal=PatchProposal(
                    summary="update both files",
                    changed_files=["a.py", "b.py"],
                    risk="low",
                    verification_plan=[PASSING],
                ),
                edits=edits,
            )

    graph = build_graph(_runtime(tmp_path, Fixer()), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume-partial-patch"}}
    await graph.ainvoke(_failure(tmp_path, [PASSING]), config, interrupt_before=["apply_candidate"])
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("apply_candidate",)
    # Reproduce the persisted-state/filesystem boundary after a process dies:
    # one atomic file replacement completed, but the node has not checkpointed.
    apply_edit(tmp_path, FileTools(tmp_path, writable=True), edits[0])
    result = await graph.ainvoke(None, config)
    observed = {
        "status": result["status"],
        "reason": result.get("failure_reason"),
        "a": (tmp_path / "a.py").read_text(),
        "b": (tmp_path / "b.py").read_text(),
    }
    assert observed["status"] == "success", observed
    assert observed["a"] == observed["b"] == after


@pytest.mark.asyncio
async def test_checkpoint_replay_accepts_already_deleted_first_edit(tmp_path):
    before = "VALUE = 0\n"
    after = "VALUE = 1\n"
    gone = tmp_path / "gone.py"
    gone.write_text("old\n")
    (tmp_path / "b.py").write_text(before)
    edits = [
        FileEdit(path="gone.py", delete=True, expected_sha256=_digest("old\n")),
        FileEdit(path="b.py", content=after, expected_sha256=_digest(before)),
    ]

    class Fixer:
        async def propose(self, **kwargs):
            return FixerOutput(
                proposal=PatchProposal(
                    summary="delete then update",
                    changed_files=["gone.py", "b.py"],
                    risk="low",
                    verification_plan=[PASSING],
                ),
                edits=edits,
            )

    graph = build_graph(_runtime(tmp_path, Fixer()), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume-partial-delete"}}
    await _pause_before_apply(graph, _failure(tmp_path, [PASSING]), config)
    gone.unlink()
    result = await graph.ainvoke(None, config)
    assert result["status"] == "success", result.get("failure_reason")
    assert not gone.exists()
    assert (tmp_path / "b.py").read_text() == after


@pytest.mark.asyncio
async def test_checkpoint_replay_accepts_fully_applied_patch(tmp_path):
    before = "VALUE = 0\n"
    after = "VALUE = 1\n"
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(before)
    edits = [
        FileEdit(path=name, content=after, expected_sha256=_digest(before))
        for name in ("a.py", "b.py")
    ]

    class Fixer:
        async def propose(self, **kwargs):
            return FixerOutput(
                proposal=PatchProposal(
                    summary="update both files",
                    changed_files=["a.py", "b.py"],
                    risk="low",
                    verification_plan=[PASSING],
                ),
                edits=edits,
            )

    graph = build_graph(_runtime(tmp_path, Fixer()), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume-full-patch"}}
    await graph.ainvoke(_failure(tmp_path, [PASSING]), config, interrupt_before=["apply_candidate"])
    tools = FileTools(tmp_path, writable=True)
    for edit in edits:
        apply_edit(tmp_path, tools, edit)
    result = await graph.ainvoke(None, config)
    assert result["status"] == "success", result.get("failure_reason")
    assert (tmp_path / "a.py").read_text() == after
    assert (tmp_path / "b.py").read_text() == after


@pytest.mark.asyncio
async def test_checkpoint_replay_preserves_third_party_conflict(tmp_path):
    before = "VALUE = 0\n"
    after = "VALUE = 1\n"
    third = "VALUE = 99\n"
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(before)
    edits = [
        FileEdit(path=name, content=after, expected_sha256=_digest(before))
        for name in ("a.py", "b.py")
    ]

    class Fixer:
        async def propose(self, **kwargs):
            return FixerOutput(
                proposal=PatchProposal(
                    summary="update both files",
                    changed_files=["a.py", "b.py"],
                    risk="low",
                    verification_plan=[PASSING],
                ),
                edits=edits,
            )

    graph = build_graph(_runtime(tmp_path, Fixer()), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume-third-party"}}
    await graph.ainvoke(_failure(tmp_path, [PASSING]), config, interrupt_before=["apply_candidate"])
    apply_edit(tmp_path, FileTools(tmp_path, writable=True), edits[0])
    (tmp_path / "b.py").write_text(third)
    result = await graph.ainvoke(None, config)
    assert result["status"] == "failed"
    assert "b.py" in (result.get("failure_reason") or "")
    assert (tmp_path / "a.py").read_text() == before
    assert (tmp_path / "b.py").read_text() == third


@pytest.mark.asyncio
async def test_rollback_deleted_executable_restores_mode(tmp_path):
    script = tmp_path / "build.sh"
    contents = "#!/bin/sh\nexit 0\n"
    script.write_text(contents)
    script.chmod(0o751)

    class DeleteFixer:
        async def propose(self, **kwargs):
            return FixerOutput(
                proposal=PatchProposal(
                    summary="remove build script",
                    changed_files=["build.sh"],
                    risk="low",
                    verification_plan=[PASSING],
                ),
                edits=[FileEdit(path="build.sh", delete=True)],
            )

    graph = build_graph(_runtime(tmp_path, DeleteFixer()), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "delete-rollback-mode"}}
    pending = await graph.ainvoke(
        _failure(tmp_path, [["python", "-c", "raise SystemExit(1)"]]), config
    )
    assert pending.get("__interrupt__"), "delete must pass through the real approval boundary"
    result = await graph.ainvoke(Command(resume=True), config)
    assert result["status"] == "failed"
    assert script.read_text() == contents
    mode = stat.S_IMODE(script.stat().st_mode)
    assert mode == 0o751, f"rollback changed executable mode from 0751 to {mode:04o}"


@pytest.mark.asyncio
async def test_rollback_deleted_regular_file_restores_mode(tmp_path):
    path = tmp_path / "notes.txt"
    contents = "keep\n"
    path.write_text(contents)
    path.chmod(0o640)

    class DeleteFixer:
        async def propose(self, **kwargs):
            return FixerOutput(
                proposal=PatchProposal(
                    summary="remove notes",
                    changed_files=["notes.txt"],
                    risk="low",
                    verification_plan=[PASSING],
                ),
                edits=[FileEdit(path="notes.txt", delete=True)],
            )

    graph = build_graph(_runtime(tmp_path, DeleteFixer()), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "delete-rollback-regular-mode"}}
    pending = await graph.ainvoke(
        _failure(tmp_path, [["python", "-c", "raise SystemExit(1)"]]), config
    )
    assert pending.get("__interrupt__")
    result = await graph.ainvoke(Command(resume=True), config)
    assert result["status"] == "failed"
    assert path.read_text() == contents
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.asyncio
async def test_single_rollback_deleted_executable_restores_mode(tmp_path):
    script = tmp_path / "build.sh"
    contents = "#!/bin/sh\nexit 0\n"
    script.write_text(contents)
    script.chmod(0o751)

    class DeleteFixer:
        async def propose(self, **kwargs):
            return FixerOutput(
                proposal=PatchProposal(
                    summary="remove build script",
                    changed_files=["build.sh"],
                    risk="low",
                    verification_plan=[PASSING],
                ),
                edits=[FileEdit(path="build.sh", delete=True)],
            )

    runtime = _runtime(tmp_path, DeleteFixer())
    result = await _drive_single(
        SimpleNamespace(runtime=runtime, recorder=TrajectoryRecorder()),
        _failure(tmp_path, [["python", "-c", "raise SystemExit(1)"]]),
    )
    assert result["status"] == "failed"
    assert script.read_text() == contents
    mode = stat.S_IMODE(script.stat().st_mode)
    assert mode == 0o751, f"single rollback changed executable mode from 0751 to {mode:04o}"
