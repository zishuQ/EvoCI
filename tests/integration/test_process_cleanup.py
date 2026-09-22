"""Cancellation must join process cleanup before callers close resources."""

import asyncio
from contextlib import suppress
from pathlib import Path

import pytest

from evoci.capability.models import GeneratedFile
from evoci.capability.validator import CandidateValidator
from evoci.tools.policy import FIXER_CAPABILITIES
from evoci.tools.registry import create_worker_registry
from evoci.tools.shell import CommandRunner
from tests.unit.test_capability import candidate, registry


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["command", "skill", "validation_pytest"])
async def test_cancel_reaps_descendants_before_return(tmp_path: Path, entry: str) -> None:
    ready = tmp_path / "ready"
    survived = tmp_path / "survived"
    script = (
        "import os, time\nfrom pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.close(1)\n    os.close(2)\n"
        "    time.sleep(0.7)\n"
        f"    Path({str(survived)!r}).write_text('survived')\n"
        "    os._exit(0)\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        "time.sleep(15)\n"
    )
    store = registry(tmp_path)
    tools = None
    if entry == "command":
        operation = CommandRunner(tmp_path).run(["python", "-c", script])
    else:
        skill = candidate(script=script).model_copy(update={"tests": []})
        if entry == "validation_pytest":
            skill = skill.model_copy(
                update={
                    "tests": [
                        GeneratedFile(
                            path="tests/test_child.py",
                            content="def test_child():\n"
                            + "\n".join("    " + line for line in script.splitlines()),
                        )
                    ]
                }
            )
        validator = CandidateValidator(timeout=10)
        if entry.startswith("validation"):
            staging, manifest = store._write_staging(skill, "pytest-import-debugging")
            record = type("Record", (), {"manifest": manifest})()
            operation = validator.avalidate_package(staging, manifest)
        else:
            record = store.create_skill(skill)
            tools = create_worker_registry(
                FIXER_CAPABILITIES,
                tmp_path,
                capability_registry=store,
                allowed_skill_refs={record.manifest.skill_id},
            )
            tools.invoke("load_skill", skill_id=record.manifest.skill_id)
            operation = tools.ainvoke(
                "run_skill_script",
                skill_id=record.manifest.skill_id,
                script_name="inspect_imports.py",
                args=[],
            )
    task = asyncio.create_task(operation)
    try:
        async with asyncio.timeout(5):
            while not ready.exists():
                if task.done():
                    pytest.fail(f"operation ended before spawning child: {task.result()}")
                await asyncio.sleep(0.01)
        task.cancel()
        # A second request must not interrupt the join and leak the worker.
        asyncio.get_running_loop().call_later(0.005, task.cancel)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        if entry.startswith("validation"):
            assert store.get(record.manifest.skill_id) is None
        if tools is not None:
            tools.close()
            tools = None
        await asyncio.sleep(0.8)
        assert not survived.exists(), "child continued running after cancellation returned"
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if tools is not None:
            tools.close()
        store.close()
