from __future__ import annotations

import subprocess
from pathlib import Path

from evoci.graph.routing import contains_workspace_review_bypass


def _git(workspace: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(workspace: Path) -> None:
    (workspace / "module.py").write_text(
        "import pytest\n\nVALUE = 1\n\ndef existing_skip():\n"
        "    pytest.skip('pre-existing')\n"
    )
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.name", "test")
    _git(workspace, "config", "user.email", "test@example.invalid")
    _git(workspace, "add", "module.py")
    _git(workspace, "commit", "-qm", "baseline")


def test_review_bypass_ignores_preexisting_directives(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    target = tmp_path / "module.py"
    target.write_text(target.read_text().replace("VALUE = 1", "VALUE = 2"))

    assert contains_workspace_review_bypass(tmp_path) == []


def test_review_bypass_ignores_internal_artifacts(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    artifact = tmp_path / ".evoci" / "state" / "artifacts" / "patch.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"content": "pytest.skip(\\\"recorded\\\")"}')

    assert contains_workspace_review_bypass(tmp_path) == []


def test_review_bypass_rejects_new_directives(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    target = tmp_path / "module.py"
    target.write_text(target.read_text() + "\npytest.skip('new bypass')\n")

    assert contains_workspace_review_bypass(tmp_path) == [
        "possible test or CI bypass in final diff: pytest.skip("
    ]
