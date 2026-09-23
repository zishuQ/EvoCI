import json
import shutil
import subprocess
from pathlib import Path

import pytest

from evoci.benchmark.adapters import CIRepairBenchAdapter
from evoci.benchmark.docker import (
    DockerImageManager,
    DockerReplayVerifier,
    PatchApplyError,
    ProtectedPatchError,
    build_candidate_patch,
    build_test_command,
    django_label,
    django_modules,
    lookup_node_status,
    parse_log_django,
)
from evoci.benchmark.models import AgentTaskView, DockerTaskSpec, PreparedTask
from evoci.benchmark.validate import validate_reference
from evoci.tools.filesystem import FileTools
from evoci.tools.isolation import copy_workspace_with_independent_git
from evoci.tools.shell import CommandRunner, reset_container_executor, set_container_executor


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.counter = 0
        self.copied: list[str] = []
        self.ftp_mode: str = "fail_until_candidate"
        self.ptp_status: str = "PASSED"
        self.patch_apply_fail: bool = False
        self.exec_stdout: str | None = None
        self.exec_code: int | None = None
        self.timeout_exec: bool = False
        self.pip_fail: bool = False
        self.ptp_fail_first: int = 0
        self._ptp_execs: int = 0
        self.unrelated_log: str = ""
        self.copied_files: dict[str, bytes] = {}
        self.copies_by_container: dict[str, set[str]] = {}

    def run(self, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        del timeout
        self.calls.append(args)
        if args[:1] == ["pull"]:
            return subprocess.CompletedProcess(args, 0, "pulled\n", "")
        if args[:2] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                args, 0, json.dumps(["swebench/test@sha256:abc"]), ""
            )
        if args[:1] == ["run"]:
            self.counter += 1
            return subprocess.CompletedProcess(args, 0, f"container-{self.counter}\n", "")
        if args[:1] == ["cp"]:
            source, dest = args[1], args[2]
            if ":" in source and ":" not in dest:
                return subprocess.CompletedProcess(args, 1, "", "no junit in fake")
            container, remote = dest.split(":", 1)
            self.copied.append(remote)
            self.copies_by_container.setdefault(container, set()).add(remote)
            if ":" not in source:
                self.copied_files[remote] = Path(source).read_bytes()
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:1] == ["exec"]:
            index = 1
            while index < len(args) and args[index] in {"--workdir", "--env"}:
                index += 2
            container = args[index]
            command = args[index + 1 :]
            return self._exec(args, command, container=container)
        if args[:1] == ["rm"]:
            name = args[-1]
            self.copies_by_container.pop(name, None)
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)

    def _exec(
        self, args: list[str], command: list[str], container: str = ""
    ) -> subprocess.CompletedProcess[str]:
        if self.timeout_exec and command[:1] == ["python"]:
            return subprocess.CompletedProcess(args, 124, "", "docker timed out")
        if command[:2] == ["git", "-C"] or command[:1] == ["patch"]:
            if self.patch_apply_fail:
                return subprocess.CompletedProcess(args, 1, "", "patch does not apply")
            return subprocess.CompletedProcess(args, 0, "applied\n", "")
        if command[:3] == ["python", "-m", "pip"]:
            if self.pip_fail:
                return subprocess.CompletedProcess(args, 1, "", "pip install failed")
            return subprocess.CompletedProcess(args, 0, "installed\n", "")
        if self.exec_stdout is not None:
            code = 1 if self.exec_code is None else self.exec_code
            return subprocess.CompletedProcess(args, code, self.exec_stdout, "")
        has_candidate = "/tmp/candidate.patch" in self.copies_by_container.get(container, set())
        nodes = [part for part in command if "::" in part or " (" in part]
        lines: list[str] = []
        failed = False
        if any("pass_reg" in node or node.endswith("test_reg") for node in nodes):
            self._ptp_execs += 1
        for node in nodes:
            if "pass_reg" in node or node.endswith("test_reg"):
                if self.ptp_fail_first and self._ptp_execs <= self.ptp_fail_first:
                    status = "FAILED"
                else:
                    status = self.ptp_status
            elif self.ftp_mode == "already_pass":
                status = "PASSED"
            elif self.ftp_mode == "fail_until_candidate":
                status = "PASSED" if has_candidate else "FAILED"
            else:
                status = "FAILED"
            if status != "PASSED":
                failed = True
            lines.append(f"{status} {node}")
        if not nodes and command[:3] == ["python", "-m", "pytest"]:
            lines.append("ERROR collecting tests")
            return subprocess.CompletedProcess(args, 2, "\n".join(lines) + "\n", "")
        stdout = "\n".join(lines) + "\n" + self.unrelated_log
        return subprocess.CompletedProcess(args, 1 if failed else 0, stdout, "")


class OverlayDocker(FakeDocker):
    """Simulate an official image /testbed that is not the host workspace."""

    def __init__(self, overlay: Path, image_files: dict[str, str]) -> None:
        super().__init__()
        self.overlay = overlay
        self.image_files = dict(image_files)
        self._reset_overlay()

    def _reset_overlay(self) -> None:
        if self.overlay.exists():
            shutil.rmtree(self.overlay)
        self.overlay.mkdir(parents=True)
        for relative, content in self.image_files.items():
            path = self.overlay / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        subprocess.run(["git", "init", "-q"], cwd=self.overlay, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=self.overlay, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@example.invalid"], cwd=self.overlay, check=True
        )
        subprocess.run(["git", "add", "."], cwd=self.overlay, check=True)
        subprocess.run(["git", "commit", "-qm", "image-baseline"], cwd=self.overlay, check=True)

    def _exec(
        self, args: list[str], command: list[str], container: str = ""
    ) -> subprocess.CompletedProcess[str]:
        if command and command[0] == "git" and "-C" in command:
            git_cmd = command[command.index("-C") + 2 :]
            if git_cmd and git_cmd[0] == "apply":
                patch_dest = git_cmd[-1]
                payload = self.copied_files.get(patch_dest)
                if payload is not None:
                    patch_file = self.overlay / ".incoming.patch"
                    patch_file.write_bytes(payload)
                    git_cmd = [*git_cmd[:-1], str(patch_file)]
            completed = subprocess.run(
                ["git", "-C", str(self.overlay), *git_cmd],
                check=False,
                capture_output=True,
                text=True,
            )
            return subprocess.CompletedProcess(
                args, completed.returncode, completed.stdout, completed.stderr
            )
        if command[:2] == ["python", "-c"]:
            completed = subprocess.run(
                command, cwd=self.overlay, check=False, capture_output=True, text=True
            )
            return subprocess.CompletedProcess(
                args, completed.returncode, completed.stdout, completed.stderr
            )
        return super()._exec(args, command, container=container)


def _task() -> PreparedTask:
    return PreparedTask(
        agent_view=AgentTaskView(
            task_id="t",
            repo_owner="o",
            repo_name="r",
            workflow_name="ci",
            workflow_path="ci.yml",
            sha_fail="fail",
            ci_failure={
                "workflow_yaml": "",
                "log_text": "",
                "failed_steps": [],
                "candidate_failed_commands": [],
            },
        )
    )


def _spec(**overrides: object) -> DockerTaskSpec:
    payload: dict[str, object] = {
        "task_id": "t",
        "official_image": "swebench/test:latest",
        "fail_to_pass": ["tests/test_x.py::test_target"],
        "pass_to_pass": ["tests/test_x.py::test_reg"],
        "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
        "protected_files": ["tests/test_x.py"],
        "reference_patch": "diff --git a/src.py b/src.py\n",
    }
    payload.update(overrides)
    return DockerTaskSpec.model_validate(payload)


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=root, check=True)
    (root / "src.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)


def test_adapter_parses_evaluator_truth_without_leaking_to_agent(tmp_path: Path) -> None:
    row = {
        "task_id": "t",
        "repo_owner": "o",
        "repo_name": "r",
        "workflow_path": "ci.yml",
        "sha_fail": "f",
        "sha_success": "s",
        "official_image": "swebench/test:latest",
        "local_replay_command": ["pytest", "-q"],
        "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
        "fail_to_pass": '["tests/test_x.py::test_target"]',
        "pass_to_pass": ["tests/test_x.py::test_reg"],
        "diff": "SECRET_REFERENCE",
        "changed_files": ["x.py"],
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    adapter = CIRepairBenchAdapter(path)
    prepared = adapter.prepare_task("t")
    serialized = prepared.agent_view.model_dump_json()
    assert "official_image" not in serialized
    assert "SECRET_REFERENCE" not in serialized
    assert "fail_to_pass" not in serialized
    assert "pass_to_pass" not in serialized
    assert "test_patch" not in serialized
    assert "reference_patch" not in serialized
    spec = adapter.docker_task_spec("t")
    assert spec is not None
    assert spec.fail_to_pass == ["tests/test_x.py::test_target"]
    assert spec.pass_to_pass == ["tests/test_x.py::test_reg"]
    assert spec.test_patch.startswith("diff --git")
    assert spec.protected_files == ["tests/test_x.py"]
    assert spec.reference_patch == "SECRET_REFERENCE"
    assert spec.replay_commands == [["pytest", "-q"]]


def test_adapter_rejects_non_test_node_ids(tmp_path: Path) -> None:
    row = {
        "task_id": "t",
        "repo_owner": "o",
        "repo_name": "r",
        "workflow_path": "ci.yml",
        "sha_fail": "f",
        "sha_success": "s",
        "official_image": "swebench/test:latest",
        "fail_to_pass": ["tests/test_x.py::test_target", "[100%]"],
        "pass_to_pass": ["tests/test_x.py::test_reg"],
        "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="invalid test node id"):
        CIRepairBenchAdapter(path).docker_task_spec("t")


def test_adapter_rejects_empty_pass_to_pass_for_official_image(tmp_path: Path) -> None:
    row = {
        "task_id": "t",
        "repo_owner": "o",
        "repo_name": "r",
        "workflow_path": "ci.yml",
        "sha_fail": "f",
        "sha_success": "s",
        "official_image": "swebench/test:latest",
        "fail_to_pass": ["tests/test_x.py::test_target"],
        "pass_to_pass": [],
        "test_patch": "diff --git a/t.py b/t.py\n",
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="pass_to_pass"):
        CIRepairBenchAdapter(path).docker_task_spec("t")


def test_adapter_rejects_empty_fail_to_pass(tmp_path: Path) -> None:
    row = {
        "task_id": "t",
        "repo_owner": "o",
        "repo_name": "r",
        "workflow_path": "ci.yml",
        "sha_fail": "f",
        "sha_success": "s",
        "official_image": "swebench/test:latest",
        "fail_to_pass": [],
        "test_patch": "diff --git a/t.py b/t.py\n",
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    adapter = CIRepairBenchAdapter(path)
    with pytest.raises(ValueError, match="fail_to_pass"):
        adapter.docker_task_spec("t")


def test_image_manager_reuses_local_digest_without_registry_request(tmp_path: Path) -> None:
    fake = FakeDocker()
    specs = [_spec(task_id="a"), _spec(task_id="b")]
    pinned = DockerImageManager(fake).prepare(specs, audit_path=tmp_path / "images.json")
    assert pinned == {"a": "swebench/test@sha256:abc", "b": "swebench/test@sha256:abc"}
    assert sum(call[:1] == ["pull"] for call in fake.calls) == 0


def test_eval_container_does_not_bind_mount_testbed() -> None:
    command = build_test_command(["tests/test_x.py::test_target"])
    assert command[:3] == ["python", "-m", "pytest"]
    assert "tests/test_x.py::test_target" in command
    assert django_label("test_negative (utils_tests.test_dateparse.DurationParseTests)") == (
        "utils_tests.test_dateparse.DurationParseTests.test_negative"
    )
    assert django_modules(
        [
            "test_negative (utils_tests.test_dateparse.DurationParseTests)",
            "test_parse_postgresql_format (utils_tests.test_dateparse.DurationParseTests)",
        ]
    ) == ["utils_tests.test_dateparse"]
    django_cmd = build_test_command(
        ["test_negative (utils_tests.test_dateparse.DurationParseTests)"]
    )
    assert django_cmd[:2] == ["python", "tests/runtests.py"]
    assert "utils_tests.test_dateparse" in django_cmd


def test_django_log_parser_handles_subtests_and_two_tests_on_one_line() -> None:
    log = (
        "test_negative (utils_tests.test_dateparse.DurationParseTests) ... "
        "test_parse_postgresql_format (utils_tests.test_dateparse.DurationParseTests) ... ok\n"
        "FAIL: test_negative (utils_tests.test_dateparse.DurationParseTests) "
        "(source='-15:30')\n"
        "FAIL: test_negative (utils_tests.test_dateparse.DurationParseTests) "
        "(source='-01:-01')\n"
    )
    parsed = parse_log_django(log)
    ftp = "test_negative (utils_tests.test_dateparse.DurationParseTests)"
    ptp = "test_parse_postgresql_format (utils_tests.test_dateparse.DurationParseTests)"
    assert lookup_node_status(parsed, ftp) == "FAILED"
    assert lookup_node_status(parsed, ptp) == "PASSED"


@pytest.mark.asyncio
async def test_eval_run_has_no_testbed_bind_mount_and_applies_test_patch_first(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    assert preflight.status == "reproduced"
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "passed"
    run_calls = [call for call in fake.calls if call[:1] == ["run"]]
    assert run_calls
    for call in run_calls:
        assert not any(arg.startswith("type=bind") and "/testbed" in arg for arg in call)
        assert "--mount" not in call
    copies = [call for call in fake.calls if call[:1] == ["cp"] and ":" in call[2]]
    dests = [call[2].split(":", 1)[1] for call in copies]
    assert dests.index("/tmp/test.patch") < dests.index("/tmp/candidate.patch")
    verifier.close()
    assert [call for call in fake.calls if call[:1] == ["rm"]]


def test_candidate_patch_includes_new_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    (tmp_path / "new.py").write_text("print('added')\n")
    patch = build_candidate_patch(tmp_path)
    assert "src.py" in patch
    assert "new.py" in patch
    assert "print('added')" in patch


def test_candidate_patch_rejects_protected_files(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_target():\n    assert False\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "tests"], cwd=tmp_path, check=True)
    (tmp_path / "tests" / "test_x.py").write_text("def test_target():\n    assert True\n")
    with pytest.raises(ProtectedPatchError):
        build_candidate_patch(tmp_path, protected_files=["tests/test_x.py"])


@pytest.mark.asyncio
async def test_assertion_failure_is_unresolved(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    fake.ftp_mode = "always_fail"
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    assert preflight.status == "reproduced"
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "failed"
    assert result.failure_class == "assertion"


@pytest.mark.asyncio
async def test_collection_import_timeout_and_patch_apply_are_infra_error(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    fake.exec_stdout = "ERROR collecting tests\nImportError: missing\n"
    fake.exec_code = 2
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    preflight = await verifier.preflight(_task(), tmp_path)
    assert preflight.status == "infra_error"

    fake = FakeDocker()
    fake.timeout_exec = True
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    preflight = await verifier.preflight(_task(), tmp_path)
    assert preflight.status == "infra_error"
    assert "timeout" in preflight.details

    fake = FakeDocker()
    fake.patch_apply_fail = True
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    preflight = await verifier.preflight(_task(), tmp_path)
    assert preflight.status == "infra_error"
    assert "patch apply" in preflight.details


@pytest.mark.asyncio
async def test_ftp_pass_ptp_fail_is_unresolved(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    fake.ptp_status = "FAILED"
    verifier = DockerReplayVerifier(
        _spec(), "swebench/test@sha256:abc", cli=fake, ptp_retries=0
    )
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "failed"
    assert result.ftp_results["tests/test_x.py::test_target"] == "PASSED"
    assert result.ptp_results["tests/test_x.py::test_reg"] == "FAILED"


@pytest.mark.asyncio
async def test_skipped_ptp_is_not_resolved(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    fake.ptp_status = "SKIPPED"
    verifier = DockerReplayVerifier(
        _spec(), "swebench/test@sha256:abc", cli=fake, ptp_retries=1
    )
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "failed"
    assert result.ptp_results["tests/test_x.py::test_reg"] == "SKIPPED"
    assert len(result.attempts) == 1


@pytest.mark.asyncio
async def test_reinstall_failure_is_infra_error(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    fake.pip_fail = True
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "infra_error"
    assert "editable reinstall failed" in result.details


@pytest.mark.asyncio
async def test_flaky_ptp_retries_for_candidate_and_gold(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    fake.ptp_fail_first = 1
    verifier = DockerReplayVerifier(
        _spec(), "swebench/test@sha256:abc", cli=fake, ptp_retries=1
    )
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "passed"
    assert len(result.attempts) == 2
    assert result.attempts[0].ptp_results["tests/test_x.py::test_reg"] == "FAILED"
    assert result.attempts[1].ptp_results["tests/test_x.py::test_reg"] == "PASSED"


@pytest.mark.asyncio
async def test_ftp_and_ptp_pass_is_resolved(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert preflight.status == "reproduced"
    assert result.status == "passed"


@pytest.mark.asyncio
async def test_baseline_already_passing_is_not_reproduced(tmp_path: Path) -> None:
    fake = FakeDocker()
    fake.ftp_mode = "already_pass"
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    preflight = await verifier.preflight(_task(), tmp_path)
    assert preflight.status == "not_reproduced"
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "not_available"


def test_reference_validation_does_not_call_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "task_id": "t",
        "repo_owner": "o",
        "repo_name": "r",
        "workflow_path": "ci.yml",
        "sha_fail": "f",
        "sha_success": "s",
        "official_image": "swebench/test:latest",
        "fail_to_pass": ["tests/test_x.py::test_target"],
        "pass_to_pass": ["tests/test_x.py::test_reg"],
        "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n",
        "diff": "diff --git a/src.py b/src.py\n",
        "changed_files": ["src.py"],
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(row) + "\n")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"task_id": "t", "category": "x"}) + "\n")
    fake = FakeDocker()

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("LLM must not be constructed during reference validation")

    monkeypatch.setattr("evoci.model.openai_compatible.OpenAICompatibleGateway.__init__", boom)
    report = validate_reference(
        dataset=dataset,
        manifest=manifest,
        docker_official_images=True,
        apply_reference_patch=True,
        timeout=30,
        cli=fake,
    )
    assert report.valid == 1
    assert report.invalid == 0
    assert report.infra_error == 0
    source = Path("src/evoci/benchmark/validate.py").read_text(encoding="utf-8")
    assert "OpenAICompatibleGateway" not in source
    assert "persist_run_outcome" not in source
    assert "SQLiteMemoryStore" not in source


def _gold_patch() -> str:
    return (
        "diff --git a/src.py b/src.py\n"
        "--- a/src.py\n"
        "+++ b/src.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )


async def _run_agent_command(
    verifier: DockerReplayVerifier, workspace: Path, argv: list[str]
) -> str:
    token = set_container_executor(verifier.execute_agent_command)
    try:
        result = await CommandRunner(workspace, timeout=5, max_chars=8000).run(argv)
    finally:
        reset_container_executor(token)
    return result.stdout


@pytest.mark.asyncio
async def test_work_container_sees_replace_text_before_run_test(tmp_path: Path) -> None:
    host = tmp_path / "host"
    host.mkdir()
    _init_repo(host)
    overlay = tmp_path / "image-testbed"
    fake = OverlayDocker(overlay, {"src.py": "VALUE = 1\n", ".image-generated": "from-image\n"})
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(host)
    verifier.start_work_container(host)
    FileTools(host, writable=True).replace_text("src.py", "VALUE = 1", "VALUE = 2")
    output = await _run_agent_command(
        verifier,
        host,
        ["python", "-c", "from pathlib import Path; print(Path('src.py').read_text())"],
    )
    assert "VALUE = 2" in output
    git_cmds = [
        call
        for call in fake.calls
        if call[:1] == ["exec"] and "git" in call
    ]
    assert any("reset" in call for call in git_cmds)
    assert any("apply" in call for call in git_cmds)
    verifier.close()


@pytest.mark.asyncio
async def test_docker_executes_staging_workspace_not_original(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.mkdir()
    _init_repo(original)
    staging = tmp_path / "staging"
    copy_workspace_with_independent_git(original, staging)
    (staging / "src.py").write_text("VALUE = staging\n")
    overlay = tmp_path / "image-testbed"
    fake = OverlayDocker(overlay, {"src.py": "VALUE = 1\n"})
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(original)
    verifier.start_work_container(original)
    output = await _run_agent_command(
        verifier,
        staging,
        ["python", "-c", "from pathlib import Path; print(Path('src.py').read_text())"],
    )
    assert "staging" in output
    assert (original / "src.py").read_text() == "VALUE = 1\n"
    verifier.close()


@pytest.mark.asyncio
async def test_image_generated_testbed_file_is_not_covered_by_host_mount(
    tmp_path: Path,
) -> None:
    host = tmp_path / "host"
    host.mkdir()
    _init_repo(host)
    assert not (host / ".image-generated").exists()
    overlay = tmp_path / "image-testbed"
    fake = OverlayDocker(overlay, {"src.py": "VALUE = 1\n", ".image-generated": "from-image\n"})
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(host)
    verifier.start_work_container(host)
    run_calls = [call for call in fake.calls if call[:1] == ["run"]]
    assert run_calls
    for call in run_calls:
        assert "--mount" not in call
        assert not any(arg.startswith("type=bind") and "/testbed" in arg for arg in call)
    output = await _run_agent_command(
        verifier,
        host,
        [
            "python",
            "-c",
            "from pathlib import Path; print(Path('.image-generated').read_text())",
        ],
    )
    assert "from-image" in output
    verifier.close()


@pytest.mark.asyncio
async def test_work_container_defers_image_baseline_conflicts_to_patch_sync(
    tmp_path: Path,
) -> None:
    host = tmp_path / "host"
    host.mkdir()
    _init_repo(host)
    overlay = tmp_path / "image-testbed"
    fake = OverlayDocker(overlay, {"src.py": "VALUE = different\n"})
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    fake.patch_apply_fail = True
    verifier.protect(host)
    verifier.start_work_container(host)
    FileTools(host, writable=True).replace_text("src.py", "VALUE = 1", "VALUE = 2")

    with pytest.raises(PatchApplyError, match="patch apply failed"):
        verifier._sync_workspace_into_container(host)

    verifier.close()


@pytest.mark.parametrize(
    "instance_id",
    ["pytest-5262", "pytest-6202", "pytest-7205", "pytest-5631"],
)
@pytest.mark.asyncio
async def test_gold_equivalent_patch_resolves_in_fresh_eval_container(
    tmp_path: Path, instance_id: str
) -> None:
    del instance_id
    _init_repo(tmp_path)
    fake = FakeDocker()
    spec = _spec(reference_patch=_gold_patch())
    verifier = DockerReplayVerifier(spec, "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    assert preflight.status == "reproduced"
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "passed"
    assert result.ftp_results["tests/test_x.py::test_target"] == "PASSED"
    assert result.ptp_results["tests/test_x.py::test_reg"] == "PASSED"
    run_calls = [call for call in fake.calls if call[:1] == ["run"]]
    assert len(run_calls) >= 2
    for call in run_calls:
        assert "--mount" not in call
    verifier.close()


@pytest.mark.asyncio
async def test_requests_2931_target_only_tests_can_pass(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    spec = _spec(
        fail_to_pass=["tests/test_requests.py::test_idna_allow_underscores"],
        pass_to_pass=["tests/test_requests.py::test_params"],
        test_patch="diff --git a/tests/test_requests.py b/tests/test_requests.py\n",
        protected_files=["tests/test_requests.py"],
    )
    verifier = DockerReplayVerifier(spec, "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "passed"
    pytest_cmds = [
        call
        for call in fake.calls
        if "pytest" in call
    ]
    assert pytest_cmds
    for call in pytest_cmds:
        joined = " ".join(call)
        assert "httpbin" not in joined
        target = "tests/test_requests.py::test_idna_allow_underscores"
        assert target in joined or "test_params" in joined
    verifier.close()


@pytest.mark.asyncio
async def test_requests_2931_unrelated_httpbin_fixture_does_not_veto_target(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    fake.unrelated_log = (
        "ERROR tests/test_httpbin.py::test_fixture - fixture 'httpbin' not found\n"
    )
    spec = _spec(
        fail_to_pass=["tests/test_requests.py::test_idna_allow_underscores"],
        pass_to_pass=["tests/test_requests.py::test_params"],
        test_patch="diff --git a/tests/test_requests.py b/tests/test_requests.py\n",
        protected_files=["tests/test_requests.py"],
    )
    verifier = DockerReplayVerifier(spec, "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    preflight = await verifier.preflight(_task(), tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "passed"
    assert result.ftp_results["tests/test_requests.py::test_idna_allow_underscores"] == "PASSED"
    verifier.close()


@pytest.mark.asyncio
async def test_final_evaluator_uses_fresh_container_and_requires_ftp_and_ptp(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    fake = FakeDocker()
    verifier = DockerReplayVerifier(_spec(), "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    verifier.start_work_container(tmp_path)
    work_runs = [call for call in fake.calls if call[:1] == ["run"]]
    preflight = await verifier.preflight(_task(), tmp_path)
    (tmp_path / "src.py").write_text("VALUE = 2\n")
    result = await verifier.verify(_task(), tmp_path, preflight)
    assert result.status == "passed"
    all_runs = [call for call in fake.calls if call[:1] == ["run"]]
    assert len(all_runs) > len(work_runs)
    eval_runs = all_runs[len(work_runs) :]
    assert eval_runs
    for call in all_runs:
        assert "--mount" not in call
    dests = [
        call[2].split(":", 1)[1]
        for call in fake.calls
        if call[:1] == ["cp"] and ":" in call[2]
    ]
    assert "/tmp/test.patch" in dests
    assert dests.index("/tmp/test.patch") < dests.index("/tmp/candidate.patch")
    verifier.close()


@pytest.mark.asyncio
async def test_agent_workspace_cannot_see_gold_or_evaluator_test_patch(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    gold = "SECRET_GOLD_PATCH_BODY"
    test_patch = "SECRET_EVALUATOR_TEST_PATCH"
    spec = _spec(reference_patch=gold, test_patch=test_patch)
    fake = FakeDocker()
    verifier = DockerReplayVerifier(spec, "swebench/test@sha256:abc", cli=fake)
    verifier.protect(tmp_path)
    verifier.start_work_container(tmp_path)
    texts: list[str] = []
    for path in tmp_path.rglob("*"):
        if not path.is_file():
            continue
        try:
            texts.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    tree = "\n".join(texts)
    assert gold not in tree
    assert test_patch not in tree
    serialized = _task().agent_view.model_dump_json()
    assert gold not in serialized
    assert test_patch not in serialized
    assert "fail_to_pass" not in serialized
    verifier.close()
