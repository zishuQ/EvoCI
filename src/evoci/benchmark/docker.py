"""Official-image evaluation that never bind-mounts over the image /testbed."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from evoci.benchmark.models import (
    BenchmarkCommandResult,
    BenchmarkPreflightResult,
    BenchmarkVerificationResult,
    DockerTaskSpec,
    EvalAttempt,
    PreparedTask,
)
from evoci.tools.shell import CommandResult

_TESTBED_PATH = (
    "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_DJANGO_NODE = re.compile(r"^(\S+)\s+\(([^)]+)\)$")
_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_INFRA_MARKERS = (
    "ERROR collecting",
    "ModuleNotFoundError",
    "ImportError",
    "InvalidVersion",
    "fixture not found",
    "recursive fixture",
    "INTERNALERROR",
    "no tests ran",
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class DockerError(RuntimeError):
    """A Docker lifecycle operation failed."""


class PatchApplyError(DockerError):
    """Applying an evaluator-owned or candidate patch failed."""


class UnsafePatchError(ValueError):
    """A generated candidate patch contained an unsafe path."""


class ProtectedPatchError(ValueError):
    """A candidate patch modified evaluator-owned test files."""


class DockerCLI(Protocol):
    def run(self, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]: ...


class SubprocessDockerCLI:
    def run(self, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["docker", *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "docker timed out")
            )
            return subprocess.CompletedProcess(["docker", *args], 124, stdout, stderr)


def _checked(
    cli: DockerCLI, args: list[str], *, timeout: float, operation: str
) -> subprocess.CompletedProcess[str]:
    result = cli.run(args, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise DockerError(f"{operation} failed: {detail or f'exit {result.returncode}'}")
    return result


class DockerImageManager:
    """Pull tags once and resolve every task to an immutable repository digest."""

    def __init__(self, cli: DockerCLI | None = None, *, timeout: float = 1800.0) -> None:
        self.cli = cli or SubprocessDockerCLI()
        self.timeout = timeout

    def prepare(
        self, specs: list[DockerTaskSpec], *, audit_path: Path | None = None
    ) -> dict[str, str]:
        images = sorted({spec.official_image for spec in specs})
        resolved: dict[str, str] = {}
        for image in images:
            inspect_args = ["image", "inspect", image, "--format", "{{json .RepoDigests}}"]
            inspected = self.cli.run(inspect_args, timeout=60)
            if inspected.returncode != 0:
                _checked(self.cli, ["pull", image], timeout=self.timeout, operation=f"pull {image}")
                inspected = _checked(
                    self.cli, inspect_args, timeout=60, operation=f"inspect {image}"
                )
            try:
                digests = json.loads(inspected.stdout.strip())
            except json.JSONDecodeError as exc:
                raise DockerError(f"inspect {image} returned invalid RepoDigests JSON") from exc
            if not isinstance(digests, list) or not digests:
                raise DockerError(f"official image {image} has no immutable repository digest")
            repository = image.rsplit(":", 1)[0]
            matching = [value for value in digests if str(value).startswith(repository + "@")]
            resolved[image] = str((matching or digests)[0])
        pinned = {spec.task_id: resolved[spec.official_image] for spec in specs}
        if audit_path is not None:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "images": [
                            {
                                "task_id": spec.task_id,
                                "source_image": spec.official_image,
                                "pinned_image": pinned[spec.task_id],
                            }
                            for spec in sorted(specs, key=lambda item: item.task_id)
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return pinned


@dataclass(frozen=True)
class _ProtectedFile:
    path: str
    content: bytes | None
    mode: int | None


class ProtectedFiles:
    def __init__(self, workspace: Path, paths: list[str]) -> None:
        self.workspace = workspace.resolve()
        self.files: list[_ProtectedFile] = []
        for relative in paths:
            candidate = self._resolve(relative)
            self.files.append(
                _ProtectedFile(
                    path=relative,
                    content=candidate.read_bytes() if candidate.is_file() else None,
                    mode=(candidate.stat().st_mode & 0o777) if candidate.is_file() else None,
                )
            )

    def _resolve(self, relative: str) -> Path:
        path = Path(relative)
        candidate = (self.workspace / path).resolve()
        if path.is_absolute() or (
            candidate != self.workspace and self.workspace not in candidate.parents
        ):
            raise ValueError(f"unsafe protected file path: {relative}")
        return candidate

    def restore(self) -> None:
        for saved in self.files:
            candidate = self._resolve(saved.path)
            if saved.content is None:
                if candidate.is_file() or candidate.is_symlink():
                    candidate.unlink()
                continue
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(saved.content)
            assert saved.mode is not None
            os.chmod(candidate, saved.mode)


def _git(
    workspace: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def paths_from_unified_diff(patch: str) -> list[str]:
    """Extract repository-relative paths from a git unified diff."""

    paths: list[str] = []
    for match in _DIFF_GIT.finditer(patch):
        path = match.group(2)
        if path not in paths:
            paths.append(path)
    return paths


def _assert_safe_patch_path(path: str, protected: set[str]) -> None:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise UnsafePatchError(f"unsafe candidate patch path: {path}")
    if ".git" in candidate.parts or path == ".git" or path.startswith(".git/"):
        raise UnsafePatchError(f"candidate patch must not touch .git: {path}")
    if path in protected:
        raise ProtectedPatchError(f"candidate patch modifies protected test file: {path}")


def build_candidate_patch(
    workspace: Path,
    base_sha: str | None = None,
    protected_files: Sequence[str] = (),
) -> str:
    """Build a binary-safe patch of agent edits relative to the task start commit."""

    workspace = workspace.resolve()
    protected = set(protected_files)
    if base_sha is None:
        parsed = _git(workspace, "rev-parse", "HEAD")
        if parsed.returncode != 0:
            raise ValueError(f"workspace is not a git repository: {parsed.stderr.strip()}")
        base_sha = parsed.stdout.strip()
    git_dir = _git(workspace, "rev-parse", "--git-dir")
    if git_dir.returncode != 0:
        raise ValueError(f"workspace is not a git repository: {git_dir.stderr.strip()}")
    git_dir_path = Path(git_dir.stdout.strip())
    if not git_dir_path.is_absolute():
        git_dir_path = workspace / git_dir_path
    index_src = git_dir_path / "index"
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "index"
        if index_src.is_file():
            shutil.copyfile(index_src, index_path)
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(index_path)
        added = _git(workspace, "add", "-A", env=env)
        if added.returncode != 0:
            raise ValueError(f"git add failed: {added.stderr.strip()}")
        diff = _git(workspace, "diff", "--cached", "--binary", base_sha, env=env)
        if diff.returncode != 0:
            raise ValueError(f"git diff failed: {diff.stderr.strip()}")
        patch = diff.stdout
    for path in paths_from_unified_diff(patch):
        _assert_safe_patch_path(path, protected)
    return patch


def is_django_node(node: str) -> bool:
    return _DJANGO_NODE.match(node) is not None


def django_label(node: str) -> str:
    match = _DJANGO_NODE.match(node)
    if match is None:
        return node
    return f"{match.group(2)}.{match.group(1)}"


def django_modules(nodes: list[str]) -> list[str]:
    """Map Django FAIL_TO_PASS ids to the modules the official harness runs."""

    modules: list[str] = []
    for node in nodes:
        match = _DJANGO_NODE.match(node)
        if match is None:
            continue
        module = match.group(2).rsplit(".", 1)[0]
        if module and module not in modules:
            modules.append(module)
    return modules


def build_test_command(nodes: list[str], *, junit_xml: str | None = None) -> list[str]:
    """Construct pytest/Django argv with each node id as its own argument."""

    if not nodes:
        raise ValueError("test command requires at least one node id")
    if all(is_django_node(node) for node in nodes):
        return [
            "python",
            "tests/runtests.py",
            "--verbosity",
            "2",
            "--settings=test_sqlite",
            "--parallel",
            "1",
            *django_modules(nodes),
        ]
    command = ["python", "-m", "pytest", "-rA", "--tb=short", "-v"]
    if junit_xml:
        command.extend(["--junitxml", junit_xml])
    command.extend(nodes)
    return command


def parse_log_pytest(log: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for raw in log.splitlines():
        line = _ANSI_ESCAPE.sub("", raw).strip()
        for prefix, status in (
            ("PASSED ", "PASSED"),
            ("FAILED ", "FAILED"),
            ("ERROR ", "ERROR"),
            ("SKIPPED ", "SKIPPED"),
            ("XFAIL ", "SKIPPED"),
            ("XPASS ", "PASSED"),
        ):
            if line.startswith(prefix):
                rest = line[len(prefix) :].strip()
                if rest[:1].isdigit() and ": " in rest:
                    rest = rest.split(": ", 1)[1]
                node = rest.split(" ", 1)[0].split(" - ", 1)[0].strip()
                if "::" in node:
                    statuses[node] = status
                break
        else:
            for suffix, status in (
                (" PASSED", "PASSED"),
                (" FAILED", "FAILED"),
                (" ERROR", "ERROR"),
                (" SKIPPED", "SKIPPED"),
            ):
                if line.endswith(suffix) and "::" in line:
                    node = line[: -len(suffix)].strip().split()[-1]
                    if "::" in node:
                        statuses[node] = status
                    break
    return statuses


@dataclass(frozen=True)
class _JUnitCase:
    classname: str
    name: str
    status: str


def parse_junit_xml(text: str) -> list[_JUnitCase]:
    root = ET.fromstring(text)
    cases: list[_JUnitCase] = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None:
            status = "FAILED"
        elif case.find("error") is not None:
            status = "ERROR"
        elif case.find("skipped") is not None:
            status = "SKIPPED"
        else:
            status = "PASSED"
        cases.append(
            _JUnitCase(
                classname=case.get("classname") or "",
                name=case.get("name") or "",
                status=status,
            )
        )
    return cases


def match_junit_status(node: str, cases: Sequence[_JUnitCase]) -> str:
    parts = node.split("::")
    expected_name = parts[-1]
    expected_class = parts[-2] if len(parts) >= 3 else None
    stem = Path(parts[0]).stem if parts else ""
    for case in cases:
        if case.name != expected_name:
            continue
        class_parts = case.classname.split(".")
        if expected_class is not None and expected_class not in class_parts:
            continue
        if stem and stem not in class_parts and stem not in case.classname:
            continue
        return case.status
    return "MISSING"


_DJANGO_INLINE_TEST = re.compile(r"(test\S+\s+\([^)]+\))")


def canonical_django_id(text: str) -> str | None:
    match = _DJANGO_NODE.match(text.strip())
    if match is not None:
        return f"{match.group(1)} ({match.group(2)})"
    match = re.match(r"^(test\S+)\s+\(([^)]+)\)", text.strip())
    if match is not None:
        return f"{match.group(1)} ({match.group(2)})"
    return None


def parse_log_django(log: str) -> dict[str, str]:
    """Parse Django runtests.py logs, including subTest and two-tests-per-line output."""

    statuses: dict[str, str] = {}
    for raw in log.splitlines():
        stripped = _ANSI_ESCAPE.sub("", raw).strip()
        for prefix, status in (("FAIL: ", "FAILED"), ("ERROR: ", "ERROR")):
            if stripped.startswith(prefix):
                ident = canonical_django_id(stripped[len(prefix) :])
                if ident is not None:
                    statuses[ident] = status
    for raw in log.splitlines():
        stripped = _ANSI_ESCAPE.sub("", raw).strip()
        names = _DJANGO_INLINE_TEST.findall(stripped)
        if not names:
            continue
        inline_status: str | None = None
        if stripped.endswith(" ... ok") or stripped.endswith(" ... OK"):
            inline_status = "PASSED"
        elif stripped.endswith(" ... FAIL"):
            inline_status = "FAILED"
        elif stripped.endswith(" ... ERROR"):
            inline_status = "ERROR"
        elif stripped.endswith(" ... skipped"):
            inline_status = "SKIPPED"
        if inline_status is None:
            continue
        statuses.setdefault(names[-1], inline_status)
    return statuses


def lookup_node_status(parsed: dict[str, str], node: str) -> str:
    if node in parsed:
        return parsed[node]
    canonical = canonical_django_id(node)
    if canonical is not None and canonical in parsed:
        return parsed[canonical]
    for key, status in parsed.items():
        if key == node or (canonical is not None and key == canonical):
            return status
    return "MISSING"


def detect_infra_reason(result: BenchmarkCommandResult, statuses: dict[str, str]) -> str | None:
    if result.timed_out or result.exit_code == 124:
        return "timeout"
    if any(status == "MISSING" for status in statuses.values()):
        missing = [node for node, status in statuses.items() if status == "MISSING"]
        return f"pytest did not collect target nodes: {missing[:5]}"
    if statuses and all(
        status in {"PASSED", "FAILED", "ERROR", "SKIPPED"} for status in statuses.values()
    ):
        return None
    text = f"{result.stdout}\n{result.stderr}"
    for marker in _INFRA_MARKERS:
        if marker in text:
            return marker
    if result.exit_code in {2, 3, 4, 5}:
        return f"test runner exit {result.exit_code}"
    return None


@dataclass(frozen=True)
class SuiteScore:
    commands: list[BenchmarkCommandResult]
    ftp_results: dict[str, str]
    ptp_results: dict[str, str]
    infra_reason: str | None
    attempts: list[EvalAttempt] = field(default_factory=list)

    @property
    def all_ftp_passed(self) -> bool:
        return bool(self.ftp_results) and all(
            status == "PASSED" for status in self.ftp_results.values()
        )

    @property
    def all_ftp_failed(self) -> bool:
        return bool(self.ftp_results) and all(
            status == "FAILED" for status in self.ftp_results.values()
        )

    @property
    def ftp_executed(self) -> bool:
        return bool(self.ftp_results) and all(
            status in {"PASSED", "FAILED", "ERROR"} for status in self.ftp_results.values()
        )

    @property
    def any_ftp_failed(self) -> bool:
        return any(status in {"FAILED", "ERROR"} for status in self.ftp_results.values())

    @property
    def all_ptp_passed(self) -> bool:
        return all(status == "PASSED" for status in self.ptp_results.values())

    def as_attempt(self, phase: str) -> EvalAttempt:
        return EvalAttempt(
            phase=phase,
            ftp_results=self.ftp_results,
            ptp_results=self.ptp_results,
            infra_reason=self.infra_reason,
        )


class DockerReplayVerifier:
    """Fail-before / pass-after scoring in fresh official images, never bind-mounting /testbed."""

    def __init__(
        self,
        spec: DockerTaskSpec,
        pinned_image: str,
        *,
        cli: DockerCLI | None = None,
        timeout: float = 1800.0,
        work_timeout: float | None = None,
        max_chars: int = 500_000,
        preflight_retries: int = 0,
        ptp_retries: int = 1,
    ) -> None:
        if "@sha256:" not in pinned_image:
            raise ValueError("Docker verifier requires an immutable sha256 repository digest")
        self.spec = spec
        self.pinned_image = pinned_image
        self.cli = cli or SubprocessDockerCLI()
        self.timeout = timeout
        self.work_timeout = work_timeout if work_timeout is not None else timeout
        self.max_chars = max_chars
        self.preflight_retries = max(0, preflight_retries)
        self.ptp_retries = max(0, ptp_retries)
        self._eval_containers: list[str] = []
        self._work_containers: list[str] = []
        self._protected: ProtectedFiles | None = None
        self._base_sha: str | None = None

    def protect(self, workspace: Path) -> None:
        self._protected = ProtectedFiles(workspace, self.spec.protected_files)
        parsed = _git(workspace, "rev-parse", "HEAD")
        if parsed.returncode == 0:
            self._base_sha = parsed.stdout.strip()

    def _container_name(self, phase: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", self.spec.task_id)[:40]
        return f"evoci-{safe}-{phase}-{uuid4().hex[:10]}"

    def _start_eval_container(self, phase: str) -> str:
        """Start the official image as-is. Never bind-mounts /testbed."""

        name = self._container_name(f"eval-{phase}")
        _checked(
            self.cli,
            [
                "run",
                "--detach",
                "--name",
                name,
                "--workdir",
                "/testbed",
                "--env",
                f"PATH={_TESTBED_PATH}",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "LANG=en_US.UTF-8",
                "--env",
                "LANGUAGE=en_US:en",
                "--env",
                "LC_ALL=en_US.UTF-8",
                "--entrypoint",
                "sleep",
                self.pinned_image,
                "infinity",
            ],
            timeout=120,
            operation=f"start {phase} eval container",
        )
        self._eval_containers.append(name)
        return name

    def start_work_container(self, workspace: Path) -> str:
        """Agent work container. Bind-mounts the workspace; not used for scoring."""

        if self._base_sha is None:
            parsed = _git(workspace, "rev-parse", "HEAD")
            if parsed.returncode == 0:
                self._base_sha = parsed.stdout.strip()
        name = self._container_name("work")
        _checked(
            self.cli,
            [
                "run",
                "--detach",
                "--name",
                name,
                "--mount",
                f"type=bind,src={workspace.resolve()},dst=/testbed",
                "--workdir",
                "/testbed",
                "--env",
                f"PATH={_TESTBED_PATH}",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--entrypoint",
                "sleep",
                self.pinned_image,
                "infinity",
            ],
            timeout=120,
            operation="start work container",
        )
        self._work_containers.append(name)
        return name

    def _remove(self, name: str) -> None:
        self.cli.run(["rm", "--force", name], timeout=60)
        if name in self._eval_containers:
            self._eval_containers.remove(name)
        if name in self._work_containers:
            self._work_containers.remove(name)

    def _exec(
        self, container: str, command: list[str], *, timeout: float | None = None
    ) -> BenchmarkCommandResult:
        completed = self.cli.run(
            [
                "exec",
                "--workdir",
                "/testbed",
                "--env",
                "CI=1",
                "--env",
                f"PATH={_TESTBED_PATH}",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "LANG=en_US.UTF-8",
                "--env",
                "LANGUAGE=en_US:en",
                "--env",
                "LC_ALL=en_US.UTF-8",
                container,
                *command,
            ],
            timeout=timeout if timeout is not None else self.timeout,
        )
        return BenchmarkCommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout[-self.max_chars :],
            stderr=completed.stderr[-self.max_chars :],
            timed_out=completed.returncode == 124,
        )

    def _copy_text(self, container: str, content: str, destination: str) -> None:
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            handle.write(content.encode("utf-8"))
            path = handle.name
        try:
            _checked(
                self.cli,
                ["cp", path, f"{container}:{destination}"],
                timeout=60,
                operation=f"copy {destination}",
            )
        finally:
            Path(path).unlink(missing_ok=True)

    def _apply_patch(
        self, container: str, content: str, destination: str
    ) -> BenchmarkCommandResult:
        self._copy_text(container, content, destination)
        applied = self._exec(
            container, ["git", "-C", "/testbed", "apply", "--whitespace=nowarn", destination]
        )
        if applied.exit_code == 0:
            return applied
        fallback = self._exec(
            container,
            ["patch", "--batch", "--forward", "-p1", "-d", "/testbed", "-i", destination],
        )
        if fallback.exit_code == 0:
            return fallback
        raise PatchApplyError(
            f"patch apply failed for {destination}: "
            f"{(fallback.stderr or fallback.stdout or applied.stderr or applied.stdout).strip()}"
        )

    def _reinstall(self, container: str) -> BenchmarkCommandResult:
        return self._exec(
            container,
            [
                "python",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--disable-pip-version-check",
                "-e",
                ".",
            ],
        )

    def _parse_nodes(self, log: str, nodes: list[str]) -> dict[str, str]:
        parsed = (
            parse_log_django(log)
            if nodes and all(is_django_node(n) for n in nodes)
            else parse_log_pytest(log)
        )
        return {node: lookup_node_status(parsed, node) for node in nodes}

    def _read_junit(
        self, container: str, remote: str, nodes: list[str]
    ) -> dict[str, str] | None:
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            host = Path(handle.name)
        try:
            copied = self.cli.run(["cp", f"{container}:{remote}", str(host)], timeout=60)
            if copied.returncode != 0 or host.stat().st_size == 0:
                return None
            cases = parse_junit_xml(host.read_text(encoding="utf-8", errors="replace"))
            if not cases:
                return None
            return {node: match_junit_status(node, cases) for node in nodes}
        except (OSError, ET.ParseError):
            return None
        finally:
            host.unlink(missing_ok=True)

    def _should_retry_ptp(self, score: SuiteScore) -> bool:
        """Retry only executed PTP assertion flakes after FTP already passed."""

        if score.infra_reason or not score.all_ftp_passed or not score.ptp_results:
            return False
        if any(status not in {"PASSED", "FAILED"} for status in score.ptp_results.values()):
            return False
        return any(status == "FAILED" for status in score.ptp_results.values())

    def _run_nodes(
        self, container: str, nodes: list[str]
    ) -> tuple[BenchmarkCommandResult, dict[str, str]]:
        django = bool(nodes) and all(is_django_node(node) for node in nodes)
        junit_remote = None if django else "/tmp/evoci-junit.xml"
        command = build_test_command(nodes, junit_xml=junit_remote)
        result = self._exec(container, command)
        statuses = self._parse_nodes(f"{result.stdout}\n{result.stderr}", nodes)
        if junit_remote is not None:
            junit_statuses = self._read_junit(container, junit_remote, nodes)
            if junit_statuses is not None:
                statuses = junit_statuses
        return result, statuses

    def _score_once(
        self,
        *,
        phase: str,
        candidate_patch: str = "",
        run_pass_to_pass: bool = True,
    ) -> SuiteScore:
        commands: list[BenchmarkCommandResult] = []
        container = self._start_eval_container(phase)
        try:
            if self.spec.test_patch.strip():
                commands.append(
                    self._apply_patch(container, self.spec.test_patch, "/tmp/test.patch")
                )
            django_eval = all(is_django_node(node) for node in self.spec.fail_to_pass)
            if candidate_patch.strip():
                commands.append(
                    self._apply_patch(container, candidate_patch, "/tmp/candidate.patch")
                )
                if not django_eval:
                    install_result = self._reinstall(container)
                    commands.append(install_result)
                    if install_result.exit_code != 0 or install_result.timed_out:
                        return SuiteScore(
                            commands=commands,
                            ftp_results={node: "MISSING" for node in self.spec.fail_to_pass},
                            ptp_results={},
                            infra_reason="editable reinstall failed",
                        )
            ptp_results: dict[str, str] = {}
            if django_eval and run_pass_to_pass and self.spec.pass_to_pass:
                combined = [*self.spec.fail_to_pass, *self.spec.pass_to_pass]
                result, statuses = self._run_nodes(container, combined)
                commands.append(result)
                ftp_results = {node: statuses[node] for node in self.spec.fail_to_pass}
                ptp_results = {node: statuses[node] for node in self.spec.pass_to_pass}
                infra = detect_infra_reason(result, statuses)
            else:
                ftp_result, ftp_results = self._run_nodes(container, self.spec.fail_to_pass)
                commands.append(ftp_result)
                infra = detect_infra_reason(ftp_result, ftp_results)
                if run_pass_to_pass and self.spec.pass_to_pass and infra is None:
                    ptp_result, ptp_results = self._run_nodes(
                        container, self.spec.pass_to_pass
                    )
                    commands.append(ptp_result)
                    infra = detect_infra_reason(ptp_result, ptp_results)
            return SuiteScore(
                commands=commands,
                ftp_results=ftp_results,
                ptp_results=ptp_results,
                infra_reason=infra,
            )
        except PatchApplyError as exc:
            return SuiteScore(
                commands=commands,
                ftp_results={node: "MISSING" for node in self.spec.fail_to_pass},
                ptp_results={},
                infra_reason=f"patch apply failed: {exc}",
            )
        except DockerError as exc:
            return SuiteScore(
                commands=commands,
                ftp_results={node: "MISSING" for node in self.spec.fail_to_pass},
                ptp_results={},
                infra_reason=str(exc),
            )
        finally:
            self._remove(container)

    def score_patches(
        self,
        *,
        phase: str,
        candidate_patch: str = "",
        run_pass_to_pass: bool = True,
    ) -> SuiteScore:
        attempts: list[EvalAttempt] = []
        commands: list[BenchmarkCommandResult] = []
        last: SuiteScore | None = None
        retries = self.ptp_retries if run_pass_to_pass else 0
        extra = 0
        while True:
            phase_name = phase if extra == 0 else f"{phase}-ptp-retry-{extra}"
            last = self._score_once(
                phase=phase_name,
                candidate_patch=candidate_patch,
                run_pass_to_pass=run_pass_to_pass,
            )
            commands.extend(last.commands)
            attempts.append(last.as_attempt(phase_name))
            extra += 1
            if extra > retries or not self._should_retry_ptp(last):
                break
        assert last is not None
        return replace(last, commands=commands, attempts=attempts)

    async def preflight(self, task: PreparedTask, workspace: Path) -> BenchmarkPreflightResult:
        del task, workspace
        if not self.spec.fail_to_pass:
            return BenchmarkPreflightResult(
                status="not_available", details="no FAIL_TO_PASS tests in dataset"
            )
        score = self.score_patches(phase="baseline", candidate_patch="", run_pass_to_pass=False)
        if score.infra_reason:
            return BenchmarkPreflightResult(
                status="infra_error",
                commands=score.commands,
                details=f"baseline infra_error: {score.infra_reason}",
                failure_class="infra",
                ftp_results=score.ftp_results,
            )
        if score.ftp_executed and score.any_ftp_failed:
            failed = sum(status == "FAILED" for status in score.ftp_results.values())
            return BenchmarkPreflightResult(
                status="reproduced",
                commands=score.commands,
                details=(
                    f"reproduced {failed} FAIL_TO_PASS test(s) in official image"
                ),
                ftp_results=score.ftp_results,
            )
        if score.all_ftp_passed:
            return BenchmarkPreflightResult(
                status="not_reproduced",
                commands=score.commands,
                details="FAIL_TO_PASS tests already passed before repair",
                ftp_results=score.ftp_results,
            )
        return BenchmarkPreflightResult(
            status="infra_error",
            commands=score.commands,
            details=f"FAIL_TO_PASS mixed baseline results: {score.ftp_results}",
            failure_class="infra",
            ftp_results=score.ftp_results,
        )

    async def verify(
        self, task: PreparedTask, workspace: Path, preflight: BenchmarkPreflightResult
    ) -> BenchmarkVerificationResult:
        del task
        if preflight.status == "infra_error":
            return BenchmarkVerificationResult(
                status="infra_error",
                commands=preflight.commands,
                details=preflight.details,
                failure_class="infra",
                ftp_results=preflight.ftp_results,
            )
        if preflight.status != "reproduced":
            return BenchmarkVerificationResult(
                status="not_available", details=f"benchmark oracle unavailable: {preflight.details}"
            )
        try:
            candidate = build_candidate_patch(workspace, self._base_sha, self.spec.protected_files)
        except ProtectedPatchError as exc:
            return BenchmarkVerificationResult(
                status="infra_error",
                details=str(exc),
                failure_class="protected_file",
            )
        except (UnsafePatchError, ValueError) as exc:
            return BenchmarkVerificationResult(
                status="infra_error",
                details=f"candidate patch could not be built: {exc}",
                failure_class="infra",
            )
        score = self.score_patches(phase="final", candidate_patch=candidate, run_pass_to_pass=True)
        if score.infra_reason:
            return BenchmarkVerificationResult(
                status="infra_error",
                commands=score.commands,
                details=f"final infra_error: {score.infra_reason}",
                failure_class="infra",
                ftp_results=score.ftp_results,
                ptp_results=score.ptp_results,
                attempts=score.attempts,
            )
        if score.all_ftp_passed and score.all_ptp_passed:
            return BenchmarkVerificationResult(
                status="passed",
                commands=score.commands,
                details="FAIL_TO_PASS and PASS_TO_PASS all passed",
                ftp_results=score.ftp_results,
                ptp_results=score.ptp_results,
                attempts=score.attempts,
            )
        return BenchmarkVerificationResult(
            status="failed",
            commands=score.commands,
            details=(
                "official-image tests failed after repair: "
                f"ftp={score.ftp_results} ptp={score.ptp_results}"
            ),
            failure_class="assertion",
            ftp_results=score.ftp_results,
            ptp_results=score.ptp_results,
            attempts=score.attempts,
        )

    def close(self) -> None:
        remaining = list(self._eval_containers) + list(self._work_containers)
        for name in remaining:
            self._remove(name)

    async def execute_agent_command(
        self, command: list[str], cwd: str = ".", network: bool = False
    ) -> CommandResult:
        del cwd, network
        if not self._work_containers:
            raise DockerError("Docker work container is not running")
        result = self._exec(self._work_containers[0], command, timeout=self.work_timeout)
        return CommandResult(
            argv=tuple(command),
            cwd="/testbed",
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
        )
