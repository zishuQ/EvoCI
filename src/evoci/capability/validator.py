"""Candidate structure, secret, safety, syntax, and isolated test validation."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from evoci.capability.models import RegisteredSkill, ValidationResult
from evoci.capability.registry import CapabilityRegistry
from evoci.tools.policy import PolicyViolation
from evoci.tools.shell import CommandRunner, run_cancellable, run_grouped_subprocess

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
)
FORBIDDEN_TEXT = (
    "rm -rf /",
    "sudo ",
    "curl | sh",
    "../",
    "/etc/",
    "~/.ssh",
    '".env"',
    "'.env'",
)
FORBIDDEN_IMPORTS = {"subprocess", "socket", "requests", "httpx", "urllib", "ftplib"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "os.system", "os.popen"}
_PYTEST_PLUGIN = """from __future__ import annotations

import json
from pathlib import Path

_COUNTS = {"collected": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}


def pytest_collection_modifyitems(items: list[object]) -> None:
    _COUNTS["collected"] = len(items)


def pytest_runtest_logreport(report: object) -> None:
    when = getattr(report, "when", "")
    if when == "setup" and getattr(report, "failed", False):
        _COUNTS["errors"] += 1
    if when == "call":
        if getattr(report, "passed", False):
            _COUNTS["passed"] += 1
        elif getattr(report, "failed", False):
            _COUNTS["failed"] += 1
        elif getattr(report, "skipped", False):
            _COUNTS["skipped"] += 1


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    del session, exitstatus
    Path("evoci-pytest-report.json").write_text(json.dumps(_COUNTS), encoding="utf-8")
"""


def _python_policy(source: str, path: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    errors.append(f"{path}: forbidden import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                errors.append(f"{path}: forbidden import {node.module}")
        elif isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                name = f"{node.func.value.id}.{node.func.attr}"
            if name in FORBIDDEN_CALLS:
                errors.append(f"{path}: forbidden call {name}")
    return errors


class CandidateValidator:
    def __init__(self, registry: CapabilityRegistry, *, timeout: float = 10.0) -> None:
        self.registry = registry
        self.timeout = timeout

    def validate(self, skill_id: str, version: int) -> ValidationResult:
        record = self.registry.get(skill_id, version)
        if record is None:
            return ValidationResult(passed=False, errors=["skill not found"])
        errors = self._validate_record(record)
        tests_run = 0
        test_files = 0
        behavior_verified = False
        if not errors:
            test_error, tests_run, test_files, behavior_verified = self._run_behavior(record)
            if test_error:
                errors.append(test_error)
        return ValidationResult(
            passed=not errors,
            errors=errors,
            tests_run=tests_run,
            test_files=test_files,
            behavior_verified=behavior_verified and not errors,
        )

    def validate_to_trial(self, skill_id: str, version: int) -> ValidationResult:
        result = self.validate(skill_id, version)
        self.registry.transition(skill_id, version, "trial" if result.passed else "rejected")
        return result

    async def avalidate_to_trial(self, skill_id: str, version: int) -> ValidationResult:
        # Transition on the caller's thread only after validation has fully completed.
        # Cancellation joins subprocess cleanup and leaves the candidate unpromoted.
        result = await run_cancellable(self.validate, skill_id, version)
        self.registry.transition(skill_id, version, "trial" if result.passed else "rejected")
        return result

    def _validate_record(self, record: RegisteredSkill) -> list[str]:
        package = Path(record.package_path).resolve()
        errors: list[str] = []
        required = {"SKILL.md", "manifest.json"}
        if not all((package / path).is_file() for path in required):
            errors.append("package is missing SKILL.md or manifest.json")
        for file in record.manifest.files:
            target = (package / file.path).resolve()
            if package not in target.parents or not target.is_file():
                errors.append(f"unsafe or missing file: {file.path}")
                continue
            content = target.read_text(encoding="utf-8", errors="replace")
            if hashlib.sha256(content.encode()).hexdigest() != file.sha256:
                errors.append(f"immutable hash mismatch: {file.path}")
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                errors.append(f"possible secret in {file.path}")
            if any(token in content for token in FORBIDDEN_TEXT):
                errors.append(f"unsafe path or command in {file.path}")
            if file.path.endswith(".py"):
                errors.extend(_python_policy(content, file.path))
            if file.path.startswith("scripts/") and not file.path.endswith(".py"):
                errors.append(f"unsupported_runner: {file.path}")
            if file.path.endswith(".sh"):
                errors.append(f"unsupported_runner: {file.path}")
        return errors

    def _run_behavior(
        self, record: RegisteredSkill
    ) -> tuple[str | None, int, int, bool]:
        package = Path(record.package_path)
        tests_dir = package / "tests"
        test_files = list(tests_dir.rglob("*.py")) if tests_dir.is_dir() else []
        commands = list(record.manifest.verification_commands)
        if not test_files and not commands:
            return None, 0, 0, False
        with tempfile.TemporaryDirectory(prefix="evoci-skill-test-") as temporary:
            copied = Path(temporary) / "package"
            shutil.copytree(package, copied)
            for item in copied.rglob("*"):
                if item.is_file():
                    item.chmod(item.stat().st_mode | 0o200)
            tests_run = 0
            if test_files:
                error, tests_run = self._run_pytest(copied, len(test_files))
                if error:
                    return error, tests_run, len(test_files), False
            if commands:
                error = self._run_verification_commands(copied, commands)
                if error:
                    return error, tests_run, len(test_files), False
        return None, tests_run, len(test_files), True

    def _run_pytest(self, copied: Path, file_count: int) -> tuple[str | None, int]:
        (copied / "evoci_skill_pytest_plugin.py").write_text(_PYTEST_PLUGIN, encoding="utf-8")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
        }
        environment["PYTHONPATH"] = str(copied)
        argv = [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            f"--rootdir={copied}",
            "-p",
            "evoci_skill_pytest_plugin",
            "-o",
            "addopts=",
            "-q",
            "--tb=short",
        ]
        completed = run_grouped_subprocess(
            argv,
            cwd=copied,
            env=environment,
            timeout=self.timeout,
            max_chars=8_000,
        )
        report_path = copied / "evoci-pytest-report.json"
        counts = {"collected": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        if report_path.is_file():
            counts.update(json.loads(report_path.read_text(encoding="utf-8")))
        executed = int(counts["passed"]) + int(counts["failed"]) + int(counts["errors"])
        collected_count = int(counts["collected"])
        if completed.timed_out:
            return "skill tests timed out", executed or file_count
        if collected_count <= 0 and completed.exit_code != 0:
            output = (completed.stdout + completed.stderr)[-4_000:]
            return f"test collection failed: {output}", 0
        if collected_count <= 0:
            return "no tests collected", 0
        if completed.exit_code != 0 or counts["failed"] or counts["errors"]:
            output = (completed.stdout + completed.stderr)[-4_000:]
            return f"skill tests failed: {output}", executed or collected_count
        if executed <= 0:
            return "zero tests executed", 0
        return None, executed

    def _run_verification_commands(self, copied: Path, commands: list[list[str]]) -> str | None:
        runner = CommandRunner(copied, timeout=self.timeout)
        for command in commands:
            try:
                completed = runner.run_sync(command, extra_env={"CI": "1"})
            except (OSError, PolicyViolation, TypeError, ValueError) as exc:
                return f"verification command could not run: {type(exc).__name__}: {exc}"
            if completed.timed_out:
                return f"verification command timed out: {' '.join(command)}"
            if completed.exit_code != 0:
                detail = completed.stderr or f"exit code {completed.exit_code}"
                return f"verification command failed: {' '.join(command)}: {detail[-2_000:]}"
        return None
