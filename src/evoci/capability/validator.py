"""Candidate structure, secret, safety, syntax, and isolated test validation."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from evoci.capability.models import RegisteredSkill, ValidationResult
from evoci.capability.registry import CapabilityRegistry

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
        if not errors:
            test_error, tests_run = self._run_tests(record)
            if test_error:
                errors.append(test_error)
        return ValidationResult(passed=not errors, errors=errors, tests_run=tests_run)

    def validate_to_trial(self, skill_id: str, version: int) -> ValidationResult:
        result = self.validate(skill_id, version)
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
            if file.path.endswith(".sh") and any(
                token in content for token in ("sudo", "curl", "wget", "rm -rf", "${HOME}")
            ):
                errors.append(f"unsafe shell in {file.path}")
        return errors

    def _run_tests(self, record: RegisteredSkill) -> tuple[str | None, int]:
        package = Path(record.package_path)
        tests = list((package / "tests").glob("test*.py")) if (package / "tests").is_dir() else []
        if not tests:
            return None, 0
        with tempfile.TemporaryDirectory(prefix="evoci-skill-test-") as temporary:
            copied = Path(temporary) / "package"
            shutil.copytree(package, copied)
            for item in copied.rglob("*"):
                if item.is_file():
                    item.chmod(item.stat().st_mode | 0o200)
            environment = {
                key: value
                for key, value in os.environ.items()
                if key in {"PATH", "LANG", "LC_ALL", "TMPDIR"}
            }
            environment["PYTHONPATH"] = str(copied)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                    cwd=copied,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return "skill tests timed out", len(tests)
            if result.returncode != 0:
                output = (result.stdout + result.stderr)[-4_000:]
                return f"skill tests failed: {output}", len(tests)
        return None, len(tests)
