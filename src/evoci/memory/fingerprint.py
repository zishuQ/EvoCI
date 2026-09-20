"""Stable fingerprints for CI failures, independent of run_id and task_id."""

from __future__ import annotations

import hashlib
import json
import re

from evoci.domain.models import CIFailure, RepoSpec

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_MEMORY_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]+\b")
_ISO_TIMESTAMP = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_NAMED_TIMESTAMP = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"
)
_DURATION = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds|m|min|minutes)\b", re.I)
_COMMIT_SHA = re.compile(r"\b(?:commit\s+)?[0-9a-f]{40}\b")
_HOST_PORT = re.compile(r"\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d{2,5})\b")
_TMP_PATH = re.compile(
    r"(?:/tmp|/var/folders|/private/var/folders|C:\\Users\\[^\\]+\\AppData\\Local\\Temp)"
    r"[/\\][^\s:,]+"
)
_PYTEST_TMP = re.compile(r"pytest-of-[^\s/\\]+")
_ABS_PREFIX = re.compile(r"(?:/home|/Users|/usr/local|/opt)/[^\s:,]*")
_CI_ID = re.compile(r"\b(?:job|run|build|github_run)[-_#]?\d+\b", re.I)
_LINE_IN = re.compile(r", line \d+")
_FILE_LINE = re.compile(r"(\.py):(\d+)(:)")
_EXCEPTION = re.compile(r"\b(?:[A-Z][A-Za-z0-9_]*)(?:Error|Exception|Failure|Warning)\b")
_PYTEST_NODE = re.compile(r"[\w./\\-]+\.py::[\w]+(?:::[\w]+)?")
_TEST_NAME = re.compile(r"\btest_[A-Za-z0-9_]+\b")
_ERROR_LINE = re.compile(
    r"(?:[A-Z][A-Za-z0-9_]*)(?:Error|Exception|Failure|Warning):[^\n]{0,200}"
)


def normalize_unstable_text(text: str) -> str:
    """Strip run-specific noise so the same fault yields a stable signature."""

    normalized = _ANSI.sub("", text)
    normalized = _UUID.sub("<uuid>", normalized)
    normalized = _MEMORY_ADDRESS.sub("<addr>", normalized)
    normalized = _ISO_TIMESTAMP.sub("<timestamp>", normalized)
    normalized = _NAMED_TIMESTAMP.sub("<timestamp>", normalized)
    normalized = _DURATION.sub("<duration>", normalized)
    normalized = _COMMIT_SHA.sub("<commit>", normalized)
    normalized = _HOST_PORT.sub("<host>:<port>", normalized)
    normalized = _TMP_PATH.sub("<tmp>", normalized)
    normalized = _PYTEST_TMP.sub("<pytest_tmp>", normalized)
    normalized = _ABS_PREFIX.sub("<abs>/", normalized)
    normalized = _CI_ID.sub("<ci-id>", normalized)
    normalized = _LINE_IN.sub(", line N", normalized)
    normalized = _FILE_LINE.sub(r"\1:N\3", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def stable_error_signature(log_excerpt: str, summary: str = "") -> dict[str, list[str]]:
    blob = normalize_unstable_text(f"{summary}\n{log_excerpt}")
    exceptions = list(dict.fromkeys(_EXCEPTION.findall(blob)))
    test_ids = list(dict.fromkeys(_PYTEST_NODE.findall(blob) + _TEST_NAME.findall(blob)))
    files = list(
        dict.fromkeys(node.split("::")[0].replace("\\", "/") for node in test_ids if ".py" in node)
    )
    messages: list[str] = []
    for item in _ERROR_LINE.findall(blob):
        stable = re.split(r"\s+(?:File|uuid=|addr=|ran in)\b", item, maxsplit=1)[0]
        messages.append(normalize_unstable_text(stable))
    return {
        "exception_types": exceptions,
        "files": files[:12],
        "messages": messages[:8],
        "test_ids": test_ids[:12],
    }


def failure_fingerprint(repo: RepoSpec, failure: CIFailure) -> str:
    payload = {
        "failed_commands": list(failure.failed_commands),
        "repo": repo.full_name,
        "signature": stable_error_signature(failure.log_excerpt, failure.summary),
        "summary": normalize_unstable_text(failure.summary),
        "task_family": failure.task_family,
        "workflow_path": failure.workflow_path,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
