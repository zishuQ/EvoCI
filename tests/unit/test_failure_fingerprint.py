from evoci.domain.models import CIFailure, RepoSpec
from evoci.memory.fingerprint import failure_fingerprint, normalize_unstable_text


def _failure(**updates: object) -> CIFailure:
    payload = {
        "summary": "test_add failed with AssertionError",
        "log_excerpt": (
            "tests/test_add.py::test_add FAILED\n"
            "AssertionError: 1 != 2\n"
            "File src/app.py, line 12, in add\n"
        ),
        "failed_commands": [["python", "-m", "pytest", "tests/test_add.py::test_add"]],
        "task_family": "test",
        "workflow_path": ".github/workflows/ci.yml",
    }
    payload.update(updates)
    return CIFailure.model_validate(payload)


def test_same_fault_with_noisy_logs_shares_fingerprint() -> None:
    repo = RepoSpec(owner="org", name="example")
    base = _failure()
    noisy = _failure(
        log_excerpt=(
            "\x1b[31m2026-09-20T12:00:01.123Z\x1b[0m tests/test_add.py::test_add FAILED\n"
            "AssertionError: 1 != 2\n"
            "File /tmp/pytest-of-ci/pytest-12/workspace/src/app.py, line 44, in add\n"
            "uuid=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee addr=0x7fabc1234567\n"
            "ran in 1.23s on localhost:54321 commit abcdef0123456789abcdef0123456789abcdef01\n"
        )
    )
    other_run = _failure()
    assert failure_fingerprint(repo, base) == failure_fingerprint(repo, noisy)
    assert failure_fingerprint(repo, base) == failure_fingerprint(repo, other_run)


def test_run_id_and_task_id_do_not_affect_fingerprint() -> None:
    repo = RepoSpec(owner="org", name="example")
    failure = _failure()
    assert "task_id" not in normalize_unstable_text(failure.log_excerpt)
    assert failure_fingerprint(repo, failure) == failure_fingerprint(
        RepoSpec(owner="org", name="example", base_commit="deadbeef"), failure
    )


def test_different_command_or_exception_changes_fingerprint() -> None:
    repo = RepoSpec(owner="org", name="example")
    base = failure_fingerprint(repo, _failure())
    command = failure_fingerprint(
        repo,
        _failure(failed_commands=[["python", "-m", "pytest", "tests/test_sub.py::test_sub"]]),
    )
    exception = failure_fingerprint(
        repo,
        _failure(
            summary="test_add failed with TypeError",
            log_excerpt="tests/test_add.py::test_add FAILED\nTypeError: unsupported operand\n",
        ),
    )
    test_id = failure_fingerprint(
        repo,
        _failure(
            summary="test_mul failed with AssertionError",
            log_excerpt="tests/test_mul.py::test_mul FAILED\nAssertionError: 1 != 2\n",
        ),
    )
    other_repo = failure_fingerprint(RepoSpec(owner="other", name="example"), _failure())
    assert len({base, command, exception, test_id, other_repo}) == 5
