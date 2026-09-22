from pathlib import Path

import pytest

from evoci.domain.models import CIFailure, RepoSpec
from evoci.learning_state import LEGACY_LEARNING_STATE_MESSAGE, LegacyLearningStateError
from evoci.memory.fingerprint import failure_fingerprint
from evoci.memory.models import Episode, LongTermFactCandidate, LongTermMemory
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import SQLiteMemoryStore


def test_long_term_memory_is_repository_scoped(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.add_long_term(
        LongTermMemory(
            id="repo-a",
            repository="org/a",
            content="pytest imports require src on PYTHONPATH",
            confidence=0.95,
            source_run_ids=["run-a"],
        )
    )
    store.add_long_term(
        LongTermMemory(
            id="repo-b",
            repository="org/b",
            content="pytest imports require src on PYTHONPATH",
            confidence=0.9,
            source_run_ids=["run-b"],
        )
    )
    failure = CIFailure(
        summary="pytest import failure",
        log_excerpt="pytest cannot import package",
        task_family="test",
    )

    repo_a = MemoryRetriever(store).retrieve(RepoSpec(owner="org", name="a"), failure)
    repo_b = MemoryRetriever(store).retrieve(RepoSpec(owner="org", name="b"), failure)

    assert {hit.memory_id for hit in repo_a.hits} == {"repo-a"}
    assert {hit.memory_id for hit in repo_b.hits} == {"repo-b"}
    store.close()


def test_failed_episode_search_keeps_outcome_and_counterevidence(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.add_episode(
        Episode(
            id="episode-fail",
            run_id="run-fail",
            repo="org/a",
            task_family="test",
            failure_summary="assertion still fails",
            root_cause="guessed missing import",
            attempts=2,
            hypotheses_attempted=["guessed missing import", "wrong operator"],
            verification_failures=["TEST_STILL_FAILS", "DO_NOT_REPEAT"],
            failure_reason="TEST_STILL_FAILS",
            success=False,
        )
    )
    hits = store.search_episodes("assertion still fails", repo="org/a", limit=3)
    assert hits
    content = hits[0].content
    assert "OUTCOME=failed" in content
    assert "TEST_STILL_FAILS" in content
    assert "DO_NOT_REPEAT" in content
    assert "HYPOTHESIS (unverified)" in content
    store.close()


def test_retrieval_truncation_keeps_outcome_prefix(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.add_episode(
        Episode(
            id="episode-long",
            run_id="run-long",
            repo="org/a",
            task_family="test",
            failure_summary="x" * 400,
            root_cause="unverified cause",
            attempts=1,
            verification_failures=["TEST_STILL_FAILS"],
            failure_reason="TEST_STILL_FAILS",
            success=False,
        )
    )
    failure = CIFailure(
        summary="assertion still fails xxxx",
        log_excerpt="TEST_STILL_FAILS",
        task_family="test",
    )
    retrieved = MemoryRetriever(store, context_limit_chars=80).retrieve(
        RepoSpec(owner="org", name="a"), failure
    )
    episode_hits = [hit for hit in retrieved.hits if hit.memory_id == "episode-long"]
    assert episode_hits
    assert episode_hits[0].content.startswith("OUTCOME=failed")
    store.close()


def test_episode_search_recalls_across_repos_with_same_repo_priority(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.add_episode(
        Episode(
            id="episode-a",
            run_id="run-a",
            repo="org/a",
            task_family="test",
            failure_summary="pytest import failed",
            root_cause="missing src path",
            attempts=1,
            successful_fix_summary="configure pythonpath",
            success=True,
        )
    )
    store.add_episode(
        Episode(
            id="episode-b",
            run_id="run-b",
            repo="org/b",
            task_family="test",
            failure_summary="pytest import error in another service",
            root_cause="module not on sys.path",
            attempts=1,
            successful_fix_summary="add src directory to pythonpath",
            success=True,
        )
    )
    hits_a = store.search_episodes("pytest import", repo="org/a", limit=3)
    hits_b = store.search_episodes("pytest import", repo="org/b", limit=3)
    assert [hit.memory_id for hit in hits_a] == ["episode-a", "episode-b"]
    assert [hit.memory_id for hit in hits_b] == ["episode-b", "episode-a"]
    store.close()


def test_same_fingerprint_returns_latest_failure_and_success(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    fingerprint = "abc123"
    store.add_episode(
        Episode(
            id="old-fail",
            run_id="run-old-fail",
            repo="org/example",
            task_family="test",
            failure_summary="assertion still fails",
            attempts=1,
            success=False,
            failure_fingerprint=fingerprint,
        )
    )
    store.add_episode(
        Episode(
            id="new-fail",
            run_id="run-new-fail",
            repo="org/example",
            task_family="test",
            failure_summary="assertion still fails",
            attempts=2,
            success=False,
            failure_fingerprint=fingerprint,
        )
    )
    store.add_episode(
        Episode(
            id="new-success",
            run_id="run-new-success",
            repo="org/example",
            task_family="test",
            failure_summary="assertion still fails",
            attempts=1,
            success=True,
            failure_fingerprint=fingerprint,
        )
    )
    hits = store.search_episodes_by_fingerprint(
        repo="org/example", fingerprint=fingerprint, limit=2
    )
    assert [hit.memory_id for hit in hits] == ["new-fail", "new-success"]
    store.close()


def test_unicode_error_summary_is_searchable(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.add_episode(
        Episode(
            id="zh-fail",
            run_id="run-zh",
            repo="org/a",
            task_family="test",
            failure_summary="类型错误 期望整数",
            attempts=1,
            success=False,
        )
    )
    hits = store.search_episodes("类型错误", repo="org/a", limit=3)
    assert hits
    assert hits[0].memory_id == "zh-fail"
    store.close()


def test_old_database_migrates_new_episode_columns(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            repo TEXT NOT NULL,
            task_family TEXT NOT NULL,
            failure_summary TEXT NOT NULL,
            root_cause TEXT,
            important_evidence TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            successful_fix_summary TEXT,
            tools_used TEXT NOT NULL,
            success INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO episodes VALUES (
            'legacy', 'run-legacy', 'org/a', 'test', 'old failure', NULL, '[]',
            1, NULL, '[]', 0, '2026-01-01T00:00:00+00:00'
        )
        """
    )
    connection.commit()
    connection.close()
    store = SQLiteMemoryStore(path)
    store = SQLiteMemoryStore(path)
    episode = store.get_episode("run-legacy")
    assert episode is not None
    assert episode.failure_class == "repair"
    assert episode.attempted_files == []
    assert episode.failure_fingerprint == ""
    store.close()


def test_empty_fts_query_still_allows_fingerprint_retrieval(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    fingerprint = "fp-empty-query"
    store.add_episode(
        Episode(
            id="exact",
            run_id="run-exact",
            repo="org/a",
            task_family="test",
            failure_summary="!!",
            attempts=1,
            success=False,
            failure_fingerprint=fingerprint,
        )
    )
    hits = store.search_episodes("??", repo="org/a", limit=3)
    exact = store.search_episodes_by_fingerprint(repo="org/a", fingerprint=fingerprint, limit=2)
    assert exact and exact[0].memory_id == "exact"
    del hits
    store.close()


def test_related_failure_uses_repo_fts_not_exact_fingerprint(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    repo = RepoSpec(owner="org", name="example")
    original = CIFailure(
        summary="tests/test_add.py::test_add failed AssertionError",
        log_excerpt="tests/test_add.py::test_add FAILED\nAssertionError: 1 != 2\n",
        failed_commands=[["python", "-m", "pytest", "tests/test_add.py::test_add"]],
        task_family="test",
    )
    related = CIFailure(
        summary="tests/test_add.py::test_add failed TypeError",
        log_excerpt="tests/test_add.py::test_add FAILED\nTypeError: bad operand\n",
        failed_commands=[["python", "-m", "pytest", "tests/test_add.py::test_add"]],
        task_family="test",
    )
    store.add_episode(
        Episode(
            id="add-assert",
            run_id="run-assert",
            repo=repo.full_name,
            task_family="test",
            failure_summary=original.summary,
            attempts=1,
            success=False,
            failure_fingerprint=failure_fingerprint(repo, original),
            attempted_fix_summaries=["change operator"],
            attempted_files=["src/app.py"],
            verification_failures=["AssertionError"],
        )
    )
    assert failure_fingerprint(repo, original) != failure_fingerprint(repo, related)
    exact = store.search_episodes_by_fingerprint(
        repo=repo.full_name,
        fingerprint=failure_fingerprint(repo, related),
        limit=2,
    )
    fts = store.search_episodes(related.summary, repo=repo.full_name, limit=3)
    retrieved = MemoryRetriever(store).retrieve(repo, related)
    assert exact == []
    assert any(hit.memory_id == "add-assert" for hit in fts)
    assert any(hit.memory_id == "add-assert" for hit in retrieved.hits)
    assert all("def add" not in hit.content for hit in retrieved.hits)
    store.close()


def test_long_term_memory_operation_key_is_exactly_once(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    memory = LongTermMemory(
        id="stable-memory",
        repository="org/a",
        content="replay side effects exactly once",
        confidence=0.9,
        source_run_ids=["run-1"],
    )

    assert store.add_long_term(memory, operation_key="ltm:run-1")
    assert not store.add_long_term(memory, operation_key="ltm:run-1")
    assert store.operation_result("ltm:run-1") == {
        "memory_id": "stable-memory",
        "created": True,
    }
    assert len(store.search_long_term("replay side effects", repository="org/a", limit=10)) == 1
    store.close()


def test_normalized_identical_facts_merge_source_run_ids(tmp_path: Path) -> None:
    from evoci.memory.consolidation import commit_candidate, fact_memory_id

    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    first = commit_candidate(
        store,
        LongTermFactCandidate(
            type="fact",
            content="The repository uses pytest.",
            confidence=0.8,
        ),
        run_id="run-1",
        repository="org/rpn",
    )
    second = commit_candidate(
        store,
        LongTermFactCandidate(
            type="fact",
            content="the repository uses pytest",
            confidence=0.9,
        ),
        run_id="run-2",
        repository="org/rpn",
    )
    expected_id = fact_memory_id("org/rpn", "The repository uses pytest.")
    assert first == expected_id
    assert second is None
    facts = store.list_long_term(repository="org/rpn", limit=8)
    assert len(facts) == 1
    assert facts[0].id == expected_id
    assert facts[0].source_run_ids == ["run-1", "run-2"]
    assert facts[0].confidence == 0.9
    assert facts[0].content == "The repository uses pytest."
    store.close()


def test_fact_memory_id_does_not_include_run_id() -> None:
    from evoci.memory.consolidation import fact_memory_id, normalize_fact_content

    left = fact_memory_id("org/rpn", "Uses pytest. Tests live in python_testcases/")
    right = fact_memory_id("org/rpn", "uses pytest. tests live in python_testcases/")
    other_repo = fact_memory_id("org/other", "Uses pytest. Tests live in python_testcases/")
    assert left == right
    assert left != other_repo
    assert normalize_fact_content("Uses pytest.") == "uses pytest"


def test_fact_extractor_cannot_choose_namespace() -> None:
    assert "namespace" not in LongTermFactCandidate.model_fields
    assert "repository" not in LongTermFactCandidate.model_fields
    candidate = LongTermFactCandidate(type="fact", content="uses pytest 3.2", confidence=0.8)
    assert candidate.content == "uses pytest 3.2"


def test_long_term_memory_fts_supports_unicode(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.add_long_term(
        LongTermMemory(
            id="zh-fact",
            repository="org/a",
            content="该仓库使用 pytest 3.2",
            confidence=0.9,
            source_run_ids=["run-zh"],
        )
    )
    hits = store.search_long_term("pytest", repository="org/a", limit=3)
    assert hits
    assert hits[0].memory_id == "zh-fact"
    store.close()


def test_legacy_memory_schema_requires_fresh_state(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE semantic_memories (id TEXT PRIMARY KEY, namespace TEXT, content TEXT)"
    )
    connection.commit()
    connection.close()
    with pytest.raises(LegacyLearningStateError, match=LEGACY_LEARNING_STATE_MESSAGE):
        SQLiteMemoryStore(path)
