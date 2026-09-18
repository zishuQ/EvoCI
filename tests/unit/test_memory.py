from pathlib import Path

from evoci.domain.models import CIFailure, RepoSpec
from evoci.memory.models import Episode, SemanticMemory
from evoci.memory.retrieval import MemoryRetriever
from evoci.memory.store import SQLiteMemoryStore


def test_memory_retrieval_respects_repo_namespace(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    store.add_semantic(
        SemanticMemory(
            id="repo-a",
            namespace="repo:org/a",
            content="pytest imports require src on PYTHONPATH",
            importance=0.9,
            confidence=0.95,
            source_run_ids=["run-a"],
        )
    )
    store.add_semantic(
        SemanticMemory(
            id="global",
            namespace="global:ci",
            content="pytest failures should be reproduced with the targeted test",
            importance=0.8,
            confidence=0.9,
            source_run_ids=["run-global"],
        )
    )
    failure = CIFailure(
        summary="pytest import failure",
        log_excerpt="pytest cannot import package",
        task_family="test",
    )

    repo_a = MemoryRetriever(store).retrieve(RepoSpec(owner="org", name="a"), failure)
    repo_b = MemoryRetriever(store).retrieve(RepoSpec(owner="org", name="b"), failure)

    assert {hit.memory_id for hit in repo_a.hits} == {"repo-a", "global"}
    assert {hit.memory_id for hit in repo_b.hits} == {"global"}
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


def test_semantic_memory_operation_key_is_exactly_once(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    memory = SemanticMemory(
        id="stable-memory",
        namespace="global:ci",
        content="replay side effects exactly once",
        importance=0.8,
        confidence=0.9,
        source_run_ids=["run-1"],
    )

    assert store.add_semantic(memory, operation_key="semantic:run-1")
    assert not store.add_semantic(memory, operation_key="semantic:run-1")
    assert store.operation_result("semantic:run-1") == {"memory_id": "stable-memory"}
    assert (
        len(store.search_semantic("replay side effects", namespaces=["global:ci"], limit=10)) == 1
    )
    store.close()
