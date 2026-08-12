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


def test_episode_search_is_repo_scoped(tmp_path: Path) -> None:
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
    assert store.search_episodes("pytest import", repo="org/a", limit=3)
    assert store.search_episodes("pytest import", repo="org/b", limit=3) == []
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
