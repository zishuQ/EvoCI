from __future__ import annotations

import pytest

import evoci.graph.builder as graph_builder
from evoci.domain.models import SkillCatalogEntry
from evoci.graph.state import LegacyOrchestrationError
from evoci.runtime.checkpoints import _serializer


def test_skill_catalog_entry_msgpack_round_trip() -> None:
    entry = SkillCatalogEntry(
        skill_id="checkpoint-skill",
        name="Checkpoint skill",
        description="serializer fixture",
    )

    serializer = _serializer()
    encoded = serializer.dumps_typed(entry)
    restored = serializer.loads_typed(encoded)

    assert isinstance(restored, SkillCatalogEntry)
    assert restored == entry


def test_legacy_dict_skill_catalog_is_upgraded() -> None:
    restored = graph_builder._coerce_skill_catalog(
        [
            {
                "skill_id": "legacy-skill",
                "name": "Legacy skill",
                "description": "stored as a plain dictionary",
            }
        ]
    )

    assert restored == [
        SkillCatalogEntry(
            skill_id="legacy-skill",
            name="Legacy skill",
            description="stored as a plain dictionary",
        )
    ]


@pytest.mark.parametrize(
    "raw",
    [
        {"skill_id": "not-a-list"},
        [{"name": "missing id", "description": "invalid"}],
        [object()],
    ],
)
def test_malformed_skill_catalog_is_not_silently_dropped(raw: object) -> None:
    with pytest.raises(
        LegacyOrchestrationError,
        match="malformed checkpoint skill_catalog",
    ):
        graph_builder._coerce_skill_catalog(raw)
