"""Shared errors for incompatible on-disk learning state."""

from __future__ import annotations

LEGACY_LEARNING_STATE_MESSAGE = (
    "Legacy learning state detected. Start with a fresh state or campaign directory."
)


class LegacyLearningStateError(RuntimeError):
    def __init__(self, message: str = LEGACY_LEARNING_STATE_MESSAGE) -> None:
        super().__init__(message)
