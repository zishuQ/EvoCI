"""Explicit feature switches for fair benchmark ablations."""

from __future__ import annotations

from dataclasses import dataclass

from evoci.benchmark.models import BenchmarkVariant


@dataclass(frozen=True, slots=True)
class VariantFeatures:
    multi_agent: bool
    long_term_memory: bool
    capabilities: bool


FEATURES: dict[BenchmarkVariant, VariantFeatures] = {
    "single": VariantFeatures(False, False, False),
    "multi": VariantFeatures(True, False, False),
    "multi-memory": VariantFeatures(True, True, False),
    "evo": VariantFeatures(True, True, True),
}


def variant_features(variant: BenchmarkVariant) -> VariantFeatures:
    return FEATURES[variant]
