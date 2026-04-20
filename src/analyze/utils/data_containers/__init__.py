"""Typed data-container utilities for MorphSeq bin-based analyses.

This package currently exposes a lean BinObject API for level-aware feature
attachment with strict grain validation.
"""

from .bin_object import BinObject, LevelCollection
from .specs import FeatureSpec

__all__ = [
    "BinObject",
    "FeatureSpec",
    "LevelCollection",
]
