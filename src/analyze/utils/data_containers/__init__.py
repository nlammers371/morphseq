"""Typed data-container utilities for MorphSeq bin-based analyses.

This package introduces a BinObject-centered API with explicit levels,
declarative reducer contracts, and auditable cross-bin reductions.
"""

from .bin_object import BinObject, LevelCollection
from .reducers import (
    get_reducer,
    make_centered_reducer,
    make_group_centered_reducer,
    make_group_difference_reducer,
    register_reducer,
)
from .reports import SupportReport
from .specs import FeatureSpec, InputRef, ReducerSpec

__all__ = [
    "BinObject",
    "FeatureSpec",
    "InputRef",
    "LevelCollection",
    "make_centered_reducer",
    "make_group_centered_reducer",
    "make_group_difference_reducer",
    "ReducerSpec",
    "SupportReport",
    "get_reducer",
    "register_reducer",
]
