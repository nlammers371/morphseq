"""Reducer registry, built-ins, and factories for cross-bin reductions."""

from .builtins import ensure_builtin_reducers_registered
from .factories import (
    make_centered_reducer,
    make_group_centered_reducer,
    make_group_difference_reducer,
)
from .registry import get_reducer, register_reducer

ensure_builtin_reducers_registered()

__all__ = [
    "get_reducer",
    "register_reducer",
    "make_centered_reducer",
    "make_group_centered_reducer",
    "make_group_difference_reducer",
    "ensure_builtin_reducers_registered",
]
