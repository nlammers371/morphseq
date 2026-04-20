from __future__ import annotations

from ..specs import ReducerSpec


_REGISTRY: dict[str, ReducerSpec] = {}


def register_reducer(reducer: ReducerSpec, *, overwrite: bool = False) -> ReducerSpec:
    if reducer.name in _REGISTRY and not overwrite:
        raise ValueError(f"Reducer {reducer.name!r} is already registered")
    _REGISTRY[reducer.name] = reducer
    return reducer


def get_reducer(reducer: str | ReducerSpec) -> ReducerSpec:
    if isinstance(reducer, ReducerSpec):
        return reducer
    try:
        return _REGISTRY[reducer]
    except KeyError as exc:
        raise KeyError(f"Unknown reducer {reducer!r}") from exc
