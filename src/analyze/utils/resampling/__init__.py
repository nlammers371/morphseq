"""
Resampling framework.

Unified engine for bootstrap, subsample, and permutation resampling with
deterministic SeedSequence-based RNG, unit-aware perturbation, streaming
reducers, and post-hoc aggregation.

Usage::

    from analyze.utils.resampling import resample

    spec = resample.indices(replacement=False, frac=0.8)
    stat = resample.statistic("silhouette", scorer_fn)
    out  = resample.run(data, spec, stat, n_iters=100, seed=42)
    summary = resample.aggregate(out)

    # Semantic aliases
    spec = resample.bootstrap(size=2500)
    spec = resample.subsample(frac=0.8)
    spec = resample.permute_labels(within="time_bin")
    spec = resample.permute_groups(a="X1", b="X2")

    # Raw callable shorthand
    out = resample.run(data, spec, scorer_fn, n_iters=100, seed=42)

.. important::

   This is a clean SeedSequence break. Old code and new code will produce
   different (but equally valid) null distributions for the same seed.
"""

from typing import Callable, List, Optional

from ._spec import (
    ResampleSpec,
    IndicesParams,
    LabelsParams,
    GroupsParams,
)
from ._statistic import Statistic
from ._engine import run, ResampleRun
from ._aggregator import aggregate, BootstrapSummary, PermutationSummary
from ._preflight import preflight
from ._reducer import PermutationReducer


# ── Factory functions ────────────────────────────────────────────────

def indices(
    *,
    replacement: bool,
    size: Optional[int] = None,
    frac: Optional[float] = None,
    unit: Optional[str] = None,
    within: Optional[str] = None,
) -> ResampleSpec:
    """Create an index-based resampling spec.

    Parameters
    ----------
    replacement : bool
        Whether to sample with replacement.
    size : int, optional
        Explicit draw count.
    frac : float, optional
        Fraction of population to draw.
    unit : str, optional
        Column name for unit-level resampling (e.g., ``"embryo_id"``).
    within : str, optional
        Column name for stratified resampling (e.g., ``"time_bin"``).
    """
    return ResampleSpec(
        kind="indices",
        params=IndicesParams(replacement=replacement, size=size, frac=frac),
        unit_key=unit,
        within_key=within,
    )


def labels(
    *,
    within: Optional[str] = None,
    unit: Optional[str] = None,
) -> ResampleSpec:
    """Create a label-permutation spec.

    Parameters
    ----------
    within : str, optional
        Shuffle labels within strata.
    unit : str, optional
        Shuffle at unit level, then broadcast to rows.
    """
    return ResampleSpec(
        kind="labels",
        params=LabelsParams(),
        unit_key=unit,
        within_key=within,
    )


def groups(
    *,
    a: str = "X1",
    b: str = "X2",
) -> ResampleSpec:
    """Create a group-permutation (pool-and-redistribute) spec.

    Parameters
    ----------
    a : str, default="X1"
        Key for the first group in the data dict.
    b : str, default="X2"
        Key for the second group in the data dict.
    """
    return ResampleSpec(
        kind="groups",
        params=GroupsParams(a_key=a, b_key=b),
    )


# ── Semantic aliases ─────────────────────────────────────────────────

def bootstrap(
    *,
    size: Optional[int] = None,
    frac: Optional[float] = None,
    unit: Optional[str] = None,
    within: Optional[str] = None,
) -> ResampleSpec:
    """Alias for ``indices(replacement=True, ...)``."""
    return indices(replacement=True, size=size, frac=frac, unit=unit, within=within)


def subsample(
    *,
    size: Optional[int] = None,
    frac: Optional[float] = None,
    unit: Optional[str] = None,
    within: Optional[str] = None,
) -> ResampleSpec:
    """Alias for ``indices(replacement=False, ...)``."""
    return indices(replacement=False, size=size, frac=frac, unit=unit, within=within)


def permute_labels(
    *,
    within: Optional[str] = None,
    unit: Optional[str] = None,
) -> ResampleSpec:
    """Alias for ``labels(within=..., unit=...)``."""
    return labels(within=within, unit=unit)


def permute_groups(
    *,
    a: str = "X1",
    b: str = "X2",
) -> ResampleSpec:
    """Alias for ``groups(a=..., b=...)``."""
    return groups(a=a, b=b)


def statistic(
    name: str,
    fn: Callable,
    *,
    description: Optional[str] = None,
    outputs: Optional[List[str]] = None,
    default_alternative: Optional[str] = None,
    is_nonnegative: Optional[bool] = None,
) -> Statistic:
    """Create a ``Statistic`` wrapper.

    Parameters
    ----------
    name : str
        Human-readable name.
    fn : callable
        ``fn(data, rng) -> scalar | array | dict``.
    description : str, optional
        Longer description.
    outputs : list of str, optional
        Dict keys for multi-output scorers.
    default_alternative : str, optional
        Default tail: ``"greater"`` | ``"less"`` | ``"two-sided"``.
    is_nonnegative : bool, optional
        True for distance-like statistics.
    """
    return Statistic(
        name=name,
        fn=fn,
        description=description,
        outputs=outputs,
        default_alternative=default_alternative,
        is_nonnegative=is_nonnegative,
    )


__all__ = [
    # Core types
    "ResampleSpec",
    "IndicesParams",
    "LabelsParams",
    "GroupsParams",
    "Statistic",
    "ResampleRun",
    "BootstrapSummary",
    "PermutationSummary",
    "PermutationReducer",
    # Factories
    "indices",
    "labels",
    "groups",
    "statistic",
    # Aliases
    "bootstrap",
    "subsample",
    "permute_labels",
    "permute_groups",
    # Engine
    "run",
    "aggregate",
    "preflight",
]
