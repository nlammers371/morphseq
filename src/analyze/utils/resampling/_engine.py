"""
Resampling engine.

Implements ``run()``: the core iteration loop with SeedSequence-based
deterministic RNG, optional retry logic, parallel execution via joblib,
and streaming reducer support.
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.random import SeedSequence, default_rng

from ._spec import ResampleSpec
from ._statistic import Statistic
from ._perturbation import _perturb
from ._preflight import preflight as _preflight

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None


# Sentinel for failed iterations (identity comparison only).
class _FailedSentinel:
    __slots__ = ()
    def __repr__(self):
        return "<FAILED>"

_FAILED = _FailedSentinel()


@dataclass
class ResampleRun:
    """Container for the output of ``run()``.

    Attributes
    ----------
    observed : Any
        Observed statistic value (computed on original data).
    samples : list or None
        Stored iteration results (if ``store="all"``).
    reducer_state : dict or None
        Finalized reducer state (if a reducer was provided).
    n_success : int
        Number of successful iterations.
    n_failed : int
        Number of failed iterations.
    spec : ResampleSpec
        The resampling specification used.
    statistic : Statistic
        The statistic wrapper used.
    seed : int
        Base seed for reproducibility.
    resolved_alternative : str or None
        Alternative hypothesis resolved once in ``run()``.
    diagnostics : dict
        Timing and other engine diagnostics.
    """
    observed: Any
    samples: Optional[list]
    reducer_state: Optional[dict]
    n_success: int
    n_failed: int
    spec: ResampleSpec
    statistic: Statistic
    seed: int
    resolved_alternative: Optional[str]
    diagnostics: dict = field(default_factory=dict)


def _resolve_alternative(
    caller_alt: Optional[str],
    stat: Statistic,
) -> Optional[str]:
    """Resolve the alternative hypothesis once.

    Priority:
    1. Caller-provided ``alternative`` arg
    2. ``stat.default_alternative``
    3. If ``stat.is_nonnegative`` is True: ``"greater"``
    4. Else: ``"two-sided"``
    """
    if caller_alt is not None:
        return caller_alt
    if stat.default_alternative is not None:
        return stat.default_alternative
    if stat.is_nonnegative is True:
        return "greater"
    return "two-sided"


def run(
    data: dict,
    spec: ResampleSpec,
    statistic: Union[Statistic, Callable],
    *,
    n_iters: int,
    seed: int,
    n_jobs: int = 1,
    store: str = "all",
    reducer=None,
    max_retries_per_iter: int = 0,
    verbose: bool = False,
    alternative: Optional[str] = None,
) -> ResampleRun:
    """Run the resampling engine.

    Parameters
    ----------
    data : dict
        Data bundle. Expected keys depend on ``spec.kind``.
    spec : ResampleSpec
        Resampling specification (from ``indices()``, ``labels()``, etc.).
    statistic : Statistic or callable
        Scorer function. If a raw callable, auto-wrapped to ``Statistic``.
    n_iters : int
        Number of resampling iterations.
    seed : int
        Base seed for full reproducibility via ``SeedSequence``.
    n_jobs : int, default=1
        Number of parallel workers. 1 = sequential.
    store : {"all", "none"}
        Whether to store all iteration results.
    reducer : object, optional
        Streaming reducer (e.g., ``PermutationReducer``). Runs in parent
        process only.
    max_retries_per_iter : int, default=0
        Number of retries per iteration on failure.
    verbose : bool, default=False
        Print progress information.
    alternative : str, optional
        Override tail for permutation p-values.

    Returns
    -------
    ResampleRun
        Container with observed value, samples, reducer state, and diagnostics.
    """
    t_start = time.monotonic()

    # Auto-wrap callable to Statistic.
    if callable(statistic) and not isinstance(statistic, Statistic):
        statistic = Statistic(
            name=getattr(statistic, "__name__", "statistic"),
            fn=statistic,
        )

    # Resolve alternative ONCE.
    resolved_alt = _resolve_alternative(alternative, statistic)

    # Emit nonneg + two-sided warning.
    if (resolved_alt == "two-sided"
            and statistic.is_nonnegative is True):
        warnings.warn(
            "two-sided assumes symmetric null about 0; for nonnegative "
            "distance-like stats, 'greater' is typically intended.",
            stacklevel=2,
        )

    # Preflight validation.
    _preflight(
        data, spec, statistic,
        n_iters=n_iters, store=store, reducer=reducer,
    )

    # Seed derivation: 3-way split.
    ss = SeedSequence(seed)
    obs_ss, reducer_ss, iter_root_ss = ss.spawn(3)
    iter_seeds = iter_root_ss.spawn(n_iters)

    # Observed — reproducible from base seed.
    rng_obs = default_rng(obs_ss)
    observed = statistic.fn(data, rng_obs)

    # Initialize reducer.
    if reducer is not None:
        reducer.init(observed, resolved_alt, seedseq=reducer_ss)

    # Build per-iteration worker.
    max_retries = max_retries_per_iter

    def _one_iter(b):
        retry_seeds = iter_seeds[b].spawn(max_retries + 1)
        for r in range(max_retries + 1):
            rng = default_rng(retry_seeds[r])
            perturbed = _perturb(data, spec, rng)
            try:
                return statistic.fn(perturbed, rng)
            except Exception:
                if r == max_retries:
                    return _FAILED
        return _FAILED  # pragma: no cover

    # Execute iterations.
    if n_jobs != 1 and Parallel is not None:
        raw = Parallel(n_jobs=n_jobs)(
            delayed(_one_iter)(b) for b in range(n_iters)
        )
    else:
        raw = [_one_iter(b) for b in range(n_iters)]

    # Partition successes / failures.
    successes = [r for r in raw if r is not _FAILED]
    n_failed = len(raw) - len(successes)

    # Reducer: apply in parent, in deterministic iteration order.
    if reducer is not None:
        for r in raw:
            if r is not _FAILED:
                reducer.update(r)

    # Store samples only if requested.
    samples = successes if store == "all" else None

    t_end = time.monotonic()

    return ResampleRun(
        observed=observed,
        samples=samples,
        reducer_state=reducer.finalize() if reducer is not None else None,
        n_success=len(successes),
        n_failed=n_failed,
        spec=spec,
        statistic=statistic,
        seed=seed,
        resolved_alternative=resolved_alt,
        diagnostics={
            "elapsed_seconds": t_end - t_start,
            "n_jobs": n_jobs,
            "store": store,
        },
    )
