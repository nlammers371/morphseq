"""
Preflight validation for resampling.

``preflight()`` is called before the iteration loop to catch configuration
errors early. It validates data keys, population size, unit/within
consistency, memory estimates, and reducer compatibility.
"""

import sys
import warnings
from typing import Optional

import numpy as np

from ._spec import ResampleSpec, IndicesParams, GroupsParams
from ._statistic import Statistic
from ._perturbation import _perturb, _population_size
from ._reducer import PermutationReducer


_MAX_WARN_BYTES = 1 * 1024**3   # 1 GB
_MAX_ERROR_BYTES = 4 * 1024**3  # 4 GB


def preflight(
    data: dict,
    spec: ResampleSpec,
    statistic: Statistic,
    *,
    n_iters: int = 1,
    store: str = "all",
    reducer=None,
) -> None:
    """Validate data/spec/statistic before the iteration loop.

    Raises ``ValueError`` on fatal errors, emits ``warnings.warn`` for
    non-fatal concerns.

    Parameters
    ----------
    data : dict
        The data bundle.
    spec : ResampleSpec
        Resampling specification.
    statistic : Statistic
        Statistic wrapper.
    n_iters : int
        Number of planned iterations (used for memory estimate).
    store : str
        ``"all"`` or ``"none"``.
    reducer : object, optional
        A reducer instance (e.g., ``PermutationReducer``).
    """
    # 1. Groups restrictions (before anything else)
    _check_groups_restrictions(spec)

    # 2. Required keys and lengths
    _check_required_keys(data, spec)

    # 3. Population size (indices mode)
    _check_population_size(data, spec)

    # 4. Unit consistency
    _check_unit_consistency(data, spec)

    # 5. Label consistency
    _check_label_consistency(data, spec)

    # 6. Strata viability
    _check_strata_viability(data, spec)

    # 7. Statistic dry run
    dry_result = _dry_run(data, spec, statistic)

    # 8. Memory estimate
    if store == "all":
        _check_memory(dry_result, n_iters)

    # 9. Reducer + scalar enforcement
    _check_reducer_compat(reducer, dry_result)

    # 10. Store policy
    _check_store_policy(spec, store, reducer)


# ── individual checks ────────────────────────────────────────────────

def _check_groups_restrictions(spec: ResampleSpec):
    if spec.kind == "groups":
        if spec.within_key is not None or spec.unit_key is not None:
            raise ValueError(
                "groups(within=...) and groups(unit=...) not yet supported in PR1."
            )


def _check_required_keys(data: dict, spec: ResampleSpec):
    """Verify expected keys exist and row-level arrays have matching lengths."""
    if spec.kind == "groups":
        params = spec.params
        for key in (params.a_key, params.b_key):
            if key not in data:
                raise ValueError(f"data[{key!r}] required for groups mode.")
        return

    if spec.kind == "labels" and "labels" not in data:
        raise ValueError("data['labels'] required for labels mode.")

    if spec.unit_key is not None and spec.unit_key not in data:
        raise ValueError(f"data[{spec.unit_key!r}] (unit_key) not found in data.")

    if spec.within_key is not None and spec.within_key not in data:
        raise ValueError(f"data[{spec.within_key!r}] (within_key) not found in data.")

    # Check lengths match across row-level arrays.
    row_keys = []
    if "labels" in data:
        row_keys.append("labels")
    if spec.unit_key is not None:
        row_keys.append(spec.unit_key)
    if spec.within_key is not None:
        row_keys.append(spec.within_key)

    if row_keys:
        lengths = {k: len(np.asarray(data[k])) for k in row_keys}
        unique_lens = set(lengths.values())
        if len(unique_lens) > 1:
            raise ValueError(
                f"Row-level arrays have inconsistent lengths: {lengths}"
            )


def _check_population_size(data: dict, spec: ResampleSpec):
    if spec.kind != "indices":
        return
    if spec.unit_key is not None:
        # N = number of unique units — always determinable.
        return
    try:
        _population_size(data)
    except ValueError:
        raise ValueError(
            "Cannot determine population size for indices mode. "
            "Provide data['n'] or data['labels']."
        )


def _check_unit_consistency(data: dict, spec: ResampleSpec):
    """If unit_key + within_key: within must be constant per unit."""
    if spec.unit_key is None or spec.within_key is None:
        return
    unit_col = np.asarray(data[spec.unit_key])
    within_col = np.asarray(data[spec.within_key])
    unique_units = np.unique(unit_col)
    for u in unique_units:
        mask = unit_col == u
        vals = within_col[mask]
        if len(np.unique(vals)) > 1:
            raise ValueError(
                f"within_key {spec.within_key!r} is not constant for "
                f"unit {u!r}. Each unit must belong to exactly one stratum."
            )


def _check_label_consistency(data: dict, spec: ResampleSpec):
    """If kind='labels' + unit_key: labels must be constant per unit."""
    if spec.kind != "labels" or spec.unit_key is None:
        return
    unit_col = np.asarray(data[spec.unit_key])
    labels = np.asarray(data["labels"])
    unique_units = np.unique(unit_col)
    for u in unique_units:
        mask = unit_col == u
        vals = labels[mask]
        if len(np.unique(vals)) > 1:
            raise ValueError(
                f"Labels are not constant for unit {u!r}. "
                f"When using unit_key with labels mode, all rows within "
                f"a unit must share the same label."
            )


def _check_strata_viability(data: dict, spec: ResampleSpec):
    """Ensure at least one stratum has size >= 2."""
    if spec.within_key is None:
        return
    if spec.kind == "groups":
        return  # Groups restrictions already handled.

    if spec.unit_key is not None:
        unit_col = np.asarray(data[spec.unit_key])
        within_col = np.asarray(data[spec.within_key])
        unique_units, inverse = np.unique(unit_col, return_inverse=True)
        unit_strata = np.array([
            within_col[inverse == u][0] for u in range(len(unique_units))
        ])
        counts = np.unique(unit_strata, return_counts=True)[1]
    else:
        within_col = np.asarray(data[spec.within_key])
        counts = np.unique(within_col, return_counts=True)[1]

    if np.all(counts < 2):
        raise ValueError(
            f"All strata in {spec.within_key!r} are singletons. "
            f"Cannot produce a non-degenerate null distribution."
        )


def _dry_run(data: dict, spec: ResampleSpec, statistic: Statistic):
    """Run statistic on one synthetic perturbation to validate output type."""
    rng = np.random.default_rng(0)
    perturbed = _perturb(data, spec, rng)
    try:
        result = statistic.fn(perturbed, rng)
    except Exception as e:
        raise ValueError(
            f"Statistic dry run failed: {e!r}. "
            f"Ensure stat.fn(data, rng) works on perturbed data."
        ) from e

    _validate_output(result)
    return result


def _validate_output(result):
    """Check that the dry-run output is numeric scalar, array, or dict."""
    if isinstance(result, dict):
        for k, v in result.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"Dict output keys must be strings, got {type(k)}"
                )
    elif isinstance(result, np.ndarray):
        if not np.issubdtype(result.dtype, np.number):
            raise ValueError(
                f"Array output must be numeric, got dtype={result.dtype}"
            )
    elif isinstance(result, (int, float, np.integer, np.floating)):
        pass
    else:
        raise ValueError(
            f"Statistic must return scalar, ndarray, or dict. "
            f"Got {type(result).__name__}."
        )


def _check_memory(dry_result, n_iters: int):
    """Estimate memory for stored results; warn or error if excessive."""
    bpi = _estimate_bytes(dry_result)
    est = bpi * n_iters
    if est > _MAX_ERROR_BYTES:
        raise ValueError(
            f"Estimated memory for {n_iters} iterations: "
            f"{est / 1024**3:.1f} GB (> 4 GB limit). "
            f"Use store='none' with a reducer, or reduce n_iters."
        )
    if est > _MAX_WARN_BYTES:
        warnings.warn(
            f"Estimated memory for {n_iters} iterations: "
            f"{est / 1024**3:.1f} GB. Consider store='none' with a reducer.",
            stacklevel=4,
        )


def _estimate_bytes(result) -> int:
    if isinstance(result, np.ndarray):
        return result.nbytes
    if isinstance(result, (int, float, np.integer, np.floating)):
        return 8
    if isinstance(result, dict):
        total = 0
        for v in result.values():
            total += _estimate_bytes(v)
        return total
    return sys.getsizeof(result)


def _check_reducer_compat(reducer, dry_result):
    if reducer is None:
        return
    if isinstance(reducer, PermutationReducer):
        if not isinstance(dry_result, (int, float, np.integer, np.floating)):
            raise ValueError(
                "PermutationReducer supports scalar outputs only. "
                f"Got {type(dry_result).__name__}."
            )


def _check_store_policy(spec: ResampleSpec, store: str, reducer):
    if spec.kind == "indices" and store == "none":
        raise ValueError(
            "indices mode requires store='all' in current version. "
            "BootstrapReducer not yet available."
        )
