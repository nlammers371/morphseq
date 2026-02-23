"""
Data perturbation logic for the resampling engine.

Implements the three perturbation strategies:
- **indices**: draw row (or unit) indices with/without replacement
- **labels**: shuffle labels (optionally within strata / at unit level)
- **groups**: pool-and-redistribute two arrays
"""

import numpy as np

from ._spec import ResampleSpec


def _perturb(data: dict, spec: ResampleSpec, rng: np.random.Generator) -> dict:
    """Create a perturbed copy of *data* according to *spec*.

    Parameters
    ----------
    data : dict
        The data bundle. Expected keys depend on ``spec.kind``.
    spec : ResampleSpec
        Resampling specification.
    rng : np.random.Generator
        Seeded RNG for this iteration.

    Returns
    -------
    dict
        Shallow copy of *data* with perturbed entries.
    """
    if spec.kind == "indices":
        return _perturb_indices(data, spec, rng)
    elif spec.kind == "labels":
        return _perturb_labels(data, spec, rng)
    elif spec.kind == "groups":
        return _perturb_groups(data, spec, rng)
    else:
        raise ValueError(f"Unknown spec kind: {spec.kind!r}")


# ── indices ──────────────────────────────────────────────────────────

def _perturb_indices(data: dict, spec: ResampleSpec, rng: np.random.Generator) -> dict:
    """Draw row indices (or unit blocks) with/without replacement."""
    out = dict(data)
    params = spec.params
    replacement = params.replacement

    if spec.unit_key is not None:
        selected = _draw_unit_indices(data, spec, rng, replacement)
    else:
        N = _population_size(data)
        n_draw = _draw_size(N, params)

        if spec.within_key is not None:
            selected = _draw_within_strata(
                strata=np.asarray(data[spec.within_key]),
                N=N, n_draw=n_draw, replacement=replacement, rng=rng,
            )
        else:
            selected = rng.choice(N, size=n_draw, replace=replacement)

    out["indices"] = selected
    return out


def _draw_unit_indices(
    data: dict, spec: ResampleSpec, rng: np.random.Generator, replacement: bool,
) -> np.ndarray:
    """Draw at the unit level, then expand to row indices."""
    unit_col = np.asarray(data[spec.unit_key])
    unique_units, inverse = np.unique(unit_col, return_inverse=True)
    n_units = len(unique_units)
    params = spec.params
    n_draw = _draw_size(n_units, params)

    if spec.within_key is not None:
        # Strata defined on units — preflight validated constancy.
        within_col = np.asarray(data[spec.within_key])
        # Take first within-value per unit (validated constant).
        unit_strata = np.array([
            within_col[inverse == u][0] for u in range(n_units)
        ])
        chosen_unit_idx = _draw_within_strata(
            strata=unit_strata, N=n_units, n_draw=n_draw,
            replacement=replacement, rng=rng,
        )
    else:
        chosen_unit_idx = rng.choice(n_units, size=n_draw, replace=replacement)

    # Expand chosen units -> row indices, preserving intra-unit order.
    row_blocks = []
    for uid in chosen_unit_idx:
        rows = np.where(inverse == uid)[0]
        row_blocks.append(rows)

    if len(row_blocks) == 0:
        return np.array([], dtype=np.intp)
    return np.concatenate(row_blocks)


def _draw_within_strata(
    strata: np.ndarray, N: int, n_draw: int,
    replacement: bool, rng: np.random.Generator,
) -> np.ndarray:
    """Draw indices within each stratum, proportional to stratum size."""
    unique_strata = np.unique(strata)
    selected = []
    remaining = n_draw

    for i, s in enumerate(unique_strata):
        mask = strata == s
        stratum_idx = np.where(mask)[0]
        n_s = len(stratum_idx)
        # Proportional allocation; last stratum gets remainder.
        if i < len(unique_strata) - 1:
            n_s_draw = int(round(n_draw * n_s / N))
            n_s_draw = min(n_s_draw, remaining)
        else:
            n_s_draw = remaining

        if n_s_draw > 0:
            if not replacement and n_s_draw > n_s:
                n_s_draw = n_s
            chosen = rng.choice(stratum_idx, size=n_s_draw, replace=replacement)
            selected.append(chosen)
            remaining -= n_s_draw

    if not selected:
        return np.array([], dtype=np.intp)
    return np.concatenate(selected)


# ── labels ───────────────────────────────────────────────────────────

def _perturb_labels(data: dict, spec: ResampleSpec, rng: np.random.Generator) -> dict:
    """Shuffle labels, optionally within strata / at unit level."""
    out = dict(data)
    labels = np.asarray(data["labels"]).copy()

    if spec.unit_key is not None:
        labels = _shuffle_labels_by_unit(data, spec, rng, labels)
    else:
        if spec.within_key is not None:
            within = np.asarray(data[spec.within_key])
            for s in np.unique(within):
                mask = within == s
                labels[mask] = rng.permutation(labels[mask])
        else:
            labels = rng.permutation(labels)

    out["labels"] = labels
    return out


def _shuffle_labels_by_unit(
    data: dict, spec: ResampleSpec, rng: np.random.Generator, labels: np.ndarray,
) -> np.ndarray:
    """Shuffle one-label-per-unit, then broadcast back to rows."""
    unit_col = np.asarray(data[spec.unit_key])
    unique_units, inverse = np.unique(unit_col, return_inverse=True)
    n_units = len(unique_units)

    # Extract one label per unit (preflight validated constancy).
    unit_labels = np.array([labels[inverse == u][0] for u in range(n_units)])

    if spec.within_key is not None:
        within_col = np.asarray(data[spec.within_key])
        unit_within = np.array([within_col[inverse == u][0] for u in range(n_units)])
        for s in np.unique(unit_within):
            mask = unit_within == s
            unit_labels[mask] = rng.permutation(unit_labels[mask])
    else:
        unit_labels = rng.permutation(unit_labels)

    # Broadcast back to row level.
    return unit_labels[inverse]


# ── groups ───────────────────────────────────────────────────────────

def _perturb_groups(data: dict, spec: ResampleSpec, rng: np.random.Generator) -> dict:
    """Pool-and-redistribute two arrays."""
    out = dict(data)
    params = spec.params
    a = np.asarray(data[params.a_key])
    b = np.asarray(data[params.b_key])

    combined = np.concatenate([a, b], axis=0)
    perm_idx = rng.permutation(len(combined))
    n_a = len(a)

    out[params.a_key] = combined[perm_idx[:n_a]]
    out[params.b_key] = combined[perm_idx[n_a:]]
    return out


# ── helpers ──────────────────────────────────────────────────────────

def _population_size(data: dict) -> int:
    """Determine N from data bundle (non-unit mode)."""
    if "n" in data:
        return int(data["n"])
    if "labels" in data:
        return len(data["labels"])
    raise ValueError(
        "Cannot determine population size. Provide data['n'] or data['labels']."
    )


def _draw_size(N: int, params) -> int:
    """Resolve the number of items to draw."""
    if params.size is not None:
        return params.size
    if params.frac is not None:
        return max(1, int(round(N * params.frac)))
    return N
