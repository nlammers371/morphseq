"""
pairwise_outputs.py
-------------------
Array-level postprocessing helpers for pairwise classification outputs.

Works on raw arrays (features, mask, labels, feature_names) — no CondensationData.
This keeps classification independent of trajectory_condensation.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class PairwiseSubsetResult:
    """Output of subset_pairwise_cube(), including audit info.

    Shapes
    ------
    features    : (N_e', T', K')
    mask        : (N_e', T') bool
    embryo_ids  : (N_e',) str
    time_values : (T',) float
    labels      : (N_e',) str
    feature_names : length K'
    """
    features: np.ndarray
    mask: np.ndarray
    embryo_ids: np.ndarray
    time_values: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    n_embryos_kept: int
    n_time_bins_kept: int
    dropped_by_name: list[str] = field(default_factory=list)
    dropped_by_nan: list[str] = field(default_factory=list)


def subset_pairwise_cube(
    features: np.ndarray,
    mask: np.ndarray,
    embryo_ids: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    keep_genotypes: Sequence[str],
    *,
    drop_irrelevant_comparisons: bool = True,
    nan_policy: str = "warn_drop",
) -> PairwiseSubsetResult:
    """Subset a pairwise feature cube to a set of genotypes.

    Steps
    -----
    1. Validate input shapes.
    2. Filter embryos to keep_genotypes.
    3. Drop time bins with no observations in the subset.
    4. (Optional) Drop feature dims A__vs__B where neither A nor B is in
       keep_genotypes — string heuristic, first pass.
    5. Detect remaining dims that are all-NaN on observed entries; apply nan_policy.

    Parameters
    ----------
    features : (N_e, T, K)
    mask : (N_e, T) bool
    embryo_ids : (N_e,) str
    time_values : (T,) float
    labels : (N_e,) str — genotype per embryo
    feature_names : length K
    keep_genotypes : genotype labels to retain
    drop_irrelevant_comparisons : drop A__vs__B cols where neither side is in keep_genotypes
    nan_policy : "warn_drop" | "error" | "keep"
        What to do when feature dims are all-NaN on observed entries after subsetting.
        "warn_drop" emits UserWarning then drops (default).
        "error" raises ValueError.
        "keep" retains them silently.

    Returns
    -------
    PairwiseSubsetResult with filtered arrays and audit fields.
    """
    assert features.ndim == 3, f"features must be (N_e, T, K), got shape {features.shape}"
    N_e, T, K = features.shape
    assert mask.shape == (N_e, T), f"mask shape {mask.shape} != (N_e={N_e}, T={T})"
    assert len(embryo_ids) == N_e, f"len(embryo_ids)={len(embryo_ids)} != N_e={N_e}"
    assert len(time_values) == T, f"len(time_values)={len(time_values)} != T={T}"
    assert len(labels) == N_e, f"len(labels)={len(labels)} != N_e={N_e}"
    assert len(feature_names) == K, f"len(feature_names)={len(feature_names)} != K={K}"

    keep_set = set(keep_genotypes)
    obs_mask = mask.astype(bool)
    embryo_sel = np.array([g in keep_set for g in labels])
    if not embryo_sel.any():
        raise ValueError(f"No embryos found for genotypes {list(keep_genotypes)!r}.")

    new_embryo_ids = embryo_ids[embryo_sel]
    new_labels = labels[embryo_sel]
    new_features = features[embryo_sel].copy()
    new_mask = obs_mask[embryo_sel]

    # Drop time bins with no observations in the subset
    observed_t = np.where(new_mask.any(axis=0))[0]
    new_features = new_features[:, observed_t, :]
    new_mask = new_mask[:, observed_t]
    new_time_values = np.asarray(time_values)[observed_t]

    feat_names = list(feature_names)

    # First-pass: string-based drop of irrelevant comparison columns
    dropped_by_name: list[str] = []
    if drop_irrelevant_comparisons:
        keep_k = []
        for k, name in enumerate(feat_names):
            if "__vs__" in name:
                lhs, rhs = name.split("__vs__", 1)
                if lhs.strip() not in keep_set and rhs.strip() not in keep_set:
                    dropped_by_name.append(name)
                    continue
            keep_k.append(k)
        new_features = new_features[:, :, keep_k]
        feat_names = [feat_names[k] for k in keep_k]

    # Second-pass: correctness check — dims that are all-NaN on observed entries
    dropped_by_nan: list[str] = []
    all_nan_k = [
        k for k in range(new_features.shape[2])
        if not np.any(np.isfinite(new_features[:, :, k]) & new_mask)
    ]
    if all_nan_k:
        names = [feat_names[k] for k in all_nan_k]
        dropped_by_nan = names
        preview = ", ".join(names[:5]) + (f" (and {len(names) - 5} more)" if len(names) > 5 else "")
        msg = (
            f"subset_pairwise_cube: {len(all_nan_k)} feature dim(s) are all-NaN "
            f"for genotypes {list(keep_genotypes)!r} "
            f"({new_embryo_ids.shape[0]} embryos, {new_time_values.shape[0]} time bins).\n"
            f"This usually means comparisons are irrelevant or the subset has no support.\n"
            f"Affected: {preview}.\n"
            f"Set nan_policy='error' to fail instead, or nan_policy='keep' to retain."
        )
        if nan_policy == "error":
            raise ValueError(msg)
        elif nan_policy == "warn_drop":
            warnings.warn(msg, UserWarning, stacklevel=2)
            keep_k2 = [k for k in range(new_features.shape[2]) if k not in set(all_nan_k)]
            new_features = new_features[:, :, keep_k2]
            feat_names = [feat_names[k] for k in keep_k2]
        # "keep": do nothing

    return PairwiseSubsetResult(
        features=new_features,
        mask=new_mask,
        embryo_ids=new_embryo_ids,
        time_values=new_time_values,
        labels=new_labels,
        feature_names=feat_names,
        n_embryos_kept=int(new_embryo_ids.shape[0]),
        n_time_bins_kept=int(new_time_values.shape[0]),
        dropped_by_name=dropped_by_name,
        dropped_by_nan=dropped_by_nan,
    )
