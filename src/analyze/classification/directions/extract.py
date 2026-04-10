"""Lightweight entry point: extract classifier direction vectors without AUROC/permutations.

extract_classifier_directions() produces the same ClassifierDirections artifact as
run_classification(..., save_classifier_directions=True), except that auroc_obs and
pval columns are NaN (since no CV scoring is performed).

Use this when you only need morphological geometry products and do not need
AUROC, permutation p-values, confusion matrices, or contrast coordinates.

The direction vectors (unit_coef, sign convention, preprocess_fingerprint) are
byte-identical to those from run_classification because both paths go through
the same fit_classifier_direction() and build_classifier_directions_payload().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..engine.comparison_resolution import (
    ComparisonScheme,
    resolve_comparisons,
)
from ..engine.loop import (
    _bin_and_aggregate,
    _build_binary_labels,
    _resolve_feature_columns,
)
from .artifact import ClassifierDirections
from .build_payload import build_classifier_directions_payload
from .fit import fit_classifier_direction


def extract_classifier_directions(
    df: pd.DataFrame,
    *,
    class_col: str,
    id_col: str,
    time_col: str,
    comparisons: ComparisonScheme,
    features: dict[str, str | list[str]],
    bin_width: float = 4.0,
    min_samples_per_group: int = 3,
    min_samples_per_member: int = 2,
    random_state: int = 42,
    class_weight: Any | None = "balanced",
    verbose: bool = False,
    save_dir: str | Path | None = None,
    overwrite: bool = False,
) -> ClassifierDirections:
    """Fit per-(feature_set, comparison, time_bin) binary logistic regressions.

    Extracts signed unit direction vectors without CV, permutation tests, AUROC,
    or confusion matrices. Substantially faster than run_classification.

    Parameters
    ----------
    df : DataFrame with id_col, time_col, class_col, and feature columns.
    class_col : column with group/genotype labels.
    id_col : column with per-embryo unique identifiers.
    time_col : column with continuous time values (hpf).
    comparisons : same format as run_classification — list of
        {"positive": ..., "negative": ...} dicts or a ComparisonScheme.
    features : same format as run_classification — dict mapping feature set
        name to a column prefix (str) or explicit column list (list[str]).
    bin_width : time bin width in hpf (must match what you will pass to
        run_morphology_geometry / project_binned_features).
    min_samples_per_group : minimum total samples for a comparison to be scored.
    min_samples_per_member : minimum samples per class within a bin.
    random_state : passed to LogisticRegression.
    class_weight : passed to LogisticRegression.
    verbose : if True, print progress messages.
    save_dir : if provided, write classifier_directions.parquet and
        classifier_directions_vectors.npz to this directory.
    overwrite : if False (default), raise FileExistsError if artifacts already exist.

    Returns
    -------
    ClassifierDirections artifact. auroc_obs and pval are NaN.
    The geometry validator will accept this with has_auroc=False, and weight_mode
    will automatically fall back to "uniform" in run_morphology_geometry.
    """
    resolved = resolve_comparisons(comparisons, class_col=class_col, df=df)
    feature_sets = _resolve_feature_columns(df, features)

    fits: list[dict[str, Any]] = []

    for fs_name, feature_cols in feature_sets.items():
        for rc in resolved:
            if verbose:
                print(f"  [{fs_name}] {rc.comparison_id}")

            labeled = _build_binary_labels(df, class_col, rc)
            if labeled[class_col].nunique() < 2:
                continue

            # Check group-level minimum before binning
            group_counts = labeled[class_col].value_counts()
            if group_counts.min() < min_samples_per_group:
                if verbose:
                    print(
                        f"    Skipping {rc.comparison_id}: "
                        f"insufficient group size ({group_counts.min()} < {min_samples_per_group})"
                    )
                continue

            binned = _bin_and_aggregate(labeled, id_col, time_col, feature_cols, bin_width)

            for time_bin_center, sub in binned.groupby("time_bin_center"):
                y = sub["_y"].to_numpy(dtype=int)
                n_pos = int((y == 1).sum())
                n_neg = int((y == 0).sum())

                if n_pos < min_samples_per_member or n_neg < min_samples_per_member:
                    continue

                X = sub[feature_cols].to_numpy(dtype=float)
                time_bin = int(np.floor(float(time_bin_center) - bin_width / 2.0))

                fit = fit_classifier_direction(
                    X=X,
                    y_binary=y,
                    feature_cols=feature_cols,
                    random_state=random_state,
                    class_weight=class_weight,
                )
                if fit is None:
                    continue

                fit.update(
                    {
                        "feature_set": fs_name,
                        "comparison_id": rc.comparison_id,
                        "positive_label": rc.positive_label,
                        "negative_label": rc.negative_label,
                        "time_bin": time_bin,
                        "time_bin_center": float(time_bin_center),
                        "bin_width": float(bin_width),
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                    }
                )
                fits.append(fit)

    directions = build_classifier_directions_payload(fits, feature_sets=feature_sets)

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        meta_path = save_path / "classifier_directions.parquet"
        vec_path = save_path / "classifier_directions_vectors.npz"
        if not overwrite:
            for p in (meta_path, vec_path):
                if p.exists():
                    raise FileExistsError(
                        f"{p} already exists. Pass overwrite=True to replace."
                    )
        directions.metadata.to_parquet(meta_path, index=False)
        directions.save(vec_path)

    return directions
