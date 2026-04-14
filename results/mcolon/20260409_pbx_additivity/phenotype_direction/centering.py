"""Centering primitives for experiment-local phenotype-direction coordinates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


INTERCEPT_CENTERED = "intercept_centered"
NEG_CENTROID_CENTERED = "neg_centroid_centered"
MIDPOINT_CENTERED = "midpoint_centered"
RAW_PROJECTION = "raw_projection"

CENTERING_VARIANTS: tuple[str, ...] = (
    INTERCEPT_CENTERED,
    NEG_CENTROID_CENTERED,
    MIDPOINT_CENTERED,
    RAW_PROJECTION,
)

_REQUIRED_PROJECTION_COLUMNS = frozenset({"embryo_id", "time_bin_center", "genotype"})


def validate_projection_frame(
    proj_bin: pd.DataFrame,
    *,
    raw_col: str = "raw_score",
    label_col: str = "genotype",
) -> None:
    """Validate the minimal projected-table contract required for centering."""
    missing = set(_REQUIRED_PROJECTION_COLUMNS) - set(proj_bin.columns)
    if missing:
        raise ValueError(f"Projected table is missing required columns: {sorted(missing)}")
    if raw_col not in proj_bin.columns:
        raise ValueError(f"Projected table is missing raw score column {raw_col!r}.")
    if label_col not in proj_bin.columns:
        raise ValueError(f"Projected table is missing label column {label_col!r}.")


def compute_center_stats(
    proj_bin: pd.DataFrame,
    *,
    intercept: float,
    coef_norm: float,
    pos_label: str,
    neg_label: str,
    raw_col: str = "raw_score",
    label_col: str = "genotype",
) -> dict[str, float | int]:
    """Compute per-bin centering statistics from a shared projected representation."""
    validate_projection_frame(proj_bin, raw_col=raw_col, label_col=label_col)
    if not np.isfinite(float(coef_norm)) or float(coef_norm) == 0.0:
        raise ValueError(f"coef_norm must be finite and non-zero; got {coef_norm!r}")

    pos_mask = proj_bin[label_col].astype(str) == str(pos_label)
    neg_mask = proj_bin[label_col].astype(str) == str(neg_label)
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Projected bin must contain both labels; got n_pos={n_pos}, n_neg={n_neg}"
        )

    pos_mean = float(proj_bin.loc[pos_mask, raw_col].mean())
    neg_mean = float(proj_bin.loc[neg_mask, raw_col].mean())
    midpoint = 0.5 * (pos_mean + neg_mean)
    boundary = -float(intercept) / float(coef_norm)
    return {
        "boundary": boundary,
        "neg_mean": neg_mean,
        "pos_mean": pos_mean,
        "midpoint": midpoint,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "coef_norm": float(coef_norm),
        "intercept": float(intercept),
    }


def apply_centering_variant(
    proj_bin: pd.DataFrame,
    *,
    variant: str,
    center_stats: Mapping[str, float | int],
    raw_col: str = "raw_score",
) -> pd.Series:
    """Apply one centering variant to a projected bin table."""
    validate_projection_frame(proj_bin, raw_col=raw_col)
    raw = proj_bin[raw_col].astype(float)
    if variant == INTERCEPT_CENTERED:
        return raw + float(center_stats["intercept"]) / float(center_stats["coef_norm"])
    if variant == NEG_CENTROID_CENTERED:
        return raw - float(center_stats["neg_mean"])
    if variant == MIDPOINT_CENTERED:
        return raw - float(center_stats["midpoint"])
    if variant == RAW_PROJECTION:
        return raw
    raise ValueError(f"Unknown centering variant {variant!r}; expected one of {CENTERING_VARIANTS}")


def compute_all_centered_scores(
    proj_bin: pd.DataFrame,
    *,
    intercept: float,
    coef_norm: float,
    pos_label: str,
    neg_label: str,
    raw_col: str = "raw_score",
    label_col: str = "genotype",
    variants: Sequence[str] = CENTERING_VARIANTS,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return a copy of ``proj_bin`` with centered score columns added."""
    center_stats = compute_center_stats(
        proj_bin,
        intercept=intercept,
        coef_norm=coef_norm,
        pos_label=pos_label,
        neg_label=neg_label,
        raw_col=raw_col,
        label_col=label_col,
    )
    out = proj_bin.copy()
    for variant in variants:
        out[variant] = apply_centering_variant(
            out,
            variant=variant,
            center_stats=center_stats,
            raw_col=raw_col,
        )
    return out, center_stats


def center_metadata_row(
    *,
    vector_id: str,
    comparison_id: str,
    time_bin_center: float,
    positive_label: str,
    negative_label: str,
    center_stats: Mapping[str, float | int],
    time_bin: int | None = None,
) -> dict[str, object]:
    """Build a stable metadata row for one direction/time-bin centering payload."""
    return {
        "vector_id": str(vector_id),
        "comparison_id": str(comparison_id),
        "time_bin_center": float(time_bin_center),
        "time_bin": int(time_bin) if time_bin is not None else None,
        "positive_label": str(positive_label),
        "negative_label": str(negative_label),
        "coef_norm": float(center_stats["coef_norm"]),
        "intercept": float(center_stats["intercept"]),
        "boundary_score": float(center_stats["boundary"]),
        "neg_mean": float(center_stats["neg_mean"]),
        "pos_mean": float(center_stats["pos_mean"]),
        "midpoint": float(center_stats["midpoint"]),
        "n_pos": int(center_stats["n_pos"]),
        "n_neg": int(center_stats["n_neg"]),
    }
