from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import skimage.io as io

from data_pipeline.shared.path_contracts import require_existing_path

from .curvature_skeletonization import extract_centerline_points


def _smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size < 3 or window <= 1:
        return values.astype(np.float64)
    window = min(int(window), int(values.size))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def compute_curvature_metrics(mask: np.ndarray, pixel_size_um: float) -> Dict[str, float]:
    """Compute curvature summaries from an embryo mask."""
    centerline_xy_px = extract_centerline_points(mask)
    if centerline_xy_px.shape[0] < 3:
        return {
            "mean_curvature_per_um": np.nan,
            "median_curvature_per_um": np.nan,
            "max_curvature_per_um": np.nan,
            "centerline_length_um": np.nan,
            "centerline_point_count": int(centerline_xy_px.shape[0]),
        }

    centerline_xy_um = centerline_xy_px * float(pixel_size_um)
    diffs = np.diff(centerline_xy_um, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    valid = s > 0
    if np.count_nonzero(valid) < 3:
        return {
            "mean_curvature_per_um": np.nan,
            "median_curvature_per_um": np.nan,
            "max_curvature_per_um": np.nan,
            "centerline_length_um": float(s[-1]),
            "centerline_point_count": int(centerline_xy_px.shape[0]),
        }

    x = _smooth_series(centerline_xy_um[:, 0])
    y = _smooth_series(centerline_xy_um[:, 1])
    dx_ds = np.gradient(x, s, edge_order=1)
    dy_ds = np.gradient(y, s, edge_order=1)
    d2x_ds2 = np.gradient(dx_ds, s, edge_order=1)
    d2y_ds2 = np.gradient(dy_ds, s, edge_order=1)
    denom = np.power(dx_ds ** 2 + dy_ds ** 2, 1.5)
    numer = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
    curvature = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 0)
    curvature = curvature[np.isfinite(curvature)]

    if curvature.size == 0:
        mean_curvature = np.nan
        median_curvature = np.nan
        max_curvature = np.nan
    else:
        mean_curvature = float(np.mean(curvature))
        median_curvature = float(np.median(curvature))
        max_curvature = float(np.max(curvature))

    return {
        "mean_curvature_per_um": mean_curvature,
        "median_curvature_per_um": median_curvature,
        "max_curvature_per_um": max_curvature,
        "centerline_length_um": float(s[-1]),
        "centerline_point_count": int(centerline_xy_px.shape[0]),
    }


def extract_curvature_metrics_batch(
    tracking_df: pd.DataFrame,
    mask_dir: Path | None = None,
    pixel_size_col: str = "micrometers_per_pixel",
    mask_path_col: str = "exported_mask_path",
) -> pd.DataFrame:
    """Extract curvature metrics for a batch of tracked snips."""
    results = []
    for _, row in tracking_df.iterrows():
        snip_id = row["snip_id"]
        mask_path = require_existing_path(
            row.get(mask_path_col),
            context="curvature_metrics",
            field_name=mask_path_col,
            row_id=str(snip_id),
        )
        mask = io.imread(mask_path)
        if pixel_size_col not in row.index or pd.isna(row[pixel_size_col]):
            raise ValueError(f"curvature_metrics: missing required pixel size column '{pixel_size_col}' for snip_id={snip_id}")
        pixel_size = float(row[pixel_size_col])
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(f"curvature_metrics: invalid pixel size {pixel_size!r} for snip_id={snip_id}")
        metrics = compute_curvature_metrics(mask, pixel_size)
        metrics["snip_id"] = snip_id
        results.append(metrics)
    return pd.DataFrame(results)
