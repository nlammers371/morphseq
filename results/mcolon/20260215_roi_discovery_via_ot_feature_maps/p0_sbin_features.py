"""
Phase 0 Step 5: Build the S-bin feature table.

Aggregates per-pixel OT feature maps into per-sample-per-bin summaries.
Output: features_sbins.parquet (tiny table, fast, reusable).

All summaries computed on UNSMOOTHED maps (smoothing is visualization-only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from roi_config import Phase0FeatureSet, Phase0SBinConfig

logger = logging.getLogger(__name__)


def define_bins(K: int) -> List[Tuple[float, float]]:
    """Define K uniform bins on S ∈ [0, 1). Returns list of (S_lo, S_hi)."""
    return [(k / K, (k + 1) / K) for k in range(K)]


def compute_bin_mask(S_map: np.ndarray, S_lo: float, S_hi: float) -> np.ndarray:
    """Return boolean mask for pixels in bin [S_lo, S_hi)."""
    valid = np.isfinite(S_map)
    return valid & (S_map >= S_lo) & (S_map < S_hi)


def build_sbin_features_v0(
    X: np.ndarray,
    y: np.ndarray,
    S_map_ref: np.ndarray,
    metadata_df: pd.DataFrame,
    outlier_flag: np.ndarray,
    K: int = 10,
) -> pd.DataFrame:
    """
    Build S-bin feature table (v0: cost_mean only).

    Parameters
    ----------
    X : (N, 512, 512, C) — channel 0 is cost_density
    y : (N,) int
    S_map_ref : (512, 512) float32 in [0,1]
    metadata_df : DataFrame with sample_id, embryo_id, etc.
    outlier_flag : (N,) bool
    K : number of bins

    Returns
    -------
    DataFrame with N×K rows
    """
    N = X.shape[0]
    bins = define_bins(K)
    rows = []

    for i in range(N):
        meta = metadata_df.iloc[i]
        for k, (s_lo, s_hi) in enumerate(bins):
            bin_mask = compute_bin_mask(S_map_ref, s_lo, s_hi)
            n_pixels = int(bin_mask.sum())

            cost_mean = float(np.mean(X[i, bin_mask, 0])) if n_pixels > 0 else 0.0

            rows.append({
                "sample_id": meta.get("sample_id", f"sample_{i}"),
                "embryo_id": meta.get("embryo_id", ""),
                "snip_id": meta.get("snip_id", ""),
                "label_int": int(y[i]),
                "qc_outlier_flag": bool(outlier_flag[i]),
                "k_bin": k,
                "S_lo": s_lo,
                "S_hi": s_hi,
                "n_pixels": n_pixels,
                "cost_mean": cost_mean,
            })

    df = pd.DataFrame(rows)
    logger.info(f"S-bin features (v0): {len(df)} rows ({N} samples × {K} bins)")
    return df


def build_sbin_features_v1(
    X: np.ndarray,
    y: np.ndarray,
    S_map_ref: np.ndarray,
    tangent_ref: np.ndarray,
    normal_ref: np.ndarray,
    metadata_df: pd.DataFrame,
    outlier_flag: np.ndarray,
    K: int = 10,
) -> pd.DataFrame:
    """
    Build S-bin feature table (v1: cost + dynamics).

    Requires V1_DYNAMICS channel set (C=5):
      ch0=cost_density, ch1=disp_u, ch2=disp_v, ch3=disp_mag, ch4=delta_mass

    Computes:
      - cost_mean
      - disp_mag_mean
      - disp_par_mean = mean(displacement · e_parallel)
      - disp_perp_mean = mean(displacement · e_perp)
    """
    N = X.shape[0]
    bins = define_bins(K)
    rows = []

    for i in range(N):
        meta = metadata_df.iloc[i]
        for k, (s_lo, s_hi) in enumerate(bins):
            bin_mask = compute_bin_mask(S_map_ref, s_lo, s_hi)
            n_pixels = int(bin_mask.sum())

            if n_pixels > 0:
                cost_mean = float(np.mean(X[i, bin_mask, 0]))
                disp_mag_mean = float(np.mean(X[i, bin_mask, 3]))

                # Project displacement onto local basis
                disp_u = X[i, bin_mask, 1]  # x component
                disp_v = X[i, bin_mask, 2]  # y component
                tan_x = tangent_ref[bin_mask, 0]
                tan_y = tangent_ref[bin_mask, 1]
                norm_x = normal_ref[bin_mask, 0]
                norm_y = normal_ref[bin_mask, 1]

                disp_par = disp_u * tan_x + disp_v * tan_y
                disp_perp = disp_u * norm_x + disp_v * norm_y

                disp_par_mean = float(np.mean(disp_par))
                disp_perp_mean = float(np.mean(disp_perp))
            else:
                cost_mean = disp_mag_mean = disp_par_mean = disp_perp_mean = 0.0

            rows.append({
                "sample_id": meta.get("sample_id", f"sample_{i}"),
                "embryo_id": meta.get("embryo_id", ""),
                "snip_id": meta.get("snip_id", ""),
                "label_int": int(y[i]),
                "qc_outlier_flag": bool(outlier_flag[i]),
                "k_bin": k,
                "S_lo": s_lo,
                "S_hi": s_hi,
                "n_pixels": n_pixels,
                "cost_mean": cost_mean,
                "disp_mag_mean": disp_mag_mean,
                "disp_par_mean": disp_par_mean,
                "disp_perp_mean": disp_perp_mean,
            })

    df = pd.DataFrame(rows)
    logger.info(f"S-bin features (v1): {len(df)} rows ({N} samples × {K} bins)")
    return df


def build_sbin_features(
    X: np.ndarray,
    y: np.ndarray,
    S_map_ref: np.ndarray,
    metadata_df: pd.DataFrame,
    outlier_flag: np.ndarray,
    feature_set: Phase0FeatureSet = Phase0FeatureSet.V0_COST,
    K: int = 10,
    tangent_ref: Optional[np.ndarray] = None,
    normal_ref: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Dispatch to v0 or v1 builder depending on feature_set.
    """
    if feature_set == Phase0FeatureSet.V0_COST:
        df = build_sbin_features_v0(X, y, S_map_ref, metadata_df, outlier_flag, K)
    elif feature_set == Phase0FeatureSet.V1_DYNAMICS:
        assert tangent_ref is not None and normal_ref is not None, \
            "V1_DYNAMICS requires tangent_ref and normal_ref"
        df = build_sbin_features_v1(
            X, y, S_map_ref, tangent_ref, normal_ref,
            metadata_df, outlier_flag, K,
        )
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    if save_path:
        df.to_parquet(save_path, index=False)
        logger.info(f"Saved S-bin features: {save_path}")

    return df


__all__ = [
    "define_bins",
    "compute_bin_mask",
    "build_sbin_features_v0",
    "build_sbin_features_v1",
    "build_sbin_features",
]
