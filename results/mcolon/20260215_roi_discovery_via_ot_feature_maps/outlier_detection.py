"""
Reusable outlier detection utilities for ROI discovery pipelines.

Phase 0 currently uses IQR filtering on total OT cost as the default QC gate.
This module centralizes that logic and provides optional robust alternatives
(MAD, z-score) for diagnostics and future extensions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class OutlierDetectionConfig:
    """Configuration for outlier detection."""

    method: str = "iqr"
    iqr_multiplier: float = 2.0
    zscore_threshold: float = 3.0
    mad_threshold: float = 3.5


@dataclass(frozen=True)
class OutlierDetectionResult:
    """Output for outlier detection."""

    method: str
    outlier_flag: np.ndarray
    lower_bound: float
    upper_bound: float
    n_total: int
    n_outliers: int

    def to_stats(self) -> Dict[str, float | int | str]:
        """Convert to JSON-serializable stats dict."""
        return {
            "method": self.method,
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "n_total": int(self.n_total),
            "n_outliers": int(self.n_outliers),
            "n_retained": int(self.n_total - self.n_outliers),
        }


def _validate_input(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError("Outlier detection requires a non-empty vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input contains NaN/Inf values")
    return arr


def detect_outliers_iqr(values: np.ndarray, multiplier: float = 2.0) -> OutlierDetectionResult:
    """IQR outlier detection with Tukey fences."""
    arr = _validate_input(values)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    flags = (arr < lower) | (arr > upper)
    return OutlierDetectionResult(
        method="iqr",
        outlier_flag=flags,
        lower_bound=lower,
        upper_bound=upper,
        n_total=arr.size,
        n_outliers=int(flags.sum()),
    )


def detect_outliers_zscore(values: np.ndarray, threshold: float = 3.0) -> OutlierDetectionResult:
    """Z-score outlier detection; mostly diagnostic (non-robust)."""
    arr = _validate_input(values)
    mean = float(arr.mean())
    std = float(arr.std())
    if std == 0.0:
        flags = np.zeros_like(arr, dtype=bool)
        lower, upper = mean, mean
    else:
        z = (arr - mean) / std
        flags = np.abs(z) > threshold
        lower = mean - threshold * std
        upper = mean + threshold * std
    return OutlierDetectionResult(
        method="zscore",
        outlier_flag=flags,
        lower_bound=lower,
        upper_bound=upper,
        n_total=arr.size,
        n_outliers=int(flags.sum()),
    )


def detect_outliers_mad(values: np.ndarray, threshold: float = 3.5) -> OutlierDetectionResult:
    """Robust MAD-based outlier detection."""
    arr = _validate_input(values)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad == 0.0:
        flags = np.zeros_like(arr, dtype=bool)
        lower, upper = med, med
    else:
        robust_z = 0.6745 * (arr - med) / mad
        flags = np.abs(robust_z) > threshold
        # Approximate equivalent bounds for logging purposes.
        delta = (threshold * mad) / 0.6745
        lower = med - delta
        upper = med + delta
    return OutlierDetectionResult(
        method="mad",
        outlier_flag=flags,
        lower_bound=lower,
        upper_bound=upper,
        n_total=arr.size,
        n_outliers=int(flags.sum()),
    )


def detect_outliers(values: np.ndarray, config: OutlierDetectionConfig) -> OutlierDetectionResult:
    """Dispatch to the configured outlier detector."""
    method = config.method.lower()
    if method == "iqr":
        return detect_outliers_iqr(values, multiplier=config.iqr_multiplier)
    if method == "zscore":
        return detect_outliers_zscore(values, threshold=config.zscore_threshold)
    if method == "mad":
        return detect_outliers_mad(values, threshold=config.mad_threshold)
    raise ValueError(f"Unknown outlier detection method: {config.method}")


def save_detection_summary(
    result: OutlierDetectionResult,
    config: OutlierDetectionConfig,
    save_path: str | Path,
) -> None:
    """Write config + result summary for reproducibility."""
    payload = {
        "config": asdict(config),
        "result": result.to_stats(),
    }
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2))


__all__ = [
    "OutlierDetectionConfig",
    "OutlierDetectionResult",
    "detect_outliers",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "detect_outliers_mad",
    "save_detection_summary",
]
