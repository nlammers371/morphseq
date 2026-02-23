"""Compact feature extraction for OT pair analysis and ML/PCA."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.fft import dctn

from .storage import upsert_pair_metrics, compute_barycentric_projection


FEATURE_VECTOR_SCHEMA_VERSION = "1.0.0"
FEATURE_KEY_COLUMNS = ("run_id", "pair_id")
FEATURE_STRING_COLUMNS = ("feature_schema_version", "run_id", "pair_id")


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def _metric_with_fallback(metrics: Mapping, *keys: str) -> float:
    for key in keys:
        if key in metrics:
            value = _safe_float(metrics.get(key))
            if not np.isnan(value):
                return value
    return float("nan")


def _support_mask(result_canon) -> np.ndarray:
    vel_mag = np.linalg.norm(result_canon.velocity_canon_px_per_step_yx, axis=-1)
    return (
        (result_canon.mass_created_canon > 0)
        | (result_canon.mass_destroyed_canon > 0)
        | (vel_mag > 0)
    )


def _divergence_and_curl(velocity_hw2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vy = velocity_hw2[..., 0]
    vx = velocity_hw2[..., 1]
    dvy_dy, _dvy_dx = np.gradient(vy)
    dvx_dy, dvx_dx = np.gradient(vx)
    divergence = dvy_dy + dvx_dx
    curl = dvx_dy - _dvy_dx
    return divergence, curl


def dct_radial_band_energy_fractions(
    field_hw: np.ndarray,
    n_bands: int = 8,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute normalized DCT radial-band energies for one scalar field."""
    if n_bands <= 0:
        raise ValueError("n_bands must be positive.")
    field = np.asarray(field_hw, dtype=np.float32)
    if support_mask is not None:
        field = np.where(np.asarray(support_mask, dtype=bool), field, 0.0)

    coeff = dctn(field, type=2, norm="ortho")
    power = coeff * coeff

    h, w = field.shape
    fy = np.arange(h, dtype=np.float32)[:, None] / max(1.0, float(h))
    fx = np.arange(w, dtype=np.float32)[None, :] / max(1.0, float(w))
    radius = np.sqrt(fy * fy + fx * fx)

    edges = np.linspace(0.0, float(radius.max()) + 1e-12, n_bands + 1)
    band_power = np.zeros(n_bands, dtype=np.float64)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bands - 1:
            band_mask = (radius >= lo) & (radius <= hi)
        else:
            band_mask = (radius >= lo) & (radius < hi)
        band_power[i] = float(power[band_mask].sum())

    total = float(band_power.sum())
    if total <= 0:
        return np.zeros(n_bands, dtype=np.float32)
    return (band_power / total).astype(np.float32)


def extract_pair_feature_record(
    *,
    run_id: str,
    pair_id: str,
    result_work,
    result_canon,
    backend: str,
    n_bands: int = 8,
    feature_schema_version: str = FEATURE_VECTOR_SCHEMA_VERSION,
) -> Dict:
    """Extract compact per-pair features for downstream PCA/ML."""
    diagnostics = result_work.diagnostics or {}
    metrics = diagnostics.get("metrics", {}) if isinstance(diagnostics, Mapping) else {}
    support_mask = _support_mask(result_canon)

    velocity = np.asarray(result_canon.velocity_canon_px_per_step_yx, dtype=np.float32)
    vy = velocity[..., 0]
    vx = velocity[..., 1]
    divergence, curl = _divergence_and_curl(velocity)

    # Convert barycentric displacement to micrometers if pair-frame metadata exists.
    bary = compute_barycentric_projection(result_work)
    disp = np.linalg.norm(bary["barycentric_velocity_yx"], axis=1).astype(np.float64)
    mass = np.asarray(bary["transported_mass_src"], dtype=np.float64)
    valid = mass > 1e-12
    disp = disp[valid] if np.any(valid) else disp
    scale_um = 1.0
    if getattr(result_work, "work_um_per_px", None) is not None and not np.isnan(float(result_work.work_um_per_px)):
        scale_um = float(result_work.work_um_per_px)
    disp_um = disp * scale_um

    record = {
        "feature_schema_version": feature_schema_version,
        "run_id": run_id,
        "pair_id": pair_id,
        "backend": str(backend),
        "n_dct_bands": int(n_bands),
        "ot_total_transport_cost_um2": _metric_with_fallback(metrics, "total_transport_cost_um2", "total_transport_cost"),
        "ot_mean_transport_distance_um": _metric_with_fallback(metrics, "mean_transport_distance_um", "mean_transport_distance"),
        "ot_specific_transport_cost_um2_per_mass": _metric_with_fallback(
            metrics, "specific_transport_cost_um2_per_mass", "specific_transport_cost"
        ),
        "ot_transported_mass_pct_src": _metric_with_fallback(metrics, "transported_mass_pct_src"),
        "ot_transported_mass_pct_tgt": _metric_with_fallback(metrics, "transported_mass_pct_tgt"),
        "ot_created_mass_pct": _metric_with_fallback(metrics, "created_mass_pct"),
        "ot_destroyed_mass_pct": _metric_with_fallback(metrics, "destroyed_mass_pct"),
        "ot_mass_ratio_crop": _metric_with_fallback(metrics, "mass_ratio_crop"),
        "ot_mass_delta_crop": _metric_with_fallback(metrics, "mass_delta_crop"),
        "ot_proportion_transported": _metric_with_fallback(metrics, "proportion_transported"),
        "bar_disp_mean_um": float(np.mean(disp_um)) if disp_um.size else float("nan"),
        "bar_disp_std_um": float(np.std(disp_um)) if disp_um.size else float("nan"),
        "bar_disp_p50_um": float(np.percentile(disp_um, 50)) if disp_um.size else float("nan"),
        "bar_disp_p90_um": float(np.percentile(disp_um, 90)) if disp_um.size else float("nan"),
        "bar_disp_p95_um": float(np.percentile(disp_um, 95)) if disp_um.size else float("nan"),
        "bar_disp_max_um": float(np.max(disp_um)) if disp_um.size else float("nan"),
    }

    for field_name, field in (
        ("vx", vx),
        ("vy", vy),
        ("div", divergence),
        ("curl", curl),
    ):
        bands = dct_radial_band_energy_fractions(field, n_bands=n_bands, support_mask=support_mask)
        for i, value in enumerate(bands):
            record[f"dct_{field_name}_band_{i:02d}"] = float(value)

    return record


def apply_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")
    if "backend" in out.columns:
        out["backend"] = out["backend"].astype("category")
    return out


def upsert_ot_feature_matrix_parquet(
    parquet_path: Path,
    incoming_rows: pd.DataFrame | Iterable[Mapping],
    key_columns: Sequence[str] = FEATURE_KEY_COLUMNS,
) -> pd.DataFrame:
    """Idempotent upsert writer for `ot_feature_matrix.parquet`."""
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(incoming_rows, pd.DataFrame):
        incoming = incoming_rows.copy()
    else:
        incoming = pd.DataFrame(list(incoming_rows))
    if incoming.empty:
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        return incoming

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
    else:
        existing = pd.DataFrame()

    merged = upsert_pair_metrics(existing, incoming, key_columns=key_columns)
    merged = apply_feature_dtypes(merged)
    merged.to_parquet(parquet_path, index=False)
    return merged
