"""Storage contract helpers for OT pair metrics and field artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy may be optional
    sp = None


FEATURE_SCHEMA_VERSION = "2.0.0"
PAIR_KEY_COLUMNS = ("run_id", "pair_id")

# Repeated low-cardinality config columns to compress parquet size.
_CATEGORICAL_COLUMNS = (
    "backend",
    "metric",
    "canonical_grid_align_mode",
    "mass_mode",
    "align_mode",
)

# Identifier/meta columns can arrive as int or str depending on upstream CSV typing.
# Force stable nullable-string dtypes to avoid mixed-type parquet write failures.
_STRING_COLUMNS = (
    "feature_schema_version",
    "run_id",
    "pair_id",
    "error_message",
    "src_embryo_id",
    "tgt_embryo_id",
    "src_experiment_id",
    "tgt_experiment_id",
    "src_experiment_date",
    "tgt_experiment_date",
    "src_well",
    "tgt_well",
    "src_snip_id",
    "tgt_snip_id",
)


def _normalize_value(value):
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def build_pair_id(
    src_embryo_id: str,
    tgt_embryo_id: str,
    src_frame_index: int,
    tgt_frame_index: int,
) -> str:
    return f"{src_embryo_id}__f{int(src_frame_index):04d}__to__{tgt_embryo_id}__f{int(tgt_frame_index):04d}"


def _cfg_to_dict(config) -> Dict:
    if config is None:
        return {}
    if is_dataclass(config):
        raw = asdict(config)
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = {}
    return {k: _normalize_value(v) for k, v in raw.items()}


def _meta_value(meta: Optional[Mapping], key: str):
    if not meta:
        return None
    return meta.get(key)


def build_pair_metrics_record(
    *,
    run_id: str,
    pair_id: str,
    result,
    src_meta: Optional[Mapping] = None,
    tgt_meta: Optional[Mapping] = None,
    config=None,
    backend: str = "unknown",
    runtime_sec: Optional[float] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> Dict:
    """Build one analysis row for `ot_pair_metrics.parquet`."""
    metrics = {}
    diagnostics = result.diagnostics or {}
    if isinstance(diagnostics, Mapping):
        metrics = dict(diagnostics.get("metrics", {}) or {})
    cfg = _cfg_to_dict(config)

    src_frame_index = _meta_value(src_meta, "frame_index")
    tgt_frame_index = _meta_value(tgt_meta, "frame_index")
    src_rel_t = _meta_value(src_meta, "relative_time_s")
    tgt_rel_t = _meta_value(tgt_meta, "relative_time_s")
    delta_time_s = None
    if src_rel_t is not None and tgt_rel_t is not None:
        delta_time_s = float(tgt_rel_t) - float(src_rel_t)

    record = {
        "feature_schema_version": feature_schema_version,
        "run_id": run_id,
        "pair_id": pair_id,
        "success": bool(success),
        "error_message": error_message,
        "runtime_sec": runtime_sec,
        "backend": str(backend),
        "cost": float(result.cost),
        "n_support_src": int(len(result.support_src_yx)),
        "n_support_tgt": int(len(result.support_tgt_yx)),
        "src_embryo_id": _meta_value(src_meta, "embryo_id"),
        "tgt_embryo_id": _meta_value(tgt_meta, "embryo_id"),
        "src_frame_index": src_frame_index,
        "tgt_frame_index": tgt_frame_index,
        "src_experiment_id": _meta_value(src_meta, "experiment_id"),
        "tgt_experiment_id": _meta_value(tgt_meta, "experiment_id"),
        "src_experiment_date": _meta_value(src_meta, "experiment_date"),
        "tgt_experiment_date": _meta_value(tgt_meta, "experiment_date"),
        "src_well": _meta_value(src_meta, "well"),
        "tgt_well": _meta_value(tgt_meta, "well"),
        "src_snip_id": _meta_value(src_meta, "snip_id"),
        "tgt_snip_id": _meta_value(tgt_meta, "snip_id"),
        "src_relative_time_s": src_rel_t,
        "tgt_relative_time_s": tgt_rel_t,
        "delta_time_s": delta_time_s,
        "yolk_src_present": bool(src_meta and ("yolk_mask" in src_meta) and (src_meta["yolk_mask"] is not None)),
        "yolk_tgt_present": bool(tgt_meta and ("yolk_mask" in tgt_meta) and (tgt_meta["yolk_mask"] is not None)),
        "epsilon": cfg.get("epsilon"),
        "marginal_relaxation": cfg.get("marginal_relaxation"),
        "metric": cfg.get("metric"),
        "downsample_factor": cfg.get("downsample_factor"),
        "coord_scale": cfg.get("coord_scale"),
        "mass_mode": cfg.get("mass_mode"),
        "align_mode": cfg.get("align_mode"),
        "use_pair_frame": cfg.get("use_pair_frame"),
        "use_canonical_grid": cfg.get("use_canonical_grid"),
        "canonical_grid_align_mode": cfg.get("canonical_grid_align_mode"),
    }

    for key, value in metrics.items():
        record[key] = _normalize_value(value)
    return record


def apply_contract_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dataframe dtypes expected by OT pair metrics storage."""
    out = df.copy()
    for col in _STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")
    for col in _CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("category")
    if "success" in out.columns:
        out["success"] = out["success"].astype("boolean")
    return out


def upsert_pair_metrics(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    key_columns: Sequence[str] = PAIR_KEY_COLUMNS,
) -> pd.DataFrame:
    """Upsert incoming rows by composite key, keeping the latest row per key."""
    key_columns = tuple(key_columns)
    if incoming.empty:
        return apply_contract_dtypes(existing)

    for col in key_columns:
        if col not in incoming.columns:
            raise ValueError(f"Incoming dataframe missing key column: {col}")
    if not existing.empty:
        for col in key_columns:
            if col not in existing.columns:
                raise ValueError(f"Existing dataframe missing key column: {col}")

    incoming_dedup = incoming.drop_duplicates(subset=list(key_columns), keep="last")
    combined = pd.concat([existing, incoming_dedup], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=list(key_columns), keep="last")
    if combined.duplicated(subset=list(key_columns)).any():
        raise RuntimeError(f"Duplicate keys detected after upsert on {key_columns}")
    return apply_contract_dtypes(combined)


def upsert_ot_pair_metrics_parquet(
    parquet_path: Path,
    incoming_rows: pd.DataFrame | Iterable[Mapping],
    key_columns: Sequence[str] = PAIR_KEY_COLUMNS,
) -> pd.DataFrame:
    """Idempotent upsert writer for `ot_pair_metrics.parquet`."""
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
    merged.to_parquet(parquet_path, index=False)
    return merged


def compute_barycentric_projection(
    result_work,
    min_mass: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """Compute source-support barycentric targets from a solved coupling."""
    if result_work.coupling is None:
        raise ValueError("Result coupling is None; cannot compute barycentric projection.")

    src = np.asarray(result_work.support_src_yx, dtype=np.float64)
    tgt = np.asarray(result_work.support_tgt_yx, dtype=np.float64)
    n_src = src.shape[0]

    if sp is not None and sp.issparse(result_work.coupling):
        coo = result_work.coupling.tocoo()
        transported_mass = np.zeros(n_src, dtype=np.float64)
        weighted_tgt = np.zeros((n_src, 2), dtype=np.float64)
        np.add.at(transported_mass, coo.row, coo.data)
        np.add.at(weighted_tgt, coo.row, coo.data[:, None] * tgt[coo.col])
    else:
        coupling = np.asarray(result_work.coupling, dtype=np.float64)
        transported_mass = coupling.sum(axis=1)
        weighted_tgt = coupling @ tgt

    bary_tgt = src.copy()
    valid = transported_mass > float(min_mass)
    bary_tgt[valid] = weighted_tgt[valid] / transported_mass[valid, None]
    bary_vel = bary_tgt - src

    return {
        "src_yx": src.astype(np.float32),
        "barycentric_tgt_yx": bary_tgt.astype(np.float32),
        "barycentric_velocity_yx": bary_vel.astype(np.float32),
        "transported_mass_src": transported_mass.astype(np.float32),
    }


def save_pair_artifacts(
    *,
    result_work,
    result_canon=None,
    artifact_root: Path,
    pair_id: str,
    float_dtype: str = "float32",
    include_barycentric: bool = True,
) -> Dict[str, Path]:
    """Save per-pair field artifacts and optional barycentric projection.

    Contract:
    - `result_work` is a `UOTResultWork` (work-grid fields + coupling + supports).
    - `result_canon` is optional `UOTResultCanonical` (canonical-grid rasters only).
    """
    artifact_root = Path(artifact_root)
    pair_dir = artifact_root / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    float_dtype_np = np.dtype(float_dtype)

    fields_path = pair_dir / "fields.npz"
    fields_payload = {
        "velocity_work_px_per_step_yx": np.asarray(result_work.velocity_work_px_per_step_yx, dtype=float_dtype_np),
        "mass_created_work": np.asarray(result_work.mass_created_work, dtype=float_dtype_np),
        "mass_destroyed_work": np.asarray(result_work.mass_destroyed_work, dtype=float_dtype_np),
        "support_src_yx": np.asarray(result_work.support_src_yx, dtype=np.float32),
        "support_tgt_yx": np.asarray(result_work.support_tgt_yx, dtype=np.float32),
    }
    if getattr(result_work, "cost_src_work", None) is not None:
        fields_payload["cost_src_work"] = np.asarray(result_work.cost_src_work, dtype=float_dtype_np)
    if getattr(result_work, "cost_tgt_work", None) is not None:
        fields_payload["cost_tgt_work"] = np.asarray(result_work.cost_tgt_work, dtype=float_dtype_np)

    if result_canon is not None:
        fields_payload.update(
            {
                "velocity_canon_px_per_step_yx": np.asarray(
                    result_canon.velocity_canon_px_per_step_yx, dtype=float_dtype_np
                ),
                "mass_created_canon": np.asarray(result_canon.mass_created_canon, dtype=float_dtype_np),
                "mass_destroyed_canon": np.asarray(result_canon.mass_destroyed_canon, dtype=float_dtype_np),
            }
        )
        if getattr(result_canon, "cost_src_canon", None) is not None:
            fields_payload["cost_src_canon"] = np.asarray(result_canon.cost_src_canon, dtype=float_dtype_np)
        if getattr(result_canon, "cost_tgt_canon", None) is not None:
            fields_payload["cost_tgt_canon"] = np.asarray(result_canon.cost_tgt_canon, dtype=float_dtype_np)
    np.savez_compressed(fields_path, **fields_payload)

    paths: Dict[str, Path] = {"fields": fields_path}
    if include_barycentric and result_work.coupling is not None:
        bary = compute_barycentric_projection(result_work)
        bary_path = pair_dir / "barycentric_projection.npz"
        np.savez_compressed(
            bary_path,
            src_yx=bary["src_yx"].astype(float_dtype_np),
            barycentric_tgt_yx=bary["barycentric_tgt_yx"].astype(float_dtype_np),
            barycentric_velocity_yx=bary["barycentric_velocity_yx"].astype(float_dtype_np),
            transported_mass_src=bary["transported_mass_src"].astype(float_dtype_np),
        )
        paths["barycentric"] = bary_path

    metadata = {
        "pair_id": pair_id,
        "float_dtype": str(float_dtype_np),
        "work_velocity_shape": list(result_work.velocity_work_px_per_step_yx.shape),
        "work_mass_shape": list(result_work.mass_created_work.shape),
        "n_support_src": int(len(result_work.support_src_yx)),
        "n_support_tgt": int(len(result_work.support_tgt_yx)),
        "has_coupling": result_work.coupling is not None,
        "has_pair_frame": result_work.pair_frame is not None,
        "has_canonical_fields": result_canon is not None,
    }
    if result_work.pair_frame is not None:
        metadata["pair_frame"] = {
            "canon_shape_hw": list(result_work.pair_frame.canon_shape_hw),
            "work_shape_hw": list(result_work.pair_frame.work_shape_hw),
            "downsample_factor": int(result_work.pair_frame.downsample_factor),
            "px_size_um": float(result_work.pair_frame.px_size_um),
        }
    if result_canon is not None:
        metadata["canonical"] = {
            "canonical_shape_hw": list(result_canon.mass_created_canon.shape),
            "canonical_um_per_px": float(result_canon.canonical_um_per_px),
        }
    metadata_path = pair_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    paths["metadata"] = metadata_path
    return paths
