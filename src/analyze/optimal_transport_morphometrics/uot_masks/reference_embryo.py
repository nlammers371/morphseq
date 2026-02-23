"""Reference embryo velocity fields and deviation metrics.

Build a reference (WT-like) velocity field from multiple embryos,
then compute deviation of individual embryos from that reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ReferenceField:
    """Mean velocity field and mass maps for a single frame pair."""
    velocity_yx: np.ndarray          # (H, W, 2), mean velocity
    mass_created: np.ndarray         # (H, W), mean mass created
    mass_destroyed: np.ndarray       # (H, W), mean mass destroyed
    support_mask: np.ndarray         # (H, W), bool: union of all input supports
    n_embryos: int                   # Number of embryos averaged


@dataclass
class ReferenceTimeseries:
    """Collection of ReferenceField objects keyed by frame pair."""
    fields: Dict[Tuple[int, int], ReferenceField] = field(default_factory=dict)

    def __getitem__(self, key: Tuple[int, int]) -> ReferenceField:
        return self.fields[key]

    def __contains__(self, key: Tuple[int, int]) -> bool:
        return key in self.fields

    @property
    def frame_pairs(self) -> List[Tuple[int, int]]:
        return sorted(self.fields.keys())


def build_reference_field(
    results: List,
    method: str = "mean",
) -> ReferenceField:
    """Build a reference field from multiple canonical-grid OT results.

    Args:
        results: List of canonical-grid results (same frame pair, different embryos).
            Each must have mass_created_canon, mass_destroyed_canon,
            velocity_canon_px_per_step_yx of the same shape.
        method: Aggregation method ("mean" only for now).

    Returns:
        ReferenceField with averaged fields.
    """
    if not results:
        raise ValueError("No results provided to build reference field.")
    if method != "mean":
        raise ValueError(f"Unsupported method: {method}. Only 'mean' is supported.")

    shape = results[0].mass_created_canon.shape
    vel_shape = results[0].velocity_canon_px_per_step_yx.shape

    vel_stack = np.zeros((*vel_shape,), dtype=np.float64)
    created_stack = np.zeros(shape, dtype=np.float64)
    destroyed_stack = np.zeros(shape, dtype=np.float64)
    support_union = np.zeros(shape, dtype=bool)

    for r in results:
        assert r.mass_created_canon.shape == shape, \
            f"Shape mismatch: {r.mass_created_canon.shape} vs {shape}"
        vel_stack += r.velocity_canon_px_per_step_yx.astype(np.float64)
        created_stack += r.mass_created_canon.astype(np.float64)
        destroyed_stack += r.mass_destroyed_canon.astype(np.float64)

        # Build support from non-zero data
        vel_mag = np.linalg.norm(r.velocity_canon_px_per_step_yx, axis=-1)
        support_union |= (r.mass_created_canon > 0) | (r.mass_destroyed_canon > 0) | (vel_mag > 0)

    n = len(results)
    return ReferenceField(
        velocity_yx=(vel_stack / n).astype(np.float32),
        mass_created=(created_stack / n).astype(np.float32),
        mass_destroyed=(destroyed_stack / n).astype(np.float32),
        support_mask=support_union,
        n_embryos=n,
    )


def compute_deviation_from_reference(
    result,
    reference: ReferenceField,
) -> Dict[str, float]:
    """Compute deviation metrics of a single canonical-grid result from a reference.

    Args:
        result: UOTResult to compare against reference.
        reference: ReferenceField to compare against.

    Returns:
        Dict with keys:
        - rmse_velocity: RMSE of velocity field on support
        - cosine_similarity: Mean per-pixel cosine similarity on support
        - rmse_mass_created: RMSE of mass created on support
        - rmse_mass_destroyed: RMSE of mass destroyed on support
    """
    mask = reference.support_mask

    # Velocity RMSE
    v_test = result.velocity_canon_px_per_step_yx.astype(np.float64)
    v_ref = reference.velocity_yx.astype(np.float64)
    diff = v_test - v_ref
    if mask.any():
        rmse_vel = float(np.sqrt(np.mean(diff[mask] ** 2)))
    else:
        rmse_vel = 0.0

    # Cosine similarity (per-pixel, then mean)
    if mask.any():
        v_test_flat = v_test[mask]  # (N, 2)
        v_ref_flat = v_ref[mask]    # (N, 2)
        dot = np.sum(v_test_flat * v_ref_flat, axis=1)
        norm_test = np.linalg.norm(v_test_flat, axis=1)
        norm_ref = np.linalg.norm(v_ref_flat, axis=1)
        denom = norm_test * norm_ref
        # Only compute cosine where both vectors are non-zero
        valid = denom > 1e-12
        if valid.any():
            cos_sim = float(np.mean(dot[valid] / denom[valid]))
        else:
            cos_sim = 1.0  # Both zero → perfect agreement
    else:
        cos_sim = 1.0

    # Mass RMSE
    mc_diff = result.mass_created_canon.astype(np.float64) - reference.mass_created.astype(np.float64)
    md_diff = result.mass_destroyed_canon.astype(np.float64) - reference.mass_destroyed.astype(np.float64)
    if mask.any():
        rmse_mc = float(np.sqrt(np.mean(mc_diff[mask] ** 2)))
        rmse_md = float(np.sqrt(np.mean(md_diff[mask] ** 2)))
    else:
        rmse_mc = 0.0
        rmse_md = 0.0

    return {
        "rmse_velocity": rmse_vel,
        "cosine_similarity": cos_sim,
        "rmse_mass_created": rmse_mc,
        "rmse_mass_destroyed": rmse_md,
    }


def compute_residual_field(
    result,
    reference: ReferenceField,
) -> np.ndarray:
    """Compute residual velocity field (test - reference).

    Returns:
        (H, W, 2) residual velocity field.
    """
    return (result.velocity_canon_px_per_step_yx.astype(np.float64) -
            reference.velocity_yx.astype(np.float64)).astype(np.float32)


def deviation_timeseries(
    embryo_results: Dict[Tuple[int, int], object],
    reference: ReferenceTimeseries,
    embryo_id: str = "",
) -> pd.DataFrame:
    """Compute deviation metrics across a timeseries.

    Args:
        embryo_results: Dict mapping (frame_src, frame_tgt) → UOTResult
        reference: ReferenceTimeseries with matching frame pairs
        embryo_id: Optional label for the embryo

    Returns:
        DataFrame with columns: embryo_id, frame_src, frame_tgt,
        rmse_velocity, cosine_similarity, rmse_mass_created, rmse_mass_destroyed
    """
    rows = []
    for (fs, ft), result in sorted(embryo_results.items()):
        if (fs, ft) not in reference:
            continue
        ref_field = reference[(fs, ft)]
        dev = compute_deviation_from_reference(result, ref_field)
        dev["embryo_id"] = embryo_id
        dev["frame_src"] = fs
        dev["frame_tgt"] = ft
        rows.append(dev)

    return pd.DataFrame(rows)
