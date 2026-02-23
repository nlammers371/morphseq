"""Summary metrics for UOT transport results."""

from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import numpy as np

from .config import Coupling

if TYPE_CHECKING:
    from .pair_frame import PairFrameGeometry


def compute_transport_metrics(
    cost: float,
    coupling: Coupling,
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    metric: str,
    m_src: float = None,
    m_tgt: float = None,
    pair_frame: "PairFrameGeometry" = None,
    coord_scale: float = 1.0,
) -> Dict[str, float]:
    if coupling is None:
        transported_mass = float("nan")
    else:
        transported_mass = float(np.asarray(coupling).sum())

    created_mass = float(mass_created_hw.sum())
    destroyed_mass = float(mass_destroyed_hw.sum())

    mean_transport_cost = float("nan")
    mean_transport_distance = float("nan")
    if transported_mass > 0:
        mean_transport_cost = float(cost / transported_mass)
        if metric == "sqeuclidean":
            mean_transport_distance = float(np.sqrt(mean_transport_cost))
        else:
            mean_transport_distance = mean_transport_cost

    # Add percentage-based metrics if source and target masses are provided
    created_mass_pct = float("nan")
    destroyed_mass_pct = float("nan")
    proportion_transported = float("nan")

    if m_tgt is not None and m_tgt > 0:
        created_mass_pct = 100.0 * created_mass / m_tgt

    if m_src is not None and m_src > 0:
        destroyed_mass_pct = 100.0 * destroyed_mass / m_src

    if m_src is not None and m_tgt is not None and min(m_src, m_tgt) > 0:
        proportion_transported = transported_mass / min(m_src, m_tgt)

    # Physical area calculations using pair_frame
    created_area_um2 = float("nan")
    destroyed_area_um2 = float("nan")
    if pair_frame is not None:
        created_area_um2 = float(created_mass * pair_frame.px_area_um2)
        destroyed_area_um2 = float(destroyed_mass * pair_frame.px_area_um2)

    # Physical transport cost conversions (if we know the work pixel size)
    total_transport_cost_um2 = float("nan")
    mean_transport_cost_um2 = float("nan")
    mean_transport_distance_um = float("nan")
    specific_transport_cost_um2_per_mass = float("nan")

    total_transport_cost_um = float("nan")
    mean_transport_cost_um = float("nan")
    specific_transport_cost_um_per_mass = float("nan")

    if pair_frame is not None and coord_scale and coord_scale > 0:
        scale = float(pair_frame.work_px_size_um / coord_scale)
        if metric == "sqeuclidean":
            scale2 = scale ** 2
            total_transport_cost_um2 = float(cost * scale2)
            mean_transport_cost_um2 = float(mean_transport_cost * scale2)
            mean_transport_distance_um = float(mean_transport_distance * scale)
            if m_src is not None and m_src > 0:
                specific_transport_cost_um2_per_mass = float(total_transport_cost_um2 / m_src)
        elif metric == "euclidean":
            total_transport_cost_um = float(cost * scale)
            mean_transport_cost_um = float(mean_transport_cost * scale)
            mean_transport_distance_um = float(mean_transport_distance * scale)
            if m_src is not None and m_src > 0:
                specific_transport_cost_um_per_mass = float(total_transport_cost_um / m_src)

    # Activity metrics (cost per unit mass)
    specific_transport_cost = float("nan")
    if m_src is not None and m_src > 0:
        specific_transport_cost = float(cost / m_src)

    # Transported mass percentages
    transported_mass_pct_src = float("nan")
    transported_mass_pct_tgt = float("nan")
    transported_mass_pct_min = float("nan")
    if m_src is not None and m_src > 0:
        transported_mass_pct_src = 100.0 * transported_mass / m_src
    if m_tgt is not None and m_tgt > 0:
        transported_mass_pct_tgt = 100.0 * transported_mass / m_tgt
    if m_src is not None and m_tgt is not None and min(m_src, m_tgt) > 0:
        transported_mass_pct_min = 100.0 * transported_mass / min(m_src, m_tgt)

    return {
        "total_transport_cost": float(cost),
        "transported_mass": transported_mass,
        "created_mass": created_mass,
        "destroyed_mass": destroyed_mass,
        "mean_transport_cost": mean_transport_cost,
        "mean_transport_distance": mean_transport_distance,
        "specific_transport_cost": specific_transport_cost,
        "transported_mass_pct_src": transported_mass_pct_src,
        "transported_mass_pct_tgt": transported_mass_pct_tgt,
        "transported_mass_pct_min": transported_mass_pct_min,
        "created_mass_pct": created_mass_pct,
        "destroyed_mass_pct": destroyed_mass_pct,
        "proportion_transported": proportion_transported,
        "created_area_um2": created_area_um2,
        "destroyed_area_um2": destroyed_area_um2,
        "total_transport_cost_um2": total_transport_cost_um2,
        "mean_transport_cost_um2": mean_transport_cost_um2,
        "mean_transport_distance_um": mean_transport_distance_um,
        "specific_transport_cost_um2_per_mass": specific_transport_cost_um2_per_mass,
        "total_transport_cost_um": total_transport_cost_um,
        "mean_transport_cost_um": mean_transport_cost_um,
        "specific_transport_cost_um_per_mass": specific_transport_cost_um_per_mass,
    }


def summarize_metrics(
    cost: float,
    coupling: Coupling,
    mass_created_hw: np.ndarray,
    mass_destroyed_hw: np.ndarray,
    metric: str,
    m_src: float = None,
    m_tgt: float = None,
    pair_frame: "PairFrameGeometry" = None,
    coord_scale: float = 1.0,
) -> Dict[str, float]:
    return compute_transport_metrics(
        cost=cost,
        coupling=coupling,
        mass_created_hw=mass_created_hw,
        mass_destroyed_hw=mass_destroyed_hw,
        metric=metric,
        m_src=m_src,
        m_tgt=m_tgt,
        pair_frame=pair_frame,
        coord_scale=coord_scale,
    )
