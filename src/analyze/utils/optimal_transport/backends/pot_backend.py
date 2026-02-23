"""POT-based unbalanced OT backend (CPU)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import ot

from analyze.utils.optimal_transport.backends.base import UOTBackend, BackendResult
from analyze.utils.optimal_transport.config import UOTSupport, UOTConfig


class POTBackend(UOTBackend):
    """Unbalanced OT backend using POT's Sinkhorn implementation."""

    def solve(self, src: UOTSupport, tgt: UOTSupport, config: UOTConfig) -> BackendResult:
        coords_src = src.coords_yx.astype(np.float64) * float(config.coord_scale)
        coords_tgt = tgt.coords_yx.astype(np.float64) * float(config.coord_scale)
        weights_src = src.weights.astype(np.float64)
        weights_tgt = tgt.weights.astype(np.float64)

        m_src = float(weights_src.sum())
        m_tgt = float(weights_tgt.sum())
        if m_src <= 0 or m_tgt <= 0:
            raise ValueError("Source/target mass must be positive for UOT solve.")

        a = weights_src / m_src
        b = weights_tgt / m_tgt

        if config.metric == "sqeuclidean":
            cost = ot.dist(coords_src, coords_tgt, metric="sqeuclidean")
        elif config.metric == "euclidean":
            cost = ot.dist(coords_src, coords_tgt, metric="euclidean")
        else:
            raise ValueError(f"Unsupported metric: {config.metric}")

        coupling, log = ot.unbalanced.sinkhorn_unbalanced(
            a,
            b,
            cost,
            reg=config.epsilon,
            reg_m=config.marginal_relaxation,
            log=True,
        )

        coupling = np.asarray(coupling, dtype=np.float64)
        coupling_rescaled = coupling * m_src
        weighted_cost = coupling * cost
        cost_value = float(weighted_cost.sum() * m_src)
        cost_per_src = (weighted_cost.sum(axis=1) * m_src).astype(np.float64)
        cost_per_tgt = (weighted_cost.sum(axis=0) * m_src).astype(np.float64)

        diagnostics: Dict = {
            "m_src": m_src,
            "m_tgt": m_tgt,
            "reg": config.epsilon,
            "reg_m": config.marginal_relaxation,
            "coord_scale": float(config.coord_scale),
            "log": log,
        }

        return BackendResult(
            coupling=coupling_rescaled if config.store_coupling else None,
            cost=cost_value,
            diagnostics=diagnostics,
            cost_per_src=cost_per_src,
            cost_per_tgt=cost_per_tgt,
        )
