"""OTT-JAX based unbalanced OT backend (CPU/GPU)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from analyze.utils.optimal_transport.backends.base import UOTBackend, BackendResult
from analyze.utils.optimal_transport.config import UOTSupport, UOTConfig

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from ott.geometry import pointcloud, costs
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    _OTT_AVAILABLE = True
except ImportError:
    _OTT_AVAILABLE = False


def ott_available() -> bool:
    """Check if ott-jax is importable."""
    return _OTT_AVAILABLE


class OTTBackend(UOTBackend):
    """Unbalanced OT backend using ott-jax Sinkhorn.

    Matches POTBackend normalization contract:
    - Normalizes weights (a, b) to sum to 1
    - Solves on normalized distributions
    - Rescales coupling by m_src

    No GPU assertion at import time. Works on CPU or GPU depending on
    jax device availability.
    """

    def __init__(self, max_iterations: int = 2000, threshold: float = 1e-4):
        if not _OTT_AVAILABLE:
            raise ImportError(
                "ott-jax is required for OTTBackend. "
                "Install with: pip install ott-jax"
            )
        self.max_iterations = max_iterations
        self.threshold = threshold

        # Log device info (warning, not error, if no GPU)
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == "gpu"]
            if gpu_devices:
                logger.info("OTTBackend: GPU available (%d device(s))", len(gpu_devices))
            else:
                logger.warning("OTTBackend: No GPU found, running on CPU")
        except Exception:
            logger.warning("OTTBackend: Could not query JAX devices")

    def solve(self, src: UOTSupport, tgt: UOTSupport, config: UOTConfig) -> BackendResult:
        coords_src = src.coords_yx.astype(np.float64) * float(config.coord_scale)
        coords_tgt = tgt.coords_yx.astype(np.float64) * float(config.coord_scale)
        weights_src = src.weights.astype(np.float64)
        weights_tgt = tgt.weights.astype(np.float64)

        m_src = float(weights_src.sum())
        m_tgt = float(weights_tgt.sum())
        if m_src <= 0 or m_tgt <= 0:
            raise ValueError("Source/target mass must be positive for UOT solve.")

        # Normalize to probability distributions (matching POTBackend)
        a = weights_src / m_src
        b = weights_tgt / m_tgt

        # Convert to JAX arrays
        x = jnp.array(coords_src, dtype=jnp.float32)
        y = jnp.array(coords_tgt, dtype=jnp.float32)
        a_jax = jnp.array(a, dtype=jnp.float32)
        b_jax = jnp.array(b, dtype=jnp.float32)

        # Select cost function
        if config.metric == "sqeuclidean":
            cost_fn = costs.SqEuclidean()
        elif config.metric == "euclidean":
            cost_fn = costs.Euclidean()
        else:
            raise ValueError(f"Unsupported metric: {config.metric}")

        # Build geometry
        geom = pointcloud.PointCloud(x, y, cost_fn=cost_fn, epsilon=config.epsilon)

        # Convert reg_m to tau: tau = reg_m / (reg_m + epsilon)
        tau = config.marginal_relaxation / (config.marginal_relaxation + config.epsilon)

        # Build problem and solver
        prob = linear_problem.LinearProblem(geom, a=a_jax, b=b_jax, tau_a=tau, tau_b=tau)
        solver = sinkhorn.Sinkhorn(
            max_iterations=self.max_iterations,
            threshold=self.threshold,
        )
        out = solver(prob)

        # Extract coupling matrix
        coupling_jax = out.matrix
        coupling_np = np.array(coupling_jax, dtype=np.float64)

        # Rescale by m_src (matching POTBackend)
        coupling_rescaled = coupling_np * m_src

        # Compute cost (matching POTBackend: weighted_cost * m_src)
        cost_matrix = np.array(geom.cost_matrix, dtype=np.float64)
        weighted_cost = coupling_np * cost_matrix
        cost_value = float(weighted_cost.sum() * m_src)
        cost_per_src = (weighted_cost.sum(axis=1) * m_src).astype(np.float64)
        cost_per_tgt = (weighted_cost.sum(axis=0) * m_src).astype(np.float64)

        diagnostics: Dict = {
            "m_src": m_src,
            "m_tgt": m_tgt,
            "reg": config.epsilon,
            "reg_m": config.marginal_relaxation,
            "coord_scale": float(config.coord_scale),
            "converged": bool(out.converged),
            "n_iters": int(out.n_iters) if hasattr(out, "n_iters") else None,
            "tau": tau,
        }

        return BackendResult(
            coupling=coupling_rescaled if config.store_coupling else None,
            cost=cost_value,
            diagnostics=diagnostics,
            cost_per_src=cost_per_src,
            cost_per_tgt=cost_per_tgt,
        )

    def solve_batch(
        self,
        problems: List[tuple],
        config: UOTConfig,
    ) -> List[BackendResult]:
        """Solve a batch of UOT problems sequentially.

        Args:
            problems: List of (src: UOTSupport, tgt: UOTSupport) tuples
            config: Shared UOT config

        Returns:
            List of BackendResult, one per problem
        """
        results = []
        for src, tgt in problems:
            results.append(self.solve(src, tgt, config))
        return results
