"""OTT-JAX based unbalanced OT backend (CPU/GPU)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

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


# Module-level flag: warn exactly once per process when solve() is called
# for a single pair (vs. using solve_batch() to amortize JIT compile cost).
_SINGLE_SOLVE_WARNED = False


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

    JIT caching: The first call for a given (n_src, n_tgt, epsilon, tau, metric)
    shape bucket triggers XLA compilation (~10–60s on CPU). Subsequent calls with
    the same shape reuse the cached compiled function.  Use solve_batch() to
    amortize this cost over many pairs.
    """

    def __init__(self, max_iterations: int = 2000, threshold: float = 1e-4):
        if not _OTT_AVAILABLE:
            raise ImportError(
                "ott-jax is required for OTTBackend. "
                "Install with: pip install ott-jax"
            )
        self.max_iterations = max_iterations
        self.threshold = threshold

        # JIT cache: keyed by (n_src, n_tgt, dtype, epsilon, tau_a, tau_b, metric,
        #                       max_iterations, threshold)
        self._jit_cache: dict[tuple, Any] = {}

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

    # ------------------------------------------------------------------
    # JIT cache helpers
    # ------------------------------------------------------------------

    def _make_cache_key(
        self, n_src: int, n_tgt: int, config: UOTConfig, dtype: str
    ) -> tuple:
        """Build a cache key that covers everything affecting the computation graph."""
        # Mirror the exact tau formula used in _solve_with_jit — must stay in sync.
        tau = config.marginal_relaxation / (config.marginal_relaxation + config.epsilon)
        return (
            n_src,
            n_tgt,
            dtype,
            config.epsilon,
            tau,  # tau_a
            tau,  # tau_b — symmetric for now; explicit for future asymmetry
            config.metric,
            self.max_iterations,
            self.threshold,
        )

    def _get_jit_solver(self, key: tuple) -> Any:
        """Return the JIT-compiled solver for this key, compiling if needed."""
        if key not in self._jit_cache:
            solver = sinkhorn.Sinkhorn(
                max_iterations=self.max_iterations,
                threshold=self.threshold,
            )

            def _solve_fn(prob):
                return solver(prob)

            self._jit_cache[key] = jax.jit(_solve_fn)
        return self._jit_cache[key]

    # ------------------------------------------------------------------
    # Core solve (shared by solve() and solve_batch())
    # ------------------------------------------------------------------

    def _solve_with_jit(
        self, src: UOTSupport, tgt: UOTSupport, config: UOTConfig
    ) -> BackendResult:
        """Solve one UOT problem, reusing the JIT-compiled function when possible."""
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

        # Build problem
        prob = linear_problem.LinearProblem(geom, a=a_jax, b=b_jax, tau_a=tau, tau_b=tau)

        # Get (or compile) JIT solver for this shape/config bucket
        key = self._make_cache_key(len(src.coords_yx), len(tgt.coords_yx), config, "float32")
        jit_fn = self._get_jit_solver(key)
        out = jit_fn(prob)

        # Block until ready — JAX dispatches async; must sync before timing or reading results.
        # Use leaf-based blocking (robust across ott-jax versions where field names vary).
        leaves = jax.tree_util.tree_leaves(out)
        if leaves:
            leaves[0].block_until_ready()

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, src: UOTSupport, tgt: UOTSupport, config: UOTConfig) -> BackendResult:
        """Solve a single UOT problem.

        On CPU the first call will trigger JIT compilation (10–60s). For many pairs,
        use solve_batch() to amortize compilation cost. For a single pair on CPU,
        POTBackend avoids compile overhead entirely.
        """
        global _SINGLE_SOLVE_WARNED
        if not _SINGLE_SOLVE_WARNED:
            logger.warning(
                "OTTBackend.solve() called for a single pair. On CPU, the first call "
                "will JIT-compile (10–60s). For many pairs, use solve_batch() to "
                "amortize compilation. For a single pair on CPU, POTBackend avoids "
                "compile overhead entirely."
            )
            _SINGLE_SOLVE_WARNED = True
        return self._solve_with_jit(src, tgt, config)

    def solve_batch(
        self,
        problems: List[tuple],
        config: UOTConfig,
    ) -> List[BackendResult]:
        """Solve a batch of UOT problems, reusing the JIT-compiled solver per shape bucket.

        Pairs are grouped by (n_src, n_tgt) to maximize JIT reuse: one compile per
        unique shape, amortized over all pairs in that bucket.  This mirrors the
        bucketing already done by solve_working_grid_batch() at the outer pipeline level.

        Args:
            problems: List of (src: UOTSupport, tgt: UOTSupport) tuples
            config: Shared UOT config

        Returns:
            List of BackendResult, one per problem (order preserved)
        """
        from collections import defaultdict

        # Bucket by (n_src, n_tgt) to reuse JIT-compiled solver within each bucket.
        buckets: dict[tuple, list] = defaultdict(list)
        for i, (src, tgt) in enumerate(problems):
            shape_key = (len(src.coords_yx), len(tgt.coords_yx))
            buckets[shape_key].append((i, src, tgt))

        results: List[Optional[BackendResult]] = [None] * len(problems)
        for (n_src, n_tgt), items in buckets.items():
            for idx, src, tgt in items:
                results[idx] = self._solve_with_jit(src, tgt, config)
        return results

    def warmup(
        self, problems: List[tuple], config: UOTConfig
    ) -> Dict[tuple, float]:
        """Pre-compile JIT functions for each unique shape bucket.

        Runs one solve per distinct (n_src, n_tgt) shape to trigger XLA compilation
        before the timed benchmark loop.

        Args:
            problems: List of (src: UOTSupport, tgt: UOTSupport) tuples
            config: UOT config

        Returns:
            Dict mapping cache_key → compile_time_s for each newly compiled bucket
        """
        import time

        seen: Dict[tuple, float] = {}
        for src, tgt in problems:
            key = self._make_cache_key(
                len(src.coords_yx), len(tgt.coords_yx), config, "float32"
            )
            if key not in seen:
                t0 = time.perf_counter()
                self._solve_with_jit(src, tgt, config)  # triggers compile on first call
                seen[key] = time.perf_counter() - t0
        return seen

    def cache_info(self) -> Dict:
        """Return summary of the JIT cache contents."""
        return {
            "n_entries": len(self._jit_cache),
            "shapes": [(k[0], k[1]) for k in self._jit_cache],  # (n_src, n_tgt) per entry
        }
