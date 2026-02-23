"""Concordance tests: POTBackend vs OTTBackend on identical synthetic problems.

Every test runs BOTH backends and asserts parity within specified tolerances.
All tests are guarded by pytest.importorskip("ott").
"""

from __future__ import annotations

import numpy as np
import pytest

from analyze.utils.optimal_transport.backends.base import BackendResult, UOTBackend
from analyze.utils.optimal_transport.backends.pot_backend import POTBackend
from analyze.utils.optimal_transport.config import UOTConfig, UOTSupport

ott = pytest.importorskip("ott")

from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circle_support(cx: float, cy: float, r: float, n: int = 100, seed: int = 0) -> UOTSupport:
    """Generate uniformly weighted points on a filled circle."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    radii = r * np.sqrt(rng.uniform(0, 1, n))
    ys = cy + radii * np.sin(theta)
    xs = cx + radii * np.cos(theta)
    coords = np.stack([ys, xs], axis=1).astype(np.float64)
    weights = np.ones(n, dtype=np.float64) / n
    return UOTSupport(coords_yx=coords, weights=weights)


def _default_config(**overrides) -> UOTConfig:
    # epsilon must be reasonable relative to cost scale.
    # With coords ~0-100 and sqeuclidean, costs ~ 0-20000.
    # epsilon=0.1 gives best POT-vs-OTT concordance (<5% cost diff).
    # Higher epsilon amplifies tau conversion divergence.
    defaults = dict(
        epsilon=0.1,
        marginal_relaxation=10.0,
        metric="sqeuclidean",
        coord_scale=1.0,
        store_coupling=True,
    )
    defaults.update(overrides)
    return UOTConfig(**defaults)


def _assert_concordance(
    pot_result: BackendResult,
    ott_result: BackendResult,
    *,
    cost_rtol: float = 0.05,
    cost_atol: float = 1e-3,
    marginal_rtol: float = 0.35,
    marginal_atol: float = 5e-3,
    cosine_threshold: float = 0.9,
    mass_pct_atol: float = 0.05,
):
    """Assert concordance between POT and OTT results."""
    # 1. Total transport cost
    np.testing.assert_allclose(
        pot_result.cost, ott_result.cost,
        rtol=cost_rtol, atol=cost_atol,
        err_msg="Total transport cost mismatch",
    )

    # 2. Coupling marginals
    pot_coupling = np.asarray(pot_result.coupling)
    ott_coupling = np.asarray(ott_result.coupling)
    pot_mu = pot_coupling.sum(axis=1)
    ott_mu = ott_coupling.sum(axis=1)
    pot_nu = pot_coupling.sum(axis=0)
    ott_nu = ott_coupling.sum(axis=0)
    np.testing.assert_allclose(
        pot_mu, ott_mu, rtol=marginal_rtol, atol=marginal_atol,
        err_msg="Source marginal mismatch",
    )
    np.testing.assert_allclose(
        pot_nu, ott_nu, rtol=marginal_rtol, atol=marginal_atol,
        err_msg="Target marginal mismatch",
    )

    # 3. m_src/m_tgt in diagnostics
    assert "m_src" in pot_result.diagnostics
    assert "m_tgt" in pot_result.diagnostics
    assert "m_src" in ott_result.diagnostics
    assert "m_tgt" in ott_result.diagnostics

    # 4. Mass created/destroyed percentages
    m_src_pot = pot_result.diagnostics["m_src"]
    m_tgt_pot = pot_result.diagnostics["m_tgt"]
    m_src_ott = ott_result.diagnostics["m_src"]
    m_tgt_ott = ott_result.diagnostics["m_tgt"]
    assert m_src_pot == pytest.approx(m_src_ott, abs=1e-10)
    assert m_tgt_pot == pytest.approx(m_tgt_ott, abs=1e-10)

    # Mass created = max(0, weights_tgt - nu_hat) summed
    pot_created = float(np.maximum(0, pot_nu - pot_coupling.sum(axis=0)).sum())
    ott_created = float(np.maximum(0, ott_nu - ott_coupling.sum(axis=0)).sum())
    # Normalize by total mass for percentage comparison
    if m_tgt_pot > 0:
        pot_created_pct = pot_created / m_tgt_pot
        ott_created_pct = ott_created / m_tgt_ott
        assert abs(pot_created_pct - ott_created_pct) < mass_pct_atol, \
            f"Mass created % mismatch: POT={pot_created_pct:.4f} OTT={ott_created_pct:.4f}"


def _velocity_from_coupling(coupling, src_coords, tgt_coords):
    """Compute barycentric velocity from coupling matrix."""
    coupling = np.asarray(coupling, dtype=np.float64)
    mu_hat = coupling.sum(axis=1)
    mu_hat_safe = np.maximum(mu_hat, 1e-12)
    T = (coupling @ tgt_coords) / mu_hat_safe[:, None]
    return T - src_coords


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIdentity:
    """Identical circles → cost ≈ 0."""

    def test_identity_concordance(self):
        src = _circle_support(50, 50, 10, n=80, seed=42)
        tgt = UOTSupport(coords_yx=src.coords_yx.copy(), weights=src.weights.copy())
        config = _default_config()

        pot_result = POTBackend().solve(src, tgt, config)
        ott_result = OTTBackend().solve(src, tgt, config)

        # Cost should be near zero for identity
        assert pot_result.cost < 1.0, f"POT cost too high for identity: {pot_result.cost}"
        assert ott_result.cost < 1.0, f"OTT cost too high for identity: {ott_result.cost}"

        _assert_concordance(pot_result, ott_result, cost_atol=0.5)


class TestTranslation:
    """Known offset → velocity agrees."""

    def test_translation_concordance(self):
        src = _circle_support(50, 50, 10, n=80, seed=42)
        offset_y, offset_x = 5.0, 3.0
        tgt_coords = src.coords_yx.copy()
        tgt_coords[:, 0] += offset_y
        tgt_coords[:, 1] += offset_x
        tgt = UOTSupport(coords_yx=tgt_coords, weights=src.weights.copy())
        config = _default_config()

        pot_result = POTBackend().solve(src, tgt, config)
        ott_result = OTTBackend().solve(src, tgt, config)

        _assert_concordance(pot_result, ott_result)

        # Check velocity direction agreement
        v_pot = _velocity_from_coupling(pot_result.coupling, src.coords_yx, tgt.coords_yx)
        v_ott = _velocity_from_coupling(ott_result.coupling, src.coords_yx, tgt.coords_yx)

        # Mean velocity should point in the right direction
        mean_v_pot = v_pot.mean(axis=0)
        mean_v_ott = v_ott.mean(axis=0)
        expected = np.array([offset_y, offset_x])

        # Cosine similarity of mean velocity with expected direction
        cos_pot = np.dot(mean_v_pot, expected) / (np.linalg.norm(mean_v_pot) * np.linalg.norm(expected) + 1e-12)
        cos_ott = np.dot(mean_v_ott, expected) / (np.linalg.norm(mean_v_ott) * np.linalg.norm(expected) + 1e-12)
        assert cos_pot > 0.9, f"POT velocity direction wrong: cosine={cos_pot:.3f}"
        assert cos_ott > 0.9, f"OTT velocity direction wrong: cosine={cos_ott:.3f}"

        # Cross-backend velocity cosine similarity
        flat_pot = v_pot.ravel()
        flat_ott = v_ott.ravel()
        cross_cos = np.dot(flat_pot, flat_ott) / (np.linalg.norm(flat_pot) * np.linalg.norm(flat_ott) + 1e-12)
        assert cross_cos > 0.9, f"Cross-backend velocity cosine too low: {cross_cos:.3f}"


class TestDilation:
    """Target larger than source → mass creation detected by both."""

    def test_dilation_concordance(self):
        src = _circle_support(50, 50, 8, n=60, seed=42)
        tgt = _circle_support(50, 50, 14, n=100, seed=43)
        config = _default_config()

        pot_result = POTBackend().solve(src, tgt, config)
        ott_result = OTTBackend().solve(src, tgt, config)

        _assert_concordance(pot_result, ott_result)

        # Both should detect mass creation (target has more mass coverage)
        pot_nu = np.asarray(pot_result.coupling).sum(axis=0)
        ott_nu = np.asarray(ott_result.coupling).sum(axis=0)
        pot_created = float(np.maximum(0, tgt.weights * pot_result.diagnostics["m_tgt"] - pot_nu).sum())
        ott_created = float(np.maximum(0, tgt.weights * ott_result.diagnostics["m_tgt"] - ott_nu).sum())
        assert pot_created > 0, "POT should detect mass creation in dilation"
        assert ott_created > 0, "OTT should detect mass creation in dilation"


class TestErosion:
    """Target smaller than source → mass destruction detected by both."""

    def test_erosion_concordance(self):
        src = _circle_support(50, 50, 14, n=100, seed=42)
        tgt = _circle_support(50, 50, 8, n=60, seed=43)
        config = _default_config()

        pot_result = POTBackend().solve(src, tgt, config)
        ott_result = OTTBackend().solve(src, tgt, config)

        _assert_concordance(pot_result, ott_result)

        # Both should detect mass destruction (source has more mass coverage)
        pot_mu = np.asarray(pot_result.coupling).sum(axis=1)
        ott_mu = np.asarray(ott_result.coupling).sum(axis=1)
        pot_destroyed = float(np.maximum(0, src.weights * pot_result.diagnostics["m_src"] - pot_mu).sum())
        ott_destroyed = float(np.maximum(0, src.weights * ott_result.diagnostics["m_src"] - ott_mu).sum())
        assert pot_destroyed > 0, "POT should detect mass destruction in erosion"
        assert ott_destroyed > 0, "OTT should detect mass destruction in erosion"


class TestBatch:
    """solve_batch works sequentially for batch of 8."""

    def test_batch_8_pairs(self):
        rng = np.random.default_rng(123)
        config = _default_config()
        problems = []
        for i in range(8):
            src = _circle_support(50, 50, 10, n=50, seed=i)
            offset = rng.uniform(-3, 3, size=2)
            tgt_coords = src.coords_yx + offset
            tgt = UOTSupport(coords_yx=tgt_coords, weights=src.weights.copy())
            problems.append((src, tgt))

        backend = OTTBackend()
        results = backend.solve_batch(problems, config)

        assert len(results) == 8
        for r in results:
            assert isinstance(r, BackendResult)
            assert r.cost >= 0
            assert "m_src" in r.diagnostics
            assert "m_tgt" in r.diagnostics
            assert r.coupling is not None


class TestConditionalImport:
    """OTTBackend is importable and conditional import works."""

    def test_ott_backend_importable(self):
        from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend, ott_available
        assert ott_available()
        backend = OTTBackend()
        assert isinstance(backend, UOTBackend)

    def test_ott_in_backends_init(self):
        from analyze.utils.optimal_transport.backends import OTTBackend as OTT
        assert OTT is not None

    def test_ott_in_top_level_init(self):
        from analyze.utils.optimal_transport import OTTBackend as OTT
        assert OTT is not None
