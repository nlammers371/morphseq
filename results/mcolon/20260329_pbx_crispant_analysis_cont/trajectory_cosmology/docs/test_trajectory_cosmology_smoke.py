from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ANALYSIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS_ROOT))

from trajectory_cosmology.schema import (
    CosmologyData,
    from_multiclass_csv,
    from_pairwise_margin_csv,
    validate,
)
from trajectory_cosmology.init_embedding import pca_init
from trajectory_cosmology.condensation.coherence import compute_coherence, gaussian_kernel
from trajectory_cosmology.condensation.forces import (
    attraction,
    repulsion,
    elasticity,
    total_energy_and_grad,
)
from trajectory_cosmology.condensation.api import run_condensation
from trajectory_cosmology.condensation.state import CondensationConfig
from trajectory_cosmology.condensation.stopping import (
    StoppingConfig,
    reference_scale_from_positions,
    displacement_metrics,
)
from slice_diagnostic import radial_spread

REPO_ROOT = ANALYSIS_ROOT.parents[2]
MULTI_PATH = REPO_ROOT / 'results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_positioning_multiclass_bridge_bin4_perm500/multiclass_probability_vectors.csv'
PAIR_PATH = REPO_ROOT / 'results/mcolon/20260326_pbx_crispant_analysis/results/misclassification/embedding/phenotypic_positioning_phase2_bin4/raw_position_vectors.csv'


def _make_toy_data(n_embryo=3, n_time=4, n_feat=2, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((n_embryo, n_time, n_feat))
    mask = np.ones((n_embryo, n_time), dtype=bool)
    embryo_ids = np.array([f'e{i:02d}' for i in range(n_embryo)])
    time_values = np.arange(n_time, dtype=float)
    labels = np.array([f'g{i}' for i in range(n_embryo)])
    embryo_index = {e: i for i, e in enumerate(embryo_ids)}
    time_index = {float(t): j for j, t in enumerate(time_values)}
    return CosmologyData(
        features=features,
        mask=mask,
        embryo_ids=embryo_ids,
        time_values=time_values,
        labels=labels,
        feature_names=[f'f{k}' for k in range(n_feat)],
        embryo_index=embryo_index,
        time_index=time_index,
    )


def test_schema_real_pbx_inputs_load():
    pair = from_pairwise_margin_csv(PAIR_PATH, label_col='genotype')
    assert pair.features.shape == (222, 27, 10)
    assert pair.mask.sum() == 1546

    prob_cols = [c for c in pd.read_csv(MULTI_PATH, nrows=1).columns if c.startswith('pred_proba_')]
    multi = from_multiclass_csv(MULTI_PATH, prob_cols=prob_cols, label_col='true_class')
    assert multi.features.shape == (304, 27, 4)
    assert multi.mask.sum() == 2570


def test_condensation_smoke_real_pbx_input_is_finite():
    prob_cols = [c for c in pd.read_csv(MULTI_PATH, nrows=1).columns if c.startswith('pred_proba_')]
    data = from_multiclass_csv(MULTI_PATH, prob_cols=prob_cols, label_col='true_class')
    x0 = pca_init(data.features, data.mask)
    coherence = compute_coherence(x0, data.mask, sigma=0.5, delta=3)
    assert np.isfinite(coherence).all()

    energies, grad = total_energy_and_grad(
        x0, x0, data.mask, coherence,
        sigma=0.5, epsilon_r=0.01, eta=1e-4,
        lambda_stretch=0.1, lambda_bend=0.05, mu=1.0,
        k_attract=None, subtract_mean_attraction=False,
    )
    assert all(np.isfinite(v) for v in energies.values())
    assert np.isfinite(grad[data.mask]).all()

    result = run_condensation(
        x0=x0,
        mask=data.mask,
        config=CondensationConfig(max_iter=5, k_attract=None, subtract_mean_attraction=False),
        stopping=StoppingConfig(patience=1000),
        log_every=1,
        verbose=False,
    )
    assert len(result.metrics_history) == 5
    assert all(np.isfinite(row['disp_rms_rel']) for row in result.metrics_history)
    assert all(np.isfinite(row['energy_total']) for row in result.metrics_history)


def test_validate_rejects_unsorted_times():
    data = _make_toy_data()
    object.__setattr__(data, 'time_values', data.time_values[::-1].copy())
    with pytest.raises(AssertionError, match='sorted'):
        validate(data)


def test_validate_rejects_non_nan_unobserved():
    data = _make_toy_data()
    data.mask[0, 0] = False
    data.features[0, 0, :] = 0.0
    with pytest.raises(AssertionError, match='NaN'):
        validate(data)


def test_validate_rejects_inconsistent_labels():
    df = pd.DataFrame({
        'embryo_id': ['e00', 'e00', 'e01', 'e01'],
        'time_bin_center': [1.0, 2.0, 1.0, 2.0],
        'genotype': ['wt', 'mut', 'wt', 'wt'],
        'p_wt': [0.9, 0.1, 0.8, 0.85],
        'p_mut': [0.1, 0.9, 0.2, 0.15],
    })
    tmp = ANALYSIS_ROOT / '_tmp_inconsistent_labels.csv'
    df.to_csv(tmp, index=False)
    try:
        with pytest.raises(AssertionError, match='inconsistent labels'):
            from_multiclass_csv(tmp, label_col='genotype')
    finally:
        tmp.unlink(missing_ok=True)


def test_coherence_delta0_equals_kernel():
    rng = np.random.default_rng(1)
    pos = rng.standard_normal((3, 4, 2))
    mask = np.ones((3, 4), dtype=bool)
    sigma = 0.5
    C = compute_coherence(pos, mask, sigma=sigma, delta=0)
    for t in range(4):
        K = gaussian_kernel(pos[:, t, :], sigma)
        np.testing.assert_allclose(C[:, :, t], K, atol=1e-10)


def test_coherence_identical_trajectories():
    pos = np.zeros((2, 5, 2))
    mask = np.ones((2, 5), dtype=bool)
    C = compute_coherence(pos, mask, sigma=0.5, delta=3)
    np.testing.assert_allclose(C[0, 1, :], 1.0, atol=1e-10)


def test_coherence_far_trajectories():
    pos = np.zeros((2, 5, 2))
    pos[1, :, 0] = 100.0
    mask = np.ones((2, 5), dtype=bool)
    C = compute_coherence(pos, mask, sigma=0.5, delta=3)
    assert C[0, 1, :].max() < 1e-10


def test_attraction_pulls_together():
    pos = np.zeros((2, 1, 2))
    pos[0, 0, :] = [-1.0, 0.0]
    pos[1, 0, :] = [1.0, 0.0]
    mask = np.ones((2, 1), dtype=bool)
    C = np.ones((2, 2, 1))
    np.fill_diagonal(C[:, :, 0], 0.0)
    _, grad = attraction(pos, mask, C, sigma=1.0)
    assert grad[0, 0, 0] < 0
    assert grad[1, 0, 0] > 0
    step = pos - 0.1 * grad
    assert abs(step[0, 0, 0] - step[1, 0, 0]) < abs(pos[0, 0, 0] - pos[1, 0, 0])


def test_repulsion_pushes_apart():
    pos = np.zeros((2, 1, 2))
    pos[0, 0, :] = [-0.5, 0.0]
    pos[1, 0, :] = [0.5, 0.0]
    mask = np.ones((2, 1), dtype=bool)
    _, grad = repulsion(pos, mask, epsilon_r=0.01, eta=1e-4)
    assert grad[0, 0, 0] > 0
    assert grad[1, 0, 0] < 0
    step = pos - 0.1 * grad
    assert abs(step[0, 0, 0] - step[1, 0, 0]) > abs(pos[0, 0, 0] - pos[1, 0, 0])


def test_elasticity_straight_line_zero_bend():
    pos = np.zeros((1, 4, 2))
    pos[0, :, 0] = [0.0, 1.0, 2.0, 3.0]
    mask = np.ones((1, 4), dtype=bool)
    energy, _ = elasticity(pos, mask, lambda_stretch=1.0, lambda_bend=1.0)
    stretch_energy = 3.0
    assert abs(energy - stretch_energy) < 1e-10


def test_finite_difference_gradient_baseline_all_pairs():
    rng = np.random.default_rng(42)
    pos = rng.standard_normal((2, 3, 2)) * 0.3
    x0 = pos.copy()
    mask = np.ones((2, 3), dtype=bool)
    C = compute_coherence(pos, mask, sigma=0.5, delta=1)
    kwargs = dict(
        x0=x0,
        mask=mask,
        coherence=C,
        sigma=0.5,
        epsilon_r=0.01,
        eta=1e-4,
        lambda_stretch=0.1,
        lambda_bend=0.05,
        mu=0.5,
        k_attract=None,
        subtract_mean_attraction=False,
    )
    _, grad_analytic = total_energy_and_grad(pos, **kwargs)
    eps = 1e-5
    grad_numerical = np.zeros_like(pos)
    for i in range(2):
        for t in range(3):
            for d in range(2):
                pos_p = pos.copy(); pos_p[i, t, d] += eps
                e_p, _ = total_energy_and_grad(pos_p, **kwargs)
                pos_m = pos.copy(); pos_m[i, t, d] -= eps
                e_m, _ = total_energy_and_grad(pos_m, **kwargs)
                grad_numerical[i, t, d] = (e_p['total'] - e_m['total']) / (2 * eps)
    np.testing.assert_allclose(grad_analytic, grad_numerical, atol=1e-4, rtol=1e-3)


def test_knn_attraction_blocks_cross_cluster_pull():
    pos = np.zeros((20, 1, 2))
    pos[:10, 0, 0] = np.linspace(-0.1, 0.1, 10)
    pos[10:, 0, 0] = 10.0 + np.linspace(-0.1, 0.1, 10)
    mask = np.ones((20, 1), dtype=bool)
    C = np.ones((20, 20, 1))
    np.fill_diagonal(C[:, :, 0], 0.0)
    _, grad_all = attraction(pos, mask, C, sigma=5.0, k_attract=None)
    _, grad_knn = attraction(pos, mask, C, sigma=5.0, k_attract=5)
    mean_cross_pull_all = float(np.abs(grad_all[:10, 0, 0]).mean())
    mean_cross_pull_knn = float(np.abs(grad_knn[:10, 0, 0]).mean())
    assert mean_cross_pull_knn < mean_cross_pull_all


def test_mean_subtracted_attraction_sums_to_zero_per_slice():
    rng = np.random.default_rng(123)
    pos = rng.standard_normal((8, 2, 2))
    mask = np.ones((8, 2), dtype=bool)
    C = np.ones((8, 8, 2))
    for t in range(2):
        np.fill_diagonal(C[:, :, t], 0.0)
    _, grad = attraction(pos, mask, C, sigma=1.0, k_attract=3, subtract_mean=True)
    np.testing.assert_allclose(grad[:, 0, :].sum(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(grad[:, 1, :].sum(axis=0), 0.0, atol=1e-10)


def test_radial_spread_is_positive_and_detects_collapse():
    wide = np.array([[0.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]])
    tight = 0.01 * wide
    assert radial_spread(wide) > radial_spread(tight)


def test_reference_scale_nonzero():
    prob_cols = [c for c in pd.read_csv(MULTI_PATH, nrows=1).columns if c.startswith('pred_proba_')]
    data = from_multiclass_csv(MULTI_PATH, prob_cols=prob_cols, label_col='true_class')
    x0 = pca_init(data.features, data.mask)
    scale = reference_scale_from_positions(x0, data.mask)
    assert np.isfinite(scale)
    assert scale > 0


def test_displacement_metrics_zero_on_identical():
    x = np.random.default_rng(7).standard_normal((5, 4, 2))
    mask = np.ones((5, 4), dtype=bool)
    scale = reference_scale_from_positions(x, mask)
    metrics = displacement_metrics(x, x, mask, scale)
    assert metrics['disp_max_rel'] == 0.0
    assert metrics['disp_rms_rel'] == 0.0


def test_displacement_metrics_proportional_to_shift():
    x = np.zeros((3, 3, 2))
    mask = np.ones((3, 3), dtype=bool)
    x[0, :, 0] = 1.0
    x[1, :, 0] = -1.0
    scale = reference_scale_from_positions(x, mask)
    shift = 0.1
    x_shifted = x + shift
    metrics = displacement_metrics(x, x_shifted, mask, scale)
    expected = np.linalg.norm([shift, shift]) / scale
    np.testing.assert_allclose(metrics['disp_max_rel'], expected, rtol=1e-6)
