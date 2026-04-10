from __future__ import annotations

import numpy as np
import pandas as pd

from analyze.trajectory_condensation.condensation import (
    CondensationConfig,
    StoppingConfig,
    describe_force_balance,
    run_condensation,
)
from analyze.trajectory_condensation.condensation.forces.repulsion import repulsion
from analyze.trajectory_condensation.init_embedding import pca_init
from analyze.trajectory_condensation.schema import from_pairwise_margin_csv


def test_pairwise_schema_preserves_nans(tmp_path):
    df = pd.DataFrame(
        {
            "embryo_id": ["e1", "e1", "e2", "e2"],
            "time_bin_center": [48.0, 52.0, 48.0, 52.0],
            "genotype": ["inj_ctrl", "inj_ctrl", "pbx4_crispant", "pbx4_crispant"],
            "inj_ctrl__vs__pbx4_crispant": [0.1, np.nan, -0.2, -0.3],
            "pbx1b_crispant__vs__pbx4_crispant": [np.nan, np.nan, 0.4, 0.5],
        }
    )
    path = tmp_path / "pairwise.csv"
    df.to_csv(path, index=False)

    data = from_pairwise_margin_csv(path, label_col="genotype")
    assert data.features.shape == (2, 2, 2)
    assert data.mask.sum() == 4
    assert np.isnan(data.features[0, 1, 0])
    assert np.isnan(data.features[0, 0, 1])


def test_condensation_smoke_runs_on_sparse_pairwise_input(tmp_path):
    df = pd.DataFrame(
        {
            "embryo_id": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "time_bin_center": [48.0, 52.0, 48.0, 52.0, 48.0, 52.0],
            "genotype": ["inj_ctrl", "inj_ctrl", "pbx4_crispant", "pbx4_crispant", "pbx1b_crispant", "pbx1b_crispant"],
            "inj_ctrl__vs__pbx4_crispant": [0.4, 0.3, -0.4, -0.3, np.nan, np.nan],
            "inj_ctrl__vs__pbx1b_crispant": [0.3, 0.2, np.nan, np.nan, -0.3, -0.2],
        }
    )
    path = tmp_path / "pairwise.csv"
    df.to_csv(path, index=False)
    data = from_pairwise_margin_csv(path, label_col="genotype")
    x0 = pca_init(data.features, data.mask, random_state=0)

    result = run_condensation(
        x0=x0,
        mask=data.mask,
        config=CondensationConfig(solver_max_iter=3, attract_k=2),
        stopping=StoppingConfig(patience=1000),
        log_every=1,
        save_every=None,
        verbose=False,
    )
    assert result.positions.shape == x0.shape
    assert len(result.metrics_history) == 3
    assert np.isfinite(result.positions[data.mask]).all()


def test_force_balance_resolves_public_elasticity_aliases():
    x0 = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
        ],
        dtype=float,
    )
    mask = np.ones((2, 3), dtype=bool)
    config = CondensationConfig(
        elastic_strength=0.5,
        elastic_mix=0.25,
        elastic_kernel="ratio_hinge",
        void_strength=0.014,
        void_bandwidth=3.0,
        local_scale_strength=0.2,
    )
    balance = describe_force_balance(x0, mask, config)
    assert np.isclose(balance["s_bend"], 1.0)
    assert balance["elastic_kernel"] == "ratio_hinge"
    assert np.isclose(balance["lambda_stretch"], 0.5 * 0.75)
    assert np.isclose(balance["lambda_bend"], 0.5 * 0.25)
    assert np.isclose(balance["void_strength"], 0.014)
    assert np.isclose(balance["void_bandwidth"], 3.0)
    assert np.isclose(balance["local_scale_strength"], 0.2)


def test_force_balance_rejects_invalid_elasticity_settings():
    with np.testing.assert_raises(ValueError):
        CondensationConfig(elastic_mix=0.5)
    with np.testing.assert_raises(ValueError):
        CondensationConfig(elastic_strength=0.5, elastic_mix=1.5)


def test_force_balance_includes_outlier_strength():
    x0 = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            [[0.5, 0.5], [1.5, 0.5], [2.5, 1.5]],
        ],
        dtype=float,
    )
    mask = np.ones((3, 3), dtype=bool)
    config = CondensationConfig(
        elastic_strength=0.5,
        elastic_mix=0.5,
        outlier_strength=1.25,
    )
    balance = describe_force_balance(x0, mask, config)
    assert np.isclose(balance["outlier_strength"], 1.25)
    assert balance["elastic_kernel"] == "quadratic"


def test_repulsion_quartic_tail_decays_fast_for_far_pairs():
    positions = np.array([[[0.0, 0.0]], [[2.0, 0.0]], [[5.0, 0.0]]], dtype=float)
    mask = np.ones((3, 1), dtype=bool)
    _, grad = repulsion(positions, mask, epsilon_r=1.0, eta=1e-6, r_cut=0.0)
    mid_norm = float(np.linalg.norm(grad[1, 0]))
    far_norm = float(np.linalg.norm(grad[2, 0]))
    assert mid_norm > 0
    assert far_norm > 0
    assert far_norm < mid_norm * 0.2
