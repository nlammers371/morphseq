"""
Phase 1 / Task 1.7 — Lambda/mu sweep and deterministic selection.

Checks:
- Sweep table has one row per (lambda, mu) combination
- AUROC values are in [0, 1]
- Pareto knee selection picks a valid (lam, mu) from the grid
- Epsilon-best selection picks smallest complexity within epsilon of best AUROC
- Sweep is deterministic (same inputs -> same selection)
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from roi_config import SelectionRule, SweepConfig, TrainerConfig
from roi_sweep import SweepResult, run_sweep


@pytest.fixture
def tiny_sweep_config():
    """Minimal sweep config: 2x2 grid, 2 folds."""
    return SweepConfig(
        lambda_values=(1e-3, 1e-2),
        mu_values=(1e-3, 1e-2),
        n_cv_folds=2,
        selection_rule=SelectionRule.PARETO_KNEE,
        pareto_beta=1.0,
        roi_quantile=0.9,
    )


@pytest.fixture
def sweep_trainer_config():
    return TrainerConfig(
        learn_res=16,
        output_res=32,
        learning_rate=1e-2,
        max_steps=50,
        convergence_tol=1e-8,
        log_every=50,
        random_seed=42,
    )


def test_sweep_table_size(planted_data, tiny_sweep_config, sweep_trainer_config):
    """Sweep table should have n_lambda * n_mu rows."""
    result = run_sweep(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        groups=planted_data["groups"],
        lambda_values=list(tiny_sweep_config.lambda_values),
        mu_values=list(tiny_sweep_config.mu_values),
        sweep_config=tiny_sweep_config,
        trainer_config=sweep_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    expected_rows = len(tiny_sweep_config.lambda_values) * len(tiny_sweep_config.mu_values)
    assert len(result.sweep_table) == expected_rows


def test_sweep_auroc_range(planted_data, tiny_sweep_config, sweep_trainer_config):
    """All AUROC values should be in [0, 1]."""
    result = run_sweep(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        groups=planted_data["groups"],
        lambda_values=list(tiny_sweep_config.lambda_values),
        mu_values=list(tiny_sweep_config.mu_values),
        sweep_config=tiny_sweep_config,
        trainer_config=sweep_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    aurocs = result.sweep_table["auroc_mean"].values
    assert np.all(aurocs >= 0.0) and np.all(aurocs <= 1.0), (
        f"AUROC out of range: min={aurocs.min()}, max={aurocs.max()}"
    )


def test_selected_params_in_grid(planted_data, tiny_sweep_config, sweep_trainer_config):
    """Selected (lam, mu) must come from the sweep grid."""
    result = run_sweep(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        groups=planted_data["groups"],
        lambda_values=list(tiny_sweep_config.lambda_values),
        mu_values=list(tiny_sweep_config.mu_values),
        sweep_config=tiny_sweep_config,
        trainer_config=sweep_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    assert result.selected_lam in tiny_sweep_config.lambda_values
    assert result.selected_mu in tiny_sweep_config.mu_values


def test_sweep_determinism(planted_data, tiny_sweep_config, sweep_trainer_config):
    """Running the same sweep twice should select the same (lam, mu)."""
    kwargs = dict(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        groups=planted_data["groups"],
        lambda_values=list(tiny_sweep_config.lambda_values),
        mu_values=list(tiny_sweep_config.mu_values),
        sweep_config=tiny_sweep_config,
        trainer_config=sweep_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    r1 = run_sweep(**kwargs)
    r2 = run_sweep(**kwargs)

    assert r1.selected_lam == r2.selected_lam
    assert r1.selected_mu == r2.selected_mu


def test_sweep_result_has_selection_metadata(planted_data, tiny_sweep_config, sweep_trainer_config):
    """SweepResult should carry selection_rule and selection_metadata."""
    result = run_sweep(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        groups=planted_data["groups"],
        lambda_values=list(tiny_sweep_config.lambda_values),
        mu_values=list(tiny_sweep_config.mu_values),
        sweep_config=tiny_sweep_config,
        trainer_config=sweep_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    assert result.selection_rule == "pareto_knee"
    assert "beta" in result.selection_metadata or "rule" in result.selection_metadata


def test_planted_signal_gets_good_auroc(planted_data, tiny_sweep_config, sweep_trainer_config):
    """
    With a strong planted signal, the sweep should find at least one (lam, mu)
    with AUROC substantially above chance (0.5).
    """
    result = run_sweep(
        X=planted_data["X"],
        y=planted_data["y"],
        mask_ref=planted_data["mask_ref"],
        groups=planted_data["groups"],
        lambda_values=list(tiny_sweep_config.lambda_values),
        mu_values=list(tiny_sweep_config.mu_values),
        sweep_config=tiny_sweep_config,
        trainer_config=sweep_trainer_config,
        channel_names=planted_data["channel_names"],
    )

    best_auroc = result.sweep_table["auroc_mean"].max()
    assert best_auroc > 0.65, f"Best AUROC={best_auroc:.3f}, expected >0.65 for planted signal"
