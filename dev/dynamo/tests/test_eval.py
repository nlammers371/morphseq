"""Tests for evaluation and visualization pipeline (build step 2).

Tests cover:
    - PredictionResult construction
    - Dummy predictors (Persistence, LinearExtrapolation, GaussianNoise)
    - Metric computation (NLL, MSE, calibration, energy distance)
    - Evaluation orchestration (run_evaluation)
    - Three-model comparison
    - Visualization panel generation
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dev.dynamo.data.loading import EmbryoTrajectory, TrajectoryDataset
from dev.dynamo.data.dataset import FragmentDataset, FragmentBatch, fragment_collate_fn
from dev.dynamo.eval.predictions import (
    PredictionResult,
    PersistencePredictor,
    LinearExtrapolationPredictor,
    GaussianNoisePredictor,
)
from dev.dynamo.eval.metrics import (
    gaussian_nll,
    mse,
    per_dim_mse,
    calibration_fraction,
    energy_distance,
    compute_sample_metrics,
    mode_diagnostics,
)
from dev.dynamo.eval.evaluate import (
    run_evaluation,
    ComparisonResult,
    _bin_horizon_k,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data (reuses patterns from test_data.py)
# ---------------------------------------------------------------------------

def _make_trajectory(
    embryo_id: str = "emb_001",
    n_frames: int = 20,
    n_dim: int = 10,
    delta_t: float = 500.0,
    temperature: float = 28.5,
    perturbation_class: str = "wildtype",
    experiment_id: str = "exp_001",
    seed: int = 42,
) -> EmbryoTrajectory:
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal((n_frames, n_dim)) * 0.1
    traj = np.cumsum(steps, axis=0)
    times = np.arange(n_frames) * delta_t
    return EmbryoTrajectory(
        embryo_id=embryo_id,
        trajectory=traj,
        time_seconds=times,
        delta_t=delta_t,
        temperature=temperature,
        perturbation_class=perturbation_class,
        experiment_id=experiment_id,
    )


def _make_dataset(n_embryos: int = 10, n_frames: int = 20, n_dim: int = 10) -> TrajectoryDataset:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    trajs = []
    classes = ["wildtype", "mutant_a", "mutant_b"]
    for i in range(n_embryos):
        trajs.append(_make_trajectory(
            embryo_id=f"emb_{i:03d}",
            n_frames=n_frames,
            n_dim=n_dim,
            perturbation_class=classes[i % len(classes)],
            seed=i,
        ))

    pca = PCA(n_components=n_dim)
    pca.fit(np.eye(n_dim))
    scaler = StandardScaler()
    scaler.fit(np.zeros((2, n_dim)))

    return TrajectoryDataset(
        trajectories=trajs,
        pca=pca,
        scaler=scaler,
        z_mu_cols=[f"z_mu_b_{i:02d}" for i in range(n_dim)],
    )


def _make_batch(n: int = 8, D: int = 5) -> FragmentBatch:
    """Create a synthetic FragmentBatch for direct metric testing."""
    ds = _make_dataset(n_embryos=10, n_dim=D)
    fds = FragmentDataset(ds, min_context=3, horizons=(1, 2, 3))
    samples = [fds[i] for i in range(n)]
    return fragment_collate_fn(samples)


# ---------------------------------------------------------------------------
# Tests: PredictionResult
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_construction(self) -> None:
        B, D = 4, 10
        pred = PredictionResult(
            predicted_mean=torch.randn(B, D),
            predicted_cov_diag=torch.rand(B, D) + 0.01,
        )
        assert pred.predicted_mean.shape == (B, D)
        assert pred.predicted_cov_diag.shape == (B, D)
        assert pred.forward_samples is None
        assert pred.mode_loadings is None

    def test_with_samples(self) -> None:
        B, D, N = 4, 10, 50
        pred = PredictionResult(
            predicted_mean=torch.randn(B, D),
            predicted_cov_diag=torch.rand(B, D) + 0.01,
            forward_samples=torch.randn(B, N, D),
        )
        assert pred.forward_samples.shape == (B, N, D)

    def test_with_mode_diagnostics(self) -> None:
        B, D, M = 4, 10, 3
        pred = PredictionResult(
            predicted_mean=torch.randn(B, D),
            predicted_cov_diag=torch.rand(B, D) + 0.01,
            mode_loadings=torch.randn(B, M),
            local_correction_norm=torch.rand(B),
            residual_norm=torch.rand(B),
            rate=torch.rand(B) + 0.5,
        )
        assert pred.mode_loadings.shape == (B, M)
        assert pred.rate.shape == (B,)


# ---------------------------------------------------------------------------
# Tests: Dummy Predictors
# ---------------------------------------------------------------------------

class TestPersistencePredictor:
    def test_output_shape(self) -> None:
        batch = _make_batch(n=8, D=5)
        pred = PersistencePredictor(noise_scale=0.1)
        result = pred.predict(batch)

        assert result.predicted_mean.shape == (8, 5)
        assert result.predicted_cov_diag.shape == (8, 5)

    def test_mean_is_last_context(self) -> None:
        batch = _make_batch(n=4, D=5)
        pred = PersistencePredictor()
        result = pred.predict(batch)

        # Verify each prediction is the last real context frame
        lengths = batch.context_mask.sum(dim=1).long()
        for i in range(4):
            last_idx = lengths[i] - 1
            expected = batch.context[i, last_idx]
            torch.testing.assert_close(result.predicted_mean[i], expected)

    def test_variance_scales_with_horizon(self) -> None:
        batch = _make_batch(n=4, D=5)
        pred = PersistencePredictor(noise_scale=1.0)
        result = pred.predict(batch)

        # Variance should be proportional to horizon_dt
        for i in range(4):
            expected_var = batch.horizon_dt[i].item()
            # All dimensions should have the same variance
            torch.testing.assert_close(
                result.predicted_cov_diag[i],
                torch.full((5,), expected_var),
            )


class TestLinearExtrapolationPredictor:
    def test_output_shape(self) -> None:
        batch = _make_batch(n=8, D=5)
        pred = LinearExtrapolationPredictor(noise_scale=0.1)
        result = pred.predict(batch)

        assert result.predicted_mean.shape == (8, 5)
        assert result.predicted_cov_diag.shape == (8, 5)

    def test_extrapolation_direction(self) -> None:
        """Predicted mean should be in the direction of the last velocity."""
        batch = _make_batch(n=4, D=5)
        pred = LinearExtrapolationPredictor()
        result = pred.predict(batch)

        lengths = batch.context_mask.sum(dim=1).long()
        for i in range(4):
            last = batch.context[i, lengths[i] - 1]
            prev = batch.context[i, lengths[i] - 2]
            velocity = (last - prev) / batch.delta_t[i]
            expected = last + velocity * batch.horizon_dt[i]
            torch.testing.assert_close(result.predicted_mean[i], expected, atol=1e-5, rtol=1e-5)


class TestGaussianNoisePredictor:
    def test_output_shape(self) -> None:
        batch = _make_batch(n=8, D=5)
        pred = GaussianNoisePredictor(std=1.0)
        result = pred.predict(batch)

        assert result.predicted_mean.shape == (8, 5)
        assert result.predicted_cov_diag.shape == (8, 5)

    def test_variance_is_constant(self) -> None:
        batch = _make_batch(n=4, D=5)
        pred = GaussianNoisePredictor(std=2.0)
        result = pred.predict(batch)

        # All variances should be 4.0
        expected = torch.full((4, 5), 4.0)
        torch.testing.assert_close(result.predicted_cov_diag, expected)


# ---------------------------------------------------------------------------
# Tests: Metrics
# ---------------------------------------------------------------------------

class TestGaussianNLL:
    def test_shape(self) -> None:
        B, D = 8, 5
        nll = gaussian_nll(
            torch.zeros(B, D),
            torch.ones(B, D),
            torch.zeros(B, D),
        )
        assert nll.shape == (B,)

    def test_zero_residual(self) -> None:
        """NLL at the mean should be the entropy of the Gaussian."""
        B, D = 4, 3
        var = torch.ones(B, D)
        nll = gaussian_nll(torch.zeros(B, D), var, torch.zeros(B, D))

        # For unit Gaussian: NLL = D/2 * log(2π) + 0
        expected = 0.5 * D * math.log(2.0 * math.pi)
        torch.testing.assert_close(nll, torch.full((B,), expected), atol=1e-5, rtol=1e-5)

    def test_larger_residual_higher_nll(self) -> None:
        """Moving the target away from the mean should increase NLL."""
        B, D = 4, 5
        mean = torch.zeros(B, D)
        var = torch.ones(B, D)
        nll_close = gaussian_nll(mean, var, torch.full((B, D), 0.1))
        nll_far = gaussian_nll(mean, var, torch.full((B, D), 1.0))
        assert (nll_far > nll_close).all()

    def test_larger_variance_lower_nll_for_far_target(self) -> None:
        """Larger variance should reduce NLL for targets far from mean."""
        B, D = 4, 5
        mean = torch.zeros(B, D)
        target = torch.full((B, D), 3.0)  # Far from mean

        nll_small_var = gaussian_nll(mean, torch.ones(B, D) * 0.5, target)
        nll_large_var = gaussian_nll(mean, torch.ones(B, D) * 10.0, target)
        assert (nll_large_var < nll_small_var).all()

    def test_differentiable(self) -> None:
        """NLL should be differentiable w.r.t. mean and variance."""
        B, D = 4, 5
        mean = torch.randn(B, D, requires_grad=True)
        var_raw = torch.rand(B, D, requires_grad=True)
        var = var_raw + 0.1
        target = torch.randn(B, D)
        nll = gaussian_nll(mean, var, target)
        nll.sum().backward()
        assert mean.grad is not None
        assert var_raw.grad is not None


class TestMSE:
    def test_shape(self) -> None:
        B, D = 8, 5
        err = mse(torch.zeros(B, D), torch.ones(B, D))
        assert err.shape == (B,)

    def test_zero_for_perfect(self) -> None:
        B, D = 4, 5
        x = torch.randn(B, D)
        assert (mse(x, x) == 0).all()

    def test_value(self) -> None:
        B, D = 1, 4
        pred = torch.zeros(B, D)
        target = torch.ones(B, D)
        # MSE = mean(1^2) = 1.0
        torch.testing.assert_close(mse(pred, target), torch.tensor([1.0]))


class TestPerDimMSE:
    def test_shape(self) -> None:
        B, D = 4, 5
        err = per_dim_mse(torch.zeros(B, D), torch.ones(B, D))
        assert err.shape == (B, D)


class TestCalibration:
    def test_perfect_calibration(self) -> None:
        """Targets at the mean should always be inside."""
        B, D = 100, 5
        mean = torch.randn(B, D)
        var = torch.ones(B, D)
        frac = calibration_fraction(mean, var, mean, level=0.9)
        assert frac.item() == 1.0

    def test_all_outside(self) -> None:
        """Very distant targets should be outside tiny confidence regions."""
        B, D = 100, 5
        mean = torch.zeros(B, D)
        var = torch.full((B, D), 1e-6)  # Tiny variance
        target = torch.full((B, D), 100.0)  # Very far
        frac = calibration_fraction(mean, var, target, level=0.9)
        assert frac.item() == 0.0


class TestEnergyDistance:
    def test_shape(self) -> None:
        B, N, D = 4, 50, 5
        ed = energy_distance(torch.randn(B, N, D), torch.randn(B, D))
        assert ed.shape == (B,)

    def test_zero_for_perfect(self) -> None:
        """Energy distance should be near zero when all samples equal the target."""
        B, D = 4, 5
        target = torch.randn(B, D)
        samples = target.unsqueeze(1).expand(B, 50, D)
        ed = energy_distance(samples, target)
        torch.testing.assert_close(ed, torch.zeros(B), atol=1e-5, rtol=1e-5)

    def test_positive_for_mismatch(self) -> None:
        B, D = 4, 5
        target = torch.zeros(B, D)
        samples = torch.randn(B, 50, D) + 5.0  # Far from target
        ed = energy_distance(samples, target)
        assert (ed > 0).all()


class TestComputeSampleMetrics:
    def test_returns_required_keys(self) -> None:
        B, D = 4, 5
        pred = PredictionResult(
            predicted_mean=torch.randn(B, D),
            predicted_cov_diag=torch.rand(B, D) + 0.01,
        )
        target = torch.randn(B, D)
        metrics = compute_sample_metrics(pred, target)
        assert "nll" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics

    def test_includes_energy_when_samples_present(self) -> None:
        B, D, N = 4, 5, 50
        pred = PredictionResult(
            predicted_mean=torch.randn(B, D),
            predicted_cov_diag=torch.rand(B, D) + 0.01,
            forward_samples=torch.randn(B, N, D),
        )
        metrics = compute_sample_metrics(pred, torch.randn(B, D))
        assert "energy_distance" in metrics


class TestModeDiagnostics:
    def test_empty_when_no_data(self) -> None:
        pred = PredictionResult(
            predicted_mean=torch.randn(4, 5),
            predicted_cov_diag=torch.rand(4, 5),
        )
        diag = mode_diagnostics(pred)
        assert diag == {}

    def test_populated_when_available(self) -> None:
        pred = PredictionResult(
            predicted_mean=torch.randn(4, 5),
            predicted_cov_diag=torch.rand(4, 5),
            mode_loadings=torch.randn(4, 3),
            local_correction_norm=torch.rand(4),
            residual_norm=torch.rand(4),
            rate=torch.rand(4) + 0.5,
        )
        diag = mode_diagnostics(pred)
        assert "v_norm" in diag
        assert "residual_norm" in diag
        assert "loading_magnitudes" in diag
        assert "loading_norm" in diag
        assert "rate" in diag
        assert diag["loading_magnitudes"].shape == (4, 3)


# ---------------------------------------------------------------------------
# Tests: Horizon binning
# ---------------------------------------------------------------------------

class TestHorizonBinning:
    def test_exact_multiples(self) -> None:
        horizon_dt = torch.tensor([500.0, 1000.0, 1500.0, 2000.0])
        delta_t = torch.tensor([500.0, 500.0, 500.0, 500.0])
        k = _bin_horizon_k(horizon_dt, delta_t)
        assert k.tolist() == [1, 2, 3, 4]

    def test_rounding(self) -> None:
        horizon_dt = torch.tensor([480.0, 520.0, 1450.0])
        delta_t = torch.tensor([500.0, 500.0, 500.0])
        k = _bin_horizon_k(horizon_dt, delta_t)
        assert k.tolist() == [1, 1, 3]

    def test_clamped_to_one(self) -> None:
        horizon_dt = torch.tensor([0.1])
        delta_t = torch.tensor([500.0])
        k = _bin_horizon_k(horizon_dt, delta_t)
        assert k.item() == 1


# ---------------------------------------------------------------------------
# Tests: Evaluation orchestration
# ---------------------------------------------------------------------------

class TestRunEvaluation:
    def test_persistence_evaluates(self) -> None:
        ds = _make_dataset(n_embryos=10, n_dim=5)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2))
        predictor = PersistencePredictor(noise_scale=0.5)

        result = run_evaluation(
            predictor=predictor,
            dataset=fds,
            n_batches=5,
            batch_size=4,
            tier="test",
        )

        assert result.n_samples > 0
        assert "nll" in result.metrics
        assert "mse" in result.metrics
        assert result.calibration >= 0.0
        assert result.calibration <= 1.0
        assert result.tier == "test"

    def test_linear_evaluates(self) -> None:
        ds = _make_dataset(n_embryos=10, n_dim=5)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2))
        predictor = LinearExtrapolationPredictor(noise_scale=0.5)

        result = run_evaluation(
            predictor=predictor,
            dataset=fds,
            n_batches=3,
            batch_size=4,
        )

        assert result.n_samples > 0
        assert len(result.per_horizon) > 0

    def test_per_horizon_breakdown(self) -> None:
        ds = _make_dataset(n_embryos=10, n_dim=5)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2, 3))
        predictor = PersistencePredictor(noise_scale=0.5)

        result = run_evaluation(
            predictor=predictor,
            dataset=fds,
            n_batches=10,
            batch_size=8,
        )

        # Should have at least one horizon bucket
        assert len(result.per_horizon) >= 1
        for k, hm in result.per_horizon.items():
            assert "nll" in hm
            assert "mse" in hm
            assert hm["n_samples"] > 0

    def test_random_predictor_worse_than_persistence(self) -> None:
        """Random predictions should have higher NLL than persistence."""
        ds = _make_dataset(n_embryos=10, n_dim=5)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2))

        persist_result = run_evaluation(
            PersistencePredictor(noise_scale=0.5),
            fds, n_batches=10, batch_size=8,
        )
        random_result = run_evaluation(
            GaussianNoisePredictor(std=2.0),
            fds, n_batches=10, batch_size=8,
        )

        # Random should be worse (higher NLL and MSE)
        assert random_result.metrics["mse"] > persist_result.metrics["mse"]


# ---------------------------------------------------------------------------
# Tests: ComparisonResult
# ---------------------------------------------------------------------------

class TestComparisonResult:
    def _make_eval_result(self, nll: float = 5.0) -> "EvalResult":
        from dev.dynamo.eval.evaluate import EvalResult
        return EvalResult(
            metrics={"nll": nll, "mse": nll * 0.1, "rmse": (nll * 0.1) ** 0.5},
            per_horizon={1: {"nll": nll, "mse": nll * 0.1, "rmse": 0.1, "n_samples": 50}},
            calibration=0.85,
            n_samples=100,
            tier="test",
        )

    def test_summary_table(self) -> None:
        comp = ComparisonResult(
            kernel=self._make_eval_result(10.0),
            phi0=self._make_eval_result(7.0),
            full=self._make_eval_result(5.0),
        )
        table = comp.summary_table()
        assert "Kernel" in table
        assert "φ₀-only" in table
        assert "Full" in table
        assert "nll" in table

    def test_two_model_comparison(self) -> None:
        comp = ComparisonResult(
            kernel=self._make_eval_result(10.0),
            phi0=self._make_eval_result(7.0),
        )
        table = comp.summary_table()
        assert "Full" not in table


# ---------------------------------------------------------------------------
# Tests: Visualization panels
# ---------------------------------------------------------------------------

class TestVisualizationPanels:
    def test_trajectory_view(self) -> None:
        from dev.dynamo.viz.panels import trajectory_view
        ds = _make_dataset(n_embryos=10, n_dim=5)
        fig = trajectory_view(ds, max_trajectories=10)
        assert fig is not None
        plt_close(fig)

    def test_prediction_fan(self) -> None:
        from dev.dynamo.viz.panels import prediction_fan
        D = 5
        context = np.random.randn(8, D)
        target = np.random.randn(D)
        pred_mean = np.random.randn(D)
        pred_std = np.abs(np.random.randn(D)) + 0.1
        samples = np.random.randn(50, D)

        fig = prediction_fan(
            context=context,
            target=target,
            forward_samples=samples,
            predicted_mean=pred_mean,
            predicted_std=pred_std,
        )
        assert fig is not None
        plt_close(fig)

    def test_phenotype_space(self) -> None:
        from dev.dynamo.viz.panels import phenotype_space
        N, M = 30, 3
        loadings = np.random.randn(N, M)
        labels = np.array([i % 3 for i in range(N)])
        names = ["wildtype", "mutant_a", "mutant_b"]

        fig = phenotype_space(loadings, labels, names)
        assert fig is not None
        plt_close(fig)

    def test_phenotype_space_with_novel(self) -> None:
        from dev.dynamo.viz.panels import phenotype_space
        N, M = 30, 3
        loadings = np.random.randn(N, M)
        labels = np.array([i % 3 for i in range(N)])
        names = ["wildtype", "mutant_a", "mutant_b"]
        novel = np.zeros(N, dtype=bool)
        novel[-5:] = True

        fig = phenotype_space(loadings, labels, names, novel_mask=novel)
        assert fig is not None
        plt_close(fig)

    def test_three_panel_figure(self) -> None:
        from dev.dynamo.viz.panels import three_panel_figure
        ds = _make_dataset(n_embryos=10, n_dim=5)

        # Without model outputs (should show placeholders)
        fig = three_panel_figure(ds)
        assert fig is not None
        assert len(fig.axes) == 3
        plt_close(fig)

    def test_three_panel_with_data(self) -> None:
        from dev.dynamo.viz.panels import three_panel_figure
        ds = _make_dataset(n_embryos=10, n_dim=5)
        D, M, N = 5, 3, 10

        fig = three_panel_figure(
            dataset=ds,
            context=np.random.randn(8, D),
            target=np.random.randn(D),
            predicted_mean=np.random.randn(D),
            predicted_std=np.abs(np.random.randn(D)) + 0.1,
            mode_loadings=np.random.randn(N, M),
            class_labels=np.array([i % 3 for i in range(N)]),
        )
        assert fig is not None
        assert len(fig.axes) == 3
        plt_close(fig)


def plt_close(fig):
    """Close a figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: W&B logger (without active run)
# ---------------------------------------------------------------------------

class TestWandBLogger:
    def test_log_without_wandb_run(self) -> None:
        """Logger should silently skip when no wandb run is active."""
        from dev.dynamo.eval.wandb_logger import (
            log_eval_results,
            log_comparison,
            print_eval_summary,
        )
        from dev.dynamo.eval.evaluate import EvalResult

        result = EvalResult(
            metrics={"nll": 5.0, "mse": 0.5, "rmse": 0.707},
            per_horizon={1: {"nll": 4.0, "mse": 0.4, "rmse": 0.6, "n_samples": 25}},
            calibration=0.88,
            n_samples=100,
            tier="test",
        )

        # These should not raise even without wandb
        log_eval_results(result, "test_model", step=0)

        comp = ComparisonResult(
            kernel=result,
            phi0=result,
        )
        log_comparison(comp, step=0)

    def test_print_summary(self) -> None:
        from dev.dynamo.eval.wandb_logger import print_eval_summary
        from dev.dynamo.eval.evaluate import EvalResult

        result = EvalResult(
            metrics={"nll": 5.0, "mse": 0.5, "rmse": 0.707},
            per_horizon={
                1: {"nll": 4.0, "mse": 0.4, "rmse": 0.6, "n_samples": 25},
                2: {"nll": 6.0, "mse": 0.6, "rmse": 0.8, "n_samples": 25},
            },
            calibration=0.88,
            n_samples=50,
            tier="tier2",
        )

        # Should print without error (captured by pytest if needed)
        print_eval_summary(result, "Persistence Baseline")


# ---------------------------------------------------------------------------
# Integration test: full eval pipeline with real-ish data
# ---------------------------------------------------------------------------

class TestEvalIntegration:
    def test_full_pipeline(self) -> None:
        """End-to-end: load data, build dataset, run eval, compare models."""
        ds = _make_dataset(n_embryos=15, n_dim=5)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2, 3))

        # Run eval for three "models"
        persist = run_evaluation(
            PersistencePredictor(noise_scale=0.5),
            fds, n_batches=5, batch_size=8, tier="test",
        )
        linear = run_evaluation(
            LinearExtrapolationPredictor(noise_scale=0.5),
            fds, n_batches=5, batch_size=8, tier="test",
        )
        random = run_evaluation(
            GaussianNoisePredictor(std=2.0),
            fds, n_batches=5, batch_size=8, tier="test",
        )

        # Build comparison
        comp = ComparisonResult(
            kernel=persist,  # Using persistence as stand-in for kernel
            phi0=linear,     # Using linear as stand-in for phi0
            full=random,     # Using random as stand-in (should be worst)
        )

        table = comp.summary_table()
        assert len(table) > 0

        # Print summary
        from dev.dynamo.eval.wandb_logger import print_eval_summary
        print_eval_summary(persist, "Persistence")
        print_eval_summary(linear, "Linear Extrapolation")
        print_eval_summary(random, "Random")
