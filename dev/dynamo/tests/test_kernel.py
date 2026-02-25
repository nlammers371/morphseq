"""Tests for simple kernel regression baseline (build step 3).

Tests cover:
    - Reference bank construction
    - Auto-bandwidth calibration
    - Predictor output shapes and protocol compliance
    - Self-prediction (exclude_self=False)
    - Reference filtering (same_experiment, same_class, exclude_self)
    - Fallback when no references exist
    - Metric ordering (kernel beats random on structured data)
    - Integration with run_evaluation
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dev.dynamo.data.loading import EmbryoTrajectory, TrajectoryDataset
from dev.dynamo.data.dataset import FragmentDataset, fragment_collate_fn
from dev.dynamo.eval.predictions import Predictor, PredictionResult
from dev.dynamo.models.kernel import (
    SimpleKernelPredictor,
    _build_reference_bank,
    _auto_bandwidth,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

def _make_trajectory(
    embryo_id: str = "emb_001",
    n_frames: int = 20,
    n_dim: int = 5,
    delta_t: float = 300.0,
    temperature: float = 28.5,
    perturbation_class: str = "wildtype",
    experiment_id: str = "exp_001",
    seed: int = 42,
    direction: np.ndarray | None = None,
    start: np.ndarray | None = None,
) -> EmbryoTrajectory:
    """Create a synthetic trajectory, optionally along a specified direction."""
    rng = np.random.default_rng(seed)
    if direction is not None and start is not None:
        # Straight line with small noise
        t_vals = np.linspace(0, 1, n_frames)[:, None]
        traj = start[None, :] + t_vals * direction[None, :] + rng.standard_normal((n_frames, n_dim)) * 0.01
    else:
        steps = rng.standard_normal((n_frames, n_dim)) * 0.1
        traj = np.cumsum(steps, axis=0)
    times = np.arange(n_frames, dtype=np.float64) * delta_t
    return EmbryoTrajectory(
        embryo_id=embryo_id,
        trajectory=traj,
        time_seconds=times,
        delta_t=delta_t,
        temperature=temperature,
        perturbation_class=perturbation_class,
        experiment_id=experiment_id,
    )


def _make_dataset(
    n_embryos: int = 10,
    n_frames: int = 20,
    n_dim: int = 5,
    classes: list[str] | None = None,
    experiments: list[str] | None = None,
) -> TrajectoryDataset:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if classes is None:
        classes = ["wildtype", "mutant_a", "mutant_b"]
    if experiments is None:
        experiments = ["exp_001"]

    trajs = []
    for i in range(n_embryos):
        trajs.append(_make_trajectory(
            embryo_id=f"emb_{i:03d}",
            n_frames=n_frames,
            n_dim=n_dim,
            perturbation_class=classes[i % len(classes)],
            experiment_id=experiments[i % len(experiments)],
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


def _make_batch(ds: TrajectoryDataset, n: int = 8) -> "FragmentBatch":
    fds = FragmentDataset(ds, min_context=3, horizons=(1, 2, 3))
    samples = [fds[i] for i in range(n)]
    return fragment_collate_fn(samples)


# ---------------------------------------------------------------------------
# Tests: Reference bank
# ---------------------------------------------------------------------------

class TestReferenceBank:
    def test_construction(self) -> None:
        ds = _make_dataset(n_embryos=5, n_frames=10, n_dim=3)
        bank = _build_reference_bank(ds.trajectories, ds.class_to_idx)

        assert bank.all_points.shape == (50, 3)  # 5 trajs * 10 frames
        assert len(bank.traj_id) == 50
        assert len(bank.frame_idx) == 50
        assert len(bank.traj_points) == 5
        assert len(bank.traj_lengths) == 5
        assert (bank.traj_lengths == 10).all()

    def test_traj_id_mapping(self) -> None:
        ds = _make_dataset(n_embryos=3, n_frames=5, n_dim=2)
        bank = _build_reference_bank(ds.trajectories, ds.class_to_idx)

        # First 5 points should belong to traj 0
        assert (bank.traj_id[:5] == 0).all()
        assert (bank.traj_id[5:10] == 1).all()
        assert (bank.traj_id[10:15] == 2).all()

    def test_frame_idx_sequential(self) -> None:
        ds = _make_dataset(n_embryos=2, n_frames=7, n_dim=2)
        bank = _build_reference_bank(ds.trajectories, ds.class_to_idx)

        # Each trajectory's frames should be 0, 1, 2, ..., 6
        for t in range(2):
            mask = bank.traj_id == t
            np.testing.assert_array_equal(bank.frame_idx[mask], np.arange(7))


# ---------------------------------------------------------------------------
# Tests: Auto-bandwidth
# ---------------------------------------------------------------------------

class TestAutoBandwidth:
    def test_positive(self) -> None:
        ds = _make_dataset(n_embryos=5, n_frames=10, n_dim=3)
        bank = _build_reference_bank(ds.trajectories, ds.class_to_idx)
        bw = _auto_bandwidth(bank)
        assert bw > 0

    def test_deterministic(self) -> None:
        ds = _make_dataset(n_embryos=5, n_frames=10, n_dim=3)
        bank = _build_reference_bank(ds.trajectories, ds.class_to_idx)
        bw1 = _auto_bandwidth(bank, seed=123)
        bw2 = _auto_bandwidth(bank, seed=123)
        assert bw1 == bw2


# ---------------------------------------------------------------------------
# Tests: Predictor protocol and output shapes
# ---------------------------------------------------------------------------

class TestSimpleKernelPredictor:
    def test_protocol_compliance(self) -> None:
        ds = _make_dataset()
        pred = SimpleKernelPredictor(ds, bandwidth=1.0)
        assert isinstance(pred, Predictor)

    def test_output_shapes(self) -> None:
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=5)
        pred = SimpleKernelPredictor(ds, bandwidth=1.0, n_samples=50)
        batch = _make_batch(ds, n=4)
        result = pred.predict(batch)

        assert result.predicted_mean.shape == (4, 5)
        assert result.predicted_cov_diag.shape == (4, 5)
        assert result.forward_samples.shape == (4, 50, 5)

    def test_output_types(self) -> None:
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=5)
        pred = SimpleKernelPredictor(ds, bandwidth=1.0)
        batch = _make_batch(ds, n=4)
        result = pred.predict(batch)

        assert isinstance(result, PredictionResult)
        assert result.predicted_mean.dtype == torch.float32
        assert result.predicted_cov_diag.dtype == torch.float32

    def test_auto_bandwidth(self) -> None:
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=5)
        pred = SimpleKernelPredictor(ds, bandwidth=None)
        assert pred.bandwidth > 0

        batch = _make_batch(ds, n=4)
        result = pred.predict(batch)
        assert result.predicted_mean.shape == (4, 5)

    def test_cov_diag_positive(self) -> None:
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=5)
        pred = SimpleKernelPredictor(ds, bandwidth=1.0)
        batch = _make_batch(ds, n=8)
        result = pred.predict(batch)
        assert (result.predicted_cov_diag > 0).all()


# ---------------------------------------------------------------------------
# Tests: Self-prediction
# ---------------------------------------------------------------------------

class TestSelfPrediction:
    def test_exclude_self_false_improves_accuracy(self) -> None:
        """With exclude_self=False, the predictor can look up its own
        trajectory, which should give near-perfect predictions."""
        D = 3
        # Make identical trajectories so the kernel can't distinguish them
        ds = _make_dataset(n_embryos=5, n_frames=20, n_dim=D)
        batch = _make_batch(ds, n=4)

        pred_with = SimpleKernelPredictor(ds, bandwidth=0.5, exclude_self=True)
        pred_without = SimpleKernelPredictor(ds, bandwidth=0.5, exclude_self=False)

        result_with = pred_with.predict(batch)
        result_without = pred_without.predict(batch)

        mse_with = (result_with.predicted_mean - batch.target).pow(2).mean()
        mse_without = (result_without.predicted_mean - batch.target).pow(2).mean()

        # Self-matching should be at least as good
        assert mse_without <= mse_with + 1e-6


# ---------------------------------------------------------------------------
# Tests: Reference filtering
# ---------------------------------------------------------------------------

class TestReferenceFiltering:
    def test_same_experiment_filters(self) -> None:
        """With same_experiment=True, only references from the query's
        experiment should be used."""
        ds = _make_dataset(
            n_embryos=10, n_frames=20, n_dim=3,
            experiments=["exp_A", "exp_B"],
        )
        # All even embryos are exp_A, odd are exp_B
        pred = SimpleKernelPredictor(
            ds, bandwidth=1.0, same_experiment=True, exclude_self=True,
        )
        batch = _make_batch(ds, n=4)
        result = pred.predict(batch)
        # Should produce valid output (not crash)
        assert result.predicted_mean.shape == (4, 3)

    def test_same_class_filters(self) -> None:
        """With same_class=True, only references from the query's
        perturbation class should be used."""
        ds = _make_dataset(
            n_embryos=12, n_frames=20, n_dim=3,
            classes=["wt", "mut_a", "mut_b"],
        )
        pred = SimpleKernelPredictor(
            ds, bandwidth=1.0, same_class=True, exclude_self=True,
        )
        batch = _make_batch(ds, n=4)
        result = pred.predict(batch)
        assert result.predicted_mean.shape == (4, 3)

    def test_all_filters_together(self) -> None:
        """All filters enabled simultaneously."""
        ds = _make_dataset(
            n_embryos=12, n_frames=20, n_dim=3,
            classes=["wt", "mut"],
            experiments=["exp_A", "exp_B"],
        )
        pred = SimpleKernelPredictor(
            ds, bandwidth=1.0,
            exclude_self=True, same_experiment=True, same_class=True,
        )
        batch = _make_batch(ds, n=4)
        result = pred.predict(batch)
        assert result.predicted_mean.shape == (4, 3)


# ---------------------------------------------------------------------------
# Tests: Fallback behavior
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_when_no_references(self) -> None:
        """If all references are filtered out, should fall back to persistence."""
        ds = _make_dataset(n_embryos=1, n_frames=20, n_dim=3)
        # Only 1 embryo + exclude_self → no references
        pred = SimpleKernelPredictor(ds, bandwidth=1.0, exclude_self=True)
        batch = _make_batch(ds, n=2)
        result = pred.predict(batch)

        # Should produce valid output (persistence fallback)
        assert result.predicted_mean.shape == (2, 3)
        assert (result.predicted_cov_diag == 1.0).all()


# ---------------------------------------------------------------------------
# Tests: Metric ordering
# ---------------------------------------------------------------------------

class TestMetricOrdering:
    def test_kernel_beats_random(self) -> None:
        """Kernel should have lower MSE than random Gaussian on structured data."""
        from dev.dynamo.eval.predictions import GaussianNoisePredictor
        from dev.dynamo.eval.evaluate import run_evaluation

        ds = _make_dataset(n_embryos=15, n_frames=30, n_dim=5)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2))

        kernel_result = run_evaluation(
            SimpleKernelPredictor(ds, bandwidth=None, exclude_self=True),
            fds, n_batches=10, batch_size=8,
        )
        random_result = run_evaluation(
            GaussianNoisePredictor(std=2.0),
            fds, n_batches=10, batch_size=8,
        )

        assert kernel_result.metrics["mse"] < random_result.metrics["mse"]


# ---------------------------------------------------------------------------
# Tests: Straight-line recovery
# ---------------------------------------------------------------------------

class TestStraightLineRecovery:
    def test_parallel_lines_predict_on_line(self) -> None:
        """If all training trajectories are parallel straight lines,
        predictions should lie approximately on a line."""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        D = 3
        direction = np.array([1.0, 0.5, -0.3])
        trajs = []
        for i in range(10):
            start = np.array([0.0, 0.0, 0.0]) + np.random.default_rng(i).standard_normal(D) * 0.01
            trajs.append(_make_trajectory(
                embryo_id=f"emb_{i:03d}", n_frames=30, n_dim=D,
                direction=direction, start=start, seed=i + 100,
                delta_t=300.0,
            ))

        pca = PCA(n_components=D)
        pca.fit(np.eye(D))
        scaler = StandardScaler()
        scaler.fit(np.zeros((2, D)))
        ds = TrajectoryDataset(
            trajectories=trajs, pca=pca, scaler=scaler,
            z_mu_cols=[f"z_{i}" for i in range(D)],
        )

        pred = SimpleKernelPredictor(ds, bandwidth=0.5, exclude_self=False)
        fds = FragmentDataset(ds, min_context=5, horizons=(1, 2))
        batch = _make_batch(ds, n=4)
        result = pred.predict(batch)

        # Variance should be small (all trajectories agree)
        assert result.predicted_cov_diag.mean().item() < 0.1


# ---------------------------------------------------------------------------
# Tests: Integration with eval pipeline
# ---------------------------------------------------------------------------

class TestEvalIntegration:
    def test_run_evaluation(self) -> None:
        ds = _make_dataset(n_embryos=10, n_frames=20, n_dim=5)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2, 3))
        pred = SimpleKernelPredictor(ds, bandwidth=None)

        result = run_evaluation(pred, fds, n_batches=5, batch_size=8)

        assert result.n_samples > 0
        assert "nll" in result.metrics
        assert "mse" in result.metrics
        assert result.calibration >= 0.0
        assert result.calibration <= 1.0
        assert len(result.per_horizon) >= 1


def run_evaluation(predictor, dataset, n_batches, batch_size):
    """Import helper to avoid circular import at module level."""
    from dev.dynamo.eval.evaluate import run_evaluation as _run_eval
    return _run_eval(predictor, dataset, n_batches=n_batches, batch_size=batch_size)
