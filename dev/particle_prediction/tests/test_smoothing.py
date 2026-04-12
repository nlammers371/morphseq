import numpy as np

from dev.particle_prediction.data.loading import EmbryoTrajectory, TrajectoryDataset
from dev.particle_prediction.data.smoothing import resolve_window_frames, smooth_dataset, smooth_trajectory


def _make_trajectory(length: int = 21, dims: int = 3) -> EmbryoTrajectory:
    time_seconds = np.arange(length, dtype=np.float64) * 60.0
    x = np.linspace(0.0, 2.0 * np.pi, length)
    trajectory = np.stack(
        [
            np.sin(x),
            np.cos(x),
            np.sin(2.0 * x),
        ],
        axis=1,
    )[:, :dims]
    trajectory = trajectory + 0.05 * np.random.default_rng(0).normal(size=trajectory.shape)
    return EmbryoTrajectory(
        embryo_id="emb_test",
        trajectory=trajectory,
        time_seconds=time_seconds,
        delta_t=60.0,
        temperature=28.5,
        perturbation_class="wt",
        experiment_id="exp_test",
    )


def test_resolve_window_frames_prefers_tres_pair() -> None:
    resolved = resolve_window_frames(
        trajectory_length=21,
        poly_order=2,
        window_frames=9,
        tres=2.0,
        smoothing_tres=10.0,
    )
    assert resolved == 5


def test_smooth_trajectory_preserves_shape_and_returns_finite_values() -> None:
    raw = _make_trajectory()
    smoothed = smooth_trajectory(raw, poly_order=2, window_frames=5)

    assert smoothed.smoothed.shape == raw.trajectory.shape
    assert smoothed.residuals is not None
    assert smoothed.residuals.shape == raw.trajectory.shape
    assert smoothed.window_frames == 5
    assert np.isfinite(smoothed.smoothed).all()
    assert smoothed.diagnostics["applied"] is True


def test_smooth_trajectory_short_series_degrades_gracefully() -> None:
    raw = _make_trajectory(length=3, dims=2)
    smoothed = smooth_trajectory(raw, poly_order=2, window_frames=5)

    np.testing.assert_allclose(smoothed.smoothed, raw.trajectory)
    assert smoothed.diagnostics["applied"] is False


def test_smooth_dataset_wraps_all_trajectories() -> None:
    raw = _make_trajectory()
    dataset = TrajectoryDataset(
        trajectories=[raw],
        pca=type("PCAStub", (), {"n_components_": 3})(),
        scaler=None,
        z_mu_cols=["z_mu_b0", "z_mu_b1", "z_mu_b2"],
    )

    smoothed_dataset = smooth_dataset(dataset, poly_order=2, window_frames=5)
    assert len(smoothed_dataset) == 1
    assert smoothed_dataset.class_names == ["wt"]