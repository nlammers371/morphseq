"""Tests for data loading and fragment sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dev.dynamo.data.loading import EmbryoTrajectory, TrajectoryDataset
from dev.dynamo.data.dataset import FragmentDataset, FragmentBatch, fragment_collate_fn


# ---------------------------------------------------------------------------
# Fixtures: synthetic trajectories
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
    # Brownian-walk trajectory
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

    # Dummy PCA + scaler (identity-like)
    pca = PCA(n_components=n_dim)
    pca.fit(np.eye(n_dim))  # trivial fit
    scaler = StandardScaler()
    scaler.fit(np.zeros((2, n_dim)))

    return TrajectoryDataset(
        trajectories=trajs,
        pca=pca,
        scaler=scaler,
        z_mu_cols=[f"z_mu_b_{i:02d}" for i in range(n_dim)],
    )


# ---------------------------------------------------------------------------
# Tests: EmbryoTrajectory
# ---------------------------------------------------------------------------

class TestEmbryoTrajectory:
    def test_shape(self) -> None:
        t = _make_trajectory(n_frames=15, n_dim=10)
        assert t.trajectory.shape == (15, 10)
        assert t.time_seconds.shape == (15,)

    def test_immutable(self) -> None:
        t = _make_trajectory()
        with pytest.raises(AttributeError):
            t.embryo_id = "new_id"


# ---------------------------------------------------------------------------
# Tests: TrajectoryDataset
# ---------------------------------------------------------------------------

class TestTrajectoryDataset:
    def test_class_to_idx(self) -> None:
        ds = _make_dataset(n_embryos=6)
        assert set(ds.class_to_idx.keys()) == {"wildtype", "mutant_a", "mutant_b"}
        # Indices should be contiguous
        assert set(ds.class_to_idx.values()) == {0, 1, 2}

    def test_filter_by_class(self) -> None:
        ds = _make_dataset(n_embryos=9)
        sub = ds.filter(perturbation_classes=["wildtype"])
        assert all(t.perturbation_class == "wildtype" for t in sub.trajectories)
        assert len(sub.trajectories) == 3  # 9 embryos, 3 classes, round-robin

    def test_filter_by_experiment(self) -> None:
        ds = _make_dataset(n_embryos=5)
        sub = ds.filter(experiment_ids=["exp_001"])
        assert len(sub.trajectories) == 5  # all are exp_001

    def test_n_components(self) -> None:
        ds = _make_dataset(n_dim=7)
        assert ds.n_components == 7


# ---------------------------------------------------------------------------
# Tests: FragmentDataset
# ---------------------------------------------------------------------------

class TestFragmentDataset:
    def test_length(self) -> None:
        ds = _make_dataset()
        fds = FragmentDataset(ds, epoch_length=50)
        assert len(fds) == 50

    def test_default_length(self) -> None:
        ds = _make_dataset(n_embryos=8)
        fds = FragmentDataset(ds)
        # Default epoch_length = number of valid trajectories
        assert len(fds) == 8

    def test_sample_shapes(self) -> None:
        ds = _make_dataset(n_dim=10)
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2))
        sample = fds[0]

        D = 10
        L = sample["context"].shape[0]
        assert sample["context"].shape == (L, D)
        assert sample["target"].shape == (D,)
        assert sample["time_deltas"].shape == (L - 1,)
        assert sample["horizon_dt"].shape == ()
        assert sample["delta_t"].shape == ()
        assert sample["class_idx"].dtype == torch.long

    def test_context_min_length(self) -> None:
        ds = _make_dataset(n_frames=20)
        fds = FragmentDataset(ds, min_context=5)
        for _ in range(50):
            sample = fds[0]
            assert sample["context"].shape[0] >= 5

    def test_horizon_within_trajectory(self) -> None:
        """Target index should never exceed trajectory length."""
        ds = _make_dataset(n_frames=10, n_dim=5)
        fds = FragmentDataset(ds, min_context=2, horizons=(1, 2, 3, 4))
        # Smoke test: no IndexError over many samples
        for _ in range(200):
            sample = fds[0]
            assert sample["target"].shape == (5,)

    def test_time_deltas_positive(self) -> None:
        ds = _make_dataset()
        fds = FragmentDataset(ds, min_context=2)
        for _ in range(50):
            sample = fds[0]
            td = sample["time_deltas"]
            assert (td >= 0).all(), "Time deltas should be non-negative"

    def test_horizon_dt_positive(self) -> None:
        ds = _make_dataset()
        fds = FragmentDataset(ds, min_context=2)
        for _ in range(50):
            sample = fds[0]
            assert sample["horizon_dt"].item() > 0

    def test_short_trajectories_filtered(self) -> None:
        """Trajectories shorter than min_context+1 should be excluded."""
        from sklearn.decomposition import PCA
        trajs = [
            _make_trajectory(embryo_id="short", n_frames=2),
            _make_trajectory(embryo_id="long", n_frames=20, seed=99),
        ]
        pca = PCA(n_components=10)
        pca.fit(np.eye(10))
        ds = TrajectoryDataset(trajs, pca=pca, scaler=None, z_mu_cols=[])
        fds = FragmentDataset(ds, min_context=3)
        # Only the long trajectory should be valid
        assert len(fds.valid_indices) == 1

    def test_no_valid_trajectories_raises(self) -> None:
        from sklearn.decomposition import PCA
        trajs = [_make_trajectory(n_frames=2)]
        pca = PCA(n_components=10)
        pca.fit(np.eye(10))
        ds = TrajectoryDataset(trajs, pca=pca, scaler=None, z_mu_cols=[])
        with pytest.raises(ValueError, match="No trajectories"):
            FragmentDataset(ds, min_context=5)


# ---------------------------------------------------------------------------
# Tests: Collation
# ---------------------------------------------------------------------------

class TestCollation:
    def _get_samples(self, n: int = 4) -> list:
        ds = _make_dataset(n_dim=5)
        fds = FragmentDataset(ds, min_context=2, max_context=8, horizons=(1, 2))
        return [fds[i] for i in range(n)]

    def test_batch_shapes(self) -> None:
        samples = self._get_samples(4)
        batch = fragment_collate_fn(samples)
        B = 4
        D = 5
        L_max = batch.context.shape[1]

        assert batch.context.shape == (B, L_max, D)
        assert batch.context_mask.shape == (B, L_max)
        assert batch.target.shape == (B, D)
        assert batch.time_deltas.shape == (B, L_max - 1)
        assert batch.horizon_dt.shape == (B,)
        assert batch.delta_t.shape == (B,)
        assert batch.temperature.shape == (B,)
        assert batch.class_idx.shape == (B,)

    def test_mask_consistency(self) -> None:
        samples = self._get_samples(4)
        batch = fragment_collate_fn(samples)

        for i, s in enumerate(samples):
            L = s["context"].shape[0]
            # Real frames should be True
            assert batch.context_mask[i, :L].all()
            # Padding should be False
            if L < batch.context_mask.shape[1]:
                assert not batch.context_mask[i, L:].any()

    def test_padded_context_values(self) -> None:
        samples = self._get_samples(4)
        batch = fragment_collate_fn(samples)

        for i, s in enumerate(samples):
            L = s["context"].shape[0]
            # Real values should match
            torch.testing.assert_close(batch.context[i, :L], s["context"])
            # Padded values should be zero
            if L < batch.context.shape[1]:
                assert (batch.context[i, L:] == 0).all()


# ---------------------------------------------------------------------------
# Integration test: real data (skipped if files missing)
# ---------------------------------------------------------------------------

class TestRealData:
    BUILD_DIR = "morphseq_playground/metadata/build06_output"
    EXP_ID = "20251207_pbx"

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self) -> None:
        from pathlib import Path
        path = Path(self.BUILD_DIR) / f"df03_final_output_with_latents_{self.EXP_ID}.csv"
        if not path.exists():
            pytest.skip(f"Real data not available: {path}")

    def test_load_real_experiment(self) -> None:
        from dev.dynamo.data.loading import load_trajectories
        ds = load_trajectories(
            experiment_ids=[self.EXP_ID],
            build_dir=self.BUILD_DIR,
            n_components=10,
            verbose=True,
        )
        assert len(ds.trajectories) > 0
        assert ds.n_components == 10

        # Check trajectory properties
        for t in ds.trajectories:
            assert t.trajectory.shape[1] == 10
            assert len(t.time_seconds) == len(t.trajectory)
            assert t.perturbation_class != ""
            assert t.experiment_id == self.EXP_ID

    def test_fragment_sampling_real(self) -> None:
        from dev.dynamo.data.loading import load_trajectories
        ds = load_trajectories(
            experiment_ids=[self.EXP_ID],
            build_dir=self.BUILD_DIR,
            n_components=10,
            verbose=False,
        )
        fds = FragmentDataset(ds, min_context=3, horizons=(1, 2, 3, 4))
        # Sample a batch
        samples = [fds[i] for i in range(8)]
        batch = fragment_collate_fn(samples)

        assert batch.context.shape[0] == 8
        assert batch.context.shape[2] == 10
        assert batch.target.shape == (8, 10)
        assert (batch.horizon_dt > 0).all()
