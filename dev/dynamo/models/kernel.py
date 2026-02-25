"""Simple kernel regression baseline (model spec §9.1).

Given a query point z_hat(t), finds nearby training trajectory points via
Gaussian kernel weighting, then reports the weighted distribution of where
those points went at the requested prediction horizon.

One hyperparameter: kernel bandwidth sigma. Dead simple, no direction
awareness, no recruitment. This is the absolute floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..data.dataset import FragmentBatch
from ..data.loading import EmbryoTrajectory, TrajectoryDataset
from ..eval.predictions import PredictionResult


# ---------------------------------------------------------------------------
# Internal data structures for the reference bank
# ---------------------------------------------------------------------------

@dataclass
class _ReferenceBank:
    """Pre-concatenated training data for fast vectorized lookup.

    Attributes:
        all_points: (N_total, D) all training trajectory points concatenated.
        traj_id: (N_total,) which trajectory each point belongs to.
        frame_idx: (N_total,) frame index within its trajectory.
        traj_lengths: (N_traj,) length of each trajectory.
        traj_points: List of (T_i, D) arrays per trajectory.
        traj_delta_t: (N_traj,) delta_t per trajectory (seconds).
        embryo_idx: (N_traj,) original index into the trajectory list.
        experiment_id: List of experiment_id strings per trajectory.
        class_idx: (N_traj,) perturbation class index per trajectory.
    """
    all_points: np.ndarray
    traj_id: np.ndarray
    frame_idx: np.ndarray
    traj_lengths: np.ndarray
    traj_points: List[np.ndarray]
    traj_delta_t: np.ndarray
    embryo_idx: np.ndarray
    experiment_id: List[str]
    class_idx: np.ndarray


def _build_reference_bank(
    trajectories: List[EmbryoTrajectory],
    class_to_idx: dict[str, int],
) -> _ReferenceBank:
    """Build a pre-concatenated reference bank from trajectory list."""
    all_pts: List[np.ndarray] = []
    all_traj_id: List[np.ndarray] = []
    all_frame_idx: List[np.ndarray] = []
    traj_points: List[np.ndarray] = []
    traj_lengths: List[int] = []
    traj_delta_t: List[float] = []
    embryo_idx: List[int] = []
    experiment_ids: List[str] = []
    class_idxs: List[int] = []

    for i, traj in enumerate(trajectories):
        T = len(traj.trajectory)
        all_pts.append(traj.trajectory)
        all_traj_id.append(np.full(T, i, dtype=np.int64))
        all_frame_idx.append(np.arange(T, dtype=np.int64))
        traj_points.append(traj.trajectory)
        traj_lengths.append(T)
        traj_delta_t.append(traj.delta_t)
        embryo_idx.append(i)
        experiment_ids.append(traj.experiment_id)
        class_idxs.append(class_to_idx.get(traj.perturbation_class, -1))

    return _ReferenceBank(
        all_points=np.concatenate(all_pts, axis=0),
        traj_id=np.concatenate(all_traj_id),
        frame_idx=np.concatenate(all_frame_idx),
        traj_lengths=np.array(traj_lengths, dtype=np.int64),
        traj_points=traj_points,
        traj_delta_t=np.array(traj_delta_t, dtype=np.float64),
        embryo_idx=np.array(embryo_idx, dtype=np.int64),
        experiment_id=experiment_ids,
        class_idx=np.array(class_idxs, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Auto-calibration
# ---------------------------------------------------------------------------

def _auto_bandwidth(bank: _ReferenceBank, n_samples: int = 1000, seed: int = 42) -> float:
    """Estimate kernel bandwidth via median heuristic.

    Subsamples points from the reference bank, computes pairwise distances,
    and returns the median as a bandwidth estimate.

    Args:
        bank: Pre-built reference bank.
        n_samples: Number of points to subsample.
        seed: RNG seed for reproducibility.

    Returns:
        Estimated bandwidth sigma (positive scalar).
    """
    rng = np.random.default_rng(seed)
    N = len(bank.all_points)
    n = min(n_samples, N)
    idx = rng.choice(N, size=n, replace=False)
    pts = bank.all_points[idx]

    # Pairwise distances (subsample pairs if too many)
    n_pairs = min(50_000, n * (n - 1) // 2)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    # Avoid self-pairs
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    dists = np.linalg.norm(pts[i_idx] - pts[j_idx], axis=1)
    return float(np.median(dists))


# ---------------------------------------------------------------------------
# Simple kernel predictor
# ---------------------------------------------------------------------------

class SimpleKernelPredictor:
    """Simple kernel regression baseline (spec §9.1).

    For each query point, weights all training trajectory points by a
    Gaussian kernel on Euclidean distance, then looks at where each
    weighted point went at the requested horizon. Returns the weighted
    distribution of outcomes.

    Args:
        dataset: TrajectoryDataset containing training trajectories.
        bandwidth: Kernel bandwidth sigma. None = auto (median heuristic).
        n_samples: Number of forward samples to resample per query.
        exclude_self: Skip query embryo's trajectory from reference set.
        same_experiment: Restrict references to same experiment as query.
        same_class: Restrict references to same perturbation class as query.
        min_weight: Minimum total weight to produce a prediction (below
            this, fall back to persistence).
    """

    def __init__(
        self,
        dataset: TrajectoryDataset,
        bandwidth: Optional[float] = None,
        n_samples: int = 100,
        exclude_self: bool = True,
        same_experiment: bool = False,
        same_class: bool = False,
        min_weight: float = 1e-10,
    ) -> None:
        self.trajectories = dataset.trajectories
        self.class_to_idx = dataset.class_to_idx
        self.bank = _build_reference_bank(self.trajectories, self.class_to_idx)
        self.n_samples = n_samples
        self.exclude_self = exclude_self
        self.same_experiment = same_experiment
        self.same_class = same_class
        self.min_weight = min_weight
        self._rng = np.random.default_rng()

        if bandwidth is not None:
            self.bandwidth = bandwidth
        else:
            self.bandwidth = _auto_bandwidth(self.bank)

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        """Produce predictions for a batch of fragments.

        Args:
            batch: FragmentBatch from the data pipeline.

        Returns:
            PredictionResult with predicted_mean, predicted_cov_diag,
            and forward_samples.
        """
        B = batch.context.shape[0]
        D = batch.context.shape[2]

        means = np.zeros((B, D))
        cov_diags = np.zeros((B, D))
        all_samples = np.zeros((B, self.n_samples, D))

        for b in range(B):
            result = self._predict_single(
                context=batch.context[b].numpy(),
                context_mask=batch.context_mask[b].numpy(),
                horizon_dt=batch.horizon_dt[b].item(),
                delta_t=batch.delta_t[b].item(),
                embryo_idx=batch.embryo_idx[b].item(),
                class_idx=batch.class_idx[b].item(),
            )
            means[b] = result["mean"]
            cov_diags[b] = result["cov_diag"]
            all_samples[b] = result["samples"]

        return PredictionResult(
            predicted_mean=torch.from_numpy(means).float(),
            predicted_cov_diag=torch.from_numpy(cov_diags).float(),
            forward_samples=torch.from_numpy(all_samples).float(),
        )

    def _predict_single(
        self,
        context: np.ndarray,
        context_mask: np.ndarray,
        horizon_dt: float,
        delta_t: float,
        embryo_idx: int,
        class_idx: int,
    ) -> dict:
        """Predict for a single query fragment.

        Args:
            context: (L_max, D) padded context array.
            context_mask: (L_max,) boolean mask for real frames.
            horizon_dt: Time gap to target in seconds.
            delta_t: Query experiment's median delta_t in seconds.
            embryo_idx: Index of the query embryo in the trajectory list.
            class_idx: Perturbation class index of the query.

        Returns:
            Dict with 'mean' (D,), 'cov_diag' (D,), 'samples' (n_samples, D).
        """
        # Extract last real context frame as query point
        L = int(context_mask.sum())
        query_point = context[L - 1]  # (D,)
        D = len(query_point)

        # Build eligibility mask over reference points
        eligible = np.ones(len(self.bank.all_points), dtype=bool)

        if self.exclude_self:
            eligible &= self.bank.traj_id != embryo_idx

        if self.same_experiment:
            query_exp = self.trajectories[embryo_idx].experiment_id
            exp_mask = np.zeros(len(self.bank.traj_points), dtype=bool)
            for i, eid in enumerate(self.bank.experiment_id):
                exp_mask[i] = (eid == query_exp)
            # Expand per-trajectory mask to per-point mask
            eligible &= exp_mask[self.bank.traj_id]

        if self.same_class:
            cls_mask = self.bank.class_idx == class_idx
            eligible &= cls_mask[self.bank.traj_id]

        if not eligible.any():
            return self._fallback(query_point, D)

        # Compute distances from query point to all eligible reference points
        eligible_indices = np.where(eligible)[0]
        ref_points = self.bank.all_points[eligible_indices]  # (N_elig, D)
        dists_sq = np.sum((ref_points - query_point) ** 2, axis=1)  # (N_elig,)

        # Gaussian kernel weights
        sigma_sq = self.bandwidth ** 2
        weights = np.exp(-dists_sq / (2.0 * sigma_sq))  # (N_elig,)

        # For each weighted point, find where it went at the requested horizon
        ref_traj_ids = self.bank.traj_id[eligible_indices]
        ref_frame_idxs = self.bank.frame_idx[eligible_indices]

        # Horizon in frames for each reference (depends on that trajectory's delta_t)
        ref_delta_ts = self.bank.traj_delta_t[ref_traj_ids]
        horizon_frames = np.round(horizon_dt / np.maximum(ref_delta_ts, 1e-6)).astype(np.int64)

        # Target frame index for each reference point
        target_frame_idxs = ref_frame_idxs + horizon_frames
        ref_traj_lengths = self.bank.traj_lengths[ref_traj_ids]

        # Only keep points whose trajectory extends far enough
        valid = target_frame_idxs < ref_traj_lengths
        if not valid.any() or weights[valid].sum() < self.min_weight:
            return self._fallback(query_point, D)

        weights = weights[valid]
        ref_traj_ids = ref_traj_ids[valid]
        target_frame_idxs = target_frame_idxs[valid]

        # Gather target positions
        targets = np.array([
            self.bank.traj_points[tid][tfid]
            for tid, tfid in zip(ref_traj_ids, target_frame_idxs)
        ])  # (N_valid, D)

        # Normalize weights
        w_sum = weights.sum()
        w_norm = weights / w_sum  # (N_valid,)

        # Weighted mean
        mean = w_norm @ targets  # (D,)

        # Weighted covariance (diagonal)
        diff = targets - mean  # (N_valid, D)
        cov_diag = (w_norm[:, None] * diff ** 2).sum(axis=0)  # (D,)
        # Floor variance to avoid zero
        cov_diag = np.maximum(cov_diag, 1e-8)

        # Resample forward samples from the weighted point cloud
        indices = self._rng.choice(
            len(targets), size=self.n_samples, replace=True, p=w_norm,
        )
        samples = targets[indices]  # (n_samples, D)

        return {"mean": mean, "cov_diag": cov_diag, "samples": samples}

    def _fallback(self, query_point: np.ndarray, D: int) -> dict:
        """Persistence fallback when no valid references exist."""
        return {
            "mean": query_point.copy(),
            "cov_diag": np.ones(D),
            "samples": np.tile(query_point, (self.n_samples, 1)),
        }
