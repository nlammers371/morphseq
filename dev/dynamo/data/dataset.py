"""PyTorch Dataset implementing fragment sampling (model spec §7.2).

Each sample is a randomly drawn contiguous fragment from one embryo's
trajectory, paired with a target observation at a randomly sampled
prediction horizon k ∈ {1, 2, 3, 4}.

Returned tensors:
    context    : (L, D)   PC-space trajectory fragment (the "observed" part)
    target     : (D,)     single PC-space vector at horizon k after context
    time_deltas: (L-1,)   inter-frame Δt values within the context (seconds)
    horizon_dt : scalar   time gap from last context frame to target (seconds)
    delta_t    : scalar   experiment-level median Δt (seconds)
    temperature: scalar   incubation temperature (°C, may be NaN)
    class_idx  : int      integer index for perturbation class
    embryo_idx : int      index into the trajectory list

The custom collate function `fragment_collate_fn` pads variable-length
context fragments and returns a boolean mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .loading import TrajectoryDataset


class FragmentDataset(Dataset):
    """Dataset that samples random fragments + prediction horizons from embryo trajectories.

    Args:
        trajectory_dataset: Loaded TrajectoryDataset from loading.py.
        min_context: Minimum number of context frames (≥2 for at least one transition).
        max_context: Maximum context frames. None = use all available (minus horizon).
        horizons: Prediction horizons to sample from (in units of frames).
        epoch_length: Virtual epoch size. Since sampling is stochastic, this controls
            how many samples constitute one "epoch" for the DataLoader.
    """

    def __init__(
        self,
        trajectory_dataset: TrajectoryDataset,
        min_context: int = 2,
        max_context: Optional[int] = None,
        horizons: Sequence[int] = (1, 2, 3, 4),
        epoch_length: Optional[int] = None,
    ) -> None:
        self.trajs = trajectory_dataset.trajectories
        self.class_to_idx = trajectory_dataset.class_to_idx
        self.n_components = trajectory_dataset.n_components
        self.min_context = min_context
        self.max_context = max_context
        self.horizons = list(horizons)
        self.max_horizon = max(self.horizons)

        # Pre-filter to trajectories long enough for at least min_context + 1 horizon
        min_len = self.min_context + 1
        self.valid_indices = [
            i for i, t in enumerate(self.trajs)
            if len(t.trajectory) >= min_len
        ]
        if not self.valid_indices:
            raise ValueError(
                f"No trajectories with ≥{min_len} frames "
                f"(min_context={min_context}, min horizon=1)"
            )
        self._epoch_length = epoch_length or len(self.valid_indices)
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return self._epoch_length

    def seed_worker(self, seed: int) -> None:
        """Re-seed the RNG (call from DataLoader worker_init_fn)."""
        self._rng = np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Sample a random fragment + horizon from a random trajectory.

        The ``idx`` argument is used only to drive the DataLoader iteration
        count; the actual embryo and fragment are sampled randomly.
        """
        rng = self._rng

        # Pick a random trajectory
        traj_idx = self.valid_indices[rng.integers(len(self.valid_indices))]
        traj = self.trajs[traj_idx]
        T = len(traj.trajectory)

        # Pick a horizon k that fits within this trajectory
        feasible_horizons = [k for k in self.horizons if T >= self.min_context + k]
        if not feasible_horizons:
            # Fallback: use k=1 (guaranteed by valid_indices filter)
            feasible_horizons = [1]
        k = int(rng.choice(feasible_horizons))

        # Determine context length bounds
        max_ctx = T - k  # leave room for horizon
        if self.max_context is not None:
            max_ctx = min(max_ctx, self.max_context)
        ctx_len = int(rng.integers(self.min_context, max_ctx + 1))

        # Pick random start position
        latest_start = T - ctx_len - k
        start = int(rng.integers(0, latest_start + 1))

        context_end = start + ctx_len  # exclusive
        target_idx = context_end + k - 1  # 0-indexed position of target

        # Extract arrays
        context = traj.trajectory[start:context_end]                 # (L, D)
        target = traj.trajectory[target_idx]                          # (D,)
        time_ctx = traj.time_seconds[start:context_end]               # (L,)
        time_target = traj.time_seconds[target_idx]

        # Inter-frame deltas within context
        time_deltas = np.diff(time_ctx)                               # (L-1,)
        # Time from last context frame to target
        horizon_dt = time_target - time_ctx[-1]

        class_idx = self.class_to_idx.get(traj.perturbation_class, -1)

        return {
            "context": torch.from_numpy(context).float(),             # (L, D)
            "target": torch.from_numpy(target).float(),               # (D,)
            "time_deltas": torch.from_numpy(time_deltas).float(),     # (L-1,)
            "horizon_dt": torch.tensor(horizon_dt, dtype=torch.float32),
            "delta_t": torch.tensor(traj.delta_t, dtype=torch.float32),
            "temperature": torch.tensor(traj.temperature, dtype=torch.float32),
            "class_idx": torch.tensor(class_idx, dtype=torch.long),
            "embryo_idx": torch.tensor(traj_idx, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

@dataclass
class FragmentBatch:
    """Padded batch of fragments.

    Attributes:
        context: (B, L_max, D) padded context trajectories.
        context_mask: (B, L_max) boolean mask — True for real frames.
        target: (B, D) target vectors.
        time_deltas: (B, L_max-1) padded inter-frame Δt values.
        horizon_dt: (B,) time gap from last context frame to target.
        delta_t: (B,) experiment-level median Δt.
        temperature: (B,) incubation temperatures.
        class_idx: (B,) perturbation class indices.
        embryo_idx: (B,) trajectory indices.
    """
    context: torch.Tensor
    context_mask: torch.Tensor
    target: torch.Tensor
    time_deltas: torch.Tensor
    horizon_dt: torch.Tensor
    delta_t: torch.Tensor
    temperature: torch.Tensor
    class_idx: torch.Tensor
    embryo_idx: torch.Tensor


def fragment_collate_fn(samples: List[Dict[str, torch.Tensor]]) -> FragmentBatch:
    """Collate variable-length fragments into a padded batch."""
    B = len(samples)
    D = samples[0]["context"].shape[-1]
    lengths = [s["context"].shape[0] for s in samples]
    L_max = max(lengths)

    context = torch.zeros(B, L_max, D)
    context_mask = torch.zeros(B, L_max, dtype=torch.bool)
    time_deltas = torch.zeros(B, L_max - 1)

    for i, s in enumerate(samples):
        L = lengths[i]
        context[i, :L] = s["context"]
        context_mask[i, :L] = True
        td = s["time_deltas"]
        time_deltas[i, :len(td)] = td

    return FragmentBatch(
        context=context,
        context_mask=context_mask,
        target=torch.stack([s["target"] for s in samples]),
        time_deltas=time_deltas,
        horizon_dt=torch.stack([s["horizon_dt"] for s in samples]),
        delta_t=torch.stack([s["delta_t"] for s in samples]),
        temperature=torch.stack([s["temperature"] for s in samples]),
        class_idx=torch.stack([s["class_idx"] for s in samples]),
        embryo_idx=torch.stack([s["embryo_idx"] for s in samples]),
    )


def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker's RNG independently."""
    dataset = torch.utils.data.get_worker_info().dataset
    dataset.seed_worker(torch.initial_seed() % (2**32) + worker_id)
