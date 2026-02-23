"""
Streaming loader for ROI FeatureDatasets.

Supports:
- Full-batch (small N) and minibatch (large N) loading
- Deterministic filtering via qc/outlier_flag
- Embryo-grouped CV splits (prevents leakage)
- Fold-local class weights (computed from training fold only)

Follows the same grouped-CV pattern used in
src/analyze/difference_detection/classification_test.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

try:
    import zarr
except ImportError:
    zarr = None

logger = logging.getLogger(__name__)


class FeatureLoader:
    """
    Loader for a validated FeatureDataset.

    Handles QC filtering, CV split generation, and class weighting.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        exclude_outliers: bool = True,
    ):
        self.root = Path(dataset_dir)

        if zarr is None:
            raise ImportError("zarr required for FeatureLoader")

        import json
        with open(self.root / "manifest.json") as f:
            self.manifest = json.load(f)

        self.metadata = pd.read_parquet(self.root / "metadata.parquet")
        self.store = zarr.open(str(self.root / "features.zarr"), mode="r")

        self.N = self.manifest["n_samples"]
        self.grid_hw = tuple(self.manifest["canonical_grid"])
        self.n_channels = self.manifest["n_channels"]
        self.group_key = self.manifest["split_policy"]["group_key"]

        # Deterministic filtering
        outlier_flag = np.array(self.store["qc"]["outlier_flag"])
        if exclude_outliers:
            self._valid_mask = ~outlier_flag
            n_excluded = int(outlier_flag.sum())
            if n_excluded > 0:
                logger.info(
                    f"Excluding {n_excluded}/{self.N} outlier samples "
                    f"(QC IQR filter)"
                )
        else:
            self._valid_mask = np.ones(self.N, dtype=bool)

        self._valid_indices = np.where(self._valid_mask)[0]

    @property
    def n_valid(self) -> int:
        return len(self._valid_indices)

    def load_full(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all valid samples into memory.

        Returns (X, y, groups) where groups are embryo_id values
        for use in GroupKFold.
        """
        idx = self._valid_indices
        X = np.array(self.store["X"])[idx]  # (N_valid, H, W, C)
        y = np.array(self.store["y"])[idx]  # (N_valid,)
        groups = self.metadata.iloc[idx][self.group_key].values
        return X, y, groups

    def load_mask_ref(self) -> np.ndarray:
        """Load the reference mask (512x512)."""
        return np.array(self.store["mask_ref"])

    def get_channel_names(self) -> Tuple[str, ...]:
        """Return channel names from the manifest channel_schema in order."""
        schema = self.manifest.get("channel_schema", [])
        if not schema:
            return tuple(f"channel_{i}" for i in range(self.n_channels))
        return tuple(str(channel["name"]) for channel in schema)

    def iter_batches(
        self,
        batch_size: int = 16,
        shuffle: bool = False,
        random_seed: int = 42,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over valid samples in batches.

        Yields (X_batch, y_batch) tuples.
        """
        idx = self._valid_indices.copy()
        if shuffle:
            rng = np.random.default_rng(random_seed)
            rng.shuffle(idx)

        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start : start + batch_size]
            X_batch = np.array(self.store["X"][batch_idx])
            y_batch = np.array(self.store["y"][batch_idx])
            yield X_batch, y_batch

    def generate_cv_splits(
        self,
        n_folds: int = 5,
    ) -> List[Dict]:
        """
        Generate GroupKFold CV splits grouped by embryo_id.

        Returns a list of dicts with keys:
            fold, train_idx, val_idx, train_groups, val_groups,
            class_weights_train

        Class weights are computed from the TRAINING fold only,
        matching the MorphSeq class_weight='balanced' convention.
        """
        X_all, y_all, groups_all = self.load_full()

        # GroupKFold ensures no embryo appears in both train and val
        gkf = GroupKFold(n_splits=n_folds)
        splits = []

        for fold_i, (train_idx, val_idx) in enumerate(
            gkf.split(X_all, y_all, groups_all)
        ):
            y_train = y_all[train_idx]

            # Compute class weights from TRAINING fold only
            classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_train,
            )
            class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

            splits.append({
                "fold": fold_i,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_groups": groups_all[train_idx],
                "val_groups": groups_all[val_idx],
                "class_weights_train": class_weight_dict,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "train_class_counts": {
                    int(c): int(np.sum(y_train == c)) for c in classes
                },
            })

            logger.info(
                f"Fold {fold_i}: train={len(train_idx)}, val={len(val_idx)}, "
                f"class_weights={class_weight_dict}"
            )

        return splits


__all__ = ["FeatureLoader"]
