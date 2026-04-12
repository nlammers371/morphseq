"""Raw null distribution storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NullDistributions:
    """Stores raw null AUROC arrays indexed by (feature_set, comparison_id, time_bin_center).

    Parameters
    ----------
    null_auc : np.ndarray
        Shape ``(N, P)`` float32 — one row per (feature_set, comparison, time_bin),
        *P* permutation AUROCs per row.
    feature_set : np.ndarray
        Shape ``(N,)`` str.
    comparison_id : np.ndarray
        Shape ``(N,)`` str.
    time_bin_center : np.ndarray
        Shape ``(N,)`` float64.
    """

    null_auc: np.ndarray
    feature_set: np.ndarray
    comparison_id: np.ndarray
    time_bin_center: np.ndarray
    _index: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = self.null_auc.shape[0]
        for name, arr in [
            ("feature_set", self.feature_set),
            ("comparison_id", self.comparison_id),
            ("time_bin_center", self.time_bin_center),
        ]:
            if arr.shape[0] != n:
                raise ValueError(
                    f"{name} length {arr.shape[0]} != null_auc rows {n}"
                )

    def _build_index(self) -> dict[tuple[str, str, float], int]:
        idx = {}
        for i in range(self.null_auc.shape[0]):
            key = (
                str(self.feature_set[i]),
                str(self.comparison_id[i]),
                float(self.time_bin_center[i]),
            )
            idx[key] = i
        return idx

    def get(
        self,
        feature_set: str,
        comparison_id: str,
        time_bin_center: float,
    ) -> np.ndarray:
        """Return the 1-D null array for the given key."""
        if self._index is None:
            object.__setattr__(self, "_index", self._build_index())
        key = (feature_set, comparison_id, time_bin_center)
        idx = self._index.get(key)  # type: ignore[union-attr]
        if idx is None:
            raise KeyError(
                f"No null distribution for {key}. "
                f"Available keys: {len(self._index)}"  # type: ignore[arg-type]
            )
        return self.null_auc[idx]

    def save(self, path: Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            null_auc=self.null_auc.astype(np.float32),
            feature_set=self.feature_set,
            comparison_id=self.comparison_id,
            time_bin_center=self.time_bin_center,
        )

    @classmethod
    def load(cls, path: Path) -> NullDistributions:
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        return cls(
            null_auc=data["null_auc"],
            feature_set=data["feature_set"],
            comparison_id=data["comparison_id"],
            time_bin_center=data["time_bin_center"],
        )
