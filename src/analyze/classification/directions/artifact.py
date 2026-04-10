"""ClassifierDirections dataclass — the persisted artifact produced by direction fitting.

This module owns the definition, save, and load of ClassifierDirections.
It is the single artifact contract between:
  - the producing subsystem (classification/directions/)
  - the consuming subsystem (morphology_geometry/)

Downstream code that only needs to *read* direction vectors should import
ClassifierDirections from here, not from engine/analysis.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ClassifierDirections:
    """Binary classifier direction metadata plus NPZ-backed unit vectors.

    Attributes
    ----------
    metadata : pd.DataFrame
        One row per (feature_set, comparison_id, time_bin_center).
        Required columns: vector_id, feature_set, comparison_id, positive_label,
        negative_label, time_bin_center, n_pos, n_neg, coef_norm, intercept,
        sign_flipped, centroid_dot, direction_space, preprocess_fingerprint.
        Optional: auroc_obs (present from run_classification, NaN from extract).
    vectors : dict[str, np.ndarray]
        Maps vector_id -> unit coefficient vector (float64, 1-D, unit-norm).
    feature_names : dict[str, list[str]]
        Maps feature_set -> ordered list of feature column names.
        This is the authoritative column order for projection math.
    """

    metadata: pd.DataFrame
    vectors: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]

    def __post_init__(self) -> None:
        required = {"feature_set", "vector_id"}
        missing = required.difference(self.metadata.columns)
        if missing:
            raise ValueError(
                f"ClassifierDirections metadata missing required columns: {sorted(missing)}"
            )
        for _, row in self.metadata.iterrows():
            feature_set = str(row["feature_set"])
            vector_id = str(row["vector_id"])
            if vector_id not in self.vectors:
                raise ValueError(f"Missing classifier direction vector {vector_id!r}")
            if feature_set not in self.feature_names:
                raise ValueError(
                    f"Missing classifier direction feature names for {feature_set!r}"
                )
            if len(self.vectors[vector_id]) != len(self.feature_names[feature_set]):
                raise ValueError(
                    f"Vector {vector_id!r} length does not match feature names for "
                    f"{feature_set!r}"
                )

    def save(self, path: Path) -> None:
        """Write vectors and feature_names to an NPZ file at *path*."""
        payload: dict[str, np.ndarray] = {
            vector_id: np.asarray(vector, dtype=np.float64)
            for vector_id, vector in self.vectors.items()
        }
        for feature_set, names in self.feature_names.items():
            payload[f"feature_names__{feature_set}"] = np.asarray(names, dtype=str)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, metadata_path: Path, vectors_path: Path) -> "ClassifierDirections":
        """Load from the two artifact files (parquet + npz)."""
        metadata = pd.read_parquet(metadata_path)
        vectors: dict[str, np.ndarray] = {}
        feature_names: dict[str, list[str]] = {}
        with np.load(vectors_path, allow_pickle=True) as data:
            for key in data.files:
                arr = data[key]
                if key.startswith("feature_names__"):
                    feature_set = key.removeprefix("feature_names__")
                    feature_names[feature_set] = [str(x) for x in arr.tolist()]
                else:
                    vectors[key] = np.asarray(arr, dtype=np.float64)
        return cls(metadata=metadata, vectors=vectors, feature_names=feature_names)

    @classmethod
    def load_from_dir(cls, save_dir: Path | str) -> "ClassifierDirections":
        """Convenience loader: load from a directory that contains the standard filenames."""
        save_dir = Path(save_dir)
        return cls.load(
            save_dir / "classifier_directions.parquet",
            save_dir / "classifier_directions_vectors.npz",
        )
