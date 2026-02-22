"""
io.phase0 — Ergonomic loader for Phase 0 feature datasets.

Single source of truth for zarr keys and sidecar file locations.
Provides lazy zarr handles (no RAM) and explicit eager numpy loading.

Usage
-----
    from io.phase0 import Phase0Loader, MissingArtifactError

    loader = Phase0Loader(run_dir)
    X = loader.get_X()
    S = loader.require("S_map")       # raises MissingArtifactError if absent
    debug = loader.alignment_debug    # pd.DataFrame or None
    print(loader)                     # Phase0Loader(run=..., n=20, artifacts=[X, y, S_map])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import zarr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional artifact registry
# ---------------------------------------------------------------------------

# Maps artifact key → (kind, path-relative-to-dataset-dir)
# kind: "zarr" means z[path], "file" means dataset_dir/path
OPTIONAL: dict[str, tuple[str, str]] = {
    "S_map":            ("zarr", "optional/S_map_ref"),
    "target_masks":     ("zarr", "optional/target_masks_canonical"),
    "tangent_ref":      ("zarr", "optional/tangent_ref"),
    "normal_ref":       ("zarr", "optional/normal_ref"),
    "alignment_debug":  ("file", "alignment_debug.parquet"),
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class MissingArtifactError(FileNotFoundError):
    def __init__(self, key: str, location: str, run_dir: Path):
        super().__init__(
            f"Artifact '{key}' not found at {location} in {run_dir}. "
            f"This run may predate '{key}' generation."
        )
        self.key = key
        self.location = location
        self.run_dir = run_dir


# ---------------------------------------------------------------------------
# Phase0Loader
# ---------------------------------------------------------------------------

class Phase0Loader:
    """
    Lazy-loading interface to a Phase 0 feature_dataset directory.

    Parameters
    ----------
    run_dir : path to the run directory (contains feature_dataset/ sub-dir)
    dataset_subdir : name of the feature dataset sub-directory (default "feature_dataset")
    """

    def __init__(self, run_dir: str | Path, dataset_subdir: str = "feature_dataset"):
        self._run_dir = Path(run_dir)
        self._dataset_dir = self._run_dir / dataset_subdir

        zarr_path = self._dataset_dir / "features.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(
                f"feature_dataset zarr not found: {zarr_path}"
            )
        self._z = zarr.open(str(zarr_path), "r")

        # Validate required keys
        for key in ("X", "y", "mask_ref", "qc/total_cost_C"):
            if key not in self._z:
                raise KeyError(f"Required zarr key '{key}' missing in {zarr_path}")

        # Lazy handles (no data loaded yet)
        self.X_zarr = self._z["X"]
        self.y_zarr = self._z["y"]
        self.mask_ref_zarr = self._z["mask_ref"]
        self.total_cost_C_zarr = self._z["qc/total_cost_C"]

        # Metadata (always loaded — it's small)
        meta_path = self._dataset_dir / "metadata.parquet"
        if meta_path.exists():
            self._metadata_df: Optional[pd.DataFrame] = pd.read_parquet(meta_path)
        else:
            self._metadata_df = None

    # ------------------------------------------------------------------
    # Eager numpy loaders
    # ------------------------------------------------------------------

    def get_X(self, indices=None) -> np.ndarray:
        """Load X (N, H, W, C) or a subset by indices."""
        if indices is None:
            return self.X_zarr[:]
        return self.X_zarr[indices]

    def get_y(self) -> np.ndarray:
        """Load label vector (N,)."""
        return self.y_zarr[:]

    def get_mask_ref(self) -> np.ndarray:
        """Load reference mask (H, W)."""
        return self.mask_ref_zarr[:]

    def get_total_cost_C(self) -> np.ndarray:
        """Load total OT cost vector (N,)."""
        return self.total_cost_C_zarr[:]

    @property
    def sample_ids(self) -> List[str]:
        """Sample IDs from metadata."""
        if self._metadata_df is not None and "sample_id" in self._metadata_df.columns:
            return self._metadata_df["sample_id"].tolist()
        N = self.X_zarr.shape[0]
        return [f"s_{i}" for i in range(N)]

    @property
    def metadata_df(self) -> Optional[pd.DataFrame]:
        return self._metadata_df

    # ------------------------------------------------------------------
    # Optional artifact properties
    # ------------------------------------------------------------------

    @property
    def S_map(self) -> Optional[np.ndarray]:
        """S coordinate map (H, W) or None."""
        return self._load_optional("S_map")

    @property
    def target_masks(self) -> Optional[np.ndarray]:
        """Canonical target masks (N, H, W) or None."""
        return self._load_optional("target_masks")

    @property
    def tangent_ref(self) -> Optional[np.ndarray]:
        """Tangent field on reference (H, W, 2) or None."""
        return self._load_optional("tangent_ref")

    @property
    def normal_ref(self) -> Optional[np.ndarray]:
        """Normal field on reference (H, W, 2) or None."""
        return self._load_optional("normal_ref")

    @property
    def alignment_debug(self) -> Optional[pd.DataFrame]:
        """Alignment debug DataFrame or None."""
        return self._load_optional("alignment_debug")

    # ------------------------------------------------------------------
    # Ergonomics: has() / require()
    # ------------------------------------------------------------------

    def has(self, key: str) -> bool:
        """Return True if the artifact exists (zarr key or sidecar file)."""
        if key not in OPTIONAL:
            return False
        kind, loc = OPTIONAL[key]
        if kind == "zarr":
            return loc in self._z
        else:  # "file"
            return (self._dataset_dir / loc).exists()

    def require(self, key: str):
        """
        Return the loaded artifact or raise MissingArtifactError.

        Returns np.ndarray for zarr artifacts, pd.DataFrame for file artifacts.
        """
        if key not in OPTIONAL:
            raise KeyError(f"Unknown artifact key '{key}'. Known keys: {list(OPTIONAL)}")
        if not self.has(key):
            kind, loc = OPTIONAL[key]
            location = f"{kind}:{loc}"
            raise MissingArtifactError(key, location, self._run_dir)
        return self._load_optional(key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_optional(self, key: str):
        if key not in OPTIONAL:
            return None
        kind, loc = OPTIONAL[key]
        if kind == "zarr":
            if loc not in self._z:
                return None
            return self._z[loc][:]
        else:  # "file"
            path = self._dataset_dir / loc
            if not path.exists():
                return None
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            raise NotImplementedError(f"Unsupported file type: {path.suffix}")

    def __repr__(self) -> str:
        N = self.X_zarr.shape[0]
        present = [k for k in OPTIONAL if self.has(k)]
        required = ["X", "y", "mask_ref", "total_cost_C"]
        artifacts = required + present
        return (
            f"Phase0Loader(run={self._run_dir.name!r}, "
            f"n={N}, artifacts={artifacts})"
        )


__all__ = ["Phase0Loader", "MissingArtifactError", "OPTIONAL"]
