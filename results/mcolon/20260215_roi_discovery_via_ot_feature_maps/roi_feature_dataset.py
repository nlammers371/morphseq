"""
FeatureDataset builder + validator for ROI discovery.

Contract-first: Zarr (arrays) + Parquet (metadata) + JSON manifest.
Follows the Parquet storage patterns from
src/analyze/optimal_transport_morphometrics/uot_masks/feature_compaction/storage.py.

Usage:
    # Build from OT results
    builder = FeatureDatasetBuilder(
        ot_results_dir="path/to/ot_outputs",
        out_dir="roi_feature_dataset_cep290",
        feature_set=FeatureSet.COST,
    )
    builder.build()

    # Validate an existing dataset
    validate_feature_dataset("roi_feature_dataset_cep290")
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import zarr
except ImportError:
    zarr = None
try:
    from numcodecs import Blosc
except Exception:
    Blosc = None

from roi_config import (
    CHANNEL_SCHEMAS,
    PHASE0_CHANNEL_SCHEMAS,
    ChannelSchema,
    FeatureDatasetConfig,
    FeatureSet,
    Phase0FeatureSet,
)

logger = logging.getLogger(__name__)

# Schema version for forward-compatibility checks
SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _build_manifest(
    config: FeatureDatasetConfig,
    feature_set: FeatureSet,
    channel_schemas: List[ChannelSchema],
    n_samples: int,
    n_channels: int,
    class_counts: Dict[int, int],
    provenance: Optional[Dict] = None,
) -> Dict:
    """Build the manifest.json dict (hard validation fields included)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_grid": list(config.canonical_grid_hw),
        "feature_set": feature_set.value,
        "channel_schema": [
            {"name": cs.name, "definition": cs.definition, "units": cs.units}
            for cs in channel_schemas
        ],
        "n_samples": n_samples,
        "n_channels": n_channels,
        "class_counts": class_counts,
        "qc_rules": {
            "iqr_multiplier": config.iqr_multiplier,
            "outlier_field": "qc/outlier_flag",
            "cost_field": "qc/total_cost_C",
            "policy": "log_and_flag_never_delete",
        },
        "split_policy": {
            "group_key": config.group_key,
            "strategy": "GroupKFold_by_embryo_id",
            "note": "MANDATORY: prevents leakage across folds",
        },
        "class_balance_strategy": {
            "method": "sklearn_balanced_class_weight",
            "scope": "train_fold_only",
            "note": "Matches existing MorphSeq classification code (class_weight='balanced')",
        },
        "chunking": {
            "X": [config.chunk_size_n, *config.canonical_grid_hw, n_channels],
            "compression": config.compression,
            "compression_level": config.compression_level,
        },
        "provenance": provenance or {},
    }


def _build_phase0_manifest(
    config: FeatureDatasetConfig,
    feature_set: Phase0FeatureSet,
    channel_schemas: List[ChannelSchema],
    n_samples: int,
    n_channels: int,
    class_counts: Dict[int, int],
    stage_window: tuple,
    ot_params_hash: str = "",
    reference_mask_id: str = "",
    provenance: Optional[Dict] = None,
) -> Dict:
    """Build manifest.json for Phase 0 FeatureDataset (adds Phase 0-specific fields)."""
    manifest = _build_manifest(
        config=config,
        feature_set=FeatureSet.COST,  # base schema check
        channel_schemas=channel_schemas,
        n_samples=n_samples,
        n_channels=n_channels,
        class_counts=class_counts,
        provenance=provenance,
    )
    # Override / add Phase 0-specific fields
    manifest["phase"] = 0
    manifest["feature_set"] = feature_set.value
    manifest["stage_window"] = list(stage_window)
    manifest["OT_params_hash"] = ot_params_hash
    manifest["reference_mask_id"] = reference_mask_id
    manifest["smoothing_policy"] = "visualization-only; stats computed on unsmoothed features"
    manifest["S_orientation"] = "S=0 head, S=1 tail"
    return manifest


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidationError(Exception):
    """Raised when a FeatureDataset fails schema validation."""
    pass


def validate_feature_dataset(dataset_dir: str | Path) -> Dict:
    """
    Validate a FeatureDataset directory against the contract.

    Performs fail-fast schema checks with informative errors.

    Returns the parsed manifest on success.
    Raises DatasetValidationError with details on failure.
    """
    root = Path(dataset_dir)
    errors = []

    # 1) Check directory exists
    if not root.is_dir():
        raise DatasetValidationError(f"Dataset directory does not exist: {root}")

    # 2) manifest.json
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise DatasetValidationError(f"Missing manifest.json in {root}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    required_keys = [
        "schema_version", "canonical_grid", "feature_set", "channel_schema",
        "n_samples", "n_channels", "qc_rules", "split_policy",
        "class_balance_strategy", "chunking",
    ]
    for key in required_keys:
        if key not in manifest:
            errors.append(f"manifest.json missing required key: '{key}'")

    if errors:
        raise DatasetValidationError(
            f"Manifest validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # 3) metadata.parquet
    meta_path = root / "metadata.parquet"
    if not meta_path.exists():
        errors.append("Missing metadata.parquet")
    else:
        df_meta = pd.read_parquet(meta_path)
        group_key = manifest["split_policy"]["group_key"]
        if group_key not in df_meta.columns:
            errors.append(
                f"metadata.parquet missing required group_key column: '{group_key}'"
            )
        required_meta_cols = [group_key, "genotype", "sample_index"]
        for col in required_meta_cols:
            if col not in df_meta.columns:
                errors.append(f"metadata.parquet missing column: '{col}'")

    # 4) features.zarr
    zarr_path = root / "features.zarr"
    if not zarr_path.exists():
        errors.append("Missing features.zarr directory")
    elif zarr is not None:
        try:
            store = zarr.open(str(zarr_path), mode="r")
        except Exception as e:
            errors.append(f"Cannot open features.zarr: {e}")
        else:
            expected_arrays = ["X", "y", "mask_ref"]
            for name in expected_arrays:
                if name not in store:
                    errors.append(f"features.zarr missing array: '{name}'")

            if "X" in store:
                X = store["X"]
                n_samples = manifest["n_samples"]
                n_channels = manifest["n_channels"]
                grid = tuple(manifest["canonical_grid"])
                expected_shape = (n_samples, *grid, n_channels)
                if X.shape != expected_shape:
                    errors.append(
                        f"X shape mismatch: got {X.shape}, expected {expected_shape}"
                    )
                if X.dtype != np.float32:
                    errors.append(f"X dtype should be float32, got {X.dtype}")

            if "y" in store:
                y = store["y"]
                if y.shape != (manifest["n_samples"],):
                    errors.append(
                        f"y shape mismatch: got {y.shape}, expected ({manifest['n_samples']},)"
                    )

            if "mask_ref" in store:
                mask = store["mask_ref"]
                grid = tuple(manifest["canonical_grid"])
                if mask.shape != grid:
                    errors.append(
                        f"mask_ref shape mismatch: got {mask.shape}, expected {grid}"
                    )

            # QC arrays
            if "qc" not in store:
                errors.append("features.zarr missing 'qc' group")
            else:
                qc = store["qc"]
                for qc_name in ["total_cost_C", "outlier_flag"]:
                    if qc_name not in qc:
                        errors.append(f"features.zarr/qc missing: '{qc_name}'")

    if errors:
        raise DatasetValidationError(
            f"Dataset validation failed ({len(errors)} errors):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(f"Dataset validated: {root} ({manifest['n_samples']} samples)")
    return manifest


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class FeatureDatasetBuilder:
    """
    Build a FeatureDataset from OT results.

    This builder collects OT-derived feature maps (cost, displacement,
    mass creation/destruction) already rasterized to the 512x512 canonical
    grid and packages them into the contract format.

    The actual OT computation is NOT done here — we consume outputs from
    the existing UOT pipeline (run_transport.py / transport_maps.py).
    """

    def __init__(
        self,
        ot_results_dir: str | Path | None = None,
        out_dir: str | Path | None = None,
        feature_set: FeatureSet = FeatureSet.COST,
        config: Optional[FeatureDatasetConfig] = None,
        genotype_col: str = "genotype",
        reference_genotype: str = "WT",
        target_genotype: str = "cep290",
    ):
        self.ot_results_dir = Path(ot_results_dir) if ot_results_dir is not None else None
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.feature_set = feature_set
        self.config = config or FeatureDatasetConfig()
        self.genotype_col = genotype_col
        self.reference_genotype = reference_genotype
        self.target_genotype = target_genotype
        self.channel_schemas = CHANNEL_SCHEMAS[feature_set]

    def build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask_ref: np.ndarray,
        metadata_df: pd.DataFrame = None,
        total_cost_C: np.ndarray = None,
        provenance: Optional[Dict] = None,
        out_dir: str | Path | None = None,
        metadata: pd.DataFrame = None,
    ) -> Path:
        """
        Build and write the FeatureDataset to disk.

        Parameters
        ----------
        X : ndarray, shape (N, 512, 512, C)
            Feature maps on the canonical grid.
        y : ndarray, shape (N,)
            Binary labels (0 = reference, 1 = target genotype).
        mask_ref : ndarray, shape (512, 512)
            Reference embryo mask on canonical grid.
        metadata_df : DataFrame
            Per-sample metadata. Must contain 'embryo_id' and 'genotype'.
        total_cost_C : ndarray, shape (N,)
            Total OT cost per sample (for QC filtering).
        provenance : dict, optional
            Provenance info (source paths, git hash, etc).

        Returns
        -------
        Path to the created dataset directory.
        """
        if zarr is None:
            raise ImportError("zarr is required to build FeatureDataset. pip install zarr")

        # Allow out_dir override from build() call
        if out_dir is not None:
            self.out_dir = Path(out_dir)
        if self.out_dir is None:
            raise ValueError("out_dir must be provided either in __init__ or build()")

        # Allow 'metadata' as alias for 'metadata_df'
        if metadata_df is None and metadata is not None:
            metadata_df = metadata
        if metadata_df is None:
            raise ValueError("metadata_df (or metadata=) is required")

        N, H, W, C = X.shape
        grid = self.config.canonical_grid_hw

        # Validate shapes
        assert (H, W) == grid, f"X spatial dims {(H, W)} != canonical_grid {grid}"
        assert y.shape == (N,), f"y shape {y.shape} != ({N},)"
        assert mask_ref.shape == grid, f"mask_ref shape {mask_ref.shape} != {grid}"
        assert total_cost_C.shape == (N,), f"total_cost_C shape != ({N},)"
        assert len(metadata_df) == N, f"metadata length {len(metadata_df)} != {N}"
        assert C == len(self.channel_schemas), (
            f"X has {C} channels but channel_schema has {len(self.channel_schemas)}"
        )

        # QC: IQR outlier detection on total_cost_C
        q1 = np.percentile(total_cost_C, 25)
        q3 = np.percentile(total_cost_C, 75)
        iqr = q3 - q1
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr
        outlier_flag = (total_cost_C < lower) | (total_cost_C > upper)
        n_outliers = int(outlier_flag.sum())
        logger.info(
            f"QC: {n_outliers}/{N} samples flagged as outliers "
            f"(IQR={iqr:.4f}, bounds=[{lower:.4f}, {upper:.4f}])"
        )

        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Write Zarr
        zarr_path = self.out_dir / "features.zarr"
        store = zarr.open(str(zarr_path), mode="w")

        chunk_n = self.config.chunk_size_n
        compressor = None
        if self.config.compression:
            if Blosc is None:
                raise ImportError("numcodecs is required for zarr compression")
            if self.config.compression.lower() != "zstd":
                raise ValueError(f"Unsupported compression {self.config.compression!r}; expected 'zstd'")
            compressor = Blosc(cname="zstd", clevel=int(self.config.compression_level), shuffle=Blosc.BITSHUFFLE)

        store.create_dataset(
            "X", data=X.astype(np.float32),
            chunks=(chunk_n, H, W, C),
            dtype=np.float32,
            compressor=compressor,
        )
        store.create_dataset("y", data=y.astype(np.int32), dtype=np.int32, compressor=compressor)
        store.create_dataset(
            "mask_ref", data=mask_ref.astype(np.uint8), dtype=np.uint8,
            compressor=compressor,
        )

        qc_group = store.create_group("qc")
        qc_group.create_dataset(
            "total_cost_C", data=total_cost_C.astype(np.float32), dtype=np.float32,
            compressor=compressor,
        )
        qc_group.create_dataset(
            "outlier_flag", data=outlier_flag, dtype=bool,
            compressor=compressor,
        )

        # Write metadata Parquet
        metadata_df = metadata_df.copy()
        metadata_df["sample_index"] = np.arange(N)
        metadata_df.to_parquet(self.out_dir / "metadata.parquet", index=False)

        # Class counts
        unique, counts = np.unique(y, return_counts=True)
        class_counts = {int(u): int(c) for u, c in zip(unique, counts)}

        # Write manifest
        manifest = _build_manifest(
            config=self.config,
            feature_set=self.feature_set,
            channel_schemas=self.channel_schemas,
            n_samples=N,
            n_channels=C,
            class_counts=class_counts,
            provenance=provenance,
        )
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # Validate what we just wrote
        validate_feature_dataset(self.out_dir)

        logger.info(f"FeatureDataset built: {self.out_dir} ({N} samples, {C} channels)")
        return self.out_dir


class Phase0FeatureDatasetBuilder:
    """
    Build a Phase 0 FeatureDataset with optional S_map_ref and basis arrays.

    Phase 0 adds:
    - stage_window constraint (single 2 hpf bin)
    - S_map_ref in optional/ group
    - tangent_ref, normal_ref in optional/ group
    - Phase 0-specific manifest fields
    """

    def __init__(
        self,
        out_dir: str | Path,
        feature_set: Phase0FeatureSet = Phase0FeatureSet.V0_COST,
        config: Optional[FeatureDatasetConfig] = None,
        stage_window: tuple = (0.0, 2.0),
        reference_mask_id: str = "",
        ot_params_hash: str = "",
    ):
        self.out_dir = Path(out_dir)
        self.feature_set = feature_set
        self.config = config or FeatureDatasetConfig()
        self.stage_window = stage_window
        self.reference_mask_id = reference_mask_id
        self.ot_params_hash = ot_params_hash
        self.channel_schemas = PHASE0_CHANNEL_SCHEMAS[feature_set]

    def build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask_ref: np.ndarray,
        metadata_df: pd.DataFrame,
        total_cost_C: np.ndarray,
        target_masks_canonical: Optional[np.ndarray] = None,
        alignment_debug_df: Optional[pd.DataFrame] = None,
        S_map_ref: Optional[np.ndarray] = None,
        tangent_ref: Optional[np.ndarray] = None,
        normal_ref: Optional[np.ndarray] = None,
        provenance: Optional[Dict] = None,
    ) -> Path:
        """
        Build and write Phase 0 FeatureDataset to disk.

        Parameters
        ----------
        X : ndarray, shape (N, 512, 512, C)
        y : ndarray, shape (N,)
        mask_ref : ndarray, shape (512, 512)
        metadata_df : DataFrame with embryo_id, snip_id, label_int, etc.
        total_cost_C : ndarray, shape (N,)
        target_masks_canonical : ndarray, shape (N, 512, 512), optional
            OT-aligned target masks on canonical grid (exact masks used by OT preprocessing).
        alignment_debug_df : DataFrame, optional
            Per-sample alignment diagnostics with columns including:
            sample_id, source_id, target_id, rotation/flip/retained ratio fields.
        S_map_ref : ndarray, shape (512, 512), optional, S in [0,1]
        tangent_ref : ndarray, shape (512, 512, 2), optional
        normal_ref : ndarray, shape (512, 512, 2), optional
        provenance : dict, optional
        """
        if zarr is None:
            raise ImportError("zarr required")

        N, H, W, C = X.shape
        grid = self.config.canonical_grid_hw
        assert (H, W) == grid
        assert y.shape == (N,)
        assert mask_ref.shape == grid
        assert total_cost_C.shape == (N,)
        if target_masks_canonical is not None:
            assert target_masks_canonical.shape == (N, *grid)
        assert len(metadata_df) == N
        assert C == len(self.channel_schemas), (
            f"X has {C} channels, schema expects {len(self.channel_schemas)}"
        )

        # QC: IQR outlier detection
        q1 = np.percentile(total_cost_C, 25)
        q3 = np.percentile(total_cost_C, 75)
        iqr = q3 - q1
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr
        outlier_flag = (total_cost_C < lower) | (total_cost_C > upper)
        n_outliers = int(outlier_flag.sum())
        logger.info(
            f"QC: {n_outliers}/{N} outliers (IQR={iqr:.4f}, bounds=[{lower:.4f}, {upper:.4f}])"
        )

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Write Zarr
        zarr_path = self.out_dir / "features.zarr"
        store = zarr.open(str(zarr_path), mode="w")

        chunk_n = self.config.chunk_size_n
        compressor = None
        if self.config.compression:
            if Blosc is None:
                raise ImportError("numcodecs is required for zarr compression")
            if self.config.compression.lower() != "zstd":
                raise ValueError(f"Unsupported compression {self.config.compression!r}; expected 'zstd'")
            compressor = Blosc(cname="zstd", clevel=int(self.config.compression_level), shuffle=Blosc.BITSHUFFLE)
        store.create_dataset("X", data=X.astype(np.float32),
                             chunks=(chunk_n, H, W, C), dtype=np.float32, compressor=compressor)
        store.create_dataset("y", data=y.astype(np.int32), dtype=np.int32, compressor=compressor)
        store.create_dataset("mask_ref", data=mask_ref.astype(np.uint8), dtype=np.uint8, compressor=compressor)

        qc_group = store.create_group("qc")
        qc_group.create_dataset("total_cost_C", data=total_cost_C.astype(np.float32), compressor=compressor)
        qc_group.create_dataset("outlier_flag", data=outlier_flag, dtype=bool, compressor=compressor)

        # Optional arrays (Phase 0 specific)
        opt_group = store.create_group("optional")
        if target_masks_canonical is not None:
            opt_group.create_dataset(
                "target_masks_canonical",
                data=target_masks_canonical.astype(np.uint8),
                chunks=(chunk_n, H, W),
                dtype=np.uint8,
                compressor=compressor,
            )
        if S_map_ref is not None:
            assert S_map_ref.shape == grid
            opt_group.create_dataset("S_map_ref", data=S_map_ref.astype(np.float32), compressor=compressor)
        if tangent_ref is not None:
            assert tangent_ref.shape == (*grid, 2)
            opt_group.create_dataset("tangent_ref", data=tangent_ref.astype(np.float32), compressor=compressor)
        if normal_ref is not None:
            assert normal_ref.shape == (*grid, 2)
            opt_group.create_dataset("normal_ref", data=normal_ref.astype(np.float32), compressor=compressor)

        # Write metadata
        metadata_df = metadata_df.copy()
        metadata_df["sample_index"] = np.arange(N)
        metadata_df["total_cost_C"] = total_cost_C
        metadata_df["qc_outlier_flag"] = outlier_flag

        if alignment_debug_df is not None:
            align_df = alignment_debug_df.copy()
            if "sample_id" not in align_df.columns:
                raise ValueError("alignment_debug_df must include 'sample_id'")
            if "sample_id" not in metadata_df.columns:
                raise ValueError("metadata_df must include 'sample_id' when alignment_debug_df is provided")

            sample_to_index = {sid: i for i, sid in enumerate(metadata_df["sample_id"].astype(str).tolist())}
            align_df["sample_id"] = align_df["sample_id"].astype(str)
            align_df["sample_index"] = align_df["sample_id"].map(sample_to_index)

            missing_samples = align_df["sample_index"].isna()
            if missing_samples.any():
                missing_ids = align_df.loc[missing_samples, "sample_id"].tolist()
                raise ValueError(
                    "alignment_debug_df has sample_id values not present in metadata_df: "
                    f"{missing_ids[:5]}"
                )

            align_df["sample_index"] = align_df["sample_index"].astype(np.int32)
            align_df = align_df.sort_values("sample_index").reset_index(drop=True)
            align_df.to_parquet(self.out_dir / "alignment_debug.parquet", index=False)

            if "source_id" in align_df.columns and "source_id" not in metadata_df.columns:
                metadata_df = metadata_df.merge(
                    align_df[["sample_id", "source_id"]],
                    on="sample_id",
                    how="left",
                )
            if "target_id" in align_df.columns and "target_id" not in metadata_df.columns:
                metadata_df = metadata_df.merge(
                    align_df[["sample_id", "target_id"]],
                    on="sample_id",
                    how="left",
                )

        metadata_df.to_parquet(self.out_dir / "metadata.parquet", index=False)

        # Class counts
        unique, counts = np.unique(y, return_counts=True)
        class_counts = {int(u): int(c) for u, c in zip(unique, counts)}

        # Write manifest
        manifest = _build_phase0_manifest(
            config=self.config,
            feature_set=self.feature_set,
            channel_schemas=self.channel_schemas,
            n_samples=N,
            n_channels=C,
            class_counts=class_counts,
            stage_window=self.stage_window,
            ot_params_hash=self.ot_params_hash,
            reference_mask_id=self.reference_mask_id,
            provenance=provenance,
        )
        manifest["debug_outputs"] = {
            "alignment_debug_parquet": bool(alignment_debug_df is not None),
            "source_target_ids_in_metadata": bool(
                ("source_id" in metadata_df.columns) and ("target_id" in metadata_df.columns)
            ),
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        validate_feature_dataset(self.out_dir)
        logger.info(f"Phase 0 FeatureDataset built: {self.out_dir} ({N} samples, {C} ch)")
        return self.out_dir


__all__ = [
    "FeatureDatasetBuilder",
    "Phase0FeatureDatasetBuilder",
    "validate_feature_dataset",
    "DatasetValidationError",
    "SCHEMA_VERSION",
]
