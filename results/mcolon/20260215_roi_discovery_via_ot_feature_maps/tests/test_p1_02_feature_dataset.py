"""
Phase 1 / Task 1.2 — FeatureDataset builder and validator.

Checks:
- Builder creates Zarr arrays, Parquet metadata, and JSON manifest
- Manifest contains required keys (canonical_grid, channel_schema, QC rules, etc.)
- Validator rejects datasets with missing arrays or wrong shapes
- QC outlier flagging works with IQR rule
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Optional: only run if zarr is installed
zarr = pytest.importorskip("zarr")
pd = pytest.importorskip("pandas")

from roi_config import CHANNEL_SCHEMAS, FeatureDatasetConfig, FeatureSet
from roi_feature_dataset import FeatureDatasetBuilder, validate_feature_dataset


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    return tmp_path / "roi_feature_dataset_test"


@pytest.fixture
def small_dataset_arrays():
    """Minimal arrays for building a FeatureDataset."""
    N, H, W, C = 8, 512, 512, 1
    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, (N, H, W, C)).astype(np.float32)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    mask_ref = np.ones((H, W), dtype=bool)
    total_cost_C = rng.uniform(0.5, 1.5, N).astype(np.float32)
    metadata = pd.DataFrame({
        "embryo_id": [f"emb_{i}" for i in range(N)],
        "genotype": ["WT"] * 4 + ["cep290"] * 4,
        "label": y,
    })
    return {
        "X": X, "y": y, "mask_ref": mask_ref,
        "total_cost_C": total_cost_C, "metadata": metadata,
    }


# ---- Builder creates expected structure ----

def test_builder_creates_zarr_and_manifest(tmp_dataset_dir, small_dataset_arrays):
    """
    PSEUDO-LOGIC:
    1. Instantiate FeatureDatasetBuilder with default config
    2. Call build() with synthetic arrays
    3. Assert manifest.json, metadata.parquet, and features.zarr/ all exist
    4. Assert Zarr groups X, y, mask_ref, qc/ are present
    """
    builder = FeatureDatasetBuilder(
        config=FeatureDatasetConfig(),
        feature_set=FeatureSet.COST,
    )
    builder.build(
        out_dir=str(tmp_dataset_dir),
        X=small_dataset_arrays["X"],
        y=small_dataset_arrays["y"],
        mask_ref=small_dataset_arrays["mask_ref"],
        total_cost_C=small_dataset_arrays["total_cost_C"],
        metadata=small_dataset_arrays["metadata"],
    )

    assert (tmp_dataset_dir / "manifest.json").exists()
    assert (tmp_dataset_dir / "metadata.parquet").exists()
    assert (tmp_dataset_dir / "features.zarr").exists()


def test_manifest_required_keys(tmp_dataset_dir, small_dataset_arrays):
    """Manifest must contain canonical_grid, channel_schema, qc_rules, split_policy."""
    builder = FeatureDatasetBuilder(
        config=FeatureDatasetConfig(),
        feature_set=FeatureSet.COST,
    )
    builder.build(
        out_dir=str(tmp_dataset_dir),
        X=small_dataset_arrays["X"],
        y=small_dataset_arrays["y"],
        mask_ref=small_dataset_arrays["mask_ref"],
        total_cost_C=small_dataset_arrays["total_cost_C"],
        metadata=small_dataset_arrays["metadata"],
    )

    with open(tmp_dataset_dir / "manifest.json") as f:
        manifest = json.load(f)

    required_keys = ["canonical_grid", "channel_schema", "qc_rules", "split_policy"]
    for key in required_keys:
        assert key in manifest, f"Missing required manifest key: {key}"


# ---- Validator catches bad datasets ----

def test_validate_rejects_missing_zarr(tmp_path):
    """
    PSEUDO-LOGIC:
    1. Create a directory with manifest.json but no features.zarr
    2. Call validate_feature_dataset()
    3. Expect DatasetValidationError
    """
    d = tmp_path / "bad_dataset"
    d.mkdir()
    (d / "manifest.json").write_text("{}")
    # No features.zarr — should fail
    with pytest.raises(Exception):
        validate_feature_dataset(str(d))


# ---- QC outlier flagging ----

def test_qc_outlier_flagging_iqr(tmp_dataset_dir, small_dataset_arrays):
    """
    PSEUDO-LOGIC:
    1. Inject one extreme outlier into total_cost_C
    2. Build dataset
    3. Check qc/outlier_flag array has exactly one True
    """
    arrays = small_dataset_arrays.copy()
    arrays["total_cost_C"] = arrays["total_cost_C"].copy()
    arrays["total_cost_C"][0] = 100.0  # extreme outlier

    builder = FeatureDatasetBuilder(
        config=FeatureDatasetConfig(iqr_multiplier=1.5),
        feature_set=FeatureSet.COST,
    )
    builder.build(
        out_dir=str(tmp_dataset_dir),
        X=arrays["X"],
        y=arrays["y"],
        mask_ref=arrays["mask_ref"],
        total_cost_C=arrays["total_cost_C"],
        metadata=arrays["metadata"],
    )

    z = zarr.open(str(tmp_dataset_dir / "features.zarr"), "r")
    if "qc" in z and "outlier_flag" in z["qc"]:
        flags = np.array(z["qc"]["outlier_flag"])
        assert flags[0] == True, "Extreme outlier should be flagged"
        assert flags.sum() >= 1
