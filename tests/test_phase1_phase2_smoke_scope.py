from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.pipeline_orchestrator.targets import (
    get_experiment_microscopes,
    get_experiment_wells,
    get_experiments,
)


SMOKE_EXPERIMENTS = ["20240418", "20240509_24hpf"]
SMOKE_WELLS = {
    "20240418": {"A01", "C01"},
    "20240509_24hpf": {"A04", "B04"},
}


def test_smoke_target_defaults() -> None:
    config = {}
    assert get_experiments(config) == SMOKE_EXPERIMENTS

    microscopes = get_experiment_microscopes(config)
    assert microscopes["20240418"] == "YX1"
    assert microscopes["20240509_24hpf"] == "Keyence"

    wells = get_experiment_wells(config)
    assert set(wells["20240418"]) == SMOKE_WELLS["20240418"]
    assert set(wells["20240509_24hpf"]) == SMOKE_WELLS["20240509_24hpf"]


@pytest.mark.parametrize("experiment", SMOKE_EXPERIMENTS)
def test_output_scope_if_artifacts_exist(experiment: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    metadata_dir = repo_root / "data_pipeline_output" / "experiment_metadata" / experiment

    stitched_index = metadata_dir / "stitched_image_index.csv"
    frame_manifest = metadata_dir / "frame_manifest.csv"

    if not stitched_index.exists() or not frame_manifest.exists():
        pytest.skip("Smoke artifacts not built yet; run Snakemake workflow first.")

    stitched_df = pd.read_csv(stitched_index)
    manifest_df = pd.read_csv(frame_manifest)

    allowed_wells = SMOKE_WELLS[experiment]

    assert set(stitched_df["well_index"].astype(str).unique()).issubset(allowed_wells)
    assert set(manifest_df["well_index"].astype(str).unique()).issubset(allowed_wells)

    base_cols = [
        "well_id",
        "well_index",
        "channel_id",
        "stitched_image_path",
        "time_int",
    ]
    for col in base_cols:
        assert col in stitched_df.columns

    # Keyence rows include tiler diagnostics emitted at materialization.
    if experiment == "20240509_24hpf":
        for col in [
            "tiler_fallback_used",
            "tiler_qc_passed",
            "tiler_qc_reasons",
            "tiler_tile_count",
            "tiler_canvas_height_px",
            "tiler_canvas_width_px",
            "tiler_max_abs_shift_px",
        ]:
            assert col in stitched_df.columns

    for col in ["well_id", "well_index", "channel_id", "channel_name_raw", "stitched_image_path", "micrometers_per_pixel"]:
        assert col in manifest_df.columns
