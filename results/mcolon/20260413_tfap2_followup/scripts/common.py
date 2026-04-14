"""Shared constants and utilities for the TFAP2 followup analysis.

Reuses the aggregated parquet from the first-pass run to avoid re-aggregating.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Experiment constants (same as first-pass)
# ---------------------------------------------------------------------------

EXPERIMENT_IDS = ["20260213", "20260223", "20260224", "20260319", "20260320"]
EXPERIMENT_LABEL = "_".join(EXPERIMENT_IDS)
OVERLAP_FEATURE = "baseline_deviation_normalized"
FEATURES = ["total_length_um", OVERLAP_FEATURE]
AGGREGATE_STEM = f"tfap2_combined_{EXPERIMENT_LABEL}"
AGGREGATE_PARQUET_NAME = f"{AGGREGATE_STEM}.parquet"

# Classification / binning parameters
BIN_WIDTH = 2.0
MIN_SUPPORT = 3

# Reuse the already-aggregated parquet from the first-pass run
_THIS_FILE = Path(__file__).resolve()
FOLLOWUP_DIR = _THIS_FILE.parents[1]
FIRST_PASS_DIR = FOLLOWUP_DIR.parent / "20260324_tfap2_crispant_first_pass"


def load_aggregate_dataframe(
    run_dir: str | Path | None = None,
    *,
    required_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Load the pre-aggregated TFAP2 parquet.

    Defaults to the first-pass results directory if run_dir is not given.
    """
    if run_dir is None:
        run_dir = FIRST_PASS_DIR
    parquet_path = Path(run_dir) / "results" / AGGREGATE_PARQUET_NAME
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Aggregate parquet not found: {parquet_path}\n"
            "Run 20260324_tfap2_crispant_first_pass/scripts/data_gen/00_aggregate_experiments.py first."
        )
    df = pd.read_parquet(parquet_path)
    if required_cols:
        missing = sorted(required_cols - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns in aggregate: {missing}")
    return df


def load_supported_window(results_dir: str | Path) -> dict:
    """Load the supported time window produced by 01_support_table.py."""
    import json
    path = Path(results_dir) / "supported_window.json"
    if not path.exists():
        raise FileNotFoundError(
            f"supported_window.json not found at {path}. "
            "Run 01_support_table.py first."
        )
    return json.loads(path.read_text())
