"""
Q3 extension: Per-embryo direction classification (run only).

Loads data, builds raw/unit direction vectors, runs classification, and saves
results to disk. Re-running is safe — existing results are overwritten.

Run this script first. Then run 07b_q3_plot_heatmaps.py to generate figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR       = Path(__file__).resolve().parent
REPO_ROOT        = SCRIPT_DIR.parents[2]
PBX_ANALYSIS_DIR = REPO_ROOT / "results" / "mcolon" / "20260407_pbx_analysis_cont"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PBX_ANALYSIS_DIR))

from common import load_combined_pbx_dataframe
from analyze.utils.binning import add_time_bins
from analyze.classification import run_classification

EXPERIMENT_IDS = ["20260304", "20260306"]
GENOTYPES      = ["wik_ab", "inj_ctrl", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant"]
TIME_COL  = "stage_hpf"
CLASS_COL = "genotype"
ID_COL    = "embryo_id"
BIN_WIDTH = 4.0
MIN_N     = 3

NON_INJ  = "wik_ab"
INJ_CTRL = "inj_ctrl"
PBX1B    = "pbx1b_crispant"
PBX4     = "pbx4_crispant"

CLF_GENOTYPES = [INJ_CTRL, PBX1B, PBX4]

RESULTS_DIR = SCRIPT_DIR / "results" / "q3_embryo_direction"


def _load_and_bin() -> tuple[pd.DataFrame, list[str]]:
    df = load_combined_pbx_dataframe(experiment_ids=EXPERIMENT_IDS, genotypes=GENOTYPES)
    df = add_time_bins(df, time_col=TIME_COL, bin_width=BIN_WIDTH, bin_col="time_bin")
    feat_cols = [c for c in df.columns if "z_mu_b" in c]
    if not feat_cols:
        raise ValueError("No VAE feature columns found matching 'z_mu_b'.")
    binned = df.groupby([ID_COL, CLASS_COL, "time_bin"], as_index=False)[feat_cols].mean()
    return binned, feat_cols


def _build_direction_dfs(
    binned: pd.DataFrame, feat_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows:  list[pd.Series] = []
    unit_rows: list[pd.Series] = []

    for tb, grp in binned.groupby("time_bin"):
        non_rows = grp[grp[CLASS_COL] == NON_INJ]
        if len(non_rows) < MIN_N:
            continue
        mu_non = non_rows[feat_cols].mean().to_numpy(dtype=float)

        for _, row in grp[grp[CLASS_COL].isin(CLF_GENOTYPES)].iterrows():
            z = row[feat_cols].to_numpy(dtype=float) - mu_non
            z_unit = z / (np.linalg.norm(z) + 1e-12)

            meta = {ID_COL: row[ID_COL], CLASS_COL: row[CLASS_COL], "time_bin": tb}
            raw_rows.append(pd.Series({**meta, **dict(zip(feat_cols, z))}))
            unit_rows.append(pd.Series({**meta, **dict(zip(feat_cols, z_unit))}))

    df_raw  = pd.DataFrame(raw_rows).reset_index(drop=True)
    df_unit = pd.DataFrame(unit_rows).reset_index(drop=True)
    return df_raw, df_unit


def _run_clf(df_feat: pd.DataFrame, feat_cols: list[str], save_subdir: Path) -> None:
    run_classification(
        df_feat,
        class_col=CLASS_COL,
        id_col=ID_COL,
        time_col="time_bin",
        features={"emb": feat_cols},
        comparisons="all_pairs",
        bin_width=BIN_WIDTH,
        n_splits=5,
        n_permutations=500,
        n_jobs=-1,
        save_null_arrays=True,
        save_dir=save_subdir,
        overwrite=True,
    )
    print(f"Saved results to {save_subdir}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    binned, feat_cols = _load_and_bin()
    print(f"Loaded {len(binned)} embryo-bin rows across {binned['time_bin'].nunique()} bins.")

    df_raw, df_unit = _build_direction_dfs(binned, feat_cols)
    print(f"Direction DataFrames built: raw={len(df_raw)}, unit={len(df_unit)} rows.")

    print("Running classification on raw direction vectors...")
    _run_clf(df_raw, feat_cols, RESULTS_DIR / "raw_direction")

    print("Running classification on unit direction vectors...")
    _run_clf(df_unit, feat_cols, RESULTS_DIR / "unit_direction")


if __name__ == "__main__":
    main()
