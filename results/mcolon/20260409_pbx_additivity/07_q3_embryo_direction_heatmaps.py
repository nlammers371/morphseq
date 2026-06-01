"""
Q3 extension: Per-embryo direction heatmaps.

Each embryo's VAE latent vector is centered relative to the non_inj mean within
its time bin, giving a direction vector z_i = x_i - mu_non.

Two feature variants are classified with all-pairs:
  raw_direction : z_i                          (direction + magnitude)
  unit_direction: z_i / ||z_i||               (direction only, magnitude removed)

Comparing AUROC between the two variants reveals whether the classifier is
separating genotypes by WHERE they go in latent space (direction) or HOW FAR
(magnitude). inj_ctrl is the negative control — should show low AUROC vs pbx targets.

Output: two faceted AUROC heatmaps (positive_label × time, faceted by negative_label).
Row order: pbx4 (top), pbx1b (middle), inj_ctrl (bottom).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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
from analyze.classification.engine.analysis import ClassificationAnalysis
from analyze.classification.viz.heatmaps import plot_auroc_heatmaps

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

# Genotypes to classify — wik_ab is used only as origin reference
CLF_GENOTYPES = [INJ_CTRL, PBX1B, PBX4]
ROW_ORDER     = [INJ_CTRL, PBX1B, PBX4]

RESULTS_DIR = SCRIPT_DIR / "results" / "q3_embryo_direction"
FIGURES_DIR = SCRIPT_DIR / "figures" / "q3_embryo_direction"


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
    """
    For each embryo in CLF_GENOTYPES, subtract the non_inj mean of its time bin.
    Returns (df_raw, df_unit) — same structure as binned, feat_cols replaced.
    """
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


def _run_clf(df_feat: pd.DataFrame, feat_cols: list[str], save_subdir: Path) -> object:
    scores_path = save_subdir / "scores.parquet"
    if scores_path.exists():
        print(f"Results exist at {save_subdir}, loading (skipping classification run).")
        return ClassificationAnalysis.load(save_subdir)
    return run_classification(
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
    )


def _add_separability(scores: pd.DataFrame, res: object, save_dir: Path) -> pd.DataFrame:
    """
    Replace signed AUROC with separability = max(auroc, 1-auroc) and compute
    symmetric permutation p-values: p = fraction of null |auroc_null - 0.5|
    >= |auroc_obs - 0.5|.

    This makes the metric orientation-independent: AUROC 0.4 and 0.6 are both
    separability 0.6 with the same p-value, which is correct because the
    positive-label assignment is arbitrary (alphabetical).

    NOTE: run_classification always assigns the alphabetically-first label as
    positive_label — so without mirroring, inj_ctrl would never appear as a
    facet column. After converting to separability, mirroring is trivial:
    A-vs-B and B-vs-A have identical separability, so we just duplicate rows
    with labels swapped.
    """
    import numpy as np

    scores = scores.copy()
    scores["separability"] = np.maximum(scores["auroc_obs"], 1.0 - scores["auroc_obs"])

    # Compute symmetric p-values from null arrays.
    # Load raw npz directly (allow_pickle=True needed for string arrays in some numpy versions).
    null_path = Path(save_dir) / "null_distributions.npz"
    if null_path.exists():
        raw = np.load(null_path, allow_pickle=True)
        feat_sets   = raw["feature_set"]
        comp_ids_arr = raw["comparison_id"]
        tbc_arr     = raw["time_bin_center"]
        null_auc    = raw["null_auc"]
        pvals = []
        for _, row in scores.iterrows():
            mask = (
                (feat_sets == row["feature_set"]) &
                (comp_ids_arr == row["comparison_id"]) &
                (np.abs(tbc_arr - row["time_bin_center"]) < 0.1)
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                pvals.append(float("nan"))
                continue
            null_aurocs = null_auc[idx[0]]
            sep_obs  = abs(row["auroc_obs"] - 0.5)
            sep_null = np.abs(null_aurocs - 0.5)
            p = (np.sum(sep_null >= sep_obs) + 1) / (len(sep_null) + 1)
            pvals.append(float(p))
        scores["pval_sym"] = pvals
    else:
        scores["pval_sym"] = np.nan

    return scores


def _mirror_separability(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror A-vs-B into B-vs-A. Because separability = max(auroc, 1-auroc)
    is symmetric, both directions have identical values — just swap labels.

    NOTE: p-values are also identical (symmetric test), so no transformation needed.
    """
    flipped = scores.copy()
    flipped["positive_label"] = scores["negative_label"].values
    flipped["negative_label"] = scores["positive_label"].values
    return pd.concat([scores, flipped], ignore_index=True)


def _plot_heatmap(res: object, save_dir: Path, out_path: Path, title: str) -> None:
    scores = res.scores if hasattr(res, "scores") else res
    scores = _add_separability(scores, res, save_dir)
    mirrored = _mirror_separability(scores)
    row_order = [g for g in ROW_ORDER if g in set(mirrored["positive_label"].unique())]

    pval_col = "pval_sym" if not mirrored["pval_sym"].isna().all() else None

    plot_auroc_heatmaps(
        mirrored,
        heatmap_row="positive_label",
        heatmap_col="time_bin_center",
        facet_col="negative_label",
        heatmap_row_order=row_order or None,
        auroc_col="separability",
        pval_col=pval_col or "pval_sym",
        show_significance=pval_col is not None,
        title=title,
        x_label="Time bin center (hpf)",
        y_label="Positive genotype",
        colorbar_label="Separability  max(AUROC, 1−AUROC)",
        vmin=0.5,
        vmax=1.0,
        sig_threshold=0.05,
        output_path=out_path,
        backend="matplotlib",
    )
    print(f"Saved: {out_path}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    binned, feat_cols = _load_and_bin()
    print(f"Loaded {len(binned)} embryo-bin rows across {binned['time_bin'].nunique()} bins.")

    df_raw, df_unit = _build_direction_dfs(binned, feat_cols)
    print(f"Direction DataFrames built: raw={len(df_raw)}, unit={len(df_unit)} rows.")

    print("Running classification on raw direction vectors...")
    res_raw = _run_clf(df_raw, feat_cols, RESULTS_DIR / "raw_direction")

    print("Running classification on unit direction vectors...")
    res_unit = _run_clf(df_unit, feat_cols, RESULTS_DIR / "unit_direction")

    _plot_heatmap(
        res_raw,
        save_dir=RESULTS_DIR / "raw_direction",
        out_path=FIGURES_DIR / "q3_sep_raw_direction.png",
        title=(
            "Q3: Separability — raw direction vectors (z_i = x_i − mu_non)\n"
            "max(AUROC, 1−AUROC): orientation-independent, 0.5 = chance\n"
            "direction + magnitude combined | inj_ctrl row should be near 0.5 vs pbx"
        ),
    )
    _plot_heatmap(
        res_unit,
        save_dir=RESULTS_DIR / "unit_direction",
        out_path=FIGURES_DIR / "q3_sep_unit_direction.png",
        title=(
            "Q3: Separability — unit direction vectors (z_i / ||z_i||)\n"
            "max(AUROC, 1−AUROC): orientation-independent, 0.5 = chance\n"
            "direction only (magnitude removed) | raw >> unit → magnitude drives sep | raw ≈ unit → direction sufficient"
        ),
    )


if __name__ == "__main__":
    main()
