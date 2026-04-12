"""
Two plots:

Plot 1 — PBX axes (3 columns, one per crispant direction).
  Genotypes: wik_ab, inj_ctrl, pbx1b, pbx4, pbx1b+pbx4.
  WT centering: per-bin wik_ab centroid.

Plot 2 — CEP290 axis (1 column: cep290_homozygous vs wik_ab direction).
  Genotypes: wik_ab, inj_ctrl, pbx1b, pbx4, pbx1b+pbx4  (no cep290 classes).
  WT centering: per-bin wik_ab centroid on the CEP290 axis.
  cep290_homozygous shown in magenta-pink as reference.

Raw frame-level time is passed to plot_feature_over_time so smoothing works.
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

from analyze.classification import run_classification
from analyze.viz.plotting import plot_feature_over_time
from analyze.viz.plotting.feature_over_time import FacetSpec
from phenotype_direction import load_classifier_directions
from common import load_combined_pbx_dataframe, SENSITIVITY_GENOTYPES

# ── shared constants ──────────────────────────────────────────────────────────
PBX_EXPERIMENT_IDS = ["20251207_pbx", "20260304", "20260306"]
PBX_GENOTYPES      = SENSITIVITY_GENOTYPES   # wik_ab + inj_ctrl + 3 crispants
WT_GENOTYPE        = "wik_ab"

CEP290_DATA_PATH = (
    REPO_ROOT
    / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)

FEATURE_SET = "vae"
TIME_COL    = "predicted_stage_hpf"
CLASS_COL   = "genotype"
ID_COL      = "embryo_id"
BIN_WIDTH   = 4.0
NWT_MIN     = 4

RESULTS_DIR = SCRIPT_DIR / "results" / "pbx_direction_smoke_5class_bin4_perm0_wt_ref"
FIGURES_DIR = SCRIPT_DIR / "figures" / "pbx_direction_smoke_5class_bin4_perm0_wt_ref"

CEP290_RESULTS_DIR = SCRIPT_DIR / "results" / "cep290_direction_for_pbx_projection"

PBX_CRISPANT_COMPARISONS = [
    "pbx1b_crispant__vs__wik_ab",
    "pbx4_crispant__vs__wik_ab",
    "pbx1b_pbx4_crispant__vs__wik_ab",
]
PBX_COL_LABELS = ["pbx1b axis", "pbx4 axis", "pbx1b+pbx4 axis"]

SCORE_COL     = "score_wt_centered"
DIRECTION_COL = "axis_direction"

PBX_COLORS = {
    "wik_ab":              "#808080",
    "inj_ctrl":            "#2166AC",
    "pbx1b_crispant":      "#9467BD",
    "pbx4_crispant":       "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}

CEP290_AXIS_COLORS = {
    **PBX_COLORS,
    "cep290_homozygous": "#FF69B4",   # magenta-pink reference
}

VECTOR_NORM_TOL = 1e-6


# ── helpers ───────────────────────────────────────────────────────────────────

def _wt_mu_by_bin(df: pd.DataFrame, wt_label: str, raw_col: str) -> dict[float, float]:
    """Per-4hpf-bin WT centroid with NWT_MIN rolling fallback."""
    df = df.copy()
    df["_bin"] = np.floor(df[TIME_COL] / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2.0
    sorted_bins = sorted(df["_bin"].unique().tolist())
    wt_by_bin = {
        tbc: df.loc[(df["_bin"] == tbc) & (df[CLASS_COL] == wt_label), raw_col].to_numpy(dtype=float)
        for tbc in sorted_bins
    }
    n = len(sorted_bins)
    mu: dict[float, float] = {}
    for i, tbc in enumerate(sorted_bins):
        wt = wt_by_bin[tbc]
        if len(wt) >= NWT_MIN:
            mu[tbc] = float(np.mean(wt))
        else:
            pooled = list(wt)
            r = 1
            while len(pooled) < NWT_MIN and r <= n:
                if i - r >= 0:
                    pooled.extend(wt_by_bin[sorted_bins[i - r]].tolist())
                if i + r < n:
                    pooled.extend(wt_by_bin[sorted_bins[i + r]].tolist())
                r += 1
            mu[tbc] = float(np.mean(pooled)) if pooled else float("nan")
    return mu


def _add_centered_score(
    df: pd.DataFrame,
    axis: np.ndarray,
    feature_names: list[str],
    wt_label: str,
) -> pd.DataFrame:
    df = df.copy()
    df["_raw"] = df[feature_names].to_numpy(dtype=float) @ axis
    mu = _wt_mu_by_bin(df, wt_label, "_raw")
    df["_bin"] = np.floor(df[TIME_COL] / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2.0
    df[SCORE_COL] = df.apply(
        lambda r: r["_raw"] - mu.get(r["_bin"], float("nan")), axis=1
    )
    return df.drop(columns=["_raw", "_bin"])


def _build_weighted_axis(
    results_dir: Path, comparison_id: str, feature_names: list[str]
) -> np.ndarray:
    directions = load_classifier_directions(results_dir)
    meta = directions.metadata
    rows = meta[
        (meta["feature_set"] == FEATURE_SET)
        & (meta["comparison_id"] == comparison_id)
        & (meta["direction_space"] == "raw_feature_space")
        & (meta["vector_kind"] == "signed_unit_coef")
        & (meta["auroc_obs"] >= 0.55)
    ].copy()
    if rows.empty:
        raise ValueError(f"No gated vectors for {comparison_id!r}.")
    weights = np.maximum(rows["auroc_obs"].to_numpy(dtype=float) - 0.5, 0.0)
    vecs = np.vstack([
        np.asarray(directions.vectors[vid], dtype=float).ravel()
        for vid in rows["vector_id"]
    ])
    axis = np.sum(vecs * weights[:, None], axis=0)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError(f"Zero axis for {comparison_id!r}.")
    return axis / norm


def _load_pbx_raw() -> pd.DataFrame:
    df = load_combined_pbx_dataframe(
        experiment_ids=PBX_EXPERIMENT_IDS,
        genotypes=PBX_GENOTYPES,
    )
    if TIME_COL not in df.columns and "stage_hpf" in df.columns:
        df = df.rename(columns={"stage_hpf": TIME_COL})
    return df


# ── Plot 1: PBX axes ──────────────────────────────────────────────────────────

def _build_pbx_long(pbx_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    segments = []
    for cid, label in zip(PBX_CRISPANT_COMPARISONS, PBX_COL_LABELS):
        axis = _build_weighted_axis(RESULTS_DIR, cid, feature_names)
        seg = _add_centered_score(pbx_df, axis, feature_names, WT_GENOTYPE)
        seg = seg[[ID_COL, CLASS_COL, TIME_COL, SCORE_COL]].copy()
        seg[DIRECTION_COL] = label
        segments.append(seg)
    long = pd.concat(segments, ignore_index=True)
    return long.dropna(subset=[SCORE_COL, TIME_COL])


# ── Plot 2: CEP290 axis, PBX genotypes projected ──────────────────────────────

def _train_cep290_direction(feature_names: list[str]) -> None:
    """Train cep290_homozygous vs cep290_wildtype and save directions."""
    CEP290_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CEP290_DATA_PATH, low_memory=False)
    df = df[df["genotype"].isin(["cep290_wildtype", "cep290_homozygous"])].copy()
    df = df[df[ID_COL].notna() & df[TIME_COL].notna()].copy()
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"CEP290 data missing VAE features: {missing[:3]}...")
    run_classification(
        df=df,
        class_col=CLASS_COL,
        id_col=ID_COL,
        time_col=TIME_COL,
        comparisons=[{"positive": "cep290_homozygous", "negative": "cep290_wildtype"}],
        features={FEATURE_SET: "z_mu_b"},
        bin_width=BIN_WIDTH,
        n_permutations=0,
        n_splits=5,
        n_jobs=1,
        random_state=42,
        save_classifier_directions=True,
        save_predictions=False,
        save_dir=CEP290_RESULTS_DIR,
        overwrite=True,
        verbose=False,
    )


def _build_cep290_axis_long(pbx_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Project PBX genotypes + cep290_homozygous onto the CEP290 axis."""
    cep290_axis = _build_weighted_axis(
        CEP290_RESULTS_DIR, "cep290_homozygous__vs__cep290_wildtype", feature_names
    )

    # PBX genotypes projected and centered by wik_ab
    pbx_seg = _add_centered_score(pbx_df, cep290_axis, feature_names, WT_GENOTYPE)
    pbx_seg = pbx_seg[[ID_COL, CLASS_COL, TIME_COL, SCORE_COL]].copy()

    # cep290_homozygous — load just homos, center by cep290_wildtype
    cep_df = pd.read_csv(CEP290_DATA_PATH, low_memory=False)
    cep_df = cep_df[cep_df["genotype"].isin(["cep290_wildtype", "cep290_homozygous"])].copy()
    cep_df = cep_df[cep_df[ID_COL].notna() & cep_df[TIME_COL].notna()].copy()
    if "use_embryo_flag" in cep_df.columns:
        cep_df = cep_df[cep_df["use_embryo_flag"]].copy()

    cep_seg = _add_centered_score(cep_df, cep290_axis, feature_names, "cep290_wildtype")
    # Keep only homozygous (wildtype was only used for centering)
    cep_seg = cep_seg[cep_seg[CLASS_COL] == "cep290_homozygous"][[ID_COL, CLASS_COL, TIME_COL, SCORE_COL]].copy()

    long = pd.concat([pbx_seg, cep_seg], ignore_index=True)
    return long.dropna(subset=[SCORE_COL, TIME_COL])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    directions = load_classifier_directions(RESULTS_DIR)
    feature_names = list(directions.feature_names.get(FEATURE_SET, []))
    if not feature_names:
        raise ValueError(f"No feature_names for {FEATURE_SET!r} in saved bundle.")

    pbx_df = _load_pbx_raw()

    # ── Plot 1: PBX axes ──
    pbx_long = _build_pbx_long(pbx_df, feature_names)
    plot_feature_over_time(
        df=pbx_long,
        features=SCORE_COL,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by=CLASS_COL,
        color_lookup=PBX_COLORS,
        facet_col=DIRECTION_COL,
        layout=FacetSpec(col_order=PBX_COL_LABELS, sharex=True, sharey=False),
        backend="matplotlib",
        output_path=FIGURES_DIR / "pbx_projection_feature_over_time.png",
        title="PBX axis projections per embryo (WT-centered, raw VAE units)",
    )
    print(f"Saved: {FIGURES_DIR / 'pbx_projection_feature_over_time.png'}")

    # ── Plot 2: CEP290 axis, PBX genotypes ──
    if not (CEP290_RESULTS_DIR / "classifier_directions.parquet").exists():
        print("Training CEP290 direction...")
        _train_cep290_direction(feature_names)

    cep290_long = _build_cep290_axis_long(pbx_df, feature_names)
    plot_feature_over_time(
        df=cep290_long,
        features=SCORE_COL,
        time_col=TIME_COL,
        id_col=ID_COL,
        color_by=CLASS_COL,
        color_lookup=CEP290_AXIS_COLORS,
        backend="matplotlib",
        output_path=FIGURES_DIR / "cep290_axis_pbx_projection_feature_over_time.png",
        title="CEP290 axis: PBX genotypes projected (WT-centered, raw VAE units)",
    )
    print(f"Saved: {FIGURES_DIR / 'cep290_axis_pbx_projection_feature_over_time.png'}")


if __name__ == "__main__":
    main()
