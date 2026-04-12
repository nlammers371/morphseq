from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib

cache_root = Path("/tmp") / "morphseq_bridge_feature_over_time_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.trajectory_condensation import render_feature_over_time_facets

from common import BRIDGE_EXPERIMENT_ID, CURRENT_EXPERIMENT_IDS, SHARED_GENOTYPES, load_bridge_ready_dataframe, short_name


FEATURE_LABELS = {
    "baseline_deviation_normalized": "Curvature",
    "total_length_um": "Length (um)",
    "embedding_pc1": "Embedding PC1",
}

EXPERIMENT_LABELS = {
    BRIDGE_EXPERIMENT_ID: "20251207_pbx",
    "20260304": "20260304",
    "20260306": "20260306",
}

EXPERIMENT_COLORS = {
    BRIDGE_EXPERIMENT_ID: "#dd8452",
    "20260304": "#4c72b0",
    "20260306": "#55a868",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bridge QC features over time by genotype and experiment.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "bridge_feature_over_time_bin4",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "figures" / "bridge_feature_over_time_bin4",
    )
    parser.add_argument("--bin-width", type=float, default=4.0)
    return parser.parse_args()


def _assign_time_bins(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    out = df.copy()
    out["time_bin"] = np.floor(out["stage_hpf_bridge"] / bin_width) * bin_width
    out["time_bin_center"] = out["time_bin"] + bin_width / 2.0
    return out


def _build_embedding_pc1(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = sorted(c for c in df.columns if c.startswith("z_mu_b_"))
    if not feature_cols:
        raise ValueError("No z_mu_b_* columns available for embedding PC1.")
    valid = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    keep = valid.notna().all(axis=1)
    if not keep.any():
        raise ValueError("No rows with complete embedding features for PC1.")
    scaler = StandardScaler()
    x = scaler.fit_transform(valid.loc[keep].to_numpy())
    pc1 = PCA(n_components=1, random_state=42).fit_transform(x).ravel()
    out = df.copy()
    out["embedding_pc1"] = np.nan
    out.loc[keep, "embedding_pc1"] = pc1
    return out


def _aggregate_feature_traces(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    df = _assign_time_bins(df, bin_width)
    group_cols = ["experiment_id", "genotype", "embryo_id", "time_bin", "time_bin_center"]
    embryo_bin = (
        df.groupby(group_cols, observed=True)
        .agg(
            baseline_deviation_normalized=("baseline_deviation_normalized", "mean"),
            total_length_um=("total_length_um", "mean"),
            embedding_pc1=("embedding_pc1", "mean"),
        )
        .reset_index()
    )
    embryo_bin["genotype_label"] = embryo_bin["genotype"].map(short_name)
    embryo_bin["experiment_label"] = embryo_bin["experiment_id"].map(lambda x: EXPERIMENT_LABELS.get(str(x), str(x)))
    embryo_bin["predicted_stage_hpf"] = embryo_bin["time_bin_center"]
    return embryo_bin


def _summarize_traces(embryo_bin: pd.DataFrame) -> pd.DataFrame:
    long_df = embryo_bin.melt(
        id_vars=["experiment_id", "experiment_label", "genotype", "genotype_label", "embryo_id", "time_bin", "time_bin_center", "predicted_stage_hpf"],
        value_vars=["baseline_deviation_normalized", "total_length_um", "embedding_pc1"],
        var_name="feature",
        value_name="value",
    ).dropna(subset=["value"])
    summary = (
        long_df.groupby(["feature", "genotype", "genotype_label", "experiment_id", "experiment_label", "time_bin_center"], observed=True)
        .agg(
            median_value=("value", "median"),
            q25=("value", lambda s: float(np.quantile(s, 0.25))),
            q75=("value", lambda s: float(np.quantile(s, 0.75))),
            n_embryos=("embryo_id", "nunique"),
        )
        .reset_index()
    )
    summary["feature_label"] = summary["feature"].map(FEATURE_LABELS)
    return summary


def _plot(embryo_bin: pd.DataFrame, output_png: Path, output_html: Path) -> None:
    genotype_order = [short_name(g) for g in SHARED_GENOTYPES]
    color_lookup = {EXPERIMENT_LABELS[k]: v for k, v in EXPERIMENT_COLORS.items()}
    figs = render_feature_over_time_facets(
        embryo_bin,
        features=["baseline_deviation_normalized", "total_length_um", "embedding_pc1"],
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by="experiment_label",
        color_lookup=color_lookup,
        facet_col="genotype_label",
        col_order=genotype_order,
        show_individual=True,
        show_trend=True,
        show_error_band=True,
        trend_statistic="median",
        legend_loc="outside",
        title="Bridge feature trajectories by genotype and experiment",
        output_png=output_png,
        output_html=output_html,
    )
    plt.close(figs["matplotlib"])


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_bridge_ready_dataframe()
    df = _build_embedding_pc1(df)
    embryo_bin = _aggregate_feature_traces(df, bin_width=float(args.bin_width))
    summary = _summarize_traces(embryo_bin)

    embryo_bin.to_csv(args.results_dir / "bridge_feature_over_time_embryo_bin.csv", index=False)
    summary.to_csv(args.results_dir / "bridge_feature_over_time_summary.csv", index=False)
    _plot(
        embryo_bin,
        args.figures_dir / "bridge_feature_over_time_faceted.png",
        args.figures_dir / "bridge_feature_over_time_faceted.html",
    )


if __name__ == "__main__":
    main()
