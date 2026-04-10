from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import warnings

import matplotlib

cache_root = Path("/tmp") / "morphseq_bridge_qc_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
warnings.filterwarnings("ignore", category=FutureWarning, message=".*multi_class.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*liblinear.*multiclass classification.*deprecated.*")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification import run_classification
from analyze.classification.viz import plot_auroc_heatmaps

from common import (
    BRIDGE_LABEL,
    CURRENT_REFERENCE_LABEL,
    REPO_ROOT as _COMMON_REPO_ROOT,
    SHARED_GENOTYPES,
    load_bridge_ready_dataframe,
    short_name,
)

assert REPO_ROOT == _COMMON_REPO_ROOT


FEATURES = {
    "curvature": ["baseline_deviation_normalized"],
    "length": ["total_length_um"],
    "embedding": "z_mu_b",
}


def _summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (genotype, feature_set), group in scores.groupby(["shared_genotype", "feature_set"], observed=True):
        rows.append(
            {
                "shared_genotype": genotype,
                "feature_set": feature_set,
                "max_auroc": float(group["auroc_obs"].max()),
                "median_auroc": float(group["auroc_obs"].median()),
                "min_pval": float(group["pval"].min()),
                "n_time_bins": int(group["time_bin_center"].nunique()),
                "n_sig_bins_p001": int((group["pval"] <= 0.01).sum()),
                "n_sig_bins_p005": int((group["pval"] <= 0.05).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_set", "shared_genotype"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge QC heatmaps for 20251207_pbx vs current PBX reference.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "bridge_qc_bin4_perm500",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "figures" / "bridge_qc_bin4_perm500",
    )
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--min-samples-per-group", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_bridge_ready_dataframe()
    feature_cols = sorted(c for c in df.columns if c.startswith("z_mu_b_"))
    if not feature_cols:
        raise ValueError("No z_mu_b embedding features found in bridge dataframe.")

    score_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for genotype in SHARED_GENOTYPES:
        df_genotype = df[df["genotype"] == genotype].copy()
        analysis = run_classification(
            df_genotype,
            class_col="source_group",
            id_col="embryo_id",
            time_col="stage_hpf_bridge",
            positive=BRIDGE_LABEL,
            negative=CURRENT_REFERENCE_LABEL,
            features={
                "curvature": FEATURES["curvature"],
                "length": FEATURES["length"],
                "embedding": feature_cols,
            },
            n_jobs=int(args.n_jobs),
            n_permutations=int(args.n_permutations),
            bin_width=float(args.bin_width),
            min_samples_per_group=int(args.min_samples_per_group),
            random_state=42,
            verbose=False,
        )
        scores = analysis.scores.copy()
        scores["shared_genotype"] = genotype
        scores["shared_genotype_short"] = short_name(genotype)
        score_frames.append(scores)
        summary_rows.append(
            {
                "shared_genotype": genotype,
                "bridge_embryos": int(df_genotype[df_genotype["source_group"] == BRIDGE_LABEL]["embryo_id"].nunique()),
                "current_embryos": int(df_genotype[df_genotype["source_group"] == CURRENT_REFERENCE_LABEL]["embryo_id"].nunique()),
            }
        )

    scores_df = pd.concat(score_frames, ignore_index=True)
    support_df = pd.DataFrame(summary_rows)
    summary_df = _summarize_scores(scores_df).merge(support_df, on="shared_genotype", how="left")

    scores_df.to_csv(args.results_dir / "bridge_qc_experiment_comparisons.csv", index=False)
    summary_df.to_csv(args.results_dir / "bridge_qc_summary.csv", index=False)

    fig = plot_auroc_heatmaps(
        scores_df,
        heatmap_row="shared_genotype",
        heatmap_col="time_bin_center",
        facet_row="feature_set",
        facet_col=None,
        heatmap_row_order=SHARED_GENOTYPES,
        facet_row_order=["curvature", "length", "embedding"],
        title="20251207_pbx vs current PBX reference within shared genotype",
        x_label="Time (hpf)",
        y_label="Shared genotype",
        colorbar_label="AUROC",
        sig_threshold=0.01,
        backend="matplotlib",
        cmap="BuPu",
        vmin=0.4,
        vmax=1.0,
    )
    out_path = args.figures_dir / "bridge_qc_experiment_source_heatmaps.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(args.results_dir / "bridge_qc_summary.csv")
    print(out_path)


if __name__ == "__main__":
    main()
