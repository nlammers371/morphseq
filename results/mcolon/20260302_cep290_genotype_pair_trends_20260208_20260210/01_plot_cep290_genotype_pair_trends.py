"""
Standard feature-over-time plots for recent CEP290 experiments (2026-02-08, 2026-02-10).

Experiments:
- 20260208
- 20260210

Plots (per experiment):
- Length + curvature over time, faceted by genotype
- Length + curvature over time, faceted by pair (genotypes overlaid)
- Pair × genotype grids for each feature (length, curvature)
- Single-panel overlaps (each feature, colored by genotype)

Outputs saved under:
- figures/
- results/
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_IDS = ["20260208", "20260210"]
FEATURES = ["total_length_um", "baseline_deviation_normalized"]


def _normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower()
    g = g.replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")

    # Common typos observed in these datasets
    g = g.replace("cep290_unkown", "cep290_unknown")
    g = g.replace("cep290_homozyous", "cep290_homozygous")

    return g


def _load_experiment_df(project_root: Path, experiment_id: str) -> pd.DataFrame:
    data_path = (
        project_root
        / "morphseq_playground"
        / "metadata"
        / "build06_output"
        / f"df03_final_output_with_latents_{experiment_id}.csv"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)
    if "experiment_id" in df.columns:
        df = df[df["experiment_id"].astype(str) == str(experiment_id)].copy()
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()

    required_cols = {"embryo_id", "genotype", "pair", "predicted_stage_hpf", *FEATURES}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df["pair"] = df["pair"].fillna("unknown").astype(str)

    # Keep only cep290 genotypes (defensive; expected already)
    df = df[df["genotype"].str.startswith("cep290")].copy()

    return df


def _save_plot_feature_over_time(figs, html_path: Path, png_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(figs, dict):
        figs["plotly"].write_html(html_path)
        figs["matplotlib"].savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(figs["matplotlib"])
    else:
        # Fallback: matplotlib-only
        figs.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(figs)


def main() -> None:
    run_dir = Path(__file__).resolve().parent
    figures_dir = run_dir / "figures"
    results_dir = run_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root / "src"))

    from analyze.trajectory_analysis.viz.styling import (
        get_color_for_genotype,
        sort_genotypes_by_suffix,
    )
    from analyze.viz.plotting import plot_feature_over_time

    for experiment_id in EXPERIMENT_IDS:
        df = _load_experiment_df(project_root, experiment_id)

        embryo_df = df.drop_duplicates(subset="embryo_id")[
            ["embryo_id", "genotype", "pair"]
        ].copy()
        total_embryos = embryo_df["embryo_id"].nunique()

        genotype_counts = (
            embryo_df.groupby("genotype", observed=True)["embryo_id"]
            .nunique()
            .rename("n_embryos")
            .sort_values(ascending=False)
            .reset_index()
        )
        genotype_counts.to_csv(
            results_dir / f"raw_genotype_counts_{experiment_id}.csv",
            index=False,
        )

        pair_counts = (
            embryo_df.groupby("pair", observed=True)["embryo_id"]
            .nunique()
            .rename("n_embryos")
            .sort_values(ascending=False)
            .reset_index()
        )
        pair_counts.to_csv(
            results_dir / f"raw_pair_counts_{experiment_id}.csv",
            index=False,
        )

        genotype_order = sort_genotypes_by_suffix(genotype_counts["genotype"].tolist())
        color_lookup = {gt: get_color_for_genotype(gt) for gt in genotype_order}

        # ------------------------------------------------------------------
        # Feature-over-time: faceted by genotype (rows=features, cols=genotype)
        # ------------------------------------------------------------------
        figs = plot_feature_over_time(
            df,
            features=FEATURES,
            color_by="genotype",
            color_lookup=color_lookup,
            facet_col="genotype",
            show_individual=True,
            show_error_band=True,
            trend_statistic="median",
            backend="both",
            title=f"{experiment_id}: Length + Curvature by Genotype",
        )
        _save_plot_feature_over_time(
            figs,
            figures_dir / f"{experiment_id}_length_curvature_by_genotype.html",
            figures_dir / f"{experiment_id}_length_curvature_by_genotype.png",
        )

        # ------------------------------------------------------------------
        # Feature-over-time: faceted by pair, genotypes overlaid
        # ------------------------------------------------------------------
        figs = plot_feature_over_time(
            df,
            features=FEATURES,
            color_by="genotype",
            color_lookup=color_lookup,
            facet_col="pair",
            show_individual=True,
            show_error_band=True,
            trend_statistic="median",
            backend="both",
            title=f"{experiment_id}: Length + Curvature by Pair (Genotypes Overlaid)",
        )
        _save_plot_feature_over_time(
            figs,
            figures_dir / f"{experiment_id}_length_curvature_by_pair_overlay_genotype.html",
            figures_dir / f"{experiment_id}_length_curvature_by_pair_overlay_genotype.png",
        )

        # ------------------------------------------------------------------
        # Pair × genotype grid, one feature at a time
        # ------------------------------------------------------------------
        for feature in FEATURES:
            figs = plot_feature_over_time(
                df,
                features=feature,
                color_by="genotype",
                color_lookup=color_lookup,
                facet_row="pair",
                facet_col="genotype",
                show_individual=True,
                show_error_band=True,
                trend_statistic="median",
                backend="both",
                title=f"{experiment_id}: {feature} by Pair × Genotype",
            )
            _save_plot_feature_over_time(
                figs,
                figures_dir / f"{experiment_id}_{feature}_grid_pair_by_genotype.html",
                figures_dir / f"{experiment_id}_{feature}_grid_pair_by_genotype.png",
            )

        # ------------------------------------------------------------------
        # Single-panel overlap per feature (colored by genotype)
        # ------------------------------------------------------------------
        for feature in FEATURES:
            figs = plot_feature_over_time(
                df,
                features=feature,
                color_by="genotype",
                color_lookup=color_lookup,
                show_individual=True,
                show_error_band=True,
                trend_statistic="median",
                backend="both",
                title=f"{experiment_id}: {feature} (Genotypes Overlaid)",
            )
            _save_plot_feature_over_time(
                figs,
                figures_dir / f"{experiment_id}_{feature}_overlap_by_genotype.html",
                figures_dir / f"{experiment_id}_{feature}_overlap_by_genotype.png",
            )

        summary_lines = [
            f"experiment_id: {experiment_id}",
            f"total_rows: {len(df)}",
            f"total_embryo_ids: {total_embryos}",
            "",
            "genotype_counts (embryos):",
        ]
        for _, row in genotype_counts.iterrows():
            summary_lines.append(f"- {row['genotype']}: n={int(row['n_embryos'])}")
        summary_lines += ["", "pair_counts (embryos):"]
        for _, row in pair_counts.iterrows():
            summary_lines.append(f"- {row['pair']}: n={int(row['n_embryos'])}")
        (results_dir / f"summary_{experiment_id}.txt").write_text("\n".join(summary_lines) + "\n")

        print("\n".join(summary_lines))
        print("\nSaved outputs under:")
        print(f"- {figures_dir}")
        print(f"- {results_dir}")


if __name__ == "__main__":
    main()

