"""
NWDB talk: curvature-over-time plots for cep290 heterozygous, wildtype, and homozygous.

Generates five plot families in two variants each:
1) heterozygous only
2) wildtype only
3) homozygous only
4) heterozygous + wildtype overlay
5) heterozygous + wildtype + homozygous overlay

Variants:
- individual traces only
- individual traces + dashed trend lines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create NWDB curvature overlays for heterozygous vs wildtype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output folder (figures/ will be created inside).",
    )
    p.add_argument(
        "--figures-subdir",
        default="curvature_het_vs_wt",
        help="Subfolder inside figures/ for outputs.",
    )
    p.add_argument(
        "--data-dir",
        default="results/mcolon/20251229_cep290_phenotype_extraction/final_data",
        help="Directory containing embryo_data_with_labels.csv.",
    )
    p.add_argument("--t-min", type=float, default=24.0, help="Minimum HPF to include.")
    p.add_argument("--t-max", type=float, default=120.0, help="Maximum HPF to include.")
    p.add_argument("--ylim", default="0,1", help="Y limits as 'low,high'.")
    p.add_argument("--bin-width", type=float, default=2.0, help="HPF bin width for trend lines.")
    p.add_argument(
        "--trend-smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for the trend line.",
    )
    return p.parse_args()


def _save(fig: plt.Figure, png_path: Path, pdf_path: Path, dpi: int) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _pretty_label(label: str) -> str:
    mapping = {
        "cep290_heterozygous": "Heterozygous",
        "cep290_homozygous": "Homozygous",
        "cep290_wildtype": "Wildtype",
    }
    return mapping.get(str(label), str(label).replace("_", " "))


def _rewrite_legends(fig: plt.Figure) -> None:
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_text(_pretty_label(text.get_text()))

    for leg in list(fig.legends):
        for text in leg.get_texts():
            text.set_text(_pretty_label(text.get_text()))


def main() -> None:
    args = _parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from analyze.utils.stats import normalize_arbitrary_feature
    from analyze.viz.plotting.feature_over_time import plot_feature_over_time
    from analyze.viz.styling.color_mapping_config import GENOTYPE_SUFFIX_COLORS
    from make_nwdb_phenotype_transition_static_plots import (
        LEGACY_AXIS_LABELSIZE,
        LEGACY_DPI,
        LEGACY_FIGSIZE_IN,
        LEGACY_TICK_LABELSIZE,
    )

    out_dir = Path(args.out_dir).resolve() / "figures" / str(args.figures_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = (_PROJECT_ROOT / args.data_dir).resolve()

    ylim_parts = [float(x.strip()) for x in str(args.ylim).split(",")]
    if len(ylim_parts) != 2:
        raise ValueError(f"--ylim must be two comma-separated floats, got {args.ylim!r}")
    ylim = (ylim_parts[0], ylim_parts[1])

    plt.rcParams.update(
        {
            "figure.dpi": LEGACY_DPI,
            "savefig.dpi": LEGACY_DPI,
            "xtick.labelsize": LEGACY_TICK_LABELSIZE,
            "ytick.labelsize": LEGACY_TICK_LABELSIZE,
            "axes.labelsize": LEGACY_AXIS_LABELSIZE,
        }
    )

    df = pd.read_csv(data_dir / "embryo_data_with_labels.csv", low_memory=False)
    df["curvature"] = normalize_arbitrary_feature(
        df["baseline_deviation_normalized"],
        low=0,
        high_percentile=100,
        clip=False,
    )
    df = df[df["predicted_stage_hpf"].between(float(args.t_min), float(args.t_max), inclusive="both")].copy()
    df = df[df["genotype"].isin(["cep290_heterozygous", "cep290_homozygous", "cep290_wildtype"])].copy()

    color_lookup = {
        "cep290_heterozygous": str(GENOTYPE_SUFFIX_COLORS["heterozygous"]),
        "cep290_homozygous": str(GENOTYPE_SUFFIX_COLORS["homozygous"]),
        "cep290_wildtype": str(GENOTYPE_SUFFIX_COLORS["wildtype"]),
    }

    variants = (
        ("without_trend", False),
        ("with_trend", True),
    )
    plot_specs = (
        ("heterozygous", ["cep290_heterozygous"]),
        ("wildtype", ["cep290_wildtype"]),
        ("homozygous", ["cep290_homozygous"]),
        ("het_vs_wt", ["cep290_heterozygous", "cep290_wildtype"]),
        ("het_vs_wt_vs_homo", ["cep290_heterozygous", "cep290_wildtype", "cep290_homozygous"]),
    )

    for plot_key, genotypes in plot_specs:
        plot_df = df[df["genotype"].isin(genotypes)].copy()
        for suffix, show_trend in variants:
            fig = plot_feature_over_time(
                plot_df,
                features="curvature",
                time_col="predicted_stage_hpf",
                id_col="embryo_id",
                color_by="genotype",
                color_lookup={k: v for k, v in color_lookup.items() if k in genotypes},
                show_individual=True,
                show_trend=bool(show_trend),
                show_error_band=False,
                trend_statistic="median",
                trend_smooth_sigma=float(args.trend_smooth_sigma),
                trend_linestyle="dashed",
                bin_width=float(args.bin_width),
                backend="matplotlib",
                ylim=ylim,
                title=None,
                legend_loc="outside",
            )
            fig.set_size_inches(*LEGACY_FIGSIZE_IN, forward=True)

            for ax in fig.axes:
                ax.set_xlabel("Hours post fertilization")
                ax.set_ylabel("Curvature")

            _rewrite_legends(fig)

            png_path = out_dir / f"curvature_{plot_key}_{suffix}.png"
            pdf_path = out_dir / f"curvature_{plot_key}_{suffix}.pdf"
            _save(fig, png_path, pdf_path, LEGACY_DPI)
            print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
