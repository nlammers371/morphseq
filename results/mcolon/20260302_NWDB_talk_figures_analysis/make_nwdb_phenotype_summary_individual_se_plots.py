"""
NWDB talk: phenotype summary plots with individual traces and API-native trend bands.

Generates three curvature plots in two variants each:
1) High_to_Low only
2) Low_to_High only
3) High_to_Low + Low_to_High overlay

Variants:
- with API error band
- without error band
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
        description="Create phenotype summary curvature plots with individual traces and SE bands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output folder (figures/ will be created inside).",
    )
    p.add_argument(
        "--figures-subdir",
        default="phenotype_transition_homozygous/summary_individual_api_errorbands_bin2",
        help="Subfolder inside figures/ for outputs.",
    )
    p.add_argument(
        "--data-dir",
        default="results/mcolon/20251229_cep290_phenotype_extraction/final_data",
        help="Directory containing embryo_data_with_labels.csv and embryo_cluster_labels.csv.",
    )
    p.add_argument("--t-min", type=float, default=24.0, help="Minimum HPF to include.")
    p.add_argument("--t-max", type=float, default=120.0, help="Maximum HPF to include.")
    p.add_argument("--ylim", default="0,1", help="Y limits as 'low,high'.")
    p.add_argument("--bin-width", type=float, default=2.0, help="HPF bin width for trend bands.")
    p.add_argument(
        "--trend-smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma for the mean trend.",
    )
    return p.parse_args()


def _save_plot(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _pretty_label(label: str) -> str:
    return str(label).replace("_", " ")


def main() -> None:
    args = _parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from analyze.utils.stats import normalize_arbitrary_feature
    from analyze.viz.plotting.feature_over_time import plot_feature_over_time
    from make_nwdb_phenotype_transition_static_plots import (
        LEGACY_AXIS_LABELSIZE,
        LEGACY_DPI,
        LEGACY_FIGSIZE_IN,
        LEGACY_TICK_LABELSIZE,
        PHENOTYPE_COLORS,
        _load_embryo_frames,
    )

    out_dir = Path(args.out_dir).resolve() / "figures" / str(args.figures_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = (_PROJECT_ROOT / args.data_dir).resolve()

    parts = [float(x.strip()) for x in str(args.ylim).split(",")]
    if len(parts) != 2:
        raise ValueError(f"--ylim must be two comma-separated floats, got {args.ylim!r}")
    ylim = (parts[0], parts[1])

    plt.rcParams.update(
        {
            "figure.dpi": LEGACY_DPI,
            "savefig.dpi": LEGACY_DPI,
            "xtick.labelsize": LEGACY_TICK_LABELSIZE,
            "ytick.labelsize": LEGACY_TICK_LABELSIZE,
            "axes.labelsize": LEGACY_AXIS_LABELSIZE,
        }
    )

    df = _load_embryo_frames(data_dir)
    df["curvature"] = normalize_arbitrary_feature(
        df["baseline_deviation_normalized"],
        low=0,
        high_percentile=100,
        clip=False,
    )
    df = df[df["predicted_stage_hpf"].between(float(args.t_min), float(args.t_max), inclusive="both")].copy()
    df = df[df["cluster_categories"].notna()].copy()
    df.loc[df["cluster_categories"] == "Intermediate", "cluster_categories"] = "Low_to_High"

    specs = (
        ("01", "High_to_Low", ("High_to_Low",)),
        ("02", "Low_to_High", ("Low_to_High",)),
        ("03", "High_to_Low__Low_to_High_overlay", ("High_to_Low", "Low_to_High")),
    )

    variants = (
        ("with_error_band", True),
        ("without_error_band", False),
    )
    legend_variants = (
        ("with_legend", True),
        ("without_legend", False),
    )

    for order, suffix, phenotypes in specs:
        plot_df = pd.concat(
            [df[df["cluster_categories"].astype(str) == phenotype].copy() for phenotype in phenotypes],
            ignore_index=True,
        )
        for variant_tag, show_error_band in variants:
            for legend_tag, show_legend in legend_variants:
                fig = plot_feature_over_time(
                    plot_df,
                    features="curvature",
                    time_col="predicted_stage_hpf",
                    id_col="embryo_id",
                    color_by="cluster_categories",
                    color_lookup={p: PHENOTYPE_COLORS[p] for p in phenotypes},
                    show_individual=True,
                    show_trend=True,
                    show_error_band=bool(show_error_band),
                    trend_statistic="median",
                    trend_smooth_sigma=float(args.trend_smooth_sigma),
                    trend_linestyle="dashed",
                    bin_width=float(args.bin_width),
                    error_type="iqr",
                    backend="matplotlib",
                    xlim=(float(args.t_min), float(args.t_max)),
                    ylim=ylim,
                    title="Curvature over hours post fertilization",
                )
                fig.set_size_inches(*LEGACY_FIGSIZE_IN, forward=True)
                for ax in fig.axes:
                    ax.set_xlabel("Hours post fertilization")
                    ax.set_ylabel("Curvature")
                    ax.set_title("Curvature over hours post fertilization")
                    leg = ax.get_legend()
                    if leg is not None:
                        if show_legend:
                            for text in leg.get_texts():
                                text.set_text(_pretty_label(text.get_text()))
                        else:
                            leg.remove()
                for leg in list(fig.legends):
                    if show_legend:
                        for text in leg.get_texts():
                            text.set_text(_pretty_label(text.get_text()))
                    else:
                        leg.remove()

                filename = (
                    f"{order}_summary_api_{variant_tag}__{suffix}"
                    f"__individual_less_smoothing__{legend_tag}.png"
                )
                out_path = out_dir / filename
                _save_plot(fig, out_path, LEGACY_DPI)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
