"""
NWDB talk: static unfaded curvature plots bridging genotype and phenotype overlays.

Generates a small set of static PNGs with:
- unfaded background traces
- background variants: homozygous, High_to_Low, Low_to_High
- overlay variants: none, Low_to_High only, High_to_Low only, both orders
- separate trace-only plots for the selected Low_to_High and High_to_Low embryos
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEGACY_FIGSIZE_IN = (5.0, 4.5)
LEGACY_DPI = 100
LEGACY_TICK_LABELSIZE = 15
LEGACY_AXIS_LABELSIZE = 17

PHENOTYPE_COLORS = {
    "High_to_Low": "#E76FA2",
    "Low_to_High": "#2FB7B0",
    "Not Penetrant": "#3A3A3A",
}


@dataclass(frozen=True)
class OverlaySpec:
    key: str
    label: str
    embryo_id: str
    color_hex: str


@dataclass(frozen=True)
class BackgroundSpec:
    key: str
    label: str
    out_prefix: str
    color_by: str
    filter_col: str
    filter_value: str
    color_lookup: dict[str, str]


@dataclass(frozen=True)
class ComboSpec:
    order: int
    suffix: str
    overlays: tuple[OverlaySpec, ...]


@dataclass(frozen=True)
class SummaryPlotSpec:
    order: int
    suffix: str
    phenotypes: tuple[str, ...]


@dataclass(frozen=True)
class SummaryRenderVariant:
    order_offset: int
    suffix_tag: str
    bin_width: float
    trend_smooth_sigma: float
    show_individual: bool
    show_trend: bool
    show_error_band: bool


OVERLAYS = {
    "homozygous_reference": OverlaySpec(
        key="homozygous_reference",
        label="Homozygous_Reference",
        embryo_id="20251205_A03_e01",
        color_hex="#B2182B",
    ),
    "low_to_high": OverlaySpec(
        key="low_to_high",
        label="Low_to_High",
        embryo_id="20251106_H04_e01",
        color_hex=PHENOTYPE_COLORS["Low_to_High"],
    ),
    "high_to_low": OverlaySpec(
        key="high_to_low",
        label="High_to_Low",
        embryo_id="20251113_A02_e01",
        color_hex=PHENOTYPE_COLORS["High_to_Low"],
    ),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create static unfaded phenotype/genotype transition plots for NWDB talk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output folder (figures/ will be created inside).",
    )
    p.add_argument(
        "--figures-subdir",
        default="phenotype_transition_homozygous/static_plots",
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
    p.add_argument("--smooth-sigma", type=float, default=2.0, help="Trace smoothing sigma.")
    p.add_argument(
        "--summary-bin-width",
        type=float,
        default=3.0,
        help="Bin width for phenotype summary error-bar plots.",
    )
    p.add_argument(
        "--summary-trend-statistic",
        default="median",
        choices=["mean", "median"],
        help="Central tendency for phenotype summary error-bar plots.",
    )
    p.add_argument(
        "--summary-error-type",
        default="iqr",
        choices=["sd", "se", "iqr", "mad"],
        help="Error metric for phenotype summary error-bar plots.",
    )
    p.add_argument(
        "--summary-trend-smooth-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for dashed phenotype summary trends.",
    )
    p.add_argument(
        "--summary-trend-linestyle",
        default="dashed",
        choices=["solid", "dashed", "dotted", "-", "--", ":"],
        help="Linestyle for phenotype summary trend curves.",
    )
    return p.parse_args()


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _coalesce_columns(df: pd.DataFrame, dst: str, src: str) -> None:
    if dst not in df.columns or src not in df.columns:
        return
    if df[dst].isna().all() and (~df[src].isna()).any():
        df[dst] = df[src]


def _load_embryo_frames(data_dir: Path) -> pd.DataFrame:
    embryo_frames_path = data_dir / "embryo_data_with_labels.csv"
    embryo_labels_path = data_dir / "embryo_cluster_labels.csv"
    if not embryo_frames_path.exists():
        raise FileNotFoundError(f"Missing: {embryo_frames_path}")
    if not embryo_labels_path.exists():
        raise FileNotFoundError(f"Missing: {embryo_labels_path}")

    labels_raw = pd.read_csv(embryo_labels_path, usecols=["embryo_id", "cluster_categories"])
    labels_raw["embryo_id"] = labels_raw["embryo_id"].astype(str)
    labels_raw["cluster_categories"] = labels_raw["cluster_categories"].astype(str).str.strip()
    labels = labels_raw.drop_duplicates(subset=["embryo_id"], keep="first").copy()

    header = pd.read_csv(embryo_frames_path, nrows=0)
    requested = [
        "embryo_id",
        "experiment_date",
        "frame_index",
        "predicted_stage_hpf",
        "genotype",
        "baseline_deviation_normalized",
        "use_embryo_flag",
    ]
    existing = [c for c in requested if c in header.columns]
    df = pd.read_csv(embryo_frames_path, usecols=existing, low_memory=False)
    df["embryo_id"] = df["embryo_id"].astype(str)
    if "use_embryo_flag" in df.columns:
        use_flag = df["use_embryo_flag"]
        if use_flag.dtype == bool:
            df = df[use_flag].copy()
        else:
            df = df[use_flag.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])].copy()

    df["predicted_stage_hpf"] = _safe_float_series(df["predicted_stage_hpf"])
    df["baseline_deviation_normalized"] = _safe_float_series(df["baseline_deviation_normalized"])
    return df.merge(labels, on="embryo_id", how="left", validate="many_to_one")


def _gaussian_smooth(vals: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or vals.size < 3:
        return vals
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(vals.astype(float), sigma=sigma, mode="reflect")


def _overlay_xy(trace_df: pd.DataFrame, smooth_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    times = trace_df["predicted_stage_hpf"].to_numpy(dtype=float)
    vals = trace_df["curvature"].to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(vals)
    times = times[finite]
    vals = vals[finite]
    order = np.argsort(times)
    times = times[order]
    vals = _gaussian_smooth(vals[order], smooth_sigma)
    return times, vals


def _save_plot(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=LEGACY_DPI)
    plt.close(fig)


def _draw_overlays(ax, overlays: tuple[OverlaySpec, ...], overlay_frames: dict[str, pd.DataFrame], smooth_sigma: float) -> None:
    for overlay in overlays:
        trace_df = overlay_frames[overlay.key]
        xs, ys = _overlay_xy(trace_df, smooth_sigma)
        if xs.size == 0:
            continue
        line = ax.plot(
            xs,
            ys,
            color=overlay.color_hex,
            linewidth=2.8,
            solid_capstyle="round",
            zorder=10,
        )[0]
        line.set_path_effects(
            [
                pe.Stroke(linewidth=4.6, foreground="white"),
                pe.Normal(),
            ]
        )


def _render_combo_plot(
    bg_df: pd.DataFrame,
    bg_spec: BackgroundSpec,
    combo_overlays: tuple[OverlaySpec, ...],
    overlay_frames: dict[str, pd.DataFrame],
    out_path: Path,
    ylim: tuple[float, float],
    smooth_sigma: float,
    show_trend: bool = True,
) -> None:
    from analyze.viz.plotting.feature_over_time import plot_feature_over_time

    plt.rcParams.update(
        {
            "figure.dpi": LEGACY_DPI,
            "savefig.dpi": LEGACY_DPI,
            "xtick.labelsize": LEGACY_TICK_LABELSIZE,
            "ytick.labelsize": LEGACY_TICK_LABELSIZE,
            "axes.labelsize": LEGACY_AXIS_LABELSIZE,
        }
    )
    fig = plot_feature_over_time(
        bg_df,
        features="curvature",
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by=bg_spec.color_by,
        color_lookup=bg_spec.color_lookup,
        show_individual=True,
        show_trend=bool(show_trend),
        show_error_band=False,
        backend="matplotlib",
        ylim=ylim,
    )
    ax = fig.axes[0]
    ax.set_xlabel("Hours post fertilization")
    ax.set_ylabel("Curvature")
    leg = ax.get_legend()
    if leg is not None:
        ax.legend(
            handles=leg.legend_handles,
            labels=[t.get_text() for t in leg.get_texts()],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            bbox_transform=ax.transAxes,
            fontsize=leg.get_texts()[0].get_fontsize() if leg.get_texts() else 12,
            frameon=True,
            framealpha=0.9,
        )
    _draw_overlays(ax, combo_overlays, overlay_frames, smooth_sigma)
    _save_plot(fig, out_path)


def _trace_only_plot(
    overlays: tuple[OverlaySpec, ...],
    overlay_frames: dict[str, pd.DataFrame],
    out_path: Path,
    t_min: float,
    t_max: float,
    ylim: tuple[float, float],
    smooth_sigma: float,
) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": LEGACY_DPI,
            "savefig.dpi": LEGACY_DPI,
            "xtick.labelsize": LEGACY_TICK_LABELSIZE,
            "ytick.labelsize": LEGACY_TICK_LABELSIZE,
            "axes.labelsize": LEGACY_AXIS_LABELSIZE,
        }
    )
    fig, ax = plt.subplots(figsize=LEGACY_FIGSIZE_IN, dpi=LEGACY_DPI)
    ax.set_xlim(float(t_min), float(t_max))
    ax.set_ylim(*ylim)
    ax.set_xlabel("Hours post fertilization")
    ax.set_ylabel("Curvature")
    ax.grid(alpha=0.15, linewidth=0.7)
    for overlay in overlays:
        xs, ys = _overlay_xy(overlay_frames[overlay.key], smooth_sigma)
        if xs.size == 0:
            continue
        line = ax.plot(xs, ys, color=overlay.color_hex, linewidth=3.0, solid_capstyle="round")[0]
        line.set_path_effects([pe.Stroke(linewidth=4.8, foreground="white"), pe.Normal()])
    _save_plot(fig, out_path)


def _summary_plot(
    phenotype_frames: dict[str, pd.DataFrame],
    phenotypes: tuple[str, ...],
    out_path: Path,
    t_min: float,
    t_max: float,
    ylim: tuple[float, float],
    *,
    bin_width: float,
    trend_statistic: str,
    trend_smooth_sigma: float,
    trend_linestyle: str,
    show_individual: bool,
    show_trend: bool,
    show_error_band: bool,
    error_type: str,
) -> None:
    from analyze.viz.plotting.feature_over_time import plot_feature_over_time

    plt.rcParams.update(
        {
            "figure.dpi": LEGACY_DPI,
            "savefig.dpi": LEGACY_DPI,
            "xtick.labelsize": LEGACY_TICK_LABELSIZE,
            "ytick.labelsize": LEGACY_TICK_LABELSIZE,
            "axes.labelsize": LEGACY_AXIS_LABELSIZE,
        }
    )
    plot_df = pd.concat([phenotype_frames[p] for p in phenotypes if p in phenotype_frames], ignore_index=True)
    fig = plot_feature_over_time(
        plot_df,
        features="curvature",
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by="cluster_categories",
        color_lookup={p: PHENOTYPE_COLORS[p] for p in phenotypes},
        show_individual=bool(show_individual),
        show_trend=bool(show_trend),
        show_error_band=bool(show_error_band),
        trend_statistic=str(trend_statistic),
        trend_smooth_sigma=float(trend_smooth_sigma),
        trend_linestyle=str(trend_linestyle),
        bin_width=float(bin_width),
        error_type=str(error_type),
        backend="matplotlib",
        xlim=(float(t_min), float(t_max)),
        ylim=ylim,
        legend_loc="outside",
    )
    fig.set_size_inches(*LEGACY_FIGSIZE_IN, forward=True)
    for ax in fig.axes:
        ax.set_xlabel("Hours post fertilization")
        ax.set_ylabel("Curvature")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    for leg in list(fig.legends):
        leg.remove()
    _save_plot(fig, out_path)


def main() -> None:
    args = _parse_args()

    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
    from analyze.utils.stats import normalize_arbitrary_feature
    from analyze.viz.styling.color_mapping_config import GENOTYPE_COLORS

    out_dir = Path(args.out_dir).resolve() / "figures" / str(args.figures_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = (_PROJECT_ROOT / args.data_dir).resolve()

    parts = [float(x.strip()) for x in str(args.ylim).split(",")]
    if len(parts) != 2:
        raise ValueError(f"--ylim must be two comma-separated floats, got {args.ylim!r}")
    ylim = (parts[0], parts[1])

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

    overlay_frames: dict[str, pd.DataFrame] = {}
    for overlay in OVERLAYS.values():
        feat_df = df[df["embryo_id"].astype(str) == overlay.embryo_id].copy().sort_values("predicted_stage_hpf")
        if feat_df.empty:
            raise RuntimeError(f"No rows found in {args.t_min}-{args.t_max} HPF for {overlay.embryo_id}")
        overlay_frames[overlay.key] = feat_df

    phenotype_frames = {
        phenotype: df[df["cluster_categories"].astype(str) == phenotype].copy().sort_values("predicted_stage_hpf")
        for phenotype in ("High_to_Low", "Low_to_High")
    }

    backgrounds = (
        BackgroundSpec(
            key="homozygous",
            label="cep290_homozygous",
            out_prefix="homozygous_bg",
            color_by="genotype",
            filter_col="genotype",
            filter_value="cep290_homozygous",
            color_lookup={"cep290_homozygous": GENOTYPE_COLORS["cep290_homozygous"]},
        ),
        BackgroundSpec(
            key="high_to_low",
            label="High_to_Low",
            out_prefix="high_to_low_bg",
            color_by="cluster_categories",
            filter_col="cluster_categories",
            filter_value="High_to_Low",
            color_lookup={"High_to_Low": PHENOTYPE_COLORS["High_to_Low"]},
        ),
        BackgroundSpec(
            key="low_to_high",
            label="Low_to_High",
            out_prefix="low_to_high_bg",
            color_by="cluster_categories",
            filter_col="cluster_categories",
            filter_value="Low_to_High",
            color_lookup={"Low_to_High": PHENOTYPE_COLORS["Low_to_High"]},
        ),
    )

    combos = (
        ComboSpec(order=0, suffix="background_only", overlays=()),
        ComboSpec(order=1, suffix=f"low_to_high_only__{OVERLAYS['low_to_high'].embryo_id}", overlays=(OVERLAYS["low_to_high"],)),
        ComboSpec(order=2, suffix=f"high_to_low_only__{OVERLAYS['high_to_low'].embryo_id}", overlays=(OVERLAYS["high_to_low"],)),
        ComboSpec(
            order=3,
            suffix=f"low_to_high_then_high_to_low__{OVERLAYS['low_to_high'].embryo_id}__{OVERLAYS['high_to_low'].embryo_id}",
            overlays=(OVERLAYS["low_to_high"], OVERLAYS["high_to_low"]),
        ),
        ComboSpec(
            order=4,
            suffix=f"high_to_low_then_low_to_high__{OVERLAYS['high_to_low'].embryo_id}__{OVERLAYS['low_to_high'].embryo_id}",
            overlays=(OVERLAYS["high_to_low"], OVERLAYS["low_to_high"]),
        ),
        ComboSpec(
            order=5,
            suffix=f"homozygous_reference_only__{OVERLAYS['homozygous_reference'].embryo_id}",
            overlays=(OVERLAYS["homozygous_reference"],),
        ),
        ComboSpec(
            order=6,
            suffix=(
                f"homozygous_reference_then_low_to_high_then_high_to_low__"
                f"{OVERLAYS['homozygous_reference'].embryo_id}__"
                f"{OVERLAYS['low_to_high'].embryo_id}__"
                f"{OVERLAYS['high_to_low'].embryo_id}"
            ),
            overlays=(OVERLAYS["homozygous_reference"], OVERLAYS["low_to_high"], OVERLAYS["high_to_low"]),
        ),
    )

    print(f"Saving static plots to: {out_dir}")
    for bg_spec in backgrounds:
        bg_df = df[df[bg_spec.filter_col].astype(str) == bg_spec.filter_value].copy()
        if bg_df.empty:
            print(f"WARNING: no rows for background {bg_spec.key}, skipping")
            continue
        print(f"Background {bg_spec.key}: {bg_df['embryo_id'].nunique()} embryos")
        for combo in combos:
            if bg_spec.key != "homozygous" and any(o.key == "homozygous_reference" for o in combo.overlays):
                continue
            out_path = out_dir / f"{combo.order:02d}_background_unfaded_{bg_spec.out_prefix}__{combo.suffix}.png"
            _render_combo_plot(
                bg_df=bg_df,
                bg_spec=bg_spec,
                combo_overlays=combo.overlays,
                overlay_frames=overlay_frames,
                out_path=out_path,
                ylim=ylim,
                smooth_sigma=float(args.smooth_sigma),
            )
            print(f"  Saved: {out_path.name}")

        if bg_spec.key == "homozygous":
            manual_sequences = (
                ("06A", "homozygous_bg__red_then_cyan_then_magenta__step_a__red_only", (OVERLAYS["homozygous_reference"],), True),
                ("06B", "homozygous_bg__red_then_cyan_then_magenta__step_b__red_plus_cyan", (OVERLAYS["homozygous_reference"], OVERLAYS["low_to_high"]), True),
                ("06C", "homozygous_bg__red_then_cyan_then_magenta__step_c__red_plus_cyan_plus_magenta", (OVERLAYS["homozygous_reference"], OVERLAYS["low_to_high"], OVERLAYS["high_to_low"]), False),
                ("06D", "homozygous_bg__red_then_magenta_then_cyan__step_a__red_only", (OVERLAYS["homozygous_reference"],), False),
                ("06E", "homozygous_bg__red_then_magenta_then_cyan__step_b__red_plus_magenta", (OVERLAYS["homozygous_reference"], OVERLAYS["high_to_low"]), False),
                ("06F", "homozygous_bg__red_then_magenta_then_cyan__step_c__red_plus_magenta_plus_cyan", (OVERLAYS["homozygous_reference"], OVERLAYS["high_to_low"], OVERLAYS["low_to_high"]), True),
            )
            for prefix, suffix, overlays, show_trend in manual_sequences:
                out_path = out_dir / f"{prefix}_{suffix}.png"
                _render_combo_plot(
                    bg_df=bg_df,
                    bg_spec=bg_spec,
                    combo_overlays=overlays,
                    overlay_frames=overlay_frames,
                    out_path=out_path,
                    ylim=ylim,
                    smooth_sigma=float(args.smooth_sigma),
                    show_trend=show_trend,
                )
                print(f"  Saved: {out_path.name}")

    for order, overlay in enumerate(OVERLAYS.values(), start=7):
        out_path = out_dir / f"{order:02d}_trace_only_{overlay.label}__{overlay.embryo_id}.png"
        _trace_only_plot(
            overlays=(overlay,),
            overlay_frames=overlay_frames,
            out_path=out_path,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            ylim=ylim,
            smooth_sigma=float(args.smooth_sigma),
        )
        print(f"Saved: {out_path.name}")

    trace_only_sequences = (
        (
            "09A",
            (
                "trace_only_progression__red_only__"
                f"{OVERLAYS['homozygous_reference'].embryo_id}"
            ),
            (OVERLAYS["homozygous_reference"],),
        ),
        (
            "09B",
            (
                "trace_only_progression__red_plus_cyan_plus_magenta__"
                f"{OVERLAYS['homozygous_reference'].embryo_id}__"
                f"{OVERLAYS['low_to_high'].embryo_id}__"
                f"{OVERLAYS['high_to_low'].embryo_id}"
            ),
            (OVERLAYS["homozygous_reference"], OVERLAYS["low_to_high"], OVERLAYS["high_to_low"]),
        ),
    )
    for prefix, suffix, overlays in trace_only_sequences:
        out_path = out_dir / f"{prefix}_{suffix}.png"
        _trace_only_plot(
            overlays=overlays,
            overlay_frames=overlay_frames,
            out_path=out_path,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            ylim=ylim,
            smooth_sigma=float(args.smooth_sigma),
        )
        print(f"Saved: {out_path.name}")

    summary_specs = (
        SummaryPlotSpec(order=10, suffix="summary_error_band__High_to_Low", phenotypes=("High_to_Low",)),
        SummaryPlotSpec(order=11, suffix="summary_error_band__Low_to_High", phenotypes=("Low_to_High",)),
        SummaryPlotSpec(
            order=12,
            suffix="summary_error_band__High_to_Low__Low_to_High_overlay",
            phenotypes=("High_to_Low", "Low_to_High"),
        ),
    )
    summary_variants = (
        SummaryRenderVariant(
            order_offset=0,
            suffix_tag="",
            bin_width=float(args.summary_bin_width),
            trend_smooth_sigma=float(args.summary_trend_smooth_sigma),
            show_individual=False,
            show_trend=True,
            show_error_band=True,
        ),
        SummaryRenderVariant(
            order_offset=3,
            suffix_tag="__less_smoothing",
            bin_width=2.0,
            trend_smooth_sigma=1.0,
            show_individual=False,
            show_trend=True,
            show_error_band=True,
        ),
        SummaryRenderVariant(
            order_offset=6,
            suffix_tag="__minimal_smoothing",
            bin_width=1.5,
            trend_smooth_sigma=0.75,
            show_individual=False,
            show_trend=True,
            show_error_band=True,
        ),
        SummaryRenderVariant(
            order_offset=9,
            suffix_tag="__line_only",
            bin_width=float(args.summary_bin_width),
            trend_smooth_sigma=float(args.summary_trend_smooth_sigma),
            show_individual=False,
            show_trend=True,
            show_error_band=False,
        ),
        SummaryRenderVariant(
            order_offset=12,
            suffix_tag="__less_smoothing__line_only",
            bin_width=2.0,
            trend_smooth_sigma=1.0,
            show_individual=False,
            show_trend=True,
            show_error_band=False,
        ),
        SummaryRenderVariant(
            order_offset=15,
            suffix_tag="__minimal_smoothing__line_only",
            bin_width=1.5,
            trend_smooth_sigma=0.75,
            show_individual=False,
            show_trend=True,
            show_error_band=False,
        ),
        SummaryRenderVariant(
            order_offset=18,
            suffix_tag="__individual_plus_trend",
            bin_width=float(args.summary_bin_width),
            trend_smooth_sigma=float(args.summary_trend_smooth_sigma),
            show_individual=True,
            show_trend=True,
            show_error_band=False,
        ),
        SummaryRenderVariant(
            order_offset=21,
            suffix_tag="__individual_only",
            bin_width=float(args.summary_bin_width),
            trend_smooth_sigma=float(args.summary_trend_smooth_sigma),
            show_individual=True,
            show_trend=False,
            show_error_band=False,
        ),
    )
    for variant in summary_variants:
        for spec in summary_specs:
            out_order = spec.order + variant.order_offset
            out_suffix = f"{spec.suffix}{variant.suffix_tag}"
            out_path = out_dir / f"{out_order:02d}_{out_suffix}.png"
            _summary_plot(
                phenotype_frames=phenotype_frames,
                phenotypes=spec.phenotypes,
                out_path=out_path,
                t_min=float(args.t_min),
                t_max=float(args.t_max),
                ylim=ylim,
                bin_width=float(variant.bin_width),
                trend_statistic=str(args.summary_trend_statistic),
                trend_smooth_sigma=float(variant.trend_smooth_sigma),
                trend_linestyle=str(args.summary_trend_linestyle),
                show_individual=bool(variant.show_individual),
                show_trend=bool(variant.show_trend),
                show_error_band=bool(variant.show_error_band),
                error_type=str(args.summary_error_type),
            )
            print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    main()
