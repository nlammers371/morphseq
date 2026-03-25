"""
NWDB talk: spawn-only transition penetrance story plots.

This script keeps the talk-facing exports in the NWDB analysis folder while
reusing the validated embryo-bin penetrance implementation from the local
penetrance results package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
_PEN_DIR = _PROJECT_ROOT / "results/mcolon/20260308_penetrance_quantile_envelope"

sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_PEN_DIR))

from analyze.utils.binning import add_time_bins
from _nwdb_transition_plot_style import PLOT_FIGSIZE_IN, style_transition_figure
from _plot_nwdb_genotype_classification_utils import save_figure
from calls import mark_penetrant
from config import (
    EMBRYO_BIN_AGG,
    EMBRYO_CALL_MODE,
    EMBRYO_COL,
    GENOTYPE_COL,
    HOMO_GENOTYPE,
    METRIC_NAME,
    MIN_WT_EMBRYOS_PER_BIN,
    PAIR_COL,
    PRESENTATION_CURVE_FRAC,
    PRESENTATION_CURVE_MODE,
    PRESENTATION_CURVE_SMOOTH_SE,
    TIME_BIN_WIDTH,
    TIME_COL,
    UPPER_BOUND_ONLY,
    WT_GENOTYPE,
)
from envelope import aggregate_embryo_bins, compute_wt_envelope
from penetrance_plots import plot_scatter_and_penetrance
from smoothing import loess_smooth
from summaries import compute_penetrance_by_group_and_time


PHENOTYPE_COLORS = {
    "Low_to_High": "#2FB7B0",
    "High_to_Low": "#E76FA2",
    "Transition_Combined": "#7FC97F",
    "WT_Reference": "#1f77b4",
}
STORY_CATEGORIES = ["Low_to_High", "High_to_Low"]
STORY_COMBINED_GROUP = "Transition_Combined"
WT_REFERENCE_GROUP = "WT_Reference"
STORY_PAIR_VALUE = "cep290_spawn"
STORY_TIME_MIN = 24.0
STORY_TIME_MAX = 110.0
STORY_CURVE_FRAC = float(PRESENTATION_CURVE_FRAC) * 2.0
STORY_LINEWIDTH = 3.0
STORY_COMBINED_LINEWIDTH = 4.2
STORY_LINESTYLE = "-"
STORY_COMBINED_LINESTYLE = "--"


def _load_story_frame_df() -> pd.DataFrame:
    data_path = _PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    df = pd.read_csv(data_path, low_memory=False)
    df = add_time_bins(df, time_col=TIME_COL, bin_width=TIME_BIN_WIDTH)

    pair_norm = df[PAIR_COL].fillna("none").astype(str).str.strip().str.lower()
    spawn_mask = pair_norm.isin({STORY_PAIR_VALUE.lower(), "none", ""})
    df = df.loc[spawn_mask].copy()
    df = df.loc[df[TIME_COL].between(STORY_TIME_MIN, STORY_TIME_MAX, inclusive="both")].copy()
    df.loc[df["cluster_categories"] == "Intermediate", "cluster_categories"] = "Low_to_High"
    return df


def _build_story_penetrance_summary(embryo_bin_df: pd.DataFrame) -> pd.DataFrame:
    homo_story = embryo_bin_df[
        (embryo_bin_df[GENOTYPE_COL] == HOMO_GENOTYPE)
        & (embryo_bin_df["cluster_categories"].isin(STORY_CATEGORIES))
    ].copy()

    pen_story = compute_penetrance_by_group_and_time(
        homo_story,
        "cluster_categories",
        unit_col=EMBRYO_COL,
        count_col_name="n_embryos",
    )

    pooled = homo_story.copy()
    pooled["story_group"] = STORY_COMBINED_GROUP
    pooled_pen = compute_penetrance_by_group_and_time(
        pooled,
        "story_group",
        unit_col=EMBRYO_COL,
        count_col_name="n_embryos",
    )

    out = pd.concat([pen_story, pooled_pen], ignore_index=True)
    for col in ["penetrance", "se", "q25", "q75"]:
        out[col] = out[col].astype(float) * 100.0
    return out.sort_values(["group", "time_bin"]).reset_index(drop=True)


def _resolve_curve_display(x, y, *, mode: str, frac: float | None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if mode == "raw" or frac is None or len(x) < 3:
        return y
    if mode != "smoothed":
        raise ValueError(f"curve mode must be 'raw' or 'smoothed', got {mode!r}")
    return loess_smooth(x, y, frac)


def _plot_story_curves(
    df: pd.DataFrame,
    *,
    group_order: list[str],
    title: str,
    show_points: bool,
    show_band: bool,
    show_legend: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_IN)

    for group in group_order:
        grp = df[df["group"] == group].sort_values("time_bin")
        if grp.empty:
            continue
        color = PHENOTYPE_COLORS[group]
        x_vals = grp["time_bin"].to_numpy(dtype=float)
        y_vals = grp["penetrance"].to_numpy(dtype=float)
        y_disp = _resolve_curve_display(
            x_vals,
            y_vals,
            mode=PRESENTATION_CURVE_MODE,
            frac=STORY_CURVE_FRAC,
        )
        is_combined = group == STORY_COMBINED_GROUP

        if show_band and not is_combined:
            spread = grp["se"].to_numpy(dtype=float)
            if PRESENTATION_CURVE_SMOOTH_SE:
                spread = _resolve_curve_display(
                    x_vals,
                    spread,
                    mode=PRESENTATION_CURVE_MODE,
                    frac=STORY_CURVE_FRAC,
                )
            ax.fill_between(x_vals, y_disp - spread, y_disp + spread, color=color, alpha=0.18)
        if show_points and not is_combined:
            ax.plot(x_vals, y_vals, "o", color=color, ms=4.5, alpha=0.9)
        ax.plot(
            x_vals,
            y_disp,
            color=color,
            lw=STORY_COMBINED_LINEWIDTH if is_combined else STORY_LINEWIDTH,
            linestyle=STORY_COMBINED_LINESTYLE if is_combined else STORY_LINESTYLE,
            label=group,
        )

    ax.set_xlim(STORY_TIME_MIN, STORY_TIME_MAX)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Hours Post Fertilization (hpf)")
    ax.set_ylabel("Embryo-level penetrance (%)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    style_transition_figure(fig)
    return fig


def _render_story_curve(
    pen_story: pd.DataFrame,
    *,
    out_png: Path,
    out_pdf: Path,
    group_order: list[str],
    title: str,
    show_points: bool,
    show_band: bool,
    show_legend: bool,
) -> None:
    fig = _plot_story_curves(
        pen_story[pen_story["group"].isin(group_order)].copy(),
        group_order=group_order,
        title=title,
        show_points=show_points,
        show_band=show_band,
        show_legend=show_legend,
    )
    tight_rect = (0.0, 0.0, 0.84, 1.0) if show_legend else None
    save_figure(fig, out_png, out_pdf, tight_layout_rect=tight_rect, use_tight_layout=True)


def main() -> None:
    out_dir = _HERE / "figures" / "spawn_transition_penetrance_story"
    supplemental_dir = out_dir / "supplemental"
    out_dir.mkdir(parents=True, exist_ok=True)
    supplemental_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spawn-only 24-120 hpf story dataset...")
    frame_df = _load_story_frame_df()
    print(f"  Frames: {len(frame_df):,}")
    print(f"  Embryos: {frame_df[EMBRYO_COL].nunique():,}")
    print(f"  Pair values: {sorted(frame_df[PAIR_COL].dropna().astype(str).unique())}")

    embryo_bin_df = aggregate_embryo_bins(frame_df, agg=EMBRYO_BIN_AGG)
    wt_story = embryo_bin_df[embryo_bin_df[GENOTYPE_COL] == WT_GENOTYPE].copy()
    embryo_env, _, _ = compute_wt_envelope(
        wt_story,
        min_units=MIN_WT_EMBRYOS_PER_BIN,
        unit_label="embryo-bin summary",
    )
    embryo_bin_df = mark_penetrant(embryo_bin_df, embryo_env, call_mode=EMBRYO_CALL_MODE)
    homo_story = embryo_bin_df[
        (embryo_bin_df[GENOTYPE_COL] == HOMO_GENOTYPE)
        & (embryo_bin_df["cluster_categories"].isin(STORY_CATEGORIES))
    ].copy()
    wt_reference = embryo_bin_df[embryo_bin_df[GENOTYPE_COL] == WT_GENOTYPE].copy()
    pen_story = _build_story_penetrance_summary(embryo_bin_df)

    main_specs = [
        (
            "01_transition_penetrance_overlay__step1_low_to_high_trend",
            ["Low_to_High"],
            "Spawn-only homozygous transition penetrance: Low_to_High",
        ),
        (
            "02_transition_penetrance_overlay__step2_add_high_to_low_trend",
            ["Low_to_High", "High_to_Low"],
            "Spawn-only homozygous transition penetrance: add High_to_Low",
        ),
        (
            "03_transition_penetrance_overlay__step3_add_combined_trend",
            ["Low_to_High", "High_to_Low", STORY_COMBINED_GROUP],
            "Spawn-only homozygous transition penetrance: add pooled transition",
        ),
    ]
    for stem, group_order, title in main_specs:
        _render_story_curve(
            pen_story,
            out_png=out_dir / f"{stem}.png",
            out_pdf=out_dir / f"{stem}.pdf",
            group_order=group_order,
            title=title,
            show_points=False,
            show_band=False,
            show_legend=False,
        )
        print(f"  Saved: {stem}.png/.pdf")

    supplemental_specs = [
        (
            "10_transition_penetrance_overlay__step1_low_to_high_points_se",
            ["Low_to_High"],
            "Spawn-only homozygous penetrance: Low_to_High",
            False,
        ),
        (
            "11_transition_penetrance_overlay__step2_add_high_to_low_points_se",
            ["Low_to_High", "High_to_Low"],
            "Spawn-only homozygous transition penetrance",
            False,
        ),
        (
            "12_transition_penetrance_overlay__step3_add_combined_points_se",
            ["Low_to_High", "High_to_Low", STORY_COMBINED_GROUP],
            "Spawn-only homozygous transition penetrance",
            False,
        ),
        (
            "13_transition_penetrance_overlay__all_groups_points_se_no_legend",
            ["Low_to_High", "High_to_Low", STORY_COMBINED_GROUP],
            "Spawn-only homozygous transition penetrance",
            False,
        ),
        (
            "14_transition_penetrance_overlay__all_groups_points_se_with_legend",
            ["Low_to_High", "High_to_Low", STORY_COMBINED_GROUP],
            "Spawn-only homozygous transition penetrance",
            True,
        ),
    ]
    for stem, group_order, title, show_legend in supplemental_specs:
        _render_story_curve(
            pen_story,
            out_png=supplemental_dir / f"{stem}.png",
            out_pdf=supplemental_dir / f"{stem}.pdf",
            group_order=group_order,
            title=title,
            show_points=True,
            show_band=True,
            show_legend=show_legend,
        )
        print(f"  Saved: supplemental/{stem}.png/.pdf")

    scatter_story = pd.concat(
        [
            wt_reference.assign(display_group=WT_REFERENCE_GROUP),
            homo_story.assign(display_group=homo_story["cluster_categories"].astype(str)),
        ],
        ignore_index=True,
    )
    fig, _ = plot_scatter_and_penetrance(
        scatter_story,
        embryo_env,
        time_col=TIME_COL,
        metric_col=METRIC_NAME,
        embryo_col=EMBRYO_COL,
        group_col="display_group",
        group_order=[WT_REFERENCE_GROUP, "Low_to_High", "High_to_Low"],
        colors=PHENOTYPE_COLORS,
        upper_only=UPPER_BOUND_ONLY,
        title="Spawn-only embryo-bin summaries + penetrance by transition phenotype",
        figsize_per_col=(7, 9),
        envelope_lower_col="smoothed_low",
        envelope_upper_col="smoothed_high",
        envelope_supported_only=False,
        top_ylabel=f"{METRIC_NAME} ({EMBRYO_BIN_AGG} per embryo/bin)",
        overall_label="Overall embryo-bin penetrant",
        bottom_label="Embryo-level penetrance (%)",
        penetrance_curve_mode=PRESENTATION_CURVE_MODE,
        penetrance_curve_frac=PRESENTATION_CURVE_FRAC,
        show_penetrance_band=True,
        show_penetrance_line=True,
        show_penetrance_points=True,
    )
    save_figure(
        fig,
        supplemental_dir / "20_scatter_penetrance_by_transition_category_spawn_24_120_homo.png",
        supplemental_dir / "20_scatter_penetrance_by_transition_category_spawn_24_120_homo.pdf",
    )
    print("  Saved: supplemental/20_scatter_penetrance_by_transition_category_spawn_24_120_homo.png/.pdf")

    print("\nDone. Story figures saved to:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
