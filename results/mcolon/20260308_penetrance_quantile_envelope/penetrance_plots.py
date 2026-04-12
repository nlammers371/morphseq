"""
penetrance_plots.py — Public plotting API for the WT quantile envelope penetrance pipeline.

Design contract:
- All functions accept ax=None (single-panel) or axes=None (multi-panel) and create
  their own figure/axes when None is passed.
- All functions return (fig, ax) or (fig, axes).
- No printing; no disk I/O.  Caller handles savefig / logging.
- Aesthetics: top/right spines off, grid(alpha=0.3), label fontsize=12, title fontsize=14 bold.

STANDARD_PALETTE fallback order is imported from src/analyze/viz/styling when available.
If the import fails (e.g. running outside the full environment), a built-in palette is used.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from src.analyze.viz.styling.color_mapping_config import GENOTYPE_SUFFIX_COLORS
    _PALETTE_FALLBACK = list(GENOTYPE_SUFFIX_COLORS.values())
except Exception:
    _PALETTE_FALLBACK = [
        "#2166AC", "#F7B267", "#B2182B", "#9467bd", "#808080",
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    ]

from smoothing import SmoothedCurveSelection, loess_smooth


# ---------------------------------------------------------------------------
# Internal aesthetic helper
# ---------------------------------------------------------------------------

def _style_ax(ax):
    """Apply standard aesthetics: remove top/right spines, add grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.title.set_fontsize(14)
    ax.title.set_fontweight("bold")


def _resolve_colors(group_order, colors):
    """Return a dict group → color, filling gaps from the fallback palette."""
    out = dict(colors) if colors else {}
    fallback = iter(_PALETTE_FALLBACK)
    for g in (group_order or []):
        if g not in out:
            out[g] = next(fallback, "#808080")
    return out


def _resolve_curve_display(x, y, *, mode: str, frac: float | None):
    """Return display y-values, optionally LOESS-smoothed."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if mode == "raw" or frac is None or len(x) < 3:
        return y
    if mode != "smoothed":
        raise ValueError(f"curve mode must be 'raw' or 'smoothed', got {mode!r}")
    return loess_smooth(x, y, frac)


def summarize_binary_penetrance(
    df: pd.DataFrame,
    *,
    group_col: str,
    bin_col: str,
    penetrant_col: str,
    unit_col: str,
    value_scale: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for (group, time_bin), grp in df.groupby([group_col, bin_col]):
        valid = grp.dropna(subset=[penetrant_col])
        if valid.empty:
            continue

        unit_flags = valid.groupby(unit_col)[penetrant_col].max().astype(float)
        n_units = int(len(unit_flags))
        if n_units == 0:
            continue

        penetrance = float(unit_flags.mean())
        se = math.sqrt(penetrance * (1.0 - penetrance) / n_units)
        scaled = unit_flags.to_numpy(dtype=float) * float(value_scale)
        rows.append(
            {
                "group": group,
                "time_bin": float(time_bin),
                "penetrance": penetrance * float(value_scale),
                "se": se * float(value_scale),
                "q25": float(np.quantile(scaled, 0.25)),
                "q75": float(np.quantile(scaled, 0.75)),
                "n_units": n_units,
                "n_penetrant": int(unit_flags.sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["group", "time_bin"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. WT envelope diagnostic
# ---------------------------------------------------------------------------

def plot_wt_envelope_diagnostic(
    wt_df: pd.DataFrame,
    df_env: pd.DataFrame,
    *,
    time_col: str,
    metric_col: str,
    embryo_col: str,
    bin_col: str = "time_bin",
    n_scatter: int = 5000,
    scatter_label: str = "WT frames",
    title: str = "WT Envelope Diagnostic",
    show_raw: bool = True,
    show_smoothed: bool = True,
    envelope_lower_col: str = "smoothed_low",
    envelope_upper_col: str = "smoothed_high",
    exclude_lower_col: str | None = None,
    exclude_upper_col: str | None = None,
    ax=None,
    figsize=(14, 8),
):
    """
    WT scatter + raw quantile dashes + smoothed envelope lines + unsupported spans.

    Parameters
    ----------
    wt_df : pd.DataFrame
        Raw WT frame data; must contain ``time_col`` and ``metric_col``.
    df_env : pd.DataFrame
        Envelope table; expected columns: [bin_col, raw_low, raw_high,
        smoothed_low, smoothed_high, supported].
    time_col, metric_col, embryo_col : str
        Column names in ``wt_df``.
    bin_col : str
        Time-bin column in ``df_env``.
    n_scatter : int
        Max scatter points (subsampled for speed).
    ax : matplotlib Axes or None
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sample = wt_df.sample(min(n_scatter, len(wt_df)), random_state=42)
    ax.scatter(sample[time_col], sample[metric_col], s=2, alpha=0.3, color="#aaaaaa", label=scatter_label)

    sup = df_env["supported"]
    tb_sup = df_env.loc[sup, bin_col]
    if show_raw:
        ax.plot(tb_sup, df_env.loc[sup, "raw_low"], "b--", lw=1, label="Raw 2.5%")
        ax.plot(tb_sup, df_env.loc[sup, "raw_high"], "r--", lw=1, label="Raw 97.5%")
        if exclude_lower_col is not None and exclude_lower_col in df_env.columns:
            low_ex = sup & df_env[exclude_lower_col].astype(bool)
            if low_ex.any():
                ax.scatter(
                    df_env.loc[low_ex, bin_col],
                    df_env.loc[low_ex, "raw_low"],
                    s=45,
                    color="#1f77b4",
                    edgecolors="black",
                    linewidths=0.5,
                    label="Excluded low-bin support",
                    zorder=4,
                )
        if exclude_upper_col is not None and exclude_upper_col in df_env.columns:
            high_ex = sup & df_env[exclude_upper_col].astype(bool)
            if high_ex.any():
                ax.scatter(
                    df_env.loc[high_ex, bin_col],
                    df_env.loc[high_ex, "raw_high"],
                    s=45,
                    color="#d62728",
                    edgecolors="black",
                    linewidths=0.5,
                    label="Excluded high-bin support",
                    zorder=4,
                )
    if show_smoothed:
        ax.plot(tb_sup, df_env.loc[sup, envelope_lower_col], "b-", lw=2, label="Envelope lower")
        ax.plot(tb_sup, df_env.loc[sup, envelope_upper_col], "r-", lw=2, label="Envelope upper")

    # Grey spans for unsupported bins
    unsup_bins = df_env.loc[~sup, bin_col]
    if len(unsup_bins) > 0:
        bin_width = float(df_env[bin_col].diff().dropna().median())
        first = True
        for ub in unsup_bins:
            label = "Unsupported bins" if first else "_nolegend_"
            ax.axvspan(ub - bin_width / 2, ub + bin_width / 2,
                       alpha=0.15, color="gray", label=label)
            first = False

    ax.set_xlabel(time_col, fontsize=12)
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Quantile smoother frac selection
# ---------------------------------------------------------------------------

def plot_quantile_smoother_selection(
    x,
    raw_low,
    raw_high,
    lower_result: SmoothedCurveSelection,
    upper_result: SmoothedCurveSelection,
    *,
    supported_mask=None,
    candidate_fracs=None,
    figsize=(16, 6),
):
    """
    1×2 subplots showing candidate LOESS curves for lower and upper quantiles.

    Parameters
    ----------
    x : array-like
        Full time-bin array (length N).
    raw_low, raw_high : array-like
        Raw quantile values (length N, NaN for unsupported bins).
    lower_result, upper_result : SmoothedCurveSelection
        Results from ``select_quantile_curve_smoother``.
    supported_mask : array-like of bool or None
        If None, derived from non-NaN in raw_low / raw_high.
    candidate_fracs : list or None
        If None, uses keys from result.candidate_curves.
    figsize : tuple

    Returns
    -------
    (fig, axes)  where axes has shape (2,)
    """
    x = np.asarray(x, dtype=float)
    raw_low = np.asarray(raw_low, dtype=float)
    raw_high = np.asarray(raw_high, dtype=float)

    if supported_mask is None:
        sup_low = ~np.isnan(raw_low)
        sup_high = ~np.isnan(raw_high)
    else:
        supported_mask = np.asarray(supported_mask, dtype=bool)
        sup_low = supported_mask
        sup_high = supported_mask

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Use non-NaN mask (matches what select_quantile_curve_smoother uses internally)
    # so that candidate_curves arrays align with tx.
    valid_low  = ~np.isnan(raw_low)
    valid_high = ~np.isnan(raw_high)

    panels = [
        ("Lower curve (2.5%)",  x[valid_low],  raw_low[valid_low],  lower_result),
        ("Upper curve (97.5%)", x[valid_high], raw_high[valid_high], upper_result),
    ]

    for ax, (title, tx, raw_vals, result) in zip(axes, panels):
        ax.plot(tx, raw_vals, "ko", ms=4, label="Raw quantile", zorder=3)

        fracs = sorted(result.candidate_curves.keys())
        cmap = plt.cm.cool
        for fi, frac in enumerate(fracs):
            sm = result.candidate_curves[frac]
            color = cmap(fi / max(len(fracs) - 1, 1))
            is_selected = frac == result.selected_frac
            lw = 2.5 if is_selected else 1.0
            zorder = 5 if is_selected else 2
            label = f"frac={frac}" + (" ← selected" if is_selected else "")
            ax.plot(tx, sm, color=color, lw=lw, zorder=zorder, label=label)

        if result.used_fallback:
            ax.set_title(f"{title}\n(fallback frac={result.selected_frac})",
                         fontsize=14, fontweight="bold")
        else:
            ax.set_title(title, fontsize=14, fontweight="bold")

        ax.set_xlabel("Time bin (hpf)", fontsize=12)
        ax.legend(fontsize=7, ncol=2)
        _style_ax(ax)

    fig.suptitle("LOESS Frac Selection (shape stability criterion)", y=1.01, fontsize=14)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3. Outside-rate by group (generic; replaces _fig_wt_het_calibration)
# ---------------------------------------------------------------------------

def plot_outside_rate_by_group(
    rate_df: pd.DataFrame,
    *,
    expected_rate=None,
    x_col: str = "time_bin",
    y_col: str = "outside_rate",
    group_col: str = "group",
    colors: dict | None = None,
    group_order=None,
    y_label: str = "Outside-envelope rate",
    ax=None,
    figsize=(10, 5),
):
    """
    Outside-envelope rate over time, one line per group.

    Parameters
    ----------
    rate_df : pd.DataFrame
        Must contain ``x_col``, ``y_col``, ``group_col``.
    expected_rate : float or None
        Draws a dotted horizontal reference line when given.
    x_col, y_col, group_col : str
    colors : dict group → hex, or None (auto-assigned from palette)
    group_order : list or None
    ax : matplotlib Axes or None
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if group_order is None:
        group_order = sorted(rate_df[group_col].unique())

    resolved = _resolve_colors(group_order, colors)

    if expected_rate is not None:
        ax.axhline(expected_rate, color="black", ls=":", lw=1.5,
                   label=f"Expected ({expected_rate:.3f})")

    for group in group_order:
        grp = rate_df[rate_df[group_col] == group].sort_values(x_col)
        if grp.empty:
            continue
        color = resolved.get(group, "#808080")
        ax.plot(grp[x_col], grp[y_col], "-o", color=color, ms=4, label=str(group))

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("Outside-Envelope Rate by Group", fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    ax.legend()
    _style_ax(ax)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Penetrance curves
# ---------------------------------------------------------------------------

def plot_penetrance_curves(
    df: pd.DataFrame,
    *,
    x_col: str = "time_bin",
    y_col: str = "penetrance",
    se_col: str = "se",
    band_lower_col: str = "q25",
    band_upper_col: str = "q75",
    group_col: str = "group",
    colors: dict | None = None,
    group_order=None,
    x_label: str | None = None,
    y_label: str = "Penetrance",
    curve_mode: str = "raw",
    curve_frac: float | None = None,
    band_mode: str = "se",
    show_band: bool = True,
    show_line: bool = True,
    show_points: bool = True,
    show_legend: bool = True,
    legend_loc: str = "upper left",
    legend_bbox_to_anchor=(1.01, 1.0),
    legend_fontsize: int = 9,
    smooth_se: bool = True,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    tick_labelsize: int = 10,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    ax=None,
    title: str = "",
    figsize=(12, 8),
):
    """
    Line + fill_between SE bands, one per group.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``x_col``, ``y_col``, ``group_col`` and any required band cols.
    x_col, y_col, se_col, group_col : str
    colors : dict group → hex, or None
    group_order : list or None
    ax : matplotlib Axes or None
    title : str
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if group_order is None:
        group_order = sorted(df[group_col].unique())

    resolved = _resolve_colors(group_order, colors)

    for group in group_order:
        grp = df[df[group_col] == group].sort_values(x_col)
        if grp.empty:
            continue
        color = resolved.get(group, "#808080")
        x_vals = grp[x_col].to_numpy(dtype=float)
        y_vals = grp[y_col].to_numpy(dtype=float)
        y_display = _resolve_curve_display(x_vals, y_vals, mode=curve_mode, frac=curve_frac)
        if show_points:
            ax.plot(x_vals, y_vals, "o", color=color, ms=4, alpha=0.8)
        if show_line:
            ax.plot(x_vals, y_display, color=color, lw=2.4, label=str(group))
        elif show_band:
            ax.plot(x_vals, y_display, color=color, lw=0.0, label=str(group))
        if show_band:
            if band_mode == "se":
                spread = grp[se_col].to_numpy(dtype=float)
                if smooth_se:
                    spread = _resolve_curve_display(x_vals, spread, mode=curve_mode, frac=curve_frac)
                lower = y_display - spread
                upper = y_display + spread
            elif band_mode == "iqr":
                lower = _resolve_curve_display(
                    x_vals,
                    grp[band_lower_col].to_numpy(dtype=float),
                    mode=curve_mode,
                    frac=curve_frac,
                )
                upper = _resolve_curve_display(
                    x_vals,
                    grp[band_upper_col].to_numpy(dtype=float),
                    mode=curve_mode,
                    frac=curve_frac,
                )
            else:
                raise ValueError(f"band mode must be 'se' or 'iqr', got {band_mode!r}")
            ax.fill_between(
                x_vals,
                lower,
                upper,
                color=color,
                alpha=0.18,
            )

    ax.set_xlabel(x_label or x_col, fontsize=label_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_title(title or "Penetrance by category over time", fontsize=title_fontsize, fontweight="bold")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif np.nanmax(df[y_col].to_numpy(dtype=float)) <= 1.5:
        ax.set_ylim(0, 1.03)
    else:
        ax.set_ylim(0, 103)
    ax.tick_params(axis="both", labelsize=tick_labelsize)
    if show_legend:
        ax.legend(
            bbox_to_anchor=legend_bbox_to_anchor,
            loc=legend_loc,
            fontsize=legend_fontsize,
            framealpha=0.9,
        )
    _style_ax(ax)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Penetrance heatmap
# ---------------------------------------------------------------------------

def plot_penetrance_heatmap(
    df: pd.DataFrame,
    *,
    x_col: str = "time_bin",
    y_col: str = "group",
    value_col: str = "penetrance",
    group_order=None,
    colorbar_label: str = "Penetrance",
    ax=None,
    title: str = "",
    figsize=(14, 6),
):
    """
    imshow heatmap (YlOrRd, vmin=0 vmax=1) of penetrance by group × time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``x_col``, ``y_col``, ``value_col``.
    x_col, y_col, value_col : str
    group_order : list or None
    ax : matplotlib Axes or None
    title : str
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    pivot = df.pivot(index=y_col, columns=x_col, values=value_col)
    if group_order is not None:
        pivot = pivot.reindex(index=[g for g in group_order if g in pivot.index])

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   origin="upper")
    plt.colorbar(im, ax=ax, label=colorbar_label)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c)}" for c in pivot.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_title(title or "Penetrance heatmap", fontsize=14, fontweight="bold")
    _style_ax(ax)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Embryo consistency histograms
# ---------------------------------------------------------------------------

def plot_embryo_consistency(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    value_col: str = "frac_penetrant",
    colors: dict | None = None,
    group_order=None,
    xlabel: str = "Frac penetrant",
    title: str = "Per-embryo penetrance consistency",
    figsize_per_panel=(5, 4),
):
    """
    Per-group histogram of per-embryo fraction-penetrant (dynamic grid, ≤3 cols).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``group_col`` and ``value_col``.
    group_col, value_col : str
    colors : dict group → hex, or None
    group_order : list or None
    figsize_per_panel : (w, h) per panel

    Returns
    -------
    (fig, axes)  where axes is 2D ndarray (squeeze=False)
    """
    if group_order is None:
        group_order = sorted(df[group_col].unique())

    groups = [g for g in group_order if g in df[group_col].unique()]
    n = len(groups)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols) if n > 0 else 1

    fw = figsize_per_panel[0] * ncols
    fh = figsize_per_panel[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh), squeeze=False)

    resolved = _resolve_colors(group_order, colors)

    for i, group in enumerate(groups):
        ax = axes[i // ncols][i % ncols]
        data = df.loc[df[group_col] == group, value_col].dropna()
        color = resolved.get(group, "#808080")
        ax.hist(data, bins=20, range=(0, 1), color=color, alpha=0.7)
        ax.set_title(f"{group} (n={len(data)})", fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("# embryos", fontsize=12)
        ax.set_xlim(0, 1)
        _style_ax(ax)

    # Hide unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 7. Scatter + penetrance panel layout (one column per group)
# ---------------------------------------------------------------------------

def plot_scatter_and_penetrance(
    df: pd.DataFrame,
    df_env: pd.DataFrame,
    *,
    time_col: str,
    metric_col: str,
    embryo_col: str,
    group_col: str,
    penetrant_col: str = "penetrant",
    bin_col: str = "time_bin",
    group_order=None,
    colors: dict | None = None,
    show_envelope: bool = True,
    envelope_lower_col: str = "smoothed_low",
    envelope_upper_col: str = "smoothed_high",
    envelope_supported_only: bool = True,
    upper_only: bool = False,
    show_penetrant_markers: bool = True,
    vline_hpf=None,
    n_scatter: int = 5000,
    scatter_color: str = "#444444",
    envelope_color: str = "#3a86ff",
    title: str = "",
    figsize_per_col=(7, 9),
    scatter_height_ratio: int = 2,
    legend_fontsize: int = 13,
    top_ylabel: str | None = None,
    overall_label: str = "Overall",
    bottom_label: str = "Penetrance (%)",
    penetrance_curve_mode: str = "raw",
    penetrance_curve_frac: float | None = None,
    penetrance_band_mode: str = "se",
    show_penetrance_band: bool = False,
    show_penetrance_line: bool = True,
    show_penetrance_points: bool = True,
):
    """
    Two-row multi-panel figure: scatter + envelope (top) and penetrance % (bottom).

    One column per group in ``group_order``. By default the same ``scatter_color``
    is used for all groups; pass ``colors`` for per-group colours.

    Top row
    -------
    - Circles (o): frames within the envelope (``penetrant == 0``).
    - X marks: frames outside the envelope (``penetrant == 1``).
    - Blue filled band + dashed lines: smoothed WT envelope from ``df_env``.
    - Optional vertical dashed line at ``vline_hpf``.

    Bottom row
    ----------
    - Line plot of embryo-level penetrance (%) over ``bin_col``.
      Penetrance = fraction of embryos that have ≥1 penetrant frame in a bin.

    Parameters
    ----------
    df : pd.DataFrame
        Frame-level data; must contain ``time_col``, ``metric_col``, ``embryo_col``,
        ``group_col``, ``penetrant_col``, ``bin_col``.
    df_env : pd.DataFrame
        Envelope table from ``compute_wt_envelope``; must contain ``bin_col``,
        ``smoothed_low``, ``smoothed_high``, ``supported``.
    time_col, metric_col, embryo_col, group_col : str
    penetrant_col : str
        Column containing 0/1/NaN penetrance flag.
    bin_col : str
        Time-bin column (used for penetrance aggregation and x-axis of bottom row).
    group_order : list or None
        Order of panels left → right.  If None, uses sorted unique values.
    colors : dict group → hex, or None
    show_envelope : bool
        Draw the WT envelope band and dashed bounds on each top panel.
    envelope_supported_only : bool
        If True, only draw bins marked ``supported``. If False, draw any bins
        with finite envelope values, allowing the smoothed envelope to continue
        through unsupported regions.
    upper_only : bool
        If True, only the upper bound line is drawn (lower bound suppressed).
        Use when penetrance is one-directional (e.g. deviation metrics where
        "too low" is not a phenotype).
    show_penetrant_markers : bool
        If True, penetrant frames shown as X; non-penetrant as o.
        If False, all frames shown as o (no classification colouring).
    vline_hpf : float or None
        Optional vertical dashed line (e.g. hybrid cutoff).
    n_scatter : int
        Max scatter points per group (subsampled when group is large).
    scatter_color : str
        Hex colour for scatter points (same for all groups).
    envelope_color : str
        Hex colour for the envelope fill and bounds.
    title : str
        Figure suptitle.
    figsize_per_col : (w, h)
        Figure size per column.
    scatter_height_ratio : int
        Height ratio of scatter row vs penetrance row (default 2).
    legend_fontsize : int
        Font size for legend entries on scatter panels (default 13).

    Returns
    -------
    (fig, axes)  where axes has shape (2, n_groups)
    """
    if group_order is None:
        group_order = sorted(df[group_col].dropna().unique())

    n_cols = len(group_order)
    fw = figsize_per_col[0] * n_cols
    fh = figsize_per_col[1]
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(fw, fh),
        gridspec_kw={"height_ratios": [scatter_height_ratio, 1]},
        squeeze=False,
    )

    if envelope_supported_only:
        env_mask = df_env["supported"].values.astype(bool)
    else:
        env_mask = np.isfinite(df_env[envelope_upper_col].to_numpy(dtype=float))
        if not upper_only:
            env_mask &= np.isfinite(df_env[envelope_lower_col].to_numpy(dtype=float))
    env_x = df_env.loc[env_mask, bin_col].values.astype(float)
    env_low = df_env.loc[env_mask, envelope_lower_col].values
    env_high = df_env.loc[env_mask, envelope_upper_col].values
    resolved = _resolve_colors(group_order, colors)

    for ci, group in enumerate(group_order):
        ax_top = axes[0, ci]
        ax_bot = axes[1, ci]

        gdf = df[df[group_col] == group]
        group_color = resolved.get(group, scatter_color)

        # ── Scatter panel ──────────────────────────────────────────────────
        # Subsample per group for speed
        if len(gdf) > n_scatter:
            gdf_plot = gdf.sample(n_scatter, random_state=42)
        else:
            gdf_plot = gdf

        if show_penetrant_markers:
            non_pen = gdf_plot[gdf_plot[penetrant_col] == 0]
            pen_pts = gdf_plot[gdf_plot[penetrant_col] == 1]

            ax_top.scatter(
                non_pen[time_col], non_pen[metric_col],
                s=10, alpha=0.15, color=group_color,
                marker="o", linewidths=0, label="Within bounds",
            )
            ax_top.scatter(
                pen_pts[time_col], pen_pts[metric_col],
                s=60, alpha=0.85, color=group_color,
                marker="X", edgecolors="black", linewidths=0.6,
                label="Outside bounds",
            )
        else:
            ax_top.scatter(
                gdf_plot[time_col], gdf_plot[metric_col],
                s=18, alpha=0.35, color=group_color,
                marker="o", linewidths=0,
            )

        # ── Envelope ───────────────────────────────────────────────────────
        if show_envelope and len(env_x) > 0:
            fill_low = np.zeros_like(env_low) if upper_only else env_low
            ax_top.fill_between(
                env_x, fill_low, env_high,
                alpha=0.12, color=envelope_color, zorder=1,
            )
            if not upper_only:
                ax_top.plot(env_x, env_low, "--", color=envelope_color, lw=1.5, alpha=0.7)
            ax_top.plot(env_x, env_high, "--", color=envelope_color, lw=1.5, alpha=0.7)

        # ── Vertical cutoff line ───────────────────────────────────────────
        if vline_hpf is not None:
            ax_top.axvline(vline_hpf, color="black", ls=":", lw=2, alpha=0.8)

        # ── Title: group name + embryo count + overall rate ─────────────────
        total_emb = gdf[embryo_col].nunique()
        pen_valid = gdf[penetrant_col].dropna()
        overall_pct = pen_valid.mean() * 100 if len(pen_valid) > 0 else float("nan")

        ax_top.set_title(
            f"{group}\n{total_emb} embryos | {overall_label}: {overall_pct:.1f}%",
            fontsize=14, fontweight="bold",
        )
        if ci == 0:
            ax_top.set_ylabel(top_ylabel or metric_col, fontsize=12)

        if show_penetrant_markers:
            ax_top.legend(loc="upper right", fontsize=legend_fontsize, framealpha=0.9)

        _style_ax(ax_top)

        # ── Penetrance curve (bottom) ──────────────────────────────────────
        pen_df = summarize_binary_penetrance(
            gdf,
            group_col=group_col,
            bin_col=bin_col,
            penetrant_col=penetrant_col,
            unit_col=embryo_col,
            value_scale=100.0,
        )
        pen_df = pen_df[pen_df["group"] == group].sort_values("time_bin")
        if not pen_df.empty:
            x_vals = pen_df["time_bin"].to_numpy(dtype=float)
            y_vals = pen_df["penetrance"].to_numpy(dtype=float)
            y_display = _resolve_curve_display(x_vals, y_vals, mode=penetrance_curve_mode, frac=penetrance_curve_frac)
            if show_penetrance_band:
                if penetrance_band_mode == "se":
                    spread = pen_df["se"].to_numpy(dtype=float)
                    lower = y_display - spread
                    upper = y_display + spread
                elif penetrance_band_mode == "iqr":
                    lower = _resolve_curve_display(
                        x_vals,
                        pen_df["q25"].to_numpy(dtype=float),
                        mode=penetrance_curve_mode,
                        frac=penetrance_curve_frac,
                    )
                    upper = _resolve_curve_display(
                        x_vals,
                        pen_df["q75"].to_numpy(dtype=float),
                        mode=penetrance_curve_mode,
                        frac=penetrance_curve_frac,
                    )
                else:
                    raise ValueError(
                        f"penetrance band mode must be 'se' or 'iqr', got {penetrance_band_mode!r}"
                    )
                ax_bot.fill_between(x_vals, lower, upper, color=group_color, alpha=0.18)
            if show_penetrance_points:
                ax_bot.plot(x_vals, y_vals, "o", color=group_color, ms=5, alpha=0.85)
            if show_penetrance_line:
                ax_bot.plot(
                    x_vals, y_display,
                    "-", color=group_color, lw=2.5,
                )

        if vline_hpf is not None:
            ax_bot.axvline(vline_hpf, color="black", ls=":", lw=2, alpha=0.8)

        ax_bot.set_xlabel("Hours Post Fertilization (hpf)", fontsize=12)
        if ci == 0:
            ax_bot.set_ylabel(bottom_label, fontsize=12)
        ax_bot.set_ylim(0, 103)
        _style_ax(ax_bot)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)

    fig.tight_layout()
    return fig, axes
