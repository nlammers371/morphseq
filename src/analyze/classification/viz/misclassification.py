from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.utils.binning import bins_in_time_window
from analyze.classification.engine.margins import coerce_margin_range
from analyze.classification.viz.utils import (
    _pretty_axis_label,
    validate_required_columns,
    validate_unique_embryo_x,
    validate_margin_range,
)



def _signed_margin_rgba(
    mean_margin: float,
    alpha: float,
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> tuple[float, float, float, float]:
    cmap = plt.cm.RdBu_r
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    base = cmap(norm(mean_margin))
    return (base[0], base[1], base[2], alpha)


def _compute_penetrance(
    embryo_df: pd.DataFrame,
    *,
    embryo_col: str,
    margin_col: str = "signed_margin",
) -> pd.DataFrame:
    """Compute per-embryo penetrance metrics."""
    rows: list[dict] = []
    for embryo_id, grp in embryo_df.groupby(embryo_col):
        rows.append(
            {
                embryo_col: str(embryo_id),
                "true_label": str(grp["true_label"].iloc[0]),
                "n_points": int(len(grp)),
                "mean_signed_margin": float(grp[margin_col].mean()),
                "abs_mean_signed_margin": float(np.abs(grp[margin_col].mean())),
                "min_signed_margin": float(grp[margin_col].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["true_label", "abs_mean_signed_margin"],
        ascending=[True, False],
    ).reset_index(drop=True)


def _draw_signed_margin_panel(
    ax,
    embryo_df: pd.DataFrame,
    penetrance_df: pd.DataFrame,
    *,
    genotype: str,
    embryo_col: str,
    x_col: str,
    margin_col: str,
    max_embryos: int,
    color_mode: str,
    discrete_class_lookup: Optional[dict[str, str]],
    discrete_class_colors: Optional[dict[str, str]],
    vmin: float,
    vmax: float,
) -> None:
    pen = penetrance_df[penetrance_df["true_label"].astype(str) == genotype].copy()
    pen = pen.sort_values("abs_mean_signed_margin", ascending=False).head(max_embryos)
    top_embryos = pen[embryo_col].astype(str).tolist()
    pen_lookup = pen.set_index(embryo_col)
    alphas = np.linspace(0.35, 0.9, len(top_embryos)) if top_embryos else []
    highlight_id = top_embryos[0] if top_embryos else None

    for alpha, embryo_id in zip(alphas, top_embryos):
        curve = embryo_df[embryo_df[embryo_col].astype(str) == embryo_id].sort_values(x_col)
        if curve.empty:
            continue

        mean_margin = float(pen_lookup.at[embryo_id, "mean_signed_margin"])
        if color_mode == "discrete" and discrete_class_lookup is not None:
            discrete_class = discrete_class_lookup.get(embryo_id)
            default_colors = {"__default__": "#B0B0B0"}
            color_map = discrete_class_colors or default_colors
            base_color = color_map.get(discrete_class or "", color_map.get("__default__", "#B0B0B0"))
            color_rgba = matplotlib.colors.to_rgba(base_color, alpha=alpha)
            if embryo_id == highlight_id:
                color_rgba = matplotlib.colors.to_rgba(base_color, alpha=0.98)
        else:
            color_rgba = _signed_margin_rgba(mean_margin, alpha, vmin=vmin, vmax=vmax)
            if embryo_id == highlight_id:
                color_rgba = _signed_margin_rgba(mean_margin, 0.98, vmin=vmin, vmax=vmax)

        linewidth = 2.8 if embryo_id == highlight_id else 1.6
        marker_size = 4.0 if embryo_id == highlight_id else 3.0

        ax.plot(
            curve[x_col].to_numpy(dtype=float),
            curve[margin_col].to_numpy(dtype=float),
            color=color_rgba,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            alpha=0.95,
        )

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.3, alpha=0.7, label="Decision boundary")
    ax.set_xlabel(_pretty_axis_label(x_col), fontsize=12)
    ax.set_ylabel("Signed margin", fontsize=12)
    ax.set_ylim([vmin, vmax])
    ax.grid(alpha=0.3)
    label = genotype.replace("_", " ")
    ax.set_title(f"{label} (n={len(top_embryos)})", fontsize=14, fontweight="bold")
    if top_embryos:
        ax.legend(loc="upper left", fontsize=9)


def _resolve_comparison_labels(
    df: pd.DataFrame,
    comparison_id: Optional[str],
    positive_label: Optional[str],
    negative_label: Optional[str],
) -> tuple[pd.DataFrame, str, str]:
    """Validate selector mode, filter df, and return (filtered_df, negative_label, positive_label).

    Exactly one selector mode must be used:
      - comparison_id mode: pass comparison_id, df must have a 'comparison_id' column.
      - explicit-label mode: pass both positive_label and negative_label, df must have
        'positive_label' and 'negative_label' columns (or 'true_label').

    Raises ValueError on any ambiguity or empty selection.
    """
    using_cid = comparison_id is not None
    using_labels = positive_label is not None or negative_label is not None

    if using_cid and using_labels:
        raise ValueError(
            "Provide either comparison_id or (positive_label, negative_label), not both."
        )
    if not using_cid and not using_labels:
        raise ValueError(
            "Provide either comparison_id or both positive_label and negative_label."
        )
    if using_labels and (positive_label is None or negative_label is None):
        raise ValueError(
            "Both positive_label and negative_label must be provided together."
        )

    if using_cid:
        if "comparison_id" not in df.columns:
            raise KeyError("df is missing 'comparison_id' column (required for comparison_id mode).")
        df = df[df["comparison_id"].astype(str) == comparison_id].copy()
        if df.empty:
            raise ValueError(f"No rows found for comparison_id={comparison_id!r}.")
        # Resolve labels from df columns if available, else parse from comparison_id
        if "positive_label" in df.columns and "negative_label" in df.columns:
            pos_vals = df["positive_label"].astype(str).unique()
            neg_vals = df["negative_label"].astype(str).unique()
            if len(pos_vals) != 1 or len(neg_vals) != 1:
                raise ValueError(
                    f"comparison_id={comparison_id!r} maps to multiple label pairs: "
                    f"positive={pos_vals.tolist()}, negative={neg_vals.tolist()}. "
                    "Filter to a single comparison before plotting."
                )
            resolved_pos = pos_vals[0]
            resolved_neg = neg_vals[0]
        else:
            parts = comparison_id.split("__vs__")
            if len(parts) != 2:
                raise ValueError(
                    f"Cannot parse labels from comparison_id={comparison_id!r} "
                    "(expected 'negative__vs__positive'). Add positive_label/negative_label "
                    "columns to df or use explicit-label mode."
                )
            resolved_neg, resolved_pos = parts
    else:
        # Explicit-label mode: filter to rows matching this ordered pair
        if "positive_label" in df.columns and "negative_label" in df.columns:
            df = df[
                (df["positive_label"].astype(str) == positive_label)
                & (df["negative_label"].astype(str) == negative_label)
            ].copy()
        elif "true_label" in df.columns:
            df = df[
                df["true_label"].astype(str).isin([positive_label, negative_label])
            ].copy()
        if df.empty:
            raise ValueError(
                f"No rows found for positive_label={positive_label!r}, "
                f"negative_label={negative_label!r}."
            )
        # If the df carries positive_label/negative_label columns, verify they match.
        # This catches swapped labels before they produce a silently wrong plot.
        if "positive_label" in df.columns and "negative_label" in df.columns:
            data_pos = df["positive_label"].astype(str).unique()
            data_neg = df["negative_label"].astype(str).unique()
            if len(data_pos) == 1 and len(data_neg) == 1:
                if data_pos[0] != positive_label or data_neg[0] != negative_label:
                    raise ValueError(
                        f"Requested positive_label={positive_label!r}, "
                        f"negative_label={negative_label!r}, "
                        f"but df contains positive_label={data_pos[0]!r}, "
                        f"negative_label={data_neg[0]!r}. "
                        "Labels will not be swapped automatically."
                    )
        resolved_pos = positive_label
        resolved_neg = negative_label

    return df, resolved_neg, resolved_pos


def plot_margin_trends(
    df: pd.DataFrame,
    *,
    comparison_id: Optional[str] = None,
    positive_label: Optional[str] = None,
    negative_label: Optional[str] = None,
    feature_col: str = "feature_set",
    feature_id: Optional[str] = None,
    margin_col: str = "truth_signed_margin",
    x_col: str = "time_bin_center",
    embryo_col: str = "embryo_id",
    color_mode: str = "continuous",
    discrete_class_lookup: Optional[dict[str, str]] = None,
    discrete_class_colors: Optional[dict[str, str]] = None,
    max_embryos: int = 30,
    vmin: float = -1.0,
    vmax: float = 1.0,
    time_window: Optional[tuple[float, float]] = None,
    output_path: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """Plot per-embryo signed-margin trajectories for a binary classification comparison.

    Exactly one selector mode must be used:

    **comparison_id mode** (pipeline-native)::

        plot_margin_trends(predictions_df, comparison_id="inj_ctrl__vs__pbx4_crispant")
        plot_margin_trends(predictions_df, comparison_id="inj_ctrl__vs__pbx4_crispant",
                           feature_id="vae")

    **Explicit-label mode** (ad hoc / scripting)::

        plot_margin_trends(df, positive_label="pbx4_crispant", negative_label="inj_ctrl")
        plot_margin_trends(df, positive_label="pbx4_crispant", negative_label="inj_ctrl",
                           feature_id="vae")

    Parameters
    ----------
    df : pd.DataFrame
        Predictions table. Extra columns are ignored. Required columns:

        - *embryo_col* (default ``"embryo_id"``)
        - *x_col* (default ``"time_bin_center"``) — x-axis values, one per embryo per x
        - *margin_col* (default ``"truth_signed_margin"``) — must be in [-1, 1]
        - ``true_label``, ``positive_label``/``negative_label``, or ``y_true``
          (used to assign each row to a group panel)

        For comparison_id mode: also ``comparison_id``.
        For explicit-label mode: also ``positive_label``/``negative_label`` or ``true_label``.

    comparison_id : str | None
        Filter df to this comparison. Mutually exclusive with positive/negative_label.
    positive_label : str | None
        Label for the positive class (right panel). Mutually exclusive with comparison_id.
    negative_label : str | None
        Label for the negative class (left panel). Mutually exclusive with comparison_id.
    feature_col : str
        Column name for the feature filter. Defaults to ``"feature_set"``. Ignored if
        *feature_id* is None.
    feature_id : str | None
        If provided, filter df to rows where ``df[feature_col] == feature_id`` after the
        comparison filter. Requires *feature_col* to be present in df.
    margin_col : str
        Column to use as the signed margin. Must exist and have values in [-1, 1].
    x_col : str
        Column to use as the x-axis. Must exist; one value per (embryo, x) pair.
    embryo_col : str
        Column identifying individual embryos.
    color_mode : {"continuous", "discrete"}
        How to color embryo lines. Continuous uses RdBu_r scaled to mean margin.
        Discrete uses discrete_class_lookup + discrete_class_colors.
    max_embryos : int
        Max embryos to draw per panel, ranked by abs mean margin.
    vmin, vmax : float
        Y-axis and colormap range. Defaults match the canonical [-1, 1] margin range.
    time_window : (t_min, t_max) | None
        Restrict to bins whose center satisfies t_min <= center <= t_max.
        Only valid when x_col is a time-like column (e.g. ``"time_bin_center"``).
    output_path : Path | None
        Save figure here at 220 dpi and close it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # --- 1. Selector validation and comparison filter ---
    df, resolved_neg, resolved_pos = _resolve_comparison_labels(
        df, comparison_id, positive_label, negative_label
    )

    # --- 2. Optional feature filter ---
    if feature_id is not None:
        if feature_col not in df.columns:
            raise KeyError(
                f"feature_col={feature_col!r} not found in df. "
                f"Available columns: {df.columns.tolist()}"
            )
        df = df[df[feature_col].astype(str) == feature_id].copy()
        if df.empty:
            raise ValueError(
                f"No rows remaining after feature filter: {feature_col}={feature_id!r}."
            )

    # --- 3. Validate required columns, margin range, and uniqueness ---
    validate_required_columns(
        df, [embryo_col, x_col, margin_col], context="plot_margin_trends"
    )
    validate_margin_range(df, margin_col)
    validate_unique_embryo_x(df, embryo_col=embryo_col, x_col=x_col)

    df = df.copy()

    # --- 4. Resolve true_label for panel assignment ---
    if "true_label" not in df.columns:
        if "positive_label" in df.columns and "negative_label" in df.columns and "y_true" in df.columns:
            df["true_label"] = np.where(
                df["y_true"].astype(int) == 1, df["positive_label"], df["negative_label"]
            )
        elif "y_true" in df.columns:
            df["true_label"] = np.where(
                df["y_true"].astype(int) == 1, resolved_pos, resolved_neg
            )
        else:
            raise ValueError(
                "df must have 'true_label', 'y_true', or "
                "'positive_label'+'negative_label'+'y_true' to assign rows to group panels."
            )

    # --- 5. Optional time window (x_col must be numeric/time-like) ---
    if time_window is not None:
        centers = df[x_col].astype(float)
        mask = bins_in_time_window(centers, centers, time_window)
        df = df[mask].copy()
        if df.empty:
            available = sorted(centers.unique())
            raise ValueError(
                f"time_window={time_window} excluded all rows. "
                f"Available {x_col} values: {available}"
            )

    penetrance_df = _compute_penetrance(df, embryo_col=embryo_col, margin_col=margin_col)

    # Render: negative panel left, positive panel right
    panels = [
        g for g in [resolved_neg, resolved_pos]
        if g in penetrance_df["true_label"].astype(str).unique()
    ]
    if not panels:
        raise ValueError(
            f"Neither '{resolved_neg}' nor '{resolved_pos}' found in true_label values."
        )

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, genotype in zip(axes, panels):
        _draw_signed_margin_panel(
            ax, df, penetrance_df,
            genotype=genotype,
            embryo_col=embryo_col,
            x_col=x_col,
            margin_col=margin_col,
            max_embryos=max_embryos,
            color_mode=color_mode if color_mode in ("continuous", "discrete") else "continuous",
            discrete_class_lookup=discrete_class_lookup,
            discrete_class_colors=discrete_class_colors,
            vmin=vmin,
            vmax=vmax,
        )

    neg_title = resolved_neg.replace("_", " ")
    pos_title = resolved_pos.replace("_", " ")
    fig.suptitle(f"Signed margin: {neg_title} vs {pos_title}", fontsize=15, fontweight="bold")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=220, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_wrongness_heatmap(
    embryo_predictions: pd.DataFrame,
    per_embryo_metrics: pd.DataFrame,
    output_dir: Path,
    *,
    row_order: str = "wrong_rate",
    cmap: str = "Reds",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = embryo_predictions.copy()
    df["is_wrong"] = (df["pred_class"] != df["true_class"]).astype(int)
    pivot = df.pivot_table(index="embryo_id", columns="time_bin", values="is_wrong", aggfunc="mean", fill_value=0)

    if row_order in per_embryo_metrics.columns:
        order = (
            per_embryo_metrics.sort_values(row_order, ascending=False)["embryo_id"].astype(str).tolist()
        )
        pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.2)))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel("time_bin")
    ax.set_ylabel("embryo_id")
    ax.set_title("Wrongness heatmap")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int).tolist(), rotation=90, fontsize=6)
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("is_wrong")
    fig.tight_layout()

    path = out / "wrongness_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_embryo_deep_dive(
    embryo_predictions: pd.DataFrame,
    embryo_id: str,
    output_dir: Path,
    *,
    class_colors: dict[str, str] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = embryo_predictions[embryo_predictions["embryo_id"].astype(str) == str(embryo_id)].copy()
    df = df.sort_values("time_bin")
    if df.empty:
        raise ValueError(f"No rows found for embryo_id={embryo_id}")

    if class_colors is None:
        classes = sorted(set(df["true_class"].astype(str).unique()) | set(df["pred_class"].astype(str).unique()))
        palette = plt.cm.tab10(np.linspace(0, 1, max(3, len(classes))))
        class_colors = {c: palette[i % len(palette)] for i, c in enumerate(classes)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    x = df["time_bin_center"].to_numpy(dtype=float)
    wrong = (df["pred_class"] != df["true_class"]).to_numpy(dtype=bool)
    for i, (_, row) in enumerate(df.iterrows()):
        c = class_colors.get(str(row["pred_class"]), "gray")
        hatch = "//" if wrong[i] else None
        ax1.bar(row["time_bin_center"], 1.0, width=0.35, color=c, alpha=0.9, hatch=hatch)
    ax1.set_yticks([])
    ax1.set_ylabel("pred_class")
    ax1.set_title(f"Embryo {embryo_id}: prediction timeline")

    ax2.plot(x, df["p_true"].to_numpy(dtype=float), marker="o", label="p_true")
    ax2.plot(x, df["p_pred"].to_numpy(dtype=float), marker="o", label="p_pred")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("probability")
    ax2.set_xlabel("time_bin_center")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path = out / f"embryo_{embryo_id}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_wrong_rate_distributions(
    per_embryo_metrics: pd.DataFrame,
    output_dir: Path,
    *,
    group_by: str = "true_class",
    show_flagged: bool = True,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = per_embryo_metrics.copy()
    groups = sorted(df[group_by].astype(str).unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [df[df[group_by].astype(str) == g]["wrong_rate"].to_numpy(dtype=float) for g in groups]
    ax.violinplot(vals, showmeans=True, showextrema=True)
    for i, g in enumerate(groups, start=1):
        y = vals[i - 1]
        ax.scatter(np.full_like(y, i, dtype=float), y, s=12, alpha=0.5)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_ylabel("wrong_rate")
    ax.set_title("Wrong-rate distributions")

    if show_flagged and "is_flagged" in df.columns:
        flagged = df[df["is_flagged"]]
        for i, g in enumerate(groups, start=1):
            y = flagged[flagged[group_by].astype(str) == g]["wrong_rate"].to_numpy(dtype=float)
            if len(y) > 0:
                ax.scatter(np.full_like(y, i, dtype=float), y, marker="D", s=18, color="crimson")

    fig.tight_layout()
    path = out / "wrong_rate_distributions.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_confusion_profile(
    embryo_predictions: pd.DataFrame,
    flagged_embryos: pd.DataFrame,
    output_dir: Path,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = embryo_predictions.copy()
    pred["is_wrong"] = pred["pred_class"] != pred["true_class"]
    pred = pred[pred["is_wrong"]]

    flagged_ids = set(flagged_embryos["embryo_id"].astype(str).tolist())
    pred["is_flagged"] = pred["embryo_id"].astype(str).isin(flagged_ids)

    agg = (
        pred.groupby(["true_class", "pred_class", "is_flagged"]).size().reset_index(name="n")
    )

    classes = sorted(pred["true_class"].astype(str).unique())
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, max(3, 2.8 * len(classes))), sharex=True)
    if len(classes) == 1:
        axes = [axes]

    for ax, true_class in zip(axes, classes):
        sub = agg[agg["true_class"].astype(str) == true_class]
        for flag_val, alpha in [(False, 0.5), (True, 0.9)]:
            s = sub[sub["is_flagged"] == flag_val]
            if s.empty:
                continue
            total = s["n"].sum()
            x = np.arange(len(s))
            y = s["n"].to_numpy(dtype=float) / float(total)
            labels = s["pred_class"].astype(str).tolist()
            ax.bar(x, y, alpha=alpha, label="flagged" if flag_val else "unflagged")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{true_class}\nfrac")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    fig.suptitle("Confusion profile by true class")
    fig.tight_layout()
    path = out / "confusion_profile_by_class.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_flagged_embryo_gallery(
    embryo_predictions: pd.DataFrame,
    flagged_embryos: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int = 20,
) -> list[Path]:
    out = Path(output_dir) / "embryo_deep_dives"
    out.mkdir(parents=True, exist_ok=True)

    ranked = flagged_embryos.sort_values("wrong_rate", ascending=False)["embryo_id"].astype(str).tolist()[:top_n]
    paths = []
    for embryo_id in ranked:
        try:
            paths.append(plot_embryo_deep_dive(embryo_predictions, embryo_id, out))
        except Exception:
            continue
    return paths
