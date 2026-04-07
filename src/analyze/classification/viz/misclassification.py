from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.utils.binning import bins_in_time_window



def _signed_margin_rgba(
    mean_margin: float,
    alpha: float,
    *,
    vmin: float = -0.5,
    vmax: float = 0.5,
) -> tuple[float, float, float, float]:
    cmap = plt.cm.RdBu_r
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    base = cmap(norm(mean_margin))
    return (base[0], base[1], base[2], alpha)


def _compute_penetrance(embryo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-embryo penetrance metrics from embryo_df with signed_margin column."""
    rows: list[dict] = []
    for embryo_id, grp in embryo_df.groupby("embryo_id"):
        grp = grp.sort_values("time_bin")
        rows.append(
            {
                "embryo_id": str(embryo_id),
                "true_label": str(grp["true_label"].iloc[0]),
                "n_time_bins": int(len(grp)),
                "mean_signed_margin": float(grp["signed_margin"].mean()),
                "abs_mean_signed_margin": float(np.abs(grp["signed_margin"].mean())),
                "min_signed_margin": float(grp["signed_margin"].min()),
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
    max_embryos: int,
    color_mode: str,
    discrete_class_lookup: Optional[dict[str, str]],
    discrete_class_colors: Optional[dict[str, str]],
    vmin: float,
    vmax: float,
) -> None:
    pen = penetrance_df[penetrance_df["true_label"].astype(str) == genotype].copy()
    pen = pen.sort_values("abs_mean_signed_margin", ascending=False).head(max_embryos)
    top_embryos = pen["embryo_id"].astype(str).tolist()
    pen_lookup = pen.set_index("embryo_id")
    alphas = np.linspace(0.35, 0.9, len(top_embryos)) if top_embryos else []
    highlight_id = top_embryos[0] if top_embryos else None

    for alpha, embryo_id in zip(alphas, top_embryos):
        curve = embryo_df[embryo_df["embryo_id"].astype(str) == embryo_id].sort_values("time_bin")
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
            curve["time_bin"].to_numpy(dtype=float),
            curve["signed_margin"].to_numpy(dtype=float),
            color=color_rgba,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            alpha=0.95,
        )

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.3, alpha=0.7, label="Decision boundary")
    ax.set_xlabel("Time (hpf)", fontsize=12)
    ax.set_ylabel("Signed margin", fontsize=12)
    ax.set_ylim([vmin, vmax])
    ax.grid(alpha=0.3)
    label = genotype.replace("_", " ")
    ax.set_title(f"{label} (n={len(top_embryos)})", fontsize=14, fontweight="bold")
    if top_embryos:
        ax.legend(loc="upper left", fontsize=9)


def plot_signed_margin_trends(
    embryo_df: pd.DataFrame,
    *,
    group1: str,
    group2: str,
    color_mode: str = "continuous",
    discrete_class_lookup: Optional[dict[str, str]] = None,
    discrete_class_colors: Optional[dict[str, str]] = None,
    max_embryos: int = 30,
    vmin: float = -0.5,
    vmax: float = 0.5,
    time_window: Optional[tuple[float, float]] = None,
    output_path: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """Plot per-embryo signed-margin trajectories for a binary classification comparison.

    Parameters
    ----------
    embryo_df : pd.DataFrame
        Two accepted shapes, detected by column presence:

        1. **Pre-computed** — must have columns:
           ``embryo_id``, ``time_bin`` (or ``time_bin_center``), ``signed_margin``,
           ``true_label``.

        2. **Raw predictions** — must have columns:
           ``embryo_id``, ``time_bin`` (or ``time_bin_center``), ``p_pos``, ``y_true``.
           ``signed_margin`` is computed as ``2 * p_pos - 1``; ``y_true`` (0/1) is
           mapped to ``true_label`` using *group1* (negative=0) and *group2* (positive=1).

    group1 : str
        Label for the negative class (y_true == 0).
    group2 : str
        Label for the positive class (y_true == 1).
    color_mode : {"continuous", "discrete"}
        ``"continuous"`` — embryo lines colored by mean signed margin on RdBu_r.
        ``"discrete"`` — lines colored by ``discrete_class_lookup``; falls back to
        continuous if lookup is None.
    discrete_class_lookup : dict[str, str] | None
        Maps ``embryo_id`` → class label string.  Used only when
        ``color_mode="discrete"``.
    discrete_class_colors : dict[str, str] | None
        Maps class label → hex color.  Falls back to ``"#B0B0B0"`` for unknown
        labels.  ``"__default__"`` key is used as the fallback color.
    max_embryos : int
        Maximum number of embryos to draw per panel (ranked by abs mean margin).
    vmin, vmax : float
        Colormap and y-axis range for signed margin.  Defaults are -0.5 / 0.5
        (the natural range when ``signed_margin = p_pos - 0.5``).
    time_window : (t_min, t_max) | None
        If provided, restricts which bins are used for penetrance computation
        AND which bins are shown in the trajectories.  A bin is included iff
        its center satisfies ``t_min <= center <= t_max`` (center-of-bin rule).
    output_path : Path | None
        If provided, saves the figure to this path at 220 dpi and closes it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = embryo_df.copy()

    # --- Resolve input shape ---
    if "signed_margin" in df.columns and "true_label" in df.columns:
        if "time_bin_center" not in df.columns:
            if "time_bin" in df.columns:
                df["time_bin_center"] = df["time_bin"].astype(float)
            else:
                raise ValueError("embryo_df must have a 'time_bin' or 'time_bin_center' column")
        if "time_bin" not in df.columns:
            df["time_bin"] = df["time_bin_center"]
    elif "p_pos" in df.columns and "y_true" in df.columns:
        if "time_bin_center" not in df.columns:
            if "time_bin" in df.columns:
                df["time_bin_center"] = df["time_bin"].astype(float)
            else:
                raise ValueError("embryo_df must have a 'time_bin' or 'time_bin_center' column")
        if "time_bin" not in df.columns:
            df["time_bin"] = df["time_bin_center"]
        df["signed_margin"] = 2.0 * df["p_pos"].astype(float) - 1.0
        df["true_label"] = np.where(df["y_true"].astype(int) == 1, group2, group1)
    else:
        raise ValueError(
            "embryo_df must have either ('signed_margin', 'true_label') columns "
            "or ('p_pos', 'y_true') columns."
        )

    # --- Apply time window (center-of-bin rule via shared utility) ---
    if time_window is not None:
        centers = df["time_bin_center"].astype(float)
        mask = bins_in_time_window(centers, centers, time_window)
        df = df[mask].copy()
        if df.empty:
            available = sorted(centers.unique())
            raise ValueError(
                f"time_window={time_window} excluded all rows. "
                f"Available time_bin_center values: {available}"
            )

    penetrance_df = _compute_penetrance(df)

    groups_present = [
        g for g in [group1, group2]
        if g in penetrance_df["true_label"].astype(str).unique()
    ]
    if not groups_present:
        raise ValueError(f"Neither '{group1}' nor '{group2}' found in embryo_df true_label values.")

    fig, axes = plt.subplots(
        1, len(groups_present),
        figsize=(8 * len(groups_present), 6),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    for ax, genotype in zip(axes, groups_present):
        _draw_signed_margin_panel(
            ax,
            df,
            penetrance_df,
            genotype=genotype,
            max_embryos=max_embryos,
            color_mode=color_mode if color_mode in ("continuous", "discrete") else "continuous",
            discrete_class_lookup=discrete_class_lookup,
            discrete_class_colors=discrete_class_colors,
            vmin=vmin,
            vmax=vmax,
        )

    title = f"Signed margin: {group1.replace('_', ' ')} vs {group2.replace('_', ' ')}"
    fig.suptitle(title, fontsize=15, fontweight="bold")
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
