from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
