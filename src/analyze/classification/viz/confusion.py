"""Confusion matrix visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def plot_confusion(
    scores: pd.DataFrame,
    confusion: pd.DataFrame,
    *,
    feature_set: str | None = None,
    time_range: tuple[float, float] | None = None,
    backend: str = "matplotlib",
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Any:
    """Plot confusion matrices from a ClassificationAnalysis.

    Parameters
    ----------
    scores
        The scores DataFrame (used for time-axis reference).
    confusion
        The confusion DataFrame from ``layers["confusion"]``.
    feature_set
        Filter to a single feature set.
    time_range
        ``(lo, hi)`` to restrict time bins.
    backend
        ``"matplotlib"`` or ``"plotly"``.
    output_path
        If given, save the figure to this path.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    df = confusion.copy()
    if feature_set is not None:
        df = df[df["feature_set"] == feature_set]
    if time_range is not None:
        lo, hi = time_range
        df = df[(df["time_bin_center"] >= lo) & (df["time_bin_center"] <= hi)]

    if df.empty:
        raise ValueError("No confusion data after filtering.")

    # Aggregate across time bins: mean proportion
    agg = (
        df.groupby(["true_class", "predicted_class"])["proportion"]
        .mean()
        .reset_index()
    )
    classes = sorted(set(agg["true_class"]) | set(agg["predicted_class"]))
    n = len(classes)

    mat = np.zeros((n, n))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    for _, row in agg.iterrows():
        i = cls_to_idx[row["true_class"]]
        j = cls_to_idx[row["predicted_class"]]
        mat[i, j] = row["proportion"]

    fig, ax = plt.subplots(figsize=(max(4, n * 1.5), max(4, n * 1.5)))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Mean confusion (proportion)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] > 0.5 else "black", fontsize=9)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
