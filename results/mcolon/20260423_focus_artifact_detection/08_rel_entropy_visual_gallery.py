"""
08_rel_entropy_visual_gallery.py
=================================
Show the 2D representative images for all 10 embryos ranked by rel_entropy_mean
so we can visually judge what different entropy values look like.

Outputs:
  - 07_focus_output_tail/rel_entropy_gallery.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
LOOKUP_CSV = Path(
    "/net/trapnell/vol1/home/mdcolon/proj/morphseq/docs/refactors/motion_blur_filtering_zstack/frame_nd2_lookup.csv"
)
SUMMARY_CSV = BASE_DIR / "07_focus_output_tail" / "focus_tail_summary.csv"
OUT_DIR = BASE_DIR / "07_focus_output_tail"

CATEGORY_COLORS = {
    "Bad Images": "#d62728",
    "Okay Images": "#ff7f0e",
    "Great Images": "#2ca02c",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_CSV).sort_values("rel_entropy_mean")
    lookup = pd.read_csv(LOOKUP_CSV)

    merged = summary.merge(
        lookup[["well", "time_int", "image_path"]],
        on=["well", "time_int"],
        how="left",
    )

    n = len(merged)
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(merged.iterrows()):
        ax = axes[i]
        rank = i + 1  # 1 = worst (most negative), n = best (closest to 0)

        img_path = row.get("image_path")
        if pd.notna(img_path) and Path(img_path).exists():
            img = np.array(Image.open(img_path).convert("L"))
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "no image", ha="center", va="center", transform=ax.transAxes)

        # Large rank stamp in top-left corner
        ax.text(
            0.03, 0.97, f"#{rank}/{n}",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )

        category = str(row.get("category", ""))
        color = CATEGORY_COLORS.get(category, "#444444")

        # Label badge bottom-right
        ax.text(
            0.97, 0.03, category.replace(" Images", ""),
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            color="white", va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, alpha=0.85),
        )

        title = (
            f"{row['well']} t{int(row['time_int'])}\n"
            f"rel_entropy_mean = {row['rel_entropy_mean']:.3f}   "
            f"bad_frac = {row['bad_pair_frac_ncc']:.2f}"
        )
        ax.set_title(title, fontsize=9, color=color, fontweight="bold", pad=4)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Embryos ranked by rel_entropy_mean   |   #1 = most negative (worst) → #10 = closest to 0 (best)\n"
        "rel_entropy = embryo texture − background texture  (negative = bg richer, i.e. blurry embryo)\n"
        "Border/label color:  red = Bad,  orange = Okay,  green = Great",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])

    out = OUT_DIR / "rel_entropy_gallery.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved gallery -> {out}")


if __name__ == "__main__":
    main()
