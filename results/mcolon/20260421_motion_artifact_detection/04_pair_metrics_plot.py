"""
04_pair_metrics_plot.py
=======================
Regenerates the adjacent Z-pair metrics plot using only the two
metrics that proved useful: NCC and phase_shift_mag.

One panel per metric, all examples overlaid, colour-coded by label.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE    = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
OUT_DIR = BASE / "results/mcolon/20260421_motion_artifact_detection"
FIG_DIR = OUT_DIR / "figures"

pair_df = pd.read_csv(OUT_DIR / "pair_metrics.csv")

LABEL_COLORS = {
    "Bad Images":   "#D62728",
    "Okay Images":  "#FF7F0E",
    "Great Images": "#1F77B4",
}
LABEL_SHORT = {
    "Bad Images":   "Bad",
    "Okay Images":  "Okay",
    "Great Images": "Great",
}

fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor="#1a1a1a")
fig.subplots_adjust(hspace=0.35, left=0.07, right=0.97, top=0.90, bottom=0.08)

panels = [
    ("ncc",            "NCC",                   0.90,  "NCC < 0.90 threshold", True),
    ("phase_shift_mag","Phase shift (pixels)",   5.0,   "5 px threshold",       False),
]

examples = pair_df[["label", "well", "time_int"]].drop_duplicates()

for ax, (col, ylabel, thresh, thresh_label, higher_is_better) in zip(axes, panels):
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")

    seen_labels = set()
    for _, ex in examples.iterrows():
        sub = pair_df[
            (pair_df["well"]     == ex["well"]) &
            (pair_df["time_int"] == ex["time_int"])
        ].sort_values("z0")

        lbl   = ex["label"]
        color = LABEL_COLORS[lbl]
        short = LABEL_SHORT[lbl]
        legend_label = f"{short} — {ex['well']} t={int(ex['time_int'])}"

        lw   = 2.0
        ms   = 5
        zord = 3 if lbl == "Bad Images" else 2

        ax.plot(sub["z0"].values, sub[col].values,
                color=color, linewidth=lw, marker="o", markersize=ms,
                alpha=0.9, zorder=zord, label=legend_label)

    # Threshold line
    ax.axhline(thresh, color="#FF3333", linestyle="--",
               linewidth=1.2, alpha=0.8, label=thresh_label)

    ax.set_xlabel("Z pair  (z₀ → z₀+1,  each step = 50 µm)", fontsize=10, color="white")
    ax.set_ylabel(ylabel, fontsize=10, color="white")
    ax.set_xticks(range(14))
    ax.set_xticklabels([f"{z}→{z+1}" for z in range(14)],
                       fontsize=7, rotation=45, color="#aaaaaa")
    ax.grid(True, alpha=0.15, color="white")

    leg = ax.legend(fontsize=8, ncol=2, loc="lower left",
                    facecolor="#222222", edgecolor="#444444",
                    labelcolor="white", framealpha=0.9)

fig.suptitle(
    "Adjacent Z-pair metrics  |  Bad = red   Okay = orange   Great = blue",
    fontsize=11, color="white", y=0.95
)

out = FIG_DIR / "pair_metrics_final.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")
