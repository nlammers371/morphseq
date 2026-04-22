"""
03_ranked_metric_viz.py
=======================
One column per example (2 Bad, 2 Okay, 2 Great).

Each column (top → bottom):
  [A] Focus-stacked JPEG, cropped to embryo, with mask contour
  [B] 15 Z slices (3 rows × 5 cols), cropped to embryo, red border if NCC < 0.90
  [C] 5 ranked metric bars with value labels

Crop: square centred on embryo centroid, side = max(bbox_h, bbox_w) + 15% margin.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import nd2
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

BASE       = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
ND2_PATH   = BASE / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
MASKS_DIR  = BASE / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
IMAGES_DIR = BASE / "morphseq_playground/sam2_pipeline_files/raw_data_organized/20250912/images"
OUT_DIR    = BASE / "results/mcolon/20260421_motion_artifact_detection"
FIG_DIR    = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
EXAMPLES = [
    ("Bad",   "B10",  92,  79, "#D62728"),
    ("Bad",   "B10",  96,  79, "#D62728"),
    ("Okay",  "C04",  24,  30, "#FF7F0E"),
    ("Okay",  "C04", 111,  30, "#FF7F0E"),
    ("Great", "C04",  28,  30, "#1F77B4"),
    ("Great", "G09",  31,  71, "#1F77B4"),
]

# Ranked metrics: (key, stack_df column, short label, good=high?)
METRICS = [
    ("ncc_min",             "ncc_min",             "NCC MIN",          True),
    ("bad_pair_frac",       "bad_pair_frac_ncc",   "BAD PAIR FRAC",    False),
    ("longest_bad_run",     "longest_bad_ncc_run", "LONGEST BAD RUN",  False),
    ("max_phase_shift_px",  "max_phase_shift_px",  "PHASE SHIFT (px)", False),
]

NCC_THRESH = 0.90
CROP_MARGIN = 0.20   # extra fraction around bbox

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def norm01(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-9)

def get_crop_box(mask, margin=CROP_MARGIN):
    ys, xs = np.where(mask)
    cy, cx = ys.mean(), xs.mean()
    half = max(ys.max()-ys.min(), xs.max()-xs.min()) / 2 * (1 + margin)
    H, W = mask.shape
    y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half))
    x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half))
    # make square
    sh, sw = y1-y0, x1-x0
    side = max(sh, sw)
    cy2, cx2 = (y0+y1)//2, (x0+x1)//2
    y0 = max(0, cy2 - side//2); y1 = min(H, y0 + side)
    x0 = max(0, cx2 - side//2); x1 = min(W, x0 + side)
    return y0, y1, x0, x1

def crop(img, box):
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]

def add_contour(ax, mask_crop, color, lw=1.5):
    if mask_crop.any():
        ax.contour(mask_crop.astype(float), levels=[0.5],
                   colors=[color], linewidths=lw)

# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------
stack_df = pd.read_csv(OUT_DIR / "stack_metrics.csv")
pair_df  = pd.read_csv(OUT_DIR / "pair_metrics.csv")

# metric ranges for consistent bar scaling
metric_ranges = {}
for _, col, _, _ in METRICS:
    if col in stack_df.columns:
        v = stack_df[col].dropna()
        metric_ranges[col] = (float(v.min()), float(v.max()))

print("Loading Z stacks...")
stacks, masks, focused_imgs, crop_boxes = {}, {}, {}, {}

with nd2.ND2File(str(ND2_PATH)) as f:
    dask = f.to_dask()
    for label, well, t, series, color in EXAMPLES:
        key = (well, t)
        print(f"  {label} {well} t={t}")
        stacks[key]  = dask[t, series-1, :, :, :].compute().astype(np.float32)

        mp = MASKS_DIR / f"20250912_{well}_ch00_t{t:04d}_masks_emnum_1.png"
        masks[key] = np.array(Image.open(mp)).astype(bool) if mp.exists() else None

        jp = IMAGES_DIR / f"20250912_{well}" / f"20250912_{well}_ch00_t{t:04d}.jpg"
        focused_imgs[key] = np.array(Image.open(jp).convert("L")).astype(np.float32) if jp.exists() else None

        if masks[key] is not None:
            crop_boxes[key] = get_crop_box(masks[key])

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
N = len(EXAMPLES)
# Row heights: focused, Z-grid (3 rows of slices), metrics
# We use a nested gridspec per column

FIG_W = N * 3.6
FIG_H = 14.0

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#1a1a1a")

# Top-level: 1 row × N columns
col_gs = gridspec.GridSpec(
    1, N, figure=fig,
    left=0.01, right=0.99, top=0.94, bottom=0.02,
    wspace=0.06
)

for col_idx, (label, well, t, series, ex_color) in enumerate(EXAMPLES):
    key = (well, t)
    stack   = stacks[key]
    mask    = masks[key]
    focused = focused_imgs[key]
    box     = crop_boxes.get(key)

    # Per-column sub-gridspec: 3 sections
    #   row 0: focused image (tall)
    #   rows 1-3: Z slices (3 rows × 5 cols grid)
    #   row 4: metric bars
    inner = gridspec.GridSpecFromSubplotSpec(
        5, 1,
        subplot_spec=col_gs[col_idx],
        hspace=0.08,
        height_ratios=[2.5, 1, 1, 1, 1.8]
    )

    # ------------------------------------------------------------------
    # [A] Focus-stacked image
    # ------------------------------------------------------------------
    ax_f = fig.add_subplot(inner[0])
    ax_f.set_facecolor("#111")
    if focused is not None and box is not None:
        fc = crop(norm01(focused), box)
        mc = crop(mask, box)
        ax_f.imshow(fc, cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        add_contour(ax_f, mc, ex_color, lw=2)
    ax_f.set_xticks([]); ax_f.set_yticks([])
    for sp in ax_f.spines.values():
        sp.set_edgecolor(ex_color); sp.set_linewidth(2.5)
    ax_f.set_title(f"{label}\n{well}  t={t}",
                   fontsize=9, color=ex_color, fontweight="bold", pad=4)

    # ------------------------------------------------------------------
    # [B] Z slices: 3 rows × 5 cols (Z0-Z14, left→right, top→bottom)
    # NCC values for border colouring
    # ------------------------------------------------------------------
    pr = pair_df[(pair_df["well"]==well) & (pair_df["time_int"]==t)].sort_values("z0")
    ncc_vals = dict(zip(pr["z0"].astype(int), pr["ncc"].values)) if "ncc" in pr.columns else {}

    z_gs = gridspec.GridSpecFromSubplotSpec(
        3, 5,
        subplot_spec=gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=inner[1:4], hspace=0.0
        )[:],
        hspace=0.04, wspace=0.04
    )

    # flatten z_gs into a 15-element list
    z_axes = [fig.add_subplot(z_gs[r, c]) for r in range(3) for c in range(5)]

    for z in range(15):
        ax_z = z_axes[z]
        ax_z.set_facecolor("#111")
        if box is not None:
            sl  = crop(norm01(stack[z]), box)
            mc  = crop(mask, box)
            ax_z.imshow(sl, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            add_contour(ax_z, mc, ex_color, lw=0.8)

        ax_z.set_xticks([]); ax_z.set_yticks([])

        # Border colour: red if NCC to next slice is bad
        ncc = ncc_vals.get(z, None)
        if ncc is not None and ncc < NCC_THRESH:
            bc, blw = "#FF3333", 2.0
        else:
            bc, blw = "#333333", 0.5
        for sp in ax_z.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(blw)

        # Small NCC label underneath
        if ncc is not None:
            tc = "#FF3333" if ncc < NCC_THRESH else "#555555"
            ax_z.set_xlabel(f"{ncc:.2f}", fontsize=4.5, color=tc, labelpad=1)

        # Z index top
        ax_z.set_title(f"Z{z}", fontsize=4, color="#666", pad=1)

    # ------------------------------------------------------------------
    # [C] Metric bars
    # ------------------------------------------------------------------
    ax_m = fig.add_subplot(inner[4])
    ax_m.set_facecolor("#111")
    ax_m.set_xlim(0, 1); ax_m.set_ylim(0, 1)
    ax_m.set_xticks([]); ax_m.set_yticks([])
    for sp in ax_m.spines.values():
        sp.set_edgecolor("#333"); sp.set_linewidth(0.5)

    sr = stack_df[(stack_df["well"]==well) & (stack_df["time_int"]==t)]
    if len(sr):
        sr = sr.iloc[0]

    n_m = len(METRICS)
    bar_h = 0.10
    y_positions = np.linspace(0.88, 0.10, n_m)

    cmap_rg = matplotlib.colormaps["RdYlGn"]

    for m_i, (mkey, col, mlabel, good_high) in enumerate(METRICS):
        y = y_positions[m_i]
        if len(sr) and col in sr.index:
            val = float(sr[col])
            lo, hi = metric_ranges.get(col, (0, 1))
            span   = hi - lo if hi > lo else 1.0
            frac   = np.clip((val - lo) / span, 0, 1)
            bar_f  = frac if good_high else (1 - frac)
            color  = cmap_rg(bar_f)

            # background track
            ax_m.barh(y, 1.0, height=bar_h, left=0,
                      color="#2a2a2a", zorder=1, transform=ax_m.transAxes)
            # value bar
            ax_m.barh(y, bar_f, height=bar_h, left=0,
                      color=color, zorder=2, transform=ax_m.transAxes)

            # label (left) and value (right)
            ax_m.text(0.02, y + bar_h*0.6, mlabel,
                      transform=ax_m.transAxes, fontsize=7,
                      color="white", va="bottom", ha="left",
                      fontweight="bold")

            val_str = (f"{val:.3f}" if abs(val) < 10
                       else f"{val:.1f}" if abs(val) < 1000
                       else f"{val:.0f}")
            ax_m.text(0.98, y, val_str,
                      transform=ax_m.transAxes, fontsize=9,
                      color="white", va="center", ha="right", fontweight="bold")
        else:
            ax_m.text(0.5, y, f"{mlabel}: n/a",
                      transform=ax_m.transAxes, fontsize=6,
                      color="#555", va="center", ha="center")

# ---------------------------------------------------------------------------
# Figure title
# ---------------------------------------------------------------------------
fig.text(
    0.5, 0.97,
    "Z-stack quality survey  |  "
    "Top: focus-stacked output  |  "
    "Middle: 15 Z slices (red border = NCC < 0.90)  |  "
    "Bottom: ranked metrics (green = good, red = bad)",
    ha="center", va="top", fontsize=9, color="white"
)

out_path = FIG_DIR / "ranked_metric_comparison.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out_path}")
