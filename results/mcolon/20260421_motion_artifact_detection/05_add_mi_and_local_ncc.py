"""
05_add_mi_and_local_ncc.py
==========================
Adds two new pair metrics to pair_metrics.csv:

  mutual_info      — MI between adjacent Z slices inside the embryo mask.
                     Uses joint intensity histogram (no linearity assumption).
                     Normalized to [0,1]: NMI = MI / sqrt(H(a)*H(b))

  local_ncc_min    — minimum NCC computed over a grid of tiles within the
                     embryo bounding box. Catches partial motion (one end
                     of embryo moves, other stays).

  local_ncc_std    — std of tile NCCs. High std = spatially non-uniform
                     motion, i.e. partial movement.

Saves updated pair_metrics_extended.csv and regenerates pair metrics figure
with NCC, phase_shift_mag, NMI, and local_ncc_min panels.

NOTE: loads Z stacks from ND2 (the only source — stacks are transient in
the pipeline). This is the last time we load raw ND2 data for this analysis.
After this we have everything we need in CSVs.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

BASE      = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
ND2_PATH  = BASE / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
MASKS_DIR = BASE / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
OUT_DIR   = BASE / "results/mcolon/20260421_motion_artifact_detection"
FIG_DIR   = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

LOOKUP_CSV = BASE / "docs/refactors/motion_blur_filtering_zstack/frame_nd2_lookup.csv"
TILE_SIZE  = 128   # px — tile side for local NCC grid
N_BINS     = 64    # histogram bins for MI

LABEL_COLORS = {
    "Bad Images":   "#D62728",
    "Okay Images":  "#FF7F0E",
    "Great Images": "#1F77B4",
}

# ---------------------------------------------------------------------------
# Load existing pair metrics
# ---------------------------------------------------------------------------
pair_df  = pd.read_csv(OUT_DIR / "pair_metrics.csv")
lookup   = pd.read_csv(LOOKUP_CSV)
examples = (
    pair_df[["label", "well", "time_int", "series", "date"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def normalized_mutual_info(a: np.ndarray, b: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Normalized Mutual Information between two 1-D pixel arrays.

    MI(a,b) = H(a) + H(b) - H(a,b)
    NMI     = MI / sqrt(H(a) * H(b))   ∈ [0, 1]

    H = Shannon entropy of the marginal/joint histogram.
    No linearity assumption — captures any statistical dependence.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    # Joint histogram
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max()) + 1e-9
    joint, _, _ = np.histogram2d(a, b, bins=n_bins, range=[[lo, hi], [lo, hi]])
    joint = joint / joint.sum()  # joint probability

    # Marginals
    p_a = joint.sum(axis=1)
    p_b = joint.sum(axis=0)

    def entropy(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)))

    ha  = entropy(p_a)
    hb  = entropy(p_b)
    hab = entropy(joint.ravel())

    mi  = ha + hb - hab
    denom = np.sqrt(ha * hb)
    return float(mi / denom) if denom > 1e-9 else 0.0


def local_ncc_over_tiles(
    s0: np.ndarray, s1: np.ndarray,
    mask: np.ndarray,
    tile_size: int = TILE_SIZE,
) -> tuple[float, float, int]:
    """
    Divide the embryo bounding box into tiles of `tile_size` px.
    For each tile that has >10% mask coverage, compute NCC between s0 and s1.

    Returns:
      ncc_min   — worst tile NCC (most motion)
      ncc_std   — std across tile NCCs (non-uniform motion flag)
      n_tiles   — number of valid tiles used
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.nan, np.nan, 0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    tile_nccs = []
    for ty in range(y0, y1, tile_size):
        for tx in range(x0, x1, tile_size):
            ty2 = min(ty + tile_size, y1)
            tx2 = min(tx + tile_size, x1)

            tm = mask[ty:ty2, tx:tx2]
            if tm.sum() < (tile_size * tile_size * 0.10):
                continue  # tile not sufficiently inside mask

            a = s0[ty:ty2, tx:tx2][tm].astype(np.float64)
            b = s1[ty:ty2, tx:tx2][tm].astype(np.float64)

            a -= a.mean(); b -= b.mean()
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                continue
            tile_nccs.append(float(np.dot(a, b) / (na * nb)))

    if not tile_nccs:
        return np.nan, np.nan, 0

    arr = np.array(tile_nccs)
    return float(arr.min()), float(arr.std()), len(arr)


# ---------------------------------------------------------------------------
# Main loop — load Z stacks, compute new metrics
# ---------------------------------------------------------------------------
new_cols = {
    "nmi":           [],
    "local_ncc_min": [],
    "local_ncc_std": [],
    "local_ncc_n":   [],
}

# We'll build a list of (well, time_int, z0) → new metrics
new_rows = {}   # key=(well, time_int, z0) → dict

print("Computing MI and local NCC from Z stacks...")

with nd2.ND2File(str(ND2_PATH)) as f:
    dask = f.to_dask()

    for _, ex in examples.iterrows():
        well     = str(ex["well"])
        t        = int(ex["time_int"])
        series   = int(ex["series"])
        label    = ex["label"]
        date     = str(ex["date"])

        print(f"  [{label}] {well} t={t}")

        stack = dask[t, series-1, :, :, :].compute().astype(np.float32)

        # Load mask
        mp = MASKS_DIR / f"{date}_{well}_ch00_t{t:04d}_masks_emnum_1.png"
        if not mp.exists():
            print(f"    no mask, skipping")
            continue
        mask = np.array(Image.open(mp)).astype(bool)

        for z in range(stack.shape[0] - 1):
            s0 = stack[z];   s1 = stack[z+1]

            # NMI — computed on mask pixels
            nmi = normalized_mutual_info(
                s0[mask].astype(np.float64),
                s1[mask].astype(np.float64)
            )

            # Local NCC
            lncc_min, lncc_std, n_tiles = local_ncc_over_tiles(s0, s1, mask)

            new_rows[(well, t, z)] = {
                "nmi":           nmi,
                "local_ncc_min": lncc_min,
                "local_ncc_std": lncc_std,
                "local_ncc_n":   n_tiles,
            }

            print(f"    z{z}→{z+1}  NCC={pair_df[(pair_df['well']==well)&(pair_df['time_int']==t)&(pair_df['z0']==z)]['ncc'].values[0]:.3f}"
                  f"  NMI={nmi:.3f}  local_ncc_min={lncc_min:.3f}  tiles={n_tiles}")

# ---------------------------------------------------------------------------
# Merge new metrics into pair_df
# ---------------------------------------------------------------------------
def lookup_new(row, col):
    key = (str(row["well"]), int(row["time_int"]), int(row["z0"]))
    return new_rows.get(key, {}).get(col, np.nan)

for col in ["nmi", "local_ncc_min", "local_ncc_std", "local_ncc_n"]:
    pair_df[col] = pair_df.apply(lambda r: lookup_new(r, col), axis=1)

pair_df.to_csv(OUT_DIR / "pair_metrics_extended.csv", index=False)
print(f"\nSaved pair_metrics_extended.csv")

# ---------------------------------------------------------------------------
# Figure: all tested metrics — complete record of what was tried
# ---------------------------------------------------------------------------
# Load rel_entropy per-Z from relative slice metrics (stack-level: mean across Z)
rel_path = OUT_DIR / "slice_metrics_relative.csv"
rel_per_pair = None
if rel_path.exists():
    rel_df = pd.read_csv(rel_path)
    rel_df["date"] = rel_df["date"].astype(str)
    # rel_entropy is per-slice not per-pair, but we align z index for plotting
    # We use z as the x axis (0..14) — for pairs we plot at z0
    rel_per_pair = rel_df[["label", "well", "time_int", "embryo", "z", "rel_entropy"]].copy()

# Each panel: (kind, col, src_df, ylabel, thresh, higher_good, thresh_lbl, status)
# kind="pair"  → x = z0 (0..13), data from pair_df
# kind="slice" → x = z  (0..14), data from rel_per_pair
# KEEPs first, then DROPs.
# local_ncc_min / local_ncc_std not shown — derivable from saved NCC grids during pipeline QC.
PANELS = [
    ("pair",  "ncc",             pair_df,      "NCC\n(global, mask)",        0.90, True,  "< 0.90",       "KEEP"),
    ("slice", "rel_entropy",     rel_per_pair, "Rel entropy\n(embryo − bg)", 0.0,  True,  "= 0 baseline", "KEEP"),
    ("pair",  "phase_shift_mag", pair_df,      "Phase shift (px)",           5.0,  False, "> 5 px",       "DROP"),
    ("pair",  "nmi",             pair_df,      "NMI\n(mutual info)",         0.30, True,  "< 0.30",       "DROP"),
    ("pair",  "ssim_score",      pair_df,      "SSIM",                       0.90, True,  "< 0.90",       "DROP"),
]

STATUS_COLORS = {"KEEP": "#2ecc71", "TESTING": "#f39c12", "DROP": "#e74c3c"}
SHORT = {"Bad Images": "Bad", "Okay Images": "Okay", "Great Images": "Great"}
examples_plot = pair_df[["label", "well", "time_int"]].drop_duplicates()

fig, axes = plt.subplots(len(PANELS), 1,
                         figsize=(14, 3.2 * len(PANELS)),
                         facecolor="#1a1a1a")
fig.subplots_adjust(hspace=0.50, left=0.08, right=0.97, top=0.96, bottom=0.03)

def style_ax(ax, ylabel, thresh, thresh_lbl, status, xlabel):
    ax.set_facecolor("#111111")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, alpha=0.12, color="white")
    ax.set_xlabel(xlabel, fontsize=8, color="#777777")
    sc = STATUS_COLORS[status]
    ax.set_ylabel(ylabel, fontsize=9, color="white")
    ax.text(0.99, 0.97, status, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=sc,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                      edgecolor=sc, linewidth=1.5))
    if thresh is not None:
        ax.axhline(thresh, color="#FF4444", linestyle="--",
                   linewidth=1.0, alpha=0.6, label=thresh_lbl)

for ax_i, (kind, col, src_df, ylabel, thresh, higher_good, thresh_lbl, status) in enumerate(PANELS):
    ax = axes[ax_i]

    if kind == "pair":
        style_ax(ax, ylabel, thresh, thresh_lbl, status, "Z pair  (z₀ → z₀+1)")
        if src_df is None:
            continue
        for _, ex in examples_plot.iterrows():
            sub = src_df[
                (src_df["well"]     == ex["well"]) &
                (src_df["time_int"] == ex["time_int"])
            ].sort_values("z0")
            if col not in sub.columns or sub[col].isna().all():
                continue
            lbl   = ex["label"]
            color = LABEL_COLORS[lbl]
            ax.plot(sub["z0"].values, sub[col].values,
                    color=color, linewidth=1.8, marker="o", markersize=4,
                    alpha=0.88, zorder=3 if "Bad" in lbl else 2,
                    label=f"{SHORT[lbl]} {ex['well']} t{int(ex['time_int'])}")
        ax.set_xticks(range(14))
        ax.set_xticklabels([f"{z}→{z+1}" for z in range(14)],
                           fontsize=6, rotation=40, color="#666666")

    else:  # kind == "slice"
        style_ax(ax, ylabel, thresh, thresh_lbl, status, "Z slice index")
        if src_df is None:
            continue
        for _, ex in src_df[["label", "well", "time_int"]].drop_duplicates().iterrows():
            sub = src_df[
                (src_df["well"]     == ex["well"]) &
                (src_df["time_int"] == ex["time_int"])
            ].sort_values("z")
            if col not in sub.columns or sub[col].isna().all():
                continue
            lbl   = ex["label"]
            color = LABEL_COLORS[lbl]
            ax.plot(sub["z"].values, sub[col].values,
                    color=color, linewidth=1.8, marker="o", markersize=4,
                    alpha=0.88, zorder=3 if "Bad" in lbl else 2,
                    label=f"{SHORT[lbl]} {ex['well']} t{int(ex['time_int'])}")
        ax.set_xticks(range(15))
        ax.set_xticklabels([f"Z{z}" for z in range(15)],
                           fontsize=6, rotation=40, color="#666666")

    ax.legend(fontsize=7, ncol=3, loc="lower left",
              facecolor="#1e1e1e", edgecolor="#444",
              labelcolor="white", framealpha=0.9)

fig.suptitle(
    "All tested pair/slice metrics  |  Bad=red   Okay=orange   Great=blue  |  "
    "Badge: KEEP / TESTING / DROP",
    fontsize=10, color="white", y=0.99
)

out = FIG_DIR / "pair_metrics_all_tested.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Quick summary: do NMI and local_ncc_min add signal?
# ---------------------------------------------------------------------------
print("\n=== Per-stack summary (mean over Z pairs) ===")
summary = (
    pair_df.groupby(["label", "well", "time_int"])
    [["ncc", "nmi", "local_ncc_min", "local_ncc_std"]]
    .agg(["mean", "min"])
)
summary.columns = ["_".join(c) for c in summary.columns]
print(summary.sort_index().to_string())
