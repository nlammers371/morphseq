"""
02_relative_metrics.py
======================
Loads pre-saved slice_metrics.csv and recomputes only the background
(outside-all-masks) focus metrics per Z slice. Then joins them to produce
relative scores: embryo_metric / background_metric at the same Z.

This is the key normalization: removes global focus variation across Z so
that a blurry embryo stands out against its sharp neighbours, rather than
being hidden by a globally soft slice.

Outputs (appended columns saved to slice_metrics_relative.csv):
  bg_log_mean, bg_tenengrad, bg_lap_var, bg_entropy   (absolute background)
  rel_log_mean, rel_tenengrad, rel_lap_var, rel_entropy  (embryo / background)

Then regenerates the focus-curves figure using relative metrics.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import nd2
import scipy.ndimage as ndi
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(BASE))

ND2_PATH  = BASE / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
MASKS_DIR = BASE / "morphseq_playground/sam2_pipeline_files/exported_masks/20250912/masks"
OUT_DIR   = BASE / "results/mcolon/20260421_motion_artifact_detection"
FIG_DIR   = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

PIXEL_UM = 3.2308
Z_STEP_UM = 50.0

LABEL_COLORS = {
    "Bad Images":   "#B2182B",
    "Okay Images":  "#F7B267",
    "Great Images": "#2166AC",
}

# ---------------------------------------------------------------------------
# Load pre-saved metrics
# ---------------------------------------------------------------------------
slice_df = pd.read_csv(OUT_DIR / "slice_metrics.csv")
stack_df = pd.read_csv(OUT_DIR / "stack_metrics.csv")

print(f"Loaded slice_metrics: {len(slice_df)} rows")
print(f"Loaded stack_metrics: {len(stack_df)} rows")

# Unique (date, well, time_int, series) combos we need to process
examples = (
    slice_df[["label", "date", "well", "time_int", "series"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
print(f"\n{len(examples)} unique frame stacks to process for background metrics")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f64(img):
    return img.astype(np.float64)


def load_masks(date: str, well: str, time_int: int) -> dict[int, np.ndarray]:
    masks = {}
    emnum = 1
    while True:
        p = MASKS_DIR / f"{date}_{well}_ch00_t{time_int:04d}_masks_emnum_{emnum}.png"
        if not p.exists():
            break
        arr = np.array(Image.open(p)).astype(bool)
        if arr.any():
            masks[emnum] = arr
        emnum += 1
    return masks


def background_metrics_for_stack(
    stack: np.ndarray,
    all_masks: dict[int, np.ndarray],
    n_bg_samples: int = 50_000,
    rng_seed: int = 42,
) -> list[dict]:
    """
    Per-Z background metrics. Background = frame pixels outside all embryo masks,
    eroded by 10px to avoid edge contamination.
    Returns list of dicts (one per Z) with bg_log_mean, bg_tenengrad, bg_entropy.
    """
    rng = np.random.default_rng(rng_seed)

    combined = np.zeros(stack.shape[1:], dtype=bool)
    for m in all_masks.values():
        combined |= m
    background = ndi.binary_erosion(~combined, iterations=10)

    bg_ys, bg_xs = np.where(background)
    if len(bg_ys) > n_bg_samples:
        idx = rng.choice(len(bg_ys), n_bg_samples, replace=False)
        bg_ys, bg_xs = bg_ys[idx], bg_xs[idx]

    rows = []
    for z in range(stack.shape[0]):
        sl = stack[z]

        lap_abs = np.abs(cv2.Laplacian(_f64(sl), cv2.CV_64F, ksize=3))
        gx = cv2.Sobel(_f64(sl), cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(_f64(sl), cv2.CV_64F, 0, 1, ksize=3)

        bg_px  = sl[bg_ys, bg_xs].astype(np.float64)
        bg_lap = lap_abs[bg_ys, bg_xs]
        bg_ten = (gx**2 + gy**2)[bg_ys, bg_xs]

        hist, _ = np.histogram(bg_px, bins=256, range=(0, 65535))
        h = hist[hist > 0].astype(np.float64); h /= h.sum()
        ent = float(-np.sum(h * np.log2(h + 1e-12)))

        rows.append({
            "z":            z,
            "bg_log_mean":  float(bg_lap.mean()),
            "bg_tenengrad": float(bg_ten.mean()),
            "bg_entropy":   ent,
        })
    return rows


# ---------------------------------------------------------------------------
# Main: one pass per unique frame stack
# ---------------------------------------------------------------------------
all_bg_rows = []

with nd2.ND2File(str(ND2_PATH)) as nd2_file:
    dask_arr = nd2_file.to_dask()  # (T, P, Z, Y, X)

    for _, ex in examples.iterrows():
        date     = str(ex["date"])
        well     = str(ex["well"])
        t        = int(ex["time_int"])
        series   = int(ex["series"])
        label    = ex["label"]

        print(f"  [{label}] {well} t={t} — loading Z-stack for background...")
        stack = dask_arr[t, series - 1, :, :, :].compute().astype(np.float32)
        masks = load_masks(date, well, t)

        if not masks:
            print(f"    WARNING: no masks, skipping")
            continue

        bg_rows = background_metrics_for_stack(stack, masks)
        for r in bg_rows:
            r.update({"date": date, "well": well, "time_int": t})
        all_bg_rows.extend(bg_rows)
        print(f"    bg_log_mean range: "
              f"{min(r['bg_log_mean'] for r in bg_rows):.1f} – "
              f"{max(r['bg_log_mean'] for r in bg_rows):.1f}")

bg_df = pd.DataFrame(all_bg_rows)

# Coerce date to string in both frames before merging
slice_df["date"] = slice_df["date"].astype(str)
bg_df["date"]    = bg_df["date"].astype(str)

# ---------------------------------------------------------------------------
# Join background onto slice_df and compute relative metrics
# ---------------------------------------------------------------------------
merged = slice_df.merge(
    bg_df[["date", "well", "time_int", "z",
           "bg_log_mean", "bg_tenengrad", "bg_entropy"]],
    on=["date", "well", "time_int", "z"],
    how="left",
)

EPS = 1e-6
merged["rel_log_mean"]  = merged["log_mean"]  / (merged["bg_log_mean"]  + EPS)
merged["rel_tenengrad"] = merged["tenengrad"]  / (merged["bg_tenengrad"] + EPS)
merged["rel_entropy"]   = merged["entropy"]    - merged["bg_entropy"]   # difference: embryo minus background

merged.to_csv(OUT_DIR / "slice_metrics_relative.csv", index=False)
print(f"\nSaved slice_metrics_relative.csv ({len(merged)} rows)")

# ---------------------------------------------------------------------------
# Figure: relative focus curves
# ---------------------------------------------------------------------------
rel_metrics = [
    ("rel_log_mean",  "log_mean / bg_log_mean",  "Higher = embryo sharper than background"),
    ("rel_tenengrad", "tenengrad / bg_tenengrad", "Higher = embryo sharper than background"),
    ("rel_entropy",   "entropy − bg_entropy",     "Positive = embryo richer texture than background"),
]

fig, axes = plt.subplots(len(rel_metrics), 1, figsize=(13, 4 * len(rel_metrics)))

uid_cols = ["label", "well", "time_int", "embryo"]
examples_e = merged[uid_cols].drop_duplicates()

for ax, (col, ylabel, note) in zip(axes, rel_metrics):
    for _, ex in examples_e.iterrows():
        sub = merged[
            (merged["well"]     == ex["well"]) &
            (merged["time_int"] == ex["time_int"]) &
            (merged["embryo"]   == ex["embryo"])
        ].sort_values("z")

        color = LABEL_COLORS.get(ex["label"], "gray")
        ax.plot(sub["z"].values, sub[col].values,
                color=color, alpha=0.85, linewidth=1.8, marker="o", markersize=4,
                label=f"{ex['label'][:3]} {ex['well']} t{ex['time_int']}")

    ax.axhline(1.0 if "rel_" in col and col != "rel_entropy" else 0.0,
               color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="baseline")
    ax.axvline(7, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Z index (0=bottom, 14=top, home=7)")
    ax.set_title(f"{col}  —  {note}", fontsize=9)
    ax.grid(True, alpha=0.25)

    # Deduplicated legend
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=7, ncol=3, loc="upper left")

plt.suptitle(
    "Relative focus metrics: embryo normalised by same-Z background\n"
    "Bad=red  Okay=amber  Great=blue  |  dashed=baseline (ratio=1 or diff=0)",
    fontsize=10,
)
plt.tight_layout()
out_fig = FIG_DIR / "focus_curves_relative.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_fig.name}")

# ---------------------------------------------------------------------------
# Quick summary: mean relative metrics per label
# ---------------------------------------------------------------------------
print("\n=== Mean relative metrics per label (averaged over all Z slices) ===")
summary = (
    merged.groupby("label")[["rel_log_mean", "rel_tenengrad", "rel_entropy"]]
    .mean()
    .round(3)
)
print(summary)

print("\n=== Per-stack: mean rel_log_mean and min NCC (from stack_df) ===")
stack_rel = (
    merged.groupby(["label", "well", "time_int", "embryo"])
    [["rel_log_mean", "rel_tenengrad", "rel_entropy"]]
    .agg(["mean", "min"])
)
stack_rel.columns = ["_".join(c) for c in stack_rel.columns]
stack_rel = stack_rel.reset_index()

# Merge in ncc_min from stack_df
ncc_cols = ["well", "time_int", "embryo", "ncc_min", "bad_pair_frac_ncc", "max_phase_shift_px"]
ncc_cols = [c for c in ncc_cols if c in stack_df.columns]
stack_rel = stack_rel.merge(stack_df[ncc_cols], on=["well", "time_int", "embryo"], how="left")
print(stack_rel.sort_values("label").to_string(index=False))
