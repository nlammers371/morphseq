"""
run_loeo_logistic_benchmark.py
==============================
Leave-one-experiment-out (LOEO) benchmark for the three logistic label-transfer
modes (A=global, B=per-bin/image, C=per-bin/embryo) x two rollups (mean, margin).

Outputs
-------
plots/loeo_logistic_balanced_acc.png       — balanced accuracy per mode×rollup across folds
plots/loeo_logistic_macro_f1.png           — macro F1 per mode×rollup across folds
plots/loeo_logistic_f1_by_class.png        — per-class F1 heat map (mode×rollup)
plots/loeo_logistic_f1_by_timebin.png      — F1 by 4-hpf bin for each mode×rollup (image-level)
loeo_logistic_results.csv                  — all metrics, one row per (fold × mode × rollup)
loeo_logistic_image_predictions.csv        — image-level predictions with true labels and time bin
"""

from __future__ import annotations

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from logistic_label_transfer import run_logistic_label_transfer

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)
OUT_DIR = os.path.join(os.path.dirname(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MIN_HPF, MAX_HPF, BIN_WIDTH = 30.0, 48.0, 4.0
LABEL_ORDER = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
MODES = ["A", "B", "C"]
ROLLUPS = ["mean", "margin"]
PRED_COLS = [f"predicted_label_{m}_{r}" for m in MODES for r in ROLLUPS]

MODE_LABELS = {"A": "Global (A)", "B": "Per-bin/image (B)", "C": "Per-bin/embryo (C)"}
ROLLUP_LABELS = {"mean": "mean", "margin": "margin-wt"}

COLORS = {
    "A_mean":   "#2166AC",
    "A_margin": "#6BAED6",
    "B_mean":   "#D6604D",
    "B_margin": "#F4A582",
    "C_mean":   "#4DAC26",
    "C_margin": "#B8E186",
}

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
data = pd.read_csv(DATA_PATH, low_memory=False)
feat_cols = [c for c in data.columns if c.startswith("z_mu_b_")]
data = data.dropna(subset=["cluster_categories"]).copy()

experiments = sorted(data["experiment_id"].unique())
print(f"Experiments: {experiments}")
print(f"Total embryos: {data['embryo_id'].nunique()}, images: {len(data)}")

# ── LOEO loop ─────────────────────────────────────────────────────────────────
fold_records = []      # one dict per fold × mode × rollup
image_records = []     # image-level predictions for time-bin analysis

for held_out in experiments:
    print(f"\n── Fold: hold out {held_out} ──")
    ref = data[data["experiment_id"] != held_out].copy()
    qry = data[data["experiment_id"] == held_out].copy()

    if qry["cluster_categories"].isna().all():
        print("  No labeled query embryos — skipping")
        continue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_logistic_label_transfer(ref, qry, feat_cols,
                                             bin_width=BIN_WIDTH)

    summary = result["embryo_label_transfer_summary"]
    img_preds = result["image_predictions"]

    # true labels at embryo level
    true_embryo = (
        qry.groupby("embryo_id")["cluster_categories"].first().reset_index()
    )
    merged = summary.merge(
        true_embryo, left_on="query_embryo_id", right_on="embryo_id", how="inner"
    )
    y_true = merged["cluster_categories"]

    for mode in MODES:
        for rollup in ROLLUPS:
            col = f"predicted_label_{mode}_{rollup}"
            if col not in merged.columns:
                continue
            y_pred = merged[col]
            bal = balanced_accuracy_score(y_true, y_pred)
            mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            raw = (y_pred == y_true).mean()
            per_class = f1_score(y_true, y_pred, average=None,
                                 labels=LABEL_ORDER, zero_division=0)
            rec = {
                "fold": held_out,
                "mode": mode,
                "rollup": rollup,
                "mode_rollup": f"{mode}_{rollup}",
                "raw_acc": raw,
                "bal_acc": bal,
                "macro_f1": mf1,
            }
            for lbl, v in zip(LABEL_ORDER, per_class):
                rec[f"f1_{lbl.replace(' ', '_')}"] = v
            fold_records.append(rec)
            print(f"  {col}: bal_acc={bal:.3f}  macro_f1={mf1:.3f}  raw={raw:.3f}")

    # ── image-level predictions for time-bin plots ──
    # img_preds has columns: query_snip_id, query_embryo_id, argmax_label, mode, ...
    # merge in true label and time
    img_time = qry[["snip_id", "predicted_stage_hpf"]].rename(
        columns={"snip_id": "query_snip_id"}
    )
    img_true = qry[["snip_id", "cluster_categories"]].rename(
        columns={"snip_id": "query_snip_id", "cluster_categories": "true_label"}
    )
    ip = img_preds.merge(img_time, on="query_snip_id", how="left")
    ip = ip.merge(img_true, on="query_snip_id", how="left")
    ip["fold"] = held_out
    image_records.append(ip)

# ── assemble results ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(fold_records)
results_df.to_csv(os.path.join(OUT_DIR, "loeo_logistic_results.csv"), index=False)

all_img = pd.concat(image_records, ignore_index=True)
all_img.to_csv(os.path.join(OUT_DIR, "loeo_logistic_image_predictions.csv"), index=False)

print(f"\n{'mode_rollup':<18} {'bal_acc mean±std':>20}  {'macro_f1 mean±std':>20}")
print("-" * 62)
for mr in [f"{m}_{r}" for m in MODES for r in ROLLUPS]:
    sub = results_df[results_df["mode_rollup"] == mr]
    print(f"{mr:<18}  {sub['bal_acc'].mean():.3f} ± {sub['bal_acc'].std():.3f}"
          f"    {sub['macro_f1'].mean():.3f} ± {sub['macro_f1'].std():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
MR_ORDER = [f"{m}_{r}" for m in MODES for r in ROLLUPS]
MR_TICK  = [f"{MODE_LABELS[m]}\n({ROLLUP_LABELS[r]})" for m in MODES for r in ROLLUPS]

# ── Fig 1: balanced accuracy per fold, box + strip ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, mr in enumerate(MR_ORDER):
    sub = results_df[results_df["mode_rollup"] == mr]["bal_acc"].values
    bp = ax.boxplot(sub, positions=[i], widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(facecolor=COLORS[mr], alpha=0.6))
    ax.scatter(np.full(len(sub), i) + np.random.uniform(-0.12, 0.12, len(sub)),
               sub, color=COLORS[mr], edgecolors="k", s=40, zorder=5, linewidths=0.6)
ax.set_xticks(range(len(MR_ORDER)))
ax.set_xticklabels(MR_TICK, fontsize=9)
ax.set_ylabel("Balanced accuracy")
ax.set_title("LOEO — balanced accuracy by mode × rollup")
ax.set_ylim(0, 1)
ax.axhline(0.25, ls="--", color="gray", lw=0.8, label="chance (4 classes)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "loeo_logistic_balanced_acc.png"), dpi=150)
plt.close()
print("Saved loeo_logistic_balanced_acc.png")

# ── Fig 2: macro F1 per fold ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, mr in enumerate(MR_ORDER):
    sub = results_df[results_df["mode_rollup"] == mr]["macro_f1"].values
    ax.boxplot(sub, positions=[i], widths=0.5,
               patch_artist=True, showfliers=False,
               medianprops=dict(color="black", linewidth=2),
               boxprops=dict(facecolor=COLORS[mr], alpha=0.6))
    ax.scatter(np.full(len(sub), i) + np.random.uniform(-0.12, 0.12, len(sub)),
               sub, color=COLORS[mr], edgecolors="k", s=40, zorder=5, linewidths=0.6)
ax.set_xticks(range(len(MR_ORDER)))
ax.set_xticklabels(MR_TICK, fontsize=9)
ax.set_ylabel("Macro F1")
ax.set_title("LOEO — macro F1 by mode × rollup")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "loeo_logistic_macro_f1.png"), dpi=150)
plt.close()
print("Saved loeo_logistic_macro_f1.png")

# ── Fig 3: per-class F1 heat map (mean across folds) ─────────────────────────
f1_cols = [f"f1_{l.replace(' ', '_')}" for l in LABEL_ORDER]
mean_f1 = (
    results_df.groupby("mode_rollup")[f1_cols].mean()
    .reindex(MR_ORDER)
)
mean_f1.columns = LABEL_ORDER

fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(mean_f1.values.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(len(MR_ORDER)))
ax.set_xticklabels(MR_TICK, fontsize=9, rotation=0)
ax.set_yticks(range(len(LABEL_ORDER)))
ax.set_yticklabels(LABEL_ORDER, fontsize=10)
ax.set_title("Mean per-class F1 (LOEO)")
plt.colorbar(im, ax=ax, label="F1")
for i, mr in enumerate(MR_ORDER):
    for j, lbl in enumerate(LABEL_ORDER):
        v = mean_f1.loc[mr, lbl]
        ax.text(i, j, f"{v:.2f}", ha="center", va="center",
                fontsize=8, color="black" if 0.3 < v < 0.7 else "white")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "loeo_logistic_f1_by_class.png"), dpi=150)
plt.close()
print("Saved loeo_logistic_f1_by_class.png")

# ── Fig 4: F1 by 4-hpf time bin, per mode×rollup ─────────────────────────────
# Use image-level argmax predictions (mode column) vs true_label
# For "margin" rollup at image level we use the same argmax_label (margin only
# affects embryo rollup aggregation). Label it as "image argmax" per mode.

all_img = all_img.dropna(subset=["true_label", "predicted_stage_hpf"])
# bin across the full HPF range present in the data
_hpf_min = float(np.floor(all_img["predicted_stage_hpf"].min() / BIN_WIDTH) * BIN_WIDTH)
all_img["time_bin"] = (
    (all_img["predicted_stage_hpf"] - _hpf_min) // BIN_WIDTH * BIN_WIDTH + _hpf_min + BIN_WIDTH / 2
).astype(float)
bins = sorted(all_img["time_bin"].unique())

# For each mode, compute per-bin macro F1 on image-level argmax
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, mode in zip(axes, MODES):
    sub = all_img[all_img["mode"] == mode]
    bin_f1 = []
    bin_bal = []
    for b in bins:
        bsub = sub[sub["time_bin"] == b]
        if len(bsub) < 5:
            bin_f1.append(np.nan)
            bin_bal.append(np.nan)
            continue
        mf1 = f1_score(bsub["true_label"], bsub["argmax_label"],
                       average="macro", labels=LABEL_ORDER, zero_division=0)
        ba = balanced_accuracy_score(bsub["true_label"], bsub["argmax_label"])
        bin_f1.append(mf1)
        bin_bal.append(ba)

    color_m = COLORS[f"{mode}_mean"]
    ax.plot(bins, bin_f1, "o-", color=color_m, lw=2, ms=6, label="macro F1")
    ax.plot(bins, bin_bal, "s--", color=color_m, lw=1.5, ms=5, alpha=0.7, label="bal acc")

    # per-class F1 by bin
    class_colors = {"Low_to_High": "#4DAC26", "High_to_Low": "#D6604D",
                    "Intermediate": "#8073AC", "Not Penetrant": "#2166AC"}
    for lbl in LABEL_ORDER:
        cf1 = []
        for b in bins:
            bsub = sub[sub["time_bin"] == b]
            if len(bsub) < 5:
                cf1.append(np.nan)
                continue
            cf1.append(f1_score(bsub["true_label"], bsub["argmax_label"],
                                average=None, labels=LABEL_ORDER, zero_division=0
                                )[LABEL_ORDER.index(lbl)])
        ax.plot(bins, cf1, ":", lw=1.2, alpha=0.55,
                color=class_colors[lbl], label=lbl)

    ax.set_title(f"Mode {mode} — {MODE_LABELS[mode]}", fontsize=9)
    ax.set_xlabel("Time bin center (hpf)")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(16))
    ax.tick_params(axis="x", labelsize=8)
    if mode == "A":
        ax.set_ylabel("F1 / balanced acc")

axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Image-level prediction quality by 4-hpf time bin (LOEO pooled)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "loeo_logistic_f1_by_timebin.png"), dpi=150)
plt.close()
print("Saved loeo_logistic_f1_by_timebin.png")

print("\nDone.")
