"""
3b - Genotype QC per plate (accuracy heatmap + confusion + accuracy-by-class).

This is the genotype-QC half of the retired monolith 3_plot_qc_and_phenotype_predictions.py,
split out and retargeted to the NEW per-bin engine outputs.

Reads the CROSS-BIN (one row per query embryo) genotype-QC predictions and scores them
against the sequencing truth (`zygosity` for b9d2/cep290, `genotype_clean` for crispant).
Genotype Excel is ground truth — we do NOT reconcile truth against sequencing codes.

Per gene it writes:
    plots/genotype_qc/<gene>_sequenced_accuracy_heatmap.png   (plate x design-stage)
    plots/genotype_qc/<gene>_sequenced_confusion.png          (row-normalized confusion)
    plots/genotype_qc/<gene>_sequenced_accuracy_by_class.png

NOT the key plot (that is 3c). This is the QC "did genotype transfer behave" read-out.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3b_genotype_qc_per_plate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))  # local plot_config

from plot_config import snap_to_design_stage  # noqa: E402

PRED_DIR = RUN_DIR / "predictions"
GENO_PLOT_DIR = RUN_DIR / "plots" / "genotype_qc"
GENO_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Cross-bin (one row per query embryo) genotype-QC predictions from step 2.
GENO_PATH = PRED_DIR / "sequenced_genotype_qc_cross_bin.csv"

GENO_LABELS = {
    "b9d2": ["wildtype", "heterozygous", "homozygous"],
    "cep290": ["wildtype", "heterozygous", "homozygous"],
    "crispant": [
        "ab_wildtype",
        "foxj1a_crispant",
        "ift88_crispant",
        "ift88_ift74_crispant",
        "sspo_crispant",
    ],
}

print("3b - genotype QC per plate (cross-bin predictions)")
geno = pd.read_csv(GENO_PATH, low_memory=False)
print(f"Loaded {len(geno)} cross-bin genotype-QC embryo predictions "
      f"across genes {sorted(geno['gene'].unique())}")


def plot_accuracy_heatmap(df: pd.DataFrame, plot_spec: dict) -> None:
    gene = plot_spec["gene"]
    truth_col = plot_spec["truth_col"]
    labels = plot_spec["labels"]
    prefix = plot_spec["prefix"]

    sub = df[(df["gene"] == gene) & df[truth_col].isin(labels)].copy()
    if sub.empty:
        print(f"  [{gene}] no rows for accuracy heatmap")
        return

    stage_raw = pd.to_numeric(sub["predicted_stage_hpf"], errors="coerce")
    sub["stage"] = stage_raw.map(snap_to_design_stage)
    sub = sub[sub["stage"].notna()].copy()
    sub["stage"] = sub["stage"].astype(int)

    sub["correct"] = sub["predicted_label"].astype(str) == sub[truth_col].astype(str)

    tbl = (
        sub.groupby(["experiment", "stage", truth_col])
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )

    plates = sorted(tbl["experiment"].unique(),
                    key=lambda p: (tbl[tbl["experiment"] == p]["stage"].min(), p))
    stages = sorted(tbl["stage"].unique())
    present = [lbl for lbl in labels if lbl in tbl[truth_col].unique()]
    if not present:
        return

    cmap = LinearSegmentedColormap.from_list("acc", ["#2166AC", "#F7F7F7", "#B2182B"])
    fig, axes = plt.subplots(
        1, len(present),
        figsize=(2.4 + 3.2 * len(present), 1.0 + 0.45 * len(plates)),
        squeeze=False,
    )
    for j, lbl in enumerate(present):
        ax = axes[0][j]
        M = np.full((len(plates), len(stages)), np.nan)
        N = np.zeros((len(plates), len(stages)), dtype=int)
        sub_lbl = tbl[tbl[truth_col] == lbl]
        for _, row in sub_lbl.iterrows():
            i = plates.index(row["experiment"])
            k = stages.index(row["stage"])
            M[i, k] = row["accuracy"]
            N[i, k] = row["n"]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"{s} hpf" for s in stages], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(plates)))
        ax.set_yticklabels(
            [p.replace(f"_{gene}", "").replace(f"{gene}_", "") for p in plates]
            if j == 0 else [""] * len(plates),
            fontsize=7,
        )
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel("predicted stage (±2 hpf bin)", fontsize=8)
        for i in range(len(plates)):
            for k in range(len(stages)):
                if N[i, k] > 0:
                    ax.text(k, i, f"{M[i, k]:.2f}\nn={N[i, k]}", ha="center", va="center",
                            fontsize=6, color="black")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="accuracy")
    fig.suptitle(f"{gene} — genotype accuracy by plate × stage", fontsize=11)
    out = GENO_PLOT_DIR / f"{prefix}_sequenced_accuracy_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/genotype_qc/{out.name}")


def plot_genotype_qc(df: pd.DataFrame, plot_spec: dict) -> None:
    gene = plot_spec["gene"]
    truth_col = plot_spec["truth_col"]
    labels = plot_spec["labels"]
    title = plot_spec["title"]
    x_label = plot_spec["x_label"]
    y_label = plot_spec["y_label"]
    prefix = plot_spec["prefix"]

    print(f"\n[{gene}] genotype QC plots")
    sub = df[(df["gene"] == gene) & df[truth_col].isin(labels)].copy()
    if sub.empty:
        print(f"  no {gene} genotype rows to plot")
        return

    cm = pd.crosstab(sub[truth_col], sub["predicted_label"], dropna=False).reindex(
        index=labels, columns=labels, fill_value=0
    )
    cmn = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(4.8 + 0.35 * len(labels), 4.2 + 0.2 * len(labels)))
    im = ax.imshow(cmn.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            ax.text(
                j, i,
                f"{cmn.loc[true_label, pred_label]:.2f}\n{int(cm.loc[true_label, pred_label])}",
                ha="center", va="center", fontsize=8,
                color="white" if cmn.loc[true_label, pred_label] > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    out = GENO_PLOT_DIR / f"{prefix}_sequenced_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/genotype_qc/{out.name}")

    sub["correct"] = sub["predicted_label"].astype(str) == sub[truth_col].astype(str)
    acc_rows = []
    for label in labels:
        group = sub[sub[truth_col] == label]
        acc_rows.append({"label": label,
                         "accuracy": float(group["correct"].mean()) if len(group) else float("nan"),
                         "n": len(group)})
    acc_tbl = pd.DataFrame(acc_rows)

    fig, ax = plt.subplots(figsize=(4.6 + 0.35 * len(labels), 3.4))
    bars = ax.bar(acc_tbl["label"], acc_tbl["accuracy"].fillna(0))
    for bar, (_, row) in zip(bars, acc_tbl.iterrows()):
        acc = 0.0 if pd.isna(row["accuracy"]) else row["accuracy"]
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
                f"{acc:.2f}\nn={int(row['n'])}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{gene} — sequenced genotype accuracy by class")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax.tick_params(axis="x", rotation=30)
    out = GENO_PLOT_DIR / f"{prefix}_sequenced_accuracy_by_class.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/genotype_qc/{out.name}")


genotype_plot_plan = [
    {"gene": "b9d2", "truth_col": "zygosity", "labels": GENO_LABELS["b9d2"],
     "title": "b9d2 — sequenced genotype QC", "x_label": "predicted zygosity",
     "y_label": "sequenced truth", "prefix": "b9d2"},
    {"gene": "cep290", "truth_col": "zygosity", "labels": GENO_LABELS["cep290"],
     "title": "cep290 — sequenced genotype QC", "x_label": "predicted zygosity",
     "y_label": "sequenced truth", "prefix": "cep290"},
    {"gene": "crispant", "truth_col": "genotype_clean", "labels": GENO_LABELS["crispant"],
     "title": "crispant — sequenced genotype QC", "x_label": "predicted genotype",
     "y_label": "sequenced truth", "prefix": "crispant"},
]

for plot_spec in genotype_plot_plan:
    plot_genotype_qc(geno, plot_spec)
    plot_accuracy_heatmap(geno, plot_spec)

print(f"\nWrote genotype-QC plots under: {GENO_PLOT_DIR.relative_to(RUN_DIR)}/")
