"""
3 - Plot QC and homozygous phenotype predictions for sequenced embryos.

This is a plain analysis script:
    load predictions -> define plot plans -> make plots -> save figures

The only loops are over explicit plot plans: one for genotype QC plots and one
for homozygous phenotype plots.

It reads the sequenced-only outputs from step 2 and writes a small set of QC
figures for genotype and homozygous phenotype predictions.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3_plot_qc_and_phenotype_predictions.py
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

PRED_DIR = RUN_DIR / "predictions"
PLOT_DIR = RUN_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)
GENO_PLOT_DIR = PLOT_DIR / "genotype_qc"
PHENO_PLOT_DIR = PLOT_DIR / "homozygous_phenotype"
GENO_PLOT_DIR.mkdir(exist_ok=True)
PHENO_PLOT_DIR.mkdir(exist_ok=True)

GENO_PATH = PRED_DIR / "sequenced_genotype_qc_predictions.csv"
PHENO_PATH = PRED_DIR / "sequenced_homozygous_phenotype_predictions.csv"

print("3 - plot QC and phenotype predictions")
print("Read saved sequenced-only predictions.")
print("Make genotype QC plots and homozygous phenotype/time-series plots.")

# ---------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------

geno = pd.read_csv(GENO_PATH, low_memory=False)
pheno = pd.read_csv(PHENO_PATH, low_memory=False)
print(f"Loaded {len(geno)} genotype and {len(pheno)} phenotype embryo-level predictions")

# ---------------------------------------------------------------------
# Shared label conventions
# ---------------------------------------------------------------------

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

PHENO_LABELS = {
    "b9d2": ["CE", "HTA"],
    "cep290": ["High_to_Low", "Low_to_High"],
}

def plot_accuracy_heatmap(df: pd.DataFrame, plot_spec: dict) -> None:
    gene = plot_spec["gene"]
    truth_col = plot_spec["truth_col"]
    labels = plot_spec["labels"]
    prefix = plot_spec["prefix"]

    DESIGN_STAGES = [14, 18, 24, 30, 48]

    sub = df[(df["gene"] == gene) & df[truth_col].isin(labels)].copy()
    if sub.empty:
        return

    # Assign each embryo to the nearest design stage if within ±2 hpf; drop the rest.
    stage_raw = pd.to_numeric(sub["predicted_stage_hpf"], errors="coerce")
    def snap_to_design(hpf):
        for s in DESIGN_STAGES:
            if abs(hpf - s) <= 2:
                return s
        return None
    sub["stage"] = stage_raw.map(lambda h: snap_to_design(h) if pd.notna(h) else None)
    sub = sub[sub["stage"].notna()].copy()
    sub["stage"] = sub["stage"].astype(int)

    sub["correct"] = sub["predicted_label"].astype(str) == sub[truth_col].astype(str)

    tbl = (
        sub.groupby(["experiment", "stage", truth_col])
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )

    plates = sorted(tbl["experiment"].unique(), key=lambda p: (tbl[tbl["experiment"] == p]["stage"].min(), p))
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
            [p.replace(f"_{gene}", "").replace(f"{gene}_", "") for p in plates] if j == 0 else [""] * len(plates),
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
    fig.savefig(GENO_PLOT_DIR / f"{prefix}_sequenced_accuracy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/genotype_qc/{prefix}_sequenced_accuracy_heatmap.png")


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
                j,
                i,
                f"{cmn.loc[true_label, pred_label]:.2f}\n{int(cm.loc[true_label, pred_label])}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cmn.loc[true_label, pred_label] > 0.55 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.savefig(GENO_PLOT_DIR / f"{prefix}_sequenced_confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/genotype_qc/{prefix}_sequenced_confusion.png")

    sub["correct"] = sub["predicted_label"].astype(str) == sub[truth_col].astype(str)
    acc_rows = []
    for label in labels:
        group = sub[sub[truth_col] == label]
        acc_rows.append({"label": label, "accuracy": float(group["correct"].mean()), "n": len(group)})
    acc_tbl = pd.DataFrame(acc_rows)

    fig, ax = plt.subplots(figsize=(4.6 + 0.35 * len(labels), 3.4))
    bars = ax.bar(acc_tbl["label"], acc_tbl["accuracy"])
    for bar, (_, row) in zip(bars, acc_tbl.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["accuracy"] + 0.02,
            f"{row['accuracy']:.2f}\nn={int(row['n'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{gene} — sequenced genotype accuracy by class")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    fig.savefig(GENO_PLOT_DIR / f"{prefix}_sequenced_accuracy_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/genotype_qc/{prefix}_sequenced_accuracy_by_class.png")


def plot_homozygous_phenotype(df: pd.DataFrame, plot_spec: dict) -> None:
    gene = plot_spec["gene"]
    labels = plot_spec["labels"]
    title = plot_spec["title"]
    prefix = plot_spec["prefix"]
    positive_label = plot_spec["positive_label"]

    print(f"\n[{gene}] homozygous phenotype plots")
    sub = df[(df["gene"] == gene) & (df["sequenced_stratum"] == "homozygous")].copy()
    if sub.empty:
        print(f"  no {gene} homozygous phenotype rows to plot")
        return

    sub["stage"] = pd.to_numeric(sub["predicted_stage_hpf"], errors="coerce").round().astype("Int64")
    sub = sub[sub["stage"].notna()].copy()
    if sub.empty:
        print(f"  no {gene} homozygous rows with stage available")
        return
    sub["stage"] = sub["stage"].astype(int)

    stages = sorted(sub["stage"].unique())
    counts = np.zeros((len(stages), len(labels)), dtype=int)
    for i, stage in enumerate(stages):
        stage_df = sub[sub["stage"] == stage]
        vc = stage_df["predicted_label"].astype(str).value_counts()
        for j, label in enumerate(labels):
            counts[i, j] = int(vc.get(label, 0))

    fig, ax = plt.subplots(figsize=(3.0 + 0.9 * len(stages), 4.0))
    bottom = np.zeros(len(stages))
    for j, label in enumerate(labels):
        vals = counts[:, j]
        ax.bar(stages, vals, bottom=bottom, label=label, edgecolor="white", linewidth=0.5)
        bottom += vals
    for i, stage in enumerate(stages):
        ax.text(stage, bottom[i] + 0.05, f"n={int(bottom[i])}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("predicted stage (rounded hpf)")
    ax.set_ylabel("sequenced homozygous embryo count")
    ax.set_title(title)
    ax.legend(title="predicted phenotype", fontsize=8)
    fig.savefig(PHENO_PLOT_DIR / f"{prefix}_homo_phenotype_by_stage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/homozygous_phenotype/{prefix}_homo_phenotype_by_stage.png")

    prob_col = f"prob_{positive_label}"
    if prob_col in sub.columns:
        fig, ax = plt.subplots(figsize=(3.6 + 0.9 * len(stages), 4.1))
        rng = np.random.default_rng(0)
        for label in labels:
            stage_df = sub[sub["predicted_label"].astype(str) == label]
            if stage_df.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(stage_df))
            ax.scatter(
                stage_df["stage"].astype(float) + jitter,
                stage_df[prob_col].astype(float),
                s=26,
                alpha=0.75,
                label=label,
                edgecolors="black",
                linewidths=0.2,
            )
        ax.axhline(0.5, ls=":", color="gray", lw=0.9)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("predicted stage (rounded hpf)")
        ax.set_ylabel(f"P({positive_label})")
        ax.set_title(f"{gene} homozygous phenotype probability trend")
        ax.legend(title="predicted phenotype", fontsize=8)
        fig.savefig(PHENO_PLOT_DIR / f"{prefix}_homo_probability_trend.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved plots/homozygous_phenotype/{prefix}_homo_probability_trend.png")


genotype_plot_plan = [
    {
        "gene": "b9d2",
        "truth_col": "zygosity",
        "labels": GENO_LABELS["b9d2"],
        "title": "b9d2 — sequenced genotype QC",
        "x_label": "predicted zygosity",
        "y_label": "sequenced truth",
        "prefix": "b9d2",
    },
    {
        "gene": "cep290",
        "truth_col": "zygosity",
        "labels": GENO_LABELS["cep290"],
        "title": "cep290 — sequenced genotype QC",
        "x_label": "predicted zygosity",
        "y_label": "sequenced truth",
        "prefix": "cep290",
    },
    {
        "gene": "crispant",
        "truth_col": "genotype_clean",
        "labels": GENO_LABELS["crispant"],
        "title": "crispant — sequenced genotype QC",
        "x_label": "predicted genotype",
        "y_label": "sequenced truth",
        "prefix": "crispant",
    },
]

phenotype_plot_plan = [
    {
        "gene": "b9d2",
        "labels": PHENO_LABELS["b9d2"],
        "positive_label": "HTA",
        "title": "b9d2 homozygous phenotype predictions by stage",
        "prefix": "b9d2",
    },
    {
        "gene": "cep290",
        "labels": PHENO_LABELS["cep290"],
        "positive_label": "Low_to_High",
        "title": "cep290 homozygous phenotype predictions by stage",
        "prefix": "cep290",
    },
]

for plot_spec in genotype_plot_plan:
    plot_genotype_qc(geno, plot_spec)
    plot_accuracy_heatmap(geno, plot_spec)

for plot_spec in phenotype_plot_plan:
    plot_homozygous_phenotype(pheno, plot_spec)

print(f"\nWrote plots under: {PLOT_DIR.relative_to(RUN_DIR)}/")
