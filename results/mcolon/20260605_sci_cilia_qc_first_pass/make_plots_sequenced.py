"""
SEQUENCED-FOCUS plots — PLOTTING ONLY. Consumes the CSVs from label_transfer_snapshots.py; runs no
transfer and loads no model. This is the sequenced-restricted mirror of make_plots.py.

We already checked the projection (no batch effects). Now hone in on the SEQUENCED embryos — the
high-confidence truth set — across the QUERY plates (the sci_ time-lapse experiments are analyzed
separately). Everything is restricted to sequenced>0 and stratified into FOUR strata:
    homozygous · heterozygous · wildtype_sibling (code 1, non-AB) · AB (Trachnal AB).

Per-stratum walk-through (the deliverable):
  (a) phenotype distribution            — what phenotypes do the sequenced embryos land in
  (b) what we collectively predict them — predicted genotype + phenotype composition
  (c) the label-transfer result          — per-embryo, via confusion + per-plate composition
  (d) genotype F1                         — REAL here (sequenced truth); 3-class zygosity, AB->wt.
      Phenotype has NO F1 by design (no phenotype truth) — distributions only.

Inputs  (transfer_results/, written by label_transfer_snapshots.py):
    genotype_transfer_predictions.csv, phenotype_transfer_predictions.csv, sequenced_registry.csv
Outputs (plots/sequenced_focus/):
    <gene>/<gene>_seq_genotype_confusion.png, _seq_genotype_f1.png,
    <gene>/<gene>_seq_predicted_genotype_by_stratum.png, _seq_predicted_phenotype_by_stratum.png,
    <gene>/<gene>_seq_phenotype_distribution_by_stratum.png,
    seq_genotype_f1_summary.csv, seq_stratum_counts.csv

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/make_plots_sequenced.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.viz.styling.color_mapping_config import (  # noqa: E402
    GENOTYPE_SUFFIX_COLORS,
    B9D2_PHENOTYPE_COLORS,
)

TR = RUN_DIR / "transfer_results"
OUT = RUN_DIR / "plots" / "sequenced_focus"

# 4 strata (sequenced>0), fixed order
STRATA = ["homozygous", "heterozygous", "wildtype_sibling", "AB"]
STRATUM_COLORS = {
    "homozygous": GENOTYPE_SUFFIX_COLORS["homozygous"],     # crimson
    "heterozygous": GENOTYPE_SUFFIX_COLORS["heterozygous"], # amber
    "wildtype_sibling": GENOTYPE_SUFFIX_COLORS["wildtype"], # blue
    "AB": "#999999",                                        # gray
}

# zygosity (predicted-genotype) palette
ZYG_ORDER = ["wildtype", "heterozygous", "homozygous"]
ZYG_COLORS = {z: GENOTYPE_SUFFIX_COLORS[z] for z in ZYG_ORDER}

# phenotype palettes per gene (match make_plots.py conventions)
CEP290_PHENO_COLORS = {"High_to_Low": "#E76FA2", "Low_to_High": "#2FB7B0",
                       "Not Penetrant": "#BBBBBB"}
B9D2_PHENO_COLORS = {"CE": B9D2_PHENOTYPE_COLORS["CE"], "HTA": B9D2_PHENOTYPE_COLORS["HTA"],
                     "BA_rescue": B9D2_PHENOTYPE_COLORS.get("BA_rescue", "#7570b3"),
                     "wildtype": GENOTYPE_SUFFIX_COLORS["wildtype"]}
PHENO_COLORS = {"cep290": CEP290_PHENO_COLORS, "b9d2": B9D2_PHENO_COLORS}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(RUN_DIR)}")


def _present_strata(df: pd.DataFrame) -> list[str]:
    return [s for s in STRATA if (df["stratum"] == s).any()]


# ── (d) genotype F1 — REAL, vs the sequenced genotype call (3-class zygosity, AB->wildtype) ──
def genotype_f1(geno_seq: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Per-gene + per-stratum F1 of predicted zygosity vs the sequenced true zygosity (AB->wt)."""
    rows = []
    cms = {}
    for gene, g in geno_seq.groupby("dataset"):
        y_true = g["true_zygosity"].astype(str)
        y_pred = g["predicted_label"].astype(str)
        labels = [z for z in ZYG_ORDER if z in set(y_true) | set(y_pred)]
        macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        rows.append({"dataset": gene, "stratum": "ALL", "class": "macro",
                     "f1": macro, "n": len(g)})
        for lab, fv in zip(labels, per):
            rows.append({"dataset": gene, "stratum": "ALL", "class": lab, "f1": fv,
                         "n": int((y_true == lab).sum())})
        # per-stratum accuracy (F1 per stratum is degenerate since a stratum ~ one true class)
        for strat, sg in g.groupby("stratum"):
            acc = (sg["predicted_label"].astype(str) == sg["true_zygosity"].astype(str)).mean()
            rows.append({"dataset": gene, "stratum": strat, "class": "accuracy",
                         "f1": acc, "n": len(sg)})
        cms[gene] = (confusion_matrix(y_true, y_pred, labels=labels), labels)
    return pd.DataFrame(rows), cms


def plot_genotype_confusion(gene: str, cm: np.ndarray, labels: list[str]):
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(labels), 1.6 + 1.1 * len(labels)))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted zygosity"); ax.set_ylabel("sequenced (true) zygosity")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\nn={cm[i, j]}", ha="center", va="center",
                    fontsize=8, color="black" if cmn[i, j] < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    ax.set_title(f"{gene} — genotype transfer on SEQUENCED embryos\n(rows = sequenced truth)",
                 fontsize=10)
    _save(fig, OUT / gene / f"{gene}_seq_genotype_confusion.png")


def plot_f1_bars(gene: str, f1tbl: pd.DataFrame):
    sub = f1tbl[(f1tbl["dataset"] == gene) & (f1tbl["stratum"] == "ALL")]
    sub = sub.sort_values("class", key=lambda s: s.map(
        {"macro": -1, "wildtype": 0, "heterozygous": 1, "homozygous": 2}).fillna(9))
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(sub), 4))
    colors = ["#333333" if c == "macro" else ZYG_COLORS.get(c, "#888") for c in sub["class"]]
    bars = ax.bar(range(len(sub)), sub["f1"], color=colors, edgecolor="k", linewidth=0.4)
    for b, (_, r) in zip(bars, sub.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, r["f1"] + 0.02, f"{r['f1']:.2f}\nn={r['n']}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub["class"], rotation=30, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("F1")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax.set_title(f"{gene} — genotype F1 on SEQUENCED embryos (vs sequenced call)", fontsize=10)
    _save(fig, OUT / gene / f"{gene}_seq_genotype_f1.png")


# ── (b/c) predicted-label composition by stratum ──────────────────────────────────
def plot_predicted_by_stratum(gene: str, df: pd.DataFrame, pred_col: str, classes: list[str],
                              palette: dict, kind: str):
    """Stacked composition of `pred_col` within each stratum (one bar per stratum)."""
    strata = _present_strata(df)
    classes = [c for c in classes if (df[pred_col].astype(str) == c).any()]
    if not strata or not classes:
        return
    frac = np.zeros((len(strata), len(classes)))
    counts = np.zeros(len(strata), dtype=int)
    for i, s in enumerate(strata):
        sub = df[df["stratum"] == s]
        vc = sub[pred_col].astype(str).value_counts()
        counts[i] = int(vc.sum())
        for j, c in enumerate(classes):
            frac[i, j] = vc.get(c, 0) / counts[i] if counts[i] else 0.0
    fig, ax = plt.subplots(figsize=(1.8 + 1.4 * len(strata), 5))
    x = np.arange(len(strata)); bottom = np.zeros(len(strata))
    for j, c in enumerate(classes):
        ax.bar(x, frac[:, j], bottom=bottom, color=palette.get(c, "#888"),
               edgecolor="white", linewidth=0.5, label=c)
        bottom += frac[:, j]
    for i, n in enumerate(counts):
        ax.text(i, 1.01, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("_", "\n") for s in strata], fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel(f"fraction (predicted {kind})", fontsize=9)
    ax.set_title(f"{gene} — predicted {kind} by sequenced stratum", fontsize=10, pad=12)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title=f"predicted {kind}")
    plt.tight_layout()
    _save(fig, OUT / gene / f"{gene}_seq_predicted_{kind}_by_stratum.png")


# ── (a) true-phenotype distribution per stratum — here phenotype has no truth, so this is the
#        PREDICTED phenotype distribution shown as counts (the "what phenotypes do they land in") ──
def plot_phenotype_distribution(gene: str, pheno_seq: pd.DataFrame, classes: list[str],
                                palette: dict):
    strata = _present_strata(pheno_seq)
    classes = [c for c in classes if (pheno_seq["predicted_label"].astype(str) == c).any()]
    if not strata or not classes:
        return
    fig, ax = plt.subplots(figsize=(1.8 + 1.5 * len(strata), 5))
    x = np.arange(len(strata)); w = 0.8 / max(len(classes), 1)
    for j, c in enumerate(classes):
        vals = [int(((pheno_seq["stratum"] == s) &
                     (pheno_seq["predicted_label"].astype(str) == c)).sum()) for s in strata]
        ax.bar(x + (j - (len(classes) - 1) / 2) * w, vals, w, color=palette.get(c, "#888"),
               edgecolor="k", linewidth=0.4, label=c)
    ax.set_xticks(x); ax.set_xticklabels([s.replace("_", "\n") for s in strata], fontsize=8)
    ax.set_ylabel("embryo count", fontsize=9)
    ax.set_title(f"{gene} — predicted phenotype counts by sequenced stratum\n"
                 f"(no phenotype truth -> distribution only, no F1)", fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title="predicted phenotype")
    plt.tight_layout()
    _save(fig, OUT / gene / f"{gene}_seq_phenotype_distribution_by_stratum.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geno = pd.read_csv(TR / "genotype_transfer_predictions.csv")
    pheno = pd.read_csv(TR / "phenotype_transfer_predictions.csv")
    reg = pd.read_csv(TR / "sequenced_registry.csv")

    # restrict to sequenced>0
    geno_seq = geno[geno["sequenced"] > 0].copy()
    pheno_seq = pheno[pheno["sequenced"] > 0].copy()

    # stratum counts (registry is the canonical sequenced set)
    counts = (reg.groupby(["dataset", "stratum"]).size().rename("n").reset_index())
    counts.to_csv(OUT / "seq_stratum_counts.csv", index=False)
    print("Sequenced stratum counts:")
    print(counts.to_string(index=False))

    # (d) genotype F1 — only where we have a real zygosity truth (drop crispant: no zygosity)
    geno_bench = geno_seq[geno_seq["true_zygosity"].isin(ZYG_ORDER)].copy()
    f1tbl, cms = genotype_f1(geno_bench)
    f1tbl.to_csv(OUT / "seq_genotype_f1_summary.csv", index=False)
    print("\nGenotype F1 (sequenced, vs sequenced call):")
    print(f1tbl[f1tbl["stratum"] == "ALL"].to_string(index=False))

    for gene in geno_seq["dataset"].unique():
        gg = geno_seq[geno_seq["dataset"] == gene]
        pg = pheno_seq[pheno_seq["dataset"] == gene]

        # (b) predicted GENOTYPE composition by stratum (all genes)
        plot_predicted_by_stratum(gene, gg, "predicted_label", ZYG_ORDER, ZYG_COLORS, "genotype")

        # genotype F1 + confusion (cep290/b9d2 only — crispant has no zygosity)
        if gene in cms:
            plot_genotype_confusion(gene, *cms[gene])
            plot_f1_bars(gene, f1tbl)

        # (b/a) phenotype: predicted composition + distribution by stratum (cep290/b9d2 only)
        if gene in PHENO_COLORS and not pg.empty:
            pal = PHENO_COLORS[gene]
            classes = list(pal.keys())
            plot_predicted_by_stratum(gene, pg, "predicted_label", classes, pal, "phenotype")
            plot_phenotype_distribution(gene, pg, classes, pal)

    print(f"\nAll sequenced-focus plots under: {OUT.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
