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
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score, confusion_matrix

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.viz.styling.color_mapping_config import (  # noqa: E402
    GENOTYPE_SUFFIX_COLORS,
    B9D2_PHENOTYPE_COLORS,
)
import build_reference_and_transfer as T  # noqa: E402
from sequenced_focus_config import PHENOTYPE_COLORS, STAGE_GRIDS  # noqa: E402

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
CEP290_PHENO_COLORS = PHENOTYPE_COLORS["cep290"]
B9D2_PHENO_COLORS = PHENOTYPE_COLORS["b9d2"]
PHENO_COLORS = {"cep290": CEP290_PHENO_COLORS, "b9d2": B9D2_PHENO_COLORS}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(RUN_DIR)}")


def _present_strata(df: pd.DataFrame) -> list[str]:
    return [s for s in STRATA if (df["stratum"] == s).any()]


def _stage_lookup(geno_seq: pd.DataFrame) -> dict[str, float]:
    """query_embryo_id -> discrete design stage from build06 start_age_hpf."""
    lookup = {}
    for exp in sorted(geno_seq["query_experiment"].dropna().astype(str).unique()):
        path = T.B6 / f"df03_final_output_with_latents_{exp}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=[T.GROUP_COL, "start_age_hpf"], low_memory=False)
        except ValueError:
            continue
        lookup.update(
            df.dropna(subset=["start_age_hpf"])
            .groupby(T.GROUP_COL)["start_age_hpf"]
            .median()
            .to_dict()
        )
    return lookup


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




# -- requested per-plate sequenced-only genotype views ---------------------------
def _truth_group(row: pd.Series) -> str:
    """Four sequenced truth groups: AB, gene_wildtype, gene_heterozygous, gene_homozygous."""
    dataset = str(row.get("dataset", ""))
    stratum = str(row.get("stratum", ""))
    true_genotype = str(row.get("true_genotype", ""))
    if stratum == "AB" or true_genotype == "ab_wildtype":
        return "AB -> wildtype"
    if stratum == "wildtype_sibling" or true_genotype.endswith("_wildtype"):
        return f"{dataset}_wildtype -> wildtype"
    if stratum == "heterozygous" or true_genotype.endswith("_heterozygous"):
        return f"{dataset}_heterozygous -> heterozygous"
    if stratum == "homozygous" or true_genotype.endswith("_homozygous"):
        return f"{dataset}_homozygous -> homozygous"
    return "unknown"


def _expected_label(group: str) -> str | None:
    if group.endswith("-> wildtype"):
        return "wildtype"
    if group.endswith("-> heterozygous"):
        return "heterozygous"
    if group.endswith("-> homozygous"):
        return "homozygous"
    return None


def _truth_group_order(gene: str) -> list[str]:
    return [
        "AB -> wildtype",
        f"{gene}_wildtype -> wildtype",
        f"{gene}_heterozygous -> heterozygous",
        f"{gene}_homozygous -> homozygous",
    ]


def _short_plate(plate: str, gene: str) -> str:
    return str(plate).replace(f"_{gene}", "").replace(gene + "_", "")


def _ordered_plates(df: pd.DataFrame) -> list[str]:
    return sorted(
        df["query_experiment"].dropna().astype(str).unique(),
        key=lambda p: (
            df.loc[df["query_experiment"].astype(str) == p, "stage"].dropna().min(),
            str(p),
        ),
    )

def _stage_grid(gene: str, observed) -> list[int]:
    if gene in STAGE_GRIDS:
        return STAGE_GRIDS[gene]
    return sorted(int(s) for s in observed)



def _cell_text_from_counts(counts: pd.Series, labels: list[str], aliases: dict[str, str]) -> str:
    n = int(counts.sum())
    if n == 0:
        return ""
    parts = []
    for label in labels:
        val = int(counts.get(label, 0))
        if val:
            parts.append(f"{aliases.get(label, label)} {val / n:.2f}")
    return "\n".join(parts + [f"n={n}"])


def sequenced_accuracy_table(geno_seq: pd.DataFrame) -> pd.DataFrame:
    """Accuracy by dataset x plate x stage x four-class sequenced truth group."""
    bench = geno_seq.dropna(subset=["stage"]).copy()
    bench = bench[bench["predicted_label"].isin(ZYG_ORDER)]
    if bench.empty:
        return pd.DataFrame()
    bench["stage"] = bench["stage"].astype(float).round().astype(int)
    bench["truth_group"] = bench.apply(_truth_group, axis=1)
    bench["expected_label"] = bench["truth_group"].map(_expected_label)
    bench = bench[bench["expected_label"].notna()]
    bench["correct"] = bench["predicted_label"].astype(str) == bench["expected_label"].astype(str)

    rows = []
    for keys, sub in bench.groupby(["dataset", "query_experiment", "stage", "truth_group", "expected_label"]):
        dataset, plate, stage, truth_group, expected = keys
        rows.append(
            {
                "dataset": dataset,
                "query_experiment": plate,
                "stage": stage,
                "truth_group": truth_group,
                "expected_label": expected,
                "n": len(sub),
                "accuracy": float(sub["correct"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_seq_accuracy_heatmap(gene: str, tbl: pd.DataFrame) -> None:
    """Four-facet per-plate x stage genotype-transfer accuracy for sequenced truth groups."""
    sub_tbl = tbl[tbl["dataset"] == gene].copy() if not tbl.empty else pd.DataFrame()
    if sub_tbl.empty:
        return
    groups = _truth_group_order(gene)
    stages = _stage_grid(gene, sub_tbl["stage"].unique())
    plates = _ordered_plates(sub_tbl)
    cmap = LinearSegmentedColormap.from_list("seq_acc", ["#2166AC", "#F7F7F7", "#B2182B"])
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(2.6 + 3.0 * len(groups), 1.2 + 0.48 * len(plates)),
        squeeze=False,
    )
    im = None
    for j, group in enumerate(groups):
        ax = axes[0][j]
        mat = np.full((len(plates), len(stages)), np.nan)
        nums = np.zeros((len(plates), len(stages)), dtype=int)
        sub = sub_tbl[sub_tbl["truth_group"] == group]
        for _, row in sub.iterrows():
            i = plates.index(row["query_experiment"])
            k = stages.index(row["stage"])
            mat[i, k] = row["accuracy"]
            nums[i, k] = int(row["n"])
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"{s} hpf" for s in stages], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(plates)))
        ax.set_yticklabels([_short_plate(p, gene) for p in plates] if j == 0 else [""] * len(plates), fontsize=7)
        ax.set_title(group, fontsize=8, color=ZYG_COLORS.get(_expected_label(group), "#333"))
        ax.set_xlabel("design stage", fontsize=8)
        for i in range(len(plates)):
            for k in range(len(stages)):
                if nums[i, k] > 0:
                    ax.text(k, i, f"{mat[i, k]:.2f}\nn={nums[i, k]}", ha="center", va="center", fontsize=6)
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="accuracy")
    fig.suptitle(f"{gene} - SEQUENCED genotype-transfer accuracy by true genotype x plate x stage", fontsize=11)
    _save(fig, OUT / gene / f"{gene}_seq_true_genotype_accuracy_heatmap_plate_stage.png")


def plot_seq_genotype_mapping_composition(gene: str, geno_seq: pd.DataFrame) -> None:
    """Per-cell predicted zygosity distribution for each true sequenced genotype group."""
    g = geno_seq[geno_seq["dataset"] == gene].dropna(subset=["stage"]).copy()
    g = g[g["predicted_label"].isin(ZYG_ORDER)]
    if g.empty:
        return
    g["stage"] = g["stage"].astype(float).round().astype(int)
    g["truth_group"] = g.apply(_truth_group, axis=1)
    groups = _truth_group_order(gene)
    stages = _stage_grid(gene, g["stage"].unique())
    plates = _ordered_plates(g)
    aliases = {"wildtype": "wt", "heterozygous": "het", "homozygous": "hom"}

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(3.2 + 3.5 * len(groups), 1.4 + 0.58 * len(plates)),
        squeeze=False,
    )
    cmap = LinearSegmentedColormap.from_list("seq_comp", ["#2166AC", "#F7F7F7", "#B2182B"])
    im = None
    for j, group in enumerate(groups):
        ax = axes[0][j]
        expected = _expected_label(group)
        mat = np.full((len(plates), len(stages)), np.nan)
        sub_group = g[g["truth_group"] == group]
        for i, plate in enumerate(plates):
            for k, stage in enumerate(stages):
                cell = sub_group[(sub_group["query_experiment"] == plate) & (sub_group["stage"] == stage)]
                if cell.empty:
                    continue
                counts = cell["predicted_label"].value_counts()
                n = int(counts.sum())
                mat[i, k] = counts.get(expected, 0) / n if n else np.nan
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"{s} hpf" for s in stages], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(plates)))
        ax.set_yticklabels([_short_plate(p, gene) for p in plates] if j == 0 else [""] * len(plates), fontsize=7)
        ax.set_title(group, fontsize=8)
        ax.set_xlabel("design stage", fontsize=8)
        for i, plate in enumerate(plates):
            for k, stage in enumerate(stages):
                cell = sub_group[(sub_group["query_experiment"] == plate) & (sub_group["stage"] == stage)]
                if cell.empty:
                    continue
                txt = _cell_text_from_counts(cell["predicted_label"].value_counts(), ZYG_ORDER, aliases)
                ax.text(k, i, txt, ha="center", va="center", fontsize=5.4)
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="expected-label fraction")
    fig.suptitle(f"{gene} - SEQUENCED genotype mapping composition by true genotype", fontsize=11)
    _save(fig, OUT / gene / f"{gene}_seq_true_genotype_mapping_composition_plate_stage.png")


def plot_seq_phenotype_mapping_composition(gene: str, pheno_seq: pd.DataFrame) -> None:
    """Per-cell mini grouped bars of predicted phenotype counts by true sequenced genotype group."""
    if gene not in PHENO_COLORS:
        return
    p = pheno_seq[pheno_seq["dataset"] == gene].dropna(subset=["stage"]).copy()
    if p.empty:
        return
    p["stage"] = p["stage"].astype(float).round().astype(int)
    p["truth_group"] = p.apply(_truth_group, axis=1)
    groups = _truth_group_order(gene)
    stages = _stage_grid(gene, p["stage"].unique())
    plates = _ordered_plates(p)
    classes = list(PHENO_COLORS[gene].keys())
    aliases = {
        "High_to_Low": "HtL",
        "Low_to_High": "LtH",
        "Not Penetrant": "NotPen",
        "wildtype": "wt",
        "HTA": "HTA",
        "CE": "CE",
        "BA_rescue": "BA",
    }

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(3.6 + 3.7 * len(groups), 1.5 + 0.70 * len(plates)),
        squeeze=False,
    )
    bar_w = 0.16 if len(classes) >= 3 else 0.22
    offsets = (np.arange(len(classes)) - (len(classes) - 1) / 2) * (bar_w * 1.15)
    max_bar_h = 0.58
    for j, group in enumerate(groups):
        ax = axes[0][j]
        sub_group = p[p["truth_group"] == group]
        ax.set_xlim(-0.5, len(stages) - 0.5)
        ax.set_ylim(len(plates) - 0.5, -0.5)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"{s} hpf" for s in stages], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(plates)))
        ax.set_yticklabels([_short_plate(x, gene) for x in plates] if j == 0 else [""] * len(plates), fontsize=7)
        ax.set_title(group, fontsize=8)
        ax.set_xlabel("design stage", fontsize=8)
        ax.set_facecolor("#FAFAFA")
        ax.set_xticks(np.arange(-0.5, len(stages), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(plates), 1), minor=True)
        ax.grid(which="minor", color="#DDDDDD", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i, plate in enumerate(plates):
            for k, stage in enumerate(stages):
                cell = sub_group[(sub_group["query_experiment"].astype(str) == plate) & (sub_group["stage"] == stage)]
                if cell.empty:
                    continue
                counts = cell["predicted_label"].astype(str).value_counts()
                n = int(counts.sum())
                baseline = i + 0.36
                ax.text(k, i - 0.39, f"n={n}", ha="center", va="top", fontsize=5.8, color="#222222")
                for cls, off in zip(classes, offsets):
                    count = int(counts.get(cls, 0))
                    frac = count / n if n else 0.0
                    if count == 0:
                        continue
                    height = frac * max_bar_h
                    left = k + off - bar_w / 2
                    top = baseline - height
                    ax.add_patch(
                        plt.Rectangle(
                            (left, top),
                            bar_w,
                            height,
                            facecolor=PHENO_COLORS[gene].get(cls, "#888"),
                            edgecolor="black",
                            linewidth=0.25,
                        )
                    )
                    ax.text(k + off, top - 0.025, str(count), ha="center", va="bottom", fontsize=5.2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PHENO_COLORS[gene].get(cls, "#888"), edgecolor="black", linewidth=0.3)
        for cls in classes
    ]
    fig.legend(
        handles,
        [aliases.get(cls, cls) for cls in classes],
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        fontsize=8,
        title="predicted phenotype",
    )
    fig.suptitle(f"{gene} - SEQUENCED predicted phenotype counts by true genotype", fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.94, 0.94])
    _save(fig, OUT / gene / f"{gene}_seq_true_genotype_phenotype_minibars_plate_stage.png")


def plot_seq_plate_confusion_grid(gene: str, geno_seq: pd.DataFrame) -> None:
    """Small-multiple per-plate confusion matrices, laid out as columns=stage/timepoint."""
    g = geno_seq[(geno_seq["dataset"] == gene) & geno_seq["true_zygosity"].isin(ZYG_ORDER)].copy()
    g = g.dropna(subset=["stage"])
    if g.empty:
        return
    g["stage"] = g["stage"].astype(float).round().astype(int)
    stages = _stage_grid(gene, g["stage"].unique())
    plates_by_stage = {
        stage: sorted(g.loc[g["stage"] == stage, "query_experiment"].astype(str).unique())
        for stage in stages
    }
    ncols = len(stages)
    nrows = max(len(v) for v in plates_by_stage.values())
    labels = [z for z in ZYG_ORDER if z in set(g["true_zygosity"]) | set(g["predicted_label"])]
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for col, stage in enumerate(stages):
        for row, plate in enumerate(plates_by_stage[stage]):
            ax = axes[row][col]
            ax.axis("on")
            sub = g[g["query_experiment"].astype(str) == plate]
            cm = confusion_matrix(sub["true_zygosity"], sub["predicted_label"], labels=labels)
            denom = cm.sum(axis=1, keepdims=True)
            cmn = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels if col == 0 else [""] * len(labels), fontsize=6)
            short = _short_plate(plate, gene)
            ax.set_title(f"{stage} hpf\n{short}\nn={len(sub)}", fontsize=7)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if cm[i, j]:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    # Stage labels are encoded in each panel title; the column layout makes timepoint grouping explicit.
    fig.supxlabel("predicted zygosity", fontsize=9)
    fig.supylabel("sequenced truth", fontsize=9)
    fig.suptitle(f"{gene} - SEQUENCED per-plate genotype assignment confusion", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, OUT / gene / f"{gene}_seq_genotype_assignment_confusion_by_plate.png")


def plot_seq_plate_predicted_composition(gene: str, geno_seq: pd.DataFrame) -> None:
    """Per-plate stacked composition of predicted genotype labels among sequenced embryos."""
    g = geno_seq[geno_seq["dataset"] == gene].copy()
    if g.empty:
        return
    plates = _ordered_plates(g)
    classes = [z for z in ZYG_ORDER if (g["predicted_label"].astype(str) == z).any()]
    frac = np.zeros((len(plates), len(classes)))
    counts = np.zeros(len(plates), dtype=int)
    for i, plate in enumerate(plates):
        sub = g[g["query_experiment"] == plate]
        vc = sub["predicted_label"].astype(str).value_counts()
        counts[i] = int(vc.sum())
        for j, cls in enumerate(classes):
            frac[i, j] = vc.get(cls, 0) / counts[i] if counts[i] else 0.0
    fig, ax = plt.subplots(figsize=(1.7 + 0.75 * len(plates), 5))
    x = np.arange(len(plates))
    bottom = np.zeros(len(plates))
    for j, cls in enumerate(classes):
        ax.bar(x, frac[:, j], bottom=bottom, color=ZYG_COLORS.get(cls, "#888"),
               edgecolor="white", linewidth=0.5, label=cls)
        bottom += frac[:, j]
    labels = []
    for plate in plates:
        stage = g.loc[g["query_experiment"] == plate, "stage"].dropna()
        short = _short_plate(plate, gene)
        labels.append(f"{int(stage.min())}hpf\n{short}" if len(stage) else short)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    for i, n in enumerate(counts):
        ax.text(i, 1.01, f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction of sequenced embryos")
    ax.set_title(f"{gene} - SEQUENCED predicted genotype composition by plate", fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title="predicted")
    plt.tight_layout()
    _save(fig, OUT / gene / f"{gene}_seq_genotype_assignment_composition_by_plate.png")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geno = pd.read_csv(TR / "genotype_transfer_predictions.csv")
    pheno = pd.read_csv(TR / "phenotype_transfer_predictions.csv")
    reg = pd.read_csv(TR / "sequenced_registry.csv")

    # restrict to sequenced>0
    geno_seq = geno[geno["sequenced"] > 0].copy()
    pheno_seq = pheno[pheno["sequenced"] > 0].copy()
    # b9d2 phenotype convention: BA_rescue is part of HTA/body-axis phenotype.
    pheno_seq.loc[(pheno_seq["dataset"] == "b9d2") &
                  (pheno_seq["predicted_label"] == "BA_rescue"), "predicted_label"] = "HTA"
    stage_lookup = _stage_lookup(geno_seq)
    geno_seq["stage"] = geno_seq["query_embryo_id"].map(stage_lookup)
    pheno_seq["stage"] = pheno_seq["query_embryo_id"].map(stage_lookup)

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

    acc_tbl = sequenced_accuracy_table(geno_seq)
    if not acc_tbl.empty:
        acc_tbl.to_csv(OUT / "seq_genotype_accuracy_by_plate_stage.csv", index=False)

    for gene in geno_seq["dataset"].unique():
        gg = geno_seq[geno_seq["dataset"] == gene]
        pg = pheno_seq[pheno_seq["dataset"] == gene]

        # (b) predicted GENOTYPE composition by stratum (all genes)
        plot_predicted_by_stratum(gene, gg, "predicted_label", ZYG_ORDER, ZYG_COLORS, "genotype")

        # genotype F1 + confusion (cep290/b9d2 only — crispant has no zygosity)
        if gene in cms:
            plot_genotype_confusion(gene, *cms[gene])
            plot_f1_bars(gene, f1tbl)
            plot_seq_accuracy_heatmap(gene, acc_tbl)
            plot_seq_genotype_mapping_composition(gene, geno_seq)
            plot_seq_plate_confusion_grid(gene, geno_seq)
            plot_seq_plate_predicted_composition(gene, geno_seq)

        # (b/a) phenotype: predicted composition + distribution by stratum (cep290/b9d2 only)
        if gene in PHENO_COLORS and not pg.empty:
            pal = PHENO_COLORS[gene]
            classes = list(pal.keys())
            plot_predicted_by_stratum(gene, pg, "predicted_label", classes, pal, "phenotype")
            plot_phenotype_distribution(gene, pg, classes, pal)
            plot_seq_phenotype_mapping_composition(gene, pg)

    print(f"\nAll sequenced-focus plots under: {OUT.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
