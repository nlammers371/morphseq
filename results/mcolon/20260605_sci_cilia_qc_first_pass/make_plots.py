"""
Cilia QC first-pass — PLOTS (run AFTER predicted_stage_hpf was backfilled).

Re-runs the per-dataset label transfer (capturing the ref_model + transfer result objects that
build_reference_and_transfer.py discards) and produces three things:

  1. Core reference-quality figures (plot_reference_quality)        — 1 per dataset.
  2. Core transfer-result figures (plot_transfer_result)           — per dataset AND per plate.
  3. Per-plate × 4-hpf-window GENOTYPE accuracy (the headline)     — heatmap + grouped bars per dataset.

Genotype (zygosity: wt/het/homo) is the BENCHMARK (scored vs known truth). Phenotype is predictions
only — plotted via plot_transfer_result but never scored.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260605_sci_cilia_qc_first_pass/make_plots.py
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

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.classification.label_transfer import (  # noqa: E402
    prepare_reference,
    transfer_labels,
    plot_reference_quality,
    plot_transfer_result,
)
import src.analyze.classification.label_transfer.core as _lt_core  # noqa: E402
from src.analyze.viz.styling.color_mapping_config import (  # noqa: E402
    GENOTYPE_SUFFIX_COLORS,
)

# reuse the exact config + loaders from the transfer script (single source of truth)
import build_reference_and_transfer as T  # noqa: E402
from sequenced_focus_config import PHENOTYPE_COLORS  # noqa: E402

# cep290 phenotype palette: Northwest Def Bio "talk" colors (src/analyze/viz/presets/nwdb.py).
# Only the two directional classes are used now (Not Penetrant / Intermediate are dropped).
CEP290_PHENO_COLORS = {
    "High_to_Low": "#E76FA2",   # NWDB pink
    "Low_to_High": "#2FB7B0",   # NWDB teal
}
CEP290_PHENO_ORDER = ["High_to_Low", "Low_to_High"]

# b9d2 phenotype palette: BA_rescue is MERGED into HTA (head-trunk angle); classes = CE / HTA / wt.
B9D2_PHENO_COLORS = PHENOTYPE_COLORS["b9d2"]
B9D2_PHENO_ORDER = ["CE", "HTA", "wildtype"]

# The core plot helpers color classes via core._CLASS_COLORS, which only defines the cep290
# PHENOTYPE palette -> zygosity/b9d2 classes fell back to gray. Inject the shared genotype +
# phenotype palettes so the core figures (reference_quality / transfer_result) are colored.
_CRISPANT_GENE_COLORS_FOR_CORE = {
    "ab_wildtype": GENOTYPE_SUFFIX_COLORS["wildtype"],
    "injection_control": "#999999",
    "foxj1a_crispant": "#1b9e77", "ift88_crispant": "#d95f02",
    "ift88_ift74_crispant": "#7570b3", "sspo_crispant": "#e7298a",
}
_lt_core._CLASS_COLORS = {
    **_lt_core._CLASS_COLORS,
    **GENOTYPE_SUFFIX_COLORS,         # wildtype/heterozygous/homozygous/crispant/unknown
    **CEP290_PHENO_COLORS,            # NWDB cep290 phenotype colors
    **B9D2_PHENO_COLORS,              # b9d2 CE/HTA/wildtype (BA_rescue merged into HTA)
    **_CRISPANT_GENE_COLORS_FOR_CORE, # crispant gene classes (foxj1a/ift88/sspo/...)
    "wt": GENOTYPE_SUFFIX_COLORS["wildtype"],   # in case bare aliases ever appear
}

# Time-bin width used ONLY for the reference model's internal precision/recall-by-time-bin
# diagnostic (the core reference_quality figure). The headline accuracy view does NOT bin —
# it uses each experiment's discrete design stage (start_age_hpf: 14/18/24/30/48) as the window.
REF_BIN_WIDTH = 4.0

# zygosity palette for the headline accuracy plots (from shared config, single source of truth)
ZYG_COLORS = {z: GENOTYPE_SUFFIX_COLORS[z] for z in ("wildtype", "heterozygous", "homozygous")}
ZYG_ORDER = ["wildtype", "heterozygous", "homozygous"]

# crispant gene-class palette (categorical, distinct per gene; ab/inj-ctrl reuse shared colors)
import matplotlib.cm as _cm  # noqa: E402
CRISPANT_GENE_COLORS = {
    "ab_wildtype": GENOTYPE_SUFFIX_COLORS["wildtype"],
    "injection_control": "#999999",
    "foxj1a_crispant": "#1b9e77",
    "ift88_crispant": "#d95f02",
    "ift88_ift74_crispant": "#7570b3",
    "sspo_crispant": "#e7298a",
}
CRISPANT_GENE_ORDER = ["ab_wildtype", "injection_control", "foxj1a_crispant",
                       "ift88_crispant", "ift88_ift74_crispant", "sspo_crispant"]


def class_palette(name: str) -> tuple[list, dict]:
    """(class_order, class_colors) for the genotype-class accuracy plots, per dataset."""
    if name == "crispant":
        return CRISPANT_GENE_ORDER, CRISPANT_GENE_COLORS
    return ZYG_ORDER, ZYG_COLORS

OUT = RUN_DIR / "plots"
OUT.mkdir(exist_ok=True)


# ── small helpers ────────────────────────────────────────────────────────────────
def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(RUN_DIR)}")


def _embryo_hpf(result: dict) -> pd.Series:
    """Median predicted_stage_hpf per query embryo (from image-level predictions)."""
    img = result["image_predictions"]
    gcol = T.GROUP_COL
    return img.groupby(gcol)[T.TIME_COL].median()


# ── per-dataset transfer, capturing model + result objects ────────────────────────
def run_dataset(name: str, cfg: dict):
    """Returns (geno_model, geno_result, pheno_model, pheno_result, geno_emb_scored)."""
    qpaths = [T.B6 / f"df03_final_output_with_latents_{e}.csv" for e in cfg["queries"]]
    present = [(e, p) for e, p in zip(cfg["queries"], qpaths) if p.exists()]
    if not present:
        print(f"[{name}] no query build06 — skip")
        return None
    feat = T.resolve_feature_cols([cfg["ref"], *[p for _, p in present]])
    gene_hint = name if name in ("cep290", "b9d2") else None
    ref = T._load(cfg["ref"], feat, gene_hint=gene_hint)
    qparts = []
    start_age = {}  # embryo_id -> discrete design stage (start_age_hpf)
    for e, p in present:
        q = T._load(p, feat, gene_hint=gene_hint)
        q["query_experiment"] = e
        qparts.append(q)
        # start_age_hpf is the discrete experiment stage (14/18/24/30/48) — the window key.
        # T._load drops it, so read it straight from the source, keyed by embryo_id.
        sa = pd.read_csv(p, usecols=[T.GROUP_COL, "start_age_hpf"], low_memory=False)
        start_age.update(sa.dropna(subset=["start_age_hpf"])
                         .groupby(T.GROUP_COL)["start_age_hpf"].median().to_dict())
    qry = pd.concat(qparts, ignore_index=True)
    print(f"[{name}] feat={len(feat)}  ref={len(ref)}  query={len(qry)} "
          f"({qry['query_experiment'].nunique()} plates)")

    res = {"name": name, "qry": qry, "feat": feat, "start_age": start_age}

    # ── GENOTYPE (benchmark) ──
    # cep290/b9d2: the genotype label is ZYGOSITY (wt/het/homo). Reference spans many experiments
    #              -> leave-one-experiment-out CV.
    # crispant:    no zygosity axis; the genotype label is the GENE CLASS (foxj1a/ift88/sspo/ab...).
    #              Reference is a SINGLE experiment -> use k-fold CV (cv_group_col=None).
    if name == "crispant":
        label_col, true_col = T.GENO_COL, T.GENO_COL          # full standardized genotype = gene class
        r = ref.dropna(subset=[label_col, T.TIME_COL])
        r = r[~r[label_col].astype(str).str.endswith("unknown")]
        cv = None                                             # k-fold (single-experiment reference)
    else:
        label_col, true_col = T.ZYG_COL, T.ZYG_COL
        r = ref.dropna(subset=[label_col, T.TIME_COL])
        r = r[r[label_col] != "unknown"]
        cv = "experiment_id" if r["experiment_id"].nunique() >= 3 else None

    gmodel = prepare_reference(r, feat, label_col=label_col, group_col=T.GROUP_COL,
                               time_col=T.TIME_COL, cv_group_col=cv, bin_width=REF_BIN_WIDTH)
    gresult = transfer_labels(gmodel, qry, skip_flagged=False)
    # attach truth + stage for scoring
    emb = gresult["embryo_predictions"].copy()
    meta = qry.drop_duplicates(T.GROUP_COL).set_index(T.GROUP_COL)
    emb["query_experiment"] = emb["query_embryo_id"].map(meta["query_experiment"])
    emb["true_zygosity"] = emb["query_embryo_id"].map(meta[true_col])
    emb["hpf"] = emb["query_embryo_id"].map(_embryo_hpf(gresult))         # continuous (drifted)
    emb["stage"] = emb["query_embryo_id"].map(start_age)                  # discrete design stage
    # benchmarkable = query has a known class that the reference model can actually predict
    known = emb["true_zygosity"].notna() & ~emb["true_zygosity"].astype(str).str.endswith("unknown")
    emb["benchmarkable"] = known & emb["true_zygosity"].isin(gmodel["classes"])
    emb["correct"] = emb["benchmarkable"] & (emb["predicted_label"] == emb["true_zygosity"])
    res.update(gmodel=gmodel, gresult=gresult, gemb=emb)

    # ── PHENOTYPE (predictions only) ──
    if cfg["phenotype"]:
        rp = ref.dropna(subset=[T.PHENO_COL, T.TIME_COL]).copy()
        rp = rp[~rp[T.PHENO_COL].astype(str).isin(["unlabeled", "nan"])]
        if name == "cep290":
            # Per NWDB: MERGE Intermediate INTO Low_to_High, then keep only the two directional
            # classes (drop Not Penetrant). Result: High_to_Low + Low_to_High(+Intermediate).
            rp.loc[rp[T.PHENO_COL] == "Intermediate", T.PHENO_COL] = "Low_to_High"
            rp = rp[rp[T.PHENO_COL].isin(["High_to_Low", "Low_to_High"])]
        elif name == "b9d2":
            # MERGE BA_rescue into HTA (head-trunk angle); classes become CE / HTA / wildtype.
            rp.loc[rp[T.PHENO_COL] == "BA_rescue", T.PHENO_COL] = "HTA"
        cv = "experiment_id" if rp["experiment_id"].nunique() >= 3 else None
        pmodel = prepare_reference(rp, feat, label_col=T.PHENO_COL, group_col=T.GROUP_COL,
                                   time_col=T.TIME_COL, cv_group_col=cv, bin_width=REF_BIN_WIDTH)
        presult = transfer_labels(pmodel, qry, skip_flagged=False)
        # per-embryo phenotype prediction with truth + stage attached
        pemb = presult["embryo_predictions"].copy()
        meta = qry.drop_duplicates(T.GROUP_COL).set_index(T.GROUP_COL)
        pemb["query_experiment"] = pemb["query_embryo_id"].map(meta["query_experiment"])
        pemb["true_zygosity"] = pemb["query_embryo_id"].map(meta[T.ZYG_COL])
        pemb["true_genotype"] = pemb["query_embryo_id"].map(meta[T.GENO_COL])
        pemb["stage"] = pemb["query_embryo_id"].map(start_age)
        res.update(pmodel=pmodel, presult=presult, pemb=pemb)

    return res


def _plates_by_stage(tbl: pd.DataFrame) -> list[str]:
    """Plate order ~ developmental stage: sort by each plate's min stage, then name."""
    key = (tbl.groupby("query_experiment")["stage"]
           .min().sort_values(kind="stable"))
    return list(key.index)


def _order_plates_by_stage_map(plates, stage_map: dict) -> list:
    """Order arbitrary plate names by a {plate: representative_stage} map, then name."""
    return sorted(plates, key=lambda p: (stage_map.get(p, 1e9), p))


# ── per-plate × discrete-stage genotype-accuracy table ────────────────────────────
def accuracy_table(gemb: pd.DataFrame) -> pd.DataFrame:
    """One row per (plate, discrete design stage, zygosity). Stage = start_age_hpf, not binned."""
    b = gemb[gemb["benchmarkable"] & gemb["stage"].notna()].copy()
    b["stage"] = b["stage"].astype(int)
    g = (b.groupby(["query_experiment", "stage", "true_zygosity"])
         .agg(n=("correct", "size"), accuracy=("correct", "mean"))
         .reset_index())
    return g


def plot_accuracy_heatmap(name: str, tbl: pd.DataFrame, class_order: list, class_colors: dict):
    """Faceted heatmap: one panel per class; rows=plate, cols=discrete stage; color=acc, text=n."""
    present = list(tbl["true_zygosity"].unique())
    zygs = [z for z in class_order if z in present] + [z for z in present if z not in class_order]
    if not zygs:
        return
    plates = _plates_by_stage(tbl)
    stages = sorted(tbl["stage"].unique())
    # high accuracy = red (top), low = blue (bottom)
    cmap = LinearSegmentedColormap.from_list("acc", ["#2166AC", "#F7F7F7", "#B2182B"])

    fig, axes = plt.subplots(1, len(zygs), figsize=(2.4 + 3.2 * len(zygs), 1.0 + 0.45 * len(plates)),
                             squeeze=False)
    for j, z in enumerate(zygs):
        ax = axes[0][j]
        M = np.full((len(plates), len(stages)), np.nan)
        N = np.zeros((len(plates), len(stages)), dtype=int)
        sub = tbl[tbl["true_zygosity"] == z]
        for _, row in sub.iterrows():
            i = plates.index(row["query_experiment"]); k = stages.index(row["stage"])
            M[i, k] = row["accuracy"]; N[i, k] = row["n"]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([f"{s} hpf" for s in stages], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(plates)))
        ax.set_yticklabels([p.replace(f"_{name}", "").replace(name + "_", "") for p in plates]
                           if j == 0 else [""] * len(plates), fontsize=7)
        ax.set_title(z, fontsize=9, color=class_colors.get(z, "#333"))
        ax.set_xlabel("start_age stage", fontsize=8)
        for i in range(len(plates)):
            for k in range(len(stages)):
                if N[i, k] > 0:
                    ax.text(k, i, f"{M[i, k]:.2f}\nn={N[i, k]}", ha="center", va="center",
                            fontsize=6, color="black")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="accuracy")
    fig.suptitle(f"{name} — genotype transfer accuracy by plate × stage", fontsize=11)
    _save(fig, OUT / name / f"{name}_accuracy_heatmap.png")


def plot_accuracy_bars(name: str, tbl: pd.DataFrame, class_order: list, class_colors: dict):
    """Grouped bars: per plate, groups = discrete stage, bars colored by class, height = accuracy."""
    plates = _plates_by_stage(tbl)
    present_all = list(tbl["true_zygosity"].unique())
    order = [z for z in class_order if z in present_all] + [z for z in present_all if z not in class_order]
    fig, axes = plt.subplots(len(plates), 1, figsize=(9, 2.2 * len(plates)), squeeze=False, sharex=False)
    for i, plate in enumerate(plates):
        ax = axes[i][0]
        sub = tbl[tbl["query_experiment"] == plate]
        stages = sorted(sub["stage"].unique())
        zygs = [z for z in order if z in sub["true_zygosity"].unique()]
        x = np.arange(len(stages)); w = 0.8 / max(len(zygs), 1)
        for jz, z in enumerate(zygs):
            vals, ns = [], []
            for s in stages:
                cell = sub[(sub["stage"] == s) & (sub["true_zygosity"] == z)]
                vals.append(cell["accuracy"].iloc[0] if len(cell) else 0.0)
                ns.append(int(cell["n"].iloc[0]) if len(cell) else 0)
            xpos = x + (jz - (len(zygs) - 1) / 2) * w
            bars = ax.bar(xpos, vals, w, color=class_colors.get(z, "#888"), edgecolor="k", linewidth=0.4,
                          label=z if i == 0 else None)
            for b_, v, nn in zip(bars, vals, ns):
                if nn:
                    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.02, f"{nn}", ha="center",
                            va="bottom", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels([f"{s} hpf" for s in stages], fontsize=8)
        ax.set_ylim(0, 1.1); ax.set_ylabel("accuracy", fontsize=8)
        ax.set_title(plate, fontsize=9)
        ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    axes[0][0].legend(fontsize=8, ncol=3, loc="upper right")
    fig.suptitle(f"{name} — genotype accuracy (bar=accuracy, text=n) per plate × stage", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, OUT / name / f"{name}_accuracy_bars.png")


# ── per-plate transfer-result figures ─────────────────────────────────────────────
def plot_per_plate_transfer(name: str, model: dict, qry: pd.DataFrame, kind: str,
                            stage_map: dict) -> dict:
    """Run transfer per single plate (in stage order) and emit plot_transfer_result for each.

    Returns {plate: embryo_predictions} so callers can build the cross-plate composition view.
    Filenames are prefixed with a stage-ordered index so they list developmentally.
    """
    plates = _order_plates_by_stage_map(qry["query_experiment"].unique(), stage_map)
    per_plate_emb = {}
    for idx, plate in enumerate(plates):
        sub = qry[qry["query_experiment"] == plate]
        if len(sub) == 0:
            continue
        res = transfer_labels(model, sub, skip_flagged=False)
        if res["embryo_predictions"].empty:
            continue
        per_plate_emb[plate] = res["embryo_predictions"]
        figs = plot_transfer_result(model, res)
        st = stage_map.get(plate)
        tag = f"{idx:02d}_{int(st)}hpf" if st is not None else f"{idx:02d}"
        _save(figs[0], OUT / name / "per_plate" / f"{kind}_transfer_{tag}_{plate}.png")
    return per_plate_emb


def plot_label_composition_across_plates(name: str, per_plate_emb: dict, classes: list,
                                         stage_map: dict, kind: str):
    """Stacked predicted-label fractions, one bar per plate, plates ordered by stage (~time).

    A rough developmental view: how the predicted-label mix shifts across stages.
    """
    if not per_plate_emb:
        return
    plates = _order_plates_by_stage_map(per_plate_emb.keys(), stage_map)
    classes = [c for c in classes if any(
        (per_plate_emb[p]["predicted_label"] == c).any() for p in plates)]
    palette = _lt_core._CLASS_COLORS

    frac = np.zeros((len(plates), len(classes)))
    counts = np.zeros(len(plates), dtype=int)
    for i, p in enumerate(plates):
        vc = per_plate_emb[p]["predicted_label"].value_counts()
        counts[i] = int(vc.sum())
        for j, c in enumerate(classes):
            frac[i, j] = vc.get(c, 0) / counts[i] if counts[i] else 0.0

    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(plates), 5))
    x = np.arange(len(plates)); bottom = np.zeros(len(plates))
    for j, c in enumerate(classes):
        ax.bar(x, frac[:, j], bottom=bottom, color=palette.get(c, "#888"),
               edgecolor="white", linewidth=0.5, label=c)
        bottom += frac[:, j]
    labels = []
    for p in plates:
        st = stage_map.get(p)
        short = p.replace(f"_{name}", "").replace(name + "_", "")
        labels.append(f"{int(st)}hpf\n{short}" if st is not None else short)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    for i, n in enumerate(counts):
        ax.text(i, 1.01, f"n={n}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, 1.06); ax.set_ylabel("fraction of embryos (predicted label)", fontsize=9)
    ax.set_title(f"{name} — predicted {kind} composition across plates (ordered by stage ≈ time)",
                 fontsize=10, pad=14)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title=kind)
    plt.tight_layout()
    _save(fig, OUT / name / f"{name}_{kind}_label_composition_by_plate.png")


# ── homozygous-focus view ─────────────────────────────────────────────────────────
def _stacked_by_stage(ax, emb: pd.DataFrame, classes: list, palette: dict, title: str):
    """Stacked composition of `predicted_label` by discrete stage (x), for one emb subset."""
    stages = sorted(emb["stage"].dropna().astype(int).unique())
    classes = [c for c in classes if (emb["predicted_label"] == c).any()]
    x = np.arange(len(stages)); bottom = np.zeros(len(stages)); counts = np.zeros(len(stages), int)
    cnt_by_stage = {s: int((emb["stage"].astype("Int64") == s).sum()) for s in stages}
    for j, c in enumerate(classes):
        fr = []
        for s in stages:
            sub = emb[emb["stage"].astype("Int64") == s]
            n = len(sub)
            fr.append((sub["predicted_label"] == c).sum() / n if n else 0.0)
        ax.bar(x, fr, bottom=bottom, color=palette.get(c, "#888"),
               edgecolor="white", linewidth=0.5, label=c)
        bottom += np.array(fr)
    for i, s in enumerate(stages):
        ax.text(i, 1.01, f"n={cnt_by_stage[s]}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([f"{s} hpf" for s in stages], fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("fraction of homozygous embryos", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")


def plot_homozygous_focus(name: str, res: dict, out_dir: Path):
    """For KNOWN-homozygous query embryos: what genotype & phenotype are they predicted as, by stage.

    Two panels:
      left  — predicted ZYGOSITY (homo recovered? or called wt/het?) by stage
      right — predicted PHENOTYPE (Low_to_High / High_to_Low / Not Penetrant / ...) by stage
    Also dumps the per-embryo rows to CSV for inspection.
    """
    gemb = res.get("gemb")
    if gemb is None:
        return
    homo_g = gemb[gemb["true_zygosity"] == "homozygous"].copy()
    if homo_g.empty:
        print(f"  [{name}] no known-homozygous query embryos — skip homo focus")
        return

    has_pheno = "pemb" in res
    homo_p = res["pemb"][res["pemb"]["true_zygosity"] == "homozygous"].copy() if has_pheno else None

    ncol = 2 if has_pheno else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol, 5), squeeze=False)
    _stacked_by_stage(axes[0][0], homo_g, ZYG_ORDER, ZYG_COLORS,
                      "predicted ZYGOSITY of known-homozygous embryos")
    if has_pheno:
        _stacked_by_stage(axes[0][1], homo_p, res["pmodel"]["classes"], _lt_core._CLASS_COLORS,
                          "predicted PHENOTYPE of known-homozygous embryos")
    fig.suptitle(f"{name} — homozygous embryos: predicted label by stage  "
                 f"(n={homo_g['query_embryo_id'].nunique()} embryos)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_dir / f"{name}_homozygous_predicted_by_stage.png")

    # per-embryo dump: true genotype + predicted geno/pheno + stage
    cols_g = homo_g[["query_embryo_id", "query_experiment", "stage",
                     "predicted_label", "top_probability", "correct"]].rename(
        columns={"predicted_label": "predicted_zygosity", "top_probability": "zyg_top_prob"})
    if has_pheno:
        pj = homo_p[["query_embryo_id", "predicted_label", "top_probability"]].rename(
            columns={"predicted_label": "predicted_phenotype", "top_probability": "pheno_top_prob"})
        cols_g = cols_g.merge(pj, on="query_embryo_id", how="left")
    cols_g = cols_g.sort_values(["stage", "query_experiment", "query_embryo_id"])
    cols_g.to_csv(out_dir / f"{name}_homozygous_predictions.csv", index=False)
    print(f"  [{name}] homozygous: {len(cols_g)} embryos -> {out_dir.name}/")


def main():
    print("Re-running transfers + plotting (post stage-backfill)\n")
    HOMO_DIR = OUT / "homozygous_focus"
    HOMO_DIR.mkdir(exist_ok=True)
    all_acc = []
    for name, cfg in T.DATASETS.items():
        res = run_dataset(name, cfg)
        if res is None:
            continue

        # plate -> representative (min) design stage, for ordering plates ~ developmental time
        sa = res["start_age"]
        emap = res["qry"].drop_duplicates(T.GROUP_COL).set_index(T.GROUP_COL)["query_experiment"]
        stage_map = {}
        for eid, plate in emap.items():
            if eid in sa and not pd.isna(sa[eid]):
                stage_map[plate] = min(stage_map.get(plate, 1e9), float(sa[eid]))

        # 1+2. genotype: reference quality (per dataset) + transfer result (dataset + per plate)
        if "gmodel" in res:
            figs = plot_reference_quality(res["gmodel"])
            _save(figs[0], OUT / name / f"{name}_genotype_reference_quality_timebin.png")
            _save(figs[1], OUT / name / f"{name}_genotype_reference_confusion.png")
            figs = plot_transfer_result(res["gmodel"], res["gresult"])
            _save(figs[0], OUT / name / f"{name}_genotype_transfer_result.png")
            ppe = plot_per_plate_transfer(name, res["gmodel"], res["qry"], "genotype", stage_map)
            # predicted-label mix across plates (ordered by stage ≈ time)
            plot_label_composition_across_plates(name, ppe, res["gmodel"]["classes"],
                                                 stage_map, "genotype")

            # 3. headline accuracy: per-plate × discrete stage
            tbl = accuracy_table(res["gemb"])
            tbl.insert(0, "dataset", name)
            all_acc.append(tbl)
            corder, ccolors = class_palette(name)
            plot_accuracy_heatmap(name, tbl, corder, ccolors)
            plot_accuracy_bars(name, tbl, corder, ccolors)

        # phenotype (predictions only): reference quality + transfer result (dataset + per plate)
        if "pmodel" in res:
            figs = plot_reference_quality(res["pmodel"])
            _save(figs[0], OUT / name / f"{name}_phenotype_reference_quality_timebin.png")
            _save(figs[1], OUT / name / f"{name}_phenotype_reference_confusion.png")
            figs = plot_transfer_result(res["pmodel"], res["presult"])
            _save(figs[0], OUT / name / f"{name}_phenotype_transfer_result.png")
            ppe = plot_per_plate_transfer(name, res["pmodel"], res["qry"], "phenotype", stage_map)
            plot_label_composition_across_plates(name, ppe, res["pmodel"]["classes"],
                                                 stage_map, "phenotype")

        # homozygous-focus: for known-homo embryos, what geno/pheno are they predicted as, by stage
        plot_homozygous_focus(name, res, HOMO_DIR)

    if all_acc:
        acc = pd.concat(all_acc, ignore_index=True)
        acc.to_csv(OUT / "genotype_accuracy_by_plate_stage.csv", index=False)
        print(f"\nWrote accuracy table: {(OUT / 'genotype_accuracy_by_plate_stage.csv').relative_to(RUN_DIR)}")
        print("\nOverall benchmarkable accuracy by dataset:")
        for ds, sub in acc.groupby("dataset"):
            tot_n = sub["n"].sum()
            wavg = (sub["accuracy"] * sub["n"]).sum() / tot_n if tot_n else float("nan")
            print(f"  {ds:10s} {wavg:.1%}  (n={tot_n})")

    print(f"\nAll plots under: {OUT.relative_to(RUN_DIR)}/")


if __name__ == "__main__":
    main()
