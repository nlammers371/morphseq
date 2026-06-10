"""
3e - Phenotype separability over time (AUROC), per gene — a sanity check on the
confidence plot's early-timepoint numbers.

Question this answers: the per-bin confidence plot (3c) shows surprisingly high precision/
recall for b9d2 at 14/18 hpf. Is that real biology or a batch/small-n artifact? At those
bins the homozygous CE/HTA reference is ~5 CE vs 12 HTA embryos from ONE experiment, scored
via the k-fold auto-fallback — so it could separate on nuisance, not phenotype.

This script re-asks the separability question with the standard classification API
(`analyze.classification.run_classification`): leave-one-embryo-out grouped CV + permutation
p-values, time-resolved, across FOUR feature sets so we can see WHICH signal carries it:
    emb       -> z_mu_b (80 biological latents)
    curvature -> baseline_deviation_normalized      (the curvature metric)
    length    -> total_length_um
    both      -> curvature + length                  (interpretable shape pair, as in 3d)
If 14/18 hpf separability is real, the interpretable shape features should carry it and the
permutation p-value should be small. If it's a batch artifact, expect the embedding to look
separable while curvature/length do not, and/or non-significant p-values.

Cohort: HOMOZYGOUS only, per gene (mirrors the 3c confidence-plot cohort):
    b9d2   -> CE vs HTA
    cep290 -> High_to_Low vs Low_to_High

Outputs (per gene):
    classification/<gene>_phenotype/                 (saved ClassificationAnalysis run)
    plots/separability/<gene>_phenotype_auroc.png    (AUROC over time, 4 feature sets)
    plots/separability/<gene>_phenotype_auroc_heatmap.png

Run:
    PYTHONPATH=src:$PYTHONPATH conda run -n segmentation_grounded_sam --no-capture-output \
        python results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3e_phenotype_separability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

from analyze.classification import run_classification  # noqa: E402
from analyze.classification.viz import plot_aurocs_over_time, plot_auroc_heatmaps  # noqa: E402

TABLE_DIR = RUN_DIR / "tables"
CLASS_DIR = RUN_DIR / "classification"
SEP_PLOT_DIR = RUN_DIR / "plots" / "separability"
CLASS_DIR.mkdir(exist_ok=True)
SEP_PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COL = "phenotype_label"   # we build this from cluster_categories, normalized to 2 classes
ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
BIN_WIDTH = 2.0                 # 2 hpf bins (finer than the 4 hpf engine bins)
N_SPLITS = 5
N_PERM = 500                    # full permutation null for p-values + the null band
MAX_HPF = 48.0                  # only earliest timepoint -> 48 hpf (the collection window)

# curvature = baseline_deviation_normalized (confirmed by user); both = curvature + length.
# Fixed key ORDER -> the plotter assigns the same default color per feature_set in BOTH
# gene figures, so the two plots are color-standardized (emb/curvature/length/both line up).
FEATURES = {
    "emb":       "z_mu_b",
    "curvature": ["baseline_deviation_normalized"],
    "length":    ["total_length_um"],
    "both":      ["baseline_deviation_normalized", "total_length_um"],
}

# Readable display names for the two phenotype classes per gene (used in titles/axes).
GENE_SPEC = {
    "b9d2":   {"positive": "HTA", "negative": "CE",
               "alias": {"BA_rescue": "HTA"},
               "pos_label": "HTA (head-trunk angle)", "neg_label": "CE (convergent ext.)"},
    "cep290": {"positive": "Low_to_High", "negative": "High_to_Low",
               "alias": {"Intermediate": "Low_to_High"},
               "pos_label": "Low→High", "neg_label": "High→Low"},
}


def load_homozygous(gene: str, spec: dict) -> pd.DataFrame:
    """Homozygous reference embryos for this gene, with a clean 2-class phenotype_label."""
    pos, neg = spec["positive"], spec["negative"]
    keep_cols = [ID_COL, TIME_COL, "zygosity", "cluster_categories", "experiment_id",
                 "baseline_deviation_normalized", "total_length_um"]
    use = set(keep_cols)
    df = pd.read_csv(
        TABLE_DIR / f"reference_{gene}_clean.csv",
        usecols=lambda c: c in use or c.startswith("z_mu_b_"),
        low_memory=False,
    )
    df = df[df["zygosity"] == "homozygous"].copy()
    df["phenotype_label"] = df["cluster_categories"].replace(spec["alias"])
    df = df[df["phenotype_label"].isin([pos, neg])]
    df = df.dropna(subset=[TIME_COL, "baseline_deviation_normalized", "total_length_um"])
    # only earliest timepoint -> 48 hpf (the collection window; ignore the long reference tail)
    df = df[df[TIME_COL] <= MAX_HPF]
    return df


print("3e - phenotype separability over time (AUROC), per gene")
for gene, spec in GENE_SPEC.items():
    print(f"\n[{gene}] homozygous {spec['negative']} vs {spec['positive']}")
    df = load_homozygous(gene, spec)
    n_emb = df[ID_COL].nunique()
    vc = df.drop_duplicates(ID_COL)["phenotype_label"].value_counts().to_dict()
    print(f"  {n_emb} homozygous embryos | class counts {vc} | "
          f"experiments {df['experiment_id'].nunique()}")

    result = run_classification(
        df,
        class_col=CLASS_COL, id_col=ID_COL, time_col=TIME_COL,
        positive=spec["positive"], negative=spec["negative"],   # binary path
        features=FEATURES,
        bin_width=BIN_WIDTH,
        n_splits=N_SPLITS,
        n_permutations=N_PERM,
        n_jobs=-1,
        save_dir=str(CLASS_DIR / f"{gene}_phenotype"),
        overwrite=True,
        verbose=False,
    )

    # report the early bins specifically (the ones we're suspicious of)
    early = result.scores[result.scores["time_bin_center"] <= 22].sort_values(
        ["time_bin_center", "feature_set"])
    if not early.empty:
        cols = [c for c in ["time_bin_center", "feature_set", "auroc_obs", "pval",
                            "n_positive", "n_negative"] if c in early.columns]
        print("  early bins (<=22 hpf):")
        print(early[cols].to_string(index=False))

    # AUROC over time, one curve per feature set. Let the plotter use its DEFAULT palette
    # (no color_lookup) — fixed FEATURES key order means emb/curvature/length/both get the
    # same color in BOTH gene figures. Keep the permutation null band + significance markers.
    comparison = f"{spec['neg_label']}  vs  {spec['pos_label']}"
    plot_aurocs_over_time(
        result.scores,
        curve_col="feature_set",
        show_null_band=True,                # permutation null mean ± std band
        show_significance=True,
        sig_threshold=0.05,
        show_chance_line=True,
        title=f"{gene} — homozygous phenotype separability over development\n"
              f"{comparison}   |   {int(BIN_WIDTH)} hpf bins, leave-one-embryo-out CV, "
              f"{N_PERM} permutations",
        y_label="cross-validated AUROC",
        backend="matplotlib",
        output_path=str(SEP_PLOT_DIR / f"{gene}_phenotype_auroc.png"),
    )
    print(f"  saved plots/separability/{gene}_phenotype_auroc.png")

    # Feature_set × time heatmap (significant cells get a black border).
    plot_auroc_heatmaps(
        result.scores,
        facet_row="feature_set",
        sig_threshold=0.05,
        title=f"{gene} — {comparison}  ({int(BIN_WIDTH)} hpf bins)  AUROC",
        backend="matplotlib",
        output_path=str(SEP_PLOT_DIR / f"{gene}_phenotype_auroc_heatmap.png"),
    )
    print(f"  saved plots/separability/{gene}_phenotype_auroc_heatmap.png")

print(f"\nWrote separability plots under: {SEP_PLOT_DIR.relative_to(RUN_DIR)}/")
print(f"Saved classification runs under: {CLASS_DIR.relative_to(RUN_DIR)}/")
print(f"NOTE: n_permutations={N_PERM}, {int(BIN_WIDTH)} hpf bins, homozygous, ≤{int(MAX_HPF)} hpf.")
