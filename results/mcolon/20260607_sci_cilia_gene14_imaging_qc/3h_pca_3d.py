"""
3h - 3D PCA of the gene14 cilia QC data.

One interactive HTML per gene (b9d2, cep290, crispant) with a dropdown to toggle:
    • HPF         — continuous predicted_stage_hpf
    • genotype    — zygosity (wildtype / heterozygous / homozygous / unknown)
    • phenotype   — predicted phenotype from cross-bin CSV (query); true cluster_categories (reference)
    • experiment  — experiment_id (batch view)
    • source      — reference vs query

PCA is fit on REFERENCE + QUERY combined (shared z_mu_b space) per gene so query
points land in the same coordinates as the reference — that is what makes batch
effects visible.

Reads:
    tables/reference_<gene>_clean.csv           (backdrop + cluster_categories truth)
    tables/query_all_rows_clean.csv             (query per-frame features)
    predictions/sequenced_homozygous_phenotype_cross_bin.csv  (predicted_label per embryo)
Writes:
    plots/pca_3d/<gene>_pca_3d.html

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/3h_pca_3d.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import plotly.express as px

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

from plot_config import PHENOTYPE_COLORS, GENOTYPE_COLORS  # noqa: E402
from src.analyze.utils.pca import fit_pca_on_embeddings, transform_embeddings_to_pca  # noqa: E402
from src.analyze.viz.plotting import plot_3d_scatter  # noqa: E402

TABLE_DIR = RUN_DIR / "tables"
PRED_DIR = RUN_DIR / "predictions"
OUT = RUN_DIR / "plots" / "pca_3d"
OUT.mkdir(parents=True, exist_ok=True)

TIME_COL = "predicted_stage_hpf"
ID_COL = "embryo_id"

ZYG_COLORS = {
    "wildtype": GENOTYPE_COLORS.get("wildtype", "#2166AC"),
    "heterozygous": GENOTYPE_COLORS.get("heterozygous", "#F7B267"),
    "homozygous": GENOTYPE_COLORS.get("homozygous", "#B2182B"),
    "unknown": GENOTYPE_COLORS.get("unknown", "#808080"),
}

SOURCE_COLORS = {"reference": "#cccccc", "snapshot": "#d62728", "timeseries (sci)": "#9467bd"}

SEQ_COLORS = {
    "reference":     "#cccccc",
    "not_sequenced": "#aaaaaa",
    "seq_wt(1)":     "#2166AC",
    "seq_mutant(2)": "#B2182B",
}

# reference cluster_categories -> canonical phenotype label
REF_LABEL_MAP = {
    "CE": "CE", "HTA": "HTA", "BA_rescue": "HTA",
    "High_to_Low": "High_to_Low", "Low_to_High": "Low_to_High",
    "Intermediate": "Low_to_High",
    "wildtype": "wildtype",
    "Not Penetrant": "Not Penetrant",
    "unlabeled": None,
}

DATASETS = {
    "b9d2":     "reference_b9d2_clean.csv",
    "cep290":   "reference_cep290_clean.csv",
    "crispant": "reference_crispant_clean.csv",
}


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"z_mu_b_\d+", c)]
    return sorted(cols, key=lambda c: int(c.split("_")[-1]))


def build_gene_figure(gene: str, ref_file: str, qry_all: pd.DataFrame, pheno_pred: pd.Series) -> None:
    ref = pd.read_csv(TABLE_DIR / ref_file, low_memory=False)
    feat_r = feature_cols(ref)
    if not feat_r:
        print(f"  [{gene}] no z_mu_b features in reference — skip")
        return

    ref = ref.dropna(subset=feat_r).copy()
    ref["source_view"] = "reference"
    ref["seq_view"] = "reference"
    ref["pheno_label"] = (
        ref["cluster_categories"].map(REF_LABEL_MAP)
        if "cluster_categories" in ref.columns else np.nan
    )
    ref["zygosity"] = ref["zygosity"].fillna("unknown") if "zygosity" in ref.columns else "unknown"

    qry = qry_all[qry_all["gene"] == gene].copy()
    if qry.empty:
        print(f"  [{gene}] no query rows — skip")
        return

    feat_q = feature_cols(qry)
    feat = sorted(set(feat_r) & set(feat_q), key=lambda c: int(c.split("_")[-1]))
    if not feat:
        print(f"  [{gene}] no overlapping z_mu_b features — skip")
        return

    qry = qry.dropna(subset=feat).copy()
    qry["pheno_label"] = qry[ID_COL].map(pheno_pred)
    qry["zygosity"] = qry["zygosity"].fillna("unknown") if "zygosity" in qry.columns else "unknown"

    # source view: distinguish sci timeseries from regular snapshots
    qry["source_view"] = qry["data_source"].map(
        {"timeseries": "timeseries (sci)", "snapshot": "snapshot"}
    ).fillna("snapshot")

    # sequenced view: 0=not_sequenced, 1=seq_wt(1), 2=seq_mutant(2)
    seq_map = {0.0: "not_sequenced", 1.0: "seq_wt(1)", 2.0: "seq_mutant(2)"}
    qry["seq_view"] = qry["sequenced"].map(seq_map).fillna("not_sequenced") \
        if "sequenced" in qry.columns else "not_sequenced"

    keep = feat + [ID_COL, TIME_COL, "gene", "zygosity", "genotype_clean", "experiment_id",
                   "source_view", "seq_view", "pheno_label"]
    comb = pd.concat([
        ref.reindex(columns=keep),
        qry.reindex(columns=keep),
    ], ignore_index=True).dropna(subset=feat)

    pca, scaler, _ = fit_pca_on_embeddings(comb, z_mu_cols=feat, n_components=3, scale=True)
    comb = transform_embeddings_to_pca(comb, pca, scaler, z_mu_cols=feat)
    evr = pca.explained_variance_ratio_
    nref = int((comb["source_view"] == "reference").sum())
    nq = int((comb["source_view"] != "reference").sum())
    print(f"  [{gene}] PCA EVR={evr.round(3).tolist()}  ref={nref}, query={nq}")

    # batch palette — auto-assign distinct colors per experiment
    batches = sorted(comb["experiment_id"].astype(str).unique())
    batch_pal = {b: px.colors.qualitative.Dark24[i % 24] for i, b in enumerate(batches)}

    axis_labels = {
        "PCA_1": f"PCA_1 ({evr[0]:.0%})",
        "PCA_2": f"PCA_2 ({evr[1]:.0%})",
        "PCA_3": f"PCA_3 ({evr[2]:.0%})",
    }

    # crispants are meaningfully distinguished by gene (genotype_clean), not zygosity
    if gene == "crispant":
        geno_view = {"label": "genotype", "color_by": "genotype_clean", "palette": GENOTYPE_COLORS}
    else:
        geno_view = {"label": "genotype", "color_by": "zygosity", "palette": ZYG_COLORS}

    views = [
        {"label": "HPF", "color_by": TIME_COL, "continuous": True,
         "colorbar_title": "hpf", "colorbar_thickness": 12, "colorbar_len": 0.35},
        geno_view,
    ]
    if comb["pheno_label"].notna().any():
        views.append({"label": "phenotype", "color_by": "pheno_label", "palette": PHENOTYPE_COLORS})
    views.append({"label": "experiment (batch)", "color_by": "experiment_id", "palette": batch_pal})
    views.append({"label": "source / data type", "color_by": "source_view", "palette": SOURCE_COLORS})
    views.append({"label": "sequenced", "color_by": "seq_view", "palette": SEQ_COLORS})

    out = OUT / f"{gene}_pca_3d.html"
    plot_3d_scatter(
        df=comb,
        coords=["PCA_1", "PCA_2", "PCA_3"],
        color_views=views,
        line_by=ID_COL,
        min_points_per_line=0,
        x_col=TIME_COL,
        point_size=3,
        point_opacity=0.6,
        hover_cols=["gene", "zygosity", "pheno_label", "source_view", "seq_view"],
        title=f"{gene} — 3D PCA",
        output_path=out,
        axis_labels=axis_labels,
    )
    print(f"  wrote plots/pca_3d/{out.name}  (views: {[v['label'] for v in views]})")


print("3h - 3D PCA plots per gene")

qry_all = pd.read_csv(TABLE_DIR / "query_all_rows_clean.csv", low_memory=False)
cross_bin = pd.read_csv(PRED_DIR / "sequenced_homozygous_phenotype_cross_bin.csv", low_memory=False)
pheno_pred = cross_bin.set_index("query_embryo_id")["predicted_label"]

for gene, ref_file in DATASETS.items():
    print(f"\n[{gene}]")
    build_gene_figure(gene, ref_file, qry_all, pheno_pred)

print(f"\nInteractive HTMLs under: plots/pca_3d/ — open in a browser, use the dropdown.")
