"""
1 - Fit reference models for sequenced-only SCI cilia QC.

This is intentionally written as a simple analysis script:
    load -> filter -> fit -> save
    load -> filter -> fit -> save

Step 0 creates the cleaned tables used here.
Step 2 will load these saved models and predict sequenced query embryos.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/1_fit_reference_models.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.analyze.classification.label_transfer import prepare_reference  # noqa: E402

TABLE_DIR = RUN_DIR / "tables"
MODEL_DIR = RUN_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

GROUP_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"

print("1 - fit reference models")
print("Reference models use all valid labeled reference embryos.")
print("Query data is used here only to choose latent features shared with each reference.")

reference_all = pd.read_csv(TABLE_DIR / "reference_all_clean.csv", low_memory=False)
query_all = pd.read_csv(TABLE_DIR / "query_all_rows_clean.csv", low_memory=False)
model_summary = []


# -----------------------------------------------------------------------------
# b9d2 genotype QC model: wildtype / heterozygous / homozygous
# -----------------------------------------------------------------------------

print("\nFitting b9d2 genotype QC model")
ref = reference_all[reference_all["gene"] == "b9d2"].copy()
query = query_all[query_all["gene"] == "b9d2"].copy()
ref = ref.dropna(subset=[TIME_COL, "zygosity"])
ref = ref[ref["zygosity"].isin(["wildtype", "heterozygous", "homozygous"])]
features = sorted(
    set(c for c in ref.columns if c.startswith("z_mu_b_")) & set(c for c in query.columns if c.startswith("z_mu_b_")),
    key=lambda c: int(c.split("_")[-1]),
)
model = prepare_reference(
    ref,
    features,
    label_col="zygosity",
    group_col=GROUP_COL,
    time_col=TIME_COL,
    cv_group_col="experiment_id",
    model_type="global",
)
with (MODEL_DIR / "b9d2_genotype_qc.pkl").open("wb") as fh:
    pickle.dump(model, fh)
model_summary.append({
    "model_id": "b9d2_genotype_qc",
    "gene": "b9d2",
    "label_col": "zygosity",
    "classes": ", ".join(model["classes"]),
    "n_reference_embryos": ref[GROUP_COL].nunique(),
    "n_features": len(features),
    "macro_f1": model["quality_report"]["macro_f1"],
    "balanced_accuracy": model["quality_report"]["balanced_accuracy"],
    "model_file": "models/b9d2_genotype_qc.pkl",
})
print(ref.groupby(GROUP_COL)["zygosity"].agg(lambda s: s.mode().iloc[0]).value_counts().to_string())


# -----------------------------------------------------------------------------
# cep290 genotype QC model: wildtype / heterozygous / homozygous
# -----------------------------------------------------------------------------

print("\nFitting cep290 genotype QC model")
ref = reference_all[reference_all["gene"] == "cep290"].copy()
query = query_all[query_all["gene"] == "cep290"].copy()
ref = ref.dropna(subset=[TIME_COL, "zygosity"])
ref = ref[ref["zygosity"].isin(["wildtype", "heterozygous", "homozygous"])]
features = sorted(
    set(c for c in ref.columns if c.startswith("z_mu_b_")) & set(c for c in query.columns if c.startswith("z_mu_b_")),
    key=lambda c: int(c.split("_")[-1]),
)
model = prepare_reference(
    ref,
    features,
    label_col="zygosity",
    group_col=GROUP_COL,
    time_col=TIME_COL,
    cv_group_col="experiment_id",
    model_type="global",
)
with (MODEL_DIR / "cep290_genotype_qc.pkl").open("wb") as fh:
    pickle.dump(model, fh)
model_summary.append({
    "model_id": "cep290_genotype_qc",
    "gene": "cep290",
    "label_col": "zygosity",
    "classes": ", ".join(model["classes"]),
    "n_reference_embryos": ref[GROUP_COL].nunique(),
    "n_features": len(features),
    "macro_f1": model["quality_report"]["macro_f1"],
    "balanced_accuracy": model["quality_report"]["balanced_accuracy"],
    "model_file": "models/cep290_genotype_qc.pkl",
})
print(ref.groupby(GROUP_COL)["zygosity"].agg(lambda s: s.mode().iloc[0]).value_counts().to_string())


# -----------------------------------------------------------------------------
# cilia crispant genotype QC model: crispant and control labels
# -----------------------------------------------------------------------------

print("\nFitting cilia crispant genotype QC model")
ref = reference_all[reference_all["gene"] == "crispant"].copy()
query = query_all[query_all["gene"] == "crispant"].copy()
ref = ref.dropna(subset=[TIME_COL, "genotype_clean"])
ref = ref[~ref["genotype_clean"].isin(["unknown", "nan"])]
features = sorted(
    set(c for c in ref.columns if c.startswith("z_mu_b_")) & set(c for c in query.columns if c.startswith("z_mu_b_")),
    key=lambda c: int(c.split("_")[-1]),
)
model = prepare_reference(
    ref,
    features,
    label_col="genotype_clean",
    group_col=GROUP_COL,
    time_col=TIME_COL,
    cv_group_col=None,
    model_type="global",
)
with (MODEL_DIR / "cilia_crispant_genotype_qc.pkl").open("wb") as fh:
    pickle.dump(model, fh)
model_summary.append({
    "model_id": "cilia_crispant_genotype_qc",
    "gene": "crispant",
    "label_col": "genotype_clean",
    "classes": ", ".join(model["classes"]),
    "n_reference_embryos": ref[GROUP_COL].nunique(),
    "n_features": len(features),
    "macro_f1": model["quality_report"]["macro_f1"],
    "balanced_accuracy": model["quality_report"]["balanced_accuracy"],
    "model_file": "models/cilia_crispant_genotype_qc.pkl",
})
print(ref.groupby(GROUP_COL)["genotype_clean"].agg(lambda s: s.mode().iloc[0]).value_counts().to_string())


# -----------------------------------------------------------------------------
# b9d2 homozygous phenotype model: CE / HTA
# -----------------------------------------------------------------------------

print("\nFitting b9d2 homozygous phenotype model")
ref = reference_all[reference_all["gene"] == "b9d2"].copy()
query = query_all[query_all["gene"] == "b9d2"].copy()
ref = ref.dropna(subset=[TIME_COL, "phenotype_clean"])
ref = ref[ref["zygosity"] == "homozygous"]
ref = ref[ref["phenotype_clean"].isin(["CE", "HTA"])]
features = sorted(
    set(c for c in ref.columns if c.startswith("z_mu_b_")) & set(c for c in query.columns if c.startswith("z_mu_b_")),
    key=lambda c: int(c.split("_")[-1]),
)
model = prepare_reference(
    ref,
    features,
    label_col="phenotype_clean",
    group_col=GROUP_COL,
    time_col=TIME_COL,
    cv_group_col="experiment_id",
    model_type="global",
)
with (MODEL_DIR / "b9d2_homozygous_phenotype.pkl").open("wb") as fh:
    pickle.dump(model, fh)
model_summary.append({
    "model_id": "b9d2_homozygous_phenotype",
    "gene": "b9d2",
    "label_col": "phenotype_clean",
    "classes": ", ".join(model["classes"]),
    "n_reference_embryos": ref[GROUP_COL].nunique(),
    "n_features": len(features),
    "macro_f1": model["quality_report"]["macro_f1"],
    "balanced_accuracy": model["quality_report"]["balanced_accuracy"],
    "model_file": "models/b9d2_homozygous_phenotype.pkl",
})
print(ref.groupby(GROUP_COL)["phenotype_clean"].agg(lambda s: s.mode().iloc[0]).value_counts().to_string())


# -----------------------------------------------------------------------------
# cep290 homozygous phenotype model: High_to_Low / Low_to_High
# -----------------------------------------------------------------------------

print("\nFitting cep290 homozygous phenotype model")
ref = reference_all[reference_all["gene"] == "cep290"].copy()
query = query_all[query_all["gene"] == "cep290"].copy()
ref = ref.dropna(subset=[TIME_COL, "phenotype_clean"])
ref = ref[ref["zygosity"] == "homozygous"]
ref = ref[ref["phenotype_clean"].isin(["High_to_Low", "Low_to_High"])]
features = sorted(
    set(c for c in ref.columns if c.startswith("z_mu_b_")) & set(c for c in query.columns if c.startswith("z_mu_b_")),
    key=lambda c: int(c.split("_")[-1]),
)
model = prepare_reference(
    ref,
    features,
    label_col="phenotype_clean",
    group_col=GROUP_COL,
    time_col=TIME_COL,
    cv_group_col="experiment_id",
    model_type="per_bin",
)
with (MODEL_DIR / "cep290_homozygous_phenotype.pkl").open("wb") as fh:
    pickle.dump(model, fh)
model_summary.append({
    "model_id": "cep290_homozygous_phenotype",
    "gene": "cep290",
    "label_col": "phenotype_clean",
    "classes": ", ".join(model["classes"]),
    "n_reference_embryos": ref[GROUP_COL].nunique(),
    "n_features": len(features),
    "macro_f1": model["quality_report"]["macro_f1"],
    "balanced_accuracy": model["quality_report"]["balanced_accuracy"],
    "model_file": "models/cep290_homozygous_phenotype.pkl",
})
print(ref.groupby(GROUP_COL)["phenotype_clean"].agg(lambda s: s.mode().iloc[0]).value_counts().to_string())


summary = pd.DataFrame(model_summary)
summary.to_csv(MODEL_DIR / "model_summary.csv", index=False)
print(f"\nWrote models and summary to: {MODEL_DIR.relative_to(RUN_DIR)}/")
