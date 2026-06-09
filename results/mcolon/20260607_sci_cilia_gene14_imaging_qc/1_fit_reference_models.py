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

from src.analyze.classification.label_transfer import prepare_reference_perbin  # noqa: E402

TABLE_DIR = RUN_DIR / "tables"
MODEL_DIR = RUN_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

GROUP_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
CV_GROUP_COL = "experiment_id"
BIN_WIDTH = 4.0

print("1 - fit reference models (per-bin engine)")
print("All models are fit per time bin: one point per (embryo, bin), no global path.")
print("CV mode is chosen explicitly per model (never inferred):")
print("  b9d2 / cep290 -> 'loeo' (leave-one-experiment-out; references span >=2 experiments)")
print("  crispant      -> 'kfold' (single-experiment reference; LOEO-experiment impossible)")
print("Features = z_mu_b_* biological latents only. Query data is used here only to choose")
print("the latent features shared with each reference.")

reference_all = pd.read_csv(TABLE_DIR / "reference_all_clean.csv", low_memory=False)
query_all = pd.read_csv(TABLE_DIR / "query_all_rows_clean.csv", low_memory=False)
model_summary = []


# Each spec is one load -> filter -> fit -> save block. cv_mode is chosen EXPLICITLY:
# 'loeo' (leave-one-experiment-out) where the reference spans >=2 experiments; 'kfold'
# for crispant whose reference is a single experiment (LOEO-experiment is impossible).
MODEL_SPECS = [
    {"model_id": "b9d2_genotype_qc", "gene": "b9d2", "label_col": "zygosity",
     "keep_labels": ["wildtype", "heterozygous", "homozygous"], "cv_mode": "loeo"},
    {"model_id": "cep290_genotype_qc", "gene": "cep290", "label_col": "zygosity",
     "keep_labels": ["wildtype", "heterozygous", "homozygous"], "cv_mode": "loeo"},
    {"model_id": "cilia_crispant_genotype_qc", "gene": "crispant", "label_col": "genotype_clean",
     "drop_labels": ["unknown", "nan"], "cv_mode": "kfold"},
    {"model_id": "b9d2_homozygous_phenotype", "gene": "b9d2", "label_col": "phenotype_clean",
     "homozygous_only": True, "keep_labels": ["CE", "HTA"], "cv_mode": "loeo"},
    {"model_id": "cep290_homozygous_phenotype", "gene": "cep290", "label_col": "phenotype_clean",
     "homozygous_only": True, "keep_labels": ["High_to_Low", "Low_to_High"], "cv_mode": "loeo"},
]


def shared_z_mu_b_features(ref: pd.DataFrame, query: pd.DataFrame) -> list[str]:
    """The 80 z_mu_b_* biological latents present in BOTH ref and query, in index order."""
    return sorted(
        {c for c in ref.columns if c.startswith("z_mu_b_")}
        & {c for c in query.columns if c.startswith("z_mu_b_")},
        key=lambda c: int(c.split("_")[-1]),
    )


for spec in MODEL_SPECS:
    model_id, gene, label_col = spec["model_id"], spec["gene"], spec["label_col"]
    print(f"\nFitting {model_id} (gene={gene}, label={label_col}, cv_mode={spec['cv_mode']})")

    ref = reference_all[reference_all["gene"] == gene].copy()
    query = query_all[query_all["gene"] == gene].copy()
    ref = ref.dropna(subset=[TIME_COL, label_col])
    if spec.get("homozygous_only"):
        ref = ref[ref["zygosity"] == "homozygous"]
    if "keep_labels" in spec:
        ref = ref[ref[label_col].isin(spec["keep_labels"])]
    if "drop_labels" in spec:
        ref = ref[~ref[label_col].isin(spec["drop_labels"])]

    features = shared_z_mu_b_features(ref, query)
    model = prepare_reference_perbin(
        ref, features,
        label_col=label_col, group_col=GROUP_COL, time_col=TIME_COL,
        bin_width=BIN_WIDTH, cv_mode=spec["cv_mode"], cv_group_col=CV_GROUP_COL,
        verbose=True,
    )

    with (MODEL_DIR / f"{model_id}.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    # Reference CV (one row per reference embryo-bin) feeds the confidence plot.
    model["per_bin"]["embryo_per_bin_prediction"].to_csv(
        MODEL_DIR / f"{model_id}_reference_cv.csv", index=False
    )

    perf = model["reference_performance"]
    model_summary.append({
        "model_id": model_id, "gene": gene, "label_col": label_col,
        "cv_mode": spec["cv_mode"],
        "classes": ", ".join(model["classes"]),
        "n_reference_embryos": ref[GROUP_COL].nunique(),
        "n_features": len(features),
        "n_bins_scored": model["embryo_support"]["n_bins_scored"],
        "n_bins_failed": len(model["missing_bins"]),
        "transferability": "; ".join(f"{k}:{v}" for k, v in perf["transferability"].items()),
        "model_file": f"models/{model_id}.pkl",
    })
    print(f"  scored bins={model['embryo_support']['n_bins_scored']}, "
          f"failed bins={len(model['missing_bins'])}, "
          f"transferability={perf['transferability']}")
    print(ref.groupby(GROUP_COL)[label_col].agg(lambda s: s.mode().iloc[0]).value_counts().to_string())


summary = pd.DataFrame(model_summary)
summary.to_csv(MODEL_DIR / "model_summary.csv", index=False)
print(f"\nWrote per-bin models + reference CV + summary to: {MODEL_DIR.relative_to(RUN_DIR)}/")
