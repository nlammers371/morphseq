"""
2 - Predict sequenced query embryos using saved reference models.

This is a plain analysis script:
    load tables -> define prediction plan -> load each saved model -> predict -> save CSVs

The only loop in this script is the prediction plan. Each row in the plan is one
saved model we want to apply to sequenced query embryos.
For each model, we:
1. subset sequenced query images to the relevant gene group
2. load the saved reference model from models/
3. run label transfer on those query images
4. attach embryo and snip metadata back onto the predictions
5. collect embryo-level and image-level outputs
6. benchmark genotype QC predictions against sequencing truth when available

It predicts only sequenced query embryos and writes out:
- embryo-level genotype QC predictions
- embryo-level homozygous phenotype predictions
- image-level prediction tables for both model types
- a compact per-model prediction summary

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260607_sci_cilia_gene14_imaging_qc/2_predict_sequenced_embryos.py
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

from src.analyze.classification.label_transfer import transfer_labels  # noqa: E402

TABLE_DIR = RUN_DIR / "tables"
MODEL_DIR = RUN_DIR / "models"
PRED_DIR = RUN_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)

QUERY_PATH = TABLE_DIR / "query_all_rows_clean.csv"
SEQUENCED_REGISTRY_PATH = TABLE_DIR / "query_sequenced_embryos.csv"

print("2 - predict sequenced embryos")
print("Load saved models.")
print("Predict only sequenced query embryos.")
print("Save genotype QC predictions and homozygous phenotype predictions.")

query_all = pd.read_csv(QUERY_PATH, low_memory=False)
registry = pd.read_csv(SEQUENCED_REGISTRY_PATH, low_memory=False)
registry = registry[["embryo_id", "sequenced", "sequenced_stratum"]].drop_duplicates("embryo_id")

query = query_all[query_all["embryo_id"].isin(registry["embryo_id"])].copy()
query = query.merge(
    registry,
    on="embryo_id",
    how="left",
    suffixes=("", "_registry"),
)
for col in ["sequenced", "sequenced_stratum"]:
    registry_col = f"{col}_registry"
    if registry_col in query.columns:
        query[col] = query[registry_col].combine_first(query[col])
        query = query.drop(columns=[registry_col])

print(f"Loaded {len(query)} sequenced query rows across {query['gene'].nunique()} genes")
print(f"Loaded sequenced registry for {len(registry)} embryos")

meta_cols = [
    "embryo_id",
    "experiment",
    "gene",
    "well",
    "sequenced",
    "sequenced_stratum",
    "genotype_clean",
    "zygosity",
    "phenotype_clean",
    "predicted_stage_hpf",
]

prediction_plan = [
    {
        "model_id": "b9d2_genotype_qc",
        "gene": "b9d2",
        "prediction_kind": "genotype_qc",
        "truth_col": "zygosity",
        "valid_truth_labels": ["wildtype", "heterozygous", "homozygous"],
    },
    {
        "model_id": "cep290_genotype_qc",
        "gene": "cep290",
        "prediction_kind": "genotype_qc",
        "truth_col": "zygosity",
        "valid_truth_labels": ["wildtype", "heterozygous", "homozygous"],
    },
    {
        "model_id": "cilia_crispant_genotype_qc",
        "gene": "crispant",
        "prediction_kind": "genotype_qc",
        "truth_col": "genotype_clean",
        "valid_truth_labels": [
            "ab_wildtype",
            "foxj1a_crispant",
            "ift88_crispant",
            "ift88_ift74_crispant",
            "sspo_crispant",
        ],
    },
    {
        "model_id": "b9d2_homozygous_phenotype",
        "gene": "b9d2",
        "prediction_kind": "homozygous_phenotype",
    },
    {
        "model_id": "cep290_homozygous_phenotype",
        "gene": "cep290",
        "prediction_kind": "homozygous_phenotype",
    },
]

all_embryo = []
all_image = []
summary_rows = []

for model_spec in prediction_plan:
    model_id = model_spec["model_id"]
    gene = model_spec["gene"]
    prediction_kind = model_spec["prediction_kind"]
    truth_col = model_spec.get("truth_col")
    valid_truth_labels = model_spec.get("valid_truth_labels", [])

    print(f"\n[{model_id}] gene={gene} | prediction_kind={prediction_kind}")
    query_for_model = query[query["gene"] == gene].copy()
    if query_for_model.empty:
        print(f"  no sequenced query rows for gene={gene} — skipping")
        continue

    print(f"  query_rows={len(query_for_model)}")

    # Load the saved reference model.
    model_path = MODEL_DIR / f"{model_id}.pkl"
    with model_path.open("rb") as fh:
        model = pickle.load(fh)

    # Predict embryo-level and image-level labels.
    result = transfer_labels(model, query_for_model, skip_flagged=False)

    # Attach embryo metadata.
    embryo_predictions = result["embryo_predictions"].copy()
    embryo_metadata = query_for_model[meta_cols].drop_duplicates("embryo_id")
    embryo_predictions = embryo_predictions.merge(
        embryo_metadata,
        left_on="query_embryo_id",
        right_on="embryo_id",
        how="left",
    )
    embryo_predictions.insert(0, "model_id", model_id)
    embryo_predictions.insert(1, "prediction_kind", prediction_kind)
    all_embryo.append(embryo_predictions)

    # Attach image metadata.
    image_predictions = result["image_predictions"].copy()
    image_metadata = query_for_model[["snip_id", *meta_cols]].drop_duplicates("snip_id")
    image_predictions = image_predictions.merge(
        image_metadata,
        left_on="query_snip_id",
        right_on="snip_id",
        how="left",
    )
    image_predictions.insert(0, "model_id", model_id)
    image_predictions.insert(1, "prediction_kind", prediction_kind)
    all_image.append(image_predictions)

    # Compact summary for this block.
    predicted_classes = sorted(
        pd.unique(embryo_predictions["predicted_label"].astype(str))
    )
    summary_rows.append(
        {
            "model_id": model_id,
            "gene": gene,
            "prediction_kind": prediction_kind,
            "n_query_embryos": int(embryo_predictions["query_embryo_id"].nunique()),
            "n_query_images": int(len(image_predictions)),
            "predicted_classes": ", ".join(predicted_classes),
            "mean_top_probability": float(embryo_predictions["top_probability"].mean()),
        }
    )

    # Only genotype QC predictions have matching sequencing truth labels.
    if prediction_kind == "genotype_qc":
        if truth_col is None or not valid_truth_labels:
            raise ValueError(f"{model_id} is genotype_qc but has no truth labels configured")

        predictions_with_truth = embryo_predictions[
            embryo_predictions[truth_col].isin(valid_truth_labels)
        ].copy()

        if predictions_with_truth.empty:
            print("  no benchmarkable sequenced truth labels")
        else:
            predicted_label = predictions_with_truth["predicted_label"].astype(str)
            true_label = predictions_with_truth[truth_col].astype(str)
            predictions_with_truth["correct"] = predicted_label == true_label
            n_correct = int(predictions_with_truth["correct"].sum())
            n_total = len(predictions_with_truth)
            accuracy = n_correct / n_total
            print(f"  benchmarkable accuracy: {accuracy:.1%} ({n_correct}/{n_total})")

embryo_out = pd.concat(all_embryo, ignore_index=True) if all_embryo else pd.DataFrame()
image_out = pd.concat(all_image, ignore_index=True) if all_image else pd.DataFrame()
summary = pd.DataFrame(summary_rows)

embryo_path = PRED_DIR / "sequenced_embryo_predictions.csv"
image_path = PRED_DIR / "sequenced_image_predictions.csv"
summary_path = PRED_DIR / "prediction_summary.csv"

embryo_out.to_csv(embryo_path, index=False)
image_out.to_csv(image_path, index=False)
summary.to_csv(summary_path, index=False)

if not embryo_out.empty:
    genotype_qc_embryo = embryo_out[embryo_out["prediction_kind"] == "genotype_qc"].copy()
    phenotype_embryo = embryo_out[embryo_out["prediction_kind"] == "homozygous_phenotype"].copy()

    genotype_qc_embryo.to_csv(
        PRED_DIR / "sequenced_genotype_qc_predictions.csv",
        index=False,
    )
    phenotype_embryo.to_csv(
        PRED_DIR / "sequenced_homozygous_phenotype_predictions.csv",
        index=False,
    )

if not image_out.empty:
    genotype_qc_image = image_out[image_out["prediction_kind"] == "genotype_qc"].copy()
    phenotype_image = image_out[image_out["prediction_kind"] == "homozygous_phenotype"].copy()

    genotype_qc_image.to_csv(
        PRED_DIR / "sequenced_genotype_qc_image_predictions.csv",
        index=False,
    )
    phenotype_image.to_csv(
        PRED_DIR / "sequenced_homozygous_phenotype_image_predictions.csv",
        index=False,
    )

print(f"\nWrote embryo predictions to: {embryo_path.relative_to(RUN_DIR)}")
print(f"Wrote image predictions to: {image_path.relative_to(RUN_DIR)}")
print(f"Wrote prediction summary to: {summary_path.relative_to(RUN_DIR)}")
print("Wrote split prediction files under predictions/ for genotype QC and phenotype plots.")
