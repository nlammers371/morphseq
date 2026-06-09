"""
2 - Predict SEQUENCED query embryos using saved per-bin reference models.

==============================================================================
SCOPE — SEQUENCED EMBRYOS ONLY.
    Every embryo predicted, benchmarked and written by this script is in the
    sequenced registry (query_sequenced_embryos.csv, sequenced > 0). Unsequenced
    query embryos are deliberately excluded — this script is the sequencing
    greenlight read-out, not a whole-cohort prediction. The filter is applied
    ONCE, up front (see the SEQUENCED-ONLY line below), so everything downstream
    is sequenced-only by construction.
==============================================================================

Plain analysis script:
    load tables -> filter to sequenced -> per model: select query -> transfer -> save

The per-bin engine returns TWO grains (both written out):
    - cross-bin : one row per query embryo (the headline prediction).
    - per-bin   : one row per query embryo x time bin (feeds the confidence plot).
Plus missing_support: query (embryo, bin) rows with no model -> no prediction.

For each model, we:
1. subset SEQUENCED query rows to the gene group (phenotype models: homozygous only)
2. apply timeseries-priority selection (plate01 LOUD rule)
3. load the saved per-bin reference model from models/
4. run transfer_labels_perbin and attach embryo metadata
5. collect cross-bin + per-bin outputs
6. benchmark genotype QC cross-bin predictions against sequencing truth

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
sys.path.insert(0, str(RUN_DIR))  # so the local cilia_qc_helpers module is importable

from src.analyze.classification.label_transfer import transfer_labels_perbin  # noqa: E402
from cilia_qc_helpers import select_for_label_transfer  # noqa: E402

TABLE_DIR = RUN_DIR / "tables"
MODEL_DIR = RUN_DIR / "models"
PRED_DIR = RUN_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)

QUERY_PATH = TABLE_DIR / "query_all_rows_clean.csv"
SEQUENCED_REGISTRY_PATH = TABLE_DIR / "query_sequenced_embryos.csv"

print("=" * 70)
print("2 - PREDICT  |  SCOPE: SEQUENCED EMBRYOS ONLY (sequenced > 0)")
print("    Unsequenced query embryos are excluded — this is the sequencing")
print("    greenlight read-out, not a whole-cohort prediction.")
print("=" * 70)

query_all = pd.read_csv(QUERY_PATH, low_memory=False)
registry = pd.read_csv(SEQUENCED_REGISTRY_PATH, low_memory=False)
registry = registry[["embryo_id", "sequenced", "sequenced_stratum"]].drop_duplicates("embryo_id")

# ==========================================================================
# SEQUENCED-ONLY FILTER (applied ONCE, here). Everything downstream inherits it.
# Every embryo evaluated below is in the sequenced registry (sequenced > 0).
# ==========================================================================
n_all = query_all["embryo_id"].nunique()
query = query_all[query_all["embryo_id"].isin(registry["embryo_id"])].copy()
print(f"\nSEQUENCED-ONLY filter: {query['embryo_id'].nunique()} of {n_all} query embryos "
      f"are sequenced; the other {n_all - query['embryo_id'].nunique()} are dropped here.")
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
    "collection_time_hpf",
    "data_source",
    "physical_embryo_id",
]

# Every field is set on every row. Nothing is implied by absence.
#
# The main fields that differ between models are:
#   homozygous_only
#       Whether to restrict the query to homozygous embryos.
#       True for phenotype models.
#
#   benchmark_truth
#       Truth column used for scoring predictions.
#       None means the model is prediction-only and not benchmarked.
#
#   valid_truth_labels
#       Allowed truth labels for benchmarkable models.
#       Empty for non-benchmarkable phenotype models.

prediction_plan = [
    {
        "model_id": "b9d2_genotype_qc",
        "gene": "b9d2",
        "prediction_kind": "genotype_qc",
        "homozygous_only": False,
        "benchmark_truth": "zygosity",
        "valid_truth_labels": ["wildtype", "heterozygous", "homozygous"],
    },
    {
        "model_id": "cep290_genotype_qc",
        "gene": "cep290",
        "prediction_kind": "genotype_qc",
        "homozygous_only": False,
        "benchmark_truth": "zygosity",
        "valid_truth_labels": ["wildtype", "heterozygous", "homozygous"],
    },
    {
        "model_id": "cilia_crispant_genotype_qc",
        "gene": "crispant",
        "prediction_kind": "genotype_qc",
        "homozygous_only": False,
        "benchmark_truth": "genotype_clean",
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
        "homozygous_only": True,
        "benchmark_truth": None,
        "valid_truth_labels": [],
    },
    {
        "model_id": "cep290_homozygous_phenotype",
        "gene": "cep290",
        "prediction_kind": "homozygous_phenotype",
        "homozygous_only": True,
        "benchmark_truth": None,
        "valid_truth_labels": [],
    },
]

all_cross_bin = []   # one row per (model, query embryo)
all_per_bin = []     # one row per (model, query embryo, time bin)
all_missing = []     # query (embryo, bin) with no model -> missing support
summary_rows = []

print("\nLOUD: query uses timeseries-priority selection per physical embryo")
print("(plate01 timeseries wins over its redundant _t02 snapshot; plate02/crispant snapshots kept).")

for model_spec in prediction_plan:
    model_id = model_spec["model_id"]
    gene = model_spec["gene"]
    prediction_kind = model_spec["prediction_kind"]
    benchmark_truth = model_spec["benchmark_truth"]
    valid_truth_labels = model_spec["valid_truth_labels"]

    print(f"\n[{model_id}] gene={gene} | kind={prediction_kind}")
    query_for_model = query[query["gene"] == gene].copy()
    if model_spec["homozygous_only"]:
        query_for_model = query_for_model[query_for_model["zygosity"] == "homozygous"]
        print("  homozygous_only: restricted query to homozygous embryos")
    if query_for_model.empty:
        print(f"  no sequenced query rows for gene={gene} — skipping")
        continue

    # plate01 special case: a plate01 48 hpf embryo has BOTH a timeseries AND a redundant
    # _t02 snapshot backup (same physical_embryo_id). Drop the backup so the model uses the
    # timeseries. plate02 (snapshot-only) and crispants (no timeseries) are untouched.
    n_before = query_for_model["embryo_id"].nunique()
    query_for_model = select_for_label_transfer(query_for_model)
    n_after = query_for_model["embryo_id"].nunique()
    print(f"  query embryos: {n_before} -> {n_after} after timeseries-priority select; "
          f"rows={len(query_for_model)}")

    with (MODEL_DIR / f"{model_id}.pkl").open("rb") as fh:
        model = pickle.load(fh)

    result = transfer_labels_perbin(model, query_for_model, verbose=True)

    embryo_metadata = query_for_model[meta_cols].drop_duplicates("embryo_id")

    # Cross-bin: one row per query embryo (the headline prediction).
    cross_bin = result["embryo_support"]["embryo_cross_bin_prediction"].merge(
        embryo_metadata, left_on="query_embryo_id", right_on="embryo_id", how="left"
    )
    cross_bin.insert(0, "model_id", model_id)
    cross_bin.insert(1, "prediction_kind", prediction_kind)
    all_cross_bin.append(cross_bin)

    # Per-bin: one row per query embryo x time bin (feeds the confidence plot).
    per_bin = result["per_bin"]["embryo_per_bin_prediction"].merge(
        embryo_metadata, left_on="query_embryo_id", right_on="embryo_id", how="left"
    )
    per_bin.insert(0, "model_id", model_id)
    per_bin.insert(1, "prediction_kind", prediction_kind)
    all_per_bin.append(per_bin)

    missing = result["missing_support"].copy()
    if not missing.empty:
        missing.insert(0, "model_id", model_id)
        all_missing.append(missing)

    perf = model["reference_performance"]
    summary_rows.append({
        "model_id": model_id, "gene": gene, "prediction_kind": prediction_kind,
        "n_query_embryos": int(cross_bin["query_embryo_id"].nunique()),
        "n_query_embryo_bins": int(len(per_bin)),
        "n_query_missing_support": int(missing["query_embryo_id"].nunique()) if not missing.empty else 0,
        "predicted_classes": ", ".join(sorted(cross_bin["predicted_label"].astype(str).unique())),
        "mean_top_probability": float(cross_bin["top_probability"].mean()) if not cross_bin.empty else float("nan"),
        "reference_transferability": "; ".join(f"{k}:{v}" for k, v in perf["transferability"].items()),
    })

    # Benchmark against sequencing truth when this model has one (genotype QC only).
    if benchmark_truth is not None:
        with_truth = cross_bin[cross_bin[benchmark_truth].isin(valid_truth_labels)].copy()
        if with_truth.empty:
            print("  no benchmarkable sequenced truth labels")
        else:
            correct = with_truth["predicted_label"].astype(str) == with_truth[benchmark_truth].astype(str)
            print(f"  benchmarkable accuracy (cross-bin): "
                  f"{correct.mean():.1%} ({int(correct.sum())}/{len(with_truth)})")

cross_bin_out = pd.concat(all_cross_bin, ignore_index=True) if all_cross_bin else pd.DataFrame()
per_bin_out = pd.concat(all_per_bin, ignore_index=True) if all_per_bin else pd.DataFrame()
missing_out = pd.concat(all_missing, ignore_index=True) if all_missing else pd.DataFrame()
summary = pd.DataFrame(summary_rows)

cross_bin_path = PRED_DIR / "sequenced_cross_bin_predictions.csv"
per_bin_path = PRED_DIR / "sequenced_per_bin_predictions.csv"
missing_path = PRED_DIR / "sequenced_missing_support.csv"
summary_path = PRED_DIR / "prediction_summary.csv"

cross_bin_out.to_csv(cross_bin_path, index=False)
per_bin_out.to_csv(per_bin_path, index=False)
missing_out.to_csv(missing_path, index=False)
summary.to_csv(summary_path, index=False)

# Split by prediction_kind so genotype-QC and phenotype plots read focused tables.
for kind, tag in [("genotype_qc", "genotype_qc"), ("homozygous_phenotype", "homozygous_phenotype")]:
    if not cross_bin_out.empty:
        cross_bin_out[cross_bin_out["prediction_kind"] == kind].to_csv(
            PRED_DIR / f"sequenced_{tag}_cross_bin.csv", index=False)
    if not per_bin_out.empty:
        per_bin_out[per_bin_out["prediction_kind"] == kind].to_csv(
            PRED_DIR / f"sequenced_{tag}_per_bin.csv", index=False)

print(f"\nWrote cross-bin (per-embryo) predictions to: {cross_bin_path.relative_to(RUN_DIR)}")
print(f"Wrote per-bin (per-embryo-bin) predictions to: {per_bin_path.relative_to(RUN_DIR)}")
print(f"Wrote missing-support rows to: {missing_path.relative_to(RUN_DIR)}")
print(f"Wrote prediction summary to: {summary_path.relative_to(RUN_DIR)}")
print("Wrote split prediction files under predictions/ for genotype QC and phenotype plots.")
