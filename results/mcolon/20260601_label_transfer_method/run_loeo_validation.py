"""
Leave-one-experiment-out (LOEO) validation for CEP290 phenotype label transfer.

For each of 7 experiments, holds it out as query data and uses the remaining
experiments as the labeled reference. Compares predicted labels to known labels.

Usage:
    conda run -n morphseq-env --no-capture-output python run_loeo_validation.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from the results directory directly
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # morphseq root
sys.path.insert(0, str(ROOT))

from results.mcolon._20260601_label_transfer_method.label_transfer_core import run_label_transfer


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
OUT_DIR = HERE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LABEL_COL = "cluster_categories"
EMBRYO_COL = "embryo_id"
SNIP_COL = "snip_id"
TIME_COL = "predicted_stage_hpf"
EXPERIMENT_COL = "experiment_id"
MIN_HPF = 30.0
MAX_HPF = 48.0
K = 15

EXPERIMENTS = [
    "20250512",
    "20251017_combined",
    "20251106",
    "20251112",
    "20251113",
    "20251205",
    "20251212",
]


def get_feature_cols(df: pd.DataFrame) -> list:
    return sorted([c for c in df.columns if c.startswith("z_mu_b_")])


def check_label_consistency(df: pd.DataFrame, embryo_col: str, label_col: str):
    """Warn if any embryo has multiple distinct labels in the data."""
    counts = df.groupby(embryo_col)[label_col].nunique()
    multi_label = counts[counts > 1]
    if len(multi_label) > 0:
        warnings.warn(
            f"{len(multi_label)} embryos have multiple distinct labels in {label_col}. "
            f"Ground truth join may be ambiguous. Embryos: {list(multi_label.index[:5])} ..."
        )


def get_embryo_true_labels(df: pd.DataFrame, embryo_col: str, label_col: str) -> dict:
    """Return most common label per embryo (mode) as ground truth."""
    return (
        df[df[label_col].notna()]
        .groupby(embryo_col)[label_col]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )


def compute_fold_metrics(fold_df: pd.DataFrame) -> dict:
    evaluated = fold_df[fold_df["status"] != "not_evaluated"]
    assigned = fold_df[fold_df["status"] == "assigned"]

    def accuracy(sub):
        if len(sub) == 0:
            return np.nan
        correct = (sub["predicted_label"] == sub["true_label"]).sum()
        return correct / len(sub)

    return {
        "n_query_embryos": len(fold_df),
        "n_assigned": (fold_df["status"] == "assigned").sum(),
        "n_ambiguous": (fold_df["status"] == "ambiguous").sum(),
        "n_low_density": (fold_df["status"] == "low_density").sum(),
        "n_not_evaluated": (fold_df["status"] == "not_evaluated").sum(),
        "accuracy_all_evaluated": accuracy(evaluated),
        "accuracy_assigned_only": accuracy(assigned),
        "mean_embryo_confidence": fold_df["embryo_confidence"].mean(),
        "mean_neighbor_agreement": fold_df["mean_image_neighbor_agreement"].mean(),
        "mean_distance_score": fold_df["embryo_distance_score"].mean(),
        "mean_consistency_score": fold_df["embryo_consistency_score"].mean(),
    }


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Loaded {len(df)} rows, {df[EXPERIMENT_COL].nunique()} experiments")

    feature_cols = get_feature_cols(df)
    print(f"  Using {len(feature_cols)} z_mu_b_* feature columns")

    # Drop rows with no label (but keep for query use)
    df_labeled = df[df[LABEL_COL].notna()].copy()
    check_label_consistency(df_labeled, EMBRYO_COL, LABEL_COL)

    experiments_in_data = df[EXPERIMENT_COL].unique()
    requested = set(EXPERIMENTS)
    missing = requested - set(experiments_in_data)
    if missing:
        warnings.warn(f"Experiments not found in data: {missing}")

    all_fold_predictions = []
    all_fold_summaries = []

    for exp_id in EXPERIMENTS:
        if exp_id not in experiments_in_data:
            print(f"\n[SKIP] {exp_id} not found in data")
            continue

        print(f"\n{'='*60}")
        print(f"Fold: hold out {exp_id}")

        # Reference: all other labeled rows
        ref_df = df_labeled[df_labeled[EXPERIMENT_COL] != exp_id].copy()

        # Query: all rows from this experiment (labels withheld from run_label_transfer)
        qry_df = df[df[EXPERIMENT_COL] == exp_id].copy()
        qry_df_no_labels = qry_df.drop(columns=[LABEL_COL], errors="ignore")

        print(f"  Reference: {len(ref_df)} rows, {ref_df[EMBRYO_COL].nunique()} embryos")
        print(f"  Query:     {len(qry_df)} rows, {qry_df[EMBRYO_COL].nunique()} embryos")

        results = run_label_transfer(
            reference_df=ref_df,
            query_df=qry_df_no_labels,
            feature_cols=feature_cols,
            label_col=LABEL_COL,
            embryo_col=EMBRYO_COL,
            snip_col=SNIP_COL,
            time_col=TIME_COL,
            experiment_col=EXPERIMENT_COL,
            min_hpf=MIN_HPF,
            max_hpf=MAX_HPF,
            k=K,
        )

        summary = results["embryo_label_transfer_summary"].copy()

        # Join true labels AFTER prediction
        true_labels = get_embryo_true_labels(qry_df, EMBRYO_COL, LABEL_COL)
        summary["true_label"] = summary["query_embryo_id"].map(true_labels)
        summary["heldout_experiment_id"] = exp_id

        all_fold_predictions.append(summary)

        # Per-fold QC
        metrics = compute_fold_metrics(summary)
        all_fold_summaries.append({"heldout_experiment_id": exp_id, **metrics})

        print(f"  Status breakdown:")
        status_counts = summary["status"].value_counts()
        for status, count in status_counts.items():
            print(f"    {status}: {count}")
        print(f"  accuracy_all_evaluated : {metrics['accuracy_all_evaluated']:.3f}")
        print(f"  accuracy_assigned_only : {metrics['accuracy_assigned_only']:.3f}")
        print(f"  mean_embryo_confidence : {metrics['mean_embryo_confidence']:.3f}")

    # Save outputs
    if all_fold_predictions:
        pred_df = pd.concat(all_fold_predictions, ignore_index=True)
        out_pred = OUT_DIR / "leave_one_experiment_out_predictions.csv"
        pred_df.to_csv(out_pred, index=False)
        print(f"\nSaved predictions to {out_pred}")

    if all_fold_summaries:
        sum_df = pd.DataFrame(all_fold_summaries)
        out_sum = OUT_DIR / "leave_one_experiment_out_summary.csv"
        sum_df.to_csv(out_sum, index=False)
        print(f"Saved summary to {out_sum}")

        print("\n" + "="*60)
        print("Overall LOEO summary:")
        print(sum_df[["heldout_experiment_id", "n_query_embryos", "accuracy_all_evaluated",
                        "accuracy_assigned_only", "mean_embryo_confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
