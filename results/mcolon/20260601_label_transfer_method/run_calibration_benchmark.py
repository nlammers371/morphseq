"""
Calibration method bake-off on the reference leave-one-out (LOO) arena.

The reference LOO table gives us ground truth: for every reference image we know its
q-vector, its local_prior vector, and its TRUE label (with that image left out of its
own neighbor search). So every proposed calibration can be scored as:

    Given q_ref and local_prior_ref, can we recover true_label better than
    raw argmax(q_ref)?

We sweep the full grid of {likelihood estimator} × {prior} and report, per combination:
    overall accuracy, balanced accuracy, macro-F1,
    per-label precision / recall,
    LtH -> NP collapse rate, NP -> LtH false-call rate,
    confusion matrix.

Decision criterion (NOT raw accuracy): we want macro-F1 / balanced accuracy up and the
Low_to_High / Intermediate recall up WITHOUT exploding false positives, especially the
Low_to_High -> Not Penetrant collapse. Raw accuracy alone is gamed by the abundant
Not Penetrant class.

Baseline is always raw argmax(q) (= current method's image-level call).

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python run_calibration_benchmark.py
"""

from pathlib import Path
import sys
import itertools
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from calibration_core import (
    MAIN_LABELS,
    build_reference_loo_table,
    LIKELIHOOD_ESTIMATORS,
    PRIOR_ESTIMATORS,
    combine_posterior,
    calibrated_predictions,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

DATA_PATH = ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
OUT_DIR = HERE

LABEL_COL = "cluster_categories"
TIME_COL = "predicted_stage_hpf"
MIN_HPF = 30.0
MAX_HPF = 48.0
K_SMALL = 15
K_PRIOR = 100

# Which prior vector each prior reads: ring (disjoint from q) vs full (overlaps q).
# We test both ring and full for the local priors; uniform ignores the vectors.
PRIOR_VECTOR_SOURCES = {
    "uniform": ["ring"],            # vectors unused
    "raw_local": ["ring", "full"],
    "prevalence_corrected": ["ring", "full"],
}


def get_feature_cols(df):
    return sorted([c for c in df.columns if c.startswith("z_mu_b_")])


def _vector_block(loo_df, prefix, labels):
    return loo_df[[f"{prefix}{lbl}" for lbl in labels]].values.astype(float)


def evaluate(true_labels, pred_labels, labels):
    """Return a dict of headline + per-label metrics + collapse rates."""
    acc = accuracy_score(true_labels, pred_labels)
    bacc = balanced_accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, labels=labels, average="macro",
                        zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=labels, zero_division=0
    )

    out = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
    }
    for i, lbl in enumerate(labels):
        out[f"recall[{lbl}]"] = rec[i]
        out[f"precision[{lbl}]"] = prec[i]

    # Collapse rates: among TRUE LtH, fraction predicted NP; among TRUE NP, fraction
    # predicted LtH. These are the asymmetric-mixing danger directions.
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    lth_mask = true_arr == "Low_to_High"
    np_mask = true_arr == "Not Penetrant"
    out["LtH->NP_collapse"] = (
        (pred_arr[lth_mask] == "Not Penetrant").mean() if lth_mask.any() else np.nan
    )
    out["NP->LtH_falsecall"] = (
        (pred_arr[np_mask] == "Low_to_High").mean() if np_mask.any() else np.nan
    )
    return out


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df[LABEL_COL].notna()].copy()
    df = df[(df[TIME_COL] >= MIN_HPF) & (df[TIME_COL] <= MAX_HPF)].copy()
    df = df[df[LABEL_COL].isin(MAIN_LABELS)].copy()
    df = df.reset_index(drop=True)

    feature_cols = get_feature_cols(df)
    print(f"  {len(df)} labeled images, {len(feature_cols)} features")
    print(f"  Label counts:\n{df[LABEL_COL].value_counts().to_string()}\n")

    ref_X = df[feature_cols].values.astype(float)
    ref_labels = df[LABEL_COL].values

    print(f"Building reference LOO table (k_small={K_SMALL}, k_prior={K_PRIOR}) ...")
    loo = build_reference_loo_table(
        ref_X, ref_labels, labels=MAIN_LABELS,
        k_small=K_SMALL, k_prior=K_PRIOR,
    )
    loo.to_csv(OUT_DIR / "reference_loo_table.csv", index=False)
    print(f"  Saved reference_loo_table.csv ({len(loo)} rows)")

    true_labels = loo["true_label"].values
    q_ref = _vector_block(loo, "q_", MAIN_LABELS)         # the q-vectors (null)
    q_query = q_ref                                        # evaluating on the null itself

    # --- Baseline: raw argmax(q) ---
    print("\n=== BASELINE: raw argmax(q) ===")
    baseline = evaluate(true_labels, loo["raw_pred_label"].values, MAIN_LABELS)
    _print_metrics(baseline)

    results = [{"likelihood": "RAW_BASELINE", "prior": "-", "prior_src": "-", **baseline}]

    # --- Grid sweep ---
    for lik_name, lik_fn in LIKELIHOOD_ESTIMATORS.items():
        likelihood = lik_fn(q_query, q_ref, true_labels, MAIN_LABELS)

        for prior_name, prior_fn in PRIOR_ESTIMATORS.items():
            for src in PRIOR_VECTOR_SOURCES[prior_name]:
                prefix = "local_prior_" if src == "ring" else "local_prior_full_"
                prior_vectors = _vector_block(loo, prefix, MAIN_LABELS)
                prior = prior_fn(prior_vectors, true_labels, MAIN_LABELS)

                posterior = combine_posterior(likelihood, prior)
                pred, _ = calibrated_predictions(posterior, MAIN_LABELS)
                metrics = evaluate(true_labels, pred, MAIN_LABELS)
                results.append({
                    "likelihood": lik_name,
                    "prior": prior_name,
                    "prior_src": src,
                    **metrics,
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "calibration_benchmark_results.csv", index=False)
    print(f"\nSaved calibration_benchmark_results.csv")

    # --- Ranked summary on the decision criteria ---
    print("\n" + "=" * 90)
    print("RANKED BY macro_f1 (primary), then balanced_accuracy")
    print("=" * 90)
    show_cols = [
        "likelihood", "prior", "prior_src",
        "accuracy", "balanced_accuracy", "macro_f1",
        "recall[Low_to_High]", "recall[Intermediate]",
        "LtH->NP_collapse", "NP->LtH_falsecall",
    ]
    ranked = res_df.sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(ranked[show_cols].round(3).to_string(index=False))

    # --- Recommended pick ---
    best = ranked.iloc[0]
    print("\n" + "=" * 90)
    print("TOP BY macro_f1:")
    print(f"  likelihood = {best['likelihood']}, prior = {best['prior']} ({best['prior_src']})")
    print(f"  macro_f1 = {best['macro_f1']:.3f}  balanced_acc = {best['balanced_accuracy']:.3f}  "
          f"acc = {best['accuracy']:.3f}")
    print(f"  LtH recall = {best['recall[Low_to_High]']:.3f}  "
          f"Intermediate recall = {best['recall[Intermediate]']:.3f}")
    print(f"  LtH->NP collapse = {best['LtH->NP_collapse']:.3f}  (baseline "
          f"{baseline['LtH->NP_collapse']:.3f})")


def _print_metrics(m):
    print(f"  accuracy           : {m['accuracy']:.3f}")
    print(f"  balanced_accuracy  : {m['balanced_accuracy']:.3f}")
    print(f"  macro_f1           : {m['macro_f1']:.3f}")
    for lbl in MAIN_LABELS:
        print(f"  {lbl:<15s} recall={m[f'recall[{lbl}]']:.3f}  prec={m[f'precision[{lbl}]']:.3f}")
    print(f"  LtH->NP collapse   : {m['LtH->NP_collapse']:.3f}")
    print(f"  NP->LtH falsecall  : {m['NP->LtH_falsecall']:.3f}")


if __name__ == "__main__":
    main()
