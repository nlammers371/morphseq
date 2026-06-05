"""
Benchmark conformal prediction sets across q generators.

Fixed wrapper:
    APS conformal prediction sets from new_files/conformal_sets.py

Compared axes:
    1. q source: KNN feature-neighbor vote vs multiclass logistic probabilities
    2. output: argmax(q) vs conformal set from q

Primary validation:
    leave-one-experiment-out. For each held-out experiment, the remaining embryos are
    split into a reference/training pool and a calibration pool. Both q generators use
    the same held-out query rows and calibration rows.

Usage:
    conda run -n morphseq-env --no-capture-output python \
        results/mcolon/20260601_label_transfer_method/run_q_conformal_benchmark.py \
        --max-folds 1
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / "new_files"))

from conformal_sets import (  # noqa: E402
    aps_quantile,
    aps_scores,
    build_sets,
    knn_probabilities,
    per_class_coverage,
)


DATA_PATH = ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
OUT_DIR = HERE

LABEL_COL = "cluster_categories"
EMBRYO_COL = "embryo_id"
SNIP_COL = "snip_id"
TIME_COL = "predicted_stage_hpf"
EXPERIMENT_COL = "experiment_id"
MIN_HPF = 30.0
MAX_HPF = 48.0
MAIN_LABELS = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("z_mu_b_")])


def add_hpf_bin(
    df: pd.DataFrame,
    bin_width: float,
    min_hpf: float = MIN_HPF,
    max_hpf: float = MAX_HPF,
) -> pd.DataFrame:
    out = df.copy()
    rel = np.floor((out[TIME_COL].astype(float) - min_hpf) / bin_width)
    out["_hpf_bin"] = min_hpf + rel * bin_width
    out["_hpf_bin"] = out["_hpf_bin"].clip(lower=min_hpf, upper=max_hpf - bin_width)
    return out


def embryo_label_table(df: pd.DataFrame) -> pd.DataFrame:
    """One label per embryo, using the modal image label."""
    tab = (
        df[df[LABEL_COL].notna()]
        .groupby(EMBRYO_COL)[LABEL_COL]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    return tab


def split_reference_calibration(
    df_train: pd.DataFrame,
    calibration_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split non-held-out embryos into reference/training and calibration pools."""
    rng = np.random.default_rng(seed)
    emb = embryo_label_table(df_train)
    cal_embryos = []
    for _, sub in emb.groupby(LABEL_COL):
        ids = sub[EMBRYO_COL].to_numpy()
        rng.shuffle(ids)
        n_cal = max(1, int(round(len(ids) * calibration_frac)))
        if len(ids) > 1:
            n_cal = min(n_cal, len(ids) - 1)
        cal_embryos.extend(ids[:n_cal].tolist())
    cal_set = set(cal_embryos)
    cal = df_train[df_train[EMBRYO_COL].isin(cal_set)].copy()
    ref = df_train[~df_train[EMBRYO_COL].isin(cal_set)].copy()
    return ref, cal


def labels_to_int(labels: np.ndarray, label_order: list[str]) -> np.ndarray:
    label_to_i = {label: i for i, label in enumerate(label_order)}
    return np.array([label_to_i[x] for x in labels], dtype=int)


def normalize_q(q: np.ndarray, smoothing: float = 1e-6) -> np.ndarray:
    q = np.asarray(q, dtype=float).copy()
    q[~np.isfinite(q)] = 0.0
    if smoothing > 0:
        q += smoothing
    denom = q.sum(axis=1, keepdims=True)
    bad = denom[:, 0] <= 0
    if bad.any():
        q[bad] = 1.0 / q.shape[1]
        denom = q.sum(axis=1, keepdims=True)
    return q / denom


def knn_q_for_targets(
    ref_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list[str],
    label_order: list[str],
    k: int,
    bin_width: float,
    min_ref_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate KNN q rows, using same-HPF-bin reference pools when possible."""
    q = np.zeros((len(target_df), len(label_order)), dtype=float)
    used_global = np.zeros(len(target_df), dtype=bool)
    ref_y = labels_to_int(ref_df[LABEL_COL].to_numpy(), label_order)
    ref_X = ref_df[feature_cols].to_numpy(dtype=float)

    for hpf_bin, target_idx in target_df.groupby("_hpf_bin").groups.items():
        idx = np.array(list(target_idx), dtype=int)
        ref_bin = ref_df[ref_df["_hpf_bin"] == hpf_bin]
        if len(ref_bin) >= min_ref_per_bin:
            Xr = ref_bin[feature_cols].to_numpy(dtype=float)
            yr = labels_to_int(ref_bin[LABEL_COL].to_numpy(), label_order)
            kk = min(k, len(ref_bin))
        else:
            Xr = ref_X
            yr = ref_y
            kk = min(k, len(ref_df))
            used_global[idx] = True
        Xt = target_df.iloc[idx][feature_cols].to_numpy(dtype=float)
        q[idx] = knn_probabilities(Xr, yr, Xt, k=kk, n_classes=len(label_order))
    return normalize_q(q), used_global


def multiclass_q_for_targets(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list[str],
    label_order: list[str],
    bin_width: float,
    min_train_per_bin: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate multiclass q rows from time-bin-local logistic models."""
    q = np.zeros((len(target_df), len(label_order)), dtype=float)
    used_global = np.zeros(len(target_df), dtype=bool)

    global_model = fit_multiclass_model(train_df, feature_cols, label_order, random_state)
    for hpf_bin, target_idx in target_df.groupby("_hpf_bin").groups.items():
        idx = np.array(list(target_idx), dtype=int)
        train_bin = train_df[train_df["_hpf_bin"] == hpf_bin]
        if len(train_bin) >= min_train_per_bin and train_bin[LABEL_COL].nunique() >= 2:
            model = fit_multiclass_model(train_bin, feature_cols, label_order, random_state)
        else:
            model = global_model
            used_global[idx] = True
        Xt = target_df.iloc[idx][feature_cols].to_numpy(dtype=float)
        q[idx] = predict_proba_full(model, Xt, label_order)
    return normalize_q(q), used_global


def fit_multiclass_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    label_order: list[str],
    random_state: int,
):
    model = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            )
        ),
    )
    X = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df[LABEL_COL].astype(str).to_numpy()
    model.fit(X, y)
    return model


def predict_proba_full(model, X: np.ndarray, label_order: list[str]) -> np.ndarray:
    proba = model.predict_proba(X)
    out = np.zeros((len(X), len(label_order)), dtype=float)
    classes = list(model.classes_)
    for j, label in enumerate(label_order):
        if label in classes:
            out[:, j] = proba[:, classes.index(label)]
    return out


def calibrate_from_q(
    q_cal: np.ndarray,
    y_cal: np.ndarray,
    label_order: list[str],
    alpha: float,
) -> dict:
    label_to_i = {label: i for i, label in enumerate(label_order)}
    y_idx = np.array([label_to_i[y] for y in y_cal], dtype=int)
    scores = aps_scores(normalize_q(q_cal))
    true_scores = scores[np.arange(len(y_idx)), y_idx]
    return {
        "alpha": float(alpha),
        "qhat": aps_quantile(true_scores, alpha),
        "n_calibration": int(len(y_idx)),
    }


def predict_sets_from_q(
    q_query: np.ndarray,
    qhat: float,
    label_order: list[str],
) -> pd.DataFrame:
    q = normalize_q(q_query)
    scores = aps_scores(q)
    sets = build_sets(scores, qhat)
    rows = []
    for i in range(len(q)):
        top_idx = int(q[i].argmax())
        set_idx = np.where(sets[i])[0]
        sorted_q = np.sort(q[i])
        rec = {
            "_row": i,
            "argmax_label": label_order[top_idx],
            "argmax_probability": float(q[i, top_idx]),
            "argmax_margin": float(sorted_q[-1] - sorted_q[-2]),
            "prediction_set": "|".join([label_order[j] for j in set_idx]),
            "set_size": int(len(set_idx)),
            "conformal_status": "assigned" if len(set_idx) == 1 else "ambiguous",
        }
        for j, label in enumerate(label_order):
            rec[f"q_{label}"] = float(q[i, j])
            rec[f"in_set_{label}"] = bool(sets[i, j])
        rows.append(rec)
    return pd.DataFrame(rows)


def argmax_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_order: list[str]) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)),
    }
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_order, zero_division=0
    )
    for i, label in enumerate(label_order):
        out[f"precision[{label}]"] = float(precision[i])
        out[f"recall[{label}]"] = float(recall[i])
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lth = y_true == "Low_to_High"
    np_mask = y_true == "Not Penetrant"
    out["LtH->NP_collapse"] = float((y_pred[lth] == "Not Penetrant").mean()) if lth.any() else np.nan
    out["NP->LtH_falsecall"] = float((y_pred[np_mask] == "Low_to_High").mean()) if np_mask.any() else np.nan
    return out


def conformal_metrics(y_true: np.ndarray, pred: pd.DataFrame, label_order: list[str], alpha: float) -> dict:
    label_to_i = {label: i for i, label in enumerate(label_order)}
    y_idx = np.array([label_to_i[y] for y in y_true], dtype=int)
    membership = pred[[f"in_set_{label}" for label in label_order]].to_numpy(dtype=bool)
    report = per_class_coverage(membership, y_idx, n_classes=len(label_order), alpha=alpha)
    out = {
        "target_coverage": float(1.0 - alpha),
        "marginal_coverage": report["marginal_coverage"],
        "mean_set_size": float(pred["set_size"].mean()),
        "singleton_rate": float((pred["set_size"] == 1).mean()),
        "ambiguous_rate": float((pred["set_size"] > 1).mean()),
        "empty_set_rate": float((pred["set_size"] == 0).mean()),
        "cov_gap": report["cov_gap"],
    }
    for i, label in enumerate(label_order):
        out[f"coverage[{label}]"] = report["per_class_coverage"][i]
        out[f"coverage_n[{label}]"] = report["per_class_n"][i]
    return out


def add_prediction_metadata(
    pred: pd.DataFrame,
    query_df: pd.DataFrame,
    method: str,
    fold: str,
    qhat: float,
    n_calibration: int,
    used_global: np.ndarray,
) -> pd.DataFrame:
    meta_cols = [SNIP_COL, EMBRYO_COL, EXPERIMENT_COL, TIME_COL, "_hpf_bin", LABEL_COL]
    out = pd.concat(
        [query_df[meta_cols].reset_index(drop=True), pred.reset_index(drop=True)],
        axis=1,
    )
    out["method"] = method
    out["q_source"] = method.split("_", 1)[0]
    out["heldout_experiment_id"] = fold
    out["qhat"] = float(qhat)
    out["n_calibration"] = int(n_calibration)
    out["used_global_fallback"] = used_global.astype(bool)
    out = out.rename(columns={LABEL_COL: "true_label"})
    return out


def summarize_fold(
    pred: pd.DataFrame,
    method: str,
    fold: str,
    label_order: list[str],
    alpha: float,
) -> list[dict]:
    y_true = pred["true_label"].to_numpy()
    rows = []
    rows.append({
        "method": method,
        "q_source": method.split("_", 1)[0],
        "output_type": "argmax",
        "heldout_experiment_id": fold,
        "n_query": int(len(pred)),
        **argmax_metrics(y_true, pred["argmax_label"].to_numpy(), label_order),
    })
    rows.append({
        "method": method,
        "q_source": method.split("_", 1)[0],
        "output_type": "conformal",
        "heldout_experiment_id": fold,
        "n_query": int(len(pred)),
        **conformal_metrics(y_true, pred, label_order, alpha),
    })
    return rows


def summarize_global(
    pred_df: pd.DataFrame,
    label_order: list[str],
    alpha: float,
) -> pd.DataFrame:
    """Pooled image-level summary across completed folds."""
    rows = []
    for method, sub in pred_df.groupby("method"):
        y_true = sub["true_label"].to_numpy()
        rows.append({
            "method": method,
            "q_source": method.split("_", 1)[0],
            "output_type": "argmax",
            "heldout_experiment_id": "ALL",
            "n_query": int(len(sub)),
            **argmax_metrics(y_true, sub["argmax_label"].to_numpy(), label_order),
        })
        rows.append({
            "method": method,
            "q_source": method.split("_", 1)[0],
            "output_type": "conformal",
            "heldout_experiment_id": "ALL",
            "n_query": int(len(sub)),
            **conformal_metrics(y_true, sub, label_order, alpha),
        })
    return pd.DataFrame(rows)


def run_fold(
    df: pd.DataFrame,
    fold: str,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[list[pd.DataFrame], list[dict]]:
    train_all = df[df[EXPERIMENT_COL] != fold].copy()
    query = df[df[EXPERIMENT_COL] == fold].copy().reset_index(drop=True)
    ref, cal = split_reference_calibration(train_all, args.calibration_frac, args.seed)
    ref = ref.reset_index(drop=True)
    cal = cal.reset_index(drop=True)
    print(
        f"  ref={len(ref):,} rows/{ref[EMBRYO_COL].nunique()} embryos; "
        f"cal={len(cal):,} rows/{cal[EMBRYO_COL].nunique()} embryos; "
        f"query={len(query):,} rows/{query[EMBRYO_COL].nunique()} embryos"
    )

    outputs = []
    summaries = []
    q_jobs = []

    q_cal_knn, _ = knn_q_for_targets(
        ref, cal, feature_cols, MAIN_LABELS, args.k, args.bin_width, args.min_ref_per_bin
    )
    q_query_knn, fallback_query_knn = knn_q_for_targets(
        ref, query, feature_cols, MAIN_LABELS, args.k, args.bin_width, args.min_ref_per_bin
    )
    q_jobs.append(("knn_q", q_cal_knn, q_query_knn, fallback_query_knn))

    q_cal_mc, _ = multiclass_q_for_targets(
        ref, cal, feature_cols, MAIN_LABELS, args.bin_width, args.min_train_per_bin, args.seed
    )
    q_query_mc, fallback_query_mc = multiclass_q_for_targets(
        ref, query, feature_cols, MAIN_LABELS, args.bin_width, args.min_train_per_bin, args.seed
    )
    q_jobs.append(("multiclass_q", q_cal_mc, q_query_mc, fallback_query_mc))

    y_cal = cal[LABEL_COL].astype(str).to_numpy()
    for method, q_cal, q_query, fallback_query in q_jobs:
        calibration = calibrate_from_q(q_cal, y_cal, MAIN_LABELS, args.alpha)
        pred = predict_sets_from_q(q_query, calibration["qhat"], MAIN_LABELS)
        pred = add_prediction_metadata(
            pred, query, method, fold, calibration["qhat"],
            calibration["n_calibration"], fallback_query,
        )
        outputs.append(pred)
        summaries.extend(summarize_fold(pred, method, fold, MAIN_LABELS, args.alpha))
        print(
            f"  {method}: qhat={calibration['qhat']:.3f}, "
            f"argmax_acc={(pred['argmax_label'] == pred['true_label']).mean():.3f}, "
            f"coverage={conformal_metrics(pred['true_label'].to_numpy(), pred, MAIN_LABELS, args.alpha)['marginal_coverage']:.3f}, "
            f"mean_set_size={pred['set_size'].mean():.2f}"
        )
    return outputs, summaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--bin-width", type=float, default=4.0)
    p.add_argument("--calibration-frac", type=float, default=0.25)
    p.add_argument("--min-ref-per-bin", type=int, default=80)
    p.add_argument("--min-train-per-bin", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--min-hpf", type=float, default=MIN_HPF)
    p.add_argument("--max-hpf", type=float, default=MAX_HPF)
    p.add_argument("--output-prefix", default="q_conformal_benchmark")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df[LABEL_COL].isin(MAIN_LABELS)].copy()
    df = df[(df[TIME_COL] >= args.min_hpf) & (df[TIME_COL] <= args.max_hpf)].copy()
    feature_cols = get_feature_cols(df)
    keep_cols = [SNIP_COL, EMBRYO_COL, EXPERIMENT_COL, TIME_COL, LABEL_COL, *feature_cols]
    df = df[keep_cols].dropna(subset=[TIME_COL, LABEL_COL, *feature_cols]).reset_index(drop=True)
    df = add_hpf_bin(df, args.bin_width, min_hpf=args.min_hpf, max_hpf=args.max_hpf)
    print(
        f"  {len(df):,} labeled images; {df[EMBRYO_COL].nunique()} embryos; "
        f"{df[EXPERIMENT_COL].nunique()} experiments; {len(feature_cols)} features; "
        f"hpf {args.min_hpf:g}-{args.max_hpf:g}"
    )

    experiments = sorted(df[EXPERIMENT_COL].dropna().unique().astype(str))
    if args.max_folds is not None:
        experiments = experiments[:args.max_folds]
    all_predictions = []
    all_summaries = []
    for fold in experiments:
        print("\n" + "=" * 80)
        print(f"Hold out experiment {fold}")
        try:
            preds, summaries = run_fold(df, fold, feature_cols, args)
        except Exception as exc:
            warnings.warn(f"Fold {fold} failed: {exc}")
            continue
        all_predictions.extend(preds)
        all_summaries.extend(summaries)

    if not all_predictions:
        raise RuntimeError("No folds completed.")

    pred_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    global_df = summarize_global(pred_df, MAIN_LABELS, args.alpha)
    pred_path = OUT_DIR / f"{args.output_prefix}_image_predictions.csv"
    summary_path = OUT_DIR / f"{args.output_prefix}_summary.csv"
    global_path = OUT_DIR / f"{args.output_prefix}_global_summary.csv"
    pred_df.to_csv(pred_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    global_df.to_csv(global_path, index=False)
    print("\nSaved:")
    print(f"  {pred_path}")
    print(f"  {summary_path}")
    print(f"  {global_path}")

    print("\nPooled global summary by method/output:")
    show_cols = [
        "n_query", "accuracy", "balanced_accuracy", "macro_f1",
        "marginal_coverage", "mean_set_size", "singleton_rate",
        "LtH->NP_collapse", "NP->LtH_falsecall",
    ]
    numeric = global_df.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [c for c in show_cols if c in numeric]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(global_df[["method", "output_type", *agg_cols]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
