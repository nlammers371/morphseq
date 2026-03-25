"""Classification factory-line functions.

Each function handles exactly one step.  The orchestrator
(``run_classification``) calls them in sequence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import analyze.utils.resampling as resample

from .comparison_resolution import ComparisonGroup, ResolvedComparison

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_logistic_classifier(n_classes: int, random_state: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        multi_class="ovr",
        class_weight="balanced",
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Step 1: Feature resolution
# ---------------------------------------------------------------------------


def _resolve_feature_columns(
    df: pd.DataFrame,
    features: dict[str, str | list[str]],
) -> dict[str, list[str]]:
    """Resolve user-facing feature spec into concrete column lists.

    - ``str`` values: prefix-match against ``df.columns``
    - ``list[str]`` values: passed through; missing columns error
    """
    resolved: dict[str, list[str]] = {}
    for name, spec in features.items():
        if isinstance(spec, str):
            cols = [c for c in df.columns if c.startswith(spec)]
            if not cols:
                raise ValueError(
                    f"Feature set {name!r}: no columns match prefix {spec!r}"
                )
            resolved[name] = sorted(cols)
        elif isinstance(spec, list):
            missing = [c for c in spec if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Feature set {name!r}: missing columns {missing}"
                )
            resolved[name] = list(spec)
        else:
            raise TypeError(
                f"Feature set {name!r}: expected str or list[str], "
                f"got {type(spec).__name__}"
            )
    return resolved


# ---------------------------------------------------------------------------
# Step 3: Binary labelling (THE pooling-aware function)
# ---------------------------------------------------------------------------


def _build_binary_labels(
    df: pd.DataFrame,
    class_col: str,
    comparison: ResolvedComparison,
) -> pd.DataFrame:
    """Filter *df* to comparison members and assign ``_y`` column.

    ``_y = 1`` for positive members, ``_y = 0`` for negative members.
    Rows from unrelated classes are dropped.
    """
    pos_set = set(comparison.positive_members)
    neg_set = set(comparison.negative_members)
    all_members = pos_set | neg_set

    mask = df[class_col].isin(all_members)
    out = df.loc[mask].copy()

    out["_y"] = out[class_col].map(
        lambda x: 1 if x in pos_set else 0
    )
    return out


# ---------------------------------------------------------------------------
# Step 4: Binning & aggregation
# ---------------------------------------------------------------------------


def _bin_and_aggregate(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature_cols: list[str],
    bin_width: float,
) -> pd.DataFrame:
    """Floor-bin by *time_col*, then mean-aggregate per (id, bin, label)."""
    out = df.copy()
    out["_time_bin"] = (np.floor(out[time_col] / bin_width) * bin_width).astype(int)
    out["time_bin_center"] = out["_time_bin"].astype(float) + bin_width / 2.0

    groupby_cols = [id_col, "_time_bin", "time_bin_center", "_y"]
    agg_cols = [c for c in feature_cols if c in out.columns]
    result = out.groupby(groupby_cols, as_index=False)[agg_cols].mean()
    return result


# ---------------------------------------------------------------------------
# Step 5 inner: Permutation test
# ---------------------------------------------------------------------------


def _permutation_test_binary(
    *,
    X: np.ndarray,
    y_binary: np.ndarray,
    n_permutations: int,
    n_splits: int,
    n_jobs: int,
    random_state: int,
    bin_index: int,
    time_bin: int,
) -> np.ndarray:
    """Permutation test for binary AUROC using the shared resampling engine."""
    if n_permutations <= 0:
        return np.array([], dtype=float)

    clf = _make_logistic_classifier(n_classes=2, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _stat_fn(data: dict, _rng: np.random.Generator) -> float:
        y_perm = np.asarray(data["labels"], dtype=int)
        if len(np.unique(y_perm)) < 2:
            return float("nan")
        probs_perm = cross_val_predict(clf, X, y_perm, cv=cv, method="predict_proba")
        # Column 1 = probability of positive class
        return float(roc_auc_score(y_perm, probs_perm[:, 1]))

    spec = resample.permute_labels(within=None)
    stat = resample.statistic(
        name="binary_auroc",
        fn=_stat_fn,
        default_alternative="greater",
        is_nonnegative=True,
    )

    data = {"labels": np.asarray(y_binary, dtype=int)}
    seed = int(random_state) + 1000003 * (bin_index + 1) + 10007 * int(time_bin)

    out = resample.run(
        data=data,
        spec=spec,
        statistic=stat,
        n_iters=n_permutations,
        seed=seed,
        n_jobs=n_jobs,
        store="all",
        alternative="greater",
    )

    null_aurocs = np.asarray(out.samples or [], dtype=float)
    return null_aurocs[np.isfinite(null_aurocs)]


def _permutation_test_ovr(
    *,
    X: np.ndarray,
    y: np.ndarray,
    class_idx: int,
    n_classes: int,
    n_permutations: int,
    n_splits: int,
    n_jobs: int,
    random_state: int,
    bin_index: int,
    time_bin: int,
) -> np.ndarray:
    """Permutation test for OvR AUROC (multiclass fast path)."""
    if n_permutations <= 0:
        return np.array([], dtype=float)

    clf = _make_logistic_classifier(n_classes=n_classes, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _stat_fn(data: dict, _rng: np.random.Generator) -> float:
        y_perm = np.asarray(data["labels"], dtype=int)
        probs_perm = cross_val_predict(clf, X, y_perm, cv=cv, method="predict_proba")
        present_classes = np.array(sorted(np.unique(y_perm)), dtype=int)
        class_index_to_col = {ci: col for col, ci in enumerate(present_classes)}
        target_col = class_index_to_col.get(class_idx)
        if target_col is None:
            return float("nan")
        y_binary_perm = (y_perm == class_idx).astype(int)
        if len(np.unique(y_binary_perm)) < 2:
            return float("nan")
        return float(roc_auc_score(y_binary_perm, probs_perm[:, target_col]))

    spec = resample.permute_labels(within=None)
    stat = resample.statistic(
        name="ovr_auroc",
        fn=_stat_fn,
        default_alternative="greater",
        is_nonnegative=True,
    )

    data = {"labels": np.asarray(y, dtype=int)}
    seed = int(random_state) + 1000003 * (bin_index + 1) + 10007 * int(time_bin) + 31 * class_idx

    out = resample.run(
        data=data,
        spec=spec,
        statistic=stat,
        n_iters=n_permutations,
        seed=seed,
        n_jobs=n_jobs,
        store="all",
        alternative="greater",
    )

    null_aurocs = np.asarray(out.samples or [], dtype=float)
    return null_aurocs[np.isfinite(null_aurocs)]


# ---------------------------------------------------------------------------
# Step 5: Classification loop (binary path)
# ---------------------------------------------------------------------------


def _run_binary_classification_loop(
    *,
    df_binned: pd.DataFrame,
    feature_cols: list[str],
    id_col: str,
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    n_jobs: int,
    random_state: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Per-bin CV + AUROC + permutation test for a single binary comparison."""
    time_bins = sorted(df_binned["_time_bin"].unique())
    results: list[dict[str, Any]] = []

    for i, t in enumerate(time_bins):
        sub = df_binned[df_binned["_time_bin"] == t]
        X = sub[feature_cols].to_numpy()
        y = sub["_y"].to_numpy().astype(int)

        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        if n_pos == 0 or n_neg == 0:
            if verbose:
                print(f"    Bin {t}: skipped (n_pos={n_pos}, n_neg={n_neg})")
            continue

        min_count = min(n_pos, n_neg)
        n_splits_actual = min(n_splits, min_count)
        if n_splits_actual < 2:
            if verbose:
                print(f"    Bin {t}: skipped (min_count={min_count} < 2)")
            continue

        clf = _make_logistic_classifier(n_classes=2, random_state=random_state)
        cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=random_state)

        try:
            probs = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
            auroc_obs = float(roc_auc_score(y, probs[:, 1]))
        except Exception as exc:
            if verbose:
                print(f"    Bin {t}: error — {exc}")
            continue

        null_aurocs = _permutation_test_binary(
            X=X, y_binary=y,
            n_permutations=n_permutations,
            n_splits=n_splits_actual,
            n_jobs=n_jobs, random_state=random_state,
            bin_index=i, time_bin=int(t),
        )

        if len(null_aurocs) == 0:
            pval = float("nan")
            null_mean = float("nan")
            null_std = float("nan")
        else:
            exceed_count = int(np.sum(null_aurocs >= auroc_obs))
            pval = float((exceed_count + 1) / (len(null_aurocs) + 1))
            null_mean = float(np.mean(null_aurocs))
            null_std = float(np.std(null_aurocs))

        # Confusion for this bin
        y_pred = (probs[:, 1] >= 0.5).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])

        # Per-embryo predictions
        embryo_ids = sub[id_col].to_numpy()
        predictions = []
        for row_idx in range(len(y)):
            predictions.append({
                id_col: embryo_ids[row_idx],
                "time_bin_center": float(t) + bin_width / 2.0,
                "y_true": int(y[row_idx]),
                "p_pos": float(probs[row_idx, 1]),
                "y_pred": int(y_pred[row_idx]),
                "is_correct": bool(y[row_idx] == y_pred[row_idx]),
            })

        entry: dict[str, Any] = {
            "time_bin": int(t),
            "time_bin_center": float(t) + bin_width / 2.0,
            "bin_width": float(bin_width),
            "auroc_obs": auroc_obs,
            "pval": pval,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "auroc_null_mean": null_mean,
            "auroc_null_std": null_std,
            "n_permutations": len(null_aurocs),
            "_null_array": null_aurocs,
            "_confusion_matrix": cm,
            "_predictions": predictions,
        }
        results.append(entry)

        if verbose:
            sig = "*" if pval < 0.05 else ""
            print(f"    Bin {t}: AUROC={auroc_obs:.3f}, p={pval:.3f}{sig}")

    return results


# ---------------------------------------------------------------------------
# Step 5 (multiclass fast path): Classification loop for all-vs-rest
# ---------------------------------------------------------------------------


def _run_multiclass_classification_loop(
    *,
    df_binned: pd.DataFrame,
    class_labels: list[str],
    feature_cols: list[str],
    id_col: str,
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    n_jobs: int,
    min_samples_per_class: int,
    random_state: int,
    verbose: bool,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Run one multiclass model per bin, extract per-class binary AUROCs.

    Returns
    -------
    ovr_results
        ``{class_label: [per_bin_dict, ...]}`` — one entry per class per bin.
    all_embryo_predictions
        Flat list of per-embryo prediction dicts (wide multiclass format).
    """
    label_to_int = {label: i for i, label in enumerate(class_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    time_bins = sorted(df_binned["_time_bin"].unique())
    ovr_results: dict[str, list[dict[str, Any]]] = {label: [] for label in class_labels}
    all_embryo_predictions: list[dict[str, Any]] = []

    for i, t in enumerate(time_bins):
        if verbose:
            print(f"\n  [{i + 1}/{len(time_bins)}] Time bin {t} hpf...")

        sub = df_binned[df_binned["_time_bin"] == t]
        X = sub[feature_cols].to_numpy()
        y_labels = sub["_class_label"].astype(str).to_numpy()
        embryo_ids = sub[id_col].astype(str).to_numpy()
        y = np.array([label_to_int[label] for label in y_labels], dtype=int)

        class_counts = {label: int(np.sum(y_labels == label)) for label in class_labels}
        present_classes = [label for label, count in class_counts.items() if count > 0]

        if len(present_classes) < 2:
            if verbose:
                print("    Skipped (need at least 2 classes present)")
            continue

        min_count = min(class_counts[label] for label in present_classes)
        if min_count < min_samples_per_class:
            if verbose:
                print(f"    Skipped (min class has {min_count} samples, need {min_samples_per_class})")
            continue

        clf = _make_logistic_classifier(n_classes=len(class_labels), random_state=random_state)
        n_splits_actual = min(n_splits, min_count)
        cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=random_state)

        try:
            probs_present = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
            present_class_indices = np.array(sorted(np.unique(y)), dtype=int)
            class_index_to_col = {
                cls_idx: col_idx for col_idx, cls_idx in enumerate(present_class_indices)
            }
        except Exception as exc:
            if verbose:
                print(f"    Error: {exc}")
            continue

        # Expand to full probability matrix
        probs_full = np.zeros((len(y), len(class_labels)), dtype=float)
        for class_idx, col_idx in class_index_to_col.items():
            probs_full[:, class_idx] = probs_present[:, col_idx]

        pred_classes = np.argmax(probs_full, axis=1)

        # Per-class OvR AUROC
        for class_label in class_labels:
            class_idx = label_to_int[class_label]
            y_binary = (y == class_idx).astype(int)
            n_positive = int(np.sum(y_binary == 1))
            n_negative = int(np.sum(y_binary == 0))
            if n_positive == 0 or n_negative == 0:
                continue

            class_probs = probs_full[:, class_idx]
            try:
                true_auroc = float(roc_auc_score(y_binary, class_probs))
            except Exception:
                continue

            null_aurocs = _permutation_test_ovr(
                X=X, y=y, class_idx=class_idx, n_classes=len(class_labels),
                n_permutations=n_permutations, n_splits=n_splits_actual,
                n_jobs=n_jobs, random_state=random_state,
                bin_index=i, time_bin=int(t),
            )
            if len(null_aurocs) == 0:
                continue

            exceed_count = int(np.sum(null_aurocs >= true_auroc))
            pval = float((exceed_count + 1) / (len(null_aurocs) + 1))

            ovr_results[class_label].append({
                "time_bin": int(t),
                "time_bin_center": float(t) + float(bin_width) / 2.0,
                "bin_width": float(bin_width),
                "auroc_obs": true_auroc,
                "pval": pval,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "auroc_null_mean": float(np.mean(null_aurocs)),
                "auroc_null_std": float(np.std(null_aurocs)),
                "n_permutations": int(len(null_aurocs)),
                "_null_array": null_aurocs,
            })

            if verbose:
                sig_marker = "*" if pval < 0.05 else ""
                print(f"    {class_label} vs Rest: AUROC={true_auroc:.3f}, p={pval:.3f}{sig_marker}")

        # Per-embryo predictions (wide multiclass format)
        for row_idx, (eid, true_label, pred_idx) in enumerate(zip(embryo_ids, y_labels, pred_classes)):
            pred_label = int_to_label[int(pred_idx)]
            pred_record: dict[str, Any] = {
                "embryo_id": str(eid),
                "time_bin": int(t),
                "time_bin_center": float(t) + float(bin_width) / 2.0,
                "true_class": str(true_label),
                "pred_class": pred_label,
                "is_correct": bool(true_label == pred_label),
            }
            for class_label in class_labels:
                p = float(probs_full[row_idx, label_to_int[class_label]])
                pred_record[f"pred_proba_{class_label}"] = p

            raw_p_pred = pred_record[f"pred_proba_{pred_label}"]
            raw_p_true = pred_record[f"pred_proba_{true_label}"]
            pred_record["p_pred"] = float(np.clip(raw_p_pred, 0.0, 1.0))
            pred_record["p_true"] = float(np.clip(raw_p_true, 0.0, 1.0))
            pred_record["is_wrong"] = int(not pred_record["is_correct"])

            all_embryo_predictions.append(pred_record)

    return ovr_results, all_embryo_predictions


# ---------------------------------------------------------------------------
# Step 6: Collect scores (THE identity+result joiner)
# ---------------------------------------------------------------------------


def _collect_scores(
    bin_results: list[dict[str, Any]],
    comparison: ResolvedComparison,
    feature_set: str,
) -> list[dict[str, Any]]:
    """Assemble canonical scores rows from raw per-bin dicts."""
    rows = []
    for br in bin_results:
        rows.append({
            "feature_set": feature_set,
            "comparison_id": comparison.comparison_id,
            "positive_label": comparison.positive_label,
            "negative_label": comparison.negative_label,
            "time_bin_center": br["time_bin_center"],
            "time_bin": br["time_bin"],
            "bin_width": br["bin_width"],
            "auroc_obs": br["auroc_obs"],
            "pval": br["pval"],
            "n_positive": br["n_positive"],
            "n_negative": br["n_negative"],
            "auroc_null_mean": br["auroc_null_mean"],
            "auroc_null_std": br["auroc_null_std"],
            "n_permutations": br["n_permutations"],
        })
    return rows


def _collect_scores_from_ovr(
    ovr_results: dict[str, list[dict[str, Any]]],
    class_labels: list[str],
    resolved_map: dict[str, ResolvedComparison],
    feature_set: str,
) -> list[dict[str, Any]]:
    """Collect scores from multiclass OvR results into canonical format."""
    rows = []
    for class_label in class_labels:
        rc = resolved_map.get(class_label)
        if rc is None:
            continue
        for br in ovr_results.get(class_label, []):
            rows.append({
                "feature_set": feature_set,
                "comparison_id": rc.comparison_id,
                "positive_label": rc.positive_label,
                "negative_label": rc.negative_label,
                "time_bin_center": br["time_bin_center"],
                "time_bin": br["time_bin"],
                "bin_width": br["bin_width"],
                "auroc_obs": br["auroc_obs"],
                "pval": br["pval"],
                "n_positive": br["n_positive"],
                "n_negative": br["n_negative"],
                "auroc_null_mean": br["auroc_null_mean"],
                "auroc_null_std": br["auroc_null_std"],
                "n_permutations": br["n_permutations"],
            })
    return rows


# ---------------------------------------------------------------------------
# Step 7a: Collect binary predictions
# ---------------------------------------------------------------------------


def _collect_binary_predictions(
    bin_results: list[dict[str, Any]],
    comparison: ResolvedComparison,
    feature_set: str,
    id_col: str,
) -> list[dict[str, Any]]:
    """Tidy format: one row per (unit, time_bin, comparison, feature_set)."""
    rows = []
    for br in bin_results:
        for pred in br.get("_predictions", []):
            rows.append({
                "feature_set": feature_set,
                "comparison_id": comparison.comparison_id,
                id_col: pred[id_col],
                "time_bin_center": pred["time_bin_center"],
                "y_true": pred["y_true"],
                "p_pos": pred["p_pos"],
                "y_pred": pred["y_pred"],
                "is_correct": pred["is_correct"],
            })
    return rows


# ---------------------------------------------------------------------------
# Step 7b: Collect multiclass predictions
# ---------------------------------------------------------------------------


def _collect_multiclass_predictions(
    all_embryo_predictions: list[dict[str, Any]],
) -> pd.DataFrame | None:
    """Wide multiclass format for the misclassification pipeline."""
    if not all_embryo_predictions:
        return None
    return pd.DataFrame(all_embryo_predictions)


# ---------------------------------------------------------------------------
# Step 8: Collect confusion
# ---------------------------------------------------------------------------


def _collect_confusion(
    bin_results: list[dict[str, Any]],
    comparison: ResolvedComparison,
    feature_set: str,
) -> list[dict[str, Any]]:
    """Per-bin confusion rows using side labels."""
    side_labels = [comparison.negative_label, comparison.positive_label]  # 0, 1
    rows = []
    for br in bin_results:
        cm = br.get("_confusion_matrix")
        if cm is None:
            continue
        total_per_true = cm.sum(axis=1)
        for true_idx, true_label in enumerate(side_labels):
            row_total = total_per_true[true_idx]
            if row_total == 0:
                continue
            for pred_idx, pred_label in enumerate(side_labels):
                count = int(cm[true_idx, pred_idx])
                rows.append({
                    "feature_set": feature_set,
                    "comparison_id": comparison.comparison_id,
                    "time_bin_center": br["time_bin_center"],
                    "true_class": true_label,
                    "predicted_class": pred_label,
                    "proportion": float(count) / float(row_total),
                    "count": count,
                    "is_correct": bool(true_label == pred_label),
                })
    return rows


def _collect_confusion_from_ovr(
    ovr_results: dict[str, list[dict[str, Any]]],
    all_embryo_predictions: list[dict[str, Any]],
    class_labels: list[str],
    resolved_map: dict[str, ResolvedComparison],
    feature_set: str,
    bin_width: float,
) -> list[dict[str, Any]]:
    """Collect confusion from multiclass OvR predictions."""
    if not all_embryo_predictions:
        return []

    pred_df = pd.DataFrame(all_embryo_predictions)
    rows = []
    for t, group in pred_df.groupby("time_bin"):
        true_labels = group["true_class"].to_numpy()
        pred_labels = group["pred_class"].to_numpy()
        tbc = float(t) + bin_width / 2.0

        for true_class in class_labels:
            true_mask = true_labels == true_class
            row_total = int(true_mask.sum())
            if row_total == 0:
                continue
            for pred_class in class_labels:
                count = int(np.sum((true_labels == true_class) & (pred_labels == pred_class)))
                rows.append({
                    "feature_set": feature_set,
                    "comparison_id": "multiclass",
                    "time_bin_center": tbc,
                    "true_class": true_class,
                    "predicted_class": pred_class,
                    "proportion": float(count) / float(row_total),
                    "count": count,
                    "is_correct": bool(true_class == pred_class),
                })
    return rows
