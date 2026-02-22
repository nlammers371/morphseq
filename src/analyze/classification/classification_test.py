"""Classification testing utilities.

This module is the unified home for multiclass/one-vs-rest and flexible
reference-based classification comparisons.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import analyze.utils.resampling as resample

from .results import ComparisonSpec, MulticlassOVRResults


def _make_logistic_classifier(n_classes: int, random_state: int) -> LogisticRegression:
    """Create a stable classifier across binary/multiclass settings."""
    return LogisticRegression(
        max_iter=1000,
        # liblinear avoids OpenMP SHM aborts in restricted environments.
        solver="liblinear",
        multi_class="ovr",
        class_weight="balanced",
        random_state=random_state,
    )


def _resolve_feature_columns(df: pd.DataFrame, features: Union[str, List[str]]) -> List[str]:
    if isinstance(features, str):
        if features == "z_mu_b":
            cols = [c for c in df.columns if "z_mu_b" in c]
            if not cols:
                raise ValueError("No z_mu_b columns found. Specify features explicitly.")
            return cols
        if features not in df.columns:
            raise ValueError(f"Feature '{features}' not found in dataframe")
        return [features]

    cols = list(features)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return cols


def _resolve_comparison_groups(
    df: pd.DataFrame,
    groupby: str,
    groups: Union[str, List[str]],
    reference: Union[str, List[Union[str, Tuple[str, ...]]]],
    embryo_id_col: str = "embryo_id",
    allow_many_comparisons: bool = False,
    max_comparisons: int = 50,
) -> List[ComparisonSpec]:
    """Parse groups/reference into explicit comparison specs."""
    if groupby not in df.columns:
        raise ValueError(f"groupby column '{groupby}' not found in DataFrame")

    all_groups_in_data = df[groupby].dropna().unique().tolist()
    if not all_groups_in_data:
        raise ValueError(f"No non-null values found in groupby column '{groupby}'")

    if groups == "all":
        positive_groups = all_groups_in_data
    elif isinstance(groups, str):
        if groups not in all_groups_in_data:
            raise ValueError(f"Group '{groups}' not found in data: {all_groups_in_data}")
        positive_groups = [groups]
    elif isinstance(groups, list):
        missing = [g for g in groups if g not in all_groups_in_data]
        if missing:
            raise ValueError(f"Groups not found in data: {missing}")
        positive_groups = groups
    else:
        raise ValueError("groups must be 'all', string, or list[str]")

    if reference == "rest":
        reference_items: List[Union[str, Tuple[str, ...]]] = ["rest"]
    elif isinstance(reference, str):
        if reference not in all_groups_in_data:
            raise ValueError(f"Reference '{reference}' not found in data")
        reference_items = [reference]
    elif isinstance(reference, tuple):
        missing = [g for g in reference if g not in all_groups_in_data]
        if missing:
            raise ValueError(f"Reference groups not found in data: {missing}")
        reference_items = [reference]
    elif isinstance(reference, list):
        reference_items = []
        for item in reference:
            if isinstance(item, str):
                if item != "rest" and item not in all_groups_in_data:
                    raise ValueError(f"Reference '{item}' not found in data")
                reference_items.append(item)
            elif isinstance(item, tuple):
                missing = [g for g in item if g not in all_groups_in_data]
                if missing:
                    raise ValueError(f"Reference groups not found in data: {missing}")
                reference_items.append(item)
            else:
                raise ValueError("reference list elements must be str or tuple[str,...]")
    else:
        raise ValueError("reference must be 'rest', string, tuple, or list")

    n_comparisons = len(positive_groups) * len(reference_items)
    if n_comparisons > max_comparisons and not allow_many_comparisons:
        raise ValueError(
            f"Too many comparisons requested: {n_comparisons} > max_comparisons={max_comparisons}"
        )

    group_to_embryos: Dict[str, List[str]] = {}
    for group in all_groups_in_data:
        group_mask = df[groupby] == group
        group_to_embryos[group] = df.loc[group_mask, embryo_id_col].dropna().astype(str).unique().tolist()

    specs: List[ComparisonSpec] = []
    for pos_group in positive_groups:
        pos_members = group_to_embryos[pos_group]

        for ref_item in reference_items:
            if ref_item == "rest":
                neg_members = []
                for g in all_groups_in_data:
                    if g != pos_group:
                        neg_members.extend(group_to_embryos[g])
                neg_label = "rest"
                neg_mode = "rest"
                neg_member_list = sorted(set(neg_members))
            elif isinstance(ref_item, str):
                neg_label = ref_item
                neg_member_list = group_to_embryos[ref_item]
                neg_mode = "single"
            elif isinstance(ref_item, tuple):
                neg_members = []
                for g in ref_item:
                    neg_members.extend(group_to_embryos[g])
                neg_label = "+".join(ref_item)
                neg_member_list = sorted(set(neg_members))
                neg_mode = "pooled"
            else:
                raise ValueError(f"Unexpected reference item type: {type(ref_item)}")

            specs.append(
                ComparisonSpec(
                    positive=pos_group,
                    negative=neg_label,
                    positive_members=pos_members,
                    negative_members=neg_member_list,
                    negative_mode=neg_mode,
                )
            )

    return specs


def _permutation_test_ovr(
    *,
    X: np.ndarray,
    y: np.ndarray,
    class_idx: int,
    n_classes: int,
    time_strata: Optional[np.ndarray],
    n_permutations: int,
    n_splits: int,
    n_jobs: int,
    random_state: int,
    bin_index: int,
    time_bin: int,
) -> np.ndarray:
    """Permutation test for OvR AUROC using the shared resampling engine."""
    if n_permutations <= 0:
        return np.array([], dtype=float)

    clf = _make_logistic_classifier(n_classes=n_classes, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _stat_fn(data: dict, _rng: np.random.Generator) -> float:
        y_perm = np.asarray(data["labels"], dtype=int)
        probs_perm = cross_val_predict(clf, X, y_perm, cv=cv, method="predict_proba")
        present_classes = np.array(sorted(np.unique(y_perm)), dtype=int)
        class_index_to_col = {cls_idx: col_idx for col_idx, cls_idx in enumerate(present_classes)}
        target_col = class_index_to_col.get(class_idx)
        if target_col is None:
            return float("nan")
        y_binary_perm = (y_perm == class_idx).astype(int)
        if len(np.unique(y_binary_perm)) < 2:
            return float("nan")
        return float(roc_auc_score(y_binary_perm, probs_perm[:, target_col]))

    spec = resample.permute_labels(within="time_strata" if time_strata is not None else None)
    stat = resample.statistic(
        name="ovr_auroc",
        fn=_stat_fn,
        default_alternative="greater",
        is_nonnegative=True,
    )

    data = {"labels": np.asarray(y, dtype=int)}
    if time_strata is not None:
        data["time_strata"] = np.asarray(time_strata)

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


def _run_multiclass_classification(
    *,
    df_binned: pd.DataFrame,
    df_raw: pd.DataFrame,
    class_labels: List[str],
    feature_cols: List[str],
    time_col: str,
    embryo_id_col: str,
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    n_jobs: int,
    min_samples_per_class: int,
    within_bin_time_stratification: bool,
    within_bin_time_strata_width: float,
    skip_bin_if_not_all_present: bool,
    random_state: int,
    verbose: bool,
) -> tuple[Dict[str, pd.DataFrame], Dict[int, pd.DataFrame], Optional[pd.DataFrame], Dict[str, int]]:
    """Run multiclass OvR AUROC classification across time bins."""
    time_bins = sorted(df_binned["_time_bin"].unique())
    total_bins = len(time_bins)
    n_complete_bins = 0
    n_incomplete_bins = 0
    n_skipped_bins = 0

    ovr_results = {label: [] for label in class_labels}
    confusion_matrices: Dict[int, pd.DataFrame] = {}
    all_embryo_predictions: List[dict[str, Any]] = []

    label_to_int = {label: i for i, label in enumerate(class_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    for i, t in enumerate(time_bins):
        if verbose:
            print(f"\n  [{i + 1}/{len(time_bins)}] Time bin {t} hpf...")

        sub = df_binned[df_binned["_time_bin"] == t]
        X = sub[feature_cols].to_numpy()
        y_labels = sub["_class_label"].astype(str).to_numpy()
        embryo_ids = sub[embryo_id_col].astype(str).to_numpy()

        y = np.array([label_to_int[label] for label in y_labels], dtype=int)

        class_counts = {label: int(np.sum(y_labels == label)) for label in class_labels}
        present_classes = [label for label, count in class_counts.items() if count > 0]
        missing_classes = [label for label, count in class_counts.items() if count == 0]

        if verbose:
            print(f"    Class counts: {class_counts}")
            print("    Using balanced class weights for this bin")

        if missing_classes:
            n_incomplete_bins += 1
        else:
            n_complete_bins += 1

        if len(present_classes) < 2:
            n_skipped_bins += 1
            if verbose:
                print("    Skipped (need at least 2 classes present)")
            continue

        if skip_bin_if_not_all_present and missing_classes:
            n_skipped_bins += 1
            if verbose:
                print(f"    Skipped (missing classes: {missing_classes})")
            continue

        min_count = min(class_counts[label] for label in present_classes)
        if min_count < min_samples_per_class:
            n_skipped_bins += 1
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
            n_skipped_bins += 1
            if verbose:
                print(f"    Error: {exc}")
            continue

        probs_full = np.zeros((len(y), len(class_labels)), dtype=float)
        for class_idx, col_idx in class_index_to_col.items():
            probs_full[:, class_idx] = probs_present[:, col_idx]

        pred_classes = np.argmax(probs_full, axis=1)

        time_strata = None
        if within_bin_time_stratification:
            sub_raw = df_raw[df_raw["_time_bin"] == t]
            embryo_mean_times = (
                sub_raw.groupby(embryo_id_col)[time_col].mean().reindex(embryo_ids).to_numpy()
            )
            time_strata = np.floor(
                (embryo_mean_times - float(t)) / float(within_bin_time_strata_width)
            ).astype(int)

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
                X=X,
                y=y,
                class_idx=class_idx,
                n_classes=len(class_labels),
                time_strata=time_strata,
                n_permutations=n_permutations,
                n_splits=n_splits_actual,
                n_jobs=n_jobs,
                random_state=random_state,
                bin_index=i,
                time_bin=int(t),
            )
            if len(null_aurocs) == 0:
                continue

            exceed_count = int(np.sum(null_aurocs >= true_auroc))
            pval = float((exceed_count + 1) / (len(null_aurocs) + 1))

            ovr_results[class_label].append(
                {
                    "time_bin": int(t),
                    "time_bin_start": float(t),
                    "time_bin_end": float(t) + float(bin_width),
                    "time_bin_center": float(t) + float(bin_width) / 2.0,
                    "bin_width": float(bin_width),
                    "auroc_observed": true_auroc,
                    "auroc_null_mean": float(np.mean(null_aurocs)),
                    "auroc_null_std": float(np.std(null_aurocs)),
                    "auroc_null_exceed_count": exceed_count,
                    "auroc_n_permutations": int(len(null_aurocs)),
                    "pval": pval,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "positive_class": class_label,
                    "negative_class": "Rest",
                }
            )

            if verbose:
                sig_marker = "*" if pval < 0.05 else ""
                print(f"    {class_label} vs Rest: AUROC={true_auroc:.3f}, p={pval:.3f}{sig_marker}")

        cm = confusion_matrix(y, pred_classes, labels=list(range(len(class_labels))))
        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        cm_df.index.name = "true_class"
        cm_df.columns.name = "predicted_class"
        if missing_classes:
            cm_df.loc[missing_classes, :] = np.nan
            cm_df.loc[:, missing_classes] = np.nan
        confusion_matrices[int(t)] = cm_df

        for row_idx, (eid, true_label, pred_idx) in enumerate(zip(embryo_ids, y_labels, pred_classes)):
            pred_label = int_to_label[int(pred_idx)]
            pred_record: Dict[str, Any] = {
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
            tol = 1e-6
            if not (-tol <= raw_p_pred <= 1 + tol):
                raise ValueError(f"p_pred={raw_p_pred} out of range for embryo_id={eid}, time_bin={t}")
            if not (-tol <= raw_p_true <= 1 + tol):
                raise ValueError(f"p_true={raw_p_true} out of range for embryo_id={eid}, time_bin={t}")

            pred_record["p_pred"] = float(np.clip(raw_p_pred, 0.0, 1.0))
            pred_record["p_true"] = float(np.clip(raw_p_true, 0.0, 1.0))
            pred_record["is_wrong"] = int(not pred_record["is_correct"])

            all_embryo_predictions.append(pred_record)

    ovr_dfs = {
        class_label: pd.DataFrame(rows) if rows else pd.DataFrame()
        for class_label, rows in ovr_results.items()
    }

    embryo_predictions_df = (
        pd.DataFrame(all_embryo_predictions) if all_embryo_predictions else None
    )

    data_quality = {
        "total_bins": total_bins,
        "bins_with_all_classes": n_complete_bins,
        "bins_with_missing_classes": n_incomplete_bins,
        "bins_skipped": n_skipped_bins,
    }

    return ovr_dfs, confusion_matrices, embryo_predictions_df, data_quality


def _compute_multiclass_summary(
    ovr_results: Dict[str, pd.DataFrame],
    embryo_predictions: Optional[pd.DataFrame],
    class_labels: List[str],
    alpha: float = 0.05,
    data_quality: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    per_class_summary: Dict[str, Any] = {}

    for class_label in class_labels:
        df = ovr_results.get(class_label)

        if df is None or df.empty:
            per_class_summary[class_label] = {
                "earliest_significant_hpf": None,
                "max_auroc": None,
                "max_auroc_hpf": None,
                "n_significant_bins": 0,
            }
            continue

        sig_mask = df["pval"] < alpha
        n_significant = int(sig_mask.sum())
        earliest_hpf = int(df.loc[sig_mask, "time_bin"].min()) if n_significant > 0 else None
        max_idx = int(df["auroc_observed"].idxmax())

        per_class_summary[class_label] = {
            "earliest_significant_hpf": earliest_hpf,
            "max_auroc": float(df.loc[max_idx, "auroc_observed"]),
            "max_auroc_hpf": int(df.loc[max_idx, "time_bin"]),
            "n_significant_bins": n_significant,
        }

    overall_accuracy = None
    if embryo_predictions is not None and not embryo_predictions.empty:
        overall_accuracy = float(embryo_predictions["is_correct"].mean())

    return {
        "per_class": per_class_summary,
        "overall_accuracy": overall_accuracy,
        "data_quality": data_quality or {},
    }


def extract_temporal_confusion_profile(
    confusion_matrices: Dict[int, pd.DataFrame],
    class_labels: List[str],
) -> pd.DataFrame:
    records = []
    for time_bin, cm_df in sorted(confusion_matrices.items()):
        for true_class in class_labels:
            if true_class not in cm_df.index:
                continue
            row_sum = cm_df.loc[true_class].sum()
            if not np.isfinite(row_sum) or row_sum == 0:
                continue

            for pred_class in class_labels:
                if pred_class not in cm_df.columns:
                    continue
                count = cm_df.loc[true_class, pred_class]
                proportion = count / row_sum
                records.append(
                    {
                        "time_bin": int(time_bin),
                        "true_class": true_class,
                        "predicted_class": pred_class,
                        "proportion": float(proportion),
                        "count": float(count),
                        "is_correct": bool(true_class == pred_class),
                    }
                )

    return pd.DataFrame(records)


def run_multiclass_classification_test(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    features: Union[str, List[str]] = "z_mu_b",
    time_col: str = "predicted_stage_hpf",
    embryo_id_col: str = "embryo_id",
    bin_width: float = 4.0,
    n_splits: int = 5,
    n_permutations: int = 100,
    n_jobs: int = 1,
    min_samples_per_class: int = 3,
    within_bin_time_stratification: bool = True,
    within_bin_time_strata_width: float = 0.5,
    skip_bin_if_not_all_present: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run multiclass OvR AUROC-based comparison for explicitly provided groups."""
    class_labels = list(groups.keys())
    n_classes = len(class_labels)
    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {n_classes}")

    if verbose:
        print(f"Multiclass comparison with {n_classes} classes: {class_labels}")
        print("Classification weighting: class_weight='balanced' (enabled by default)")

    embryo_to_class: Dict[str, str] = {}
    for class_label, embryo_ids in groups.items():
        for eid in embryo_ids:
            embryo_to_class[str(eid)] = class_label

    all_embryo_ids = list(embryo_to_class.keys())
    df_filtered = df[df[embryo_id_col].astype(str).isin(all_embryo_ids)].copy()
    if len(df_filtered) == 0:
        raise ValueError("No data found for specified embryo IDs")

    df_filtered["_class_label"] = df_filtered[embryo_id_col].astype(str).map(embryo_to_class)
    feature_cols = _resolve_feature_columns(df_filtered, features)

    df_filtered["_time_bin"] = (
        np.floor(df_filtered[time_col] / bin_width) * bin_width
    ).astype(int)

    groupby_cols = [embryo_id_col, "_time_bin", "_class_label"]
    df_binned = df_filtered.groupby(groupby_cols, as_index=False)[feature_cols].mean()

    ovr_results, confusion_matrices, embryo_predictions, data_quality = _run_multiclass_classification(
        df_binned=df_binned,
        df_raw=df_filtered,
        class_labels=class_labels,
        feature_cols=feature_cols,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        bin_width=bin_width,
        n_splits=n_splits,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        min_samples_per_class=min_samples_per_class,
        within_bin_time_stratification=within_bin_time_stratification,
        within_bin_time_strata_width=within_bin_time_strata_width,
        skip_bin_if_not_all_present=skip_bin_if_not_all_present,
        random_state=random_state,
        verbose=verbose,
    )

    summary = _compute_multiclass_summary(
        ovr_results,
        embryo_predictions,
        class_labels,
        data_quality=data_quality,
    )

    config = {
        "class_labels": class_labels,
        "n_classes": n_classes,
        "features": feature_cols,
        "bin_width": float(bin_width),
        "n_permutations": int(n_permutations),
        "n_splits": int(n_splits),
        "n_jobs": int(n_jobs),
        "group_sizes": {label: len(ids) for label, ids in groups.items()},
        "within_bin_time_stratification": bool(within_bin_time_stratification),
        "within_bin_time_strata_width": float(within_bin_time_strata_width),
        "skip_bin_if_not_all_present": bool(skip_bin_if_not_all_present),
    }

    return {
        "ovr_classification": ovr_results,
        "confusion_matrices": confusion_matrices,
        "embryo_predictions": embryo_predictions,
        "summary": summary,
        "config": config,
    }


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return ""


def _time_bin_definition_from_df(df: pd.DataFrame, bin_width: float) -> List[int]:
    bins = sorted(df["time_bin"].dropna().astype(int).unique().tolist())
    if not bins:
        return []
    return bins + [bins[-1] + int(round(bin_width))]


def _time_edges_hash(edges: List[int]) -> str:
    raw = json.dumps(edges, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def run_classification_test(
    df: pd.DataFrame,
    groupby: str,
    groups: Union[str, List[str]] = "all",
    reference: Union[str, List[Union[str, Tuple[str, ...]]]] = "rest",
    features: Union[str, List[str]] = "z_mu_b",
    time_col: str = "predicted_stage_hpf",
    embryo_id_col: str = "embryo_id",
    bin_width: float = 4.0,
    n_splits: int = 5,
    n_permutations: int = 100,
    n_jobs: int = 1,
    min_samples_per_class: int = 3,
    within_bin_time_stratification: bool = True,
    within_bin_time_strata_width: float = 0.5,
    allow_many_comparisons: bool = False,
    max_comparisons: int = 50,
    random_state: int = 42,
    verbose: bool = True,
    save_null: bool = True,
    null_save_mode: str = "summary",
) -> MulticlassOVRResults:
    """Run flexible group comparison tests and return a consolidated result object."""
    if null_save_mode not in {"summary", "full"}:
        raise ValueError("null_save_mode must be one of {'summary','full'}")

    if verbose:
        print("=== Flexible Group Classification Test ===")
        print(f"Groupby column: {groupby}")
        print(f"Groups: {groups}")
        print(f"Reference: {reference}")
        print("Classifier policy: class_weight='balanced' for all logistic models")

    comparison_specs = _resolve_comparison_groups(
        df=df,
        groupby=groupby,
        groups=groups,
        reference=reference,
        embryo_id_col=embryo_id_col,
        allow_many_comparisons=allow_many_comparisons,
        max_comparisons=max_comparisons,
    )

    # Fast path: true multiclass all-vs-rest in one run, preserving full per-class probabilities.
    all_group_values = sorted(df[groupby].dropna().astype(str).unique().tolist())
    selected_groups = all_group_values if groups == "all" else [str(g) for g in groups]
    is_all_rest_path = (
        reference == "rest"
        and set(selected_groups) == set(all_group_values)
    )

    all_rows: List[dict[str, Any]] = []
    embryo_predictions_augmented: Optional[pd.DataFrame] = None
    null_summary_df: Optional[pd.DataFrame] = None
    confusion_profile_df: Optional[pd.DataFrame] = None

    if is_all_rest_path:
        if verbose:
            print("\nDetected groups='all' and reference='rest': running one unified multiclass model.")

        groups_map = {
            g: df.loc[df[groupby].astype(str) == g, embryo_id_col].dropna().astype(str).unique().tolist()
            for g in selected_groups
        }

        result = run_multiclass_classification_test(
            df=df,
            groups=groups_map,
            features=features,
            time_col=time_col,
            embryo_id_col=embryo_id_col,
            bin_width=bin_width,
            n_splits=n_splits,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            min_samples_per_class=min_samples_per_class,
            within_bin_time_stratification=within_bin_time_stratification,
            within_bin_time_strata_width=within_bin_time_strata_width,
            skip_bin_if_not_all_present=False,
            random_state=random_state,
            verbose=verbose,
        )

        for class_label, class_df in result["ovr_classification"].items():
            if class_df is None or class_df.empty:
                continue
            spec = next((s for s in comparison_specs if s.positive == class_label and s.negative == "rest"), None)
            if spec is None:
                # Should not happen in this path, but keep robust.
                continue
            for _, row in class_df.iterrows():
                row_dict = row.to_dict()
                row_dict["comparison_id"] = spec.comparison_id
                row_dict["groupby"] = groupby
                row_dict["positive"] = spec.positive
                row_dict["negative"] = spec.negative
                row_dict["negative_members"] = json.dumps(spec.negative_members)
                row_dict["negative_mode"] = spec.negative_mode
                if "auroc_observed" in row_dict:
                    row_dict["auroc_obs"] = row_dict.pop("auroc_observed")
                all_rows.append(row_dict)

        embryo_predictions_augmented = result.get("embryo_predictions")
        if embryo_predictions_augmented is not None and not embryo_predictions_augmented.empty:
            for c in selected_groups:
                col = f"pred_proba_{c}"
                if col not in embryo_predictions_augmented.columns:
                    raise ValueError(f"Hard contract violation: missing probability column {col}")

            # Hard criterion: all rows must have probability values for all classes.
            prob_cols = [f"pred_proba_{c}" for c in selected_groups]
            if embryo_predictions_augmented[prob_cols].isna().any().any():
                bad = embryo_predictions_augmented[prob_cols].isna().sum().to_dict()
                raise ValueError(f"Hard contract violation: NaN in pred_proba columns: {bad}")

        null_rows = []
        for class_label, class_df in result["ovr_classification"].items():
            if class_df is None or class_df.empty:
                continue
            for _, row in class_df.iterrows():
                null_rows.append(
                    {
                        "class": class_label,
                        "time_bin": int(row["time_bin"]),
                        "exceed_count": int(row.get("auroc_null_exceed_count", 0)),
                        "null_mean": float(row.get("auroc_null_mean", np.nan)),
                        "null_std": float(row.get("auroc_null_std", np.nan)),
                        "n_permutations": int(row.get("auroc_n_permutations", n_permutations)),
                    }
                )
        null_summary_df = pd.DataFrame(null_rows)

        confusion_profile_df = extract_temporal_confusion_profile(
            result["confusion_matrices"],
            selected_groups,
        )
    else:
        for spec in comparison_specs:
            if verbose:
                print(f"\n--- Running: {spec.positive} vs {spec.negative} ---")

            groups_dict = {
                spec.positive: spec.positive_members,
                spec.negative: spec.negative_members,
            }

            result = run_multiclass_classification_test(
                df=df,
                groups=groups_dict,
                features=features,
                time_col=time_col,
                embryo_id_col=embryo_id_col,
                bin_width=bin_width,
                n_splits=n_splits,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                min_samples_per_class=min_samples_per_class,
                within_bin_time_stratification=within_bin_time_stratification,
                within_bin_time_strata_width=within_bin_time_strata_width,
                skip_bin_if_not_all_present=True,
                random_state=random_state,
                verbose=verbose,
            )

            positive_df = result["ovr_classification"].get(spec.positive)
            if positive_df is None or positive_df.empty:
                if verbose:
                    print(f"  Warning: No results for {spec.positive}")
                continue

            for _, row in positive_df.iterrows():
                row_dict = row.to_dict()
                row_dict["comparison_id"] = spec.comparison_id
                row_dict["groupby"] = groupby
                row_dict["positive"] = spec.positive
                row_dict["negative"] = spec.negative
                row_dict["negative_members"] = json.dumps(spec.negative_members)
                row_dict["negative_mode"] = spec.negative_mode
                if "auroc_observed" in row_dict:
                    row_dict["auroc_obs"] = row_dict.pop("auroc_observed")
                all_rows.append(row_dict)

    if not all_rows:
        raise ValueError("No valid comparisons produced results")

    comparisons_df = pd.DataFrame(all_rows)

    metadata = {
        "groupby": groupby,
        "groups": groups if isinstance(groups, list) else [groups] if groups != "all" else "all",
        "reference": reference,
        "features": features if isinstance(features, list) else [features],
        "bin_width": float(bin_width),
        "n_permutations": int(n_permutations),
        "n_splits": int(n_splits),
        "n_jobs": int(n_jobs),
        "min_samples_per_class": int(min_samples_per_class),
        "within_bin_time_stratification": bool(within_bin_time_stratification),
        "within_bin_time_strata_width": float(within_bin_time_strata_width),
        "max_comparisons": int(max_comparisons),
        "n_comparisons": len(comparison_specs),
        "timestamp": datetime.now().isoformat(),
        "random_state": int(random_state),
        "save_null": bool(save_null),
        "null_save_mode": null_save_mode,
    }

    if embryo_predictions_augmented is not None and not embryo_predictions_augmented.empty:
        class_labels = sorted(set(embryo_predictions_augmented["true_class"].unique()))
        edges = _time_bin_definition_from_df(embryo_predictions_augmented, bin_width)
        metadata["stage1_null_metadata"] = {
            "class_labels": class_labels,
            "time_bin_definition": edges,
            "time_bin_center_formula": "midpoint",
            "time_bin_edges_sha256": _time_edges_hash(edges),
            "seed": int(random_state),
            "n_permutations": int(n_permutations),
            "git_commit": _git_commit(),
            "timestamp": datetime.now().isoformat(),
            "schema_version": "classification_v1",
        }

    results = MulticlassOVRResults(
        comparisons=comparisons_df,
        metadata=metadata,
        embryo_predictions=embryo_predictions_augmented,
        null_summary=null_summary_df if save_null else None,
        confusion_profile=confusion_profile_df,
    )

    if verbose:
        print("\n=== Complete ===")
        print(f"{len(results)} comparisons")
        print(f"{len(comparisons_df)} total rows")

    return results


__all__ = [
    "run_multiclass_classification_test",
    "run_classification_test",
    "extract_temporal_confusion_profile",
]
