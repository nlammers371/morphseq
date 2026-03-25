"""Top-level orchestrator: ``run_classification()``."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .engine.analysis import ClassificationAnalysis, _LazyLayers
from .engine.comparison_resolution import (
    ComparisonScheme,
    ResolvedComparison,
    UserComparisonSpec,
    check_min_samples,
    resolve_comparisons,
)
from .engine.loop import (
    _bin_and_aggregate,
    _build_binary_labels,
    _collect_binary_predictions,
    _collect_confusion,
    _collect_confusion_from_ovr,
    _collect_multiclass_predictions,
    _collect_scores,
    _collect_scores_from_ovr,
    _resolve_feature_columns,
    _run_binary_classification_loop,
    _run_multiclass_classification_loop,
)
from .engine.null import NullDistributions


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        )
        return out.strip()
    except Exception:
        return ""


def _is_all_vs_rest_mode(
    positive: UserComparisonSpec | None,
    negative: UserComparisonSpec | None,
    comparisons: ComparisonScheme,
) -> bool:
    """Detect whether we're in true all-vs-rest mode (multiclass fast path)."""
    if negative is not None:
        return False
    if isinstance(comparisons, (pd.DataFrame, list)):
        return False
    if comparisons == "all_pairs":
        return False
    # all_vs_rest or None with no negative → yes
    return True


def run_classification(
    df: pd.DataFrame,
    *,
    class_col: str,
    id_col: str,
    time_col: str,
    positive: UserComparisonSpec | None = None,
    negative: UserComparisonSpec | None = None,
    comparisons: ComparisonScheme = None,
    features: dict[str, str | list[str]],
    bin_width: float = 4.0,
    n_permutations: int = 100,
    n_splits: int = 5,
    min_samples_per_group: int = 3,
    min_samples_per_member: int = 2,
    n_jobs: int = 1,
    random_state: int = 42,
    verbose: bool = True,
    save_predictions: bool = False,
    save_multiclass_predictions: bool = False,
    save_null_arrays: bool = False,
    save_dir: str | Path | None = None,
) -> ClassificationAnalysis:
    """Run classification comparisons and return a ``ClassificationAnalysis``.

    This is the single entry point for all classification modes.
    """
    # -- 1. Resolve features -------------------------------------------------
    resolved_features = _resolve_feature_columns(df, features)

    # -- 2. Resolve comparisons ----------------------------------------------
    available_labels = set(df[class_col].dropna().unique().astype(str))
    resolved = resolve_comparisons(
        positive=positive,
        negative=negative,
        comparisons=comparisons,
        available_labels=available_labels,
        class_col=class_col,
    )

    # -- 3. Check min samples ------------------------------------------------
    label_counts = (
        df.groupby(class_col)[id_col].nunique().to_dict()
    )
    check_min_samples(
        resolved, label_counts,
        min_samples_per_group=min_samples_per_group,
        min_samples_per_member=min_samples_per_member,
    )

    # -- 4. Detect mode ------------------------------------------------------
    # The multiclass fast path runs a single multiclass model per bin and
    # extracts per-class binary AUROCs.  It is appropriate when every
    # comparison is "one class vs rest" with a single-label positive side.
    # The negative side will be pooled (the rest), which is expected.
    use_multiclass_fast_path = (
        _is_all_vs_rest_mode(positive, negative, comparisons)
        and all(not rc.is_pooled_positive for rc in resolved)
    )

    if verbose:
        print(f"=== run_classification ===")
        print(f"  {len(resolved_features)} feature set(s): {list(resolved_features.keys())}")
        print(f"  {len(resolved)} comparison(s)")
        if use_multiclass_fast_path:
            print(f"  Mode: multiclass all-vs-rest (fast path)")
        else:
            print(f"  Mode: binary per-comparison")

    # -- 5. Main loop --------------------------------------------------------
    all_score_rows: list[dict[str, Any]] = []
    all_pred_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    all_null_arrays: list[tuple[str, str, float, np.ndarray]] = []
    multiclass_preds_df: pd.DataFrame | None = None

    for fs_name, feature_cols in resolved_features.items():
        if verbose:
            print(f"\n--- Feature set: {fs_name} ({len(feature_cols)} cols) ---")

        if use_multiclass_fast_path:
            # Build resolved_map: class_label → ResolvedComparison
            resolved_map: dict[str, ResolvedComparison] = {}
            for rc in resolved:
                if len(rc.positive_members) == 1:
                    resolved_map[rc.positive_members[0]] = rc

            class_labels = sorted(resolved_map.keys())

            # Prepare binned data for multiclass
            df_str = df.copy()
            df_str[class_col] = df_str[class_col].astype(str)
            df_mc = df_str[df_str[class_col].isin(class_labels)].copy()
            df_mc["_class_label"] = df_mc[class_col]
            df_mc["_time_bin"] = (
                np.floor(df_mc[time_col] / bin_width) * bin_width
            ).astype(int)
            groupby_cols = [id_col, "_time_bin", "_class_label"]
            df_binned = df_mc.groupby(groupby_cols, as_index=False)[feature_cols].mean()

            ovr_results, embryo_preds = _run_multiclass_classification_loop(
                df_binned=df_binned,
                class_labels=class_labels,
                feature_cols=feature_cols,
                id_col=id_col,
                bin_width=bin_width,
                n_splits=n_splits,
                n_permutations=n_permutations,
                n_jobs=n_jobs,
                min_samples_per_class=min_samples_per_group,
                random_state=random_state,
                verbose=verbose,
            )

            # Collect scores
            score_rows = _collect_scores_from_ovr(
                ovr_results, class_labels, resolved_map, fs_name,
            )
            all_score_rows.extend(score_rows)

            # Collect confusion from multiclass predictions
            confusion_rows = _collect_confusion_from_ovr(
                ovr_results, embryo_preds, class_labels,
                resolved_map, fs_name, bin_width,
            )
            all_confusion_rows.extend(confusion_rows)

            # Multiclass predictions (wide format)
            if save_multiclass_predictions and embryo_preds:
                mc_df = _collect_multiclass_predictions(embryo_preds)
                if mc_df is not None:
                    mc_df["feature_set"] = fs_name
                    multiclass_preds_df = mc_df

            # Null arrays
            if save_null_arrays:
                for cl in class_labels:
                    for br in ovr_results.get(cl, []):
                        arr = br.get("_null_array")
                        if arr is not None and len(arr) > 0:
                            rc = resolved_map[cl]
                            all_null_arrays.append(
                                (fs_name, rc.comparison_id, br["time_bin_center"], arr)
                            )

        else:
            # Binary per-comparison path
            for rc in resolved:
                if verbose:
                    print(f"  {rc.positive_label} vs {rc.negative_label}")

                df_labeled = _build_binary_labels(df, class_col, rc)
                df_binned = _bin_and_aggregate(
                    df_labeled, id_col, time_col, feature_cols, bin_width,
                )

                bin_results = _run_binary_classification_loop(
                    df_binned=df_binned,
                    feature_cols=feature_cols,
                    id_col=id_col,
                    bin_width=bin_width,
                    n_splits=n_splits,
                    n_permutations=n_permutations,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbose=verbose,
                )

                all_score_rows.extend(
                    _collect_scores(bin_results, rc, fs_name)
                )

                if save_predictions:
                    all_pred_rows.extend(
                        _collect_binary_predictions(bin_results, rc, fs_name, id_col)
                    )

                all_confusion_rows.extend(
                    _collect_confusion(bin_results, rc, fs_name)
                )

                if save_null_arrays:
                    for br in bin_results:
                        arr = br.get("_null_array")
                        if arr is not None and len(arr) > 0:
                            all_null_arrays.append(
                                (fs_name, rc.comparison_id, br["time_bin_center"], arr)
                            )

    # -- 6. Assemble outputs -------------------------------------------------
    if not all_score_rows:
        raise ValueError("No valid comparisons produced results.")

    scores = pd.DataFrame(all_score_rows)

    # Build uns
    uns: dict[str, Any] = {
        "schema_version": "classification_v1",
        "created_at": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "class_col": class_col,
        "id_col": id_col,
        "time_col": time_col,
        "bin_width": bin_width,
        "n_permutations": n_permutations,
        "feature_sets": {
            name: {
                "spec": features[name],
                "columns": cols,
            }
            for name, cols in resolved_features.items()
        },
        "comparisons": {
            rc.comparison_id: {
                "positive_members": list(rc.positive_members),
                "negative_members": list(rc.negative_members),
                "positive_label": rc.positive_label,
                "negative_label": rc.negative_label,
            }
            for rc in resolved
        },
    }

    # Build layers
    layers = _LazyLayers()

    if all_pred_rows:
        layers.store("predictions", pd.DataFrame(all_pred_rows))

    if multiclass_preds_df is not None:
        layers.store("multiclass_predictions", multiclass_preds_df)

    if all_confusion_rows:
        layers.store("confusion", pd.DataFrame(all_confusion_rows))

    if all_null_arrays:
        n_rows = len(all_null_arrays)
        max_p = max(len(arr) for _, _, _, arr in all_null_arrays)
        null_auc = np.full((n_rows, max_p), np.nan, dtype=np.float32)
        fs_arr = np.empty(n_rows, dtype=object)
        cid_arr = np.empty(n_rows, dtype=object)
        tbc_arr = np.empty(n_rows, dtype=np.float64)
        for i, (fs, cid, tbc, arr) in enumerate(all_null_arrays):
            null_auc[i, : len(arr)] = arr.astype(np.float32)
            fs_arr[i] = fs
            cid_arr[i] = cid
            tbc_arr[i] = tbc
        layers.store("null_full", NullDistributions(
            null_auc=null_auc,
            feature_set=fs_arr,
            comparison_id=cid_arr,
            time_bin_center=tbc_arr,
        ))

    result = ClassificationAnalysis(scores=scores, uns=uns, layers=layers)

    if verbose:
        print(f"\n=== Complete ===")
        print(f"  {len(scores)} score rows")
        print(f"  {len(resolved)} comparison(s)")
        print(f"  Layers: {layers.cached()}")

    # Save immediately if save_dir is provided
    if save_dir is not None:
        result.save(save_dir)
        if verbose:
            print(f"  Saved to {save_dir}")

    return result
