"""
FIXED VERSION: Robust cross-validation that ensures consistent predictions
across different imbalance-handling methods.

KEY FIX: Pre-split CV folds ONCE per time bin, then use the same folds for
all methods. This ensures identical embryo coverage across methods.
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, brier_score_loss

# ============================================================================
# CONFIGURATION
# ============================================================================

N_CV_SPLITS = 5
RANDOM_SEED = 42

# ============================================================================
# HELPER FUNCTIONS FOR WEIGHTING
# ============================================================================

def compute_embryo_weights(embryo_ids, y):
    """Compute sample weights so each embryo contributes equally."""
    unique_embryos, counts = np.unique(embryo_ids, return_counts=True)
    n_embryos = len(unique_embryos)

    embryo_weight_map = {embryo: 1.0 / (n_embryos * count)
                         for embryo, count in zip(unique_embryos, counts)}

    weights = np.array([embryo_weight_map[eid] for eid in embryo_ids])
    weights = weights * len(weights) / weights.sum()

    return weights


def compute_combined_weights(embryo_ids, y):
    """Combine embryo-level weighting with class balancing."""
    from sklearn.utils.class_weight import compute_class_weight

    embryo_weights = compute_embryo_weights(embryo_ids, y)

    classes = np.unique(y)
    class_weights_dict = dict(zip(
        classes,
        compute_class_weight('balanced', classes=classes, y=y)
    ))
    class_weights = np.array([class_weights_dict[label] for label in y])

    combined = embryo_weights * class_weights
    combined = combined * len(combined) / combined.sum()

    return combined


# ============================================================================
# FIXED CLASSIFICATION METHODS WITH SHARED CV FOLDS
# ============================================================================

def fit_predict_with_shared_folds(
    method_name,
    X_train, y_train, X_test, y_test,
    embryo_ids_train=None,
    embryo_ids_test=None,
    random_state=None
):
    """
    Fit and predict using specified method.
    All methods now use the SAME train/test split passed in.
    """
    if method_name == 'baseline':
        model = LogisticRegression(max_iter=200, random_state=random_state)
        model.fit(X_train, y_train)

    elif method_name == 'class_weight':
        model = LogisticRegression(max_iter=200, class_weight='balanced', random_state=random_state)
        model.fit(X_train, y_train)

    elif method_name == 'embryo_weight':
        if embryo_ids_train is None:
            raise ValueError("embryo_ids_train required for embryo_weight")
        sample_weights = compute_embryo_weights(embryo_ids_train, y_train)
        model = LogisticRegression(max_iter=200, random_state=random_state)
        model.fit(X_train, y_train, sample_weight=sample_weights)

    elif method_name == 'combined_weight':
        if embryo_ids_train is None:
            raise ValueError("embryo_ids_train required for combined_weight")
        sample_weights = compute_combined_weights(embryo_ids_train, y_train)
        model = LogisticRegression(max_iter=200, random_state=random_state)
        model.fit(X_train, y_train, sample_weight=sample_weights)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Predict
    pred_proba = model.predict_proba(X_test)[:, 1]

    return pred_proba, model


def predictive_signal_test_robust(
    df_binned,
    methods,
    group_col="genotype",
    time_col="time_bin",
    z_cols=None,
    n_splits=5,
    random_state=None,
    return_embryo_probs=True
):
    """
    ROBUST VERSION: Run predictive signal test with multiple methods,
    ensuring all methods see the exact same CV splits.

    This guarantees identical embryo coverage across methods.
    """
    rng = np.random.default_rng(random_state)
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]

    # Initialize storage for each method
    results_by_method = {method_name: {'results': [], 'embryo_preds': []}
                         for method_name in methods}

    for t, sub in df_binned.groupby(time_col):
        X = sub[z_cols].values
        y = sub[group_col].values
        embryo_ids = sub['embryo_id'].values

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            continue

        min_samples_per_class = min([np.sum(y == c) for c in unique_classes])
        if min_samples_per_class < n_splits:
            continue

        # ====================================================================
        # KEY FIX: Create CV splits ONCE for this time bin
        # All methods will use these same splits
        # ====================================================================
        skf = StratifiedKFold(
            n_splits=min(n_splits, min_samples_per_class),
            shuffle=True,
            random_state=random_state
        )

        # Pre-generate all fold indices
        fold_splits = list(skf.split(X, y))

        # Run each method using the SAME fold splits
        for method_name in methods:
            try:
                method_aucs = []
                method_ba = []
                method_prauc = []
                method_brier = []

                for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                    # Get embryo IDs
                    embryo_train = embryo_ids[train_idx]
                    embryo_test = embryo_ids[test_idx]

                    # Fit and predict using this method
                    pred_proba, model = fit_predict_with_shared_folds(
                        method_name,
                        X[train_idx], y[train_idx],
                        X[test_idx], y[test_idx],
                        embryo_ids_train=embryo_train,
                        embryo_ids_test=embryo_test,
                        random_state=random_state
                    )

                    # Convert labels to binary
                    class_order = model.classes_
                    positive_class = class_order[1]
                    y_test_binary = (y[test_idx] == positive_class).astype(int)

                    # Compute metrics
                    method_aucs.append(roc_auc_score(y_test_binary, pred_proba))
                    # For balanced accuracy, use 0.5 threshold
                    pred_label = (pred_proba > 0.5).astype(int)
                    method_ba.append(balanced_accuracy_score(y_test_binary, pred_label))
                    method_prauc.append(average_precision_score(y_test_binary, pred_proba))
                    method_brier.append(brier_score_loss(y_test_binary, pred_proba))

                    # Store embryo-level predictions
                    if return_embryo_probs:
                        for i, idx in enumerate(test_idx):
                            true_label = y[idx]
                            p_pos = pred_proba[i]
                            support_true = p_pos if true_label == positive_class else 1.0 - p_pos
                            signed_margin = (1 if true_label == positive_class else -1) * (p_pos - 0.5)

                            results_by_method[method_name]['embryo_preds'].append({
                                'embryo_id': embryo_ids[idx],
                                'time_bin': t,
                                'fold_idx': fold_idx,
                                'true_label': true_label,
                                'pred_proba': p_pos,
                                'confidence': np.abs(p_pos - 0.5),
                                'predicted_label': positive_class if p_pos > 0.5 else class_order[0],
                                'support_true': support_true,
                                'signed_margin': signed_margin
                            })

                # Store aggregate metrics
                results_by_method[method_name]['results'].append({
                    'time_bin': t,
                    'AUROC_mean': np.mean(method_aucs),
                    'AUROC_std': np.std(method_aucs),
                    'balanced_accuracy_mean': np.mean(method_ba),
                    'balanced_accuracy_std': np.std(method_ba),
                    'PR_AUC_mean': np.mean(method_prauc),
                    'PR_AUC_std': np.std(method_prauc),
                    'brier_score_mean': np.mean(method_brier),
                    'brier_score_std': np.std(method_brier),
                    'n_samples': len(y)
                })

            except Exception as e:
                print(f"    Error with method {method_name} at time {t}: {e}")
                continue

    # Convert lists to DataFrames
    output = {}
    for method_name in methods:
        df_results = pd.DataFrame(results_by_method[method_name]['results'])
        df_embryo_probs = None
        if return_embryo_probs and results_by_method[method_name]['embryo_preds']:
            df_embryo_probs = pd.DataFrame(results_by_method[method_name]['embryo_preds'])

        output[method_name] = {
            'df_results': df_results,
            'df_embryo_probs': df_embryo_probs
        }

    return output


# ============================================================================
# VERIFICATION FUNCTION
# ============================================================================

def verify_coverage_consistency(results_by_method):
    """
    Verify that all methods have identical embryo coverage.

    Returns True if all methods have same embryos × time bins, False otherwise.
    """
    print("\n" + "="*80)
    print("COVERAGE CONSISTENCY VERIFICATION")
    print("="*80)

    method_names = list(results_by_method.keys())

    for method_name, data in results_by_method.items():
        df_embryo = data['df_embryo_probs']
        if df_embryo is None or df_embryo.empty:
            print(f"  {method_name}: NO DATA")
            continue

        n_predictions = len(df_embryo)
        n_embryos = df_embryo['embryo_id'].nunique()
        n_time_bins = df_embryo['time_bin'].nunique()

        print(f"  {method_name}:")
        print(f"    Total predictions: {n_predictions}")
        print(f"    Unique embryos: {n_embryos}")
        print(f"    Unique time bins: {n_time_bins}")
        print(f"    Predictions/embryo: {n_predictions / n_embryos:.2f}")

    # Check if coverage is identical across methods
    print("\n" + "-"*80)
    print("Pairwise Coverage Comparison:")
    print("-"*80)

    all_consistent = True

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            method1 = method_names[i]
            method2 = method_names[j]

            df1 = results_by_method[method1]['df_embryo_probs']
            df2 = results_by_method[method2]['df_embryo_probs']

            if df1 is None or df2 is None:
                continue

            # Create sets of (embryo_id, time_bin) tuples
            coverage1 = set(zip(df1['embryo_id'], df1['time_bin']))
            coverage2 = set(zip(df2['embryo_id'], df2['time_bin']))

            if coverage1 == coverage2:
                print(f"  {method1} vs {method2}: ✓ IDENTICAL ({len(coverage1)} predictions)")
            else:
                print(f"  {method1} vs {method2}: ✗ DIFFERENT")
                print(f"    {method1} only: {len(coverage1 - coverage2)}")
                print(f"    {method2} only: {len(coverage2 - coverage1)}")
                all_consistent = False

    print("\n" + "="*80)
    if all_consistent:
        print("✓ ALL METHODS HAVE IDENTICAL COVERAGE")
    else:
        print("✗ COVERAGE INCONSISTENCY DETECTED")
    print("="*80)

    return all_consistent


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ROBUST CV METHOD - VERIFICATION TEST")
    print("="*80)

    print("\nThis module provides fixed CV methods that ensure consistent")
    print("embryo coverage across all imbalance-handling methods.")
    print("\nKey function: predictive_signal_test_robust()")
    print("Key verification: verify_coverage_consistency()")
    print("\nImport this module into your main analysis script to use the fixed methods.")
