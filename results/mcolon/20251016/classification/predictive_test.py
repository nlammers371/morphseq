"""
Predictive classification and permutation testing for phenotype emergence.

This module implements the core predictive signal test that evaluates whether
genotype labels can be predicted from morphological features better than chance.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone


def _identify_wt_mutant_classes(class_order: np.ndarray) -> Tuple[str, str, int, int]:
    """
    Identify which class corresponds to WT vs mutant labels.

    Parameters
    ----------
    class_order : np.ndarray
        Class labels returned by a fitted classifier via `classes_`.

    Returns
    -------
    Tuple[str, str, int, int]
        wt_class, mutant_class, wt_idx, mutant_idx
    """
    wt_candidates = [
        c for c in class_order
        if 'wildtype' in str(c).lower() or str(c).lower() in ['wik', 'ab', 'wik-ab']
    ]
    mutant_candidates = [c for c in class_order if c not in wt_candidates]

    if len(wt_candidates) == 1 and len(mutant_candidates) == 1:
        wt_class = wt_candidates[0]
        mutant_class = mutant_candidates[0]
        wt_idx = int(np.where(class_order == wt_class)[0][0])
        mutant_idx = int(np.where(class_order == mutant_class)[0][0])
        return wt_class, mutant_class, wt_idx, mutant_idx

    # Fallback: assume the second class is the positive/mutant class
    wt_class = class_order[0]
    mutant_class = class_order[1]
    return wt_class, mutant_class, 0, 1


def _run_single_cv_fold(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    embryo_ids_test: Optional[np.ndarray],
    time_bin,
    return_predictions: bool = False
) -> Tuple[float, Optional[List[dict]]]:
    """
    Train and evaluate a single cross-validation fold.

    Parameters
    ----------
    model :
        Base estimator to clone and fit for the fold.
    X_train, y_train, X_test, y_test :
        Training and test splits for the fold.
    embryo_ids_test : np.ndarray or None
        Embryo identifiers corresponding to X_test (required if return_predictions=True).
    time_bin :
        Time bin label associated with the fold.
    return_predictions : bool, default=False
        Whether to collect per-embryo prediction diagnostics.

    Returns
    -------
    Tuple[float, Optional[List[dict]]]
        AUROC for the fold and optional list of embryo-level prediction records.
    """
    estimator = clone(model)
    estimator.fit(X_train, y_train)
    proba = estimator.predict_proba(X_test)

    class_order = estimator.classes_
    if len(class_order) != 2:
        raise ValueError("Expected binary classification with two classes.")

    wt_class, mutant_class, wt_idx, mutant_idx = _identify_wt_mutant_classes(class_order)
    mutant_probs = proba[:, mutant_idx]
    auroc = roc_auc_score(y_test, mutant_probs)

    if not return_predictions:
        return auroc, None

    if embryo_ids_test is None:
        raise ValueError("embryo_ids_test is required when return_predictions=True.")

    predictions: List[dict] = []
    for i, (true_label, embryo_id) in enumerate(zip(y_test, embryo_ids_test)):
        row_proba = proba[i]
        predicted_idx = int(np.argmax(row_proba))
        predicted_label = class_order[predicted_idx]

        # Probability columns with explicit class labels
        proba_cols = {
            f"pred_proba_{str(class_order[0])}": row_proba[0],
            f"pred_proba_{str(class_order[1])}": row_proba[1],
        }

        true_idx_matches = np.where(class_order == true_label)[0]
        true_idx = int(true_idx_matches[0]) if len(true_idx_matches) else predicted_idx
        support_true = row_proba[true_idx]
        mutant_prob = row_proba[mutant_idx]
        signed_margin = (1 if true_label == mutant_class else -1) * (mutant_prob - 0.5)

        predictions.append({
            'embryo_id': embryo_id,
            'time_bin': time_bin,
            'true_label': true_label,
            **proba_cols,
            'predicted_label': predicted_label,
            'confidence': abs(mutant_prob - 0.5),
            'support_true': support_true,
            'signed_margin': signed_margin,
            'mutant_class': mutant_class,
            'wt_class': wt_class
        })

    return auroc, predictions


def predictive_signal_test(
    df_binned: pd.DataFrame,
    group_col: str = "genotype",
    time_col: str = "time_bin",
    z_cols: Optional[list] = None,
    n_splits: int = 5,
    n_perm: int = 100,
    random_state: Optional[int] = None,
    return_embryo_probs: bool = True,
    use_class_weights: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Predictive classifier + label-shuffling test across time bins.

    This test evaluates whether genotype labels can be predicted from
    morphological features (VAE embeddings) better than chance, using
    a logistic regression classifier with cross-validation.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data (output of bin_by_embryo_time).
    group_col : str, default="genotype"
        Column specifying experimental group (e.g., genotype).
    time_col : str, default="time_bin"
        Column specifying time bins.
    z_cols : list or None
        Latent columns to use as features. Auto-detected if None.
    n_splits : int, default=5
        Number of cross-validation splits.
    n_perm : int, default=100
        Number of permutations for null distribution.
    random_state : int or None
        Random seed for reproducibility.
    return_embryo_probs : bool, default=True
        If True, return per-embryo prediction probabilities in addition to aggregate stats.
    use_class_weights : bool, default=True
        If True, use balanced class weights to handle class imbalance.
        This helps prevent bias when one class (e.g., wildtype) is much larger.
        **FIXED: This parameter is now actually used in the LogisticRegression model.**

    Returns
    -------
    df_results : pd.DataFrame
        One row per time_bin with AUROC statistics and p-values.
        Columns: time_bin, AUROC_obs, AUROC_null_mean, AUROC_null_std, pval, n_samples
    df_embryo_probs : pd.DataFrame or None
        Per-embryo prediction probabilities if return_embryo_probs=True.
        Columns include: embryo_id, time_bin, true_label,
        pred_proba_<class0>, pred_proba_<class1>, predicted_label, confidence,
        support_true, signed_margin, mutant_class, wt_class

    Notes
    -----
    The signed margin is a key metric defined as:
        signed_margin = sign(true_label == positive_class) * (pred_prob - 0.5)

    This makes it:
    - Positive when classifier correctly predicts the positive class
    - Negative when classifier incorrectly predicts the negative class
    - Zero at the decision boundary
    """
    rng = np.random.default_rng(random_state)

    # Auto-detect latent columns if not specified
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
        if not z_cols:
            raise ValueError("No latent columns found. Specify z_cols explicitly.")

    results = []
    embryo_predictions = [] if return_embryo_probs else None

    # Process each time bin independently
    for t, sub in df_binned.groupby(time_col):
        X = sub[z_cols].values
        y = sub[group_col].values
        embryo_ids = sub['embryo_id'].values

        # Only handle two-class problems for now
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            continue

        # Check for minimum sample size
        min_samples_per_class = min([np.sum(y == c) for c in unique_classes])
        if min_samples_per_class < n_splits:
            print(f"Skipping time bin {t}: insufficient samples "
                  f"({min_samples_per_class} < {n_splits})")
            continue

        # --- Configure class weights ---
        # FIXED: Actually use the use_class_weights parameter!
        if use_class_weights:
            class_weight = 'balanced'
        else:
            class_weight = None

        # --- True AUROC via cross-validation ---
        skf = StratifiedKFold(
            n_splits=min(n_splits, min_samples_per_class),
            shuffle=True,
            random_state=random_state
        )

        base_model = LogisticRegression(
            max_iter=200,
            random_state=random_state,
            class_weight=class_weight
        )

        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            fold_auc, fold_predictions = _run_single_cv_fold(
                base_model,
                X_train=X[train_idx],
                y_train=y[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                embryo_ids_test=embryo_ids[test_idx],
                time_bin=t,
                return_predictions=return_embryo_probs
            )
            aucs.append(fold_auc)

            if return_embryo_probs and fold_predictions:
                embryo_predictions.extend(fold_predictions)

        true_auc = np.mean(aucs)

        # --- Null distribution via shuffled labels ---
        null_aucs = []
        for _ in range(n_perm):
            y_shuff = rng.permutation(y)
            perm_aucs = []

            for train_idx, test_idx in skf.split(X, y_shuff):
                perm_auc, _ = _run_single_cv_fold(
                    base_model,
                    X_train=X[train_idx],
                    y_train=y_shuff[train_idx],
                    X_test=X[test_idx],
                    y_test=y_shuff[test_idx],
                    embryo_ids_test=None,
                    time_bin=t,
                    return_predictions=False
                )
                perm_aucs.append(perm_auc)

            null_aucs.append(np.mean(perm_aucs))

        null_aucs = np.array(null_aucs)
        pval = (np.sum(null_aucs >= true_auc) + 1) / (len(null_aucs) + 1)

        results.append({
            "time_bin": t,
            "AUROC_obs": true_auc,
            "AUROC_null_mean": null_aucs.mean(),
            "AUROC_null_std": null_aucs.std(),
            "pval": pval,
            "n_samples": len(y)
        })

    df_results = pd.DataFrame(results)

    if return_embryo_probs:
        df_embryo_probs = pd.DataFrame(embryo_predictions)
        return df_results, df_embryo_probs

    return df_results, None
