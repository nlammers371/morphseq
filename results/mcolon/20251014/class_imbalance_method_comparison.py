"""
Class Imbalance Method Comparison for Genotype Classification

This script systematically tests different methods for handling class imbalance
in logistic regression models, comparing their effectiveness at reducing bias
toward over-represented classes.

Methods tested:
1. Baseline - No adjustment
2. Class Weights - balanced class weights
3. Embryo-Equal Weights - each embryo contributes equally
4. Combined - embryo weights × class weights
5. Threshold Moving - optimize decision threshold
6. Calibration - isotonic/sigmoid calibration
7. Balanced Bootstrap - resample with equal class representation
"""

import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014"
data_dir_base = os.path.join(results_dir, "imbalance_methods", "data")
plot_dir_base = os.path.join(results_dir, "imbalance_methods", "plots")

print(f"Results directory: {results_dir}")
os.makedirs(plot_dir_base, exist_ok=True)
os.makedirs(data_dir_base, exist_ok=True)

morphseq_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"
print(f"MORPHSEQ_REPO_ROOT: {morphseq_root}")

import sys
sys.path.insert(0, morphseq_root)

# Analysis parameters
N_PERMUTATIONS = int(os.environ.get("MORPHSEQ_N_PERMUTATIONS", 100))
N_CV_SPLITS = 5
RANDOM_SEED = 42
ALPHA = 0.05

print(f"\nConfiguration:")
print(f"  Permutations: {N_PERMUTATIONS}")
print(f"  CV splits: {N_CV_SPLITS}")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Alpha: {ALPHA}")

# ============================================================================
# BINNING FUNCTION (from original script)
# ============================================================================

def bin_by_embryo_time(
    df,
    time_col="predicted_stage_hpf",
    z_cols=None,
    bin_width=2.0,
    suffix="_binned"
):
    """Bin VAE embeddings by predicted time and embryo."""
    df = df.copy()

    if z_cols is None:
        z_cols = [c for c in df.columns if "z_mu_b" in c]
        if not z_cols:
            raise ValueError("No latent columns found matching pattern 'z_mu_b'.")

    df["time_bin"] = (np.floor(df[time_col] / bin_width) * bin_width).astype(int)

    agg = (
        df.groupby(["embryo_id", "time_bin"], as_index=False)[z_cols]
        .mean()
    )

    agg.rename(columns={c: f"{c}{suffix}" for c in z_cols}, inplace=True)

    meta_cols = [c for c in df.columns if c not in z_cols + [time_col, "time_bin"]]
    meta_df = (
        df[meta_cols]
        .drop_duplicates(subset=["embryo_id"])
    )

    out = agg.merge(meta_df, on="embryo_id", how="left")
    out = out.sort_values(["embryo_id", "time_bin"]).reset_index(drop=True)

    return out


# ============================================================================
# HELPER FUNCTIONS FOR WEIGHTING
# ============================================================================

def compute_embryo_weights(embryo_ids, y):
    """
    Compute sample weights so each embryo contributes equally.

    Parameters
    ----------
    embryo_ids : array-like
        Embryo identifier for each sample
    y : array-like
        Class labels (not used, but kept for API consistency)

    Returns
    -------
    weights : np.ndarray
        Sample weights normalized so each embryo has equal total weight
    """
    unique_embryos, counts = np.unique(embryo_ids, return_counts=True)
    n_embryos = len(unique_embryos)

    # Each embryo should contribute 1/n_embryos total weight
    embryo_weight_map = {embryo: 1.0 / (n_embryos * count)
                         for embryo, count in zip(unique_embryos, counts)}

    weights = np.array([embryo_weight_map[eid] for eid in embryo_ids])

    # Normalize to sum to n_samples for sklearn compatibility
    weights = weights * len(weights) / weights.sum()

    return weights


def compute_combined_weights(embryo_ids, y):
    """
    Combine embryo-level weighting with class balancing.

    Each embryo contributes equally, AND minority class is upweighted.
    """
    # Get embryo weights
    embryo_weights = compute_embryo_weights(embryo_ids, y)

    # Get class weights
    classes = np.unique(y)
    class_weights_dict = dict(zip(
        classes,
        compute_class_weight('balanced', classes=classes, y=y)
    ))
    class_weights = np.array([class_weights_dict[label] for label in y])

    # Combine multiplicatively
    combined = embryo_weights * class_weights

    # Normalize
    combined = combined * len(combined) / combined.sum()

    return combined


def optimal_threshold_from_roc(y_true, y_pred_proba):
    """
    Find optimal decision threshold that maximizes balanced accuracy.

    Returns
    -------
    threshold : float
        Optimal decision threshold
    balanced_acc : float
        Balanced accuracy at optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Balanced accuracy = (TPR + TNR) / 2 = (TPR + (1 - FPR)) / 2
    balanced_accuracies = (tpr + (1 - fpr)) / 2

    optimal_idx = np.argmax(balanced_accuracies)
    optimal_threshold = thresholds[optimal_idx]
    optimal_ba = balanced_accuracies[optimal_idx]

    return optimal_threshold, optimal_ba


# ============================================================================
# CLASSIFICATION METHODS
# ============================================================================

class ClassificationMethod:
    """Base class for classification methods with different imbalance handling."""

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def fit_predict(self, X_train, y_train, X_test, y_test,
                    embryo_ids_train=None, embryo_ids_test=None,
                    random_state=None):
        """
        Fit model and return predictions.

        Returns
        -------
        dict with keys:
            - pred_proba: predicted probabilities
            - pred_label: predicted labels (at 0.5 threshold)
            - optimal_threshold: optimal threshold from training set
            - optimal_pred_label: predictions at optimal threshold
            - model: fitted model (for inspection)
        """
        raise NotImplementedError


class BaselineMethod(ClassificationMethod):
    """Logistic regression with no adjustments."""

    def __init__(self):
        super().__init__(
            name="baseline",
            description="No imbalance correction"
        )

    def fit_predict(self, X_train, y_train, X_test, y_test,
                    embryo_ids_train=None, embryo_ids_test=None,
                    random_state=None):
        model = LogisticRegression(max_iter=200, random_state=random_state)
        model.fit(X_train, y_train)

        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_label = (pred_proba > 0.5).astype(int)

        # Find optimal threshold on training set
        train_proba = model.predict_proba(X_train)[:, 1]
        y_train_binary = (y_train == model.classes_[1]).astype(int)
        optimal_thresh, _ = optimal_threshold_from_roc(y_train_binary, train_proba)
        optimal_pred = (pred_proba > optimal_thresh).astype(int)

        return {
            'pred_proba': pred_proba,
            'pred_label': pred_label,
            'optimal_threshold': optimal_thresh,
            'optimal_pred_label': optimal_pred,
            'model': model
        }


class ClassWeightMethod(ClassificationMethod):
    """Logistic regression with balanced class weights."""

    def __init__(self):
        super().__init__(
            name="class_weight",
            description="Balanced class weights"
        )

    def fit_predict(self, X_train, y_train, X_test, y_test,
                    embryo_ids_train=None, embryo_ids_test=None,
                    random_state=None):
        model = LogisticRegression(
            max_iter=200,
            class_weight='balanced',
            random_state=random_state
        )
        model.fit(X_train, y_train)

        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_label = (pred_proba > 0.5).astype(int)

        train_proba = model.predict_proba(X_train)[:, 1]
        y_train_binary = (y_train == model.classes_[1]).astype(int)
        optimal_thresh, _ = optimal_threshold_from_roc(y_train_binary, train_proba)
        optimal_pred = (pred_proba > optimal_thresh).astype(int)

        return {
            'pred_proba': pred_proba,
            'pred_label': pred_label,
            'optimal_threshold': optimal_thresh,
            'optimal_pred_label': optimal_pred,
            'model': model
        }


class EmbryoWeightMethod(ClassificationMethod):
    """Logistic regression with embryo-equal sample weights."""

    def __init__(self):
        super().__init__(
            name="embryo_weight",
            description="Embryo-equal weighting"
        )

    def fit_predict(self, X_train, y_train, X_test, y_test,
                    embryo_ids_train=None, embryo_ids_test=None,
                    random_state=None):
        if embryo_ids_train is None:
            raise ValueError("embryo_ids_train required for EmbryoWeightMethod")

        sample_weights = compute_embryo_weights(embryo_ids_train, y_train)

        model = LogisticRegression(max_iter=200, random_state=random_state)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_label = (pred_proba > 0.5).astype(int)

        train_proba = model.predict_proba(X_train)[:, 1]
        y_train_binary = (y_train == model.classes_[1]).astype(int)
        optimal_thresh, _ = optimal_threshold_from_roc(y_train_binary, train_proba)
        optimal_pred = (pred_proba > optimal_thresh).astype(int)

        return {
            'pred_proba': pred_proba,
            'pred_label': pred_label,
            'optimal_threshold': optimal_thresh,
            'optimal_pred_label': optimal_pred,
            'model': model
        }


class CombinedWeightMethod(ClassificationMethod):
    """Logistic regression with embryo weights × class weights."""

    def __init__(self):
        super().__init__(
            name="combined_weight",
            description="Embryo-equal × class weights"
        )

    def fit_predict(self, X_train, y_train, X_test, y_test,
                    embryo_ids_train=None, embryo_ids_test=None,
                    random_state=None):
        if embryo_ids_train is None:
            raise ValueError("embryo_ids_train required for CombinedWeightMethod")

        sample_weights = compute_combined_weights(embryo_ids_train, y_train)

        model = LogisticRegression(max_iter=200, random_state=random_state)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_label = (pred_proba > 0.5).astype(int)

        train_proba = model.predict_proba(X_train)[:, 1]
        y_train_binary = (y_train == model.classes_[1]).astype(int)
        optimal_thresh, _ = optimal_threshold_from_roc(y_train_binary, train_proba)
        optimal_pred = (pred_proba > optimal_thresh).astype(int)

        return {
            'pred_proba': pred_proba,
            'pred_label': pred_label,
            'optimal_threshold': optimal_thresh,
            'optimal_pred_label': optimal_pred,
            'model': model
        }


class CalibratedMethod(ClassificationMethod):
    """Logistic regression with probability calibration (isotonic)."""

    def __init__(self, base_method='class_weight'):
        super().__init__(
            name=f"calibrated_{base_method}",
            description=f"Calibrated ({base_method})"
        )
        self.base_method = base_method

    def fit_predict(self, X_train, y_train, X_test, y_test,
                    embryo_ids_train=None, embryo_ids_test=None,
                    random_state=None):

        # Build base model based on method
        if self.base_method == 'class_weight':
            base_model = LogisticRegression(
                max_iter=200,
                class_weight='balanced',
                random_state=random_state
            )
            sample_weights = None
        elif self.base_method == 'embryo_weight':
            if embryo_ids_train is None:
                raise ValueError("embryo_ids_train required")
            base_model = LogisticRegression(max_iter=200, random_state=random_state)
            sample_weights = compute_embryo_weights(embryo_ids_train, y_train)
        elif self.base_method == 'combined_weight':
            if embryo_ids_train is None:
                raise ValueError("embryo_ids_train required")
            base_model = LogisticRegression(max_iter=200, random_state=random_state)
            sample_weights = compute_combined_weights(embryo_ids_train, y_train)
        else:  # baseline
            base_model = LogisticRegression(max_iter=200, random_state=random_state)
            sample_weights = None

        # Wrap with calibration
        # Note: CalibratedClassifierCV does its own CV, so we fit on train set
        model = CalibratedClassifierCV(
            base_model,
            method='isotonic',
            cv=min(3, N_CV_SPLITS)  # Use fewer splits for speed
        )

        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_label = (pred_proba > 0.5).astype(int)

        train_proba = model.predict_proba(X_train)[:, 1]
        y_train_binary = (y_train == model.classes_[1]).astype(int)
        optimal_thresh, _ = optimal_threshold_from_roc(y_train_binary, train_proba)
        optimal_pred = (pred_proba > optimal_thresh).astype(int)

        return {
            'pred_proba': pred_proba,
            'pred_label': pred_label,
            'optimal_threshold': optimal_thresh,
            'optimal_pred_label': optimal_pred,
            'model': model
        }


class BalancedBootstrapMethod(ClassificationMethod):
    """Logistic regression with balanced bootstrapping."""

    def __init__(self):
        super().__init__(
            name="balanced_bootstrap",
            description="Balanced bootstrap resampling"
        )

    def fit_predict(self, X_train, y_train, X_test, y_test,
                    embryo_ids_train=None, embryo_ids_test=None,
                    random_state=None):
        rng = np.random.default_rng(random_state)

        # Balance classes by upsampling minority
        classes, counts = np.unique(y_train, return_counts=True)
        max_count = counts.max()

        balanced_indices = []
        for cls in classes:
            cls_indices = np.where(y_train == cls)[0]
            # Resample with replacement to match max_count
            resampled = rng.choice(cls_indices, size=max_count, replace=True)
            balanced_indices.extend(resampled)

        balanced_indices = np.array(balanced_indices)
        rng.shuffle(balanced_indices)

        X_train_balanced = X_train[balanced_indices]
        y_train_balanced = y_train[balanced_indices]

        model = LogisticRegression(max_iter=200, random_state=random_state)
        model.fit(X_train_balanced, y_train_balanced)

        pred_proba = model.predict_proba(X_test)[:, 1]
        pred_label = (pred_proba > 0.5).astype(int)

        train_proba = model.predict_proba(X_train)[:, 1]  # Use original train set
        y_train_binary = (y_train == model.classes_[1]).astype(int)
        optimal_thresh, _ = optimal_threshold_from_roc(y_train_binary, train_proba)
        optimal_pred = (pred_proba > optimal_thresh).astype(int)

        return {
            'pred_proba': pred_proba,
            'pred_label': pred_label,
            'optimal_threshold': optimal_thresh,
            'optimal_pred_label': optimal_pred,
            'model': model
        }


# Initialize all methods
METHODS = [
    BaselineMethod(),
    ClassWeightMethod(),
    EmbryoWeightMethod(),
    CombinedWeightMethod(),
    CalibratedMethod(base_method='class_weight'),
    CalibratedMethod(base_method='combined_weight'),
    BalancedBootstrapMethod()
]

print(f"\nMethods to test: {len(METHODS)}")
for method in METHODS:
    print(f"  - {method.name}: {method.description}")


# ============================================================================
# MULTI-METHOD PREDICTIVE SIGNAL TEST
# ============================================================================

def predictive_signal_test_multimethod(
    df_binned,
    methods,
    group_col="genotype",
    time_col="time_bin",
    z_cols=None,
    n_splits=5,
    n_perm=100,
    random_state=None,
    return_embryo_probs=True
):
    """
    Run predictive signal test with multiple imbalance-handling methods.

    Returns
    -------
    results_by_method : dict
        Keys are method names, values are dicts with:
        - df_results: time-bin level metrics
        - df_embryo_probs: per-embryo predictions (if return_embryo_probs=True)
    """
    rng = np.random.default_rng(random_state)
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]

    # Initialize storage for each method
    results_by_method = {method.name: {'results': [], 'embryo_preds': []}
                         for method in methods}

    for t, sub in df_binned.groupby(time_col):
        X = sub[z_cols].values
        y = sub[group_col].values
        embryo_ids = sub['embryo_id'].values

        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            continue

        min_samples_per_class = min([np.sum(y == c) for c in unique_classes])
        if min_samples_per_class < n_splits:
            print(f"  Skipping time bin {t}: insufficient samples ({min_samples_per_class} < {n_splits})")
            continue

        # Run cross-validation for each method
        skf = StratifiedKFold(n_splits=min(n_splits, min_samples_per_class),
                             shuffle=True, random_state=random_state)

        for method in methods:
            try:
                method_aucs = []
                method_ba = []  # balanced accuracy
                method_prauc = []  # PR-AUC
                method_brier = []  # Brier score

                for train_idx, test_idx in skf.split(X, y):
                    # Get embryo IDs for train/test
                    embryo_train = embryo_ids[train_idx]
                    embryo_test = embryo_ids[test_idx]

                    # Fit and predict
                    result = method.fit_predict(
                        X[train_idx], y[train_idx],
                        X[test_idx], y[test_idx],
                        embryo_ids_train=embryo_train,
                        embryo_ids_test=embryo_test,
                        random_state=random_state
                    )

                    pred_proba = result['pred_proba']
                    pred_label = result['pred_label']
                    optimal_pred = result['optimal_pred_label']

                    # Convert labels to binary (positive class = 1)
                    class_order = result['model'].classes_
                    positive_class = class_order[1]
                    y_test_binary = (y[test_idx] == positive_class).astype(int)

                    # Compute metrics
                    method_aucs.append(roc_auc_score(y_test_binary, pred_proba))
                    method_ba.append(balanced_accuracy_score(y_test_binary, optimal_pred))
                    method_prauc.append(average_precision_score(y_test_binary, pred_proba))
                    method_brier.append(brier_score_loss(y_test_binary, pred_proba))

                    # Store embryo-level predictions
                    if return_embryo_probs:
                        for i, idx in enumerate(test_idx):
                            true_label = y[idx]
                            p_pos = pred_proba[i]
                            support_true = p_pos if true_label == positive_class else 1.0 - p_pos
                            signed_margin = (1 if true_label == positive_class else -1) * (p_pos - 0.5)

                            results_by_method[method.name]['embryo_preds'].append({
                                'embryo_id': embryo_ids[idx],
                                'time_bin': t,
                                'true_label': true_label,
                                'pred_proba': p_pos,
                                'confidence': np.abs(p_pos - 0.5),
                                'predicted_label': positive_class if p_pos > 0.5 else class_order[0],
                                'optimal_threshold': result['optimal_threshold'],
                                'optimal_pred_label': positive_class if optimal_pred[i] > 0.5 else class_order[0],
                                'support_true': support_true,
                                'signed_margin': signed_margin
                            })

                # Store aggregate metrics
                results_by_method[method.name]['results'].append({
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
                print(f"  Error with method {method.name} at time {t}: {e}")
                continue

    # Convert lists to DataFrames
    output = {}
    for method in methods:
        df_results = pd.DataFrame(results_by_method[method.name]['results'])
        df_embryo_probs = None
        if return_embryo_probs and results_by_method[method.name]['embryo_preds']:
            df_embryo_probs = pd.DataFrame(results_by_method[method.name]['embryo_preds'])

        output[method.name] = {
            'df_results': df_results,
            'df_embryo_probs': df_embryo_probs
        }

    return output


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_method_comparison_auroc(results_by_method, group1, group2, output_path=None):
    """Compare AUROC across methods over time."""
    fig, ax = plt.subplots(figsize=(14, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_method)))

    for idx, (method_name, data) in enumerate(results_by_method.items()):
        df_results = data['df_results']
        if df_results.empty:
            continue

        ax.plot(df_results['time_bin'], df_results['AUROC_mean'],
               label=method_name, color=colors[idx], linewidth=2, marker='o', markersize=5)
        ax.fill_between(
            df_results['time_bin'],
            df_results['AUROC_mean'] - df_results['AUROC_std'],
            df_results['AUROC_mean'] + df_results['AUROC_std'],
            color=colors[idx], alpha=0.2
        )

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Chance')
    ax.set_xlabel('Time Bin (hpf)', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=13, fontweight='bold')
    ax.set_title(f'Method Comparison: AUROC over Time\n{group1} vs {group2}',
                fontsize=15, fontweight='bold')
    ax.set_ylim([0.4, 1.0])
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_method_comparison_metrics(results_by_method, group1, group2, output_path=None):
    """Compare multiple metrics across methods."""
    metrics = [
        ('AUROC_mean', 'AUROC', (0.4, 1.0)),
        ('balanced_accuracy_mean', 'Balanced Accuracy', (0.4, 1.0)),
        ('PR_AUC_mean', 'PR-AUC', (0.0, 1.0)),
        ('brier_score_mean', 'Brier Score (lower=better)', (0.0, 0.5))
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_method)))

    for ax_idx, (metric_col, metric_name, ylim) in enumerate(metrics):
        ax = axes[ax_idx]

        for method_idx, (method_name, data) in enumerate(results_by_method.items()):
            df_results = data['df_results']
            if df_results.empty or metric_col not in df_results.columns:
                continue

            ax.plot(df_results['time_bin'], df_results[metric_col],
                   label=method_name, color=colors[method_idx],
                   linewidth=2, marker='o', markersize=4)

        if metric_col == 'AUROC_mean' or metric_col == 'balanced_accuracy_mean':
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Time Bin (hpf)', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylim(ylim)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)

    fig.suptitle(f'Method Comparison: Multiple Metrics\n{group1} vs {group2}',
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_probability_calibration(results_by_method, group1, group2, time_bin=None, output_path=None):
    """Plot calibration curves showing predicted vs actual probabilities."""

    # Pick a time bin to analyze (middle of time range, or user-specified)
    all_time_bins = set()
    for data in results_by_method.values():
        if data['df_embryo_probs'] is not None:
            all_time_bins.update(data['df_embryo_probs']['time_bin'].unique())

    if not all_time_bins:
        print("  No embryo probability data available for calibration plot")
        return None

    if time_bin is None:
        time_bin = sorted(all_time_bins)[len(all_time_bins) // 2]

    n_methods = len([d for d in results_by_method.values() if d['df_embryo_probs'] is not None])
    if n_methods == 0:
        return None

    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(7 * ((n_methods + 1) // 2), 10))
    axes = np.atleast_1d(axes).flatten()

    ax_idx = 0
    for method_name, data in results_by_method.items():
        if data['df_embryo_probs'] is None or data['df_embryo_probs'].empty:
            continue

        df_probs = data['df_embryo_probs']
        df_time = df_probs[df_probs['time_bin'] == time_bin]

        if df_time.empty:
            continue

        ax = axes[ax_idx]

        # Get true labels and predictions
        # Assuming positive class is the second genotype
        positive_class = df_time['true_label'].unique()[1] if len(df_time['true_label'].unique()) > 1 else df_time['true_label'].unique()[0]
        y_true = (df_time['true_label'] == positive_class).astype(int)
        y_pred_proba = df_time['pred_proba'].values

        # Bin predictions and compute observed frequency
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        bin_true_freqs = []
        bin_pred_means = []
        bin_counts = []

        for i in range(n_bins):
            mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i+1])
            if i == n_bins - 1:  # Include right edge in last bin
                mask = (y_pred_proba >= bins[i]) & (y_pred_proba <= bins[i+1])

            if mask.sum() > 0:
                bin_true_freqs.append(y_true[mask].mean())
                bin_pred_means.append(y_pred_proba[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_true_freqs.append(np.nan)
                bin_pred_means.append(np.nan)
                bin_counts.append(0)

        # Plot calibration curve
        valid = ~np.isnan(bin_true_freqs)
        ax.plot(np.array(bin_pred_means)[valid], np.array(bin_true_freqs)[valid],
               marker='o', markersize=8, linewidth=2, label=method_name)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Perfect calibration')

        # Compute mean absolute calibration error
        mace = np.nanmean(np.abs(np.array(bin_true_freqs) - np.array(bin_pred_means)))

        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Frequency', fontsize=11)
        ax.set_title(f'{method_name}\nMACE={mace:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        ax_idx += 1

    # Hide unused axes
    for i in range(ax_idx, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Probability Calibration at Time Bin {time_bin} hpf\n{group1} vs {group2}',
                fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_probability_shift_analysis(results_by_method, group1, group2, output_path=None):
    """
    Analyze systematic probability shifts by comparing mean predicted probabilities
    for each true class across methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_method)))

    # Panel 1: Mean predicted probability by true class over time
    ax1 = axes[0]
    ax2 = axes[1]

    for method_idx, (method_name, data) in enumerate(results_by_method.items()):
        if data['df_embryo_probs'] is None or data['df_embryo_probs'].empty:
            continue

        df_probs = data['df_embryo_probs']

        # Get unique classes
        classes = sorted(df_probs['true_label'].unique())
        if len(classes) != 2:
            continue

        # For each class, compute mean predicted probability over time
        for cls_idx, cls in enumerate(classes):
            df_cls = df_probs[df_probs['true_label'] == cls]

            mean_probs = df_cls.groupby('time_bin')['pred_proba'].mean()

            ax = ax1 if cls_idx == 0 else ax2
            ax.plot(mean_probs.index, mean_probs.values,
                   label=method_name, color=colors[method_idx],
                   linewidth=2, marker='o', markersize=4)

    ax1.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision boundary')
    ax1.set_xlabel('Time Bin (hpf)', fontsize=12)
    ax1.set_ylabel('Mean Predicted Probability', fontsize=12)
    ax1.set_title(f'True Class: {classes[0]}', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(alpha=0.3)

    ax2.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision boundary')
    ax2.set_xlabel('Time Bin (hpf)', fontsize=12)
    ax2.set_ylabel('Mean Predicted Probability', fontsize=12)
    ax2.set_title(f'True Class: {classes[1]}', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(alpha=0.3)

    fig.suptitle(f'Probability Shift Analysis: Mean Predictions by True Class\n{group1} vs {group2}',
                fontsize=15, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

# Import experiment definitions
WT_experiments = ["20230615","20230531", "20230525", "20250912"]
b9d2_experiments = ["20250519","20250520"]
cep290_experiments = ["20250305", "20250416", "20250512", "20250515_part2", "20250519"]
tmem67_experiments = ["20250711"]

experiments = WT_experiments + b9d2_experiments + cep290_experiments + tmem67_experiments

build06_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output"

# Load all experiments
dfs = []
for exp in experiments:
    try:
        file_path = f"{build06_dir}/df03_final_output_with_latents_{exp}.csv"
        df = pd.read_csv(file_path)
        df['source_experiment'] = exp
        dfs.append(df)
        print(f"Loaded {exp}: {len(df)} rows")
    except:
        print(f"Missing: {exp}")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(combined_df)} rows from {len(dfs)} experiments")

GENOTYPE_GROUPS = {
    "cep290": ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous'],
    "b9d2": ['b9d2_wildtype', 'b9d2_heterozygous', 'b9d2_homozygous'],
    "tmem67": ['tmem67_wildtype', 'tmem67_heterozygote', 'tmem67_homozygous'],
}

print("\n" + "="*80)
print("CLASS IMBALANCE METHOD COMPARISON ANALYSIS")
print("="*80)

for genotype_label, genotype_values in GENOTYPE_GROUPS.items():
    print("\n" + "="*80)
    print(f"ANALYSIS FOR {genotype_label.upper()}")
    print("="*80)

    # Create output directories
    data_dir = os.path.join(data_dir_base, genotype_label)
    plot_dir = os.path.join(plot_dir_base, genotype_label)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Filter to genotype family
    df_family = combined_df[combined_df['genotype'].isin(genotype_values)].copy()
    if df_family.empty:
        print(f"No data found for genotype group '{genotype_label}', skipping.")
        continue

    print(f"\nFiltered to {genotype_label} genotypes: {len(df_family)} rows")
    print(f"Genotype distribution:\n{df_family['genotype'].value_counts()}")

    # Bin embeddings
    print("\nBinning embeddings by embryo and time...")
    df_binned = bin_by_embryo_time(df_family, time_col="predicted_stage_hpf")
    print(f"Binned data: {len(df_binned)} rows")

    # Drop NaNs
    binned_z_cols = [c for c in df_binned.columns if "_binned" in c]
    df_binned = df_binned.dropna(subset=binned_z_cols)
    print(f"After dropping NaNs: {len(df_binned)} rows")

    # Get present genotypes
    present_genotypes = [g for g in genotype_values if g in df_binned['genotype'].unique()]
    print(f"Present genotypes: {present_genotypes}")

    # Run pairwise comparisons
    pairwise_comparisons = list(combinations(present_genotypes, 2))
    print(f"Running {len(pairwise_comparisons)} pairwise comparisons...")

    for idx, (group1, group2) in enumerate(pairwise_comparisons, 1):
        print(f"\n[{idx}/{len(pairwise_comparisons)}] Comparing: {group1} vs {group2}")

        # Filter to just these two genotypes
        df_pair = df_binned[df_binned['genotype'].isin([group1, group2])].copy()

        if len(df_pair) < 10:
            print(f"  Skipping: insufficient data ({len(df_pair)} samples)")
            continue

        print(f"  Class distribution: {df_pair['genotype'].value_counts().to_dict()}")

        # Run multi-method test
        print(f"  Running multi-method classification...")
        results_by_method = predictive_signal_test_multimethod(
            df_pair,
            methods=METHODS,
            group_col="genotype",
            n_splits=N_CV_SPLITS,
            n_perm=N_PERMUTATIONS,
            random_state=RANDOM_SEED,
            return_embryo_probs=True
        )

        # Check if any method succeeded
        if not any(data['df_results'] is not None and not data['df_results'].empty
                  for data in results_by_method.values()):
            print(f"  No valid results from any method")
            continue

        # Create safe filename
        safe_comp_name = f"{group1.split('_')[-1]}_vs_{group2.split('_')[-1]}"

        # Save results for each method
        for method_name, data in results_by_method.items():
            if data['df_results'] is not None and not data['df_results'].empty:
                results_path = os.path.join(data_dir, f'results_{method_name}_{safe_comp_name}.csv')
                data['df_results'].to_csv(results_path, index=False)

            if data['df_embryo_probs'] is not None and not data['df_embryo_probs'].empty:
                probs_path = os.path.join(data_dir, f'embryo_probs_{method_name}_{safe_comp_name}.csv')
                data['df_embryo_probs'].to_csv(probs_path, index=False)

        # Generate comparison plots
        print(f"  Generating comparison plots...")

        plot_method_comparison_auroc(
            results_by_method,
            group1,
            group2,
            output_path=os.path.join(plot_dir, f'method_comparison_auroc_{safe_comp_name}.png')
        )

        plot_method_comparison_metrics(
            results_by_method,
            group1,
            group2,
            output_path=os.path.join(plot_dir, f'method_comparison_metrics_{safe_comp_name}.png')
        )

        plot_probability_calibration(
            results_by_method,
            group1,
            group2,
            output_path=os.path.join(plot_dir, f'probability_calibration_{safe_comp_name}.png')
        )

        plot_probability_shift_analysis(
            results_by_method,
            group1,
            group2,
            output_path=os.path.join(plot_dir, f'probability_shift_analysis_{safe_comp_name}.png')
        )

        # Print summary statistics
        print(f"\n  Method Performance Summary:")
        print(f"  {'Method':<25} {'Mean AUROC':<12} {'Mean Bal.Acc':<15} {'Mean PR-AUC':<12}")
        print(f"  {'-'*70}")

        for method_name, data in results_by_method.items():
            if data['df_results'] is not None and not data['df_results'].empty:
                df_res = data['df_results']
                mean_auroc = df_res['AUROC_mean'].mean()
                mean_ba = df_res['balanced_accuracy_mean'].mean() if 'balanced_accuracy_mean' in df_res.columns else np.nan
                mean_prauc = df_res['PR_AUC_mean'].mean() if 'PR_AUC_mean' in df_res.columns else np.nan

                print(f"  {method_name:<25} {mean_auroc:<12.3f} {mean_ba:<15.3f} {mean_prauc:<12.3f}")

print("\n" + "="*80)
print("CLASS IMBALANCE METHOD COMPARISON COMPLETE")
print("="*80)
print(f"\nResults saved to: {data_dir_base}")
print(f"Plots saved to: {plot_dir_base}")
