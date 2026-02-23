"""
Binary classification tests for difference detection.

Provides group-label assignment plus time-resolved AUROC permutation testing.

Key functions:
- assign_group_labels(): Add group labels to a DataFrame (manual or from k_results)
- run_binary_classification_test(): AUROC-based comparison between two groups
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

try:
    from joblib import Parallel, delayed
except Exception:  # joblib is optional
    Parallel = None
    delayed = None


def assign_group_labels(
    df: pd.DataFrame,
    # Mode 1: Manual group assignment
    groups: Optional[Dict[str, List[str]]] = None,
    # Mode 2: From k_results
    k_results: Optional[Dict] = None,
    k: Optional[int] = None,
    cluster_names: Optional[Dict[int, str]] = None,
    membership: Optional[str] = None,
    # Common params
    group_col: str = 'group',
    embryo_id_col: str = 'embryo_id',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Assign group labels to a DataFrame based on embryo ID membership.

    Two modes of operation:
    1. Manual: Pass groups={'CE': [ids], 'WT': [ids]}
    2. From k_results: Pass k_results, k, and cluster_names dict

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame with embryo_id column

    # Mode 1: Manual
    groups : Dict[str, List[str]], optional
        Mapping of group_name -> list of embryo IDs
        Example: {'CE': ['embryo1', 'embryo2'], 'WT': ['embryo3', 'embryo4']}

    # Mode 2: From k_results (integrates with k-selection pipeline)
    k_results : Dict, optional
        Output from run_k_selection_with_plots()
    k : int, optional
        Which k value to use
    cluster_names : Dict[int, str], optional
        Mapping cluster_id -> group name
        Example: {0: 'wildtype_like', 1: 'phenotype', 2: 'severe'}
    membership : str, optional
        Filter by membership quality: 'core', 'uncertain', or None for all

    # Common
    group_col : str
        Name for the new group column (default: 'group')
    embryo_id_col : str
        Name of embryo ID column (default: 'embryo_id')
    inplace : bool
        If True, modify df in place; otherwise return copy

    Returns
    -------
    pd.DataFrame
        DataFrame with new group column added

    Examples
    --------
    # Mode 1: Manual
    >>> df = assign_group_labels(df_raw, groups={'CE': ce_ids, 'WT': wt_ids})

    # Mode 2: From k_results
    >>> df = assign_group_labels(df_raw, k_results=k_results, k=4,
    ...                       cluster_names={0: 'WT', 1: 'phenotype'})

    # With membership filtering
    >>> df = assign_group_labels(df_raw, k_results=k_results, k=4,
    ...                       cluster_names={0: 'WT', 1: 'phenotype'},
    ...                       membership='core')
    """
    if not inplace:
        df = df.copy()

    # Validate: exactly one mode must be specified
    manual_mode = groups is not None
    k_results_mode = k_results is not None

    if not manual_mode and not k_results_mode:
        raise ValueError(
            "Must specify either 'groups' (manual mode) or "
            "'k_results' + 'k' + 'cluster_names' (k_results mode)"
        )

    if manual_mode and k_results_mode:
        raise ValueError(
            "Cannot specify both 'groups' and 'k_results'. Choose one mode."
        )

    # Mode 1: Manual group assignment
    if manual_mode:
        # Build embryo_id -> group_name mapping
        embryo_to_group = {}
        for group_name, embryo_ids in groups.items():
            for eid in embryo_ids:
                if eid in embryo_to_group:
                    print(f"Warning: {eid} appears in multiple groups. "
                          f"Using '{group_name}' (overwrites '{embryo_to_group[eid]}')")
                embryo_to_group[eid] = group_name

        df[group_col] = df[embryo_id_col].map(embryo_to_group)

    # Mode 2: From k_results
    else:
        if k is None:
            raise ValueError("Must specify 'k' when using k_results mode")
        if cluster_names is None:
            raise ValueError("Must specify 'cluster_names' when using k_results mode")

        # Validate k exists in k_results
        if 'clustering_by_k' not in k_results:
            raise ValueError(
                "k_results must contain 'clustering_by_k'. "
                "Expected output from run_k_selection_with_plots() or evaluate_k_range()"
            )

        if k not in k_results['clustering_by_k']:
            available_k = list(k_results['clustering_by_k'].keys())
            raise ValueError(
                f"k={k} not found in k_results. Available: {available_k}"
            )

        # Get assignments for this k
        k_data = k_results['clustering_by_k'][k]
        embryo_to_cluster = k_data['assignments']['embryo_to_cluster']

        # Get membership quality if filtering requested
        if membership is not None:
            if membership not in ['core', 'uncertain', 'outlier']:
                raise ValueError(
                    f"membership must be 'core', 'uncertain', or 'outlier', got '{membership}'"
                )
            embryo_to_membership = k_data['membership']['embryo_to_membership_quality']
        else:
            embryo_to_membership = None

        # Build embryo_id -> group_name mapping
        embryo_to_group = {}
        for embryo_id, cluster_id in embryo_to_cluster.items():
            # Check membership filter
            if embryo_to_membership is not None:
                if embryo_to_membership.get(embryo_id) != membership:
                    continue  # Skip embryos not matching membership filter

            # Map cluster_id to group name
            if cluster_id in cluster_names:
                embryo_to_group[embryo_id] = cluster_names[cluster_id]
            # If cluster_id not in cluster_names, embryo won't get a group label

        df[group_col] = df[embryo_id_col].map(embryo_to_group)

        # Report filtering stats
        if membership is not None:
            n_labeled = df[group_col].notna().sum()
            n_rows = len(df)
            n_embryos = df[df[group_col].notna()][embryo_id_col].nunique()
            print(f"Added '{group_col}' column: {n_embryos} embryos "
                  f"({n_labeled}/{n_rows} rows) with membership='{membership}'")

    return df


def run_binary_classification_test(
    df: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    features: Union[str, List[str]] = 'z_mu_b',
    morphology_metric: Optional[str] = 'total_length_um',
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    bin_width: float = 4.0,
    n_splits: int = 5,
    n_permutations: int = 100,
    n_jobs: int = 1,
    min_samples_per_bin: int = 5,
    within_bin_time_stratification: bool = True,
    within_bin_time_strata_width: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a time-resolved binary classification test between two groups.

    AUROC direction is defined explicitly:
    - `group1` is treated as the positive/"phenotype" class (AUROC > 0.5 means model favors group1).
    - `group2` is treated as the negative/"reference" class.

    Parameters
    ----------
    df : DataFrame
        Raw data with group column already added (via assign_group_labels)
    group_col : str
        Column containing group labels
    group1, group2 : str
        Group names to compare (must exist in group_col).
        By convention and by implementation, `group1` is the positive/"phenotype"
        class and `group2` is the negative/"reference" class.
    features : str or List[str]
        'z_mu_b' to auto-select VAE biological features, or list of column names
    morphology_metric : str, optional
        Column for morphological divergence (e.g., 'total_length_um')
    time_col : str
        Time column (default: 'predicted_stage_hpf')
    embryo_id_col : str
        Embryo ID column (default: 'embryo_id')
    bin_width : float
        Time binning width in hours (default: 4.0)
    n_splits : int
        Number of cross-validation folds (default: 5)
    n_permutations : int
        Number of permutations for p-value estimation (default: 100)
    min_samples_per_bin : int
        Minimum samples per class per time bin (default: 5)
    within_bin_time_stratification : bool
        If True, permutation testing shuffles labels within fine time strata
        to control for within-bin age confounding. Recommended for early
        developmental stages where small age differences matter. (default: True)
    within_bin_time_strata_width : float
        Width (hours) of time strata for stratified permutation (default: 0.5)
        Only used if within_bin_time_stratification=True
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print progress (default: True)

    Returns
    -------
    results : Dict[str, Any]
        {
            'classification': DataFrame with time_bin, auroc, pvalue, n_samples
            'divergence': DataFrame with hpf, group1_mean, group2_mean, abs_diff
            'embryo_predictions': DataFrame with embryo_id, time_bin, pred_proba
            'diagnostics': DataFrame with confound checks (time skew, n_obs AUROC)
            'summary': {
                'earliest_significant_hpf': float or None,
                'max_auroc': float,
                'max_auroc_hpf': float,
                'n_significant_bins': int,
            },
            'config': {
                'group1': str, 'group2': str,
                'group_col': str,
                'features': List[str],
                'morphology_metric': str,
                'bin_width': float,
                'n_permutations': int,
                'within_bin_time_stratification': bool,
            }
        }
    """
    # Validate group column exists
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame")

    # Filter to just the two groups
    df_comparison = df[df[group_col].isin([group1, group2])].copy()

    if len(df_comparison) == 0:
        raise ValueError(
            f"No data found for groups '{group1}' or '{group2}' in column '{group_col}'"
        )

    # Get unique embryo counts
    n_group1 = df_comparison[df_comparison[group_col] == group1][embryo_id_col].nunique()
    n_group2 = df_comparison[df_comparison[group_col] == group2][embryo_id_col].nunique()

    if verbose:
        print(f"Comparing {group1} (n={n_group1}) vs {group2} (n={n_group2})")
        print("Classification weighting: class_weight='balanced' (enabled by default)")

    # Determine feature columns
    if isinstance(features, str):
        if features == 'z_mu_b':
            feature_cols = [c for c in df.columns if 'z_mu_b' in c]
            if not feature_cols:
                raise ValueError("No z_mu_b columns found. Specify features explicitly.")
        else:
            feature_cols = [features]
    else:
        feature_cols = list(features)

    if verbose:
        print(f"Using {len(feature_cols)} feature columns")

    # Create time bins
    df_comparison['time_bin'] = (
        np.floor(df_comparison[time_col] / bin_width) * bin_width
    ).astype(int)

    # Bin embeddings: average per embryo x time_bin
    groupby_cols = [embryo_id_col, 'time_bin', group_col]
    df_binned = df_comparison.groupby(groupby_cols, as_index=False)[feature_cols].mean()

    # Run classification
    classification_df, embryo_predictions, diagnostics_df = _run_classification(
        df_binned=df_binned,
        df_raw=df_comparison,
        group_col=group_col,
        positive_label=group1,
        negative_label=group2,
        feature_cols=feature_cols,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        bin_width=bin_width,
        n_splits=n_splits,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        min_samples_per_bin=min_samples_per_bin,
        within_bin_time_stratification=within_bin_time_stratification,
        within_bin_time_strata_width=within_bin_time_strata_width,
        random_state=random_state,
        verbose=verbose
    )

    # Compute divergence if morphology_metric specified
    if morphology_metric and morphology_metric in df.columns:
        divergence_df = compute_timeseries_divergence(
            df=df_comparison,
            group_col=group_col,
            group1=group1,
            group2=group2,
            metric_col=morphology_metric,
            time_col=time_col,
            embryo_id_col=embryo_id_col
        )
    else:
        divergence_df = None

    # Compute summary statistics
    summary = _compute_summary(classification_df)

    # Build config dict
    config = {
        'group1': group1,
        'group2': group2,
        'group_col': group_col,
        'positive_class': group1,
        'negative_class': group2,
        'features': feature_cols,
        'morphology_metric': morphology_metric,
        'bin_width': bin_width,
        'n_permutations': n_permutations,
        'n_splits': n_splits,
        'n_jobs': n_jobs,
        'n_group1': n_group1,
        'n_group2': n_group2,
        'within_bin_time_stratification': within_bin_time_stratification,
        'within_bin_time_strata_width': within_bin_time_strata_width,
    }

    return {
        'classification': classification_df,
        'divergence': divergence_df,
        'embryo_predictions': embryo_predictions,
        'diagnostics': diagnostics_df,
        'summary': summary,
        'config': config,
    }


def _run_classification(
    df_binned: pd.DataFrame,
    df_raw: pd.DataFrame,
    group_col: str,
    positive_label: str,
    negative_label: str,
    feature_cols: List[str],
    time_col: str,
    embryo_id_col: str,
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    n_jobs: int,
    min_samples_per_bin: int,
    within_bin_time_stratification: bool,
    within_bin_time_strata_width: float,
    random_state: int,
    verbose: bool
) -> tuple:
    """Run AUROC classification with permutation testing."""
    time_bins = sorted(df_binned['time_bin'].unique())
    results = []
    embryo_predictions = []

    for i, t in enumerate(time_bins):
        if verbose:
            print(f"  [{i+1}/{len(time_bins)}] Time bin {t} hpf...", end=' ', flush=True)

        sub = df_binned[df_binned['time_bin'] == t]
        X = sub[feature_cols].values
        y_labels = sub[group_col].values
        embryo_ids = sub['embryo_id'].values

        n_positive = int(np.sum(y_labels == positive_label))
        n_negative = int(np.sum(y_labels == negative_label))

        if verbose:
            print(
                f"    Class counts (positive={positive_label}: {n_positive}, "
                f"negative={negative_label}: {n_negative}); using balanced class weights"
            )

        # Check class count (must contain both positive and negative labels)
        if n_positive == 0 or n_negative == 0:
            if verbose:
                print("skipped (need both classes present)")
            continue

        # Check minimum samples per class
        min_count = min(n_positive, n_negative)
        if min_count < min_samples_per_bin:
            if verbose:
                print(f"skipped (min class has {min_count} samples, need {min_samples_per_bin})")
            continue

        # Explicitly encode labels to enforce AUROC direction:
        # 1 = positive (group1/phenotype), 0 = negative (group2/reference)
        y = (y_labels == positive_label).astype(int)

        # Set up classifier
        clf = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=random_state
        )

        # Cross-validated predictions
        n_splits_actual = min(n_splits, min_count)
        cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=random_state)

        try:
            probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
            true_auroc = roc_auc_score(y, probs[:, 1])
        except Exception as e:
            if verbose:
                print(f"error: {e}")
            continue

        # Compute time strata if stratification is enabled
        time_strata = None
        if within_bin_time_stratification:
            # Get per-embryo mean time from raw data for this bin
            sub_raw = df_raw[df_raw['time_bin'] == t]
            embryo_mean_times = (
                sub_raw.groupby(embryo_id_col)[time_col]
                .mean()
                .reindex(embryo_ids)
                .values
            )
            # Compute strata (floor to strata_width intervals)
            time_strata = np.floor(
                (embryo_mean_times - float(t)) / float(within_bin_time_strata_width)
            ).astype(int)

        # Permutation test
        def _single_perm_auc(seed: int) -> float:
            local_rng = np.random.default_rng(seed)
            
            # Stratified permutation: shuffle labels within time strata
            if time_strata is not None:
                y_perm = y.copy()
                for stratum_id in np.unique(time_strata):
                    stratum_mask = (time_strata == stratum_id)
                    if np.sum(stratum_mask) > 1:
                        y_perm[stratum_mask] = local_rng.permutation(y[stratum_mask])
            else:
                # Standard permutation: shuffle all labels randomly
                y_perm = local_rng.permutation(y)
            
            try:
                # Use StratifiedKFold on permuted labels to keep folds comparable
                probs_perm = cross_val_predict(clf, X, y_perm, cv=cv, method='predict_proba')
                return float(roc_auc_score(y_perm, probs_perm[:, 1]))
            except Exception:
                return float('nan')

        if n_permutations <= 0:
            null_aurocs = np.array([], dtype=float)
        else:
            use_parallel = (
                n_jobs is not None
                and n_jobs != 1
                and Parallel is not None
                and delayed is not None
                and n_permutations > 1
            )

            if use_parallel:
                base_seed = int(random_state) + 1000003 * (i + 1) + 10007 * int(t)
                seeds = [base_seed + j for j in range(n_permutations)]
                null_list = Parallel(n_jobs=n_jobs)(delayed(_single_perm_auc)(s) for s in seeds)
                null_aurocs = np.asarray(null_list, dtype=float)
            else:
                null_vals = []
                for j in range(n_permutations):
                    # Keep deterministic behavior even without joblib
                    seed = int(random_state) + 1000003 * (i + 1) + 10007 * int(t) + j
                    null_vals.append(_single_perm_auc(seed))
                null_aurocs = np.asarray(null_vals, dtype=float)

        null_aurocs = null_aurocs[np.isfinite(null_aurocs)]

        if len(null_aurocs) == 0:
            if verbose:
                print("skipped (no valid permutations)")
            continue

        # Compute p-value
        k = np.sum(null_aurocs >= true_auroc)
        pval = (k + 1) / (len(null_aurocs) + 1)

        results.append({
            'time_bin': t,
            'time_bin_start': float(t),
            'time_bin_end': float(t) + float(bin_width),
            'time_bin_center': float(t) + float(bin_width) / 2.0,
            'bin_width': float(bin_width),
            'auroc_observed': true_auroc,
            'auroc_null_mean': np.mean(null_aurocs),
            'auroc_null_std': np.std(null_aurocs),
            'pval': pval,
            'n_samples': len(y),
            'n_positive': n_positive,
            'n_negative': n_negative,
            'positive_class': positive_label,
            'negative_class': negative_label,
        })

        # Store embryo predictions
        for eid, prob, true_label in zip(embryo_ids, probs[:, 1], y_labels):
            embryo_predictions.append({
                'embryo_id': eid,
                'time_bin': t,
                'time_bin_start': float(t),
                'time_bin_end': float(t) + float(bin_width),
                'time_bin_center': float(t) + float(bin_width) / 2.0,
                'bin_width': float(bin_width),
                'true_label': true_label,
                'pred_proba_positive': prob,
                'positive_class': positive_label,
                'negative_class': negative_label,
                'true_is_positive': bool(true_label == positive_label),
            })

        if verbose:
            print(f"AUROC={true_auroc:.3f}, p={pval:.3f}")

    df_results = pd.DataFrame(results)
    df_embryo_probs = pd.DataFrame(embryo_predictions) if embryo_predictions else None
    df_diagnostics = None  # Placeholder for future diagnostics

    return df_results, df_embryo_probs, df_diagnostics


def compute_timeseries_divergence(
    df: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    metric_col: str,
    time_col: str,
    embryo_id_col: str
) -> pd.DataFrame:
    """
    Compute morphological divergence between groups over time.
    """
    from scipy import stats

    # Filter and add group labels
    df_filtered = df.dropna(subset=[time_col, metric_col]).copy()

    # Interpolate trajectories to common grid
    grid_step = 0.5
    time_min = np.floor(df_filtered[time_col].min() / grid_step) * grid_step
    time_max = np.ceil(df_filtered[time_col].max() / grid_step) * grid_step
    common_grid = np.arange(time_min, time_max + grid_step, grid_step)

    interpolated_records = []
    for embryo_id in df_filtered[embryo_id_col].unique():
        embryo_data = df_filtered[df_filtered[embryo_id_col] == embryo_id].sort_values(time_col)

        if len(embryo_data) < 2:
            continue

        group = embryo_data[group_col].iloc[0]
        embryo_time_min = embryo_data[time_col].min()
        embryo_time_max = embryo_data[time_col].max()

        interp_values = np.interp(
            common_grid,
            embryo_data[time_col].values,
            embryo_data[metric_col].values
        )

        for t, v in zip(common_grid, interp_values):
            if embryo_time_min <= t <= embryo_time_max:
                interpolated_records.append({
                    'embryo_id': embryo_id,
                    'hpf': t,
                    'metric_value': v,
                    'group': group
                })

    df_interp = pd.DataFrame(interpolated_records)

    # Compute stats per timepoint
    divergence_records = []
    for hpf in sorted(df_interp['hpf'].unique()):
        df_t = df_interp[df_interp['hpf'] == hpf]

        g1_values = df_t[df_t['group'] == group1]['metric_value'].values
        g2_values = df_t[df_t['group'] == group2]['metric_value'].values

        if len(g1_values) > 0 and len(g2_values) > 0:
            g1_mean = np.mean(g1_values)
            g1_sem = stats.sem(g1_values) if len(g1_values) > 1 else 0

            g2_mean = np.mean(g2_values)
            g2_sem = stats.sem(g2_values) if len(g2_values) > 1 else 0

            divergence_records.append({
                'hpf': hpf,
                'group1_mean': g1_mean,
                'group1_sem': g1_sem,
                'group2_mean': g2_mean,
                'group2_sem': g2_sem,
                'abs_difference': abs(g2_mean - g1_mean),
                'n_group1': len(g1_values),
                'n_group2': len(g2_values)
            })

    return pd.DataFrame(divergence_records)


def _compute_summary(classification_df: pd.DataFrame, alpha: float = 0.05) -> Dict[str, Any]:
    """Compute summary statistics from classification results."""
    if classification_df.empty:
        return {
            'earliest_significant_hpf': None,
            'max_auroc': None,
            'max_auroc_hpf': None,
            'n_significant_bins': 0,
        }

    # Find significant bins
    sig_mask = classification_df['pval'] < alpha
    n_significant = sig_mask.sum()

    # Earliest significant
    if n_significant > 0:
        earliest_hpf = classification_df.loc[sig_mask, 'time_bin'].min()
    else:
        earliest_hpf = None

    # Max AUROC
    max_idx = classification_df['auroc_observed'].idxmax()
    max_auroc = classification_df.loc[max_idx, 'auroc_observed']
    max_auroc_hpf = classification_df.loc[max_idx, 'time_bin']

    return {
        'earliest_significant_hpf': earliest_hpf,
        'max_auroc': max_auroc,
        'max_auroc_hpf': max_auroc_hpf,
        'n_significant_bins': n_significant,
    }


__all__ = [
    "assign_group_labels",
    "run_binary_classification_test",
    "compute_timeseries_divergence",
]
