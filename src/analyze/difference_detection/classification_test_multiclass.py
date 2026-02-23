"""
Multiclass classification tests for difference detection.

Extends binary classification to N-class comparisons using One-vs-Rest (OvR)
AUROC with permutation-based null distributions.

Key functions:
- run_multiclass_classification_test(): Run multiclass OvR AUROC-based comparison

Example
-------
>>> from morphseq.difference_detection import run_multiclass_classification_test
>>>
>>> # Define groups
>>> groups = {
...     'CE': ce_ids,
...     'HTA': hta_ids,
...     'NonPenHet': nonpen_het_ids,
...     'WT': wt_ids
... }
>>>
>>> # Run multiclass comparison
>>> results = run_multiclass_classification_test(df, groups=groups, features='z_mu_b')
>>> results['ovr_classification']['CE']  # OvR AUROC for CE vs Rest
>>> results['confusion_matrices'][20]    # Confusion matrix at 20 hpf
>>> results['embryo_predictions']        # Per-embryo prediction details
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix

from .results import ComparisonSpec, MulticlassOVRResults

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None


def _make_logistic_classifier(n_classes: int, random_state: int) -> LogisticRegression:
    """Create a stable classifier across binary/multiclass settings.

    Binary comparisons use ``liblinear`` to avoid OpenMP SHM failures seen with
    ``lbfgs`` in restricted environments. True multiclass keeps multinomial
    ``lbfgs`` behavior.
    """
    if n_classes <= 2:
        return LogisticRegression(
            max_iter=1000,
            solver='liblinear',
            multi_class='ovr',
            class_weight='balanced',
            random_state=random_state,
        )
    return LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        class_weight='balanced',
        random_state=random_state,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _resolve_comparison_groups(
    df: pd.DataFrame,
    groupby: str,
    groups: Union[str, List[str]],
    reference: Union[str, List[Union[str, Tuple[str, ...]]]],
    embryo_id_col: str = 'embryo_id',
    allow_many_comparisons: bool = False,
    max_comparisons: int = 50,
) -> List[ComparisonSpec]:
    """
    Parse flexible groups/reference inputs into structured ComparisonSpec objects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing the groupby column and embryo IDs
    groupby : str
        Column name containing group labels
    groups : str or list of str
        Which groups to test. 
        - "all": test all unique values in groupby column
        - ["CE", "HTA"]: test only these groups
    reference : str or list of str/tuple
        Reference group(s) for comparison.
        - "rest": compare each group against all others combined
        - "WT": compare each group against this single group
        - ["WT", "Het"]: compare each group against WT AND separately against Het
        - [("WT", "Het")]: compare each group against pooled WT+Het
        - ["WT", ("WT", "Het")]: both separate WT and pooled WT+Het comparisons
    embryo_id_col : str
        Column name for embryo IDs
    allow_many_comparisons : bool
        If True, allow > max_comparisons
    max_comparisons : int
        Maximum allowed comparisons before raising error (default 50)
    
    Returns
    -------
    List[ComparisonSpec]
        List of comparison specifications to run
    
    Raises
    ------
    ValueError
        If input format is invalid or too many comparisons requested
    
    Examples
    --------
    >>> # CE and HTA each vs WT
    >>> specs = _resolve_comparison_groups(
    ...     df, groupby='cluster', groups=['CE', 'HTA'], reference='WT'
    ... )
    
    >>> # All groups vs rest (classic OvR)
    >>> specs = _resolve_comparison_groups(
    ...     df, groupby='cluster', groups='all', reference='rest'
    ... )
    
    >>> # CE vs WT and CE vs pooled WT+Het
    >>> specs = _resolve_comparison_groups(
    ...     df, groupby='cluster', groups='CE', reference=['WT', ('WT', 'Het')]
    ... )
    """
    # Validate groupby column exists
    if groupby not in df.columns:
        raise ValueError(
            f"groupby column '{groupby}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Get all unique groups from data
    all_groups_in_data = df[groupby].dropna().unique().tolist()
    
    if not all_groups_in_data:
        raise ValueError(f"No non-null values found in groupby column '{groupby}'")
    
    # Parse 'groups' parameter
    if groups == "all":
        positive_groups = all_groups_in_data
    elif isinstance(groups, str):
        # Single group specified
        if groups not in all_groups_in_data:
            raise ValueError(
                f"Group '{groups}' not found in data. "
                f"Available groups: {all_groups_in_data}"
            )
        positive_groups = [groups]
    elif isinstance(groups, list):
        # List of groups
        missing = [g for g in groups if g not in all_groups_in_data]
        if missing:
            raise ValueError(
                f"Groups not found in data: {missing}. "
                f"Available groups: {all_groups_in_data}"
            )
        positive_groups = groups
    else:
        raise ValueError(
            f"'groups' must be 'all', a string, or a list of strings. "
            f"Got: {type(groups)}. "
            f"Example: groups='CE' or groups=['CE', 'HTA'] or groups='all'"
        )
    
    # Parse 'reference' parameter into list of reference items
    if reference == "rest":
        reference_items = ["rest"]
    elif isinstance(reference, str):
        # Single reference group
        if reference not in all_groups_in_data:
            raise ValueError(
                f"Reference '{reference}' not found in data. "
                f"Available groups: {all_groups_in_data}"
            )
        reference_items = [reference]
    elif isinstance(reference, tuple):
        # Single pooled reference
        missing = [g for g in reference if g not in all_groups_in_data]
        if missing:
            raise ValueError(
                f"Reference groups not found in data: {missing}. "
                f"Available groups: {all_groups_in_data}"
            )
        reference_items = [reference]
    elif isinstance(reference, list):
        # List of references (can be strings or tuples)
        reference_items = []
        for item in reference:
            if isinstance(item, str):
                if item == "rest":
                    reference_items.append("rest")
                elif item not in all_groups_in_data:
                    raise ValueError(
                        f"Reference '{item}' not found in data. "
                        f"Available groups: {all_groups_in_data}"
                    )
                else:
                    reference_items.append(item)
            elif isinstance(item, tuple):
                missing = [g for g in item if g not in all_groups_in_data]
                if missing:
                    raise ValueError(
                        f"Reference groups not found in data: {missing}. "
                        f"Available groups: {all_groups_in_data}"
                    )
                reference_items.append(item)
            else:
                raise ValueError(
                    f"Reference list items must be strings or tuples. "
                    f"Got: {type(item)}. "
                    f"Example: reference=['WT', ('WT', 'Het')]"
                )
    else:
        raise ValueError(
            f"'reference' must be 'rest', a string, a tuple, or a list. "
            f"Got: {type(reference)}. "
            f"Examples:\n"
            f"  reference='rest' → compare against all other groups\n"
            f"  reference='WT' → compare against WT only\n"
            f"  reference=['WT', 'Het'] → separate comparisons vs WT and vs Het\n"
            f"  reference=[('WT', 'Het')] → one comparison vs pooled WT+Het"
        )
    
    # Check comparison count
    n_comparisons = len(positive_groups) * len(reference_items)
    if n_comparisons > max_comparisons and not allow_many_comparisons:
        raise ValueError(
            f"Too many comparisons requested: {n_comparisons} "
            f"({len(positive_groups)} groups × {len(reference_items)} references). "
            f"Maximum allowed: {max_comparisons}.\n"
            f"To proceed anyway, set allow_many_comparisons=True.\n"
            f"Or reduce comparisons by:\n"
            f"  - Specifying fewer groups: groups=['CE', 'HTA'] instead of groups='all'\n"
            f"  - Using fewer references: reference='WT' instead of reference={reference_items}"
        )
    
    # Build ComparisonSpec objects
    # Get embryo IDs for each group
    group_to_embryos = {}
    for group in all_groups_in_data:
        group_mask = df[groupby] == group
        group_to_embryos[group] = df.loc[group_mask, embryo_id_col].unique().tolist()
    
    specs = []
    for pos_group in positive_groups:
        pos_members = group_to_embryos[pos_group]
        
        for ref_item in reference_items:
            if ref_item == "rest":
                # Reference is all groups except positive
                neg_members = []
                for g in all_groups_in_data:
                    if g != pos_group:
                        neg_members.extend(group_to_embryos[g])
                neg_label = "rest"
                neg_mode = "rest"
                neg_member_list = list(set(neg_members))  # deduplicate
                
            elif isinstance(ref_item, str):
                # Single reference group
                neg_label = ref_item
                neg_member_list = group_to_embryos[ref_item]
                neg_mode = "single"
                
            elif isinstance(ref_item, tuple):
                # Pooled reference groups
                neg_members = []
                for g in ref_item:
                    neg_members.extend(group_to_embryos[g])
                neg_label = "+".join(ref_item)  # Preserve user order
                neg_member_list = list(set(neg_members))  # deduplicate
                neg_mode = "pooled"
            else:
                # Should not reach here due to validation above
                raise ValueError(f"Unexpected reference item type: {type(ref_item)}")
            
            # Create ComparisonSpec
            spec = ComparisonSpec(
                positive=pos_group,
                negative=neg_label,
                positive_members=pos_members,
                negative_members=neg_member_list,
                negative_mode=neg_mode,
            )
            specs.append(spec)
    
    return specs


def run_multiclass_classification_test(
    df: pd.DataFrame,
    groups: Dict[str, List[str]],
    features: Union[str, List[str]] = 'z_mu_b',
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
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
    """
    Run a time-resolved multiclass classification test (One-vs-Rest AUROC).

    For each class, computes AUROC separating that class from all others combined,
    with permutation-based null distributions for significance testing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trajectory data with embryo_id and time columns
    groups : Dict[str, List[str]]
        Mapping of class_label -> list of embryo IDs
        Example: {'CE': [id1, id2], 'HTA': [id3, id4], 'WT': [id5, id6]}
    features : str or List[str]
        'z_mu_b' to auto-select VAE biological features, or list of column names
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
    n_jobs : int
        Number of parallel jobs for permutation testing (default: 1)
    min_samples_per_class : int
        Minimum samples per class per time bin (default: 3)
    within_bin_time_stratification : bool
        If True, permutation testing shuffles labels within fine time strata
        (default: True)
    within_bin_time_strata_width : float
        Width (hours) of time strata for stratified permutation (default: 0.5)
    skip_bin_if_not_all_present : bool
        If True, skip time bins where any class is missing. If False, proceed
        but warn that interpretability is reduced due to absent classes.
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print progress (default: True)

    Returns
    -------
    results : Dict[str, Any]
        {
            'ovr_classification': {
                'ClassA': DataFrame with auroc, null stats, pval per time bin,
                'ClassB': DataFrame with auroc, null stats, pval per time bin,
                ...
            },
            'confusion_matrices': {
                time_bin: DataFrame (confusion matrix with class labels),
                ...
            },
            'embryo_predictions': DataFrame with per-embryo prediction details,
            'summary': {
                'per_class': {class: {earliest_sig, max_auroc, ...}},
                'overall_accuracy': float,
                'data_quality': {total_bins, bins_with_all_classes, ...},
            },
            'config': {...}
        }
    """
    # Validate inputs
    class_labels = list(groups.keys())
    n_classes = len(class_labels)

    if n_classes < 2:
        raise ValueError(f"Need at least 2 classes, got {n_classes}")

    if verbose:
        print(f"Multiclass comparison with {n_classes} classes: {class_labels}")
        print("Classification weighting: class_weight='balanced' (enabled by default)")

    # Build embryo_id -> class_label mapping
    embryo_to_class = {}
    for class_label, embryo_ids in groups.items():
        for eid in embryo_ids:
            if eid in embryo_to_class:
                print(f"Warning: {eid} appears in multiple groups. "
                      f"Using '{class_label}' (overwrites '{embryo_to_class[eid]}')")
            embryo_to_class[eid] = class_label

    # Filter data to only include embryos in groups
    all_embryo_ids = list(embryo_to_class.keys())
    df_filtered = df[df[embryo_id_col].isin(all_embryo_ids)].copy()

    if len(df_filtered) == 0:
        raise ValueError("No data found for specified embryo IDs")

    # Add class label column
    df_filtered['_class_label'] = df_filtered[embryo_id_col].map(embryo_to_class)

    # Report group sizes
    if verbose:
        print("\nGroup sizes:")
        for class_label in class_labels:
            n_embryos = len(groups[class_label])
            n_rows = (df_filtered['_class_label'] == class_label).sum()
            print(f"  {class_label}: {n_embryos} embryos, {n_rows} rows")

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
        print(f"\nUsing {len(feature_cols)} feature columns")

    # Create time bins
    df_filtered['_time_bin'] = (
        np.floor(df_filtered[time_col] / bin_width) * bin_width
    ).astype(int)

    # Bin embeddings: average per embryo x time_bin
    groupby_cols = [embryo_id_col, '_time_bin', '_class_label']
    df_binned = df_filtered.groupby(groupby_cols, as_index=False)[feature_cols].mean()

    # Run multiclass classification
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
        verbose=verbose
    )

    # Compute summary statistics
    summary = _compute_multiclass_summary(
        ovr_results,
        embryo_predictions,
        class_labels,
        data_quality=data_quality
    )

    # Build config dict
    config = {
        'class_labels': class_labels,
        'n_classes': n_classes,
        'features': feature_cols,
        'bin_width': bin_width,
        'n_permutations': n_permutations,
        'n_splits': n_splits,
        'n_jobs': n_jobs,
        'group_sizes': {label: len(ids) for label, ids in groups.items()},
        'within_bin_time_stratification': within_bin_time_stratification,
        'within_bin_time_strata_width': within_bin_time_strata_width,
        'skip_bin_if_not_all_present': skip_bin_if_not_all_present,
    }

    return {
        'ovr_classification': ovr_results,
        'confusion_matrices': confusion_matrices,
        'embryo_predictions': embryo_predictions,
        'summary': summary,
        'config': config,
    }


def _run_multiclass_classification(
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
    verbose: bool
) -> tuple:
    """
    Run multiclass OvR classification with permutation testing.

    Returns
    -------
    ovr_results : Dict[str, pd.DataFrame]
        Per-class AUROC DataFrames
    confusion_matrices : Dict[int, pd.DataFrame]
        Confusion matrix per time bin
    embryo_predictions : pd.DataFrame
        Per-embryo prediction details
    data_quality : Dict[str, int]
        Bin-level data quality counts
    """
    time_bins = sorted(df_binned['_time_bin'].unique())
    total_bins = len(time_bins)
    n_complete_bins = 0
    n_incomplete_bins = 0
    n_skipped_bins = 0

    # Initialize containers
    ovr_results = {label: [] for label in class_labels}
    confusion_matrices = {}
    all_embryo_predictions = []

    # Create label encoder mapping
    label_to_int = {label: i for i, label in enumerate(class_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    for i, t in enumerate(time_bins):
        if verbose:
            print(f"\n  [{i+1}/{len(time_bins)}] Time bin {t} hpf...")

        # Get data for this time bin
        sub = df_binned[df_binned['_time_bin'] == t]
        X = sub[feature_cols].values
        y_labels = sub['_class_label'].values
        embryo_ids = sub[embryo_id_col].values

        # Encode labels as integers
        y = np.array([label_to_int[label] for label in y_labels])

        # Check class counts
        class_counts = {label: np.sum(y_labels == label) for label in class_labels}
        if verbose:
            print(f"    Class counts: {class_counts}")
            print("    Using balanced class weights for this bin")
        present_classes = [label for label, count in class_counts.items() if count > 0]
        missing_classes = [label for label, count in class_counts.items() if count == 0]

        if missing_classes:
            n_incomplete_bins += 1
        else:
            n_complete_bins += 1

        if len(present_classes) < 2:
            if verbose:
                print("    Skipped (need at least 2 classes present)")
                print(f"    Class counts: {class_counts}")
            n_skipped_bins += 1
            continue

        if skip_bin_if_not_all_present and missing_classes:
            if verbose:
                print(f"    Skipped (missing classes: {missing_classes})")
            n_skipped_bins += 1
            continue

        if not skip_bin_if_not_all_present and missing_classes and verbose:
            print(f"    Warning: missing classes {missing_classes}; interpretability may be reduced")

        min_count = min(class_counts[label] for label in present_classes)

        if min_count < min_samples_per_class:
            if verbose:
                print(f"    Skipped (min class has {min_count} samples, need {min_samples_per_class})")
                print(f"    Class counts: {class_counts}")
            n_skipped_bins += 1
            continue

        clf = _make_logistic_classifier(
            n_classes=len(class_labels),
            random_state=random_state,
        )

        # Cross-validated predictions
        n_splits_actual = min(n_splits, min_count)
        cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=random_state)

        try:
            # Get probability predictions for all classes
            probs = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
            present_class_indices = np.array(sorted(np.unique(y)))
            class_index_to_col = {cls_idx: col_idx for col_idx, cls_idx in enumerate(present_class_indices)}
            pred_col_idx = np.argmax(probs, axis=1)
            pred_classes = present_class_indices[pred_col_idx]
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            n_skipped_bins += 1
            continue

        # Compute time strata if stratification is enabled
        time_strata = None
        if within_bin_time_stratification:
            sub_raw = df_raw[df_raw['_time_bin'] == t]
            embryo_mean_times = (
                sub_raw.groupby(embryo_id_col)[time_col]
                .mean()
                .reindex(embryo_ids)
                .values
            )
            time_strata = np.floor(
                (embryo_mean_times - float(t)) / float(within_bin_time_strata_width)
            ).astype(int)

        # Compute OvR AUROC for each class
        for class_label in class_labels:
            class_idx = label_to_int[class_label]
            col_idx = class_index_to_col.get(class_idx)
            if col_idx is None:
                continue

            # Create binary labels for this class vs rest
            y_binary = (y == class_idx).astype(int)
            class_probs = probs[:, col_idx]

            # Compute observed AUROC
            n_positive = int(np.sum(y_binary == 1))
            n_negative = int(np.sum(y_binary == 0))

            if n_positive == 0 or n_negative == 0:
                continue

            try:
                true_auroc = roc_auc_score(y_binary, class_probs)
            except Exception:
                continue

            # Permutation test for this class
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
                time_bin=t
            )

            # Compute p-value
            null_aurocs = null_aurocs[np.isfinite(null_aurocs)]
            if len(null_aurocs) == 0:
                continue

            k = np.sum(null_aurocs >= true_auroc)
            pval = (k + 1) / (len(null_aurocs) + 1)

            # Store results
            ovr_results[class_label].append({
                'time_bin': t,
                'time_bin_start': float(t),
                'time_bin_end': float(t) + float(bin_width),
                'time_bin_center': float(t) + float(bin_width) / 2.0,
                'bin_width': float(bin_width),
                'auroc_observed': true_auroc,
                'auroc_null_mean': np.mean(null_aurocs),
                'auroc_null_std': np.std(null_aurocs),
                'pval': pval,
                'n_positive': n_positive,
                'n_negative': n_negative,
                'positive_class': class_label,
                'negative_class': 'Rest',
            })

            if verbose:
                sig_marker = "*" if pval < 0.05 else ""
                print(f"    {class_label} vs Rest: AUROC={true_auroc:.3f}, p={pval:.3f}{sig_marker}")

        # Build confusion matrix for this time bin
        cm = confusion_matrix(y, pred_classes, labels=list(range(len(class_labels))))
        cm_df = pd.DataFrame(
            cm,
            index=class_labels,
            columns=class_labels
        )
        cm_df.index.name = 'true_class'
        cm_df.columns.name = 'predicted_class'
        if missing_classes:
            cm_df.loc[missing_classes, :] = np.nan
            cm_df.loc[:, missing_classes] = np.nan
        confusion_matrices[t] = cm_df

        # Store embryo predictions
        for j, (eid, true_label, pred_idx) in enumerate(zip(embryo_ids, y_labels, pred_classes)):
            pred_record = {
                'embryo_id': eid,
                'time_bin': t,
                'time_bin_center': float(t) + float(bin_width) / 2.0,
                'true_class': true_label,
                'pred_class': int_to_label[pred_idx],
                'is_correct': true_label == int_to_label[pred_idx],
            }
            # Add probability for each class
            for class_label in class_labels:
                class_idx = label_to_int[class_label]
                col_idx = class_index_to_col.get(class_idx)
                if col_idx is None:
                    pred_record[f'pred_proba_{class_label}'] = np.nan
                else:
                    pred_record[f'pred_proba_{class_label}'] = probs[j, col_idx]

            all_embryo_predictions.append(pred_record)

    # Convert results to DataFrames
    ovr_dfs = {}
    for class_label in class_labels:
        if ovr_results[class_label]:
            ovr_dfs[class_label] = pd.DataFrame(ovr_results[class_label])
        else:
            ovr_dfs[class_label] = pd.DataFrame()

    embryo_predictions_df = pd.DataFrame(all_embryo_predictions) if all_embryo_predictions else None

    data_quality = {
        'total_bins': total_bins,
        'bins_with_all_classes': n_complete_bins,
        'bins_with_missing_classes': n_incomplete_bins,
        'bins_skipped': n_skipped_bins,
    }

    return ovr_dfs, confusion_matrices, embryo_predictions_df, data_quality


def _permutation_test_ovr(
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
    time_bin: int
) -> np.ndarray:
    """
    Run permutation test for OvR AUROC of a specific class.

    Shuffles multiclass labels (not binary), then computes OvR AUROC.
    """
    def _single_perm_ovr_auc(seed: int) -> float:
        local_rng = np.random.default_rng(seed)

        # Stratified permutation: shuffle multiclass labels within time strata
        if time_strata is not None:
            y_perm = y.copy()
            for stratum_id in np.unique(time_strata):
                stratum_mask = (time_strata == stratum_id)
                if np.sum(stratum_mask) > 1:
                    y_perm[stratum_mask] = local_rng.permutation(y[stratum_mask])
        else:
            y_perm = local_rng.permutation(y)

        try:
            clf = _make_logistic_classifier(
                n_classes=n_classes,
                random_state=random_state,
            )

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            probs_perm = cross_val_predict(clf, X, y_perm, cv=cv, method='predict_proba')

            # Compute OvR AUROC for the target class
            y_binary_perm = (y_perm == class_idx).astype(int)

            # Check if both classes present
            if len(np.unique(y_binary_perm)) < 2:
                return float('nan')

            return float(roc_auc_score(y_binary_perm, probs_perm[:, class_idx]))
        except Exception:
            return float('nan')

    if n_permutations <= 0:
        return np.array([], dtype=float)

    # Generate deterministic seeds
    base_seed = int(random_state) + 1000003 * (bin_index + 1) + 10007 * int(time_bin) + 31 * class_idx
    seeds = [base_seed + j for j in range(n_permutations)]

    # Run permutations
    use_parallel = (
        n_jobs is not None
        and n_jobs != 1
        and Parallel is not None
        and delayed is not None
        and n_permutations > 1
    )

    if use_parallel:
        null_list = Parallel(n_jobs=n_jobs)(delayed(_single_perm_ovr_auc)(s) for s in seeds)
        null_aurocs = np.asarray(null_list, dtype=float)
    else:
        null_aurocs = np.array([_single_perm_ovr_auc(s) for s in seeds], dtype=float)

    return null_aurocs


def _compute_multiclass_summary(
    ovr_results: Dict[str, pd.DataFrame],
    embryo_predictions: Optional[pd.DataFrame],
    class_labels: List[str],
    alpha: float = 0.05,
    data_quality: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Compute summary statistics from multiclass results."""

    per_class_summary = {}

    for class_label in class_labels:
        df = ovr_results.get(class_label)

        if df is None or df.empty:
            per_class_summary[class_label] = {
                'earliest_significant_hpf': None,
                'max_auroc': None,
                'max_auroc_hpf': None,
                'n_significant_bins': 0,
            }
            continue

        # Find significant bins
        sig_mask = df['pval'] < alpha
        n_significant = sig_mask.sum()

        # Earliest significant
        if n_significant > 0:
            earliest_hpf = df.loc[sig_mask, 'time_bin'].min()
        else:
            earliest_hpf = None

        # Max AUROC
        max_idx = df['auroc_observed'].idxmax()
        max_auroc = df.loc[max_idx, 'auroc_observed']
        max_auroc_hpf = df.loc[max_idx, 'time_bin']

        per_class_summary[class_label] = {
            'earliest_significant_hpf': earliest_hpf,
            'max_auroc': max_auroc,
            'max_auroc_hpf': max_auroc_hpf,
            'n_significant_bins': n_significant,
        }

    # Overall accuracy from embryo predictions
    overall_accuracy = None
    if embryo_predictions is not None and not embryo_predictions.empty:
        overall_accuracy = embryo_predictions['is_correct'].mean()

    return {
        'per_class': per_class_summary,
        'overall_accuracy': overall_accuracy,
        'data_quality': data_quality or {},
    }


def extract_temporal_confusion_profile(
    confusion_matrices: Dict[int, pd.DataFrame],
    class_labels: List[str]
) -> pd.DataFrame:
    """
    Extract temporal confusion profile from confusion matrices.

    For each class and time bin, shows the proportion classified as each class.

    Parameters
    ----------
    confusion_matrices : Dict[int, pd.DataFrame]
        Confusion matrices keyed by time bin
    class_labels : List[str]
        List of class labels in order

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        time_bin, true_class, predicted_class, proportion, count
    """
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

                records.append({
                    'time_bin': time_bin,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'proportion': proportion,
                    'count': count,
                    'is_correct': true_class == pred_class,
                })

    return pd.DataFrame(records)


def run_classification_test(
    df: pd.DataFrame,
    groupby: str,
    groups: Union[str, List[str]] = "all",
    reference: Union[str, List[Union[str, Tuple[str, ...]]]] = "rest",
    features: Union[str, List[str]] = 'z_mu_b',
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
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
) -> MulticlassOVRResults:
    """
    Run flexible group comparison tests with Scanpy-style API.
    
    This is the new recommended API for multiclass comparisons. It provides
    flexible control over which groups to compare and what reference(s) to use.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw trajectory data with embryo_id and time columns
    groupby : str
        Column name containing group labels (e.g., 'cluster_categories', 'genotype')
    groups : str or list of str
        Which groups to test.
        - "all" (default): Test all unique values in groupby column
        - ["CE", "HTA"]: Test only these specific groups
        - "CE": Test single group
    reference : str or list of str/tuple
        Reference group(s) for comparison.
        - "rest" (default): Compare each group against all others combined (classic OvR)
        - "WT": Compare each group against this single group
        - ["WT", "Het"]: Compare each group separately against WT AND against Het
        - [("WT", "Het")]: Compare each group against pooled WT+Het
        - ["WT", ("WT", "Het")]: Both separate WT comparison and pooled comparison
    features : str or list of str
        'z_mu_b' to auto-select VAE biological features, or list of column names
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
    n_jobs : int
        Number of parallel jobs for permutation testing (default: 1)
    min_samples_per_class : int
        Minimum samples per class per time bin (default: 3)
    within_bin_time_stratification : bool
        If True, permutation testing shuffles labels within fine time strata
    within_bin_time_strata_width : float
        Width (hours) of time strata for stratified permutation (default: 0.5)
    allow_many_comparisons : bool
        If True, allow more than max_comparisons (default: False)
    max_comparisons : int
        Maximum allowed comparisons before raising error (default: 50)
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print progress (default: True)
    
    Returns
    -------
    MulticlassOVRResults
        Result container with dict-like access and iteration methods.
        - results['CE', 'WT'] → DataFrame for CE vs WT comparison
        - results.iter_comparisons() → iterate over all comparisons
        - results.comparisons → full long-format DataFrame
        - results.save(path) → save to disk
    
    Examples
    --------
    >>> # Classic One-vs-Rest (each class vs all others)
    >>> results = run_classification_test(
    ...     df, groupby='cluster_categories', groups='all', reference='rest'
    ... )
    
    >>> # CE and HTA each compared against WT
    >>> results = run_classification_test(
    ...     df, groupby='cluster', groups=['CE', 'HTA'], reference='WT'
    ... )
    
    >>> # CE against WT and against pooled WT+Het
    >>> results = run_classification_test(
    ...     df, groupby='cluster', groups='CE', 
    ...     reference=['WT', ('WT', 'Het')]
    ... )
    
    >>> # Access results
    >>> ce_vs_wt = results['CE', 'WT']
    >>> for (pos, neg), df in results.iter_comparisons():
    ...     print(f"{pos} vs {neg}: max AUROC = {df['auroc_obs'].max():.2f}")
    
    >>> # Save/load
    >>> results.save('results/my_analysis/')
    >>> loaded = MulticlassOVRResults.from_dir('results/my_analysis/')
    
    Notes
    -----
    This function provides a Scanpy/Seurat-style API for flexible comparisons.
    For the legacy dict-based API, use `run_multiclass_classification_test()`.
    """
    from datetime import datetime
    import json
    
    if verbose:
        print(f"=== Flexible Group Classification Test ===")
        print(f"Groupby column: {groupby}")
        print(f"Groups: {groups}")
        print(f"Reference: {reference}")
        print("Classifier policy: class_weight='balanced' for all logistic models")
    
    # Resolve comparisons
    comparison_specs = _resolve_comparison_groups(
        df=df,
        groupby=groupby,
        groups=groups,
        reference=reference,
        embryo_id_col=embryo_id_col,
        allow_many_comparisons=allow_many_comparisons,
        max_comparisons=max_comparisons,
    )
    
    if verbose:
        print(f"\n{len(comparison_specs)} comparisons to run:")
        for spec in comparison_specs:
            print(f"  {spec.positive} vs {spec.negative}")
    
    # Run each comparison using the existing function
    all_rows = []
    
    for spec in comparison_specs:
        if verbose:
            print(f"\n--- Running: {spec.positive} vs {spec.negative} ---")
        
        # Create groups dict for legacy API
        groups_dict = {
            spec.positive: spec.positive_members,
            spec.negative: spec.negative_members,
        }
        
        # Run comparison
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
        
        # Extract OvR AUROC for the positive class
        positive_df = result['ovr_classification'].get(spec.positive)
        if positive_df is None or positive_df.empty:
            if verbose:
                print(f"  Warning: No results for {spec.positive}")
            continue
        
        # Add comparison metadata to each row
        for _, row in positive_df.iterrows():
            row_dict = row.to_dict()
            row_dict['comparison_id'] = spec.comparison_id
            row_dict['groupby'] = groupby
            row_dict['positive'] = spec.positive
            row_dict['negative'] = spec.negative
            row_dict['negative_members'] = json.dumps(spec.negative_members)
            row_dict['negative_mode'] = spec.negative_mode
            # Rename for consistency
            if 'auroc_observed' in row_dict:
                row_dict['auroc_obs'] = row_dict.pop('auroc_observed')
            all_rows.append(row_dict)
    
    if not all_rows:
        raise ValueError("No valid comparisons produced results")
    
    # Build comparisons DataFrame
    comparisons_df = pd.DataFrame(all_rows)
    
    # Build metadata
    metadata = {
        'groupby': groupby,
        'groups': groups if isinstance(groups, list) else [groups] if groups != "all" else "all",
        'reference': reference,
        'features': features if isinstance(features, list) else [features],
        'bin_width': bin_width,
        'n_permutations': n_permutations,
        'n_splits': n_splits,
        'n_jobs': n_jobs,
        'min_samples_per_class': min_samples_per_class,
        'within_bin_time_stratification': within_bin_time_stratification,
        'within_bin_time_strata_width': within_bin_time_strata_width,
        'max_comparisons': max_comparisons,
        'n_comparisons': len(comparison_specs),
        'timestamp': datetime.now().isoformat(),
        'random_state': random_state,
    }
    
    # Create and return results object
    results = MulticlassOVRResults(
        comparisons=comparisons_df,
        metadata=metadata,
    )
    
    if verbose:
        print(f"\n=== Complete ===")
        print(f"{len(results)} comparisons")
        print(f"{len(comparisons_df)} total rows")
    
    return results


__all__ = [
    "run_multiclass_classification_test",
    "run_classification_test",
    "extract_temporal_confusion_profile",
]
