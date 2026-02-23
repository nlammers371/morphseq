import pandas as pd
import os
import numpy as np
import seaborn as sns 
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from matplotlib.colors import Normalize
try:
    from matplotlib import colormaps as mpl_colormaps
except ImportError:  # pragma: no cover - older matplotlib fallback
    mpl_colormaps = None
from pathlib import Path
from joblib import Parallel, delayed

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
except ImportError:  # pragma: no cover - SciPy optional
    linkage = None
    leaves_list = None
    pdist = None


# Use the parent directory of this file for results
# results_dir = os.getcwd()
results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014"
data_dir_base = os.path.join(results_dir, "data")
plot_dir_base = os.path.join(results_dir, "plots")

print(f"Results directory: {results_dir}")
os.makedirs(plot_dir_base, exist_ok=True)
os.makedirs(data_dir_base, exist_ok=True)


morphseq_root = os.environ.get('MORPHSEQ_REPO_ROOT')
morphseq_root = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"
print(f"MORPHSEQ_REPO_ROOT: {morphseq_root}")
os.chdir(morphseq_root)

# Add morphseq root to Python path
import sys
sys.path.insert(0, morphseq_root)

# ============================================================================
# CLASSIFICATION ANALYSIS CONFIGURATION
# ============================================================================

# Number of permutations for null distribution (higher = more accurate p-values)
# Quick test: 100, Standard: 500, Publication: 1000-5000
N_PERMUTATIONS = int(os.environ.get("MORPHSEQ_N_PERMUTATIONS", 100))

# Number of cross-validation folds for AUROC estimation
#   This means:
#   - The data at each time bin is split into 5 parts
#   - The model trains on 4 parts and tests on 1 part
#   - This repeats 5 times so every sample gets tested once
#   - The 5 AUROC scores are averaged to get the final observed AUROC

#   Common values:

#   - n_splits=3: Fast, less stable (66% train / 33% test each fold)
#   - n_splits=5: Standard, good balance (80% train / 20% test) ✓ Current
#   - n_splits=10: More stable, slower (90% train / 10% test)
#   - n_splits=n: Leave-one-out (very slow, max stability)
N_CV_SPLITS = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Significance threshold for detecting onset
ALPHA = 0.05

print(f"Classification configuration:")
print(f"  Permutations: {N_PERMUTATIONS}")
print(f"  CV splits: {N_CV_SPLITS}")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Alpha: {ALPHA}")

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


# ============================================================================
# BINNING FUNCTIONS (copied from detection script to avoid running its main code)
# ============================================================================

def bin_by_embryo_time(
    df,
    time_col="predicted_stage_hpf",
    z_cols=None,
    bin_width=2.0,
    suffix="_binned"
):
    """
    Bin VAE embeddings by predicted time and embryo.

    Always averages embeddings per embryo_id × time_bin,
    keeping all non-latent metadata columns (e.g., genotype).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'embryo_id', 'predicted_stage_hpf', and latent columns.
    time_col : str
        Column name to bin by.
    z_cols : list or None
        Columns to average. If None, auto-detect those containing 'z_mu_b'.
    bin_width : float
        Width of time bins (same units as time_col, usually hours).
    suffix : str
        Suffix to append to averaged latent column names.

    Returns
    -------
    pd.DataFrame
        One row per (embryo_id, time_bin) containing averaged latent columns and preserved metadata.
    """

    df = df.copy()

    # detect latent columns
    if z_cols is None:
        z_cols = [c for c in df.columns if "z_mu_b" in c]
        if not z_cols:
            raise ValueError("No latent columns found matching pattern 'z_mu_b'.")

    # create time bins
    df["time_bin"] = (np.floor(df[time_col] / bin_width) * bin_width).astype(int)

    # average latent vectors per embryo × time_bin
    agg = (
        df.groupby(["embryo_id", "time_bin"], as_index=False)[z_cols]
        .mean()
    )

    # rename averaged latent columns
    agg.rename(columns={c: f"{c}{suffix}" for c in z_cols}, inplace=True)

    # merge back non-latent metadata (take first unique per embryo)
    # Exclude time_bin and time_col from meta_cols to avoid conflicts
    meta_cols = [c for c in df.columns if c not in z_cols + [time_col, "time_bin"]]
    meta_df = (
        df[meta_cols]
        .drop_duplicates(subset=["embryo_id"])
    )

    # merge metadata back in
    out = agg.merge(meta_df, on="embryo_id", how="left")

    # ensure sorting
    out = out.sort_values(["embryo_id", "time_bin"]).reset_index(drop=True)

    return out

# from src.functions.embryo_df_performance_metrics import *
# from src.functions.spline_morph_spline_metrics import *

# Import TZ experiments
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
        print(df['genotype'].value_counts())
        dfs.append(df)
        print(f"Loaded {exp}: {len(df)} rows")
    except:
        print(f"Missing: {exp}")

# Combine all data
combined_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(combined_df)} rows from {len(dfs)} experiments")


# ============================================================================
# PREDICTIVE SIGNAL TEST FUNCTION
# ============================================================================

def predictive_signal_test(
    df_binned,
    group_col="genotype",
    time_col="time_bin",
    z_cols=None,
    n_splits=5,
    n_perm=100,
    random_state=None,
    return_embryo_probs=True,
    use_class_weights=True,
):
    """
    Predictive classifier + label-shuffling test across time bins.

    This test evaluates whether genotype labels can be predicted from
    morphological features (VAE embeddings) better than chance, using
    a logistic regression classifier with cross-validation.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data (output of bin_by_embryo_time).
    group_col : str
        Column specifying experimental group (e.g., genotype).
    time_col : str
        Column specifying time bins.
    z_cols : list or None
        Latent columns to use as features. Auto-detected if None.
    n_splits : int
        Number of cross-validation splits.
    n_perm : int
        Number of permutations for null distribution.
    random_state : int or None
        Random seed for reproducibility.
    return_embryo_probs : bool
        If True, return per-embryo prediction probabilities in addition to aggregate stats.
    use_class_weights : bool
        If True, use balanced class weights to handle class imbalance. Default: True.
        This helps prevent bias when one class (e.g., wildtype/het) is much larger.

    Returns
    -------
    df_results : pd.DataFrame
        One row per time_bin with AUROC statistics and p-values.
    df_embryo_probs : pd.DataFrame (optional)
        Per-embryo prediction probabilities if return_embryo_probs=True.
        Columns include: embryo_id, time_bin, true_label, pred_prob,
        confidence, predicted_label, support_true, signed_margin
    """
    rng = np.random.default_rng(random_state)
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]

    results = []
    embryo_predictions = [] if return_embryo_probs else None

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
            print(f"Skipping time bin {t}: insufficient samples ({min_samples_per_class} < {n_splits})")
            continue

        # --- True AUROC via cross-validation ---
        skf = StratifiedKFold(n_splits=min(n_splits, min_samples_per_class), shuffle=True, random_state=random_state)
        aucs = []
        for train_idx, test_idx in skf.split(X, y):
            model = LogisticRegression(max_iter=200, random_state=random_state)
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])
            # Ensure we consistently reference the positive-class column
            class_order = model.classes_
            if len(class_order) != 2:
                raise ValueError("Expected binary classification with two classes.")
            positive_class = class_order[1]
            positive_prob = proba[:, 1]
            aucs.append(roc_auc_score(y[test_idx], positive_prob))

            # Collect per-embryo predictions
            if return_embryo_probs:
                for i, idx in enumerate(test_idx):
                    true_label = y[idx]
                    p_pos = positive_prob[i]
                    support_true = p_pos if true_label == positive_class else 1.0 - p_pos
                    signed_margin = (1 if true_label == positive_class else -1) * (p_pos - 0.5)
                    embryo_predictions.append({
                        'embryo_id': embryo_ids[idx],
                        'time_bin': t,
                        'true_label': true_label,
                        'pred_prob': p_pos,
                        'confidence': np.abs(p_pos - 0.5),
                        'predicted_label': positive_class if p_pos > 0.5 else class_order[0],
                        'support_true': support_true,
                        'signed_margin': signed_margin
                    })

        true_auc = np.mean(aucs)

        # --- Null distribution via shuffled labels ---
        null_aucs = []
        for _ in range(n_perm):
            y_shuff = rng.permutation(y)
            perm_aucs = []
            for train_idx, test_idx in skf.split(X, y_shuff):
                model = LogisticRegression(max_iter=200, random_state=random_state)
                model.fit(X[train_idx], y_shuff[train_idx])
                prob = model.predict_proba(X[test_idx])[:, 1]
                perm_aucs.append(roc_auc_score(y_shuff[test_idx], prob))
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

    return df_results


# ============================================================================
# EMBRYO-LEVEL PENETRANCE ANALYSIS
# ============================================================================

def compute_embryo_penetrance(df_embryo_probs, confidence_threshold=0.1):
    """
    Compute per-embryo penetrance metrics from prediction probabilities.

    This quantifies how consistently each embryo expresses a classifiable
    phenotype across developmental time.

    Parameters
    ----------
    df_embryo_probs : pd.DataFrame
        Per-embryo predictions from predictive_signal_test().
        Must have columns: embryo_id, time_bin, true_label, pred_prob, confidence
    confidence_threshold : float
        Minimum confidence (|p - 0.5|) to consider a prediction "confident"

    Returns
    -------
    pd.DataFrame
        One row per embryo with penetrance metrics:
        - mean_confidence: Average prediction confidence magnitude across time
        - mean_support_true: Average probability assigned to the true class
        - mean_signed_margin: Average signed margin relative to 0.5 decision boundary
        - temporal_consistency: Fraction of time bins correctly classified
        - max_confidence: Peak confidence magnitude across development
        - min_support_true: Lowest probability assigned to the true class
        - min_signed_margin: Most negative margin (worst wrong-side confidence)
        - first_confident_time: First time bin with confidence > threshold
        - n_time_bins: Number of time bins embryo was observed
    """
    if df_embryo_probs.empty:
        return pd.DataFrame()

    penetrance_metrics = []

    for embryo_id, grp in df_embryo_probs.groupby('embryo_id'):
        # Sort by time
        grp = grp.sort_values('time_bin')

        # Compute metrics (guard against missing columns during experimentation)
        mean_conf = grp['confidence'].mean()
        max_conf = grp['confidence'].max()
        n_bins = len(grp)
        mean_support_true = grp['support_true'].mean() if 'support_true' in grp.columns else np.nan
        min_support_true = grp['support_true'].min() if 'support_true' in grp.columns else np.nan
        mean_signed_margin = grp['signed_margin'].mean() if 'signed_margin' in grp.columns else np.nan
        min_signed_margin = grp['signed_margin'].min() if 'signed_margin' in grp.columns else np.nan

        # Temporal consistency: fraction correctly classified
        correct = (grp['true_label'] == grp['predicted_label']).sum()
        temporal_consistency = correct / n_bins if n_bins > 0 else 0.0

        # First confident prediction time
        confident_bins = grp[grp['confidence'] > confidence_threshold]
        first_confident_time = confident_bins['time_bin'].min() if len(confident_bins) > 0 else np.nan

        # Get true label (should be constant per embryo)
        true_label = grp['true_label'].iloc[0]

        penetrance_metrics.append({
            'embryo_id': embryo_id,
            'true_label': true_label,
            'mean_confidence': mean_conf,
            'mean_support_true': mean_support_true,
            'mean_signed_margin': mean_signed_margin,
            'temporal_consistency': temporal_consistency,
            'max_confidence': max_conf,
            'min_support_true': min_support_true,
            'min_signed_margin': min_signed_margin,
            'first_confident_time': first_confident_time,
            'n_time_bins': n_bins,
            'mean_pred_prob': grp['pred_prob'].mean()
        })

    df_penetrance = pd.DataFrame(penetrance_metrics)

    # Classify embryos by penetrance level
    df_penetrance['penetrance_category'] = pd.cut(
        df_penetrance['mean_confidence'],
        bins=[0, 0.1, 0.2, 0.5],
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )

    return df_penetrance


def plot_auroc_over_time(df_auc, group1, group2, output_path=None):
    """
    Plot observed AUROC vs null distribution over time.

    Parameters
    ----------
    df_auc : pd.DataFrame
        Output from predictive_signal_test.
    group1, group2 : str
        Names of the two groups being compared.
    output_path : str or None
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_auc["time_bin"], df_auc["AUROC_obs"],
            label="Observed AUROC", color="black", linewidth=2, marker='o')

    ax.fill_between(
        df_auc["time_bin"],
        df_auc["AUROC_null_mean"] - df_auc["AUROC_null_std"],
        df_auc["AUROC_null_mean"] + df_auc["AUROC_null_std"],
        color="gray", alpha=0.3, label="Null ± 1σ"
    )

    ax.axhline(0.5, color="gray", ls="--", linewidth=1.5, alpha=0.7, label="Chance level")
    ax.set_xlabel("Predicted stage (hpf bin)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(f"Predictive Signal Test: {group1} vs {group2}", fontsize=14, fontweight='bold')
    ax.set_ylim([0.4, 1.0])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_auroc_with_significance(df_auc, group1, group2, alpha=0.05, output_path=None):
    """
    Plot AUROC with significance markers.

    Parameters
    ----------
    df_auc : pd.DataFrame
        Output from predictive_signal_test.
    group1, group2 : str
        Names of the two groups being compared.
    alpha : float
        Significance threshold.
    output_path : str or None
        Path to save figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: AUROC over time
    ax1 = axes[0]
    ax1.plot(df_auc["time_bin"], df_auc["AUROC_obs"],
            label="Observed AUROC", color="black", linewidth=2, marker='o')
    ax1.fill_between(
        df_auc["time_bin"],
        df_auc["AUROC_null_mean"] - df_auc["AUROC_null_std"],
        df_auc["AUROC_null_mean"] + df_auc["AUROC_null_std"],
        color="gray", alpha=0.3, label="Null ± 1σ"
    )
    ax1.axhline(0.5, color="gray", ls="--", linewidth=1.5, alpha=0.7)

    # Mark significant bins
    sig_bins = df_auc[df_auc["pval"] < alpha]
    if len(sig_bins) > 0:
        ax1.scatter(sig_bins["time_bin"], sig_bins["AUROC_obs"],
                   color='red', s=150, marker='*', zorder=5,
                   label=f'Significant (p < {alpha})', edgecolors='black', linewidth=1)

    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title(f"Predictive Signal Test: {group1} vs {group2}", fontsize=14, fontweight='bold')
    ax1.set_ylim([0.4, 1.0])
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: P-values over time
    ax2 = axes[1]
    ax2.plot(df_auc["time_bin"], df_auc["pval"],
            'o-', color='steelblue', linewidth=2, label='P-value')
    ax2.axhline(alpha, color='red', ls='--', linewidth=2, label=f'α = {alpha}')
    ax2.set_xlabel("Predicted stage (hpf bin)", fontsize=12)
    ax2.set_ylabel("P-value", fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


# ============================================================================
# EMBRYO-LEVEL VISUALIZATION FUNCTIONS
# ============================================================================

def plot_signed_margin_heatmap(df_embryo_probs, df_penetrance, group1, group2, output_path=None):
    """
    Heatmap showing signed-margin trajectories per embryo, split by genotype.

    Signed margin measures distance from the 0.5 decision boundary with sign,
    making it a direct proxy for phenotype penetrance (positive = predicted
    genotype, negative = wildtype-like).
    """
    if df_embryo_probs.empty or df_penetrance.empty:
        print("  Skipping heatmap: no embryo data")
        return None

    if 'signed_margin' not in df_embryo_probs.columns:
        print("  Skipping heatmap: signed_margin column not found")
        return None

    embryo_to_label = df_penetrance.set_index('embryo_id')['true_label'].to_dict()
    label_order = [label for label in [group1, group2] if label in df_embryo_probs['true_label'].unique()]
    if not label_order:
        label_order = list(df_embryo_probs['true_label'].unique())

    time_bins = sorted(df_embryo_probs['time_bin'].unique())

    pivot = df_embryo_probs.pivot_table(
        index='embryo_id',
        columns='time_bin',
        values='signed_margin',
        aggfunc='mean'
    )

    def _order_embryos_for_label(label, pivot_matrix):
        label_embryos = [eid for eid, lbl in embryo_to_label.items() if lbl == label]
        subset = pivot_matrix.loc[pivot_matrix.index.intersection(label_embryos)]
        if subset.empty:
            return []

        ordered_index = None
        if linkage and pdist and subset.shape[0] > 1:
            filled = subset.copy()
            col_means = filled.mean(axis=0)
            filled = filled.fillna(col_means)
            filled = filled.fillna(0.0)
            try:
                dist = pdist(filled.values, metric='euclidean')
                if np.any(dist > 0):
                    cluster = linkage(dist, method='ward')
                    ordered_index = list(filled.index[leaves_list(cluster)])
            except Exception as exc:
                print(f"  Ward clustering failed for {label}: {exc}")

        if ordered_index is None:
            ranking_metric = 'mean_signed_margin' if 'mean_signed_margin' in df_penetrance.columns else 'mean_confidence'
            ordered_index = list(
                df_penetrance[df_penetrance['embryo_id'].isin(subset.index)]
                .sort_values(ranking_metric, ascending=False)
                ['embryo_id']
            )

        return ordered_index

    row_sections = []
    for label in label_order:
        ordered_ids = _order_embryos_for_label(label, pivot)
        if ordered_ids:
            row_sections.append((label, ordered_ids))

    row_order = [eid for _, ids in row_sections for eid in ids]
    if not row_order:
        print("  Skipping heatmap: no embryos to plot")
        return None

    fig_height = max(8.0, len(row_order) * 0.2)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    pivot_plot = pivot.reindex(row_order).reindex(columns=time_bins)
    data = pivot_plot.values.astype(float)
    mask = np.isnan(data)

    cmap_name = 'RdBu_r'
    if mpl_colormaps is not None:
        cmap_instance = mpl_colormaps[cmap_name].copy()
    else:
        cmap_instance = plt.get_cmap(cmap_name).copy()
    cmap_instance.set_bad(color='lightgray')
    data_masked = np.ma.masked_array(data, mask=mask)

    im = ax.imshow(data_masked, aspect='auto', cmap=cmap_instance, vmin=-0.5, vmax=0.5)

    ax.set_xticks(np.arange(len(time_bins)))
    ax.set_xticklabels(time_bins, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Developmental Time (hpf)', fontsize=13, fontweight='bold')

    row_labels = [str(eid)[:15] for eid in row_order]
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_ylabel('Embryo ID', fontsize=13, fontweight='bold')

    section_sizes = [len(ids) for _, ids in row_sections if ids]
    section_boundaries = np.cumsum(section_sizes)

    for idx, (label, ids) in enumerate(row_sections):
        if not ids:
            continue
        start_pos = section_boundaries[idx - 1] if idx > 0 else 0
        end_pos = section_boundaries[idx]
        center_pos = (start_pos + end_pos) / 2

        ax.text(len(time_bins) + 0.6, center_pos, label.split('_')[-1].upper(),
                fontsize=12, fontweight='bold', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=2))

        if idx < len(row_sections) - 1:
            ax.axhline(end_pos - 0.5, color='black', linewidth=3, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Signed Margin\n(RED: predicted genotype, BLUE: wildtype-like)',
                   rotation=270, labelpad=25, fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        f'Embryo Signed-Margin Trajectories: {group1.split("_")[-1].upper()} vs {group2.split("_")[-1].upper()}',
        fontsize=16,
        fontweight='bold',
        pad=22
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_signed_margin_trajectories(df_embryo_probs, df_penetrance, group1, group2,
                                    output_path=None, max_embryos=30):
    """
    Plot signed-margin trajectories per genotype highlighting decision-boundary crossings.
    Line colours encode the embryo's mean signed margin (blue = negative / wildtype-like, red = positive).
    """
    if df_embryo_probs.empty or df_penetrance.empty:
        print("  Skipping trajectories: no embryo data")
        return None

    if 'signed_margin' not in df_embryo_probs.columns:
        print("  Skipping trajectories: signed_margin column not found")
        return None

    genotype_candidates = df_penetrance['true_label'].dropna().unique()
    genotypes = [g for g in [group1, group2] if g in genotype_candidates]
    if not genotypes:
        genotypes = sorted(genotype_candidates)
    if len(genotypes) == 0:
        print("  Skipping trajectories: no genotypes found")
        return None

    fig, axes = plt.subplots(1, len(genotypes), figsize=(8 * max(1, len(genotypes)), 6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, genotype in zip(axes, genotypes):
        genotype_penetrance = df_penetrance[df_penetrance['true_label'] == genotype].copy()
        if genotype_penetrance.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{genotype} (n=0)', fontsize=13, fontweight='bold')
            continue

        genotype_penetrance = genotype_penetrance.assign(
            abs_margin=np.abs(genotype_penetrance.get('mean_signed_margin', np.nan))
        )
        genotype_penetrance = genotype_penetrance.sort_values(
            by=['abs_margin', 'mean_signed_margin'], ascending=[False, False]
        ).head(max_embryos)
        top_embryos = genotype_penetrance['embryo_id'].values

        norm = Normalize(vmin=-0.5, vmax=0.5)
        cmap = plt.cm.RdBu_r
        alphas = np.linspace(0.35, 0.9, len(top_embryos)) if len(top_embryos) > 0 else []
        highlight_id = genotype_penetrance.iloc[0]['embryo_id'] if len(genotype_penetrance) > 0 else None
        penetrance_lookup = genotype_penetrance.set_index('embryo_id')

        for alpha, embryo_id in zip(alphas, top_embryos):
            embryo_curve = df_embryo_probs[df_embryo_probs['embryo_id'] == embryo_id].sort_values('time_bin')
            if embryo_curve.empty:
                continue

            mean_margin = np.nan
            if 'mean_signed_margin' in penetrance_lookup.columns:
                mean_margin = penetrance_lookup.at[embryo_id, 'mean_signed_margin']
            if np.isnan(mean_margin):
                mean_margin = embryo_curve['signed_margin'].mean()

            base_color = cmap(norm(mean_margin))
            color_rgba = (base_color[0], base_color[1], base_color[2], alpha)
            linewidth = 1.6
            marker_size = 3

            if embryo_id == highlight_id:
                linewidth = 2.8
                marker_size = 4
                color_rgba = cmap(norm(mean_margin))

            ax.plot(
                embryo_curve['time_bin'],
                embryo_curve['signed_margin'],
                color=color_rgba,
                linewidth=linewidth,
                marker='o',
                markersize=marker_size,
                alpha=0.95
            )

        ax.axhline(0.0, color='red', linestyle='--', linewidth=1.3, alpha=0.7, label='Decision boundary')
        ax.set_xlabel('Time (hpf)', fontsize=12)
        ax.set_ylabel('Signed Margin vs 0.5', fontsize=12)
        ax.set_ylim([-0.5, 0.5])
        ax.grid(alpha=0.3)
        ax.set_title(f'{genotype} (n={len(top_embryos)})', fontsize=14, fontweight='bold')

        if len(top_embryos) > 0:
            ax.legend(loc='upper left', fontsize=9)

    fig.suptitle(
        f'Embryo Signed Margin Trajectories: {group1.split("_")[-1]} vs {group2.split("_")[-1]}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_penetrance_distribution(df_penetrance, group1, group2, output_path=None):
    """
    Histogram and summary of penetrance distribution split by genotype.

    Displays distributions for multiple metrics including confidence, support
    for true class, signed margin, and temporal consistency. Each metric is
    shown with overlaid histograms colored by genotype for easy comparison.

    Parameters
    ----------
    df_penetrance : pd.DataFrame
        Penetrance metrics from compute_embryo_penetrance()
    group1, group2 : str
        Comparison groups (genotype labels)
    output_path : str or None
        Path to save figure
    """
    if df_penetrance.empty:
        print("  Skipping penetrance distribution: no data")
        return None

    # Detect which metrics are available
    metrics_config = [
        ('mean_signed_margin', 'Mean Signed Margin', 'coral', (-0.5, 0.5)),
        ('mean_support_true', 'Mean Support (True Class)', 'forestgreen', (0, 1.0)),
        ('mean_confidence', 'Mean Confidence (|p - 0.5|)', 'steelblue', (0, 0.5)),
        ('temporal_consistency', 'Temporal Consistency', 'darkorange', (0, 1.0))
    ]
    available_metrics = [(col, title, color, xlim) for col, title, color, xlim in metrics_config if col in df_penetrance.columns]

    if len(available_metrics) == 0:
        print("  Skipping penetrance distribution: no metrics found")
        return None

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(7 * ((n_metrics + 1) // 2), 10))
    axes = np.atleast_2d(axes).flatten()

    # Get unique genotypes
    genotypes = [g for g in [group1, group2] if g in df_penetrance['true_label'].unique()]
    palette = ['dodgerblue', 'orangered', 'mediumseagreen', 'mediumpurple']
    colors_by_genotype = {genotype: palette[idx % len(palette)] for idx, genotype in enumerate(genotypes)}

    for idx, (col, title, base_color, (xmin, xmax)) in enumerate(available_metrics):
        ax = axes[idx]

        # Plot histograms for each genotype
        for genotype in genotypes:
            subset = df_penetrance[df_penetrance['true_label'] == genotype][col].dropna()
            if len(subset) == 0:
                continue

            ax.hist(subset, bins=15, alpha=0.6, color=colors_by_genotype.get(genotype, 'gray'),
                   edgecolor='black', linewidth=0.5, label=f'{genotype} (n={len(subset)})')

            # Add median line for this genotype
            median_val = subset.median()
            ax.axvline(median_val, color=colors_by_genotype.get(genotype, 'gray'),
                      linestyle='--', linewidth=2, alpha=0.8)

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Number of Embryos', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Penetrance Distributions by Genotype: {group1} vs {group2}',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    return fig


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

GENOTYPE_GROUPS = {
    "cep290": ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous'],
    "b9d2": ['b9d2_wildtype', 'b9d2_heterozygous', 'b9d2_homozygous'],
    "tmem67": ['tmem67_wildtype', 'tmem67_heterozygote', 'tmem67_homozygous'],
}

print("\n" + "="*80)
print("PREDICTIVE CLASSIFICATION ANALYSIS")
print("="*80)

from itertools import combinations

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

    # Bin embeddings by embryo and time
    print("\nBinning embeddings by embryo and time...")
    df_binned = bin_by_embryo_time(df_family, time_col="predicted_stage_hpf")
    print(f"Binned data: {len(df_binned)} rows")

    # Drop rows with NaN in latent columns
    binned_z_cols = [c for c in df_binned.columns if "_binned" in c]
    nan_counts = df_binned[binned_z_cols].isna().sum()
    if nan_counts.sum() > 0:
        dropped = df_binned[binned_z_cols].isna().any(axis=1).sum()
        print(f"Dropping {dropped} rows with NaNs in latent columns")
        df_binned = df_binned.dropna(subset=binned_z_cols)
        print(f"Remaining rows: {len(df_binned)}")

    # Get all genotypes present in the data
    present_genotypes = [g for g in genotype_values if g in df_binned['genotype'].unique()]
    print(f"\nPresent genotypes: {present_genotypes}")

    # Run pairwise comparisons
    pairwise_comparisons = list(combinations(present_genotypes, 2))
    print(f"Running {len(pairwise_comparisons)} pairwise comparisons...")

    all_results = []

    for idx, (group1, group2) in enumerate(pairwise_comparisons, 1):
        print(f"\n[{idx}/{len(pairwise_comparisons)}] Comparing: {group1} vs {group2}")

        # Filter to just these two genotypes
        df_pair = df_binned[df_binned['genotype'].isin([group1, group2])].copy()

        if len(df_pair) < 10:
            print(f"  Skipping: insufficient data ({len(df_pair)} samples)")
            continue

        # Run predictive signal test (now returns embryo-level data too)
        print(f"  Running predictive signal test...")
        df_auc, df_embryo_probs = predictive_signal_test(
            df_pair,
            group_col="genotype",
            n_splits=N_CV_SPLITS,
            n_perm=N_PERMUTATIONS,
            random_state=RANDOM_SEED,
            return_embryo_probs=True
        )

        if df_auc.empty:
            print(f"  No valid time bins for this comparison")
            continue

        # Add comparison info
        df_auc['group1'] = group1
        df_auc['group2'] = group2
        df_auc['comparison'] = f"{group1}_vs_{group2}"
        all_results.append(df_auc)

        print(f"  Results for {len(df_auc)} time bins, {len(df_embryo_probs['embryo_id'].unique())} embryos")

        # Find onset of predictive signal
        sig_bins = df_auc[df_auc["pval"] < ALPHA].sort_values("time_bin")
        if len(sig_bins) > 0:
            first_sig = sig_bins.iloc[0]
            print(f"  ✓ First significant predictive signal:")
            print(f"    Time: {first_sig['time_bin']} hpf")
            print(f"    AUROC: {first_sig['AUROC_obs']:.3f}")
            print(f"    P-value: {first_sig['pval']:.4f}")
        else:
            print(f"  ⚠ No significant predictive signal detected")

        # Compute embryo-level penetrance metrics
        print(f"  Computing embryo-level penetrance...")
        df_penetrance = compute_embryo_penetrance(df_embryo_probs, confidence_threshold=0.1)

        if not df_penetrance.empty:
            print(f"  Penetrance summary:")
            print(f"    Mean confidence: {df_penetrance['mean_confidence'].mean():.3f} ± {df_penetrance['mean_confidence'].std():.3f}")
            if 'mean_support_true' in df_penetrance.columns:
                print(f"    Mean support (true class): {df_penetrance['mean_support_true'].mean():.3f} ± {df_penetrance['mean_support_true'].std():.3f}")
            if 'mean_signed_margin' in df_penetrance.columns:
                print(f"    Mean signed margin: {df_penetrance['mean_signed_margin'].mean():.3f} ± {df_penetrance['mean_signed_margin'].std():.3f}")
            print(f"    Temporal consistency: {df_penetrance['temporal_consistency'].mean():.3f} ± {df_penetrance['temporal_consistency'].std():.3f}")

            # Count penetrance categories
            if 'penetrance_category' in df_penetrance.columns:
                cat_counts = df_penetrance['penetrance_category'].value_counts()
                print(f"    Penetrance categories: {dict(cat_counts)}")

        # Generate plots for this comparison
        print(f"  Generating plots...")

        # Create safe filename
        safe_comp_name = f"{group1.split('_')[-1]}_vs_{group2.split('_')[-1]}"

        # AUROC plots
        plot_auroc_over_time(
            df_auc,
            group1,
            group2,
            output_path=os.path.join(plot_dir, f'classification_auroc_{safe_comp_name}.png')
        )

        plot_auroc_with_significance(
            df_auc,
            group1,
            group2,
            alpha=ALPHA,
            output_path=os.path.join(plot_dir, f'classification_auroc_with_pvalues_{safe_comp_name}.png')
        )

        # Embryo-level signed margin heatmap with genotype emphasis
        plot_signed_margin_heatmap(
            df_embryo_probs,
            df_penetrance,
            group1,
            group2,
            output_path=os.path.join(plot_dir, f'embryo_signed_margin_heatmap_{safe_comp_name}.png')
        )

        # Signed margin trajectory plot
        if 'signed_margin' in df_embryo_probs.columns:
            plot_signed_margin_trajectories(
                df_embryo_probs,
                df_penetrance,
                group1,
                group2,
                max_embryos=30,
                output_path=os.path.join(plot_dir, f'embryo_signed_margin_trajectories_{safe_comp_name}.png')
            )

        # Save embryo-level data
        if not df_embryo_probs.empty:
            embryo_probs_path = os.path.join(data_dir, f'embryo_predictions_{safe_comp_name}.csv')
            df_embryo_probs.to_csv(embryo_probs_path, index=False)
            print(f"  Saved embryo predictions: {embryo_probs_path}")

        if not df_penetrance.empty:
            penetrance_path = os.path.join(data_dir, f'embryo_penetrance_{safe_comp_name}.csv')
            df_penetrance.to_csv(penetrance_path, index=False)
            print(f"  Saved penetrance metrics: {penetrance_path}")

    # Combine all results and save
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        results_path = os.path.join(data_dir, 'classification_results_all_comparisons.csv')
        combined_results.to_csv(results_path, index=False)
        print(f"\n{'='*80}")
        print(f"All results saved: {results_path}")
        print(f"Plots saved: {plot_dir}")

        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL COMPARISONS")
        print(f"{'='*80}")
        for comp in combined_results['comparison'].unique():
            comp_data = combined_results[combined_results['comparison'] == comp]
            sig_count = (comp_data['pval'] < ALPHA).sum()
            first_sig = comp_data[comp_data['pval'] < ALPHA].sort_values('time_bin')
            if len(first_sig) > 0:
                onset = first_sig.iloc[0]['time_bin']
                auroc = first_sig.iloc[0]['AUROC_obs']
                print(f"\n{comp}:")
                print(f"  Onset: {onset} hpf (AUROC={auroc:.3f})")
                print(f"  Significant bins: {sig_count}/{len(comp_data)}")
            else:
                print(f"\n{comp}:")
                print(f"  No significant onset detected")
    else:
        print(f"\n⚠ No valid comparisons completed for {genotype_label}")

print("\n" + "="*80)
print("CLASSIFICATION ANALYSIS COMPLETE")
print("="*80)
