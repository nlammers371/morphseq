import pandas as pd
import os
import numpy as np
import seaborn as sns 
import plotly.express as px
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.cm as cm
from pathlib import Path
from joblib import Parallel, delayed


# Use the parent directory of this file for results
# results_dir = os.getcwd()
results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251013"
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

# Bootstrap configuration (defaults kept tiny for quick smoke-tests)
BOOTSTRAP_MAX_ITERATIONS = int(os.environ.get("MORPHSEQ_BOOTSTRAP_MAX", 300))
BOOTSTRAP_PERMUTATIONS = int(os.environ.get("MORPHSEQ_BOOTSTRAP_PERMUTATIONS", 150))
BOOTSTRAP_USE_ADAPTIVE = os.environ.get("MORPHSEQ_BOOTSTRAP_ADAPTIVE", "1") != "0"
BOOTSTRAP_CONVERGENCE_WINDOW = int(os.environ.get("MORPHSEQ_BOOTSTRAP_WINDOW", 20))
BOOTSTRAP_CONVERGENCE_TOL = float(os.environ.get("MORPHSEQ_BOOTSTRAP_TOL", 1.0))

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



import numpy as np
import pandas as pd

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


def get_z_columns(df, z_cols=None, suffix="_binned"):
    
    """
    Identify latent (embedding) columns for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (already binned by embryo/time).
    z_cols : list or None
        Optional explicit list. If None, automatically detect by suffix or 'z_mu_b' pattern.
    suffix : str
        Column suffix used in binning (default '_binned').

    Returns
    -------
    list
        Names of latent columns.
    """
    if z_cols is None:
        z_cols = [c for c in df.columns if c.endswith(suffix) or "z_mu_b" in c]
    if not z_cols:
        raise ValueError("No latent columns detected for analysis.")
    return z_cols

from itertools import combinations
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

# -- helper stats --

def energy_distance(X, Y):
    XY = cdist(X, Y).mean()
    XX = cdist(X, X).mean()
    YY = cdist(Y, Y).mean()
    return 2*XY - XX - YY

def energy_perm_test(X, Y, n_perm=500, rng=None):
    rng = np.random.default_rng(rng)
    obs = energy_distance(X, Y)
    Z = np.vstack([X, Y])
    nx = len(X)
    perm_stats = []
    for _ in range(n_perm):
        rng.shuffle(Z)
        perm_stats.append(energy_distance(Z[:nx], Z[nx:]))
    p = (np.sum(perm_stats >= obs) + 1) / (n_perm + 1)
    return obs, p

def hotellings_T2(X, Y):
    """
    Compute Hotelling's T-squared statistic with robust covariance estimation.
    Raises ValueError if data contains NaN or infinite values.
    """
    # Check for NaN or infinite values
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("Data contains NaN or infinite values")

    n, m = len(X), len(Y)
    mean_diff = X.mean(0) - Y.mean(0)
    Sx = LedoitWolf().fit(X).covariance_
    Sy = LedoitWolf().fit(Y).covariance_
    Sp = ((n-1)*Sx + (m-1)*Sy) / (n+m-2)
    invSp = np.linalg.pinv(Sp)
    return (n*m)/(n+m) * float(mean_diff @ invSp @ mean_diff)

# -- main analysis --

def _process_single_time_bin(time_data, group_col, z_cols, tests, n_perm, min_n, random_state_seed):
    """
    Helper function to process a single time bin. Designed to be called in parallel by joblib.
    """
    time_val, df_t = time_data
    rng = np.random.default_rng(random_state_seed)
    results_for_bin = []

    groups = sorted(df_t[group_col].dropna().unique())
    for g1, g2 in combinations(groups, 2):
        X = df_t.loc[df_t[group_col] == g1, z_cols].values
        Y = df_t.loc[df_t[group_col] == g2, z_cols].values
        if len(X) < min_n or len(Y) < min_n:
            continue

        rec = dict(time_bin=time_val, group1=g1, group2=g2)

        if "energy" in tests:
            stat, p = energy_perm_test(X, Y, n_perm=n_perm, rng=rng)
            rec.update(energy_stat=stat, energy_p=p)

        if "hotelling" in tests:
            try:
                T2 = hotellings_T2(X, Y)
                Z = np.vstack([X, Y])
                nx = len(X)
                perm_stats = [hotellings_T2(Z[:nx], Z[nx:]) for _ in range(n_perm) if (rng.shuffle(Z) is None)]
                p = (np.sum(np.array(perm_stats) >= T2) + 1) / (n_perm + 1)
                rec.update(hotelling_T2=T2, hotelling_p=p)
            except (np.linalg.LinAlgError, ValueError):
                rec.update(hotelling_T2=np.nan, hotelling_p=np.nan)

        results_for_bin.append(rec)
    return results_for_bin

def run_distribution_tests(
    df_binned,
    group_col="genotype",
    time_col="time_bin",
    z_cols=None,
    tests=("energy",),
    n_perm=500,
    min_n=4,
    random_state=None
):
    """
    Run pairwise group comparisons per time bin in parallel.
    Parameters
    ----------
    df_binned : pd.DataFrame
        Output of bin_by_embryo_time().
    group_col : str
        Column specifying experimental group (e.g., genotype).
    time_col : str
        Column specifying time bins.
    z_cols : list or None
        Columns to analyze (auto-detected if None).
    tests : tuple
        Which tests to run.
    n_perm : int
        Number of permutations for nonparametric tests.
    min_n : int
        Minimum per-group sample size per bin.
    random_state : int or None
        RNG seed for reproducibility.
    Returns
    -------
    pd.DataFrame
        One row per (time_bin, group1, group2, test).
    """

    if z_cols is None:
        z_cols = get_z_columns(df_binned)

    # Main RNG for generating seeds for child processes
    rng = np.random.default_rng(random_state)

    # Group data by time bin to prepare for parallel processing
    grouped_data = list(df_binned.groupby(time_col))
    
    # Generate independent random seeds for each parallel job
    # This is crucial for reproducibility and statistical independence.
    child_seeds = rng.integers(0, 2**32 - 1, len(grouped_data))

    # Use joblib to run the analysis for each time bin in parallel
    # n_jobs=-1 automatically uses all available CPU cores
    print(f"Running pairwise tests in parallel on {os.cpu_count()} cores...")
    results_lists = Parallel(n_jobs=-1, verbose=10)(
        delayed(_process_single_time_bin)(
            time_data, group_col, z_cols, tests, n_perm, min_n, seed
        ) for time_data, seed in zip(grouped_data, child_seeds)
    )

    # The result is a list of lists; flatten it
    results = [item for sublist in results_lists for item in sublist if sublist]

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================================
# BOOTSTRAP AND RESAMPLING FUNCTIONS FOR ROBUSTNESS ANALYSIS
# ============================================================================

def find_onset_from_results(results_df, test_col="energy_p", alpha=0.05, K=2):
    """
    Find phenotype onset from test results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_distribution_tests
    test_col : str
        P-value column to use
    alpha : float
        Significance threshold
    K : int
        Number of consecutive significant bins required

    Returns
    -------
    float or np.nan
        Time bin of phenotype onset
    """
    if results_df is None or 'time_bin' not in results_df.columns:
        return np.nan

    sub = results_df.sort_values("time_bin")
    if sub.empty:
        return np.nan
    rej, p_corr, _, _ = multipletests(sub[test_col].values, alpha=alpha, method="fdr_bh")
    consec = 0
    times = sub["time_bin"].values
    for i, sig in enumerate(rej):
        consec = consec + 1 if sig else 0
        if consec >= K:
            return times[i - (K - 1)]
    return np.nan


def _resample_embryos_loo(df_binned, group_col):
    """Yield leave-one-out resamples without turning ``resample_embryos`` into a generator."""
    all_ids = df_binned["embryo_id"].unique()
    for drop_id in all_ids:
        yield df_binned.loc[df_binned["embryo_id"] != drop_id].copy()


def resample_embryos(df_binned, group_col, strategy="bootstrap", rate=1.0, rng=None):
    """
    Returns a resampled dataframe at the embryo (cluster) level.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data
    group_col : str
        Column specifying groups (e.g., genotype)
    strategy : str
        'bootstrap' - sample with replacement
        'subsample' - sample without replacement at rate
        'loo' - yields generator for leave-one-out
    rate : float
        Subsampling rate (for subsample strategy)
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    pd.DataFrame or generator
        Resampled dataframe(s)
    """
    if strategy == "loo":
        return _resample_embryos_loo(df_binned, group_col)

    rng = np.random.default_rng(rng)
    groups = df_binned[group_col].dropna().unique()
    by_group = {g: df_binned.loc[df_binned[group_col]==g, "embryo_id"].unique() for g in groups}

    rows = []
    for g in groups:
        ids = by_group[g]
        if len(ids) == 0:
            continue  # Skip groups with no embryos

        if strategy == "bootstrap":
            n = len(ids)
            chosen = rng.choice(ids, size=n, replace=True)
        elif strategy == "subsample":
            m = int(np.ceil(rate * len(ids)))
            m = max(1, min(m, len(ids)))
            chosen = rng.choice(ids, size=m, replace=False)
        else:
            raise ValueError("strategy must be 'bootstrap', 'subsample', or 'loo'")
        rows.append(df_binned[df_binned["embryo_id"].isin(chosen) & (df_binned[group_col]==g)])

    if len(rows) == 0:
        return df_binned.iloc[:0].copy()  # Return empty DataFrame with same structure

    return pd.concat(rows, ignore_index=True)


def bootstrap_onset(
    df_binned,
    run_distribution_tests_fn,
    group_col="genotype",
    test_col="energy_p",
    n_boot=300,
    n_perm=300,
    alpha=0.05,
    K=2,
    strategy="bootstrap",
    rate=1.0,
    rng=None,
    adaptive=False,
    convergence_window=20,
    convergence_tol=1.0
):
    """
    Bootstrap analysis of phenotype onset detection.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data
    run_distribution_tests_fn : callable
        Function to run distribution tests
    group_col : str
        Column specifying groups
    test_col : str
        P-value column to use
    n_boot : int
        Number of bootstrap iterations
    n_perm : int
        Permutations per test (reduced for speed)
    alpha : float
        Significance threshold
    K : int
        Consecutive significant bins required
    strategy : str
        'bootstrap' or 'subsample'
    rate : float
        Subsampling rate
    rng : np.random.Generator
        Random number generator
    adaptive : bool
        If True, stop early once the CI stabilizes within tolerance
    convergence_window : int
        Number of most-recent iterations to track for adaptive stopping
    convergence_tol : float
        Maximum CI width (in time-bin units) before declaring convergence

    Returns
    -------
    dict
        Bootstrap statistics including median, CI, effective sample size, and iteration metadata
    """
    rng = np.random.default_rng(rng)
    onsets = []
    valid_onsets = []
    iterations_run = 0
    adaptive_stop = False

    for b in range(n_boot):
        if strategy not in ["bootstrap", "subsample"]:
            raise ValueError(f"Invalid strategy for bootstrap_onset: {strategy}")

        boot_df = resample_embryos(df_binned, group_col, strategy=strategy, rate=rate, rng=rng)

        if not isinstance(boot_df, pd.DataFrame):
            raise TypeError(f"resample_embryos returned {type(boot_df)}, expected DataFrame. Check strategy={strategy}, type={type(boot_df)}")

        if 'embryo_id' not in boot_df.columns:
            raise ValueError("boot_df missing embryo_id column")

        res = run_distribution_tests_fn(boot_df, group_col=group_col, n_perm=n_perm, random_state=rng.integers(0, 1e9))
        onset = find_onset_from_results(res, test_col=test_col, alpha=alpha, K=K)
        onsets.append(onset)
        iterations_run += 1

        if not np.isnan(onset):
            valid_onsets.append(onset)

        if adaptive and len(valid_onsets) >= convergence_window:
            window_onsets = valid_onsets[-convergence_window:]
            ci_low, ci_high = np.percentile(window_onsets, [2.5, 97.5])
            if (ci_high - ci_low) <= convergence_tol:
                adaptive_stop = True
                break

    onsets = np.array(onsets, dtype=float)
    onsets = onsets[~np.isnan(onsets)]

    if len(onsets) == 0:
        return {
            "onset_median": np.nan,
            "onset_low": np.nan,
            "onset_high": np.nan,
            "n_eff": 0,
            "strategy": strategy,
            "rate": rate,
            "all_onsets": [],
            "iterations_run": iterations_run,
            "adaptive_stop": adaptive_stop
        }

    return {
        "onset_median": float(np.median(onsets)),
        "onset_low": float(np.percentile(onsets, 2.5)),
        "onset_high": float(np.percentile(onsets, 97.5)),
        "n_eff": int(len(onsets)),
        "strategy": strategy,
        "rate": rate,
        "all_onsets": onsets.tolist(),
        "iterations_run": iterations_run,
        "adaptive_stop": adaptive_stop
    }


def loo_onset_range(
    df_binned,
    run_distribution_tests_fn,
    group_col="genotype",
    test_col="energy_p",
    alpha=0.05,
    K=2
):
    """
    Leave-one-out analysis to find min/max onset times.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data
    run_distribution_tests_fn : callable
        Function to run distribution tests
    group_col : str
        Column specifying groups
    test_col : str
        P-value column to use
    alpha : float
        Significance threshold
    K : int
        Consecutive significant bins required

    Returns
    -------
    tuple
        (min_onset, max_onset) across LOO iterations
    """
    onsets = []
    for df_loo in resample_embryos(df_binned, group_col, strategy="loo"):
        res = run_distribution_tests_fn(df_loo, group_col=group_col)
        onsets.append(find_onset_from_results(res, test_col=test_col, alpha=alpha, K=K))

    onsets = np.array(onsets, dtype=float)
    onsets = onsets[~np.isnan(onsets)]

    return (
        float(np.min(onsets)) if len(onsets) else np.nan,
        float(np.max(onsets)) if len(onsets) else np.nan,
        onsets.tolist()
    )


from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

def summarize_test_results(results_df, test_col="energy_p", alpha=0.05, consecutive=2):
    """
    FDR-correct p-values across time bins and compute earliest
    consecutive-significant onset for each group pair.

    Returns
    -------
    summary : pd.DataFrame
        Columns: group1, group2, onset_bin, n_significant_bins, etc.
    """

    summaries = []
    for (g1, g2), sub in results_df.groupby(["group1", "group2"]):
        sub = sub.sort_values("time_bin")
        # FDR correction
        rej, p_corr, _, _ = multipletests(sub[test_col], alpha=alpha, method="fdr_bh")
        sub = sub.assign(p_corr=p_corr, reject=rej)

        # detect first consecutive-significant stretch
        consec = 0
        onset = None
        for t, sig in zip(sub["time_bin"], sub["reject"]):
            consec = consec + 1 if sig else 0
            if consec >= consecutive:
                onset = t - (consecutive-1)*2  # subtract bins if needed
                break

        summaries.append(dict(
            group1=g1,
            group2=g2,
            onset_bin=onset,
            n_sig_bins=rej.sum(),
            first_sig_bin=sub.loc[sub["reject"], "time_bin"].min() if rej.any() else None
        ))

    return pd.DataFrame(summaries)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_energy_distance_over_time(results_df, output_path=None):
    """
    Plot energy distance statistic over time for each pairwise comparison.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_distribution_tests()
    output_path : str or None
        Path to save figure. If None, displays interactively.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for (g1, g2), sub in results_df.groupby(['group1', 'group2']):
        sub = sub.sort_values('time_bin')
        label = f"{g1} vs {g2}"
        ax.plot(sub['time_bin'], sub['energy_stat'], marker='o', label=label, linewidth=2)

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('Energy Distance', fontsize=12)
    ax.set_title('Energy Distance Between Genotypes Over Development', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_pvalues_over_time(results_df, test_col='energy_p', alpha=0.05, output_path=None):
    """
    Plot p-values over time with significance threshold.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_distribution_tests()
    test_col : str
        Column containing p-values to plot
    alpha : float
        Significance threshold to mark on plot
    output_path : str or None
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for (g1, g2), sub in results_df.groupby(['group1', 'group2']):
        sub = sub.sort_values('time_bin')
        label = f"{g1} vs {g2}"
        ax.plot(sub['time_bin'], sub[test_col], marker='o', label=label, linewidth=2)

    # Add significance threshold line
    ax.axhline(y=alpha, color='red', linestyle='--', linewidth=2, label=f'α = {alpha}')

    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('P-value', fontsize=12)
    ax.set_title('Statistical Significance of Genotype Differences Over Time', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_phenotype_onset_heatmap(summary_df, output_path=None):
    """
    Create heatmap showing phenotype emergence onset times.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from summarize_test_results()
    output_path : str or None
        Path to save figure
    """
    # Create pivot table for heatmap
    pivot_data = summary_df.pivot_table(
        index='group1',
        columns='group2',
        values='onset_bin',
        aggfunc='first'
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlGn_r', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_data.index)

    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{int(val)}',
                             ha="center", va="center", color="black", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Onset Time (hpf)', rotation=270, labelpad=20, fontsize=12)

    ax.set_title('Phenotype Emergence Onset Times', fontsize=14, fontweight='bold')
    ax.set_xlabel('Group 2', fontsize=12)
    ax.set_ylabel('Group 1', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_significance_timeline(results_df, test_col='energy_p', alpha=0.05, output_path=None):
    """
    Plot timeline showing which bins are significant for each comparison.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_distribution_tests()
    test_col : str
        Column containing p-values
    alpha : float
        Significance threshold
    output_path : str or None
        Path to save figure
    """
    from statsmodels.stats.multitest import multipletests

    fig, axes = plt.subplots(len(results_df.groupby(['group1', 'group2'])), 1,
                              figsize=(12, 3 * len(results_df.groupby(['group1', 'group2']))),
                              sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for idx, ((g1, g2), sub) in enumerate(results_df.groupby(['group1', 'group2'])):
        sub = sub.sort_values('time_bin')

        # FDR correction
        rej, p_corr, _, _ = multipletests(sub[test_col], alpha=alpha, method='fdr_bh')

        ax = axes[idx]

        # Plot raw p-values
        ax.plot(sub['time_bin'], sub[test_col], 'o-', color='steelblue', label='Raw p-value', linewidth=2)

        # Plot corrected p-values
        ax.plot(sub['time_bin'], p_corr, 's-', color='orange', label='FDR-corrected p-value', linewidth=2)

        # Mark significant bins
        sig_bins = sub.loc[rej, 'time_bin']
        if len(sig_bins) > 0:
            ax.scatter(sig_bins, [alpha/10]*len(sig_bins), color='red', s=100,
                      marker='v', label='Significant', zorder=5)

        ax.axhline(y=alpha, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_yscale('log')
        ax.set_ylabel('P-value', fontsize=10)
        ax.set_title(f'{g1} vs {g2}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Time (hpf)', fontsize=12)
    fig.suptitle('Phenotype Emergence Timeline with FDR Correction',
                 fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_all_results(results_df, summary_df, plot_dir, test_col='energy_p', alpha=0.05):
    """
    Generate all plots and save to directory.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from run_distribution_tests()
    summary_df : pd.DataFrame
        Output from summarize_test_results()
    plot_dir : str
        Directory to save plots
    test_col : str
        P-value column to use
    alpha : float
        Significance threshold
    """
    os.makedirs(plot_dir, exist_ok=True)

    print("\nGenerating plots...")

    # 1. Energy distance over time
    plot_energy_distance_over_time(
        results_df,
        output_path=os.path.join(plot_dir, 'energy_distance_over_time.png')
    )

    # 2. P-values over time
    plot_pvalues_over_time(
        results_df,
        test_col=test_col,
        alpha=alpha,
        output_path=os.path.join(plot_dir, 'pvalues_over_time.png')
    )

    # 3. Onset heatmap (only if we have onset data)
    if not summary_df['onset_bin'].isna().all():
        plot_phenotype_onset_heatmap(
            summary_df,
            output_path=os.path.join(plot_dir, 'phenotype_onset_heatmap.png')
        )
    else:
        print("No onset data to plot heatmap")

    # 4. Significance timeline
    plot_significance_timeline(
        results_df,
        test_col=test_col,
        alpha=alpha,
        output_path=os.path.join(plot_dir, 'significance_timeline.png')
    )

    print(f"\nAll plots saved to: {plot_dir}")


# ============================================================================
# BOOTSTRAP PLOTTING FUNCTIONS
# ============================================================================

def plot_onset_with_confidence_intervals(bootstrap_results_df, summary_df, output_path=None):
    """
    Bar chart showing onset times with bootstrap confidence intervals.

    Parameters
    ----------
    bootstrap_results_df : pd.DataFrame
        DataFrame with bootstrap results (group1, group2, onset_median, onset_low, onset_high)
    summary_df : pd.DataFrame
        DataFrame with point estimates from summarize_test_results
    output_path : str or None
        Path to save figure
    """
    # Merge bootstrap and point estimates
    merged = bootstrap_results_df.merge(
        summary_df[['group1', 'group2', 'onset_bin']],
        on=['group1', 'group2'],
        how='left'
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(merged))
    labels = [f"{row['group1']}\nvs\n{row['group2']}" for _, row in merged.iterrows()]

    # Plot bootstrap median with error bars
    yerr_low = merged['onset_median'] - merged['onset_low']
    yerr_high = merged['onset_high'] - merged['onset_median']

    ax.bar(x_pos, merged['onset_median'], alpha=0.7, label='Bootstrap Median', color='steelblue')
    ax.errorbar(x_pos, merged['onset_median'], yerr=[yerr_low, yerr_high],
                fmt='none', ecolor='black', capsize=5, linewidth=2, label='95% CI')

    # Add point estimates as markers
    ax.scatter(x_pos, merged['onset_bin'], color='red', s=100, zorder=5,
              marker='D', label='Point Estimate', edgecolors='black', linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Onset Time (hpf)', fontsize=12)
    ax.set_title('Phenotype Onset with Bootstrap Confidence Intervals', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_bootstrap_distributions(bootstrap_results_df, output_path=None):
    """
    Histograms of bootstrap onset distributions for each comparison.

    Parameters
    ----------
    bootstrap_results_df : pd.DataFrame
        DataFrame with bootstrap results including 'all_onsets' column
    output_path : str or None
        Path to save figure
    """
    n_comparisons = len(bootstrap_results_df)
    fig, axes = plt.subplots(n_comparisons, 1, figsize=(12, 4 * n_comparisons))

    if n_comparisons == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(bootstrap_results_df.iterrows()):
        ax = axes[idx]

        onsets = row['all_onsets']
        if len(onsets) == 0:
            ax.text(0.5, 0.5, 'No significant onset detected',
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"{row['group1']} vs {row['group2']}", fontsize=11, fontweight='bold')
            continue

        # Histogram
        ax.hist(onsets, bins=20, alpha=0.7, color='steelblue', edgecolor='black')

        # Mark median and CI
        ax.axvline(row['onset_median'], color='red', linewidth=2, linestyle='-', label='Median')
        ax.axvline(row['onset_low'], color='orange', linewidth=2, linestyle='--', label='2.5%')
        ax.axvline(row['onset_high'], color='orange', linewidth=2, linestyle='--', label='97.5%')

        ax.set_xlabel('Onset Time (hpf)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f"{row['group1']} vs {row['group2']} (n={row['n_eff']} bootstrap samples)",
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_loo_sensitivity(loo_results_df, summary_df, output_path=None):
    """
    Range plot showing leave-one-out sensitivity analysis.

    Parameters
    ----------
    loo_results_df : pd.DataFrame
        DataFrame with LOO results (group1, group2, loo_min, loo_max)
    summary_df : pd.DataFrame
        DataFrame with point estimates
    output_path : str or None
        Path to save figure
    """
    merged = loo_results_df.merge(
        summary_df[['group1', 'group2', 'onset_bin']],
        on=['group1', 'group2'],
        how='left'
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(merged))
    labels = [f"{row['group1']}\nvs\n{row['group2']}" for _, row in merged.iterrows()]

    # Plot LOO range as error bars
    for i, row in enumerate(merged.itertuples()):
        ax.plot([i, i], [row.loo_min, row.loo_max], 'o-', linewidth=3,
               markersize=8, color='steelblue', label='LOO Range' if i == 0 else '')

    # Add point estimate
    ax.scatter(x_pos, merged['onset_bin'], color='red', s=100, zorder=5,
              marker='D', label='Point Estimate', edgecolors='black', linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Onset Time (hpf)', fontsize=12)
    ax.set_title('Leave-One-Out Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_subsampling_stability(subsampling_results_df, output_path=None):
    """
    Line plots showing onset stability across subsampling rates.

    Parameters
    ----------
    subsampling_results_df : pd.DataFrame
        DataFrame with subsampling results (group1, group2, rate, onset_median, onset_low, onset_high)
    output_path : str or None
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for (g1, g2), sub in subsampling_results_df.groupby(['group1', 'group2']):
        sub = sub.sort_values('rate')
        label = f"{g1} vs {g2}"

        ax.plot(sub['rate'] * 100, sub['onset_median'], marker='o', linewidth=2, label=label)
        ax.fill_between(sub['rate'] * 100, sub['onset_low'], sub['onset_high'], alpha=0.2)

    ax.set_xlabel('Subsampling Rate (%)', fontsize=12)
    ax.set_ylabel('Onset Time (hpf)', fontsize=12)
    ax.set_title('Onset Stability Across Subsampling Rates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    return fig


def plot_bootstrap_summary_panel(bootstrap_results_df, loo_results_df, summary_df, output_path=None):
    """
    Multi-panel overview of all robustness metrics.

    Parameters
    ----------
    bootstrap_results_df : pd.DataFrame
        Bootstrap results
    loo_results_df : pd.DataFrame
        LOO results
    summary_df : pd.DataFrame
        Point estimates
    output_path : str or None
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Merge all data
    merged = bootstrap_results_df.merge(
        loo_results_df[['group1', 'group2', 'loo_min', 'loo_max']],
        on=['group1', 'group2'],
        how='left'
    ).merge(
        summary_df[['group1', 'group2', 'onset_bin']],
        on=['group1', 'group2'],
        how='left'
    )

    # Ensure numeric columns are floats; drop rows with no data at all
    numeric_cols = ['onset_bin', 'onset_median', 'onset_low', 'onset_high', 'loo_min', 'loo_max']
    for col in numeric_cols:
        if col in merged:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    merged = merged.dropna(subset=['onset_bin', 'onset_median'], how='all')

    if merged.empty:
        print("No data available to plot bootstrap summary panel.")
        plt.close(fig)
        return fig

    x_pos = np.arange(len(merged))
    labels = [f"{row['group1']}\nvs\n{row['group2']}" for _, row in merged.iterrows()]

    # Panel 1: Point estimates
    ax1 = axes[0]
    ax1.bar(x_pos, merged['onset_bin'], alpha=0.7, color='steelblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Onset Time (hpf)', fontsize=10)
    ax1.set_title('Point Estimates', fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')

    # Panel 2: Bootstrap CI
    ax2 = axes[1]
    yerr_low = merged['onset_median'] - merged['onset_low']
    yerr_high = merged['onset_high'] - merged['onset_median']
    ax2.bar(x_pos, merged['onset_median'], alpha=0.7, color='steelblue')
    ax2.errorbar(x_pos, merged['onset_median'], yerr=[yerr_low, yerr_high],
                fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('Onset Time (hpf)', fontsize=10)
    ax2.set_title('Bootstrap 95% CI', fontsize=11, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # Panel 3: LOO range
    ax3 = axes[2]
    for i, row in enumerate(merged.itertuples()):
        if not np.isnan(row.loo_min) and not np.isnan(row.loo_max):
            ax3.plot([i, i], [row.loo_min, row.loo_max], 'o-', linewidth=3,
                   markersize=8, color='steelblue')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel('Onset Time (hpf)', fontsize=10)
    ax3.set_title('Leave-One-Out Range', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    fig.suptitle('Robustness Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
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

aggregate_robustness = []

for genotype_label, genotype_values in GENOTYPE_GROUPS.items():
    print("\n" + "="*80)
    print(f"ANALYSIS FOR {genotype_label.upper()}")
    print("="*80)

    data_dir = os.path.join(data_dir_base, genotype_label)
    plot_dir = os.path.join(plot_dir_base, genotype_label)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    df_family = combined_df[combined_df['genotype'].isin(genotype_values)].copy()
    if df_family.empty:
        print(f"No data found for genotype group '{genotype_label}', skipping.")
        continue

    print(f"\nFiltered to {genotype_label} genotypes: {len(df_family)} rows")
    print(f"Genotype distribution:\n{df_family['genotype'].value_counts()}")

    # 1️⃣ bin embeddings per embryo/time
    print("\n=== DEBUGGING: Checking for NaNs in input data ===")
    print(f"NaNs in df_{genotype_label}: {df_family.isna().sum().sum()}")
    print(f"NaNs in predicted_stage_hpf: {df_family['predicted_stage_hpf'].isna().sum()}")

    # Check latent columns for NaNs
    z_cols_check = [c for c in df_family.columns if "z_mu_b" in c]
    print(f"Number of latent columns: {len(z_cols_check)}")
    for col in z_cols_check[:5]:  # Show first 5
        print(f"  {col}: {df_family[col].isna().sum()} NaNs")

    df_binned = bin_by_embryo_time(df_family, time_col="predicted_stage_hpf")
    print(f"\nBinned data: {len(df_binned)} rows")

    # Check binned data for NaNs
    print("\n=== DEBUGGING: Checking for NaNs in binned data ===")
    binned_z_cols = [c for c in df_binned.columns if "_binned" in c]
    print(f"Number of binned latent columns: {len(binned_z_cols)}")
    nan_counts = df_binned[binned_z_cols].isna().sum()
    if nan_counts.sum() > 0:
        print(f"Total NaNs in binned latent columns: {nan_counts.sum()}")
        print("Columns with NaNs (first 10):")
        for col in nan_counts[nan_counts > 0].index[:10]:
            print(f"  {col}: {nan_counts[col]} NaNs")
    else:
        print("No NaNs found in binned latent columns")

    # Check for infinite values
    inf_check = np.isinf(df_binned[binned_z_cols].values).sum()
    print(f"Infinite values in binned data: {inf_check}")

    # Drop rows with NaN in binned latent columns
    if nan_counts.sum() > 0:
        dropped = df_binned[binned_z_cols].isna().any(axis=1).sum()
        print(f"\nDropping {dropped} rows with NaNs in latent columns")
        df_binned = df_binned.dropna(subset=binned_z_cols)
        print(f"Remaining rows after dropping NaNs: {len(df_binned)}")

    # TEST SYMMETRY FIRST with real data
    print("\n=== TESTING ENERGY DISTANCE SYMMETRY WITH REAL DATA ===")
    if df_binned.empty:
        print("Binned dataframe is empty; skipping symmetry test.")
    else:
        test_time_bin = df_binned['time_bin'].mode()[0]
        test_df = df_binned[df_binned['time_bin'] == test_time_bin]

        z_test_cols = [c for c in test_df.columns if "_binned" in c]
        present_groups = [g for g in genotype_values if g in test_df['genotype'].unique()]

        if len(present_groups) >= 2:
            g1, g2 = present_groups[:2]
            data1 = test_df[test_df['genotype'] == g1][z_test_cols].values
            data2 = test_df[test_df['genotype'] == g2][z_test_cols].values

            if len(data1) > 0 and len(data2) > 0:
                print(f"Time bin: {test_time_bin}")
                print(f"{g1} samples: {len(data1)}")
                print(f"{g2} samples: {len(data2)}")

                energy_12 = energy_distance(data1, data2)
                energy_21 = energy_distance(data2, data1)

                print(f"\nenergy_distance({g1}, {g2}) = {energy_12:.10f}")
                print(f"energy_distance({g2}, {g1}) = {energy_21:.10f}")
                print(f"Difference: {abs(energy_12 - energy_21):.10e}")
                print(f"Symmetric: {np.isclose(energy_12, energy_21)}")
            else:
                print("Not enough samples in the most common time bin to test symmetry.")
        else:
            print("Not enough genotype groups present in the most common time bin to test symmetry.")

    # 2️⃣ run pairwise distribution tests
    results_df = run_distribution_tests(df_binned, group_col="genotype")
    print(f"\nTest results: {len(results_df)} comparisons")

    # Debug: Check for duplicate comparisons
    print("\n=== DEBUGGING: Checking for duplicate comparisons ===")
    unique_pairs = results_df[['group1', 'group2']].drop_duplicates()
    print(f"Unique pairs: {len(unique_pairs)}")
    print(unique_pairs)

    for _, row in unique_pairs.iterrows():
        g1, g2 = row['group1'], row['group2']
        reverse = results_df[(results_df['group1'] == g2) & (results_df['group2'] == g1)]
        if len(reverse) > 0:
            print(f"WARNING: Found both {g1} vs {g2} AND {g2} vs {g1}")
            forward = results_df[(results_df['group1'] == g1) & (results_df['group2'] == g2)]
            merged = forward.merge(reverse, on='time_bin', suffixes=('_fwd', '_rev'))
            if len(merged) > 0:
                energy_diff = (merged['energy_stat_fwd'] - merged['energy_stat_rev']).abs()
                print(f"  Max energy stat difference: {energy_diff.max():.6f}")
                print(f"  Mean energy stat difference: {energy_diff.mean():.6f}")

    # 3️⃣ summarize onset
    summary_df = summarize_test_results(results_df, test_col="energy_p")

    print("\n=== PHENOTYPE EMERGENCE SUMMARY ===")
    print(summary_df)

    # 4️⃣ generate all plots
    plot_all_results(results_df, summary_df, plot_dir, test_col='energy_p', alpha=0.05)

    # Save results to CSV
    results_output_path = os.path.join(data_dir, 'distribution_tests.csv')
    summary_output_path = os.path.join(data_dir, 'phenotype_emergence_summary.csv')

    results_df.to_csv(results_output_path, index=False)
    summary_df.to_csv(summary_output_path, index=False)

    print(f"\n=== RESULTS SAVED ===")
    print(f"Full test results: {results_output_path}")
    print(f"Summary: {summary_output_path}")
    print(f"Plots: {plot_dir}")

    # Print detailed summary
    print("\n=== DETAILED SUMMARY ===")
    for _, row in summary_df.iterrows():
        print(f"\n{row['group1']} vs {row['group2']}:")
        if pd.notna(row['onset_bin']):
            print(f"  Phenotype onset: {row['onset_bin']} hpf")
        else:
            print("  No consistent phenotype emergence detected")
        print(f"  First significant bin: {row['first_sig_bin']} hpf")
        print(f"  Total significant bins: {row['n_sig_bins']}")

    print("\n=== ANALYSIS COMPLETE ===")

    # ============================================================================
    # 5️⃣ BOOTSTRAP ROBUSTNESS ANALYSIS
    # ============================================================================

    print("\n" + "="*80)
    print("BOOTSTRAP ROBUSTNESS ANALYSIS")
    print("="*80)

    unique_comparisons = results_df[['group1', 'group2']].drop_duplicates()

    print(f"\nRunning bootstrap analysis for {len(unique_comparisons)} comparisons...")
    print("This may take several minutes...")

    bootstrap_results = []
    loo_results = []
    subsampling_results = []

    for idx, (_, row) in enumerate(unique_comparisons.iterrows(), 1):
        g1, g2 = row['group1'], row['group2']
        print(f"\n[{idx}/{len(unique_comparisons)}] Analyzing: {g1} vs {g2}")

        comp_df = df_binned[df_binned['genotype'].isin([g1, g2])].copy()

        bootstrap_kwargs = dict(
            group_col='genotype',
            test_col='energy_p',
            n_boot=BOOTSTRAP_MAX_ITERATIONS,
            n_perm=BOOTSTRAP_PERMUTATIONS,
            alpha=0.05,
            K=2,
            rng=42,
            adaptive=BOOTSTRAP_USE_ADAPTIVE,
            convergence_window=BOOTSTRAP_CONVERGENCE_WINDOW,
            convergence_tol=BOOTSTRAP_CONVERGENCE_TOL
        )

        print(f"  Running bootstrap (up to {bootstrap_kwargs['n_boot']} iterations)...")
        boot_result = bootstrap_onset(
            comp_df,
            run_distribution_tests,
            **bootstrap_kwargs
        )
        boot_result['group1'] = g1
        boot_result['group2'] = g2
        bootstrap_results.append(boot_result)
        if boot_result.get("adaptive_stop"):
            print(f"    Adaptive stop after {boot_result['iterations_run']} iterations (CI width ≤ {BOOTSTRAP_CONVERGENCE_TOL})")
        else:
            print(f"    Completed {boot_result['iterations_run']} iterations")
        print(f"    Median onset: {boot_result['onset_median']:.1f} hpf")
        print(f"    95% CI: [{boot_result['onset_low']:.1f}, {boot_result['onset_high']:.1f}] hpf")

        print(f"  Running leave-one-out analysis...")
        loo_min, loo_max, loo_all = loo_onset_range(
            comp_df,
            run_distribution_tests,
            group_col='genotype',
            test_col='energy_p',
            alpha=0.05,
            K=2
        )
        loo_results.append({
            'group1': g1,
            'group2': g2,
            'loo_min': loo_min,
            'loo_max': loo_max,
            'loo_range': loo_max - loo_min if not np.isnan(loo_min) else np.nan,
            'all_loo_onsets': loo_all
        })
        print(f"    LOO range: [{loo_min:.1f}, {loo_max:.1f}] hpf")

        print(f"  Running subsampling stability analysis...")
        for rate in [0.5, 0.75, 0.9, 1.0]:
            subsample_kwargs = {
                **bootstrap_kwargs,
                "strategy": 'subsample' if rate < 1.0 else 'bootstrap',
                "rate": rate
            }
            subsample_result = bootstrap_onset(
                comp_df,
                run_distribution_tests,
                **subsample_kwargs
            )
            subsample_result['group1'] = g1
            subsample_result['group2'] = g2
            subsampling_results.append(subsample_result)
            if subsample_result.get("adaptive_stop"):
                print(f"    (rate={rate:0.2f}) adaptive stop after {subsample_result['iterations_run']} iterations")
            else:
                print(f"    (rate={rate:0.2f}) completed {subsample_result['iterations_run']} iterations")

    bootstrap_df = pd.DataFrame(bootstrap_results)
    loo_df = pd.DataFrame(loo_results)
    subsampling_df = pd.DataFrame(subsampling_results)

    print("\n" + "="*80)
    print("BOOTSTRAP ANALYSIS COMPLETE")
    print("="*80)

    # ============================================================================
    # 6️⃣ GENERATE BOOTSTRAP PLOTS
    # ============================================================================

    print("\nGenerating bootstrap visualization plots...")
    plot_onset_with_confidence_intervals(
        bootstrap_df,
        summary_df,
        output_path=os.path.join(plot_dir, 'bootstrap_onset_with_ci.png')
    )
    plot_bootstrap_distributions(
        bootstrap_df,
        output_path=os.path.join(plot_dir, 'bootstrap_distributions.png')
    )
    plot_loo_sensitivity(
        loo_df,
        summary_df,
        output_path=os.path.join(plot_dir, 'loo_sensitivity.png')
    )
    plot_subsampling_stability(
        subsampling_df,
        output_path=os.path.join(plot_dir, 'subsampling_stability.png')
    )
    plot_bootstrap_summary_panel(
        bootstrap_df,
        loo_df,
        summary_df,
        output_path=os.path.join(plot_dir, 'bootstrap_summary_panel.png')
    )

    print(f"\nBootstrap plots saved to: {plot_dir}")

    # ============================================================================
    # 7️⃣ SAVE BOOTSTRAP RESULTS
    # ============================================================================

    print("\nSaving bootstrap results...")

    bootstrap_summary_path = os.path.join(data_dir, 'bootstrap_onset_summary.csv')
    bootstrap_df.drop(columns=['all_onsets'], errors='ignore').to_csv(bootstrap_summary_path, index=False)
    print(f"Bootstrap summary: {bootstrap_summary_path}")

    loo_summary_path = os.path.join(data_dir, 'loo_sensitivity.csv')
    loo_df.drop(columns=['all_loo_onsets'], errors='ignore').to_csv(loo_summary_path, index=False)
    print(f"LOO results: {loo_summary_path}")

    subsampling_path = os.path.join(data_dir, 'subsampling_stability.csv')
    subsampling_df.drop(columns=['all_onsets'], errors='ignore').to_csv(subsampling_path, index=False)
    print(f"Subsampling results: {subsampling_path}")

    # ============================================================================
    # 8️⃣ ROBUSTNESS METRICS SUMMARY
    # ============================================================================

    print("\n" + "="*80)
    print("ROBUSTNESS METRICS SUMMARY")
    print("="*80)

    robustness_summary = summary_df[['group1', 'group2', 'onset_bin']].merge(
        bootstrap_df[['group1', 'group2', 'onset_median', 'onset_low', 'onset_high', 'n_eff']],
        on=['group1', 'group2'],
        how='left'
    ).merge(
        loo_df[['group1', 'group2', 'loo_min', 'loo_max', 'loo_range']],
        on=['group1', 'group2'],
        how='left'
    )

    robustness_summary['bootstrap_ci_width'] = robustness_summary['onset_high'] - robustness_summary['onset_low']
    robustness_summary['point_vs_bootstrap_diff'] = np.abs(robustness_summary['onset_bin'] - robustness_summary['onset_median'])
    robustness_summary['high_uncertainty'] = (
        (robustness_summary['bootstrap_ci_width'] > 4) |
        (robustness_summary['loo_range'] > 6)
    )

    robustness_summary_path = os.path.join(data_dir, 'robustness_summary.csv')
    robustness_summary.to_csv(robustness_summary_path, index=False)
    print(f"\nComprehensive robustness summary saved: {robustness_summary_path}")

    print("\n" + "-"*120)
    print(f"{'Comparison':<35} {'Point':<8} {'Boot Med':<10} {'95% CI':<20} {'CI Width':<10} {'LOO Range':<15} {'Uncertain'}")
    print("-"*120)

    for _, row in robustness_summary.iterrows():
        comp = f"{row['group1']} vs {row['group2']}"
        if len(comp) > 35:
            comp = comp[:32] + "..."

        point = f"{row['onset_bin']:.1f}" if pd.notna(row['onset_bin']) else "N/A"
        boot_med = f"{row['onset_median']:.1f}" if pd.notna(row['onset_median']) else "N/A"

        if pd.notna(row['onset_low']) and pd.notna(row['onset_high']):
            ci = f"[{row['onset_low']:.1f}, {row['onset_high']:.1f}]"
            ci_width = f"{row['bootstrap_ci_width']:.1f}"
        else:
            ci = "N/A"
            ci_width = "N/A"

        if pd.notna(row['loo_min']) and pd.notna(row['loo_max']):
            loo = f"[{row['loo_min']:.1f}, {row['loo_max']:.1f}]"
        else:
            loo = "N/A"

        uncertain = "⚠️  YES" if row['high_uncertainty'] else "✓ No"

        print(f"{comp:<35} {point:<8} {boot_med:<10} {ci:<20} {ci_width:<10} {loo:<15} {uncertain}")

    print("-"*120)

    if robustness_summary['high_uncertainty'].any():
        print("\n⚠️  WARNING: Some comparisons show high uncertainty!")
        print("   Consider collecting more data or investigating influential samples.")
    else:
        print("\n✓ All comparisons show good robustness across resampling methods.")

    print("\n" + "="*80)
    print("COMPLETE ANALYSIS FINISHED")
    print("="*80)
    print(f"\nAll results saved to:")
    print(f"  Data: {data_dir}")
    print(f"  Plots: {plot_dir}")

    robustness_summary = robustness_summary.assign(genotype_group=genotype_label)
    aggregate_robustness.append(robustness_summary)

if aggregate_robustness:
    combined_robustness = pd.concat(aggregate_robustness, ignore_index=True)
    combined_path = os.path.join(data_dir_base, 'all_genotypes_robustness_summary.csv')
    combined_robustness.to_csv(combined_path, index=False)
    print(f"\nCombined robustness summary saved: {combined_path}")
