"""
Investigate why baseline and balanced methods produce different numbers of
predictions per embryo during cross-validation.

This script traces through the CV process to identify where and why embryos
get different temporal coverage between methods.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014"
data_dir = os.path.join(results_dir, "imbalance_methods", "data")
output_dir = os.path.join(results_dir, "imbalance_methods", "diagnostics", "cv_folds")
os.makedirs(output_dir, exist_ok=True)

build06_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output"

N_CV_SPLITS = 5
RANDOM_SEED = 42

print(f"CV fold diagnostic output: {output_dir}\n")

# ============================================================================
# BINNING FUNCTION
# ============================================================================

def bin_by_embryo_time(df, time_col="predicted_stage_hpf", z_cols=None, bin_width=2.0, suffix="_binned"):
    """Bin VAE embeddings by predicted time and embryo."""
    df = df.copy()

    if z_cols is None:
        z_cols = [c for c in df.columns if "z_mu_b" in c]
        if not z_cols:
            raise ValueError("No latent columns found matching pattern 'z_mu_b'.")

    df["time_bin"] = (np.floor(df[time_col] / bin_width) * bin_width).astype(int)

    agg = df.groupby(["embryo_id", "time_bin"], as_index=False)[z_cols].mean()
    agg.rename(columns={c: f"{c}{suffix}" for c in z_cols}, inplace=True)

    meta_cols = [c for c in df.columns if c not in z_cols + [time_col, "time_bin"]]
    meta_df = df[meta_cols].drop_duplicates(subset=["embryo_id"])

    out = agg.merge(meta_df, on="embryo_id", how="left")
    out = out.sort_values(["embryo_id", "time_bin"]).reset_index(drop=True)

    return out


# ============================================================================
# CV FOLD TRACKING
# ============================================================================

def trace_cv_predictions(df_binned, group1, group2, time_bin, method_name, use_class_weights=False):
    """
    Trace through one time bin's CV process and record which embryos
    get predictions in which folds.

    Returns DataFrame with columns: embryo_id, fold_idx, was_in_test_set
    """
    # Filter to this time bin and two groups
    sub = df_binned[(df_binned['time_bin'] == time_bin) &
                    (df_binned['genotype'].isin([group1, group2]))].copy()

    if len(sub) < N_CV_SPLITS * 2:
        return None

    z_cols = [c for c in sub.columns if c.endswith("_binned")]
    X = sub[z_cols].values
    y = sub['genotype'].values
    embryo_ids = sub['embryo_id'].values

    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        return None

    min_samples_per_class = min([np.sum(y == c) for c in unique_classes])
    if min_samples_per_class < N_CV_SPLITS:
        return None

    # Run CV
    skf = StratifiedKFold(n_splits=min(N_CV_SPLITS, min_samples_per_class),
                         shuffle=True, random_state=RANDOM_SEED)

    cv_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Build model
        if use_class_weights:
            model = LogisticRegression(max_iter=200, class_weight='balanced', random_state=RANDOM_SEED)
        else:
            model = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)

        try:
            model.fit(X[train_idx], y[train_idx])
            pred_proba = model.predict_proba(X[test_idx])

            # Record which embryos were in test set this fold
            for idx in test_idx:
                cv_records.append({
                    'embryo_id': embryo_ids[idx],
                    'fold_idx': fold_idx,
                    'time_bin': time_bin,
                    'was_in_test': True,
                    'genotype': y[idx]
                })
        except Exception as e:
            print(f"    Fold {fold_idx} failed: {e}")
            continue

    return pd.DataFrame(cv_records)


def analyze_cv_coverage(gene, group1, group2):
    """
    Analyze CV fold coverage for baseline vs balanced across all time bins.
    """
    print("="*80)
    print(f"CV FOLD COVERAGE ANALYSIS")
    print(f"Gene: {gene}, Groups: {group1} vs {group2}")
    print("="*80)

    safe_name = f"{group1.split('_')[-1]}_vs_{group2.split('_')[-1]}"

    # Load and bin data
    print(f"\nLoading data...")

    # Determine which experiments to load based on gene
    if 'cep290' in gene:
        experiments = ["20250305", "20250416", "20250512", "20250515_part2", "20250519"]
    elif 'b9d2' in gene:
        experiments = ["20250519", "20250520"]
    elif 'tmem67' in gene:
        experiments = ["20250711"]
    else:
        print(f"Unknown gene: {gene}")
        return

    dfs = []
    for exp in experiments:
        try:
            file_path = f"{build06_dir}/df03_final_output_with_latents_{exp}.csv"
            df = pd.read_csv(file_path)
            df['source_experiment'] = exp
            dfs.append(df)
        except:
            pass

    if not dfs:
        print(f"No data loaded for {gene}")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows")

    # Filter to gene family
    genotype_values = [group1, group2]
    df_family = combined_df[combined_df['genotype'].isin(genotype_values)].copy()
    print(f"Filtered to {len(df_family)} rows for {group1} and {group2}")

    # Bin
    print(f"Binning data...")
    df_binned = bin_by_embryo_time(df_family, time_col="predicted_stage_hpf")

    binned_z_cols = [c for c in df_binned.columns if "_binned" in c]
    df_binned = df_binned.dropna(subset=binned_z_cols)
    print(f"After binning and dropping NaNs: {len(df_binned)} rows")

    # Get time bins
    time_bins = sorted(df_binned['time_bin'].unique())
    print(f"Time bins: {len(time_bins)} bins from {time_bins[0]} to {time_bins[-1]}")

    # Trace CV for baseline and balanced at each time bin
    print(f"\nTracing CV folds across time bins...")

    baseline_records = []
    balanced_records = []

    for t_idx, t in enumerate(time_bins):
        print(f"  Time bin {t} ({t_idx+1}/{len(time_bins)})...")

        # Baseline
        baseline_cv = trace_cv_predictions(df_binned, group1, group2, t,
                                          method_name='baseline', use_class_weights=False)
        if baseline_cv is not None:
            baseline_records.append(baseline_cv)

        # Balanced
        balanced_cv = trace_cv_predictions(df_binned, group1, group2, t,
                                          method_name='balanced', use_class_weights=True)
        if balanced_cv is not None:
            balanced_records.append(balanced_cv)

    # Combine records
    if not baseline_records or not balanced_records:
        print("No CV records generated - insufficient data")
        return

    df_baseline_cv = pd.concat(baseline_records, ignore_index=True)
    df_balanced_cv = pd.concat(balanced_records, ignore_index=True)

    print(f"\nCV records generated:")
    print(f"  Baseline: {len(df_baseline_cv)} predictions across time bins")
    print(f"  Balanced: {len(df_balanced_cv)} predictions across time bins")

    # Aggregate by embryo
    print(f"\nAggregating by embryo...")

    baseline_by_embryo = df_baseline_cv.groupby('embryo_id').agg(
        n_time_bins=('time_bin', 'nunique'),
        n_predictions=('time_bin', 'count'),
        time_bins=('time_bin', lambda x: sorted(x.unique())),
        genotype=('genotype', 'first')
    ).reset_index()

    balanced_by_embryo = df_balanced_cv.groupby('embryo_id').agg(
        n_time_bins=('time_bin', 'nunique'),
        n_predictions=('time_bin', 'count'),
        time_bins=('time_bin', lambda x: sorted(x.unique())),
        genotype=('genotype', 'first')
    ).reset_index()

    print(f"\nEmbryo coverage:")
    print(f"  Baseline: {len(baseline_by_embryo)} embryos")
    print(f"    Mean time bins/embryo: {baseline_by_embryo['n_time_bins'].mean():.2f}")
    print(f"    Min: {baseline_by_embryo['n_time_bins'].min()}, Max: {baseline_by_embryo['n_time_bins'].max()}")

    print(f"  Balanced: {len(balanced_by_embryo)} embryos")
    print(f"    Mean time bins/embryo: {balanced_by_embryo['n_time_bins'].mean():.2f}")
    print(f"    Min: {balanced_by_embryo['n_time_bins'].min()}, Max: {balanced_by_embryo['n_time_bins'].max()}")

    # Find embryos with different coverage
    merged = baseline_by_embryo.merge(
        balanced_by_embryo,
        on='embryo_id',
        how='outer',
        suffixes=('_baseline', '_balanced')
    )

    merged['diff_time_bins'] = merged['n_time_bins_balanced'].fillna(0) - merged['n_time_bins_baseline'].fillna(0)

    print(f"\nEmbryo coverage differences:")
    print(f"  Same coverage: {(merged['diff_time_bins'] == 0).sum()}")
    print(f"  Balanced has more: {(merged['diff_time_bins'] > 0).sum()}")
    print(f"  Baseline has more: {(merged['diff_time_bins'] < 0).sum()}")

    # Show top discrepancies
    print(f"\nTop 10 embryos with most coverage difference (Balanced - Baseline):")
    top_diff = merged.sort_values('diff_time_bins', ascending=False).head(10)
    for idx, row in top_diff.iterrows():
        print(f"  {row['embryo_id']}: Baseline={row['n_time_bins_baseline']:.0f}, "
              f"Balanced={row['n_time_bins_balanced']:.0f}, Diff={row['diff_time_bins']:.0f}")

    # Visualize
    print(f"\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Distribution of time bins per embryo
    ax = axes[0, 0]
    ax.hist(baseline_by_embryo['n_time_bins'], bins=20, alpha=0.6,
           label='Baseline', color='steelblue', density=True)
    ax.hist(balanced_by_embryo['n_time_bins'], bins=20, alpha=0.6,
           label='Balanced', color='coral', density=True)
    ax.set_xlabel('Number of Time Bins per Embryo', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Time Bin Coverage Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: Coverage difference
    ax = axes[0, 1]
    ax.hist(merged['diff_time_bins'].dropna(), bins=30, color='purple', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    ax.set_xlabel('Coverage Difference (Balanced - Baseline)', fontsize=11)
    ax.set_ylabel('Number of Embryos', fontsize=11)
    ax.set_title('Per-Embryo Coverage Difference', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Coverage by genotype (baseline)
    ax = axes[1, 0]
    for genotype in [group1, group2]:
        subset = baseline_by_embryo[baseline_by_embryo['genotype'] == genotype]
        ax.hist(subset['n_time_bins'], bins=15, alpha=0.6, label=genotype.split('_')[-1])
    ax.set_xlabel('Number of Time Bins', fontsize=11)
    ax.set_ylabel('Number of Embryos', fontsize=11)
    ax.set_title('Baseline: Coverage by Genotype', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Coverage by genotype (balanced)
    ax = axes[1, 1]
    for genotype in [group1, group2]:
        subset = balanced_by_embryo[balanced_by_embryo['genotype'] == genotype]
        ax.hist(subset['n_time_bins'], bins=15, alpha=0.6, label=genotype.split('_')[-1])
    ax.set_xlabel('Number of Time Bins', fontsize=11)
    ax.set_ylabel('Number of Embryos', fontsize=11)
    ax.set_title('Balanced: Coverage by Genotype', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f'CV Fold Coverage Analysis\n{gene.upper()}: {safe_name}',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'cv_coverage_{gene}_{safe_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot: {plot_path}")
    plt.close()

    # Save detailed tables
    baseline_csv = os.path.join(output_dir, f'cv_coverage_baseline_{gene}_{safe_name}.csv')
    balanced_csv = os.path.join(output_dir, f'cv_coverage_balanced_{gene}_{safe_name}.csv')
    comparison_csv = os.path.join(output_dir, f'cv_coverage_comparison_{gene}_{safe_name}.csv')

    baseline_by_embryo.to_csv(baseline_csv, index=False)
    balanced_by_embryo.to_csv(balanced_csv, index=False)
    merged.to_csv(comparison_csv, index=False)

    print(f"  Saved CSVs:")
    print(f"    {baseline_csv}")
    print(f"    {balanced_csv}")
    print(f"    {comparison_csv}")

    return {
        'baseline_by_embryo': baseline_by_embryo,
        'balanced_by_embryo': balanced_by_embryo,
        'comparison': merged
    }


# ============================================================================
# RUN ANALYSIS
# ============================================================================

print("="*80)
print("CV FOLD ISSUE INVESTIGATION")
print("="*80)

# Focus on cep290 wildtype vs heterozygous
result = analyze_cv_coverage('cep290', 'cep290_wildtype', 'cep290_heterozygous')

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {output_dir}")
