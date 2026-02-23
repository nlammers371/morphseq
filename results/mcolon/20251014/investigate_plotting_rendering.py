"""
Pre-plotting diagnostic to investigate why baseline and balanced trajectory
plots appear to have different numbers of lines despite having identical data.

This script replicates the exact plotting logic and captures metadata about
what would be plotted, including color assignments, alpha values, and line
segment structure.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014"
data_dir = os.path.join(results_dir, "imbalance_methods", "data")
output_dir = os.path.join(results_dir, "imbalance_methods", "diagnostics", "plotting")
os.makedirs(output_dir, exist_ok=True)

print(f"Plotting diagnostic output: {output_dir}\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_embryo_penetrance(df_embryo_probs, confidence_threshold=0.1):
    """Compute per-embryo penetrance metrics."""
    if df_embryo_probs.empty:
        return pd.DataFrame()

    penetrance_metrics = []

    for embryo_id, grp in df_embryo_probs.groupby('embryo_id'):
        grp = grp.sort_values('time_bin')

        mean_conf = grp['confidence'].mean()
        mean_signed_margin = grp['signed_margin'].mean() if 'signed_margin' in grp.columns else np.nan

        penetrance_metrics.append({
            'embryo_id': embryo_id,
            'true_label': grp['true_label'].iloc[0],
            'mean_confidence': mean_conf,
            'mean_signed_margin': mean_signed_margin,
        })

    return pd.DataFrame(penetrance_metrics)


def analyze_plotting_data(df_probs, df_penetrance, genotype, max_embryos=30):
    """
    Analyze exactly what data would be plotted for a given genotype.

    Returns detailed metadata about each embryo trajectory that would be drawn.
    """
    if df_probs is None or df_probs.empty or df_penetrance is None or df_penetrance.empty:
        return pd.DataFrame()

    # Select embryos using same logic as plotting code
    genotype_penetrance = df_penetrance[df_penetrance['true_label'] == genotype].copy()
    if genotype_penetrance.empty:
        return pd.DataFrame()

    genotype_penetrance['abs_margin'] = np.abs(genotype_penetrance.get('mean_signed_margin', np.nan))
    genotype_penetrance = genotype_penetrance.sort_values(
        by=['abs_margin', 'mean_signed_margin'], ascending=[False, False]
    ).head(max_embryos)

    selected_embryos = genotype_penetrance['embryo_id'].values

    # Set up color mapping (same as plotting code)
    norm = Normalize(vmin=-0.5, vmax=0.5)
    cmap = plt.cm.RdBu_r
    alphas = np.linspace(0.35, 0.9, len(selected_embryos)) if len(selected_embryos) > 0 else []
    highlight_id = genotype_penetrance.iloc[0]['embryo_id'] if len(genotype_penetrance) > 0 else None
    penetrance_lookup = genotype_penetrance.set_index('embryo_id')

    # Analyze each embryo trajectory
    trajectory_metadata = []

    for idx, (alpha, embryo_id) in enumerate(zip(alphas, selected_embryos)):
        embryo_curve = df_probs[df_probs['embryo_id'] == embryo_id].sort_values('time_bin')

        if embryo_curve.empty:
            continue

        # Get mean margin for color
        mean_margin = np.nan
        if 'mean_signed_margin' in penetrance_lookup.columns:
            mean_margin = penetrance_lookup.at[embryo_id, 'mean_signed_margin']
        if np.isnan(mean_margin):
            mean_margin = embryo_curve['signed_margin'].mean()

        # Compute color
        base_color = cmap(norm(mean_margin))
        is_highlight = (embryo_id == highlight_id)

        if is_highlight:
            color_rgba = base_color
            linewidth = 2.8
            marker_size = 4
            final_alpha = 1.0
        else:
            color_rgba = (base_color[0], base_color[1], base_color[2], alpha)
            linewidth = 1.6
            marker_size = 3
            final_alpha = alpha

        # Convert to hex for easier comparison
        color_hex = to_hex(color_rgba[:3])

        # Check for NaN gaps in trajectory
        time_bins = embryo_curve['time_bin'].values
        margins = embryo_curve['signed_margin'].values

        n_points = len(time_bins)
        n_nans = np.isnan(margins).sum()
        n_valid_points = n_points - n_nans

        # Find gaps (consecutive time bins missing)
        if len(time_bins) > 1:
            time_diffs = np.diff(time_bins)
            # Assuming time bins are spaced by 2 (from bin_width=2.0)
            expected_diff = 2
            n_gaps = np.sum(time_diffs > expected_diff)
        else:
            n_gaps = 0

        # Compute how many separate line segments would be drawn
        # A segment breaks when there's a NaN or a time gap
        n_segments = 1
        for i in range(len(margins) - 1):
            if np.isnan(margins[i]) or np.isnan(margins[i+1]):
                n_segments += 1
            elif time_bins[i+1] - time_bins[i] > expected_diff:
                n_segments += 1

        trajectory_metadata.append({
            'embryo_id': embryo_id,
            'selection_rank': idx + 1,
            'is_highlight': is_highlight,
            'mean_signed_margin': mean_margin,
            'abs_mean_signed_margin': np.abs(mean_margin),
            'n_data_points': n_points,
            'n_valid_points': n_valid_points,
            'n_nan_points': n_nans,
            'n_time_gaps': n_gaps,
            'n_line_segments': n_segments,
            'time_start': time_bins.min(),
            'time_end': time_bins.max(),
            'time_span': time_bins.max() - time_bins.min(),
            'color_hex': color_hex,
            'color_r': color_rgba[0],
            'color_g': color_rgba[1],
            'color_b': color_rgba[2],
            'alpha': final_alpha,
            'linewidth': linewidth,
            'marker_size': marker_size
        })

    return pd.DataFrame(trajectory_metadata)


def compare_plotting_metadata(metadata_baseline, metadata_balanced, genotype, output_prefix):
    """
    Compare plotting metadata between baseline and balanced and generate diagnostic report.
    """
    print(f"\n{'='*80}")
    print(f"PLOTTING METADATA COMPARISON: {genotype}")
    print(f"{'='*80}")

    if metadata_baseline.empty or metadata_balanced.empty:
        print("  One or both metadata tables are empty - skipping")
        return

    print(f"\nBaseline: {len(metadata_baseline)} embryos to plot")
    print(f"Balanced: {len(metadata_balanced)} embryos to plot")

    # Compare totals
    print(f"\nTotal valid data points:")
    print(f"  Baseline: {metadata_baseline['n_valid_points'].sum()}")
    print(f"  Balanced: {metadata_balanced['n_valid_points'].sum()}")

    print(f"\nTotal line segments:")
    print(f"  Baseline: {metadata_baseline['n_line_segments'].sum()}")
    print(f"  Balanced: {metadata_balanced['n_line_segments'].sum()}")

    print(f"\nMean line segments per embryo:")
    print(f"  Baseline: {metadata_baseline['n_line_segments'].mean():.2f}")
    print(f"  Balanced: {metadata_balanced['n_line_segments'].mean():.2f}")

    print(f"\nNaN points:")
    print(f"  Baseline: {metadata_baseline['n_nan_points'].sum()} "
          f"({100 * metadata_baseline['n_nan_points'].sum() / metadata_baseline['n_data_points'].sum():.1f}%)")
    print(f"  Balanced: {metadata_balanced['n_nan_points'].sum()} "
          f"({100 * metadata_balanced['n_nan_points'].sum() / metadata_balanced['n_data_points'].sum():.1f}%)")

    print(f"\nTime gaps:")
    print(f"  Baseline: {metadata_baseline['n_time_gaps'].sum()}")
    print(f"  Balanced: {metadata_balanced['n_time_gaps'].sum()}")

    # Color distinctiveness
    print(f"\nColor distinctiveness:")
    print(f"  Baseline unique colors: {metadata_baseline['color_hex'].nunique()}")
    print(f"  Balanced unique colors: {metadata_balanced['color_hex'].nunique()}")

    # Alpha distribution
    print(f"\nAlpha values:")
    print(f"  Baseline: mean={metadata_baseline['alpha'].mean():.3f}, "
          f"min={metadata_baseline['alpha'].min():.3f}, max={metadata_baseline['alpha'].max():.3f}")
    print(f"  Balanced: mean={metadata_balanced['alpha'].mean():.3f}, "
          f"min={metadata_balanced['alpha'].min():.3f}, max={metadata_balanced['alpha'].max():.3f}")

    # Check for very low alpha (nearly invisible)
    threshold_alpha = 0.4
    n_low_alpha_baseline = (metadata_baseline['alpha'] < threshold_alpha).sum()
    n_low_alpha_balanced = (metadata_balanced['alpha'] < threshold_alpha).sum()
    print(f"\nLow alpha (< {threshold_alpha}) embryos:")
    print(f"  Baseline: {n_low_alpha_baseline}/{len(metadata_baseline)}")
    print(f"  Balanced: {n_low_alpha_balanced}/{len(metadata_balanced)}")

    # Embryo overlap
    baseline_embryos = set(metadata_baseline['embryo_id'])
    balanced_embryos = set(metadata_balanced['embryo_id'])
    overlap = baseline_embryos & balanced_embryos

    print(f"\nEmbryo selection overlap: {len(overlap)}/{len(baseline_embryos)}")

    if len(baseline_embryos - balanced_embryos) > 0:
        print(f"  Baseline-only embryos: {baseline_embryos - balanced_embryos}")
    if len(balanced_embryos - baseline_embryos) > 0:
        print(f"  Balanced-only embryos: {balanced_embryos - baseline_embryos}")

    # Generate visual comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Line segments per embryo
    ax = axes[0, 0]
    x_baseline = np.arange(len(metadata_baseline))
    x_balanced = np.arange(len(metadata_balanced))
    ax.bar(x_baseline - 0.2, metadata_baseline['n_line_segments'], width=0.4,
           label='Baseline', alpha=0.7, color='steelblue')
    ax.bar(x_balanced + 0.2, metadata_balanced['n_line_segments'], width=0.4,
           label='Balanced', alpha=0.7, color='coral')
    ax.set_xlabel('Embryo Rank', fontsize=11)
    ax.set_ylabel('Number of Line Segments', fontsize=11)
    ax.set_title('Line Segments per Embryo', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Panel 2: Alpha values
    ax = axes[0, 1]
    ax.bar(x_baseline - 0.2, metadata_baseline['alpha'], width=0.4,
           label='Baseline', alpha=0.7, color='steelblue')
    ax.bar(x_balanced + 0.2, metadata_balanced['alpha'], width=0.4,
           label='Balanced', alpha=0.7, color='coral')
    ax.axhline(threshold_alpha, color='red', linestyle='--', linewidth=1.5,
              label=f'Low visibility threshold ({threshold_alpha})')
    ax.set_xlabel('Embryo Rank', fontsize=11)
    ax.set_ylabel('Alpha Value', fontsize=11)
    ax.set_title('Alpha Transparency per Embryo', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Panel 3: Time coverage (gantt-style)
    ax = axes[1, 0]
    for idx, row in metadata_baseline.iterrows():
        ax.barh(idx, row['time_span'], left=row['time_start'], height=0.8,
               color='steelblue', alpha=0.6)
    ax.set_xlabel('Time (hpf)', fontsize=11)
    ax.set_ylabel('Embryo Index', fontsize=11)
    ax.set_title('Baseline: Time Coverage per Embryo', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    ax = axes[1, 1]
    for idx, row in metadata_balanced.iterrows():
        ax.barh(idx, row['time_span'], left=row['time_start'], height=0.8,
               color='coral', alpha=0.6)
    ax.set_xlabel('Time (hpf)', fontsize=11)
    ax.set_ylabel('Embryo Index', fontsize=11)
    ax.set_title('Balanced: Time Coverage per Embryo', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    fig.suptitle(f'Plotting Metadata Comparison\n{genotype}',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'{output_prefix}_plotting_metadata.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved plot: {plot_path}")
    plt.close()

    # Save metadata tables
    baseline_csv = os.path.join(output_dir, f'{output_prefix}_baseline_metadata.csv')
    balanced_csv = os.path.join(output_dir, f'{output_prefix}_balanced_metadata.csv')

    metadata_baseline.to_csv(baseline_csv, index=False)
    metadata_balanced.to_csv(balanced_csv, index=False)

    print(f"  Saved CSVs:")
    print(f"    {baseline_csv}")
    print(f"    {balanced_csv}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

# Focus on cep290 wildtype vs heterozygous as test case
GENE = "cep290"
GROUP1 = "cep290_wildtype"
GROUP2 = "cep290_heterozygous"

print("="*80)
print(f"PLOTTING RENDERING INVESTIGATION")
print(f"Test Case: {GENE} - {GROUP1} vs {GROUP2}")
print("="*80)

safe_name = f"{GROUP1.split('_')[-1]}_vs_{GROUP2.split('_')[-1]}"
gene_data_dir = os.path.join(data_dir, GENE)

# Load data
baseline_path = os.path.join(gene_data_dir, f"embryo_probs_baseline_{safe_name}.csv")
balanced_path = os.path.join(gene_data_dir, f"embryo_probs_class_weight_{safe_name}.csv")

print(f"\nLoading data...")
print(f"  Baseline: {baseline_path}")
print(f"  Balanced: {balanced_path}")

df_baseline = pd.read_csv(baseline_path)
df_balanced = pd.read_csv(balanced_path)

print(f"\nData loaded:")
print(f"  Baseline: {len(df_baseline)} rows, {df_baseline['embryo_id'].nunique()} embryos")
print(f"  Balanced: {len(df_balanced)} rows, {df_balanced['embryo_id'].nunique()} embryos")

# Compute penetrance
print(f"\nComputing penetrance metrics...")
penetrance_baseline = compute_embryo_penetrance(df_baseline)
penetrance_balanced = compute_embryo_penetrance(df_balanced)

print(f"  Baseline: {len(penetrance_baseline)} embryos")
print(f"  Balanced: {len(penetrance_balanced)} embryos")

# Analyze plotting data for each genotype
for genotype in [GROUP1, GROUP2]:
    print(f"\n{'='*80}")
    print(f"ANALYZING: {genotype}")
    print(f"{'='*80}")

    # Get plotting metadata
    print(f"\nExtracting plotting metadata...")
    metadata_baseline = analyze_plotting_data(df_baseline, penetrance_baseline, genotype, max_embryos=30)
    metadata_balanced = analyze_plotting_data(df_balanced, penetrance_balanced, genotype, max_embryos=30)

    print(f"  Baseline: {len(metadata_baseline)} embryos")
    print(f"  Balanced: {len(metadata_balanced)} embryos")

    # Compare
    output_prefix = f"{GENE}_{safe_name}_{genotype.split('_')[-1]}"
    compare_plotting_metadata(metadata_baseline, metadata_balanced, genotype, output_prefix)

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {output_dir}")
