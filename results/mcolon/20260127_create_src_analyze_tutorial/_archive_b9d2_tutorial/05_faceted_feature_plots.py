"""
Tutorial 05: Multi-Feature Faceted Plots

Demonstrates complex faceted plotting with multiple features and groupings.

Key patterns:
- Multi-feature plots (rows = features)
- Column faceting by cluster label
- Combined row + column faceting

Key API usage:
- plot_feature_over_time() with feature as list
- row_by and col_by parameters for faceting
"""

import pandas as pd
from pathlib import Path

# Setup directories
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Load data from both B9D2 experiments
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

print("Loading experiment data...")
df1 = load_experiment_dataframe('20251121', format_version='df03')
df2 = load_experiment_dataframe('20251125', format_version='df03')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['use_embryo_flag']].copy()

# Load cluster labels
membership_df = pd.read_csv(RESULTS_DIR / "cluster_membership_labeled.csv")
embryo_labels = membership_df[['embryo_id', 'cluster_label']].copy()
df = df.merge(embryo_labels, on='embryo_id', how='left')

print(f"Loaded {len(df['embryo_id'].unique())} embryos")
print(f"With cluster labels: {df['cluster_label'].notna().sum()} rows")

# ============================================================================
# Setup plotting
# ============================================================================
from src.analyze.viz.plotting import plot_feature_over_time
from src.analyze.trajectory_analysis.viz.styling import get_genotype_color

# Color lookup for genotypes
genotypes = df['genotype'].unique()
genotype_color_lookup = {gt: get_genotype_color(gt) for gt in genotypes}

# Color lookup for clusters
CLUSTER_COLOR_MAP = {
    'Short Body Axis': '#d62728',      # Red
    'Homozygous B9D2': '#ff7f0e',      # Orange
    'Not Penetrant': '#2ca02c',        # Green
}

# Features to plot
FEATURES = ['baseline_deviation_normalized', 'total_length_um']

# ============================================================================
# Example 1: Multi-feature plot, faceted by cluster label
# ============================================================================
print("\n1. Multi-feature plot faceted by cluster label...")
fig = plot_feature_over_time(
    df.dropna(subset=['cluster_label']),  # Remove embryos without cluster labels
    feature=FEATURES,  # Rows = features (automatic)
    color_by='genotype',
    color_lookup=genotype_color_lookup,
    col_by='cluster_label',  # Columns = cluster labels
    backend='plotly',
)
fig.write_html(FIGURES_DIR / "15_multi_feature_by_cluster.html")
print(f"   Saved: {FIGURES_DIR / '15_multi_feature_by_cluster.html'}")

# ============================================================================
# Example 2: Single feature, faceted by cluster label
# ============================================================================
print("\n2. Single feature (baseline deviation) faceted by cluster...")
fig = plot_feature_over_time(
    df.dropna(subset=['cluster_label']),
    feature='baseline_deviation_normalized',
    color_by='genotype',
    color_lookup=genotype_color_lookup,
    col_by='cluster_label',
    backend='plotly',
)
fig.write_html(FIGURES_DIR / "16_baseline_deviation_by_cluster.html")
print(f"   Saved: {FIGURES_DIR / '16_baseline_deviation_by_cluster.html'}")

# ============================================================================
# Example 3: Single feature, faceted by genotype (for comparison)
# ============================================================================
print("\n3. Single feature (baseline deviation) faceted by genotype...")
fig = plot_feature_over_time(
    df.dropna(subset=['cluster_label']),
    feature='baseline_deviation_normalized',
    color_by='cluster_label',
    color_lookup=CLUSTER_COLOR_MAP,
    col_by='genotype',
    backend='plotly',
)
fig.write_html(FIGURES_DIR / "17_baseline_deviation_by_genotype.html")
print(f"   Saved: {FIGURES_DIR / '17_baseline_deviation_by_genotype.html'}")

# ============================================================================
# Example 4: Multi-feature, faceted by genotype AND cluster
# ============================================================================
print("\n4. Multi-feature with row AND column faceting...")
# Add experiment_id for row faceting
df['experiment_id'] = df['embryo_id'].str[:8]

fig = plot_feature_over_time(
    df.dropna(subset=['cluster_label']),
    feature='total_length_um',  # Single feature
    color_by='genotype',
    color_lookup=genotype_color_lookup,
    row_by='experiment_id',  # Rows = experiment
    col_by='cluster_label',  # Columns = cluster
    backend='plotly',
)
fig.write_html(FIGURES_DIR / "18_length_by_experiment_and_cluster.html")
print(f"   Saved: {FIGURES_DIR / '18_length_by_experiment_and_cluster.html'}")

# ============================================================================
# Example 5: Show individual trajectories for specific cluster
# ============================================================================
print("\n5. Individual trajectories for 'Short Body Axis' cluster...")
df_short_axis = df[df['cluster_label'] == 'Short Body Axis'].copy()

fig = plot_feature_over_time(
    df_short_axis,
    feature=FEATURES,
    color_by='genotype',
    color_lookup=genotype_color_lookup,
    show_individual=True,  # Show individual trajectories
    show_error_band=True,
    error_type='iqr',  # IQR error bands
    backend='plotly',
)
fig.write_html(FIGURES_DIR / "19_short_axis_individual_trajectories.html")
print(f"   Saved: {FIGURES_DIR / '19_short_axis_individual_trajectories.html'}")

# ============================================================================
# Example 6: Compare clusters within homozygous embryos only
# ============================================================================
print("\n6. Compare clusters within b9d2_homozygous genotype only...")
df_homo = df[df['genotype'] == 'b9d2_homozygous'].dropna(subset=['cluster_label'])

fig = plot_feature_over_time(
    df_homo,
    feature=FEATURES,
    color_by='cluster_label',
    color_lookup=CLUSTER_COLOR_MAP,
    backend='plotly',
)
fig.write_html(FIGURES_DIR / "20_homozygous_by_cluster.html")
print(f"   Saved: {FIGURES_DIR / '20_homozygous_by_cluster.html'}")

print("\n✓ Tutorial 05 complete!")
print(f"  All figures saved to: {FIGURES_DIR}")

# ============================================================================
# Optional parameters reference
# ============================================================================
print("\n" + "="*80)
print("OPTIONAL PARAMETERS REFERENCE")
print("="*80)
print("""
plot_feature_over_time() supports many optional parameters:

Time binning and smoothing:
- bin_width: float (default: 2.0) - Time bin width in hours
- smooth_method: str ('gaussian', 'lowess', None) - Smoothing method

Visualization options:
- show_individual: bool (default: False) - Show individual trajectories
- show_error_band: bool (default: True) - Show error bands
- error_type: str ('iqr', 'std', 'sem') - Error type for bands
- trend_statistic: str ('mean', 'median') - Central tendency statistic

Backend options:
- backend: str ('plotly', 'matplotlib', 'both') - Plotting backend

Faceting:
- row_by: str - Column name for row faceting
- col_by: str - Column name for column faceting
- feature: str or List[str] - When list, each feature gets its own row
""")
