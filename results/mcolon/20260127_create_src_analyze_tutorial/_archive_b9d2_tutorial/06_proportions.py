"""
Tutorial 06: Proportions Plot

Demonstrates visualization of cluster distribution by genotype.

Key API usage:
- plot_proportion_faceted() for stacked bar charts
- Cross-tabulation of clusters vs genotypes
"""

import pandas as pd
from pathlib import Path

# Setup directories
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Load data
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

print("Loading experiment data...")
df1 = load_experiment_dataframe('20251121', format_version='df03')
df2 = load_experiment_dataframe('20251125', format_version='df03')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['use_embryo_flag']].copy()

# Load cluster labels
membership_df = pd.read_csv(RESULTS_DIR / "cluster_membership_labeled.csv")
embryo_labels = membership_df[['embryo_id', 'cluster_label', 'membership_class']].copy()

# Get one row per embryo
df_embryos = df.drop_duplicates('embryo_id')[['embryo_id', 'genotype']].copy()
df_embryos = df_embryos.merge(embryo_labels, on='embryo_id', how='left')
df_embryos = df_embryos.dropna(subset=['cluster_label'])

print(f"\nAnalyzing {len(df_embryos)} embryos")

# ============================================================================
# Compute proportions
# ============================================================================
print("\n" + "="*80)
print("CLUSTER PROPORTIONS BY GENOTYPE")
print("="*80)

# Count embryos per genotype-cluster combination
counts = df_embryos.groupby(['genotype', 'cluster_label']).size().reset_index(name='count')

# Compute proportions within each genotype
genotype_totals = df_embryos.groupby('genotype').size().reset_index(name='total')
counts = counts.merge(genotype_totals, on='genotype')
counts['proportion'] = counts['count'] / counts['total']

print("\nCounts and proportions:")
print(counts)

# Save proportions table
counts.to_csv(RESULTS_DIR / "cluster_proportions_by_genotype.csv", index=False)
print(f"\nSaved to: {RESULTS_DIR / 'cluster_proportions_by_genotype.csv'}")

# ============================================================================
# Pivot for visualization
# ============================================================================
proportions_pivot = counts.pivot(
    index='genotype',
    columns='cluster_label',
    values='proportion'
).fillna(0)

print("\nProportions (pivot table):")
print(proportions_pivot)

# ============================================================================
# Plot proportions
# ============================================================================
print("\n" + "="*80)
print("CREATING PROPORTION PLOTS")
print("="*80)

from src.analyze.viz.plotting import plot_proportion_faceted

# Define cluster colors
CLUSTER_COLOR_MAP = {
    'Short Body Axis': '#d62728',      # Red
    'Homozygous B9D2': '#ff7f0e',      # Orange
    'Not Penetrant': '#2ca02c',        # Green
}

print("\nPlotting cluster proportions by genotype...")
fig = plot_proportion_faceted(
    df_embryos,
    group_by='genotype',
    category='cluster_label',
    color_lookup=CLUSTER_COLOR_MAP,
    # Optional parameters:
    # normalize=True,  # Show as proportions (0-1) vs counts
    # show_counts=True,  # Annotate bars with counts
)
fig.write_html(FIGURES_DIR / "21_proportions_by_genotype.html")
print(f"   Saved: {FIGURES_DIR / '21_proportions_by_genotype.html'}")

# ============================================================================
# Alternative: Proportions by experiment
# ============================================================================
print("\nPlotting cluster proportions by experiment...")
df_embryos['experiment_id'] = df_embryos['embryo_id'].str[:8]

fig = plot_proportion_faceted(
    df_embryos,
    group_by='experiment_id',
    category='cluster_label',
    color_lookup=CLUSTER_COLOR_MAP,
)
fig.write_html(FIGURES_DIR / "22_proportions_by_experiment.html")
print(f"   Saved: {FIGURES_DIR / '22_proportions_by_experiment.html'}")

# ============================================================================
# Filter to core members only
# ============================================================================
print("\nPlotting proportions (core members only)...")
df_core = df_embryos[df_embryos['membership_class'] == 'core'].copy()

fig = plot_proportion_faceted(
    df_core,
    group_by='genotype',
    category='cluster_label',
    color_lookup=CLUSTER_COLOR_MAP,
)
fig.write_html(FIGURES_DIR / "23_proportions_core_only.html")
print(f"   Saved: {FIGURES_DIR / '23_proportions_core_only.html'}")

# ============================================================================
# Summary statistics
# ============================================================================
print("\n" + "="*80)
print("PROPORTION SUMMARY")
print("="*80)

for genotype in df_embryos['genotype'].unique():
    print(f"\n{genotype}:")
    gt_data = df_embryos[df_embryos['genotype'] == genotype]
    total = len(gt_data)
    print(f"  Total embryos: {total}")

    for cluster_label in sorted(gt_data['cluster_label'].unique()):
        n = len(gt_data[gt_data['cluster_label'] == cluster_label])
        pct = n / total * 100
        print(f"  {cluster_label}: {n} ({pct:.1f}%)")

print("\n✓ Tutorial 06 complete!")
print(f"  Figures saved to: {FIGURES_DIR}")
print(f"  Results saved to: {RESULTS_DIR}")
