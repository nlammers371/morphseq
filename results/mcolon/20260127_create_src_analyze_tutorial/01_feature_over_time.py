"""
Tutorial 01: Feature Over Time Plotting

Demonstrates basic time series plotting with the domain-agnostic API.
Shows single-feature, multi-feature, and faceted plotting patterns.

Key API usage:
- load_experiment_dataframe() for data loading
- plot_feature_over_time() with updated facet_row/facet_col parameters
- Multi-feature support (feature as list)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pandas as pd

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data from both CEP290 experiments
from src.analyze.trajectory_analysis.io import load_experiment_dataframe

print("Loading experiment data...")
# Note: qc_staged files have curvature data and staging already applied
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'

df1 = pd.read_csv(meta_dir / 'qc_staged_20260122.csv')
df2 = pd.read_csv(meta_dir / 'qc_staged_20260124.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Filter to valid embryos
df = df[df['use_embryo_flag']].copy()
print(f"Loaded {len(df['embryo_id'].unique())} embryos across {len(df)} timepoints")

# Define color lookup (domain-specific)
from src.analyze.trajectory_analysis.viz.styling import get_color_for_genotype
genotypes = df['genotype'].unique()
color_lookup = {gt: get_color_for_genotype(gt) for gt in genotypes}
print(f"Genotypes: {list(genotypes)}")

# Import plotting function
from src.analyze.viz.plotting import plot_feature_over_time

# ============================================================================
# Example 1: Basic single-feature plot
# ============================================================================
print("\n1. Creating basic single-feature plot...")
figs = plot_feature_over_time(
    df,
    feature='baseline_deviation_normalized',
    color_by='genotype',
    color_lookup=color_lookup,
    backend='both',
    # Optional parameters:
    # show_individual=True,  # Show individual trajectory traces
    # show_error_band=True,  # Show error bands
    # error_type='iqr',  # Error type: 'iqr', 'std', 'sem'
    # trend_statistic='median',  # Trend line: 'mean' or 'median'
    # bin_width=2.0,  # Time bin width in hours
    # smooth_method='gaussian',  # Smoothing: 'gaussian', 'lowess', None
)
# backend='both' returns dict with 'plotly' and 'matplotlib' keys
figs['plotly'].write_html(FIGURES_DIR / "01_single_feature.html")
figs['matplotlib'].savefig(FIGURES_DIR / "01_single_feature.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / '01_single_feature.html'}")
print(f"   Saved: {FIGURES_DIR / '01_single_feature.png'}")

# ============================================================================
# Example 2: Individual feature plots
# ============================================================================
print("\n2. Creating individual feature plots...")
figs1 = plot_feature_over_time(
    df,
    feature='baseline_deviation_normalized',
    color_by='genotype',
    color_lookup=color_lookup,
    backend='both',
)
figs1['plotly'].write_html(FIGURES_DIR / "02a_baseline_deviation.html")
figs1['matplotlib'].savefig(FIGURES_DIR / "02a_baseline_deviation.png", dpi=300, bbox_inches='tight')

figs2 = plot_feature_over_time(
    df,
    feature='total_length_um',
    color_by='genotype',
    color_lookup=color_lookup,
    backend='both',
)
figs2['plotly'].write_html(FIGURES_DIR / "02b_total_length.html")
figs2['matplotlib'].savefig(FIGURES_DIR / "02b_total_length.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / '02a_baseline_deviation.html/.png'}")
print(f"   Saved: {FIGURES_DIR / '02b_total_length.html/.png'}")

# ============================================================================
# Example 3: Faceted plot (NEW: facet_row/facet_col API)
# ============================================================================
print("\n3. Creating faceted plot by genotype...")
figs = plot_feature_over_time(
    df,
    feature='baseline_deviation_normalized',
    color_by='genotype',
    color_lookup=color_lookup,
    facet_col='genotype',  # Renamed from facet_col
    backend='both',
    # Can also use facet_row for row faceting
    # facet_row='some_other_column',
)
figs['plotly'].write_html(FIGURES_DIR / "03_faceted_by_genotype.html")
figs['matplotlib'].savefig(FIGURES_DIR / "03_faceted_by_genotype.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / '03_faceted_by_genotype.html/.png'}")

# ============================================================================
# Example 4: Multi-feature with column faceting (rows=features, cols=genotype)
# ============================================================================
print("\n4. Creating multi-feature plot with column faceting...")
figs = plot_feature_over_time(
    df,
    feature=['baseline_deviation_normalized', 'total_length_um'],  # Each feature gets a row
    color_by='genotype',
    color_lookup=color_lookup,
    facet_col='genotype',  # Columns = genotype
    backend='both',
)
figs['plotly'].write_html(FIGURES_DIR / "04_multi_feature_by_genotype.html")
figs['matplotlib'].savefig(FIGURES_DIR / "04_multi_feature_by_genotype.png", dpi=300, bbox_inches='tight')
print(f"   Saved: {FIGURES_DIR / '04_multi_feature_by_genotype.html/.png'}")

print("\n✓ Tutorial 01 complete!")
print(f"  All figures saved to: {FIGURES_DIR}")
