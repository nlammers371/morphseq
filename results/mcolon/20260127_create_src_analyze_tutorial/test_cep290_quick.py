"""
Quick test: CEP290 experiments (20260122, 20260124)
Plot curvature and length over time, faceted by genotype
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Setup output
OUTPUT_DIR = Path(__file__).parent / "output" / "test_cep290"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading CEP290 data (20260122, 20260124) from build04...")
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'
df1 = pd.read_csv(meta_dir / 'qc_staged_20260122.csv')
df2 = pd.read_csv(meta_dir / 'qc_staged_20260124.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Filter to valid embryos
df = df[df['use_embryo_flag']].copy()
print(f"Loaded {len(df['embryo_id'].unique())} embryos across {len(df)} timepoints")
print(f"Genotypes: {df['genotype'].unique()}")

# Define color lookup
from src.analyze.trajectory_analysis.config import GENOTYPE_COLORS
color_lookup = GENOTYPE_COLORS

print("\nGenerating plot: curvature + length by genotype...")
from src.analyze.viz.plotting import plot_feature_over_time

# Plot curvature and length, columns by genotype
figs = plot_feature_over_time(
    df,
    features=['baseline_deviation_normalized', 'total_length_um'],  # Curvature first
    color_by='genotype',
    color_lookup=color_lookup,
    facet_col='genotype',
    backend='both',
    show_individual=True,
    show_error_band=True,
    error_type='iqr',
)

# Save outputs
figs['plotly'].write_html(OUTPUT_DIR / 'cep290_features_by_genotype.html')
figs['matplotlib'].savefig(OUTPUT_DIR / 'cep290_features_by_genotype.png', dpi=300, bbox_inches='tight')

print(f"\n✓ Done!")
print(f"  HTML: {OUTPUT_DIR / 'cep290_features_by_genotype.html'}")
print(f"  PNG:  {OUTPUT_DIR / 'cep290_features_by_genotype.png'}")

print("\nGenerating plot: curvature faceted by genotype (rows) x experiment (cols)...")
figs_curvature = plot_feature_over_time(
    df,
    features='baseline_deviation_normalized',
    color_by='genotype',
    color_lookup=color_lookup,
    facet_row='genotype',
    facet_col='experiment_id',
    backend='both',
    show_individual=True,
    show_error_band=True,
    error_type='iqr',
)

figs_curvature['plotly'].write_html(OUTPUT_DIR / 'cep290_curvature_by_genotype_x_experiment.html')
figs_curvature['matplotlib'].savefig(OUTPUT_DIR / 'cep290_curvature_by_genotype_x_experiment.png', dpi=300, bbox_inches='tight')

print(f"\n✓ Done!")
print(f"  HTML: {OUTPUT_DIR / 'cep290_curvature_by_genotype_x_experiment.html'}")
print(f"  PNG:  {OUTPUT_DIR / 'cep290_curvature_by_genotype_x_experiment.png'}")
