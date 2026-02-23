"""
Debug script to test coloring on a small subset of data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

from analyze.trajectory_analysis.data_loading import load_experiment_dataframe

# Load a small subset
print("Loading data...")
df = load_experiment_dataframe('20251121', format_version='qc_staged')

# Filter to b9d2 and a few embryos for testing
df_b9d2 = df[df['genotype'].str.contains('b9d2', na=False)].copy()

# Take just 10 embryos
embryos = df_b9d2['embryo_id'].unique()[:10]
df_subset = df_b9d2[df_b9d2['embryo_id'].isin(embryos)].copy()

# Check what values we have
print(f"\nSubset: {len(df_subset)} rows, {df_subset['embryo_id'].nunique()} embryos")
print(f"\nGenotype distribution:")
print(df_subset.groupby('embryo_id')['genotype'].first().value_counts())

if 'pair' in df_subset.columns:
    print(f"\nPair distribution:")
    print(df_subset.groupby('embryo_id')['pair'].first().value_counts())

# Add a fake cluster for testing
# Let's assign half to cluster 0 and half to cluster 1
np.random.seed(42)
embryo_clusters = {e: np.random.choice([0, 1]) for e in embryos}
df_subset['test_cluster'] = df_subset['embryo_id'].map(embryo_clusters)

print(f"\nCluster distribution:")
print(df_subset.groupby('embryo_id')['test_cluster'].first().value_counts())

# Show a few rows
print(f"\nSample rows:")
print(df_subset[['embryo_id', 'genotype', 'pair', 'test_cluster']].head(20))

# Save subset for plotting test
output_file = Path(__file__).parent / 'debug_subset.csv'
df_subset.to_csv(output_file, index=False)
print(f"\nSaved subset to: {output_file}")
