"""
Quick test for row label functionality.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis import plot_trajectories_faceted, plot_pairs_overview

# Create synthetic data with multiple genes and pairs
np.random.seed(42)
data = []

genes = ['cep290', 'b9d2']
genotypes = ['wildtype', 'homozygous']
pairs = ['pair_1', 'pair_2']

for gene in genes:
    for genotype in genotypes:
        for pair in pairs:
            for embryo_i in range(2):
                embryo_id = f'{gene}_{genotype}_{pair}_embryo{embryo_i}'
                times = np.linspace(20, 36, 15)
                base = {'wildtype': 0, 'homozygous': 1}[genotype]
                metrics = np.cumsum(np.random.normal(base, 0.2, len(times))) / 5

                for time, metric in zip(times, metrics):
                    data.append({
                        'embryo_id': embryo_id,
                        'predicted_stage_hpf': time,
                        'baseline_deviation_normalized': metric,
                        'genotype': f'{gene}_{genotype}',
                        'pair': f'{gene}_{pair}',
                        'gene': gene,
                    })

df = pd.DataFrame(data)

print("Testing row labels with plot_trajectories_faceted...")
print("=" * 60)

# Test 1: Multi-row plot with gene as rows
print("\n1. Testing gene (rows) × genotype (columns)")
fig = plot_trajectories_faceted(
    df,
    row_by='gene',
    col_by='genotype',
    color_by='genotype',
    backend='matplotlib',
    title='Test: Row Labels (Gene × Genotype)',
)
print("✓ Matplotlib multi-row plot created")

# Test 2: Test with plot_pairs_overview (should have pairs as rows)
print("\n2. Testing plot_pairs_overview (pairs as rows)")
fig = plot_pairs_overview(
    df[df['gene'] == 'cep290'],
    backend='matplotlib',
    title='Test: CEP290 Pairs Overview with Row Labels',
)
print("✓ plot_pairs_overview with row labels created")

# Test 3: Single row (should NOT show row labels)
print("\n3. Testing single row (no row labels expected)")
fig = plot_trajectories_faceted(
    df[df['gene'] == 'cep290'],
    row_by=None,
    col_by='pair',
    color_by_grouping='genotype',
    backend='matplotlib',
    title='Test: Single Row (No Row Labels)',
)
print("✓ Single row plot created (no row labels)")

print("\n" + "=" * 60)
print("✅ All row label tests completed!")
print("Row labels should appear on the left side for multi-row plots.")
