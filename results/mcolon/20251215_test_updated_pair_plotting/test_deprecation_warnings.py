"""
Test deprecation warnings for old function names.
"""

import sys
import warnings
import pandas as pd
import numpy as np

sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis import (
    plot_genotypes_overlaid,
    plot_all_pairs_overview,
    plot_homozygous_across_pairs,
)

# Create minimal synthetic data
np.random.seed(42)
data = []
for genotype in ['cep290_wildtype', 'cep290_homozygous']:
    for pair in ['pair_1', 'pair_2']:
        for embryo_i in range(2):
            embryo_id = f'{genotype}_{pair}_embryo{embryo_i}'
            times = np.linspace(20, 36, 10)
            metrics = np.cumsum(np.random.normal(0, 0.1, len(times))) / 5

            for time, metric in zip(times, metrics):
                data.append({
                    'embryo_id': embryo_id,
                    'predicted_stage_hpf': time,
                    'baseline_deviation_normalized': metric,
                    'genotype': genotype,
                    'pair': pair,
                })

df = pd.DataFrame(data)

print("Testing deprecation warnings...")
print("=" * 60)

# Enable all warnings
warnings.simplefilter("always", DeprecationWarning)

# Test 1: plot_genotypes_overlaid
print("\n1. Testing plot_genotypes_overlaid (deprecated)")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fig = plot_genotypes_overlaid(df, backend='matplotlib')
    if len(w) == 1 and issubclass(w[0].category, DeprecationWarning):
        print(f"✓ DeprecationWarning raised: {w[0].message}")
    else:
        print("✗ No deprecation warning!")

# Test 2: plot_all_pairs_overview
print("\n2. Testing plot_all_pairs_overview (deprecated)")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fig = plot_all_pairs_overview(df, backend='matplotlib')
    if len(w) == 1 and issubclass(w[0].category, DeprecationWarning):
        print(f"✓ DeprecationWarning raised: {w[0].message}")
    else:
        print("✗ No deprecation warning!")

# Test 3: plot_homozygous_across_pairs
print("\n3. Testing plot_homozygous_across_pairs (deprecated)")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fig = plot_homozygous_across_pairs(df, backend='matplotlib')
    if len(w) == 1 and issubclass(w[0].category, DeprecationWarning):
        print(f"✓ DeprecationWarning raised: {w[0].message}")
    else:
        print("✗ No deprecation warning!")

print("\n" + "=" * 60)
print("✅ All deprecation warning tests completed!")
