"""
Tutorial 08: Difference Detection (Multiclass)

Demonstrates statistical comparison between phenotype clusters using
classification-based testing.

Key comparisons:
1. One-vs-Rest: Each cluster vs all others
2. Phenotype clusters vs Not Penetrant (wildtype-like)
3. Within Not Penetrant: Het vs WT

Key API usage:
- run_classification_test() (NEW NAME, renamed from run_comparison_test)
- Time-resolved AUROC for detecting group separability
- Permutation testing for significance
"""

import pandas as pd
import numpy as np
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
embryo_labels = membership_df[['embryo_id', 'cluster_label']].copy()
df = df.merge(embryo_labels, on='embryo_id', how='left')
df = df.dropna(subset=['cluster_label'])

print(f"Loaded {len(df['embryo_id'].unique())} embryos with cluster labels")

# ============================================================================
# Import difference detection API (NEW NAME)
# ============================================================================
from src.analyze.difference_detection import run_classification_test

# ============================================================================
# Comparison 1: One-vs-Rest for all cluster labels
# ============================================================================
print("\n" + "="*80)
print("COMPARISON 1: ONE-VS-REST FOR ALL CLUSTERS")
print("="*80)

print("\nRunning classification test: Each cluster vs all others...")
print("(This may take several minutes...)")

results_ovr = run_classification_test(
    df,
    groupby='cluster_label',
    groups='all',  # Test all groups
    reference='rest',  # Each vs all others
    features='z_mu_b',  # Use VAE embeddings
    bin_width=4.0,  # Time bin width in hours
    n_permutations=100,  # Number of permutation iterations
    n_splits=5,  # Cross-validation splits
    n_jobs=4,  # Parallel jobs
)

print("\n✓ One-vs-Rest comparison complete!")
print(f"  Comparisons: {len(results_ovr.comparisons)}")
print(f"  Time bins: {results_ovr.comparisons['time_bin'].nunique()}")

# Save results
results_ovr.comparisons.to_csv(RESULTS_DIR / "comparison_ovr.csv", index=False)
print(f"  Saved to: {RESULTS_DIR / 'comparison_ovr.csv'}")

# Preview results
print("\nResults preview (first few time bins):")
print(results_ovr.comparisons.head(10))

# ============================================================================
# Comparison 2: Phenotype clusters vs Not Penetrant
# ============================================================================
print("\n" + "="*80)
print("COMPARISON 2: PHENOTYPE CLUSTERS VS NOT PENETRANT")
print("="*80)

print("\nComparing 'Short Body Axis' and 'Homozygous B9D2' vs 'Not Penetrant'...")

results_vs_np = run_classification_test(
    df,
    groupby='cluster_label',
    groups=['Short Body Axis', 'Homozygous B9D2'],
    reference='Not Penetrant',
    features='z_mu_b',
    bin_width=4.0,
    n_permutations=100,
    n_splits=5,
    n_jobs=4,
)

print("\n✓ Phenotype vs Not Penetrant comparison complete!")

# Save results
results_vs_np.comparisons.to_csv(RESULTS_DIR / "comparison_phenotypes_vs_not_penetrant.csv", index=False)
print(f"  Saved to: {RESULTS_DIR / 'comparison_phenotypes_vs_not_penetrant.csv'}")

# ============================================================================
# Comparison 3: Within Not Penetrant - Het vs WT
# ============================================================================
print("\n" + "="*80)
print("COMPARISON 3: WITHIN NOT PENETRANT - HET VS WT")
print("="*80)

print("\nFiltering to 'Not Penetrant' embryos only...")
df_np = df[df['cluster_label'] == 'Not Penetrant'].copy()
print(f"  {len(df_np['embryo_id'].unique())} embryos in Not Penetrant cluster")

# Check genotype distribution
print("\nGenotype distribution in Not Penetrant:")
print(df_np.drop_duplicates('embryo_id')['genotype'].value_counts())

print("\nComparing heterozygous vs wildtype within Not Penetrant...")

results_het_vs_wt = run_classification_test(
    df_np,
    groupby='genotype',
    groups='b9d2_heterozygous',
    reference='b9d2_wildtype',
    features='z_mu_b',
    bin_width=4.0,
    n_permutations=100,
    n_splits=5,
    n_jobs=4,
)

print("\n✓ Het vs WT comparison complete!")

# Save results
results_het_vs_wt.comparisons.to_csv(RESULTS_DIR / "comparison_het_vs_wt_in_not_penetrant.csv", index=False)
print(f"  Saved to: {RESULTS_DIR / 'comparison_het_vs_wt_in_not_penetrant.csv'}")

# ============================================================================
# Summary of results
# ============================================================================
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print("\n1. One-vs-Rest comparisons:")
print(f"   Tested {results_ovr.comparisons['group'].nunique()} clusters")
print(f"   Time range: {results_ovr.comparisons['time_bin'].min():.1f} - {results_ovr.comparisons['time_bin'].max():.1f} hpf")
print(f"   Mean AUROC range: {results_ovr.comparisons['auroc_mean'].min():.3f} - {results_ovr.comparisons['auroc_mean'].max():.3f}")

print("\n2. Phenotype vs Not Penetrant:")
print(f"   Mean AUROC range: {results_vs_np.comparisons['auroc_mean'].min():.3f} - {results_vs_np.comparisons['auroc_mean'].max():.3f}")

print("\n3. Het vs WT (within Not Penetrant):")
print(f"   Mean AUROC range: {results_het_vs_wt.comparisons['auroc_mean'].min():.3f} - {results_het_vs_wt.comparisons['auroc_mean'].max():.3f}")

# ============================================================================
# Interpretation guide
# ============================================================================
print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
Results structure:
- .comparisons: DataFrame with AUROC per time bin
- .metadata: Analysis parameters

Key columns in comparisons DataFrame:
- time_bin: Time bin center (hpf)
- group: Group being tested
- reference: Reference group
- auroc_mean: Mean AUROC across cross-validation splits
- auroc_std: Standard deviation of AUROC
- p_value: P-value from permutation test

Interpretation:
- AUROC > 0.5: Groups are separable (model can distinguish them)
- AUROC = 0.5: Groups are indistinguishable (random chance)
- AUROC < 0.5: Inverse relationship (should not occur in one-sided tests)
- p_value < 0.05: Difference is statistically significant

Time-resolved analysis:
- AUROC curves over time show WHEN phenotypes diverge
- Early divergence: Groups differ from early stages
- Late divergence: Groups start similar, diverge later
""")

print("\n✓ Tutorial 08 complete!")
print(f"  Results saved to: {RESULTS_DIR}")

# ============================================================================
# Save metadata for reference
# ============================================================================
metadata_summary = {
    'ovr': results_ovr.metadata,
    'phenotypes_vs_np': results_vs_np.metadata,
    'het_vs_wt': results_het_vs_wt.metadata,
}

import pickle
with open(RESULTS_DIR / "comparison_metadata.pkl", 'wb') as f:
    pickle.dump(metadata_summary, f)
print(f"  Metadata saved to: {RESULTS_DIR / 'comparison_metadata.pkl'}")
