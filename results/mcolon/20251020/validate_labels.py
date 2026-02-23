#!/usr/bin/env python3
"""
Diagnostic script to validate labels and check correlation direction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "20251016"))

import pandas as pd
import numpy as np

# Load the saved data
data_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251020/data/penetrance")

print("=" * 80)
print("LABEL VALIDATION DIAGNOSTIC")
print("=" * 80)

# Check CEP290
print("\n" + "=" * 80)
print("CEP290 DATA")
print("=" * 80)

# Load distances
df_dist = pd.read_csv(data_dir / "cep290_distances.csv")
print(f"\nDistances loaded: {len(df_dist)} rows")
print(f"Genotypes in distance data: {df_dist['genotype'].unique()}")

# Summary stats by genotype
print("\nDistance summary by genotype:")
for genotype in df_dist['genotype'].unique():
    subset = df_dist[df_dist['genotype'] == genotype]
    mean_dist = subset['euclidean_distance'].mean()
    std_dist = subset['euclidean_distance'].std()
    print(f"  {genotype}: mean={mean_dist:.3f}, std={std_dist:.3f}, n={len(subset)}")

# Load predictions
df_pred = pd.read_csv(data_dir / "cep290_predictions.csv")
print(f"\nPredictions loaded: {len(df_pred)} rows")
print(f"Prediction columns: {df_pred.columns.tolist()}")
print(f"Unique true_label values: {df_pred['true_label'].unique()}")

# Check if pred_prob_mutant exists
if 'pred_prob_mutant' in df_pred.columns:
    print("\n✓ pred_prob_mutant column exists")
else:
    print("\n✗ pred_prob_mutant column MISSING - fix didn't apply!")

# Summary stats by true label
print("\nPrediction summary by true_label:")
for label in df_pred['true_label'].unique():
    subset = df_pred[df_pred['true_label'] == label]

    if 'pred_prob_mutant' in df_pred.columns:
        mean_prob_mut = subset['pred_prob_mutant'].mean()
        print(f"  {label}:")
        print(f"    mean pred_proba (original): {subset['pred_proba'].mean():.3f}")
        print(f"    mean pred_prob_mutant (corrected): {mean_prob_mut:.3f}")
        print(f"    n={len(subset)}")
    else:
        mean_prob = subset['pred_proba'].mean()
        print(f"  {label}: mean pred_proba={mean_prob:.3f}, n={len(subset)}")

# Load per-embryo metrics
df_embryo = pd.read_csv(data_dir / "cep290_homozygous_per_embryo_metrics.csv")
print(f"\nPer-embryo metrics loaded: {len(df_embryo)} embryos")
print(f"Columns: {df_embryo.columns.tolist()}")

# Show sample of embryos
print("\nSample of per-embryo data (first 5 embryos):")
print(df_embryo[['embryo_id', 'mean_distance', 'mean_prob']].head())

# Manual correlation check
print("\n" + "=" * 80)
print("MANUAL CORRELATION CHECK")
print("=" * 80)

from scipy.stats import pearsonr, spearmanr

distances = df_embryo['mean_distance'].values
probs = df_embryo['mean_prob'].values

r_pearson, p_pearson = pearsonr(distances, probs)
r_spearman, p_spearman = spearmanr(distances, probs)

print(f"\nCorrelation between mean_distance and mean_prob:")
print(f"  Pearson r:  {r_pearson:.4f} (p={p_pearson:.3e})")
print(f"  Spearman ρ: {r_spearman:.4f} (p={p_spearman:.3e})")

# Interpretation
print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if r_pearson > 0.5:
    print("\n✓ POSITIVE strong correlation (r > 0.5)")
    print("  → Embryos farther from WT have HIGHER mutant probability")
    print("  → Distance is a good phenotypic readout")
    print("  → Labels are CORRECT")
elif r_pearson > 0:
    print(f"\n⚠ POSITIVE weak correlation (r = {r_pearson:.3f})")
    print("  → Embryos farther from WT have HIGHER mutant probability (weakly)")
    print("  → Distance has some phenotypic signal but it's weak")
    print("  → Labels are CORRECT but signal is noisy")
elif r_pearson < -0.5:
    print("\n✗ NEGATIVE strong correlation (r < -0.5)")
    print("  → Embryos farther from WT have LOWER mutant probability")
    print("  → This is BACKWARDS! Labels are FLIPPED")
    print("  → Need to invert either distance or probability")
else:
    print(f"\n⚠ NEGATIVE weak correlation (r = {r_pearson:.3f})")
    print("  → Embryos farther from WT have LOWER mutant probability (weakly)")
    print("  → Labels might be flipped OR signal is very weak")

# Sanity check: WT should have lower distance than homozygous
print("\n" + "=" * 80)
print("SANITY CHECK: Do homozygous embryos have higher distance than WT?")
print("=" * 80)

wt_dist = df_dist[df_dist['genotype'] == 'cep290_wildtype']['euclidean_distance'].mean()
hom_dist = df_dist[df_dist['genotype'] == 'cep290_homozygous']['euclidean_distance'].mean()

print(f"\nMean distance to WT reference:")
print(f"  cep290_wildtype:    {wt_dist:.3f}")
print(f"  cep290_homozygous:  {hom_dist:.3f}")
print(f"  Difference:         {hom_dist - wt_dist:.3f}")

if hom_dist > wt_dist:
    print("\n✓ CORRECT: Homozygous embryos are farther from WT (as expected)")
else:
    print("\n✗ PROBLEM: Homozygous embryos are CLOSER to WT than WT is to itself!")
    print("  → This suggests reference calculation issue")

# Sanity check: homozygous should have higher mutant probability than WT
print("\n" + "=" * 80)
print("SANITY CHECK: Do homozygous embryos have higher mutant probability?")
print("=" * 80)

if 'pred_prob_mutant' in df_pred.columns:
    wt_prob = df_pred[df_pred['true_label'] == 'cep290_wildtype']['pred_prob_mutant'].mean()
    hom_prob = df_pred[df_pred['true_label'] == 'cep290_homozygous']['pred_prob_mutant'].mean()

    print(f"\nMean predicted mutant probability:")
    print(f"  cep290_wildtype:    {wt_prob:.3f}")
    print(f"  cep290_homozygous:  {hom_prob:.3f}")
    print(f"  Difference:         {hom_prob - wt_prob:.3f}")

    if hom_prob > wt_prob:
        print("\n✓ CORRECT: Homozygous embryos have higher mutant probability (as expected)")
    else:
        print("\n✗ PROBLEM: Homozygous embryos have LOWER mutant probability than WT!")
        print("  → Probability labels are FLIPPED")
else:
    print("\n⚠ Cannot check - pred_prob_mutant column not found")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
