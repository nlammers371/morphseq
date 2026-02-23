#!/usr/bin/env python3
"""
Quick script to run SET 4 comparisons (c), (d), and (e)

CRITICAL VALIDATION: Shows that pair_2 and pair_8 hets have a phenotype,
but wildtype controls do NOT (proving het signal is real, not artifact).
"""

import sys
from pathlib import Path

# Import everything from the extended script
from b9d2_phenotype_comparison_extended import (
    load_experiment_data,
    parse_phenotype_file,
    run_single_comparison_extended,
    get_non_penetrant_hets,
    get_embryos_by_pair_and_genotype,
    EMBRYO_ID_COL,
    GENOTYPE_COL,
    CLASSIFICATION_DIR,
    FIGURES_DIR
)

print("="*80)
print("RUNNING SET 4 COMPARISONS (c), (d), and (e)")
print("="*80)
print("CRITICAL VALIDATION:")
print("  (e) pair_2 het vs pair_2 WT - NEW")
print("  (c) pair_8 het vs pair_8 WT")
print("  (d) pair_2 WT vs pair_8 WT (negative control)")
print("="*80)

# Load data
print("\n[Step 1/3] Loading phenotype lists...")
phenotype_dir = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251219_b9d2_phenotype_extraction/phenotype_lists')
phenotypes = {
    'CE': parse_phenotype_file(phenotype_dir / 'b9d2-CE-phenotype.txt'),
    'HTA': parse_phenotype_file(phenotype_dir / 'b9d2-HTA-embryos.txt'),
    'BA_rescue': parse_phenotype_file(phenotype_dir / 'b9d2-curved-rescue.txt')
}
all_phenotype_ids = set(phenotypes['CE'] + phenotypes['HTA'] + phenotypes['BA_rescue'])
print(f"Total phenotype embryos: {len(all_phenotype_ids)}")

print("\n[Step 2/3] Loading experiment data...")
df_raw = load_experiment_data()
print(f"  Loaded {len(df_raw)} rows, {df_raw[EMBRYO_ID_COL].nunique()} unique embryos")

# Filter to 20251125 experiment
df_20251125 = df_raw[df_raw['experiment_id'] == '20251125'].copy()

print("\n[Step 3/3] Preparing pair-specific groups...")

# pair_2 groups
het_pair2_nonpen = get_non_penetrant_hets(df_20251125, 'b9d2_pair_2', all_phenotype_ids)
wt_pair2 = get_embryos_by_pair_and_genotype(df_20251125, 'b9d2_pair_2', 'b9d2_wildtype')
wt_pair2 = [eid for eid in wt_pair2 if eid not in all_phenotype_ids]

print(f"  pair_2 non-penetrant hets: {len(het_pair2_nonpen)}")
print(f"  pair_2 WTs: {len(wt_pair2)}")

# pair_8 groups
het_pair8_nonpen = get_non_penetrant_hets(df_20251125, 'b9d2_pair_8', all_phenotype_ids)
wt_pair8 = get_embryos_by_pair_and_genotype(df_20251125, 'b9d2_pair_8', 'b9d2_wildtype')
wt_pair8 = [eid for eid in wt_pair8 if eid not in all_phenotype_ids]

print(f"  pair_8 non-penetrant hets: {len(het_pair8_nonpen)}")
print(f"  pair_8 WTs: {len(wt_pair8)}")

# -------------------------------------------------------------------------
# Comparison (e): pair_2 non-penetrant hets vs pair_2 WTs (NEW)
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("COMPARISON (e): pair_2 het vs WT - NEW")
print("="*80)
if len(het_pair2_nonpen) >= 3 and len(wt_pair2) >= 3:
    run_single_comparison_extended(
        df_20251125, het_pair2_nonpen, wt_pair2,
        'pair2_Het_nonpen', 'pair2_WT',
        'pair2_het_nonpen_vs_WT',
        metric_col='total_length_um'
    )
else:
    print(f"  SKIPPED: Not enough embryos (need >=3 in each group)")
    print(f"    het_pair2_nonpen: {len(het_pair2_nonpen)}")
    print(f"    wt_pair2: {len(wt_pair2)}")

# -------------------------------------------------------------------------
# Comparison (c): pair_8 non-penetrant hets vs pair_8 WTs
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("COMPARISON (c): pair_8 het vs WT")
print("="*80)
if len(het_pair8_nonpen) >= 3 and len(wt_pair8) >= 3:
    run_single_comparison_extended(
        df_20251125, het_pair8_nonpen, wt_pair8,
        'pair8_Het_nonpen', 'pair8_WT',
        'pair8_het_nonpen_vs_WT',
        metric_col='total_length_um'
    )
else:
    print(f"  SKIPPED: Not enough embryos (need >=3 in each group)")
    print(f"    het_pair8_nonpen: {len(het_pair8_nonpen)}")
    print(f"    wt_pair8: {len(wt_pair8)}")

# -------------------------------------------------------------------------
# Comparison (d): pair_2 WTs vs pair_8 WTs (NEGATIVE CONTROL)
# -------------------------------------------------------------------------
print("\n" + "="*80)
print("COMPARISON (d): pair_2 WT vs pair_8 WT (NEGATIVE)")
print("="*80)
if len(wt_pair2) >= 3 and len(wt_pair8) >= 3:
    run_single_comparison_extended(
        df_20251125, wt_pair2, wt_pair8,
        'pair2_WT', 'pair8_WT',
        'NEGATIVE_pair2_WT_vs_pair8_WT',
        metric_col='total_length_um'
    )
else:
    print(f"  SKIPPED: Not enough embryos (need >=3 in each group)")
    print(f"    wt_pair2: {len(wt_pair2)}")
    print(f"    wt_pair8: {len(wt_pair8)}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print(f"\nResults saved to:")
print(f"  Classification: {CLASSIFICATION_DIR}")
print(f"  Figures: {FIGURES_DIR}")
