#!/usr/bin/env python3
"""
Check which embryos are in the POOLED het vs WT comparison
"""

import sys
from pathlib import Path
import pandas as pd

# Import everything from the extended script
from b9d2_phenotype_comparison_extended import (
    load_experiment_data,
    parse_phenotype_file,
    EMBRYO_ID_COL,
    GENOTYPE_COL,
)

# Load phenotypes
phenotype_dir = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251219_b9d2_phenotype_extraction/phenotype_lists')
phenotypes = {
    'CE': parse_phenotype_file(phenotype_dir / 'b9d2-CE-phenotype.txt'),
    'HTA': parse_phenotype_file(phenotype_dir / 'b9d2-HTA-embryos.txt'),
    'BA_rescue': parse_phenotype_file(phenotype_dir / 'b9d2-curved-rescue.txt')
}
all_phenotype_ids = set(phenotypes['CE'] + phenotypes['HTA'] + phenotypes['BA_rescue'])

# Load data
df_raw = load_experiment_data()

# Pooled non-penetrant hets (all pairs, excluding CE/HTA/BA-rescue)
het_mask = df_raw[GENOTYPE_COL] == 'b9d2_heterozygous'
all_hets = set(df_raw[het_mask][EMBRYO_ID_COL].unique())
het_nonpen_pooled = list(all_hets - all_phenotype_ids)

# Pooled WTs (all pairs, excluding CE/HTA/BA-rescue)
wt_mask = df_raw[GENOTYPE_COL] == 'b9d2_wildtype'
all_wt = set(df_raw[wt_mask][EMBRYO_ID_COL].unique())
wt_pooled = list(all_wt - all_phenotype_ids)

print("="*80)
print("POOLED HET NON-PENETRANT vs WT COMPARISON DATA BREAKDOWN")
print("="*80)

print(f"\n[1] Total non-penetrant hets: {len(het_nonpen_pooled)}")
print(f"[2] Total WTs: {len(wt_pooled)}")

# Break down by experiment and pair
het_df = df_raw[df_raw[EMBRYO_ID_COL].isin(het_nonpen_pooled)][[EMBRYO_ID_COL, 'experiment_id', 'pair']].drop_duplicates()
wt_df = df_raw[df_raw[EMBRYO_ID_COL].isin(wt_pooled)][[EMBRYO_ID_COL, 'experiment_id', 'pair']].drop_duplicates()

print("\n" + "="*80)
print("NON-PENETRANT HETS BREAKDOWN BY EXPERIMENT AND PAIR")
print("="*80)
for exp in ['20251121', '20251125']:
    het_exp = het_df[het_df['experiment_id'] == exp]
    print(f"\nExperiment {exp}: {len(het_exp)} hets")
    for pair in sorted(het_exp['pair'].unique()):
        pair_count = len(het_exp[het_exp['pair'] == pair])
        print(f"  {pair}: {pair_count} hets")

print("\n" + "="*80)
print("WTs BREAKDOWN BY EXPERIMENT AND PAIR")
print("="*80)
for exp in ['20251121', '20251125']:
    wt_exp = wt_df[wt_df['experiment_id'] == exp]
    print(f"\nExperiment {exp}: {len(wt_exp)} WTs")
    for pair in sorted(wt_exp['pair'].unique()):
        pair_count = len(wt_exp[wt_exp['pair'] == pair])
        print(f"  {pair}: {pair_count} WTs")

# Check early timepoints specifically
print("\n" + "="*80)
print("EARLY TIMEPOINT DATA (8-20 hpf)")
print("="*80)

early_data = df_raw[(df_raw['hpf'] >= 8) & (df_raw['hpf'] <= 20)]
het_early = early_data[early_data[EMBRYO_ID_COL].isin(het_nonpen_pooled)]
wt_early = early_data[early_data[EMBRYO_ID_COL].isin(wt_pooled)]

print(f"\nNon-penetrant hets with data at 8-20 hpf:")
print(f"  Unique embryos: {het_early[EMBRYO_ID_COL].nunique()}")
print(f"  Total datapoints: {len(het_early)}")

print(f"\nWTs with data at 8-20 hpf:")
print(f"  Unique embryos: {wt_early[EMBRYO_ID_COL].nunique()}")
print(f"  Total datapoints: {len(wt_early)}")

# Break down by specific time bins
for time_bin in [8, 12, 16, 20]:
    het_bin = het_early[(het_early['hpf'] >= time_bin) & (het_early['hpf'] < time_bin + 4)]
    wt_bin = wt_early[(wt_early['hpf'] >= time_bin) & (wt_early['hpf'] < time_bin + 4)]
    print(f"\n  {time_bin} hpf bin: {het_bin[EMBRYO_ID_COL].nunique()} hets, {wt_bin[EMBRYO_ID_COL].nunique()} WTs")

    # Show breakdown by pair
    print(f"    Hets by pair:")
    het_bin_df = het_bin[[EMBRYO_ID_COL, 'pair']].drop_duplicates()
    for pair in sorted(het_bin_df['pair'].unique()):
        count = len(het_bin_df[het_bin_df['pair'] == pair])
        print(f"      {pair}: {count}")

    print(f"    WTs by pair:")
    wt_bin_df = wt_bin[[EMBRYO_ID_COL, 'pair']].drop_duplicates()
    for pair in sorted(wt_bin_df['pair'].unique()):
        count = len(wt_bin_df[wt_bin_df['pair'] == pair])
        print(f"      {pair}: {count}")
