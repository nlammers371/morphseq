#!/usr/bin/env python3
"""
Recompute distances for WT and Homo embryos across FULL timeline.

The existing distances file only has WT and Homo up to 72 hpf, but Het goes to 120 hpf.
This script recomputes distances for ALL available timepoints to enable full time-matrix analysis.

Approach:
1. Load ALL raw data (no time filtering)
2. Bin by 2 hpf
3. Compute WT reference per time bin
4. Compute Euclidean distances for WT and Homo at ALL time bins
5. Replace existing distances in the file
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).parent.parent / '20251016'))

from utils.data_loading import load_experiments
from utils.binning import bin_embryos_by_time
import config

# ============================================================================
# Configuration
# ============================================================================

GENE = 'cep290'
DATA_DIR = Path(__file__).parent / 'data' / 'penetrance'
DISTANCES_FILE = DATA_DIR / f'{GENE}_distances.csv'

# ============================================================================
# Distance Computation
# ============================================================================

def compute_euclidean_distance(X, mu_ref):
    """Compute Euclidean distance from reference centroid."""
    if X.ndim == 1:
        X = X.reshape(1, -1)
    distances = np.sqrt(np.sum((X - mu_ref)**2, axis=1))
    return distances


def main():
    print("\n" + "="*80)
    print("RECOMPUTING WT AND HOMO DISTANCES FOR FULL TIMELINE")
    print("="*80)

    # ------------------------------------------------------------------------
    # Step 1: Backup existing file
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: BACKING UP EXISTING DISTANCES")
    print("="*80)

    if not DISTANCES_FILE.exists():
        print(f"ERROR: Distances file not found: {DISTANCES_FILE}")
        return

    df_existing = pd.read_csv(DISTANCES_FILE)
    print(f"\nExisting distances:")
    print(f"  Total rows: {len(df_existing)}")
    for genotype in df_existing['genotype'].unique():
        subset = df_existing[df_existing['genotype'] == genotype]
        print(f"  {genotype}: {len(subset)} rows, time {subset['time_bin'].min():.1f}-{subset['time_bin'].max():.1f} hpf")

    backup_file = DISTANCES_FILE.parent / f'{GENE}_distances_backup_before_recompute.csv'
    print(f"\nBacking up to: {backup_file}")
    df_existing.to_csv(backup_file, index=False)

    # ------------------------------------------------------------------------
    # Step 2: Load ALL raw data
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: LOADING ALL RAW DATA")
    print("="*80)

    print(f"\nLoading CEP290 experiments: {config.CEP290_EXPERIMENTS}")

    df_raw = load_experiments(
        experiment_ids=config.CEP290_EXPERIMENTS,
        build_dir=config.BUILD06_DIR,
        verbose=False
    )

    print(f"\nTotal raw data: {len(df_raw)} timepoints")
    print(f"Genotypes:")
    for genotype in df_raw['genotype'].unique():
        subset = df_raw[df_raw['genotype'] == genotype]
        print(f"  {genotype}: {len(subset)} timepoints, {subset['embryo_id'].nunique()} embryos")
        # Raw data uses 'time_bin' column, not 'time_hpf'
        if 'time_bin' in subset.columns:
            print(f"    Time range: {subset['time_bin'].min():.1f} - {subset['time_bin'].max():.1f} hpf")
        elif 'time_hpf' in subset.columns:
            print(f"    Time range: {subset['time_hpf'].min():.1f} - {subset['time_hpf'].max():.1f} hpf")

    # Filter genotypes
    df_wt = df_raw[df_raw['genotype'] == 'cep290_wildtype'].copy()
    df_het = df_raw[df_raw['genotype'] == 'cep290_heterozygous'].copy()
    df_homo = df_raw[df_raw['genotype'] == 'cep290_homozygous'].copy()

    # Bin data
    print(f"\nBinning by time (2 hpf bins)...")
    df_wt_binned = bin_embryos_by_time(df_wt, bin_width=2.0)
    df_het_binned = bin_embryos_by_time(df_het, bin_width=2.0)
    df_homo_binned = bin_embryos_by_time(df_homo, bin_width=2.0)

    print(f"  WT: {len(df_wt_binned)} timepoints, time {df_wt_binned['time_bin'].min():.1f}-{df_wt_binned['time_bin'].max():.1f} hpf")
    print(f"  Het: {len(df_het_binned)} timepoints, time {df_het_binned['time_bin'].min():.1f}-{df_het_binned['time_bin'].max():.1f} hpf")
    print(f"  Homo: {len(df_homo_binned)} timepoints, time {df_homo_binned['time_bin'].min():.1f}-{df_homo_binned['time_bin'].max():.1f} hpf")

    # ------------------------------------------------------------------------
    # Step 3: Compute WT reference per time bin
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: COMPUTING WT REFERENCE DISTRIBUTIONS")
    print("="*80)

    # Get embedding columns
    z_cols = [col for col in df_wt_binned.columns if col.startswith('z_mu_b_') and col.endswith('_binned')]

    if len(z_cols) == 0:
        print("ERROR: No embedding columns found!")
        return

    print(f"\nEmbedding dimensions: {len(z_cols)}")

    # Compute WT mean per time bin
    wt_refs = {}

    for time_bin in sorted(df_wt_binned['time_bin'].unique()):
        wt_at_time = df_wt_binned[df_wt_binned['time_bin'] == time_bin]

        if len(wt_at_time) < 2:
            continue

        # Compute mean
        mu = wt_at_time[z_cols].mean().values
        wt_refs[time_bin] = mu

    print(f"\nWT references computed for {len(wt_refs)} time bins")
    print(f"  Time range: {min(wt_refs.keys()):.1f} - {max(wt_refs.keys()):.1f} hpf")

    # ------------------------------------------------------------------------
    # Step 4: Compute distances for ALL genotypes
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: COMPUTING DISTANCES FOR ALL TIMEPOINTS")
    print("="*80)

    all_distances = []

    for genotype_name, df_binned in [
        ('cep290_wildtype', df_wt_binned),
        ('cep290_heterozygous', df_het_binned),
        ('cep290_homozygous', df_homo_binned)
    ]:
        print(f"\nProcessing {genotype_name}...")
        genotype_distances = []

        for time_bin in sorted(df_binned['time_bin'].unique()):
            if time_bin not in wt_refs:
                print(f"  Skipping {time_bin:.1f} hpf (no WT reference)")
                continue

            data_at_time = df_binned[df_binned['time_bin'] == time_bin]

            if len(data_at_time) == 0:
                continue

            # Get embeddings
            X = data_at_time[z_cols].values
            mu_wt = wt_refs[time_bin]

            # Compute distances
            distances = compute_euclidean_distance(X, mu_wt)

            # Store
            for i, (idx, row) in enumerate(data_at_time.iterrows()):
                genotype_distances.append({
                    'embryo_id': row['embryo_id'],
                    'time_bin': time_bin,
                    'genotype': genotype_name,
                    'euclidean_distance': distances[i]
                })

        print(f"  Computed {len(genotype_distances)} distances")
        if len(genotype_distances) > 0:
            df_gen = pd.DataFrame(genotype_distances)
            print(f"    Time range: {df_gen['time_bin'].min():.1f} - {df_gen['time_bin'].max():.1f} hpf")
            print(f"    Embryos: {df_gen['embryo_id'].nunique()}")

        all_distances.extend(genotype_distances)

    print(f"\n✓ Computed {len(all_distances)} total distances")

    # ------------------------------------------------------------------------
    # Step 5: Save updated distances
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: SAVING UPDATED DISTANCES")
    print("="*80)

    df_new_distances = pd.DataFrame(all_distances)

    print(f"\nNew distances file:")
    print(f"  Total rows: {len(df_new_distances)}")
    for genotype in df_new_distances['genotype'].unique():
        subset = df_new_distances[df_new_distances['genotype'] == genotype]
        print(f"  {genotype}: {len(subset)} rows, time {subset['time_bin'].min():.1f}-{subset['time_bin'].max():.1f} hpf")

    print(f"\nSaving to: {DISTANCES_FILE}")
    df_new_distances.to_csv(DISTANCES_FILE, index=False)

    # ------------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  ✓ Recomputed distances for full timeline")
    print(f"  ✓ Updated: {DISTANCES_FILE}")
    print(f"  ✓ Backup: {backup_file}")
    print("\nComparison:")
    print(f"  Before: {len(df_existing)} rows")
    print(f"  After:  {len(df_new_distances)} rows")
    print(f"  Change: {len(df_new_distances) - len(df_existing):+d} rows")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
