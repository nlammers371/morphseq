#!/usr/bin/env python3
"""
Compute distances for heterozygous embryos.

Adds heterozygous embryo distances to the existing cep290_distances.csv file.
Uses the same WT reference distribution and distance metric as the original.
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
    """
    Compute Euclidean distance from reference centroid.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data points
    mu_ref : np.ndarray, shape (n_features,)
        Reference mean

    Returns
    -------
    np.ndarray, shape (n_samples,)
        Distances
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    distances = np.sqrt(np.sum((X - mu_ref)**2, axis=1))
    return distances


def main():
    print("\n" + "="*80)
    print("COMPUTING DISTANCES FOR HETEROZYGOUS EMBRYOS")
    print("="*80)

    # ------------------------------------------------------------------------
    # Step 1: Load existing distances to get WT reference
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING EXISTING DISTANCES")
    print("="*80)

    if not DISTANCES_FILE.exists():
        print(f"ERROR: Distances file not found: {DISTANCES_FILE}")
        return

    df_existing = pd.read_csv(DISTANCES_FILE)
    print(f"\nExisting distances file:")
    print(f"  Total rows: {len(df_existing)}")
    print(f"  Genotypes: {df_existing['genotype'].unique()}")
    print(f"  Time bins: {sorted(df_existing['time_bin'].unique())}")

    # ------------------------------------------------------------------------
    # Step 2: Load WT and Het data
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: LOADING RAW DATA")
    print("="*80)

    print(f"\nLoading CEP290 experiments: {config.CEP290_EXPERIMENTS}")

    df_raw = load_experiments(
        experiment_ids=config.CEP290_EXPERIMENTS,
        build_dir=config.BUILD06_DIR,
        verbose=False
    )

    print(f"\nTotal raw data loaded: {len(df_raw)} timepoints")
    print(f"  Genotypes: {df_raw['genotype'].value_counts().to_dict()}")

    # Filter genotypes
    df_wt = df_raw[df_raw['genotype'] == 'cep290_wildtype'].copy()
    df_het = df_raw[df_raw['genotype'] == 'cep290_heterozygous'].copy()

    print(f"\nWT: {len(df_wt)} timepoints, {df_wt['embryo_id'].nunique()} embryos")
    print(f"Het: {len(df_het)} timepoints, {df_het['embryo_id'].nunique()} embryos")

    if len(df_het) == 0:
        print("\nERROR: No heterozygous data found!")
        return

    # Bin data
    print(f"\nBinning by time (2 hpf bins)...")
    df_wt_binned = bin_embryos_by_time(df_wt, bin_width=2.0)
    df_het_binned = bin_embryos_by_time(df_het, bin_width=2.0)

    print(f"  WT binned: {len(df_wt_binned)} timepoints")
    print(f"  Het binned: {len(df_het_binned)} timepoints")

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
    print(f"  Time bins: {sorted(wt_refs.keys())}")

    # ------------------------------------------------------------------------
    # Step 4: Compute distances for Het embryos
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: COMPUTING HET DISTANCES")
    print("="*80)

    het_distances = []

    for time_bin in sorted(df_het_binned['time_bin'].unique()):
        if time_bin not in wt_refs:
            print(f"  Skipping {time_bin} hpf (no WT reference)")
            continue

        het_at_time = df_het_binned[df_het_binned['time_bin'] == time_bin]

        if len(het_at_time) == 0:
            continue

        # Get embeddings
        X_het = het_at_time[z_cols].values
        mu_wt = wt_refs[time_bin]

        # Compute Euclidean distance
        distances = compute_euclidean_distance(X_het, mu_wt)

        # Store
        for i, (idx, row) in enumerate(het_at_time.iterrows()):
            het_distances.append({
                'embryo_id': row['embryo_id'],
                'time_bin': time_bin,
                'genotype': 'cep290_heterozygous',
                'euclidean_distance': distances[i]
            })

        print(f"  {time_bin} hpf: {len(het_at_time)} Het embryos, mean distance = {distances.mean():.4f}")

    print(f"\n✓ Computed {len(het_distances)} Het distances")

    # ------------------------------------------------------------------------
    # Step 5: Merge with existing distances
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: MERGING WITH EXISTING DISTANCES")
    print("="*80)

    df_het_distances = pd.DataFrame(het_distances)

    print(f"\nNew Het distances:")
    print(f"  Rows: {len(df_het_distances)}")
    print(f"  Embryos: {df_het_distances['embryo_id'].nunique()}")
    print(f"  Time range: {df_het_distances['time_bin'].min():.1f} - {df_het_distances['time_bin'].max():.1f} hpf")

    # Combine
    df_all_distances = pd.concat([df_existing, df_het_distances], ignore_index=True)

    print(f"\nCombined distances:")
    print(f"  Total rows: {len(df_all_distances)}")
    print(f"  Genotypes: {df_all_distances['genotype'].value_counts().to_dict()}")

    # Save
    backup_file = DISTANCES_FILE.parent / f'{GENE}_distances_backup.csv'
    print(f"\nBacking up original to: {backup_file}")
    df_existing.to_csv(backup_file, index=False)

    print(f"Saving updated distances to: {DISTANCES_FILE}")
    df_all_distances.to_csv(DISTANCES_FILE, index=False)

    # ------------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nSummary:")
    print(f"  Added {len(df_het_distances)} heterozygous distances")
    print(f"  Total distances now: {len(df_all_distances)}")
    print(f"  ✓ Updated: {DISTANCES_FILE}")
    print(f"  ✓ Backup: {backup_file}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
