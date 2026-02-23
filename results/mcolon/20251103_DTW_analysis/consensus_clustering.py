#!/usr/bin/env python3
"""
Consensus Clustering from Bootstrap Co-occurrence Matrix

This script implements the FORWARD approach to robust clustering:
  Bootstrap → Consensus Clustering → Membership Classification

Instead of computing baseline clusters then validating with bootstrap,
we use bootstrap directly to find clusters that are stable across resampling.

The consensus labels are the PRIMARY output, not a secondary validation.
Outliers are identified from co-occurrence patterns, not post-hoc.

Usage
-----
python consensus_clustering.py

Configuration is read from config.py (K_VALUES, N_BOOTSTRAP, etc.)
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import (
    OUTPUT_DIR, K_VALUES, N_BOOTSTRAP, BOOTSTRAP_FRAC,
    CORE_THRESHOLD, OUTLIER_THRESHOLD, RANDOM_SEED, VERBOSE_OUTPUT
)

# Import pipeline modules (handling hyphenated filenames)
import importlib.util

def load_module(name, filepath):
    """Load a module from a file path (handles hyphens in filenames)."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules with hyphens in names
cluster_module = load_module("cluster_module", "cluster-module.py")
select_k_module = load_module("select_k_module", "select-k-simple.py")
membership_module = load_module("membership_module", "membership-module.py")
io_module = load_module("io_module", "io-module.py")
dtw_precompute_module = load_module("dtw_precompute", "0_dtw_precompute.py")

# Extract functions
run_bootstrap = cluster_module.run_bootstrap
consensus_clustering = select_k_module.consensus_clustering
analyze_membership = membership_module.analyze_membership
load_data = io_module.load_data
save_data = io_module.save_data
precompute_dtw = dtw_precompute_module.precompute_dtw

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONSENSUS CLUSTERING PIPELINE
# ============================================================================

def compute_consensus_clustering(D, K_VALUES, OUTPUT_DIR, core_threshold=CORE_THRESHOLD,
                                 n_bootstrap=N_BOOTSTRAP, frac=BOOTSTRAP_FRAC,
                                 verbose=True):
    """
    Find robust consensus clusters from DTW distance matrix via bootstrap.

    This is the FORWARD approach:
      DTW Matrix → Bootstrap (100x resampling) → Co-association Matrix
      → Consensus Clustering (hierarchical on co-occurrence)
      → Membership Classification (core/uncertain/outlier)

    Parameters
    ----------
    D : np.ndarray
        DTW distance matrix (n_embryos, n_embryos)
    K_VALUES : list
        k values to cluster for (e.g., [2, 3, 4, 5])
    OUTPUT_DIR : str or Path
        Output directory for results
    core_threshold : float
        Threshold for classifying core members (default 0.7)
    n_bootstrap : int
        Number of bootstrap iterations (default 100)
    frac : float
        Fraction of embryos to sample each iteration (default 0.8)
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results for each k: {
            k: {
                'labels': consensus cluster assignments (n_embryos,),
                'coassoc': co-association matrix (n_embryos, n_embryos),
                'membership': membership classification dict,
                'silhouette_scores': stability metrics from bootstrap,
                'ari_scores': agreement metrics from bootstrap,
                'n_core': number of core members,
                'n_uncertain': number of uncertain members,
                'n_outlier': number of outliers
            }
        }
    """
    if verbose:
        print("\n" + "="*80)
        print("CONSENSUS CLUSTERING FROM BOOTSTRAP")
        print("="*80)

    results = {}

    for k in K_VALUES:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Computing consensus clustering for k={k}")
            print(f"{'='*80}")

        # ===== STEP 1: BOOTSTRAP RESAMPLING =====
        if verbose:
            print(f"\nStep 1: Running {n_bootstrap} bootstrap iterations...")

        bootstrap_results = run_bootstrap(
            D, k,
            n_bootstrap=n_bootstrap,
            frac=frac,
            verbose=verbose
        )

        C = bootstrap_results['coassoc']  # Co-association matrix

        if verbose:
            print(f"  ✓ Co-association matrix shape: {C.shape}")
            print(f"  ✓ Co-association range: [{C.min():.3f}, {C.max():.3f}]")
            print(f"  ✓ Mean within-cluster co-occurrence: ~{bootstrap_results.get('mean_ari', 0):.3f}")

        # ===== STEP 2: CONSENSUS CLUSTERING =====
        if verbose:
            print(f"\nStep 2: Computing consensus clusters from co-association matrix...")

        consensus_labels = consensus_clustering(C, k)

        if verbose:
            cluster_sizes = np.bincount(consensus_labels)
            print(f"  ✓ Consensus cluster sizes: {cluster_sizes}")
            print(f"  ✓ Largest cluster: {cluster_sizes.max()} embryos")
            print(f"  ✓ Smallest cluster: {cluster_sizes.min()} embryos")

        # ===== STEP 3: MEMBERSHIP CLASSIFICATION =====
        if verbose:
            print(f"\nStep 3: Classifying membership (core/uncertain/outlier)...")

        membership_results = analyze_membership(
            D, consensus_labels, C,
            core_thresh=core_threshold,
            verbose=verbose
        )

        if verbose:
            summary = membership_results['summary']
            print(f"  ✓ Core members: {summary['n_core']} ({summary['core_fraction']*100:.1f}%)")
            print(f"  ✓ Uncertain: {summary['n_uncertain']} ({(summary['n_uncertain']/len(consensus_labels))*100:.1f}%)")
            print(f"  ✓ Outliers: {summary['n_outlier']} ({(summary['n_outlier']/len(consensus_labels))*100:.1f}%)")

        # ===== SAVE RESULTS =====
        if verbose:
            print(f"\nStep 4: Saving results...")

        consensus_result = {
            'labels': consensus_labels,
            'coassoc': C,
            'membership': membership_results,
            'silhouette_scores': bootstrap_results.get('silhouette_scores', None),
            'ari_scores': bootstrap_results.get('ari_scores', None),
            'n_core': membership_results['summary']['n_core'],
            'n_uncertain': membership_results['summary']['n_uncertain'],
            'n_outlier': membership_results['summary']['n_outlier']
        }

        results[k] = consensus_result

        # Save to disk
        save_data(6, f'consensus_k{k}', consensus_result, OUTPUT_DIR, verbose=verbose)

        if verbose:
            print(f"  ✓ Saved consensus results for k={k}")

    return results


def main():
    """Main entry point for consensus clustering."""
    if VERBOSE_OUTPUT:
        print("\nLoading DTW distance matrix...")

    # Load or precompute DTW matrix
    try:
        D = load_data(0, 'distance_matrix', OUTPUT_DIR, verbose=VERBOSE_OUTPUT)
        if VERBOSE_OUTPUT:
            print(f"  ✓ Loaded DTW distance matrix: {D.shape}")
    except FileNotFoundError:
        if VERBOSE_OUTPUT:
            print(f"  DTW matrix not found, precomputing...")
        precomp = precompute_dtw(verbose=VERBOSE_OUTPUT)
        D = precomp['distance_matrix']

    # Run consensus clustering for all k values
    consensus_results = compute_consensus_clustering(
        D,
        K_VALUES,
        OUTPUT_DIR,
        core_threshold=CORE_THRESHOLD,
        n_bootstrap=N_BOOTSTRAP,
        frac=BOOTSTRAP_FRAC,
        verbose=VERBOSE_OUTPUT
    )

    # Summary
    if VERBOSE_OUTPUT:
        print("\n" + "="*80)
        print("CONSENSUS CLUSTERING COMPLETE")
        print("="*80)
        for k in K_VALUES:
            if k in consensus_results:
                res = consensus_results[k]
                print(f"\nk={k}:")
                print(f"  Consensus labels: {np.unique(res['labels'], return_counts=True)[1].tolist()}")
                print(f"  Core/Uncertain/Outlier: {res['n_core']}/{res['n_uncertain']}/{res['n_outlier']}")

    return consensus_results


if __name__ == '__main__':
    results = main()
