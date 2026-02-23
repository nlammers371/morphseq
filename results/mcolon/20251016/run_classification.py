#!/usr/bin/env python3
"""
Run classification-based phenotype emergence analysis.

This script uses the simplified unified interface to run the
classification approach for detecting phenotypic differences.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config_new as config
from utils.data_loading import load_experiments
from utils.binning import bin_by_embryo_time
from difference_detection import run_classification_test
from visualization import (
    plot_auroc_with_significance,
    plot_signed_margin_trajectories,
    plot_signed_margin_heatmap,
    plot_penetrance_distribution
)


def main():
    """Run classification analysis for cep290."""
    print("="*80)
    print("CLASSIFICATION-BASED PHENOTYPE EMERGENCE ANALYSIS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = load_experiments(
        experiment_ids=config.CEP290_EXPERIMENTS,
        build_dir=config.BUILD06_DIR
    )
    print(f"Loaded {len(df)} embryos")
    
    # Bin by time
    print("\nBinning by time...")
    df_binned = bin_by_embryo_time(
        df,
        time_col="predicted_stage_hpf",
        bin_width=2.0
    )
    print(f"Created {df_binned['time_bin'].nunique()} time bins")
    
    # Run classification test
    print("\nRunning classification test: wildtype vs homozygous...")
    results = run_classification_test(
        df_binned,
        group1="cep290_wildtype",
        group2="cep290_homozygous",
        n_cv_splits=5,
        n_permutations=100,  # Set via MORPHSEQ_N_PERMUTATIONS env var for more
        use_class_weights=True,
        random_state=42
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print("\nTime-resolved AUROC:")
    print(results['time_results'][['time_bin', 'AUROC_obs', 'pval', 'n_samples']])
    
    if results['onset_info']['is_significant']:
        print(f"\n✓ Onset detected at {results['onset_info']['onset_time']} hpf")
        print(f"  p-value: {results['onset_info']['pvalue']:.4f}")
    else:
        print("\n✗ No significant onset detected")
    
    if results['embryo_results'] is not None:
        print("\nPenetrance summary:")
        print(results['embryo_results'].groupby('true_label')['mean_confidence'].describe())
    
    # Save results
    print("\nSaving results...")
    output_dir = Path(config.DATA_DIR) / "cep290"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results['time_results'].to_csv(
        output_dir / "wt_vs_hom_time_results.csv",
        index=False
    )
    
    if results['embryo_results'] is not None:
        results['embryo_results'].to_csv(
            output_dir / "wt_vs_hom_embryo_penetrance.csv",
            index=False
        )
    
    # Create plots
    print("\nGenerating plots...")
    plot_dir = Path(config.PLOT_DIR) / "cep290"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # AUROC plot
    fig_auroc = plot_auroc_with_significance(
        results['time_results'],
        alpha=0.05,
        title="CEP290: Wildtype vs Homozygous"
    )
    fig_auroc.savefig(plot_dir / "wt_vs_hom_auroc.png", dpi=300, bbox_inches='tight')
    print(f"  Saved AUROC plot")
    
    # Trajectory plots
    if results['embryo_probs'] is not None:
        fig_traj = plot_signed_margin_trajectories(
            results['embryo_probs'],
            max_embryos=30
        )
        fig_traj.savefig(plot_dir / "wt_vs_hom_trajectories.png", dpi=300, bbox_inches='tight')
        print(f"  Saved trajectory plot")
        
        fig_heatmap = plot_signed_margin_heatmap(
            results['embryo_probs'],
            vmin=-0.5,
            vmax=0.5
        )
        fig_heatmap.savefig(plot_dir / "wt_vs_hom_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"  Saved heatmap")
    
    # Penetrance plot
    if results['embryo_results'] is not None:
        fig_pen = plot_penetrance_distribution(
            results['embryo_results']
        )
        fig_pen.savefig(plot_dir / "wt_vs_hom_penetrance.png", dpi=300, bbox_inches='tight')
        print(f"  Saved penetrance plot")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
