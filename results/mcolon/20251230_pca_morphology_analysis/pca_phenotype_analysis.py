"""
PCA Morphology Space Analysis for B9D2 Phenotypes

Visualizes b9d2 phenotypes (CE, HTA, BA-rescue, non-penetrant) in PCA space
derived from VAE embeddings. Tests whether phenotypes separate in morphology space.

Workflow:
1. Load data from experiments 20251121, 20251125
2. Fit PCA on z_mu_b columns (80 biological VAE features)
3. Transform to PCA space
4. Compute WT reference (time-binned average)
5. Subtract WT reference to get deviation trajectories
6. Visualize in 3D PCA space colored by phenotype

Usage:
    python pca_phenotype_analysis.py

Output:
    - output/pca_raw_by_phenotype.html (raw PCA colored by phenotype)
    - output/pca_raw_by_stage.html (raw PCA colored by developmental stage)
    - output/pca_raw_by_pair.html (raw PCA colored by pair)
    - output/pca_raw_trajectories.html (raw PCA with trajectory lines by phenotype)
    - output/pca_raw_individual_trajectories.html (raw PCA with individual embryo trajectories)
    - output/pca_delta_3d.html (WT-subtracted PCA colored by phenotype)

Author: Generated via Claude Code
Date: 2025-12-30
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

# Import from trajectory analysis
from src.analyze.trajectory_analysis import (
    plot_3d_scatter,
    fit_pca_on_embeddings,
    transform_embeddings_to_pca,
    compute_wt_reference_by_time,
    subtract_wt_reference,
    plot_multimetric_trajectories,
)
from src.analyze.trajectory_analysis.plot_config import PHENOTYPE_COLORS, PHENOTYPE_ORDER

# Import from existing b9d2 analysis
sys.path.insert(0, str(Path(__file__).parent.parent / '20251228_b9d2_phenotype_comparisons'))
from b9d2_phenotype_comparison import (
    load_all_phenotypes,
    load_experiment_data,
    extract_wildtype_embryos,
)
from b9d2_phenotype_distribution_by_pair import prepare_phenotype_dataframe

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_IDS = ['20251121', '20251125']
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PCA settings
N_PCA_COMPONENTS = 3
BIN_WIDTH = 2.0  # hours for WT reference

# Column names
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 70)
    print("PCA MORPHOLOGY SPACE ANALYSIS FOR B9D2 PHENOTYPES")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[Step 1/7] Loading data...")
    phenotypes = load_all_phenotypes()
    df = load_experiment_data()
    wildtype_ids = extract_wildtype_embryos(df, phenotypes)

    print(f"  Total rows: {len(df)}")
    print(f"  Total embryos: {df[EMBRYO_ID_COL].nunique()}")

    # =========================================================================
    # Step 2: Assign Phenotype Labels
    # =========================================================================
    print("\n[Step 2/7] Assigning phenotype labels...")
    df = prepare_phenotype_dataframe(df, phenotypes, wildtype_ids, EXPERIMENT_IDS)

    print(f"  Phenotype distribution:")
    for pheno in PHENOTYPE_ORDER:
        if pheno in df['phenotype'].unique():
            n = df[df['phenotype'] == pheno][EMBRYO_ID_COL].nunique()
            print(f"    {pheno}: {n} embryos")

    # =========================================================================
    # Step 3: Fit PCA on z_mu_b columns
    # =========================================================================
    print("\n[Step 3/7] Fitting PCA on VAE embeddings...")
    z_mu_cols = [c for c in df.columns if 'z_mu_b' in c]
    print(f"  Using {len(z_mu_cols)} z_mu_b columns")

    pca, scaler, z_mu_cols = fit_pca_on_embeddings(
        df, z_mu_cols, n_components=N_PCA_COMPONENTS
    )

    # =========================================================================
    # Step 4: Transform to PCA space
    # =========================================================================
    print("\n[Step 4/7] Transforming embeddings to PCA space...")
    df = transform_embeddings_to_pca(df, pca, scaler, z_mu_cols)

    pca_cols = [f'PCA_{i+1}' for i in range(N_PCA_COMPONENTS)]
    print(f"  Added columns: {pca_cols}")

    # =========================================================================
    # Step 5: Compute WT Reference
    # =========================================================================
    print("\n[Step 5/7] Computing wildtype reference...")
    wt_reference = compute_wt_reference_by_time(
        df,
        pca_cols=pca_cols,
        time_col=TIME_COL,
        wt_embryo_ids=wildtype_ids,
        embryo_id_col=EMBRYO_ID_COL,
        bin_width=BIN_WIDTH,
    )
    print(f"  Reference time bins: {len(wt_reference)}")
    print(f"  Time range: {wt_reference['time_bin'].min():.1f} - {wt_reference['time_bin'].max():.1f} hpf")

    # =========================================================================
    # Step 6: Subtract WT Reference
    # =========================================================================
    print("\n[Step 6/7] Subtracting WT reference...")
    df = subtract_wt_reference(
        df,
        wt_reference,
        pca_cols=pca_cols,
        time_col=TIME_COL,
        bin_width=BIN_WIDTH,
    )

    delta_cols = [f'PCA_{i+1}_delta' for i in range(N_PCA_COMPONENTS)]
    print(f"  Added columns: {delta_cols}")

    # =========================================================================
    # Step 7: Visualize
    # =========================================================================
    print("\n[Step 7/7] Creating visualizations...")

    # 7a. Raw PCA colored by phenotype
    print("\n  Creating raw PCA 3D plot (colored by phenotype)...")
    fig_raw = plot_3d_scatter(
        df,
        coords=pca_cols,
        color_by='phenotype',
        color_palette=PHENOTYPE_COLORS,
        color_order=PHENOTYPE_ORDER,
        line_by=EMBRYO_ID_COL,
        min_points_per_line=10,
        title='B9D2 Phenotypes in PCA Space (Raw)',
        output_path=OUTPUT_DIR / 'pca_raw_by_phenotype',
    )

    # 7b. Raw PCA colored by predicted_stage_hpf
    print("\n  Creating raw PCA 3D plot (colored by developmental stage)...")
    fig_stage = plot_3d_scatter(
        df,
        coords=pca_cols,
        color_by=TIME_COL,
        color_continuous=True,
        colorscale='Viridis',
        colorbar_title='Developmental Stage (hpf)',
        line_by=EMBRYO_ID_COL,
        min_points_per_line=10,
        title='Embryo Progression Through PCA Space Over Time',
        output_path=OUTPUT_DIR / 'pca_raw_by_stage',
    )

    # 7c. Raw PCA colored by pair
    print("\n  Creating raw PCA 3D plot (colored by pair)...")
    fig_pair = plot_3d_scatter(
        df,
        coords=pca_cols,
        color_by='pair',
        line_by=EMBRYO_ID_COL,
        min_points_per_line=10,
        title='B9D2 Phenotypes in PCA Space by Pair',
        output_path=OUTPUT_DIR / 'pca_raw_by_pair',
    )

    # 7d. Raw PCA colored by phenotype with trajectory lines
    print("\n  Creating raw PCA 3D plot (colored by phenotype with trajectories)...")
    fig_trajectories = plot_3d_scatter(
        df,
        coords=pca_cols,
        color_by='phenotype',
        color_palette=PHENOTYPE_COLORS,
        color_order=PHENOTYPE_ORDER,
        line_by=EMBRYO_ID_COL,
        min_points_per_line=10,
        show_lines=True,
        x_col=TIME_COL,
        line_opacity=0.3,
        line_width=1.5,
        title='B9D2 Phenotype Trajectories in PCA Space',
        output_path=OUTPUT_DIR / 'pca_raw_trajectories',
    )

    # 7e. Raw PCA colored by embryo_id to see individual trajectories
    print("\n  Creating raw PCA 3D plot (colored by individual embryo with trajectories)...")
    fig_individual = plot_3d_scatter(
        df,
        coords=pca_cols,
        color_by=EMBRYO_ID_COL,
        line_by=EMBRYO_ID_COL,
        min_points_per_line=10,
        show_lines=True,
        x_col=TIME_COL,
        line_opacity=0.8,
        line_width=2.0,
        title='Individual Embryo Trajectories in PCA Space',
        output_path=OUTPUT_DIR / 'pca_raw_individual_trajectories',
    )

    # 7f. PCA Delta (with WT subtraction) colored by phenotype
    print("\n  Creating PCA delta 3D plot...")
    fig_delta = plot_3d_scatter(
        df,
        coords=delta_cols,
        color_by='phenotype',
        color_palette=PHENOTYPE_COLORS,
        color_order=PHENOTYPE_ORDER,
        line_by=EMBRYO_ID_COL,
        min_points_per_line=10,
        title='B9D2 Phenotypes in PCA Space (WT-subtracted)',
        output_path=OUTPUT_DIR / 'pca_delta_3d',
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"  {f.name}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
