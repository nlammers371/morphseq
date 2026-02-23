"""Genotype comparison analysis: Homo vs WT, Homo vs Het, Het vs WT.

Uses the new refactored plotting structure with clean separation between
data preprocessing and visualization.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import classification API
from analyze.difference_detection.comparison import compare_groups

# Import new modular plotting utilities
from utils.preprocessing import prepare_auroc_data
from utils.plotting_functions import plot_multiple_aurocs
from utils.plotting_layouts import create_feature_comparison_panels

# Configuration
BIN_WIDTH = 2  # 2-hour bins as in reference function
BOOTSTRAP_PERMS = 500
OUTPUT_DIR = Path(__file__).parent / "output" / "genotype_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_comparison_with_features(
    df_with_groups,
    comparisons_dict,
    features,
    feature_label,
    bin_width=2,
    n_permutations=500
):
    """Run genotype comparisons using specified features.

    Parameters
    ----------
    df_with_groups : pd.DataFrame
        Data with genotype groups defined
    comparisons_dict : dict
        {comparison_label: (group1_ids, group2_ids, group1_name, group2_name)}
    features : str or list
        Feature specification for compare_groups()
    feature_label : str
        Descriptive label for logging
    bin_width : float
        Time binning width
    n_permutations : int
        Number of permutations for p-value estimation

    Returns
    -------
    dict
        {comparison_label: {'classification': df, 'auroc_data': df, 'summary': dict}}
    """
    print(f"\n{'=' * 70}")
    print(f"Running classifications with {feature_label}")
    print("=" * 70)

    all_results = {}

    for comparison_label, (group1_ids, group2_ids, group1_name, group2_name) in comparisons_dict.items():
        print(f"\n{'-' * 50}")
        print(f"Comparison: {comparison_label}")
        print(f"  {group1_name}: {len(group1_ids)} embryos")
        print(f"  {group2_name}: {len(group2_ids)} embryos")

        # Prepare data with group column
        df_comp = df_with_groups[df_with_groups['embryo_id'].isin(group1_ids + group2_ids)].copy()
        df_comp['group'] = df_comp['embryo_id'].apply(
            lambda x: group1_name if x in group1_ids else group2_name
        )

        # Run classification
        results = compare_groups(
            df_comp,
            group_col='group',
            group1=group1_name,
            group2=group2_name,
            features=features,
            morphology_metric=None,
            bin_width=bin_width,
            n_permutations=n_permutations,
            n_jobs=-1,
            verbose=True
        )

        # Prepare AUROC data for plotting
        classification_df = results['classification']
        auroc_data = prepare_auroc_data(classification_df)

        # Store results
        all_results[comparison_label] = {
            'classification': classification_df,
            'auroc_data': auroc_data,
            'summary': results['summary']
        }

        print(f"  Earliest significant: {results['summary']['earliest_significant_hpf']} hpf")
        print(f"  Max AUROC: {results['summary']['max_auroc']:.3f} at {results['summary']['max_auroc_hpf']} hpf")

    return all_results


def main():
    print("=" * 70)
    print("CEP290 Genotype Comparison Analysis")
    print("Homo vs WT | Homo vs Het | Het vs WT")
    print("=" * 70)

    # Load data
    DATA_PATH = PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} rows, {df['embryo_id'].nunique()} unique embryos")

    # Check genotype column
    if 'genotype' not in df.columns:
        print("ERROR: No 'genotype' column found in data")
        print(f"Available columns: {df.columns.tolist()}")
        return

    print(f"\nGenotype distribution:")
    print(df.groupby('genotype')['embryo_id'].nunique())

    # Define genotype groups
    # Map genotype values to standardized labels
    genotype_mapping = {
        'cep290_wildtype': 'WT',
        'cep290_heterozygous': 'Het',
        'cep290_homozygous': 'Homo'
    }

    # Add standardized genotype column
    df['genotype_std'] = df['genotype'].map(genotype_mapping)

    # Filter to only include rows with valid genotypes
    df = df[df['genotype_std'].notna()].copy()

    # Get embryo IDs per genotype
    homo_ids = df[df['genotype_std'] == 'Homo']['embryo_id'].unique().tolist()
    het_ids = df[df['genotype_std'] == 'Het']['embryo_id'].unique().tolist()
    wt_ids = df[df['genotype_std'] == 'WT']['embryo_id'].unique().tolist()

    print(f"\nGenotype groups:")
    print(f"  Homozygous (Homo): {len(homo_ids)} embryos")
    print(f"  Heterozygous (Het): {len(het_ids)} embryos")
    print(f"  Wild-type (WT): {len(wt_ids)} embryos")

    # Define comparisons
    comparisons = {
        'Homo_vs_WT': (homo_ids, wt_ids, 'Homo', 'WT'),
        'Homo_vs_Het': (homo_ids, het_ids, 'Homo', 'Het'),
        'Het_vs_WT': (het_ids, wt_ids, 'Het', 'WT'),
    }

    # Run classifications with different feature types
    # 1. Curvature only
    results_curvature = run_comparison_with_features(
        df_with_groups=df,
        comparisons_dict=comparisons,
        features=['baseline_deviation_normalized'],
        feature_label='Curvature (baseline_deviation_normalized)',
        bin_width=BIN_WIDTH,
        n_permutations=BOOTSTRAP_PERMS
    )

    # 2. Length only
    results_length = run_comparison_with_features(
        df_with_groups=df,
        comparisons_dict=comparisons,
        features=['total_length_um'],
        feature_label='Length (total_length_um)',
        bin_width=BIN_WIDTH,
        n_permutations=BOOTSTRAP_PERMS
    )

    # 3. VAE embedding (original)
    results_embedding = run_comparison_with_features(
        df_with_groups=df,
        comparisons_dict=comparisons,
        features='z_mu_b',  # Auto-expands to all z_mu_b_* columns
        feature_label='VAE Embedding (z_mu_b features)',
        bin_width=BIN_WIDTH,
        n_permutations=BOOTSTRAP_PERMS
    )

    # Store all results (use embedding as primary for backwards compatibility)
    all_results = results_embedding

    # =========================================================================
    # Create the pooled AUROC comparison plot
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("Creating pooled AUROC comparison plot...")

    # Define colors and styles (matching reference function)
    comparison_colors = {
        'Homo_vs_WT': '#D32F2F',
        'Homo_vs_Het': '#9467BD',
        'Het_vs_WT': '#888888',
    }

    comparison_styles = {
        'Homo_vs_WT': '-',
        'Homo_vs_Het': '-',
        'Het_vs_WT': '--',
    }

    # Extract pre-processed AUROC data
    auroc_dfs_dict = {
        label: result['auroc_data']
        for label, result in all_results.items()
    }

    # Create plot using new modular function (VAE only)
    fig = plot_multiple_aurocs(
        auroc_dfs_dict=auroc_dfs_dict,
        colors_dict=comparison_colors,
        styles_dict=comparison_styles,
        title=f'Pooled Classification Performance Over Time\n(VAE Latent Features, {BIN_WIDTH}-hour bins, shaded = null mean ± 1 SD)',
        figsize=(14, 7),
        ylim=(0.3, 1.05),
        save_path=OUTPUT_DIR / 'genotype_comparison_auroc.png'
    )
    plt.close(fig)

    # =========================================================================
    # Create 1x3 Feature Comparison Panel
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("Creating 1x3 feature comparison panel...")

    fig_features = create_feature_comparison_panels(
        results_curvature=results_curvature,
        results_length=results_length,
        results_embedding=results_embedding,
        colors_dict=comparison_colors,
        styles_dict=comparison_styles,
        title=f'Feature Comparison: Genotype Classification Performance\n({BIN_WIDTH}-hour bins, shaded = null mean ± 1 SD)',
        figsize=(18, 5),
        ylim=(0.3, 1.05),
        save_path=OUTPUT_DIR / 'genotype_comparison_by_feature.png'
    )
    plt.close(fig_features)

    # =========================================================================
    # Save results to CSV
    # =========================================================================
    print(f"\n{'=' * 50}")
    print("Saving results...")

    for comparison_label, result in all_results.items():
        csv_path = OUTPUT_DIR / f'{comparison_label}_auroc.csv'
        result['classification'].to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Summary table
    summary_df = pd.DataFrame([
        {
            'comparison': label,
            'earliest_significant_hpf': result['summary']['earliest_significant_hpf'],
            'max_auroc': result['summary']['max_auroc'],
            'max_auroc_hpf': result['summary']['max_auroc_hpf'],
        }
        for label, result in all_results.items()
    ])
    summary_path = OUTPUT_DIR / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  - genotype_comparison_auroc.png (VAE only)")
    print("  - genotype_comparison_by_feature.png (1x3 panel: Curvature | Length | VAE)")
    print("  - Homo_vs_WT_auroc.csv")
    print("  - Homo_vs_Het_auroc.csv")
    print("  - Het_vs_WT_auroc.csv")
    print("  - summary.csv")

    return {
        'curvature': results_curvature,
        'length': results_length,
        'embedding': results_embedding
    }


if __name__ == '__main__':
    results = main()
