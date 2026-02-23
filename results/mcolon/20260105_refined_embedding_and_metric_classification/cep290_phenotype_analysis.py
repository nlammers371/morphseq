"""CEP290 Phenotype Analysis - Data Generation

Runs 30 classifications (10 comparisons × 3 feature types):

Trajectory (Homo only) vs WT:
1. Low_to_High vs WT
2. High_to_Low vs WT
3. Intermediate vs WT

Trajectory (Homo only) vs Het:
4. Low_to_High vs Het
5. High_to_Low vs Het
6. Intermediate vs Het

Control:
7. Het vs WT

Cross-trajectory Comparisons (Homo only):
8. Low_to_High vs High_to_Low
9. Low_to_High vs Intermediate
10. High_to_Low vs Intermediate

Each comparison uses curvature, length, and embedding features.

Output: CSV files organized by comparison and feature type.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
from analyze.difference_detection.comparison import compare_groups

# Configuration
BIN_WIDTH = 2  # 2-hour bins
N_PERMUTATIONS = 500
OUTPUT_DIR = Path(__file__).parent / "output" / "cep290_phenotype"
DATA_PATH = PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"


def load_and_prepare_data():
    """Load CEP290 data.

    Returns
    -------
    df : pd.DataFrame
        Prepared data with genotype and cluster_categories
    """
    print("="*70)
    print("LOADING CEP290 DATA")
    print("="*70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"\nLoaded {len(df)} rows, {df['embryo_id'].nunique()} unique embryos")

    # Verify columns
    print("\nGenotype distribution:")
    print(df.groupby('genotype')['embryo_id'].nunique())

    print("\nCluster categories distribution:")
    print(df.groupby('cluster_categories')['embryo_id'].nunique())

    return df


def define_groups(df):
    """Define phenotype groups (Homo only) and control groups (WT, Het).

    Parameters
    ----------
    df : pd.DataFrame
        Data with genotype and cluster_categories

    Returns
    -------
    dict
        {comparison_name: (group1_ids, group2_ids, group1_label, group2_label)}
    """
    print("\n" + "="*70)
    print("DEFINING GROUPS")
    print("="*70)

    # Phenotype groups (Homozygous only)
    lowtohigh_homo_ids = df[
        (df['cluster_categories'] == 'Low_to_High') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    hightolow_homo_ids = df[
        (df['cluster_categories'] == 'High_to_Low') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    intermediate_homo_ids = df[
        (df['cluster_categories'] == 'Intermediate') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    # Control groups (all embryos of that genotype)
    wt_ids = df[df['genotype'] == 'cep290_wildtype']['embryo_id'].unique().tolist()
    het_ids = df[df['genotype'] == 'cep290_heterozygous']['embryo_id'].unique().tolist()

    print(f"\nPhenotype Groups (Homozygous only):")
    print(f"  Low_to_High (Homo): {len(lowtohigh_homo_ids)} embryos")
    print(f"  High_to_Low (Homo): {len(hightolow_homo_ids)} embryos")
    print(f"  Intermediate (Homo): {len(intermediate_homo_ids)} embryos")

    print(f"\nControl Groups (all embryos):")
    print(f"  WT: {len(wt_ids)} embryos")
    print(f"  Het: {len(het_ids)} embryos")

    # Define comparisons
    comparisons = {
        # Trajectory vs WT
        'LowToHigh_vs_WT': (lowtohigh_homo_ids, wt_ids, 'LowToHigh', 'WT'),
        'HighToLow_vs_WT': (hightolow_homo_ids, wt_ids, 'HighToLow', 'WT'),
        'Intermediate_vs_WT': (intermediate_homo_ids, wt_ids, 'Intermediate', 'WT'),

        # Trajectory vs Het
        'LowToHigh_vs_Het': (lowtohigh_homo_ids, het_ids, 'LowToHigh', 'Het'),
        'HighToLow_vs_Het': (hightolow_homo_ids, het_ids, 'HighToLow', 'Het'),
        'Intermediate_vs_Het': (intermediate_homo_ids, het_ids, 'Intermediate', 'Het'),

        # Control
        'Het_vs_WT': (het_ids, wt_ids, 'Het', 'WT'),

        # Cross-trajectory (all Homo)
        'LowToHigh_vs_HighToLow': (lowtohigh_homo_ids, hightolow_homo_ids, 'LowToHigh', 'HighToLow'),
        'LowToHigh_vs_Intermediate': (lowtohigh_homo_ids, intermediate_homo_ids, 'LowToHigh', 'Intermediate'),
        'HighToLow_vs_Intermediate': (hightolow_homo_ids, intermediate_homo_ids, 'HighToLow', 'Intermediate'),
    }

    return comparisons


def run_classifications(df, comparisons):
    """Run 27 classifications (9 comparisons × 3 features).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    comparisons : dict
        Comparison definitions

    Returns
    -------
    dict
        Nested dict: {comparison_name: {feature_name: results_dict}}
    """
    print("\n" + "="*70)
    print("RUNNING CLASSIFICATIONS")
    print("="*70)

    # Feature configurations
    feature_configs = {
        'curvature': ['baseline_deviation_normalized'],
        'length': ['total_length_um'],
        'embedding': 'z_mu_b'  # Auto-expands to all z_mu_b_* columns
    }

    all_results = {}
    summary_rows = []

    # Loop over comparisons
    for comp_name, (group1_ids, group2_ids, group1_label, group2_label) in comparisons.items():
        print(f"\n{'='*60}")
        print(f"Comparison: {comp_name}")
        print(f"  {group1_label}: {len(group1_ids)} embryos")
        print(f"  {group2_label}: {len(group2_ids)} embryos")
        print("="*60)

        # Prepare data for this comparison
        df_comp = df[df['embryo_id'].isin(group1_ids + group2_ids)].copy()
        df_comp['group'] = df_comp['embryo_id'].apply(
            lambda x: group1_label if x in group1_ids else group2_label
        )

        # Run for each feature type
        comp_results = {}
        for feature_name, features in feature_configs.items():
            print(f"\n  Feature: {feature_name}")

            try:
                results = compare_groups(
                    df_comp,
                    group_col='group',
                    group1=group1_label,
                    group2=group2_label,
                    features=features,
                    morphology_metric=None,
                    bin_width=BIN_WIDTH,
                    n_permutations=N_PERMUTATIONS,
                    n_jobs=-1,
                    verbose=False
                )

                comp_results[feature_name] = results

                # Save to CSV
                output_subdir = OUTPUT_DIR / comp_name.lower()
                output_subdir.mkdir(parents=True, exist_ok=True)

                results['classification'].to_csv(
                    output_subdir / f'classification_{feature_name}.csv',
                    index=False
                )

                # Print summary
                earliest_sig = results['summary']['earliest_significant_hpf']
                max_auroc = results['summary']['max_auroc']
                max_auroc_hpf = results['summary']['max_auroc_hpf']

                print(f"    ✓ Saved: {output_subdir / f'classification_{feature_name}.csv'}")
                print(f"    Earliest sig: {earliest_sig} hpf")
                print(f"    Max AUROC: {max_auroc:.3f} at {max_auroc_hpf} hpf")

                # Store for summary table
                summary_rows.append({
                    'comparison': comp_name,
                    'feature': feature_name,
                    'group1_label': group1_label,
                    'group2_label': group2_label,
                    'n_group1': len(group1_ids),
                    'n_group2': len(group2_ids),
                    'earliest_significant_hpf': earliest_sig,
                    'max_auroc': max_auroc,
                    'max_auroc_hpf': max_auroc_hpf,
                })

            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                comp_results[feature_name] = None

        all_results[comp_name] = comp_results

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Saved summary: {summary_path}")
    print("="*60)

    return all_results, summary_df


def print_final_summary(summary_df):
    """Print formatted summary table.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics for all classifications
    """
    print("\n" + "="*70)
    print("FINAL SUMMARY: CEP290 PHENOTYPE ANALYSIS")
    print("="*70)
    print("\n" + summary_df.to_string(index=False))
    print("\n" + "="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

    # Key findings
    print("\nKEY FINDINGS:")
    print("-"*70)

    # Trajectory vs WT
    print("\n1. Trajectory Detection vs WT:")
    for traj in ['LowToHigh', 'HighToLow', 'Intermediate']:
        traj_wt = summary_df[summary_df['comparison'] == f'{traj}_vs_WT']
        if len(traj_wt) > 0:
            print(f"\n  {traj} vs WT:")
            for _, row in traj_wt.iterrows():
                print(f"    {row['feature']:12s}: AUROC={row['max_auroc']:.3f} "
                      f"(earliest sig: {row['earliest_significant_hpf']} hpf)")

    # Trajectory vs Het
    print("\n2. Trajectory Detection vs Het:")
    for traj in ['LowToHigh', 'HighToLow', 'Intermediate']:
        traj_het = summary_df[summary_df['comparison'] == f'{traj}_vs_Het']
        if len(traj_het) > 0:
            print(f"\n  {traj} vs Het:")
            for _, row in traj_het.iterrows():
                print(f"    {row['feature']:12s}: AUROC={row['max_auroc']:.3f} "
                      f"(earliest sig: {row['earliest_significant_hpf']} hpf)")

    # Cross-trajectory
    print("\n3. Cross-Trajectory Distinguishability:")
    cross_traj_comps = ['LowToHigh_vs_HighToLow', 'LowToHigh_vs_Intermediate', 'HighToLow_vs_Intermediate']
    for comp in cross_traj_comps:
        comp_data = summary_df[summary_df['comparison'] == comp]
        if len(comp_data) > 0:
            print(f"\n  {comp}:")
            for _, row in comp_data.iterrows():
                print(f"    {row['feature']:12s}: AUROC={row['max_auroc']:.3f} "
                      f"(earliest sig: {row['earliest_significant_hpf']} hpf)")

    print("\n" + "="*70)
    print("NOTE: Phenotype groups are Homozygous only to avoid confounding signals")
    print("="*70)


def main():
    """Main execution function."""
    # Load data
    df = load_and_prepare_data()

    # Define groups
    comparisons = define_groups(df)

    # Run classifications
    all_results, summary_df = run_classifications(df, comparisons)

    # Print summary
    print_final_summary(summary_df)

    return all_results, summary_df


if __name__ == '__main__':
    results, summary = main()
