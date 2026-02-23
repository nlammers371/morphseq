"""B9D2 Cross-Phenotype Analysis - Data Generation

Runs 12 classifications (4 comparisons × 3 feature types):
1. HTA vs CE (primary cross-phenotype comparison)
2. HTA vs NonPenHets (reference - shows HTA is detectable)
3. CE vs NonPenHets (reference - shows CE is detectable)
4. NonPenHets vs WT (control validation)

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
OUTPUT_DIR = Path(__file__).parent / "output" / "b9d2_cross_phenotype"
DATA_PATH = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"


def load_and_prepare_data():
    """Load b9d2 data and apply cluster_categories mapping.

    Returns
    -------
    df : pd.DataFrame
        Prepared data with mapped cluster_categories
    """
    print("="*70)
    print("LOADING B9D2 DATA")
    print("="*70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"\nLoaded {len(df)} rows, {df['embryo_id'].nunique()} unique embryos")

    # CRITICAL: Map cluster_categories
    print("\nApplying cluster_categories mapping...")
    print("  unlabeled → Not_Penetrant")
    print("  wildtype → Not_Penetrant")

    df['cluster_categories'] = df['cluster_categories'].replace({
        'unlabeled': 'Not_Penetrant',
        'wildtype': 'Not_Penetrant'
    })

    # Verify mapping
    print("\nCluster categories distribution (after mapping):")
    print(df.groupby('cluster_categories')['embryo_id'].nunique())

    print("\nGenotype distribution:")
    print(df.groupby('genotype')['embryo_id'].nunique())

    return df


def define_groups(df):
    """Define HTA, CE, Non-Penetrant Het, and WT groups.

    Parameters
    ----------
    df : pd.DataFrame
        Data with mapped cluster_categories

    Returns
    -------
    dict
        {comparison_name: (group1_ids, group2_ids, group1_label, group2_label)}
    """
    print("\n" + "="*70)
    print("DEFINING GROUPS")
    print("="*70)

    # HTA phenotype
    hta_ids = df[df['cluster_categories'] == 'HTA']['embryo_id'].unique().tolist()

    # CE phenotype
    ce_ids = df[df['cluster_categories'] == 'CE']['embryo_id'].unique().tolist()

    # Non-penetrant hets: heterozygous AND Not_Penetrant
    nonpen_het_ids = df[
        (df['genotype'] == 'b9d2_heterozygous') &
        (df['cluster_categories'] == 'Not_Penetrant')
    ]['embryo_id'].unique().tolist()

    # WT: wildtype genotype
    wt_ids = df[df['genotype'] == 'b9d2_wildtype']['embryo_id'].unique().tolist()

    print(f"\nGroup sizes:")
    print(f"  HTA: {len(hta_ids)} embryos")
    print(f"  CE: {len(ce_ids)} embryos")
    print(f"  Non-Pen Hets: {len(nonpen_het_ids)} embryos")
    print(f"  WT: {len(wt_ids)} embryos")

    # Define comparisons
    comparisons = {
        'HTA_vs_CE': (hta_ids, ce_ids, 'HTA', 'CE'),
        'HTA_vs_NonPenHets': (hta_ids, nonpen_het_ids, 'HTA', 'NonPenHet'),
        'CE_vs_NonPenHets': (ce_ids, nonpen_het_ids, 'CE', 'NonPenHet'),
        'NonPenHets_vs_WT': (nonpen_het_ids, wt_ids, 'NonPenHet', 'WT'),
    }

    return comparisons


def run_classifications(df, comparisons):
    """Run 12 classifications (4 comparisons × 3 features).

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
    print("FINAL SUMMARY: CROSS-PHENOTYPE ANALYSIS")
    print("="*70)
    print("\n" + summary_df.to_string(index=False))
    print("\n" + "="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

    # Key findings
    print("\nKEY FINDINGS:")
    print("-"*70)

    # Control validation
    control_val = summary_df[summary_df['comparison'] == 'NonPenHets_vs_WT']
    if len(control_val) > 0:
        print("\n1. Control Validation (Non-Pen Hets vs WT):")
        for _, row in control_val.iterrows():
            print(f"   {row['feature']:12s}: AUROC={row['max_auroc']:.3f} "
                  f"(earliest sig: {row['earliest_significant_hpf']} hpf)")
        print("   → If AUROC close to 0.5, non-pen hets are good controls")

    # Cross-phenotype comparison
    hta_ce = summary_df[summary_df['comparison'] == 'HTA_vs_CE']
    if len(hta_ce) > 0:
        print("\n2. Cross-Phenotype Distinguishability (HTA vs CE):")
        for _, row in hta_ce.iterrows():
            print(f"   {row['feature']:12s}: AUROC={row['max_auroc']:.3f} "
                  f"(earliest sig: {row['earliest_significant_hpf']} hpf)")
        print("   → Higher AUROC = better separation between HTA and CE phenotypes")

    # HTA reference
    hta_nonpen = summary_df[summary_df['comparison'] == 'HTA_vs_NonPenHets']
    if len(hta_nonpen) > 0:
        print("\n3. HTA Detection (HTA vs Non-Pen Hets):")
        for _, row in hta_nonpen.iterrows():
            print(f"   {row['feature']:12s}: AUROC={row['max_auroc']:.3f} "
                  f"(earliest sig: {row['earliest_significant_hpf']} hpf)")

    # CE reference
    ce_nonpen = summary_df[summary_df['comparison'] == 'CE_vs_NonPenHets']
    if len(ce_nonpen) > 0:
        print("\n4. CE Detection (CE vs Non-Pen Hets):")
        for _, row in ce_nonpen.iterrows():
            print(f"   {row['feature']:12s}: AUROC={row['max_auroc']:.3f} "
                  f"(earliest sig: {row['earliest_significant_hpf']} hpf)")

    print("\n" + "="*70)


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
