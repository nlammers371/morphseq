"""Control vs Control Analysis - Data Generation

CRITICAL VALIDATION: Shows that pair_2 and pair_8 hets have a phenotype,
but wildtype controls do NOT (proving het signal is real, not artifact).

Runs 18 classifications (6 comparisons × 3 feature types):
1. pair_2 NonPenHet vs pair_2 WT (het phenotype in pair_2, 20251125 only)
2. pair_8 NonPenHet vs pair_8 WT (het phenotype in pair_8, 20251125 only)
3. pair_2 WT vs pair_8 WT (NEGATIVE CONTROL - should show no difference)
4. All NonPenHets vs All WTs (pooled across all pairs and experiments)
5. Exp 20251121 NonPenHets vs WTs (experiment-specific)
6. Exp 20251125 NonPenHets vs WTs (experiment-specific)

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
OUTPUT_DIR = Path(__file__).parent / "output" / "control_controls"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
DATA_PATH = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
PHENOTYPE_DIR = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/phenotype_lists"


def load_and_prepare_data():
    """Load b9d2 data and apply cluster_categories mapping.

    Returns
    -------
    df : pd.DataFrame
        Prepared data with mapped cluster_categories (all experiments)
    """
    print("="*70)
    print("LOADING B9D2 DATA (ALL EXPERIMENTS)")
    print("="*70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"\nLoaded {len(df)} rows, {df['embryo_id'].nunique()} unique embryos")

    # Show experiment distribution (experiment_id is integer, not string)
    print(f"\nExperiment distribution:")
    print(df.groupby('experiment_id')['embryo_id'].nunique())

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

    print("\nPair distribution:")
    print(df.groupby('pair')['embryo_id'].nunique())

    return df


def load_phenotype_lists():
    """Load phenotype lists to identify penetrant embryos.

    Returns
    -------
    set
        All embryo IDs that are in any phenotype list (CE, HTA, BA_rescue)
    """
    print("\n" + "="*70)
    print("LOADING PHENOTYPE LISTS")
    print("="*70)

    phenotype_files = {
        'CE': PHENOTYPE_DIR / 'b9d2-CE-phenotype.txt',
        'HTA': PHENOTYPE_DIR / 'b9d2-HTA-embryos.txt',
        'BA_rescue': PHENOTYPE_DIR / 'b9d2-curved-rescue.txt'
    }

    all_phenotype_ids = set()
    for name, filepath in phenotype_files.items():
        with open(filepath) as f:
            ids = [line.strip() for line in f if line.strip()]
            all_phenotype_ids.update(ids)
            print(f"  {name}: {len(ids)} embryos")

    print(f"\nTotal unique phenotype embryos: {len(all_phenotype_ids)}")
    return all_phenotype_ids


def define_groups(df, all_phenotype_ids):
    """Define pair-specific het and WT groups, plus experiment-specific groups.

    Parameters
    ----------
    df : pd.DataFrame
        Data with mapped cluster_categories (all experiments)
    all_phenotype_ids : set
        Embryo IDs that have penetrant phenotypes (to exclude)

    Returns
    -------
    dict
        {comparison_name: (group1_ids, group2_ids, group1_label, group2_label)}
    """
    print("\n" + "="*70)
    print("DEFINING GROUPS")
    print("="*70)

    # Filter to 20251125 experiment for pair-specific comparisons
    df_20251125 = df[df['experiment_id'] == 20251125].copy()

    # pair_2 non-penetrant hets (20251125 only)
    pair2_het_nonpen = df_20251125[
        (df_20251125['pair'] == 'b9d2_pair_2') &
        (df_20251125['genotype'] == 'b9d2_heterozygous') &
        (df_20251125['cluster_categories'] == 'Not_Penetrant') &
        (~df_20251125['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    # pair_2 WTs (20251125 only)
    pair2_wt = df_20251125[
        (df_20251125['pair'] == 'b9d2_pair_2') &
        (df_20251125['genotype'] == 'b9d2_wildtype') &
        (~df_20251125['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    # pair_8 non-penetrant hets (20251125 only)
    pair8_het_nonpen = df_20251125[
        (df_20251125['pair'] == 'b9d2_pair_8') &
        (df_20251125['genotype'] == 'b9d2_heterozygous') &
        (df_20251125['cluster_categories'] == 'Not_Penetrant') &
        (~df_20251125['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    # pair_8 WTs (20251125 only)
    pair8_wt = df_20251125[
        (df_20251125['pair'] == 'b9d2_pair_8') &
        (df_20251125['genotype'] == 'b9d2_wildtype') &
        (~df_20251125['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    # All non-penetrant hets (across all pairs and experiments)
    all_het_nonpen = df[
        (df['genotype'] == 'b9d2_heterozygous') &
        (df['cluster_categories'] == 'Not_Penetrant') &
        (~df['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    # All WTs (across all pairs and experiments)
    all_wt = df[
        (df['genotype'] == 'b9d2_wildtype') &
        (~df['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    # Experiment-specific groups (20251121)
    exp_20251121_het_nonpen = df[
        (df['experiment_id'] == 20251121) &
        (df['genotype'] == 'b9d2_heterozygous') &
        (df['cluster_categories'] == 'Not_Penetrant') &
        (~df['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    exp_20251121_wt = df[
        (df['experiment_id'] == 20251121) &
        (df['genotype'] == 'b9d2_wildtype') &
        (~df['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    # Experiment-specific groups (20251125)
    exp_20251125_het_nonpen = df[
        (df['experiment_id'] == 20251125) &
        (df['genotype'] == 'b9d2_heterozygous') &
        (df['cluster_categories'] == 'Not_Penetrant') &
        (~df['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    exp_20251125_wt = df[
        (df['experiment_id'] == 20251125) &
        (df['genotype'] == 'b9d2_wildtype') &
        (~df['embryo_id'].isin(all_phenotype_ids))
    ]['embryo_id'].unique().tolist()

    print(f"\nGroup sizes (20251125 pair-specific):")
    print(f"  pair_2 NonPen Hets: {len(pair2_het_nonpen)} embryos")
    print(f"  pair_2 WTs: {len(pair2_wt)} embryos")
    print(f"  pair_8 NonPen Hets: {len(pair8_het_nonpen)} embryos")
    print(f"  pair_8 WTs: {len(pair8_wt)} embryos")

    print(f"\nGroup sizes (all pairs, all experiments):")
    print(f"  All NonPen Hets: {len(all_het_nonpen)} embryos")
    print(f"  All WTs: {len(all_wt)} embryos")

    print(f"\nGroup sizes (experiment-specific):")
    print(f"  20251121 NonPen Hets: {len(exp_20251121_het_nonpen)} embryos")
    print(f"  20251121 WTs: {len(exp_20251121_wt)} embryos")
    print(f"  20251125 NonPen Hets: {len(exp_20251125_het_nonpen)} embryos")
    print(f"  20251125 WTs: {len(exp_20251125_wt)} embryos")

    # Define comparisons
    comparisons = {
        'pair2_Het_vs_WT': (pair2_het_nonpen, pair2_wt, 'pair2_Het', 'pair2_WT'),
        'pair8_Het_vs_WT': (pair8_het_nonpen, pair8_wt, 'pair8_Het', 'pair8_WT'),
        'pair2_WT_vs_pair8_WT': (pair2_wt, pair8_wt, 'pair2_WT', 'pair8_WT'),
        'AllNonPenHets_vs_WT': (all_het_nonpen, all_wt, 'AllNonPenHets', 'AllWT'),
        'Exp20251121_Het_vs_WT': (exp_20251121_het_nonpen, exp_20251121_wt, 'Exp20251121_Het', 'Exp20251121_WT'),
        'Exp20251125_Het_vs_WT': (exp_20251125_het_nonpen, exp_20251125_wt, 'Exp20251125_Het', 'Exp20251125_WT'),
    }

    return comparisons


def run_classifications(df, comparisons):
    """Run 18 classifications (6 comparisons × 3 features).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (20251125 only)
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

        # Check minimum sample size
        if len(group1_ids) < 3 or len(group2_ids) < 3:
            print(f"  SKIPPED: Not enough embryos (need >=3 per group)")
            continue

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
                import traceback
                traceback.print_exc()
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
    """Print formatted summary table with interpretation.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary statistics for all classifications
    """
    print("\n" + "="*70)
    print("FINAL SUMMARY: CONTROL VS CONTROL ANALYSIS")
    print("="*70)
    print("\n" + summary_df.to_string(index=False))
    print("\n" + "="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

    # Check if dataframe is empty
    if summary_df.empty or 'comparison' not in summary_df.columns:
        print("\nNO RESULTS: All comparisons were skipped (insufficient data)")
        print("Check that the experiment filter and group definitions are correct.")
        return

    # Key findings
    print("\nKEY FINDINGS & INTERPRETATION:")
    print("-"*70)

    # Negative control (WT vs WT)
    wt_vs_wt = summary_df[summary_df['comparison'] == 'pair2_WT_vs_pair8_WT']
    if len(wt_vs_wt) > 0:
        print("\n1. NEGATIVE CONTROL (pair_2 WT vs pair_8 WT):")
        print("   Expected: AUROC ~ 0.5, no significant difference")
        for _, row in wt_vs_wt.iterrows():
            status = "✓ PASS" if abs(row['max_auroc'] - 0.5) < 0.1 else "⚠ FAIL"
            print(f"   {row['feature']:12s}: AUROC={row['max_auroc']:.3f} {status}")
        print("   → If PASS: WTs are good controls (no spurious difference)")
        print("   → If FAIL: Batch effects or other confounds present")

    # Het phenotype detection
    pair2_het = summary_df[summary_df['comparison'] == 'pair2_Het_vs_WT']
    pair8_het = summary_df[summary_df['comparison'] == 'pair8_Het_vs_WT']

    if len(pair2_het) > 0:
        print("\n2. HET PHENOTYPE (pair_2 het vs pair_2 WT):")
        print("   Expected: AUROC > 0.6, p < 0.05")
        for _, row in pair2_het.iterrows():
            status = "✓ DETECTED" if row['max_auroc'] > 0.6 else "✗ NOT DETECTED"
            print(f"   {row['feature']:12s}: AUROC={row['max_auroc']:.3f} {status}")
            print(f"                 (earliest sig: {row['earliest_significant_hpf']} hpf)")

    if len(pair8_het) > 0:
        print("\n3. HET PHENOTYPE (pair_8 het vs pair_8 WT):")
        print("   Expected: AUROC > 0.6, p < 0.05")
        for _, row in pair8_het.iterrows():
            status = "✓ DETECTED" if row['max_auroc'] > 0.6 else "✗ NOT DETECTED"
            print(f"   {row['feature']:12s}: AUROC={row['max_auroc']:.3f} {status}")
            print(f"                 (earliest sig: {row['earliest_significant_hpf']} hpf)")

    print("\n" + "="*70)
    print("VALIDATION LOGIC:")
    print("  - If hets differ from WT (AUROC > 0.6) AND WTs don't differ from")
    print("    each other (AUROC ~ 0.5), then het phenotype is REAL.")
    print("  - If WTs differ from each other, there may be batch effects that")
    print("    confound the het signal.")
    print("="*70)


def main():
    """Main execution function."""
    # Load data
    df = load_and_prepare_data()

    # Load phenotype lists
    all_phenotype_ids = load_phenotype_lists()

    # Define groups
    comparisons = define_groups(df, all_phenotype_ids)

    # Run classifications
    all_results, summary_df = run_classifications(df, comparisons)

    # Print summary
    print_final_summary(summary_df)

    return all_results, summary_df


if __name__ == '__main__':
    results, summary = main()
