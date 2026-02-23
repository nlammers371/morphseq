"""
B9D2 Extended Phenotype Comparison Analysis

Extended comparisons including:
- Set 1: Within-pair CE validation (pair_7 and pair_8)
- Set 2: Negative controls (should show NO difference)
- Set 3: HTA/BA-rescue vs Het and WT

Usage:
    python b9d2_phenotype_comparison_extended.py

Author: Generated via Claude Code
Date: 2025-12-28
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

# Import functions from the main script
from b9d2_phenotype_comparison import (
    load_experiment_dataframe,
    parse_phenotype_file,
    load_all_phenotypes,
    load_experiment_data,
    prepare_comparison_data,
    run_difference_detection,
    compute_morphological_divergence,
    create_comprehensive_figure,
    EXPERIMENT_IDS,
    PHENOTYPE_DIR,
    CE_FILE,
    HTA_FILE,
    BA_RESCUE_FILE,
    TIME_COL,
    EMBRYO_ID_COL,
    GENOTYPE_COL,
    METRICS,
    RANDOM_STATE,
)

# Output directories for extended analysis
OUTPUT_DIR = Path(__file__).parent / 'output_extended'
CLASSIFICATION_DIR = OUTPUT_DIR / 'classification_results'
FIGURES_DIR = OUTPUT_DIR / 'figures'


# =============================================================================
# Helper Functions for Extended Analysis
# =============================================================================

def get_embryos_by_pair_and_genotype(df: pd.DataFrame, pair: str, genotype: str) -> list:
    """Get embryo IDs for a specific pair and genotype."""
    mask = (df['pair'] == pair) & (df[GENOTYPE_COL] == genotype)
    return df[mask][EMBRYO_ID_COL].unique().tolist()


def get_non_penetrant_hets(df: pd.DataFrame, pair: str, phenotype_ids: set) -> list:
    """
    Get heterozygous embryos from a pair that are NOT in any phenotype list.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe
    pair : str
        Pair name (e.g., 'b9d2_pair_7')
    phenotype_ids : set
        Set of all phenotype embryo IDs (CE, HTA, BA-rescue)

    Returns
    -------
    list of embryo IDs
    """
    mask = (df['pair'] == pair) & (df[GENOTYPE_COL] == 'b9d2_heterozygous')
    all_hets = set(df[mask][EMBRYO_ID_COL].unique())
    non_penetrant = all_hets - phenotype_ids
    return list(non_penetrant)


def get_ce_embryos_by_pair(ce_ids: list, df: pd.DataFrame, pair: str) -> list:
    """Get CE embryos that belong to a specific pair."""
    pair_embryos = set(df[df['pair'] == pair][EMBRYO_ID_COL].unique())
    return [eid for eid in ce_ids if eid in pair_embryos]


def split_embryos_randomly(embryo_ids: list, seed: int = 42) -> tuple:
    """Split embryo list randomly in half."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(embryo_ids)
    mid = len(shuffled) // 2
    return list(shuffled[:mid]), list(shuffled[mid:])


def run_single_comparison_extended(
    df_raw: pd.DataFrame,
    group1_ids: list,
    group2_ids: list,
    group1_label: str,
    group2_label: str,
    comparison_name: str,
    metric_col: str = 'total_length_um'
):
    """
    Run a single comparison between two groups (wrapper for extended analysis).
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: {comparison_name}")
    print(f"{'='*80}")
    print(f"Group 1 ({group1_label}): {len(group1_ids)} embryos")
    print(f"Group 2 ({group2_label}): {len(group2_ids)} embryos")

    # Check minimum embryos
    if len(group1_ids) < 3 or len(group2_ids) < 3:
        print(f"  SKIPPED: Not enough embryos (need >= 3 per group)")
        return

    # Step 1: Prepare data
    print("\n[1/4] Preparing comparison data...")
    try:
        df_binned = prepare_comparison_data(
            df_raw, group1_ids, group2_ids, group1_label, group2_label
        )
    except Exception as e:
        print(f"  SKIPPED: Error preparing data - {e}")
        return

    if len(df_binned) == 0:
        print("  SKIPPED: No binned samples after filtering")
        return

    # Step 2: Run difference detection
    print("\n[2/4] Running difference detection...")
    try:
        df_results, df_embryo_probs = run_difference_detection(df_binned, comparison_name)
    except Exception as e:
        print(f"  SKIPPED: Error in difference detection - {e}")
        return

    if len(df_results) == 0:
        print("  SKIPPED: No results from difference detection")
        return

    # Step 3: Compute morphological divergence
    print("\n[3/4] Computing morphological divergence...")

    # Check if metric exists
    if metric_col not in df_raw.columns:
        print(f"  Warning: {metric_col} not found, using total_length_um")
        metric_col = 'total_length_um'

    divergence_df = compute_morphological_divergence(
        df_raw, group1_ids, group2_ids, metric_col
    )

    if len(divergence_df) == 0:
        print("  Warning: No divergence data, skipping figure")
        # Still save classification results
        results_path = CLASSIFICATION_DIR / f'{comparison_name}.csv'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(results_path, index=False)
        print(f"  Saved classification results: {results_path}")
        return

    # Metric label for plotting
    metric_labels = {
        'total_length_um': 'Total Length (µm)',
        'baseline_deviation_normalized': 'Baseline Deviation (normalized)',
        'normalized_baseline_deviation': 'Baseline Deviation (normalized)'
    }
    metric_label = metric_labels.get(metric_col, metric_col)

    # Step 4: Create figure
    print("\n[4/4] Creating comprehensive figure...")
    figure_path = FIGURES_DIR / f'{comparison_name}_comprehensive.png'
    create_comprehensive_figure(
        df_results, divergence_df, df_raw,
        group1_ids, group2_ids,
        group1_label, group2_label,
        metric_col, metric_label,
        figure_path
    )

    # Save classification results
    results_path = CLASSIFICATION_DIR / f'{comparison_name}.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_path, index=False)
    print(f"  Saved classification results: {results_path}")

    print(f"\n{comparison_name} complete!")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function for extended analysis."""
    print("="*80)
    print("B9D2 EXTENDED PHENOTYPE COMPARISON ANALYSIS")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)

    # Create output directories
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load phenotypes
    print("\n[Step 1/4] Loading phenotype lists...")
    phenotypes = load_all_phenotypes()

    # Create set of all phenotype embryo IDs
    all_phenotype_ids = set()
    for pheno_ids in phenotypes.values():
        all_phenotype_ids.update(pheno_ids)
    print(f"Total phenotype embryos: {len(all_phenotype_ids)}")

    # Load experiment data
    print("\n[Step 2/4] Loading experiment data...")
    df_raw = load_experiment_data()

    # Check what pairs are available
    print("\n[Step 3/4] Checking available pairs...")
    pairs_available = df_raw['pair'].unique()
    print(f"  Available pairs: {sorted(pairs_available)}")

    # Get wildtype embryos (excluding phenotype embryos)
    wt_mask = df_raw[GENOTYPE_COL] == 'b9d2_wildtype'
    all_wt = set(df_raw[wt_mask][EMBRYO_ID_COL].unique())
    wt_ids = list(all_wt - all_phenotype_ids)
    print(f"  Total wildtype embryos (non-phenotype): {len(wt_ids)}")

    # =========================================================================
    # SET 1: Within-Pair CE Validation
    # =========================================================================
    print("\n" + "="*80)
    print("SET 1: WITHIN-PAIR CE VALIDATION")
    print("="*80)

    for pair in ['b9d2_pair_7', 'b9d2_pair_8']:
        if pair not in pairs_available:
            print(f"\n  Skipping {pair} - not in data")
            continue

        print(f"\n--- Pair: {pair} ---")

        # Get CE embryos for this pair
        ce_pair = get_ce_embryos_by_pair(phenotypes['CE'], df_raw, pair)
        print(f"  CE embryos in {pair}: {len(ce_pair)}")

        # Get non-penetrant hets for this pair
        het_non_penetrant = get_non_penetrant_hets(df_raw, pair, all_phenotype_ids)
        print(f"  Non-penetrant hets in {pair}: {len(het_non_penetrant)}")

        # Get wildtype for this pair
        wt_pair = get_embryos_by_pair_and_genotype(df_raw, pair, 'b9d2_wildtype')
        wt_pair = [eid for eid in wt_pair if eid not in all_phenotype_ids]
        print(f"  Wildtype in {pair}: {len(wt_pair)}")

        # 1a. CE vs Het (non-penetrant)
        if len(ce_pair) >= 3 and len(het_non_penetrant) >= 3:
            run_single_comparison_extended(
                df_raw, ce_pair, het_non_penetrant,
                'CE', 'Het_nonpen',
                f'{pair}_CE_vs_Het',
                metric_col='total_length_um'
            )

        # 1b. CE vs Wildtype
        if len(ce_pair) >= 3 and len(wt_pair) >= 3:
            run_single_comparison_extended(
                df_raw, ce_pair, wt_pair,
                'CE', 'WT',
                f'{pair}_CE_vs_WT',
                metric_col='total_length_um'
            )

        # 1c. Non-penetrant Het vs Wildtype
        if len(het_non_penetrant) >= 3 and len(wt_pair) >= 3:
            run_single_comparison_extended(
                df_raw, het_non_penetrant, wt_pair,
                'Het_nonpen', 'WT',
                f'{pair}_Het_vs_WT',
                metric_col='total_length_um'
            )

    # =========================================================================
    # SET 2: Negative Controls (should show NO difference)
    # =========================================================================
    print("\n" + "="*80)
    print("SET 2: NEGATIVE CONTROLS (should show NO difference)")
    print("="*80)

    # Filter to 20251125 experiment only
    df_20251125 = df_raw[df_raw['experiment_id'] == '20251125'].copy()
    print(f"\n  Experiment 20251125: {df_20251125[EMBRYO_ID_COL].nunique()} embryos")

    # 2a. Non-penetrant pair_2 hets vs Non-penetrant pair_8 hets
    het_pair2 = get_non_penetrant_hets(df_20251125, 'b9d2_pair_2', all_phenotype_ids)
    het_pair8 = get_non_penetrant_hets(df_20251125, 'b9d2_pair_8', all_phenotype_ids)
    print(f"  Non-penetrant hets (excluded CE/HTA/BA-rescue) - pair_2: {len(het_pair2)}, pair_8: {len(het_pair8)}")

    if len(het_pair2) >= 3 and len(het_pair8) >= 3:
        run_single_comparison_extended(
            df_20251125, het_pair2, het_pair8,
            'pair2_Het_nonpen', 'pair8_Het_nonpen',
            'NEGATIVE_pair2_vs_pair8_nonpen_hets',
            metric_col='total_length_um'
        )

    # 2b. pair_2 WT vs pair_8 WT (excluded CE/HTA/BA-rescue)
    wt_pair2 = get_embryos_by_pair_and_genotype(df_20251125, 'b9d2_pair_2', 'b9d2_wildtype')
    wt_pair2 = [eid for eid in wt_pair2 if eid not in all_phenotype_ids]
    wt_pair8 = get_embryos_by_pair_and_genotype(df_20251125, 'b9d2_pair_8', 'b9d2_wildtype')
    wt_pair8 = [eid for eid in wt_pair8 if eid not in all_phenotype_ids]
    print(f"  Wildtype (excluded CE/HTA/BA-rescue) - pair_2: {len(wt_pair2)}, pair_8: {len(wt_pair8)}")

    if len(wt_pair2) >= 3 and len(wt_pair8) >= 3:
        run_single_comparison_extended(
            df_20251125, wt_pair2, wt_pair8,
            'pair2_WT_nonpen', 'pair8_WT_nonpen',
            'NEGATIVE_pair2_vs_pair8_nonpen_wt',
            metric_col='total_length_um'
        )

    # 2c. Triple-sure: Split 20251125 wildtypes in half
    wt_20251125 = get_embryos_by_pair_and_genotype(df_20251125, None, 'b9d2_wildtype')
    # Actually get ALL WT from this experiment
    wt_mask = df_20251125[GENOTYPE_COL] == 'b9d2_wildtype'
    wt_20251125 = list(set(df_20251125[wt_mask][EMBRYO_ID_COL].unique()) - all_phenotype_ids)
    print(f"  Total WT in 20251125: {len(wt_20251125)}")

    if len(wt_20251125) >= 6:
        wt_half1, wt_half2 = split_embryos_randomly(wt_20251125, seed=RANDOM_STATE)
        print(f"  Split: half1={len(wt_half1)}, half2={len(wt_half2)}")

        run_single_comparison_extended(
            df_20251125, wt_half1, wt_half2,
            'WT_half1', 'WT_half2',
            'NEGATIVE_wt_split_half',
            metric_col='total_length_um'
        )

    # =========================================================================
    # SET 3: HTA/BA-rescue vs Het and WT (pooled across pairs, non-penetrant controls)
    # =========================================================================
    print("\n" + "="*80)
    print("SET 3: HTA/BA-RESCUE ANALYSIS (pooled across pairs)")
    print("="*80)

    # Get all non-penetrant hets (across all pairs, excluded from phenotype lists)
    het_mask = df_raw[GENOTYPE_COL] == 'b9d2_heterozygous'
    all_hets = set(df_raw[het_mask][EMBRYO_ID_COL].unique())
    het_non_penetrant_all = list(all_hets - all_phenotype_ids)
    print(f"  All non-penetrant hets (excluded CE/HTA/BA-rescue): {len(het_non_penetrant_all)}")
    print(f"  All wildtype (excluded CE/HTA/BA-rescue): {len(wt_ids)}")
    print(f"  HTA embryos: {len(phenotypes['HTA'])}")
    print(f"  BA-rescue embryos: {len(phenotypes['BA_rescue'])}")

    # 3a. HTA vs Het (non-penetrant, excluded from phenotype lists)
    if len(phenotypes['HTA']) >= 3 and len(het_non_penetrant_all) >= 3:
        run_single_comparison_extended(
            df_raw, phenotypes['HTA'], het_non_penetrant_all,
            'HTA', 'Het_nonpen_ctrl',
            'HTA_vs_Het_nonpen',
            metric_col='baseline_deviation_normalized'
        )

    # 3b. HTA vs Wildtype (excluded from phenotype lists)
    if len(phenotypes['HTA']) >= 3 and len(wt_ids) >= 3:
        run_single_comparison_extended(
            df_raw, phenotypes['HTA'], wt_ids,
            'HTA', 'WT_nonpen_ctrl',
            'HTA_vs_WT_nonpen',
            metric_col='baseline_deviation_normalized'
        )

    # 3c. BA-rescue vs Het (non-penetrant, excluded from phenotype lists)
    if len(phenotypes['BA_rescue']) >= 3 and len(het_non_penetrant_all) >= 3:
        run_single_comparison_extended(
            df_raw, phenotypes['BA_rescue'], het_non_penetrant_all,
            'BA_rescue', 'Het_nonpen_ctrl',
            'BA_rescue_vs_Het_nonpen',
            metric_col='baseline_deviation_normalized'
        )

    # 3d. BA-rescue vs Wildtype (excluded from phenotype lists)
    if len(phenotypes['BA_rescue']) >= 3 and len(wt_ids) >= 3:
        run_single_comparison_extended(
            df_raw, phenotypes['BA_rescue'], wt_ids,
            'BA_rescue', 'WT_nonpen_ctrl',
            'BA_rescue_vs_WT_nonpen',
            metric_col='baseline_deviation_normalized'
        )

    # =========================================================================
    # SET 4: HET vs WT COMPARISONS (pair-specific and pooled)
    # =========================================================================
    print("\n" + "="*80)
    print("SET 4: HET vs WT COMPARISONS (pair-specific and pooled)")
    print("="*80)

    # -------------------------------------------------------------------------
    # Prepare pooled groups
    # -------------------------------------------------------------------------
    print("\n--- Preparing pooled het/WT groups ---")

    # Pooled non-penetrant hets (all pairs, excluding CE/HTA/BA-rescue)
    het_mask = df_raw[GENOTYPE_COL] == 'b9d2_heterozygous'
    all_hets = set(df_raw[het_mask][EMBRYO_ID_COL].unique())
    het_nonpen_pooled = list(all_hets - all_phenotype_ids)

    # Pooled WTs (all pairs, excluding CE/HTA/BA-rescue)
    wt_mask = df_raw[GENOTYPE_COL] == 'b9d2_wildtype'
    all_wt = set(df_raw[wt_mask][EMBRYO_ID_COL].unique())
    wt_pooled = list(all_wt - all_phenotype_ids)

    print(f"  Pooled non-penetrant hets: {len(het_nonpen_pooled)}")
    print(f"  Pooled WTs: {len(wt_pooled)}")

    # -------------------------------------------------------------------------
    # Prepare pair-specific groups
    # -------------------------------------------------------------------------
    # Filter to 20251125 experiment only (where pair_2 and pair_8 exist)
    df_20251125 = df_raw[df_raw['experiment_id'] == '20251125'].copy()

    print("\n--- Preparing pair-specific het/WT groups ---")

    # pair_2 groups
    het_pair2_nonpen = get_non_penetrant_hets(df_20251125, 'b9d2_pair_2', all_phenotype_ids)
    wt_pair2 = get_embryos_by_pair_and_genotype(df_20251125, 'b9d2_pair_2', 'b9d2_wildtype')
    wt_pair2 = [eid for eid in wt_pair2 if eid not in all_phenotype_ids]

    print(f"  pair_2 non-penetrant hets: {len(het_pair2_nonpen)}")
    print(f"  pair_2 WTs: {len(wt_pair2)}")

    # pair_8 groups
    het_pair8_nonpen = get_non_penetrant_hets(df_20251125, 'b9d2_pair_8', all_phenotype_ids)
    wt_pair8 = get_embryos_by_pair_and_genotype(df_20251125, 'b9d2_pair_8', 'b9d2_wildtype')
    wt_pair8 = [eid for eid in wt_pair8 if eid not in all_phenotype_ids]

    print(f"  pair_8 non-penetrant hets: {len(het_pair8_nonpen)}")
    print(f"  pair_8 WTs: {len(wt_pair8)}")

    # -------------------------------------------------------------------------
    # Run comparisons
    # -------------------------------------------------------------------------

    # Comparison (a): Pooled non-penetrant hets vs pooled WTs
    print("\n--- Comparison (a): Pooled het vs WT ---")
    if len(het_nonpen_pooled) >= 3 and len(wt_pooled) >= 3:
        run_single_comparison_extended(
            df_raw, het_nonpen_pooled, wt_pooled,
            'Het_nonpen_pooled', 'WT_pooled',
            'POOLED_het_nonpen_vs_WT',
            metric_col='total_length_um'
        )
    else:
        print("  SKIPPED: Not enough embryos")

    # Comparison (b): pair_2 non-penetrant hets vs pair_2 WTs
    print("\n--- Comparison (b): pair_2 het vs WT ---")
    if len(het_pair2_nonpen) >= 3 and len(wt_pair2) >= 3:
        run_single_comparison_extended(
            df_20251125, het_pair2_nonpen, wt_pair2,
            'pair2_Het_nonpen', 'pair2_WT',
            'pair2_het_nonpen_vs_WT',
            metric_col='total_length_um'
        )
    else:
        print("  SKIPPED: Not enough embryos")

    # Comparison (c): pair_8 non-penetrant hets vs pair_8 WTs
    print("\n--- Comparison (c): pair_8 het vs WT ---")
    if len(het_pair8_nonpen) >= 3 and len(wt_pair8) >= 3:
        run_single_comparison_extended(
            df_20251125, het_pair8_nonpen, wt_pair8,
            'pair8_Het_nonpen', 'pair8_WT',
            'pair8_het_nonpen_vs_WT',
            metric_col='total_length_um'
        )
    else:
        print("  SKIPPED: Not enough embryos")

    # Comparison (d): pair_2 WTs vs pair_8 WTs (NEGATIVE CONTROL)
    print("\n--- Comparison (d): pair_2 WT vs pair_8 WT (NEGATIVE) ---")
    if len(wt_pair2) >= 3 and len(wt_pair8) >= 3:
        run_single_comparison_extended(
            df_20251125, wt_pair2, wt_pair8,
            'pair2_WT', 'pair8_WT',
            'NEGATIVE_pair2_WT_vs_pair8_WT',
            metric_col='total_length_um'
        )
    else:
        print("  SKIPPED: Not enough embryos")

    # Final summary
    print("\n" + "="*80)
    print("EXTENDED ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Classification: {CLASSIFICATION_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
