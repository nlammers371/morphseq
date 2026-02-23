"""
CEP290 Multiclass Benchmark Analysis

Goal: Compare multiclass (2-class OvR) outputs to the original binary pipeline.
This replicates the key comparisons from cep290_phenotype_analysis.py:
- Each homozygous trajectory vs WT
- Each homozygous trajectory vs Het
- Het vs WT (control)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from analyze.difference_detection.comparison_multiclass import compare_groups_multiclass

# Configuration
BIN_WIDTH = 2  # 2-hour bins
N_PERMUTATIONS = 500
OUTPUT_DIR = Path(__file__).parent / "output" / "cep290_multiclass_benchmark"
DATA_PATH = PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"


def load_and_prepare_data():
    """Load CEP290 data and print basic distributions."""
    print("=" * 70)
    print("LOADING CEP290 DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"\nLoaded {len(df)} rows, {df['embryo_id'].nunique()} unique embryos")

    print("\nGenotype distribution:")
    print(df.groupby('genotype')['embryo_id'].nunique())

    print("\nCluster categories distribution:")
    print(df.groupby('cluster_categories')['embryo_id'].nunique())

    return df


def define_groups(df):
    """Define trajectory and control groups (matching cep290_phenotype_analysis)."""
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

    wt_ids = df[df['genotype'] == 'cep290_wildtype']['embryo_id'].unique().tolist()
    het_ids = df[df['genotype'] == 'cep290_heterozygous']['embryo_id'].unique().tolist()

    return {
        'LowToHigh': lowtohigh_homo_ids,
        'HighToLow': hightolow_homo_ids,
        'Intermediate': intermediate_homo_ids,
        'WT': wt_ids,
        'Het': het_ids,
    }


def build_comparisons(groups):
    """Define benchmark comparisons and positive labels."""
    return {
        'LowToHigh_vs_WT': ('LowToHigh', groups['LowToHigh'], 'WT', groups['WT']),
        'HighToLow_vs_WT': ('HighToLow', groups['HighToLow'], 'WT', groups['WT']),
        'Intermediate_vs_WT': ('Intermediate', groups['Intermediate'], 'WT', groups['WT']),
        'LowToHigh_vs_Het': ('LowToHigh', groups['LowToHigh'], 'Het', groups['Het']),
        'HighToLow_vs_Het': ('HighToLow', groups['HighToLow'], 'Het', groups['Het']),
        'Intermediate_vs_Het': ('Intermediate', groups['Intermediate'], 'Het', groups['Het']),
        'Het_vs_WT': ('Het', groups['Het'], 'WT', groups['WT']),
    }


def run_benchmark(df, groups):
    """Run multiclass (2-class) comparisons for benchmarking."""
    feature_configs = {
        'curvature': ['baseline_deviation_normalized'],
        'length': ['total_length_um'],
        'embedding': 'z_mu_b'
    }

    comparisons = build_comparisons(groups)
    summary_rows = []

    for comp_name, (pos_label, pos_ids, neg_label, neg_ids) in comparisons.items():
        comp_dir = OUTPUT_DIR / comp_name.lower()
        print("\n" + "=" * 60)
        print(f"Benchmark: {comp_name}")
        print(f"  {pos_label}: {len(pos_ids)} embryos")
        print(f"  {neg_label}: {len(neg_ids)} embryos")
        print("=" * 60)

        groups_dict = {
            pos_label: pos_ids,
            neg_label: neg_ids,
        }

        for feature_name, features in feature_configs.items():
            print(f"\n  Feature: {feature_name}")
            results = compare_groups_multiclass(
                df,
                groups=groups_dict,
                features=features,
                bin_width=BIN_WIDTH,
                n_permutations=N_PERMUTATIONS,
                n_jobs=-1,
                skip_bin_if_not_all_present=True,
                verbose=False
            )

            feature_dir = comp_dir / feature_name
            feature_dir.mkdir(parents=True, exist_ok=True)

            for class_label, df_auroc in results['ovr_classification'].items():
                df_auroc.to_csv(feature_dir / f'ovr_auroc_{class_label}.csv', index=False)

            summary_rows.append({
                'comparison': comp_name,
                'feature': feature_name,
                'positive_label': pos_label,
                'negative_label': neg_label,
                'n_positive': len(pos_ids),
                'n_negative': len(neg_ids),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'benchmark_summary.csv', index=False)
    print(f"\nSaved summary: {OUTPUT_DIR / 'benchmark_summary.csv'}")


if __name__ == '__main__':
    print("CEP290 Multiclass Benchmark Analysis")

    df = load_and_prepare_data()
    groups = define_groups(df)
    run_benchmark(df, groups)

    print("\n" + "=" * 70)
    print(f"Benchmark complete. Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
