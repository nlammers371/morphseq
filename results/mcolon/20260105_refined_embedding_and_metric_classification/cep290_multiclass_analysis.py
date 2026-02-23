"""
CEP290 Multiclass Classification Analysis

Configurations:
1) HL vs (LH+Int) vs (WT+Het) - 3 classes (biological grouping)
2) LowToHigh vs HighToLow vs Intermediate vs WT - 4 classes
3) LowToHigh vs HighToLow vs Intermediate vs WT vs Het - 5 classes

Features tested:
- Curvature: baseline_deviation_normalized
- Length: total_length_um
- Embedding: z_mu_b (VAE biological latent features)
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
OUTPUT_DIR = Path(__file__).parent / "output" / "cep290_multiclass"
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


def report_group_overlaps(groups, label):
    """Warn if embryo IDs appear in multiple groups."""
    id_to_labels = {}
    for group_label, ids in groups.items():
        for eid in ids:
            id_to_labels.setdefault(eid, []).append(group_label)

    overlaps = {eid: labels for eid, labels in id_to_labels.items() if len(labels) > 1}
    if overlaps:
        print(f"\nWARNING: {len(overlaps)} embryos appear in multiple groups for {label}.")
        for eid, labels in list(overlaps.items())[:10]:
            print(f"  {eid}: {labels}")
    else:
        print(f"\nNo overlapping embryo IDs for {label}.")


def define_groups_config1(df):
    """Config 1: HL vs (LH+Int) vs (WT+Het)."""
    hl_ids = df[
        (df['cluster_categories'] == 'High_to_Low') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    lh_int_ids = df[
        (df['cluster_categories'].isin(['Low_to_High', 'Intermediate'])) &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    wt_het_ids = df[
        df['genotype'].isin(['cep290_wildtype', 'cep290_heterozygous'])
    ]['embryo_id'].unique().tolist()

    groups = {
        'HL': hl_ids,
        'LH_Int': lh_int_ids,
        'WT_Het': wt_het_ids,
    }

    print("\n" + "=" * 70)
    print("CONFIG 1 GROUPS: HL vs (LH+Int) vs (WT+Het)")
    print("=" * 70)
    for label, ids in groups.items():
        print(f"{label:8s}: {len(ids):3d} embryos")

    report_group_overlaps(groups, "Config 1")
    return groups


def define_groups_config2(df):
    """Config 2: LowToHigh vs HighToLow vs Intermediate vs WT."""
    lowtohigh_ids = df[
        (df['cluster_categories'] == 'Low_to_High') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    hightolow_ids = df[
        (df['cluster_categories'] == 'High_to_Low') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    intermediate_ids = df[
        (df['cluster_categories'] == 'Intermediate') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    wt_ids = df[
        df['genotype'] == 'cep290_wildtype'
    ]['embryo_id'].unique().tolist()

    groups = {
        'LowToHigh': lowtohigh_ids,
        'HighToLow': hightolow_ids,
        'Intermediate': intermediate_ids,
        'WT': wt_ids,
    }

    print("\n" + "=" * 70)
    print("CONFIG 2 GROUPS: LowToHigh vs HighToLow vs Intermediate vs WT")
    print("=" * 70)
    for label, ids in groups.items():
        print(f"{label:12s}: {len(ids):3d} embryos")

    report_group_overlaps(groups, "Config 2")
    return groups


def define_groups_config3(df):
    """Config 3: LowToHigh vs HighToLow vs Intermediate vs WT vs Het."""
    lowtohigh_ids = df[
        (df['cluster_categories'] == 'Low_to_High') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    hightolow_ids = df[
        (df['cluster_categories'] == 'High_to_Low') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    intermediate_ids = df[
        (df['cluster_categories'] == 'Intermediate') &
        (df['genotype'] == 'cep290_homozygous')
    ]['embryo_id'].unique().tolist()

    wt_ids = df[
        df['genotype'] == 'cep290_wildtype'
    ]['embryo_id'].unique().tolist()

    het_ids = df[
        df['genotype'] == 'cep290_heterozygous'
    ]['embryo_id'].unique().tolist()

    groups = {
        'LowToHigh': lowtohigh_ids,
        'HighToLow': hightolow_ids,
        'Intermediate': intermediate_ids,
        'WT': wt_ids,
        'Het': het_ids,
    }

    print("\n" + "=" * 70)
    print("CONFIG 3 GROUPS: LowToHigh vs HighToLow vs Intermediate vs WT vs Het")
    print("=" * 70)
    for label, ids in groups.items():
        print(f"{label:12s}: {len(ids):3d} embryos")

    report_group_overlaps(groups, "Config 3")
    return groups


def run_multiclass_config(df, config_name, groups):
    """Run multiclass classification for a single configuration."""
    print("\n" + "=" * 70)
    print(f"RUNNING MULTICLASS: {config_name}")
    print("=" * 70)

    feature_configs = {
        'curvature': ['baseline_deviation_normalized'],
        'length': ['total_length_um'],
        'embedding': 'z_mu_b'
    }

    output_dir = OUTPUT_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for feature_name, features in feature_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Feature: {feature_name.upper()}")
        print("=" * 60)

        results = compare_groups_multiclass(
            df,
            groups=groups,
            features=features,
            bin_width=BIN_WIDTH,
            n_permutations=N_PERMUTATIONS,
            n_jobs=-1,
            skip_bin_if_not_all_present=True,
            verbose=True
        )

        # Save outputs
        feature_dir = output_dir / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)

        for class_label, df_auroc in results['ovr_classification'].items():
            df_auroc.to_csv(feature_dir / f'ovr_auroc_{class_label}.csv', index=False)

        cm_dir = feature_dir / 'confusion_matrices'
        cm_dir.mkdir(exist_ok=True)
        for time_bin, cm_df in results['confusion_matrices'].items():
            cm_df.to_csv(cm_dir / f'cm_t{time_bin}.csv')

        if results['embryo_predictions'] is not None:
            results['embryo_predictions'].to_csv(
                feature_dir / 'embryo_predictions.csv', index=False
            )

        # Print summary
        print(f"\n{feature_name.upper()} Summary:")
        dq = results['summary']['data_quality']
        print("Data Quality:")
        print(f"  Total bins: {dq.get('total_bins')}")
        print(f"  Complete bins (all classes present): {dq.get('bins_with_all_classes')}")
        print(f"  Incomplete bins (missing classes): {dq.get('bins_with_missing_classes')}")
        print(f"  Bins skipped: {dq.get('bins_skipped')}")

        print("\nPer-Class Performance:")
        for class_label, class_summary in results['summary']['per_class'].items():
            earliest = class_summary['earliest_significant_hpf']
            max_auroc = class_summary['max_auroc']
            max_hpf = class_summary['max_auroc_hpf']
            n_sig = class_summary['n_significant_bins']
            print(f"  {class_label}:")
            print(f"    Earliest sig: {earliest if earliest is not None else 'None'} hpf")
            print(f"    Max AUROC: {max_auroc:.3f} at {max_hpf} hpf")
            print(f"    Significant bins: {n_sig}")

            summary_rows.append({
                'config': config_name,
                'feature': feature_name,
                'class_label': class_label,
                'earliest_significant_hpf': earliest,
                'max_auroc': max_auroc,
                'max_auroc_hpf': max_hpf,
                'n_significant_bins': n_sig,
                'total_bins': dq.get('total_bins'),
                'bins_with_all_classes': dq.get('bins_with_all_classes'),
                'bins_with_missing_classes': dq.get('bins_with_missing_classes'),
                'bins_skipped': dq.get('bins_skipped'),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nSaved summary: {output_dir / 'summary.csv'}")


if __name__ == '__main__':
    print("CEP290 Multiclass Classification Analysis")

    df = load_and_prepare_data()

    config1_groups = define_groups_config1(df)
    run_multiclass_config(df, "config1_biological", config1_groups)

    config2_groups = define_groups_config2(df)
    run_multiclass_config(df, "config2_trajectory", config2_groups)

    config3_groups = define_groups_config3(df)
    run_multiclass_config(df, "config3_trajectory_wt_het", config3_groups)

    print("\n" + "=" * 70)
    print(f"Analysis complete. Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
