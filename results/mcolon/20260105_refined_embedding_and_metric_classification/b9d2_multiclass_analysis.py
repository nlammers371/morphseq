"""
B9D2 Multiclass Classification Analysis

Three-class comparison (cluster category-based):
- CE: Cystic eye phenotype
- HTA_BA: HTA + BA_rescue combined (late-onset phenotypes)
- Not_Penetrant: Non-penetrant embryos (cluster category)

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
# Set to (min_hpf, max_hpf) for a focused window, or None for full time range.
TIME_RANGE_HPF = None
# Optional label appended to output dir to keep debug runs separate.
OUTPUT_LABEL = None
BASE_OUTPUT_DIR = Path(__file__).parent / "output" / "b9d2_multiclass"
OUTPUT_DIR = BASE_OUTPUT_DIR / OUTPUT_LABEL if OUTPUT_LABEL else BASE_OUTPUT_DIR
DATA_PATH = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"


def load_and_prepare_data(time_range: tuple | None = None):
    """Load B9D2 data and prepare for multiclass analysis."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print(f"Loaded {len(df)} rows, {df['embryo_id'].nunique()} embryos")

    # CRITICAL: Map cluster categories
    # unlabeled and wildtype are both non-penetrant
    df['cluster_categories'] = df['cluster_categories'].replace({
        'unlabeled': 'Not_Penetrant',
        'wildtype': 'Not_Penetrant'
    })

    print("\nCluster category distribution:")
    print(df['cluster_categories'].value_counts())

    if time_range is not None:
        time_min, time_max = time_range
        before = len(df)
        df = df[(df['predicted_stage_hpf'] >= time_min) & (df['predicted_stage_hpf'] <= time_max)].copy()
        print(f"\nTime filter: {time_min:g}-{time_max:g} hpf "
              f"({before} -> {len(df)} rows, {df['embryo_id'].nunique()} embryos)")

    return df


def report_group_overlaps(
    df: pd.DataFrame,
    groups: dict,
    output_path: Path,
    max_print: int = 20
):
    """Report embryo IDs that appear in multiple groups."""
    id_to_labels = {}
    for label, ids in groups.items():
        for eid in ids:
            id_to_labels.setdefault(eid, []).append(label)

    duplicates = {eid: labels for eid, labels in id_to_labels.items() if len(labels) > 1}

    if not duplicates:
        print("\nNo overlapping embryo IDs across groups.")
        return

    print(f"\nFound {len(duplicates)} embryos in multiple groups.")
    print(f"Showing up to {max_print} examples:")

    records = []
    for i, (eid, labels) in enumerate(duplicates.items()):
        subset = df[df['embryo_id'] == eid]
        genotypes = sorted(subset['genotype'].dropna().unique().tolist())
        categories = sorted(subset['cluster_categories'].dropna().unique().tolist())
        record = {
            'embryo_id': eid,
            'groups': ','.join(labels),
            'genotype': ','.join(genotypes),
            'cluster_categories': ','.join(categories),
        }
        records.append(record)
        if i < max_print:
            print(f"  {eid}: groups={record['groups']} "
                  f"genotype={record['genotype']} "
                  f"cluster_categories={record['cluster_categories']}")

    report_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    print(f"Full overlap report saved to: {output_path}")


def define_groups(df):
    """Define 3-class groups from cluster categories."""

    # CE phenotype
    ce_ids = df[df['cluster_categories'] == 'CE']['embryo_id'].unique().tolist()

    # HTA + BA_rescue combined
    hta_ba_ids = df[
        df['cluster_categories'].isin(['HTA', 'BA_rescue'])
    ]['embryo_id'].unique().tolist()

    # Non-penetrant (cluster category; genotype-agnostic)
    not_penetrant_ids = df[
        df['cluster_categories'] == 'Not_Penetrant'
    ]['embryo_id'].unique().tolist()

    groups = {
        'CE': ce_ids,
        'HTA_BA': hta_ba_ids,
        'Not_Penetrant': not_penetrant_ids,
    }

    report_group_overlaps(df, groups, OUTPUT_DIR / "group_overlap_report.csv")

    print("\n" + "="*60)
    print("Group Definitions")
    print("="*60)
    for label, ids in groups.items():
        print(f"{label:12s}: {len(ids):3d} embryos")

    return groups


def run_multiclass_classifications(df, groups):
    """Run multiclass classification with curvature, length, embedding."""

    feature_configs = {
        'curvature': ['baseline_deviation_normalized'],
        'length': ['total_length_um'],
        'embedding': 'z_mu_b'
    }

    all_results = {}

    for feature_name, features in feature_configs.items():
        print(f"\n{'='*60}")
        print(f"Feature: {feature_name.upper()}")
        print("="*60)

        results = compare_groups_multiclass(
            df,
            groups=groups,
            features=features,
            bin_width=BIN_WIDTH,
            n_permutations=N_PERMUTATIONS,
            n_jobs=-1,
            verbose=True
        )

        all_results[feature_name] = results

        # Save outputs
        output_subdir = OUTPUT_DIR / feature_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Save per-class OvR AUROC CSVs
        for class_label, df_auroc in results['ovr_classification'].items():
            df_auroc.to_csv(output_subdir / f'ovr_auroc_{class_label}.csv', index=False)

        # Save confusion matrices
        cm_dir = output_subdir / 'confusion_matrices'
        cm_dir.mkdir(exist_ok=True)
        for time_bin, cm_df in results['confusion_matrices'].items():
            cm_df.to_csv(cm_dir / f'cm_t{time_bin}.csv')

        # Save embryo predictions
        if results['embryo_predictions'] is not None:
            results['embryo_predictions'].to_csv(
                output_subdir / 'embryo_predictions.csv', index=False
            )

        # Print summary
        print(f"\n{feature_name.upper()} Summary:")
        print(f"Data Quality:")
        dq = results['summary']['data_quality']
        print(f"  Total bins: {dq['total_bins']}")
        print(f"  Complete bins (all classes present): {dq['bins_with_all_classes']}")
        print(f"  Incomplete bins (missing classes): {dq['bins_with_missing_classes']}")
        print(f"  Bins skipped: {dq['bins_skipped']}")

        print(f"\nPer-Class Performance:")
        for class_label, class_summary in results['summary']['per_class'].items():
            print(f"  {class_label}:")
            earliest = class_summary['earliest_significant_hpf']
            print(f"    Earliest sig: {earliest if earliest is not None else 'None'} hpf")
            max_auroc = class_summary['max_auroc']
            max_hpf = class_summary['max_auroc_hpf']
            print(f"    Max AUROC: {max_auroc:.3f} at {max_hpf} hpf")

        if results['summary']['overall_accuracy'] is not None:
            print(f"\n  Overall Accuracy: {results['summary']['overall_accuracy']:.3f}")

    return all_results


if __name__ == '__main__':
    print("B9D2 Multiclass Classification Analysis")
    print("="*60)

    if TIME_RANGE_HPF is None:
        print("Time filter: full range")
    else:
        print(f"Time filter: {TIME_RANGE_HPF[0]:g}-{TIME_RANGE_HPF[1]:g} hpf")
    print(f"Output directory: {OUTPUT_DIR}")

    df = load_and_prepare_data(time_range=TIME_RANGE_HPF)
    groups = define_groups(df)
    results = run_multiclass_classifications(df, groups)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)
