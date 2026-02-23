"""
Diagnostic script to investigate why baseline and class_weight methods
are missing from probability shift analysis plots for specific comparisons.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

results_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251014"
data_dir_base = os.path.join(results_dir, "imbalance_methods", "data")
output_dir = os.path.join(results_dir, "imbalance_methods", "diagnostics", "missing_lines")
os.makedirs(output_dir, exist_ok=True)

print(f"Diagnostic output: {output_dir}\n")

# Methods to check
METHODS_TO_CHECK = [
    'baseline',
    'class_weight',
    'embryo_weight',
    'combined_weight',
    'calibrated_class_weight',
    'calibrated_combined_weight',
    'balanced_bootstrap'
]

# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def diagnose_method_data(gene, comparison, method_name):
    """
    Check data availability for a specific method.

    Returns dict with diagnostic info.
    """
    data_dir = os.path.join(data_dir_base, gene)
    csv_path = os.path.join(data_dir, f"embryo_probs_{method_name}_{comparison}.csv")

    info = {
        'gene': gene,
        'comparison': comparison,
        'method': method_name,
        'file_exists': os.path.exists(csv_path),
        'file_path': csv_path
    }

    if not info['file_exists']:
        info['error'] = 'File does not exist'
        return info

    try:
        df = pd.read_csv(csv_path)

        info['n_rows'] = len(df)
        info['n_embryos'] = df['embryo_id'].nunique() if 'embryo_id' in df.columns else 0
        info['n_time_bins'] = df['time_bin'].nunique() if 'time_bin' in df.columns else 0

        if 'true_label' in df.columns:
            info['classes'] = sorted(df['true_label'].unique())
            info['n_classes'] = len(info['classes'])

            # Per-class breakdown
            class_breakdown = {}
            for cls in info['classes']:
                df_cls = df[df['true_label'] == cls]
                class_breakdown[cls] = {
                    'n_predictions': len(df_cls),
                    'n_embryos': df_cls['embryo_id'].nunique() if 'embryo_id' in df_cls.columns else 0,
                    'n_time_bins': df_cls['time_bin'].nunique() if 'time_bin' in df_cls.columns else 0,
                    'time_bin_range': (df_cls['time_bin'].min(), df_cls['time_bin'].max()) if 'time_bin' in df_cls.columns else (None, None)
                }
            info['class_breakdown'] = class_breakdown

        if 'pred_proba' in df.columns:
            info['has_pred_proba'] = True
            info['pred_proba_range'] = (df['pred_proba'].min(), df['pred_proba'].max())
            info['pred_proba_nans'] = df['pred_proba'].isna().sum()
        else:
            info['has_pred_proba'] = False
            info['error'] = 'Missing pred_proba column'

        info['loaded_successfully'] = True

    except Exception as e:
        info['error'] = str(e)
        info['loaded_successfully'] = False

    return info


def test_groupby_aggregation(gene, comparison, method_name):
    """
    Test the exact groupby operation used in the plotting function.

    Returns dict with aggregation results per class.
    """
    data_dir = os.path.join(data_dir_base, gene)
    csv_path = os.path.join(data_dir, f"embryo_probs_{method_name}_{comparison}.csv")

    if not os.path.exists(csv_path):
        return {'error': 'File does not exist'}

    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            return {'error': 'DataFrame is empty'}

        if 'true_label' not in df.columns or 'time_bin' not in df.columns or 'pred_proba' not in df.columns:
            return {'error': 'Missing required columns'}

        classes = sorted(df['true_label'].unique())

        results = {}
        for cls in classes:
            df_cls = df[df['true_label'] == cls]

            if df_cls.empty:
                results[cls] = {'error': 'No data for this class'}
                continue

            # This is the exact operation from plot_probability_shift_analysis
            mean_probs = df_cls.groupby('time_bin')['pred_proba'].mean()

            results[cls] = {
                'n_time_bins': len(mean_probs),
                'time_bins': sorted(mean_probs.index.tolist()),
                'mean_probs_range': (mean_probs.min(), mean_probs.max()),
                'has_data': len(mean_probs) > 0
            }

        return results

    except Exception as e:
        return {'error': str(e)}


def compare_genes(gene1, gene2, comparison1, comparison2):
    """
    Compare data availability between two genes.
    """
    print(f"\n{'='*80}")
    print(f"COMPARING: {gene1} {comparison1} vs {gene2} {comparison2}")
    print(f"{'='*80}")

    for method_name in METHODS_TO_CHECK:
        print(f"\n{method_name}:")

        info1 = diagnose_method_data(gene1, comparison1, method_name)
        info2 = diagnose_method_data(gene2, comparison2, method_name)

        print(f"  {gene1}:")
        if info1.get('file_exists'):
            print(f"    Rows: {info1.get('n_rows', 'N/A')}")
            print(f"    Embryos: {info1.get('n_embryos', 'N/A')}")
            print(f"    Time bins: {info1.get('n_time_bins', 'N/A')}")
        else:
            print(f"    FILE MISSING")

        print(f"  {gene2}:")
        if info2.get('file_exists'):
            print(f"    Rows: {info2.get('n_rows', 'N/A')}")
            print(f"    Embryos: {info2.get('n_embryos', 'N/A')}")
            print(f"    Time bins: {info2.get('n_time_bins', 'N/A')}")
        else:
            print(f"    FILE MISSING")


def comprehensive_diagnostic(gene, comparison):
    """
    Run comprehensive diagnostic for a specific gene × comparison.
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE DIAGNOSTIC: {gene} - {comparison}")
    print(f"{'='*80}")

    all_info = []

    for method_name in METHODS_TO_CHECK:
        print(f"\n{'-'*80}")
        print(f"Method: {method_name}")
        print(f"{'-'*80}")

        # Get basic info
        info = diagnose_method_data(gene, comparison, method_name)
        all_info.append(info)

        if not info.get('file_exists'):
            print(f"  ✗ FILE DOES NOT EXIST: {info['file_path']}")
            continue

        if not info.get('loaded_successfully'):
            print(f"  ✗ FAILED TO LOAD: {info.get('error', 'Unknown error')}")
            continue

        print(f"  ✓ File loaded successfully")
        print(f"  Total rows: {info['n_rows']}")
        print(f"  Total embryos: {info['n_embryos']}")
        print(f"  Total time bins: {info['n_time_bins']}")

        if not info.get('has_pred_proba'):
            print(f"  ✗ MISSING pred_proba COLUMN")
            continue

        print(f"  Pred proba range: {info['pred_proba_range']}")
        print(f"  Pred proba NaNs: {info['pred_proba_nans']}")

        # Per-class breakdown
        if 'class_breakdown' in info:
            print(f"\n  Per-class breakdown:")
            for cls, cls_info in info['class_breakdown'].items():
                print(f"    {cls}:")
                print(f"      Predictions: {cls_info['n_predictions']}")
                print(f"      Embryos: {cls_info['n_embryos']}")
                print(f"      Time bins: {cls_info['n_time_bins']}")
                print(f"      Time range: {cls_info['time_bin_range']}")

        # Test groupby aggregation
        print(f"\n  Testing groupby aggregation (for plotting):")
        agg_results = test_groupby_aggregation(gene, comparison, method_name)

        if 'error' in agg_results:
            print(f"    ✗ AGGREGATION FAILED: {agg_results['error']}")
        else:
            for cls, cls_results in agg_results.items():
                if 'error' in cls_results:
                    print(f"    {cls}: ✗ {cls_results['error']}")
                else:
                    print(f"    {cls}:")
                    print(f"      Time bins with data: {cls_results['n_time_bins']}")
                    print(f"      Mean prob range: {cls_results['mean_probs_range']}")
                    if cls_results['n_time_bins'] > 0:
                        print(f"      ✓ Would plot successfully")
                    else:
                        print(f"      ✗ NO DATA - would not plot")

    # Create summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")

    df_summary = pd.DataFrame(all_info)

    print(f"\n{df_summary[['method', 'file_exists', 'loaded_successfully', 'n_rows', 'n_embryos', 'n_time_bins']].to_string(index=False)}")

    # Save summary
    summary_path = os.path.join(output_dir, f'diagnostic_summary_{gene}_{comparison}.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Visualize data availability
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: File existence
    ax = axes[0, 0]
    methods = df_summary['method'].values
    exists = df_summary['file_exists'].values
    colors = ['green' if e else 'red' for e in exists]
    ax.barh(methods, exists, color=colors, alpha=0.6)
    ax.set_xlabel('File Exists', fontsize=11)
    ax.set_title('File Availability', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1.2])

    # Panel 2: Number of rows
    ax = axes[0, 1]
    n_rows = df_summary['n_rows'].fillna(0).values
    ax.barh(methods, n_rows, alpha=0.6)
    ax.set_xlabel('Number of Rows', fontsize=11)
    ax.set_title('Data Volume', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # Panel 3: Number of embryos
    ax = axes[1, 0]
    n_embryos = df_summary['n_embryos'].fillna(0).values
    ax.barh(methods, n_embryos, alpha=0.6, color='steelblue')
    ax.set_xlabel('Number of Embryos', fontsize=11)
    ax.set_title('Embryo Coverage', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    # Panel 4: Number of time bins
    ax = axes[1, 1]
    n_time_bins = df_summary['n_time_bins'].fillna(0).values
    ax.barh(methods, n_time_bins, alpha=0.6, color='coral')
    ax.set_xlabel('Number of Time Bins', fontsize=11)
    ax.set_title('Temporal Coverage', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')

    fig.suptitle(f'Data Availability Diagnostic\n{gene} - {comparison}',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'diagnostic_plot_{gene}_{comparison}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Diagnostic plot saved to: {plot_path}")
    plt.close()


# ============================================================================
# RUN DIAGNOSTICS
# ============================================================================

print("="*80)
print("PROBABILITY SHIFT MISSING LINES DIAGNOSTIC")
print("="*80)

# Focus on cep290 wildtype vs heterozygous (where lines are missing)
comprehensive_diagnostic('cep290', 'wildtype_vs_heterozygous')

# Also check tmem67 for comparison (where lines appear)
comprehensive_diagnostic('tmem67', 'wildtype_vs_heterozygote')

# Direct comparison
compare_genes('cep290', 'tmem67', 'wildtype_vs_heterozygous', 'wildtype_vs_heterozygote')

print(f"\n{'='*80}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*80}")
print(f"\nOutputs saved to: {output_dir}")
