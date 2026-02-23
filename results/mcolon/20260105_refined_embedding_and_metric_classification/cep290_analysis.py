"""CEP290 cryptic phenotype analysis.

Tests hypothesis: Embedding detects phenotype at ~18 hpf, before curvature at ~24 hpf.

Scientific Background:
- Cep290 is a ciliopathy gene; mutants show body axis curvature defects
- Hypothesis: Earliest morphological difference (~18 hpf) is ONLY detectable in embedding space
- Curvature metric differences appear later (>24 hpf)
- The ~18 hpf signal represents a "cryptic phenotype" - subtle shape changes before overt curvature
"""
import sys
from pathlib import Path

# Add project root and src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # results/mcolon/20260105_.../cep290.py -> morphseq/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # For analyze.* imports

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Existing APIs (NO reimplementation)
from analyze.difference_detection.comparison import compare_groups

# Local thin wrappers
from utils.data_prep import prepare_comparison_data
from utils.divergence import compute_multi_metric_divergence, zscore_divergence
from utils.cryptic_window import detect_cryptic_window, detect_cryptic_window_from_aurocs
from utils.plotting import create_comparison_figure

BIN_WIDTH = 4 
BOOTSTRAP_PERMS = 500

def main():
    print("=" * 70)
    print("CEP290 Cryptic Phenotype Analysis")
    print("=" * 70)

    # =========================================================================
    # Configuration
    # =========================================================================
    DATA_PATH = PROJECT_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    OUTPUT_DIR = Path(__file__).parent / "output" / f"cep290_binwidth_{BIN_WIDTH}_nperms_{BOOTSTRAP_PERMS}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Data Loading
    # =========================================================================
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {len(df)} rows, {df['embryo_id'].nunique()} unique embryos")

    # Define groups based on cluster_categories
    penetrant_categories = ['Low_to_High', 'High_to_Low', 'Intermediate']
    penetrant_ids = df[df['cluster_categories'].isin(penetrant_categories)]['embryo_id'].unique().tolist()
    control_ids = df[df['cluster_categories'] == 'Not Penetrant']['embryo_id'].unique().tolist()

    print(f"\nGroup sizes:")
    print(f"  Penetrant embryos: {len(penetrant_ids)}")
    print(f"  Control (Not Penetrant) embryos: {len(control_ids)}")

    # Prepare data with group column
    df_prep = prepare_comparison_data(
        df,
        group1_ids=penetrant_ids,
        group2_ids=control_ids,
        group1_label='Penetrant',
        group2_label='Control'
    )
    print(f"  Total rows after filtering: {len(df_prep)}")

    # =========================================================================
    # Classification: Metric-based (curvature as features)
    # =========================================================================
    print("\n" + "-" * 50)
    print("Running metric-based classification...")

    metric_results = compare_groups(
        df_prep,
        group_col='group',
        group1='Penetrant',      # POSITIVE class
        group2='Control',        # NEGATIVE class
        features=['baseline_deviation_normalized'],  # Curvature as classification features
        morphology_metric=None,  # Don't compute internal divergence (we do it separately)
        bin_width=BIN_WIDTH,
        n_permutations=BOOTSTRAP_PERMS,
        n_jobs=-1,  # Use all available CPUs
        verbose=True,
    )
    metric_auroc = metric_results['classification']
    print(f"Metric AUROC computed for {len(metric_auroc)} time bins")

    # =========================================================================
    # Classification: Embedding-based
    # =========================================================================
    print("\n" + "-" * 50)
    print("Running embedding-based classification...")

    embedding_results = compare_groups(
        df_prep,
        group_col='group',
        group1='Penetrant',
        group2='Control',
        features='z_mu_b',  # Auto-detects all z_mu_b_* columns
        morphology_metric=None,
        bin_width=BIN_WIDTH,
        n_permutations=BOOTSTRAP_PERMS,
        n_jobs=-1,  # Use all available CPUs
        verbose=True,
    )

    # CAUTION (early-time discrepancies):
    # With bin_width=4.0, time_bin=12 corresponds to the full 12–16 hpf window.
    # If Penetrant vs Control embryos have different within-bin stage distributions
    # (e.g. Penetrant skew older within the bin), a classifier can appear significant
    # even if "by-eye" trajectories overlap. Use `cep290_validation_analysis.py`
    # to audit bin-width sensitivity and within-bin time confounds.
    embedding_auroc = embedding_results['classification']
    print(f"Embedding AUROC computed for {len(embedding_auroc)} time bins")

    # =========================================================================
    # Metric Divergence: Multiple metrics
    # =========================================================================
    print("\n" + "-" * 50)
    print("Computing metric divergence...")

    metrics = ['baseline_deviation_normalized', 'total_length_um']
    divergence = compute_multi_metric_divergence(
        df_prep,
        group_col='group',
        group1_label='Penetrant',
        group2_label='Control',
        metric_cols=metrics,
    )
    divergence = zscore_divergence(divergence)
    print(f"Divergence computed: {len(divergence)} rows")

    # =========================================================================
    # Cryptic Window Detection
    # =========================================================================
    print("\n" + "-" * 50)
    print("Detecting cryptic window...")

    # Method 1: Using divergence z-scores
    cryptic = detect_cryptic_window(
        embedding_auroc,
        divergence,
        auroc_threshold=0.6,
        auroc_pval_threshold=0.05,
        divergence_zscore_threshold=1.0,
    )

    # Method 2: Using AUROC comparison directly
    cryptic_auroc = detect_cryptic_window_from_aurocs(
        embedding_auroc,
        metric_auroc,
        auroc_threshold=0.6,
        auroc_pval_threshold=0.05,
    )

    print("\nCryptic Window Analysis (method 1: embedding vs divergence):")
    print(f"  Has cryptic window: {cryptic['has_cryptic_window']}")
    print(f"  Embedding first signal: {cryptic['embedding_first_signal_hpf']} hpf")
    print(f"  Metric first divergence: {cryptic['metric_first_divergence_hpf']} hpf")
    print(f"  Window duration: {cryptic['cryptic_window_duration_hours']} hours")

    print("\nCryptic Window Analysis (method 2: embedding AUROC vs metric AUROC):")
    print(f"  Has cryptic window: {cryptic_auroc['has_cryptic_window']}")
    print(f"  Embedding first signal: {cryptic_auroc['embedding_first_signal_hpf']} hpf")
    print(f"  Metric first signal: {cryptic_auroc['metric_first_divergence_hpf']} hpf")
    print(f"  Window duration: {cryptic_auroc['cryptic_window_duration_hours']} hours")

    # =========================================================================
    # Plotting
    # =========================================================================
    print("\n" + "-" * 50)
    print("Creating plots...")

    metric_label_map = {
        'baseline_deviation_normalized': 'Curvature',
        'total_length_um': 'Body Length',
    }

    # Plot 1: Metric AUROC only
    fig1 = create_comparison_figure(
        auroc_df=metric_auroc,
        divergence_df=divergence,
        df_trajectories=df_prep,
        group1_label='Penetrant',
        group2_label='Control',
        metric_cols=metrics,
        metric_labels=metric_label_map,
        time_landmarks={24.0: '24 hpf'},
        title='CEP290: Metric-Based Classification',
        save_path=OUTPUT_DIR / 'cep290_metric_auroc_only.png',
    )
    plt.close(fig1)

    # Plot 2: Embedding vs Metric overlay
    fig2 = create_comparison_figure(
        auroc_df=metric_auroc,
        divergence_df=divergence,
        df_trajectories=df_prep,
        group1_label='Penetrant',
        group2_label='Control',
        metric_cols=metrics,
        embedding_auroc_df=embedding_auroc,  # Add overlay
        metric_labels=metric_label_map,
        time_landmarks={24.0: '24 hpf'},
        title='CEP290: Embedding vs Metric Classification',
        save_path=OUTPUT_DIR / 'cep290_embedding_vs_metric.png',
    )
    plt.close(fig2)

    # =========================================================================
    # Plot 3: Temporal Emergence Bar Plot
    # =========================================================================
    print("\n" + "-" * 50)
    print("Creating temporal emergence bar plot...")

    from utils.plotting import plot_temporal_emergence

    # Prepare results dict for temporal emergence plot
    results_dict = {
        'Penetrant_vs_Control': metric_results,
    }
    colors = {'Penetrant_vs_Control': '#D32F2F'}

    fig3 = plot_temporal_emergence(
        results_dict,
        colors=colors,
        time_bin_width=4.0,
        title_prefix='CEP290: ',
        save_path=OUTPUT_DIR / 'cep290_temporal_emergence.png',
    )
    plt.close(fig3)

    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n" + "-" * 50)
    print("Saving CSV results...")

    metric_auroc.to_csv(OUTPUT_DIR / 'metric_auroc.csv', index=False)
    embedding_auroc.to_csv(OUTPUT_DIR / 'embedding_auroc.csv', index=False)
    divergence.to_csv(OUTPUT_DIR / 'divergence.csv', index=False)

    # Save cryptic window summary
    cryptic_summary = pd.DataFrame([{
        'comparison': 'CEP290_Penetrant_vs_Control',
        'method': 'divergence_zscore',
        **cryptic
    }, {
        'comparison': 'CEP290_Penetrant_vs_Control',
        'method': 'auroc_comparison',
        **cryptic_auroc
    }])
    cryptic_summary.to_csv(OUTPUT_DIR / 'cryptic_window_analysis.csv', index=False)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nEarliest significant AUROC:")
    print(f"  Embedding: {embedding_results['summary']['earliest_significant_hpf']} hpf")
    print(f"  Metric: {metric_results['summary']['earliest_significant_hpf']} hpf")

    print(f"\nMax AUROC:")
    print(f"  Embedding: {embedding_results['summary']['max_auroc']:.3f} at {embedding_results['summary']['max_auroc_hpf']} hpf")
    print(f"  Metric: {metric_results['summary']['max_auroc']:.3f} at {metric_results['summary']['max_auroc_hpf']} hpf")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  - cep290_metric_auroc_only.png")
    print("  - cep290_embedding_vs_metric.png")
    print("  - cep290_temporal_emergence.png")
    print("  - metric_auroc.csv")
    print("  - embedding_auroc.csv")
    print("  - divergence.csv")
    print("  - cryptic_window_analysis.csv")

    return {
        'metric_auroc': metric_auroc,
        'embedding_auroc': embedding_auroc,
        'divergence': divergence,
        'cryptic_window': cryptic,
        'cryptic_window_auroc': cryptic_auroc,
        'summary': {
            'embedding': embedding_results['summary'],
            'metric': metric_results['summary'],
        }
    }


if __name__ == '__main__':
    results = main()
