#!/usr/bin/env python
"""Control vs Control Analysis - Plotting

Visualizes the 6 control validation comparisons:
- Pair-specific het phenotype detection (pair_2, pair_8)
- Negative control (pair_2 WT vs pair_8 WT)
- Pooled signal (all non-pen hets vs all WTs)
- Experiment-specific signals (20251121, 20251125)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.preprocessing import prepare_auroc_data
from utils.plotting_functions import plot_multiple_aurocs, plot_auroc_with_null

# ============================================================================
# Configuration
# ============================================================================
OUTPUT_DIR = Path(__file__).parent / "output" / "control_controls"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Comparison definitions (display name -> directory name)
COMPARISONS = {
    'pair2_Het_vs_WT': 'pair2_het_vs_wt',
    'pair8_Het_vs_WT': 'pair8_het_vs_wt',
    'pair2_WT_vs_pair8_WT': 'pair2_wt_vs_pair8_wt',
    'AllNonPenHets_vs_WT': 'allnonpenhets_vs_wt',
    'Exp20251121_Het_vs_WT': 'exp20251121_het_vs_wt',
    'Exp20251125_Het_vs_WT': 'exp20251125_het_vs_wt',
}

# Color scheme
COLORS = {
    'pair2_Het_vs_WT': '#D32F2F',        # Red (pair-specific validation)
    'pair8_Het_vs_WT': '#FF6F00',        # Orange (pair-specific validation)
    'pair2_WT_vs_pair8_WT': '#888888',   # Gray (negative control)
    'AllNonPenHets_vs_WT': '#1976D2',    # Blue (pooled signal)
    'Exp20251121_Het_vs_WT': '#388E3C',  # Green (experiment 1)
    'Exp20251125_Het_vs_WT': '#7B1FA2',  # Purple (experiment 2)
}

# Line styles
STYLES = {
    'pair2_Het_vs_WT': '-',
    'pair8_Het_vs_WT': '-',
    'pair2_WT_vs_pair8_WT': '--',        # Dashed for negative control
    'AllNonPenHets_vs_WT': '-',
    'Exp20251121_Het_vs_WT': '-',
    'Exp20251125_Het_vs_WT': '-',
}

# ============================================================================
# Helper Functions
# ============================================================================
def load_classification_data(comparison_dir, feature):
    """Load and prepare classification CSV for a specific comparison and feature.

    Parameters
    ----------
    comparison_dir : str
        Directory name for the comparison (e.g., 'pair2_het_vs_wt')
    feature : str
        Feature type ('curvature', 'length', or 'embedding')

    Returns
    -------
    pd.DataFrame or None
        Prepared AUROC data with significance flags, or None if file not found
    """
    csv_path = OUTPUT_DIR / comparison_dir / f'classification_{feature}.csv'
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return None

    # Check if file is empty (size <= 1 byte)
    if csv_path.stat().st_size <= 1:
        print(f"Warning: {csv_path} is empty (comparison likely had insufficient data)")
        return None

    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print(f"Warning: {csv_path} has no data rows")
            return None
        return prepare_auroc_data(df)
    except Exception as e:
        print(f"Warning: Error loading {csv_path}: {e}")
        return None


def load_all_features_for_comparison(comparison_dir):
    """Load all three feature types for a comparison.

    Parameters
    ----------
    comparison_dir : str
        Directory name for the comparison

    Returns
    -------
    dict
        Dictionary with keys 'curvature', 'length', 'embedding'
    """
    return {
        'curvature': load_classification_data(comparison_dir, 'curvature'),
        'length': load_classification_data(comparison_dir, 'length'),
        'embedding': load_classification_data(comparison_dir, 'embedding'),
    }


# ============================================================================
# Plotting Functions
# ============================================================================
def create_feature_comparison_figure():
    """Create 1x3 feature comparison panel (all comparisons overlaid).

    Creates three subplots showing AUROC over time for:
    - Panel 1: Curvature-based classification
    - Panel 2: Length-based classification
    - Panel 3: Embedding-based classification

    All 6 comparisons are overlaid on each panel.
    """
    print("\n  Loading data for all comparisons and features...")

    # Load data for all comparisons and features
    data_by_feature = {
        'curvature': {},
        'length': {},
        'embedding': {}
    }

    for comp_name, comp_dir in COMPARISONS.items():
        for feature in ['curvature', 'length', 'embedding']:
            data = load_classification_data(comp_dir, feature)
            if data is not None:
                data_by_feature[feature][comp_name] = data

    # Create 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    feature_titles = {
        'curvature': 'NONPEN Het Validation: Curvature',
        'length': 'NONPEN Het Validation: Length',
        'embedding': 'NONPEN Het Validation: Embedding'
    }

    for ax, (feature, title) in zip(axes, feature_titles.items()):
        # Plot all comparisons for this feature using standardized function
        for comp_name, auroc_df in data_by_feature[feature].items():
            time_col = 'time_bin_center' if 'time_bin_center' in auroc_df.columns else 'time_bin'

            # Use standardized plotting function
            plot_auroc_with_null(
                ax=ax,
                auroc_df=auroc_df,
                color=COLORS[comp_name],
                label=comp_name,
                style=STYLES[comp_name],
                time_col=time_col,
                show_null_band=True,
                show_significance=True,
                sig_05_marker_size=150
            )

        # Reference line and formatting
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        ax.set_ylabel('AUROC', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend only on first panel (include significance marker)
        if feature == 'curvature':
            # Add significance marker to legend
            ax.scatter([], [], s=150, facecolors='none', edgecolors='black',
                      linewidths=2.5, label='p < 0.05')
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    save_path = FIGURES_DIR / 'feature_comparison.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")

    plt.close(fig)


def create_individual_comparison_plots():
    """Create separate plots for each comparison showing all 3 features overlaid."""
    print("\n  Creating individual comparison plots...")

    for comp_name, comp_dir in COMPARISONS.items():
        # Load all features for this comparison
        feature_data = load_all_features_for_comparison(comp_dir)

        # Skip if no data
        if all(v is None for v in feature_data.values()):
            print(f"    Skipping {comp_name} (no data found)")
            continue

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        feature_colors = {
            'curvature': '#E57373',
            'length': '#64B5F6',
            'embedding': '#81C784'
        }

        feature_labels = {
            'curvature': 'Curvature',
            'length': 'Length',
            'embedding': 'Embedding'
        }

        for feature, auroc_df in feature_data.items():
            if auroc_df is None:
                continue

            time_col = 'time_bin_center' if 'time_bin_center' in auroc_df.columns else 'time_bin'

            # Use standardized plotting function
            plot_auroc_with_null(
                ax=ax,
                auroc_df=auroc_df,
                color=feature_colors[feature],
                label=feature_labels[feature],
                style='-',
                time_col=time_col,
                show_null_band=True,
                show_significance=True,
                sig_05_marker_size=150
            )

        # Formatting
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'{comp_name}', fontsize=14, fontweight='bold')
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        # Add significance marker to legend
        ax.scatter([], [], s=150, facecolors='none', edgecolors='black',
                  linewidths=2.5, label='p < 0.05')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save
        safe_name = comp_dir.replace('_', '_')
        save_path = FIGURES_DIR / f'{safe_name}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: {save_path}")

        plt.close(fig)


def create_embedding_overlay():
    """Create overlay plot focused on embedding feature (all comparisons)."""
    print("\n  Creating embedding overlay...")

    # Load embedding data for all comparisons
    embedding_data = {}
    for comp_name, comp_dir in COMPARISONS.items():
        data = load_classification_data(comp_dir, 'embedding')
        if data is not None:
            embedding_data[comp_name] = data

    if not embedding_data:
        print("    Warning: No embedding data found")
        return

    # Create figure using plot_multiple_aurocs utility
    fig = plot_multiple_aurocs(
        auroc_dfs_dict=embedding_data,
        colors_dict=COLORS,
        styles_dict=STYLES,
        title='Control Comparisons: Embedding-based Classification',
        save_path=str(FIGURES_DIR / 'embedding_overlay.png'),
        figsize=(12, 7),
        ylim=(0.3, 1.0)
    )

    print(f"    ✓ Saved: {FIGURES_DIR / 'embedding_overlay.png'}")
    plt.close(fig)


def create_experiment_comparison():
    """Create overlay comparing experiment-specific and pooled comparisons."""
    print("\n  Creating experiment comparison overlay...")

    # Focus on experiment-specific comparisons
    exp_comparisons = {
        'Exp20251121_Het_vs_WT': 'exp20251121_het_vs_wt',
        'Exp20251125_Het_vs_WT': 'exp20251125_het_vs_wt',
        'AllNonPenHets_vs_WT': 'allnonpenhets_vs_wt',
    }

    # Load embedding data
    embedding_data = {}
    for comp_name, comp_dir in exp_comparisons.items():
        data = load_classification_data(comp_dir, 'embedding')
        if data is not None:
            embedding_data[comp_name] = data

    if not embedding_data:
        print("    Warning: No experiment comparison data found")
        return

    # Create figure
    fig = plot_multiple_aurocs(
        auroc_dfs_dict=embedding_data,
        colors_dict=COLORS,
        styles_dict=STYLES,
        title='Experiment Comparison: Non-Penetrant Hets vs WT',
        save_path=str(FIGURES_DIR / 'experiment_comparison.png'),
        figsize=(12, 7),
        ylim=(0.3, 1.0)
    )

    print(f"    ✓ Saved: {FIGURES_DIR / 'experiment_comparison.png'}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*70)
    print("CREATING PLOTS FOR CONTROL VS CONTROL ANALYSIS")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"\nNumber of comparisons: {len(COMPARISONS)}")

    # 1. Feature comparison panel (1x3)
    print("\n[1/4] Creating feature comparison panel...")
    create_feature_comparison_figure()

    # 2. Individual comparison plots
    print("\n[2/4] Creating individual comparison plots...")
    create_individual_comparison_plots()

    # 3. Embedding overlay (all comparisons)
    print("\n[3/4] Creating embedding overlay...")
    create_embedding_overlay()

    # 4. Experiment comparison
    print("\n[4/4] Creating experiment comparison overlay...")
    create_experiment_comparison()

    print("\n" + "="*70)
    print(f"DONE! All figures saved to: {FIGURES_DIR}")
    print("="*70)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
