#!/usr/bin/env python
"""CEP290 Phenotype Analysis - Plotting

Visualizes CEP290 phenotypic trajectory comparisons:
- Trajectory (Homo only) vs WT
- Trajectory (Homo only) vs Het
- Cross-trajectory comparisons

Scientific focus: Do different phenotypic trajectories emerge from the same
genetic background (homozygous), and can we distinguish them from controls
and from each other?
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
OUTPUT_DIR = Path(__file__).parent / "output" / "cep290_phenotype"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Comparison definitions
COMPARISONS = {
    # Trajectory vs WT
    'LowToHigh_vs_WT': 'lowtohigh_vs_wt',
    'HighToLow_vs_WT': 'hightolow_vs_wt',
    'Intermediate_vs_WT': 'intermediate_vs_wt',

    # Trajectory vs Het
    'LowToHigh_vs_Het': 'lowtohigh_vs_het',
    'HighToLow_vs_Het': 'hightolow_vs_het',
    'Intermediate_vs_Het': 'intermediate_vs_het',

    # Cross-trajectory
    'LowToHigh_vs_HighToLow': 'lowtohigh_vs_hightolow',
    'LowToHigh_vs_Intermediate': 'lowtohigh_vs_intermediate',
    'HighToLow_vs_Intermediate': 'hightolow_vs_intermediate',
}

# Color scheme
COLORS = {
    # Trajectory vs WT comparisons
    'LowToHigh_vs_WT': '#2E7D32',           # Green (Low->High trajectory)
    'HighToLow_vs_WT': '#C62828',           # Red (High->Low trajectory)
    'Intermediate_vs_WT': '#F57C00',        # Orange (Intermediate)

    # Trajectory vs Het comparisons
    'LowToHigh_vs_Het': '#66BB6A',          # Light Green
    'HighToLow_vs_Het': '#EF5350',          # Light Red
    'Intermediate_vs_Het': '#FFA726',       # Light Orange

    # Cross-trajectory comparisons
    'LowToHigh_vs_HighToLow': '#1976D2',    # Blue (primary cross-trajectory)
    'LowToHigh_vs_Intermediate': '#7B1FA2', # Purple
    'HighToLow_vs_Intermediate': '#00796B', # Teal
}

# Line styles
STYLES = {
    # Trajectory vs WT - solid
    'LowToHigh_vs_WT': '-',
    'HighToLow_vs_WT': '-',
    'Intermediate_vs_WT': '-',

    # Trajectory vs Het - dashed
    'LowToHigh_vs_Het': '--',
    'HighToLow_vs_Het': '--',
    'Intermediate_vs_Het': '--',

    # Cross-trajectory - solid
    'LowToHigh_vs_HighToLow': '-',
    'LowToHigh_vs_Intermediate': '-',
    'HighToLow_vs_Intermediate': '-',
}

# ============================================================================
# Helper Functions
# ============================================================================
def load_classification_data(comparison_dir, feature):
    """Load and prepare classification CSV for a specific comparison and feature."""
    csv_path = OUTPUT_DIR / comparison_dir / f'classification_{feature}.csv'

    if not csv_path.exists():
        print(f"  ⚠ Not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    return prepare_auroc_data(df)


def load_all_features_for_comparison(comparison_dir):
    """Load classification data for all three features for a given comparison.

    Returns
    -------
    dict
        {'curvature': df, 'length': df, 'embedding': df}
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
    """Create 1x3 panel showing all comparisons for each feature type.

    Each panel shows curvature, length, or embedding with all 9 comparisons overlaid.
    """
    print("\n  Creating feature comparison figure (1x3 panel)...")

    # Load all data
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
        'curvature': 'Curvature-based Classification',
        'length': 'Length-based Classification',
        'embedding': 'Embedding-based Classification'
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
            ax.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)

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
        ax.set_title(f'CEP290 Phenotype: {comp_name}', fontsize=14, fontweight='bold')
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
        print("    ⚠ No embedding data found, skipping overlay")
        return

    # Create overlay using plot_multiple_aurocs
    fig = plot_multiple_aurocs(
        auroc_dfs_dict=embedding_data,
        colors_dict=COLORS,
        styles_dict=STYLES,
        title='CEP290 Phenotype: Embedding-based Classification',
        save_path=str(FIGURES_DIR / 'cep290_embedding_overlay.png'),
        figsize=(14, 8),
        ylim=(0.3, 1.0)
    )

    print(f"    ✓ Saved: {FIGURES_DIR / 'cep290_embedding_overlay.png'}")

    plt.close(fig)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Generate all plots."""
    print("="*70)
    print("CEP290 PHENOTYPE PLOTTING")
    print("="*70)
    print(f"\nOutput directory: {FIGURES_DIR}")

    # Create plots
    create_feature_comparison_figure()
    create_individual_comparison_plots()
    create_embedding_overlay()

    print("\n" + "="*70)
    print("PLOTTING COMPLETE")
    print("="*70)
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("\nGenerated plots:")
    print("  - feature_comparison.png (1x3 panel)")
    print("  - lowtohigh_vs_wt.png")
    print("  - lowtohigh_vs_het.png")
    print("  - hightolow_vs_wt.png")
    print("  - hightolow_vs_het.png")
    print("  - intermediate_vs_wt.png")
    print("  - intermediate_vs_het.png")
    print("  - lowtohigh_vs_hightolow.png")
    print("  - lowtohigh_vs_intermediate.png")
    print("  - hightolow_vs_intermediate.png")
    print("  - cep290_embedding_overlay.png")
    print("="*70)


if __name__ == '__main__':
    main()
