#!/usr/bin/env python
"""B9D2 HTA Phenotype Analysis - Plotting

Visualizes HTA (Head-Trunk Angle) phenotype comparisons:
- HTA vs Non-Penetrant Hets (primary comparison)
- HTA vs WT (secondary comparison)
- Non-Penetrant Hets vs WT (baseline - tests if controls are equivalent)

Scientific focus: HTA is a late-onset phenotype (>60 hpf). Both metrics
and embeddings should detect HTA at similar times.
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
OUTPUT_DIR = Path(__file__).parent / "output" / "b9d2_hta"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Comparison definitions
COMPARISONS = {
    'HTA_vs_NonPenHets': 'hta_vs_nonpenhets',
    'HTA_vs_WT': 'hta_vs_wt',
    'NonPenHets_vs_WT': 'nonpenhets_vs_wt',
}

# Color scheme
COLORS = {
    'HTA_vs_NonPenHets': '#D32F2F',  # Red (primary comparison)
    'HTA_vs_WT': '#9467BD',          # Purple (secondary comparison)
    'NonPenHets_vs_WT': '#888888',   # Gray (baseline validation)
}

# Line styles
STYLES = {
    'HTA_vs_NonPenHets': '-',        # Solid
    'HTA_vs_WT': '-',                # Solid
    'NonPenHets_vs_WT': '--',        # Dashed (weaker signal expected)
}

# ============================================================================
# Helper Functions
# ============================================================================
def load_classification_data(comparison_dir, feature):
    """Load and prepare classification CSV for a specific comparison and feature."""
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
    """Load all three feature types for a comparison."""
    return {
        'curvature': load_classification_data(comparison_dir, 'curvature'),
        'length': load_classification_data(comparison_dir, 'length'),
        'embedding': load_classification_data(comparison_dir, 'embedding'),
    }


# ============================================================================
# Plotting Functions
# ============================================================================
def create_feature_comparison_figure():
    """Create 1x3 feature comparison panel (all comparisons overlaid)."""
    print("\n  Loading data for all comparisons and features...")

    # Load data
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

        # # Add 60 hpf landmark (expected HTA onset)
        # ax.axvline(x=60, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
        # ax.text(60, 0.95, '60 hpf', ha='center', va='top', fontsize=9,
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Reference line and formatting
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        ax.set_ylabel('AUROC', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend on first panel (include significance marker)
        if feature == 'curvature':
            ax.scatter([], [], s=150, facecolors='none', edgecolors='black',
                      linewidths=2.5, label='p < 0.05')
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    plt.suptitle('B9D2 HTA Phenotype: Feature Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / 'hta_feature_comparison.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")

    plt.close(fig)


def create_individual_comparison_plots():
    """Create separate plots for each comparison showing all 3 features overlaid."""
    print("\n  Creating individual comparison plots...")

    for comp_name, comp_dir in COMPARISONS.items():
        # Load all features
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

        # # Add 60 hpf landmark
        # ax.axvline(x=60, color='red', linestyle='--', alpha=0.4, linewidth=2)
        # ax.text(60, 0.35, '60 hpf\n(Expected HTA onset)', ha='center', va='bottom',
        #         fontsize=10, bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
        #         alpha=0.8, edgecolor='red'))

        # Formatting
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        # Add context to title based on comparison type
        if 'NonPenHets_vs_WT' in comp_name:
            title_prefix = 'NONPEN Het Validation'
        else:
            title_prefix = 'B9D2 HTA'
        ax.set_title(f'{title_prefix}: {comp_name}', fontsize=14, fontweight='bold')
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

    # Create figure
    fig = plot_multiple_aurocs(
        auroc_dfs_dict=embedding_data,
        colors_dict=COLORS,
        styles_dict=STYLES,
        title='B9D2 HTA: Embedding-based Classification',
        save_path=str(FIGURES_DIR / 'hta_embedding_overlay.png'),
        figsize=(12, 7),
        ylim=(0.3, 1.0)
    )

    # # Add 60 hpf landmark to the existing figure
    # ax = fig.gca()
    # ax.axvline(x=60, color='red', linestyle='--', alpha=0.4, linewidth=2)
    # ax.text(60, 0.95, '60 hpf (HTA onset)', ha='center', va='top', fontsize=10,
    #         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='red'))

    # Re-save with landmark (disabled)
    fig.savefig(FIGURES_DIR / 'hta_embedding_overlay.png', dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {FIGURES_DIR / 'hta_embedding_overlay.png'}")

    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*70)
    print("CREATING PLOTS FOR B9D2 HTA PHENOTYPE ANALYSIS")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"\nNumber of comparisons: {len(COMPARISONS)}")
    print("\nScientific note: HTA is a late-onset phenotype (>60 hpf)")

    # 1. Feature comparison panel (1x3)
    print("\n[1/4] Creating feature comparison panel...")
    create_feature_comparison_figure()

    # 2. Individual comparison plots
    print("\n[2/4] Creating individual comparison plots...")
    create_individual_comparison_plots()

    # 3. Embedding overlay (all comparisons)
    print("\n[3/4] Creating embedding overlay...")
    create_embedding_overlay()

    print("\n" + "="*70)
    print(f"DONE! All figures saved to: {FIGURES_DIR}")
    print("="*70)

    # List generated files
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
