"""
B9D2 Multiclass Plotting

Generates visualizations for B9D2 multiclass comparison:
1. Per-class OvR AUROC curves (one plot per feature)
2. Feature-comparison panel (all classes overlaid per feature)
3. Embedding overlay (all classes, embedding only)
4. Per-class AUROC comparisons across features
5. Temporal confusion profiles (how classification breaks down over time)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.plotting_functions import (
    plot_multiclass_ovr_aurocs,
    plot_all_temporal_confusion_profiles,
    plot_multiple_aurocs,
    plot_auroc_with_null
)
from utils.preprocessing import extract_temporal_confusion_profile, prepare_auroc_data

OUTPUT_DIR = Path(__file__).parent / "output" / "b9d2_multiclass"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'CE': '#D32F2F',            # Red (strong phenotype)
    'HTA_BA': '#FF9800',        # Orange (late-onset)
    'Not_Penetrant': '#9E9E9E', # Gray (non-penetrant)
}

CLASS_LABELS = ['CE', 'HTA_BA', 'Not_Penetrant']
FEATURES = ['curvature', 'length', 'embedding']

FEATURE_COLORS = {
    'curvature': '#7E57C2',
    'length': '#43A047',
    'embedding': '#1E88E5',
}


def plot_ovr_aurocs_per_feature():
    """Plot OvR AUROC for each feature type."""
    print("\n" + "="*60)
    print("Plotting OvR AUROC curves...")
    print("="*60)

    for feature in FEATURES:
        print(f"\nProcessing {feature}...")
        feature_dir = OUTPUT_DIR / feature

        if not feature_dir.exists():
            print(f"  Skipping {feature} - directory not found")
            continue

        # Load per-class OvR results
        ovr_results = {}
        for class_label in CLASS_LABELS:
            csv_path = feature_dir / f'ovr_auroc_{class_label}.csv'
            if csv_path.exists():
                ovr_results[class_label] = pd.read_csv(csv_path)
                print(f"  Loaded {class_label}: {len(ovr_results[class_label])} time bins")
            else:
                print(f"  Warning: {class_label} data not found")

        if not ovr_results:
            print(f"  No data found for {feature}")
            continue

        # Plot
        save_path = str(FIGURES_DIR / f'ovr_auroc_{feature}.png')
        plot_multiclass_ovr_aurocs(
            ovr_results=ovr_results,
            colors_dict=COLORS,
            title=f'B9D2 Multiclass OvR AUROC: {feature.capitalize()}',
            save_path=save_path
        )
        print(f"  Saved: {save_path}")


def plot_feature_comparison_panel():
    """Create a 1x3 panel with all classes overlaid per feature."""
    print("\n" + "="*60)
    print("Plotting feature comparison panel...")
    print("="*60)

    data_by_feature = {feature: {} for feature in FEATURES}

    for feature in FEATURES:
        feature_dir = OUTPUT_DIR / feature
        for class_label in CLASS_LABELS:
            csv_path = feature_dir / f'ovr_auroc_{class_label}.csv'
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue

            data_by_feature[feature][class_label] = prepare_auroc_data(
                df,
                significance_threshold=0.01
            )

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(18, 5))
    feature_titles = {
        'curvature': 'Curvature-based Classification',
        'length': 'Length-based Classification',
        'embedding': 'Embedding-based Classification',
    }

    for ax, feature in zip(axes, FEATURES):
        for class_label, auroc_df in data_by_feature[feature].items():
            plot_auroc_with_null(
                ax=ax,
                auroc_df=auroc_df,
                color=COLORS.get(class_label, '#000000'),
                label=class_label,
                style='-',
                time_col='time_bin_center',
                show_null_band=True,
                show_significance=True,
                sig_05_marker_size=150
            )

        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        ax.set_ylabel('AUROC', fontsize=11)
        ax.set_title(feature_titles.get(feature, feature), fontsize=12, fontweight='bold')
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if feature == 'curvature':
            ax.scatter([], [], s=150, facecolors='none', edgecolors='black',
                       linewidths=2.5, label='p <= 0.01')
            ax.legend(loc='upper left', fontsize=8)

    plt.suptitle('B9D2 Multiclass: Feature Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / 'multiclass_feature_comparison.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_embedding_overlay():
    """Create an embedding-only overlay of all classes."""
    print("\n" + "="*60)
    print("Plotting embedding overlay...")
    print("="*60)

    embedding_data = {}
    for class_label in CLASS_LABELS:
        csv_path = OUTPUT_DIR / 'embedding' / f'ovr_auroc_{class_label}.csv'
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        embedding_data[class_label] = prepare_auroc_data(
            df,
            significance_threshold=0.01
        )

    if not embedding_data:
        print("  No embedding AUROC data found.")
        return

    fig = plot_multiple_aurocs(
        auroc_dfs_dict=embedding_data,
        colors_dict=COLORS,
        title='B9D2 Multiclass: Embedding OvR AUROC',
        figsize=(12, 7),
        ylim=(0.3, 1.0)
    )

    save_path = FIGURES_DIR / 'multiclass_embedding_overlay.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_auroc_feature_comparisons():
    """Plot per-class AUROC comparisons across features."""
    print("\n" + "="*60)
    print("Plotting AUROC comparisons across features...")
    print("="*60)

    for class_label in CLASS_LABELS:
        auroc_dfs = {}
        colors_dict = {}

        for feature in FEATURES:
            feature_dir = OUTPUT_DIR / feature
            csv_path = feature_dir / f'ovr_auroc_{class_label}.csv'
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            df = prepare_auroc_data(df, significance_threshold=0.01)
            auroc_dfs[feature.capitalize()] = df
            colors_dict[feature.capitalize()] = FEATURE_COLORS.get(feature, '#000000')

        if not auroc_dfs:
            print(f"  Skipping {class_label} - no AUROC data found")
            continue

        fig = plot_multiple_aurocs(
            auroc_dfs_dict=auroc_dfs,
            colors_dict=colors_dict,
            title=f'AUROC by Feature: {class_label} vs Rest'
        )

        save_path = FIGURES_DIR / f'ovr_auroc_feature_comparison_{class_label}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_temporal_profiles():
    """Plot temporal confusion profiles for all classes."""
    print("\n" + "="*60)
    print("Plotting temporal confusion profiles...")
    print("="*60)

    # Load confusion matrices from embedding feature
    feature_dir = OUTPUT_DIR / 'embedding'
    cm_dir = feature_dir / 'confusion_matrices'

    if not cm_dir.exists():
        print("Confusion matrices directory not found. Run analysis first.")
        return

    # Load all confusion matrices
    confusion_matrices = {}
    for cm_path in sorted(cm_dir.glob('cm_t*.csv')):
        time_bin = int(cm_path.stem.split('_t')[1])
        confusion_matrices[time_bin] = pd.read_csv(cm_path, index_col=0)

    if not confusion_matrices:
        print("No confusion matrices found.")
        return

    print(f"Loaded {len(confusion_matrices)} confusion matrices")

    # Extract temporal profile
    temporal_profile = extract_temporal_confusion_profile(
        confusion_matrices=confusion_matrices,
        class_labels=CLASS_LABELS
    )

    if temporal_profile.empty:
        print("Temporal profile is empty - no valid data to plot")
        return

    print(f"Temporal profile: {len(temporal_profile)} records")

    # Plot grid
    save_path = str(FIGURES_DIR / 'temporal_confusion_profiles.png')
    plot_all_temporal_confusion_profiles(
        temporal_profile_df=temporal_profile,
        class_labels=CLASS_LABELS,
        colors_dict=COLORS,
        save_path=save_path
    )
    print(f"  Saved: {save_path}")


def plot_per_class_temporal_profiles():
    """Plot individual temporal confusion profiles (one per class)."""
    print("\n" + "="*60)
    print("Plotting per-class temporal confusion profiles...")
    print("="*60)

    # Load confusion matrices
    feature_dir = OUTPUT_DIR / 'embedding'
    cm_dir = feature_dir / 'confusion_matrices'

    if not cm_dir.exists():
        print("Confusion matrices directory not found. Run analysis first.")
        return

    confusion_matrices = {}
    for cm_path in sorted(cm_dir.glob('cm_t*.csv')):
        time_bin = int(cm_path.stem.split('_t')[1])
        confusion_matrices[time_bin] = pd.read_csv(cm_path, index_col=0)

    if not confusion_matrices:
        print("No confusion matrices found.")
        return

    # Extract temporal profile
    temporal_profile = extract_temporal_confusion_profile(
        confusion_matrices=confusion_matrices,
        class_labels=CLASS_LABELS
    )

    if temporal_profile.empty:
        print("Temporal profile is empty - no valid data to plot")
        return

    # Import plotting function
    from utils.plotting_functions import plot_temporal_confusion_profile

    # Plot for each class
    for class_label in CLASS_LABELS:
        class_profile = temporal_profile[temporal_profile['true_class'] == class_label].copy()

        if class_profile.empty:
            print(f"  Skipping {class_label} - no data")
            continue

        fig = plot_temporal_confusion_profile(
            temporal_profile_df=class_profile,
            class_label=class_label,
            colors_dict=COLORS,
            title=f'{class_label} Classification Over Time'
        )

        save_path = FIGURES_DIR / f'temporal_profile_{class_label}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


if __name__ == '__main__':
    print("B9D2 Multiclass Plotting")
    print("="*60)

    # Generate all plots
    plot_ovr_aurocs_per_feature()
    plot_feature_comparison_panel()
    plot_embedding_overlay()
    plot_auroc_feature_comparisons()
    plot_temporal_profiles()
    plot_per_class_temporal_profiles()

    print("\n" + "="*60)
    print("Plotting Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("="*60)
