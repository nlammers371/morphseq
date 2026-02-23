#!/usr/bin/env python
"""CEP290 Multiclass Plotting

Generates visualizations for CEP290 multiclass comparisons:
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.plotting_functions import (
    plot_multiclass_ovr_aurocs,
    plot_all_temporal_confusion_profiles,
    plot_multiple_aurocs,
    plot_auroc_with_null,
    plot_temporal_confusion_profile,
)
from utils.preprocessing import extract_temporal_confusion_profile, prepare_auroc_data

OUTPUT_DIR = Path(__file__).parent / "output" / "cep290_multiclass"

FEATURES = ['curvature', 'length', 'embedding']

FEATURE_COLORS = {
    'curvature': '#7E57C2',
    'length': '#43A047',
    'embedding': '#1E88E5',
}

CONFIGS = {
    'config1_biological': {
        'title': 'CEP290 Multiclass: HL vs (LH+Int) vs (WT+Het)',
        'class_labels': ['HL', 'LH_Int', 'WT_Het'],
        'colors': {
            'HL': '#C62828',      # Red (High_to_Low)
            'LH_Int': '#F57C00',  # Orange (Low_to_High + Intermediate)
            'WT_Het': '#9E9E9E',  # Gray (controls)
        },
    },
    'config2_trajectory': {
        'title': 'CEP290 Multiclass: Trajectories vs WT',
        'class_labels': ['LowToHigh', 'HighToLow', 'Intermediate', 'WT'],
        'colors': {
            'LowToHigh': '#2E7D32',   # Green
            'HighToLow': '#C62828',   # Red
            'Intermediate': '#F57C00',# Orange
            'WT': '#9E9E9E',          # Gray
        },
    },
    'config3_trajectory_wt_het': {
        'title': 'CEP290 Multiclass: Trajectories vs WT vs Het',
        'class_labels': ['LowToHigh', 'HighToLow', 'Intermediate', 'WT', 'Het'],
        'colors': {
            'LowToHigh': '#2E7D32',   # Green
            'HighToLow': '#C62828',   # Red
            'Intermediate': '#F57C00',# Orange
            'WT': '#9E9E9E',          # Gray
            'Het': '#42A5F5',         # Blue
        },
    },
}


def plot_ovr_aurocs_per_feature(config_name, class_labels, colors, figures_dir, config_output_dir):
    """Plot OvR AUROC for each feature type."""
    print("\n" + "=" * 60)
    print(f"[{config_name}] Plotting OvR AUROC curves...")
    print("=" * 60)

    for feature in FEATURES:
        feature_dir = config_output_dir / feature
        if not feature_dir.exists():
            print(f"  Skipping {feature} - directory not found")
            continue

        ovr_results = {}
        for class_label in class_labels:
            csv_path = feature_dir / f'ovr_auroc_{class_label}.csv'
            if csv_path.exists():
                ovr_results[class_label] = pd.read_csv(csv_path)
            else:
                print(f"  Warning: {class_label} data not found for {feature}")

        if not ovr_results:
            print(f"  No data found for {feature}")
            continue

        save_path = figures_dir / f'ovr_auroc_{feature}.png'
        plot_multiclass_ovr_aurocs(
            ovr_results=ovr_results,
            colors_dict=colors,
            title=f'{config_name}: OvR AUROC ({feature.capitalize()})',
            save_path=str(save_path)
        )
        print(f"  Saved: {save_path}")


def plot_feature_comparison_panel(config_name, class_labels, colors, figures_dir, config_output_dir):
    """Create a 1x3 panel with all classes overlaid per feature."""
    print("\n" + "=" * 60)
    print(f"[{config_name}] Plotting feature comparison panel...")
    print("=" * 60)

    data_by_feature = {feature: {} for feature in FEATURES}

    for feature in FEATURES:
        feature_dir = config_output_dir / feature
        for class_label in class_labels:
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
                color=colors.get(class_label, '#000000'),
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

    plt.suptitle(f'{config_name}: Feature Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = figures_dir / 'feature_comparison.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_embedding_overlay(config_name, class_labels, colors, figures_dir, config_output_dir):
    """Create an embedding-only overlay of all classes."""
    print("\n" + "=" * 60)
    print(f"[{config_name}] Plotting embedding overlay...")
    print("=" * 60)

    embedding_data = {}
    for class_label in class_labels:
        csv_path = config_output_dir / 'embedding' / f'ovr_auroc_{class_label}.csv'
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
        colors_dict=colors,
        title=f'{config_name}: Embedding OvR AUROC',
        figsize=(12, 7),
        ylim=(0.3, 1.0)
    )

    save_path = figures_dir / 'embedding_overlay.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_auroc_feature_comparisons(config_name, class_labels, figures_dir, config_output_dir):
    """Plot per-class AUROC comparisons across features."""
    print("\n" + "=" * 60)
    print(f"[{config_name}] Plotting AUROC comparisons across features...")
    print("=" * 60)

    for class_label in class_labels:
        auroc_dfs = {}
        colors_dict = {}

        for feature in FEATURES:
            feature_dir = config_output_dir / feature
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

        save_path = figures_dir / f'ovr_auroc_feature_comparison_{class_label}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_temporal_profiles(config_name, class_labels, colors, figures_dir, config_output_dir):
    """Plot temporal confusion profiles for all classes."""
    print("\n" + "=" * 60)
    print(f"[{config_name}] Plotting temporal confusion profiles...")
    print("=" * 60)

    cm_dir = config_output_dir / 'embedding' / 'confusion_matrices'
    if not cm_dir.exists():
        print("  Confusion matrices directory not found. Run analysis first.")
        return

    confusion_matrices = {}
    for cm_path in sorted(cm_dir.glob('cm_t*.csv')):
        time_bin = int(cm_path.stem.split('_t')[1])
        confusion_matrices[time_bin] = pd.read_csv(cm_path, index_col=0)

    if not confusion_matrices:
        print("  No confusion matrices found.")
        return

    temporal_profile = extract_temporal_confusion_profile(
        confusion_matrices=confusion_matrices,
        class_labels=class_labels
    )

    if temporal_profile.empty:
        print("  Temporal profile is empty - no valid data to plot")
        return

    save_path = figures_dir / 'temporal_confusion_profiles.png'
    plot_all_temporal_confusion_profiles(
        temporal_profile_df=temporal_profile,
        class_labels=class_labels,
        colors_dict=colors,
        save_path=str(save_path)
    )
    print(f"  Saved: {save_path}")


def plot_per_class_temporal_profiles(config_name, class_labels, colors, figures_dir, config_output_dir):
    """Plot individual temporal confusion profiles (one per class)."""
    print("\n" + "=" * 60)
    print(f"[{config_name}] Plotting per-class temporal confusion profiles...")
    print("=" * 60)

    cm_dir = config_output_dir / 'embedding' / 'confusion_matrices'
    if not cm_dir.exists():
        print("  Confusion matrices directory not found. Run analysis first.")
        return

    confusion_matrices = {}
    for cm_path in sorted(cm_dir.glob('cm_t*.csv')):
        time_bin = int(cm_path.stem.split('_t')[1])
        confusion_matrices[time_bin] = pd.read_csv(cm_path, index_col=0)

    if not confusion_matrices:
        print("  No confusion matrices found.")
        return

    temporal_profile = extract_temporal_confusion_profile(
        confusion_matrices=confusion_matrices,
        class_labels=class_labels
    )

    if temporal_profile.empty:
        print("  Temporal profile is empty - no valid data to plot")
        return

    for class_label in class_labels:
        class_profile = temporal_profile[temporal_profile['true_class'] == class_label].copy()
        if class_profile.empty:
            print(f"  Skipping {class_label} - no data")
            continue

        fig = plot_temporal_confusion_profile(
            temporal_profile_df=class_profile,
            class_label=class_label,
            colors_dict=colors,
            title=f'{class_label} Classification Over Time'
        )

        save_path = figures_dir / f'temporal_profile_{class_label}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def run_config_plots(config_name, config):
    """Run all plots for a single configuration."""
    class_labels = config['class_labels']
    colors = config['colors']
    config_output_dir = OUTPUT_DIR / config_name
    figures_dir = config_output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"CEP290 MULTICLASS PLOTTING: {config_name}")
    print("=" * 70)
    print(f"Output directory: {config_output_dir}")
    print(f"Figures directory: {figures_dir}")

    plot_ovr_aurocs_per_feature(config_name, class_labels, colors, figures_dir, config_output_dir)
    plot_feature_comparison_panel(config_name, class_labels, colors, figures_dir, config_output_dir)
    plot_embedding_overlay(config_name, class_labels, colors, figures_dir, config_output_dir)
    plot_auroc_feature_comparisons(config_name, class_labels, figures_dir, config_output_dir)
    plot_temporal_profiles(config_name, class_labels, colors, figures_dir, config_output_dir)
    plot_per_class_temporal_profiles(config_name, class_labels, colors, figures_dir, config_output_dir)

    print("\n" + "=" * 70)
    print(f"Done. Figures saved to: {figures_dir}")
    print("=" * 70)


if __name__ == '__main__':
    print("CEP290 Multiclass Plotting")

    for name, cfg in CONFIGS.items():
        run_config_plots(name, cfg)
