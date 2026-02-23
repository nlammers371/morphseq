#!/usr/bin/env python
"""CEP290 One-vs-WT vs One-vs-Rest Plotting

Overlays binary (phenotype vs WT) against multiclass OvR (phenotype vs rest)
for each phenotype and feature, plus a single 3x2 summary figure that
contrasts binary vs multiclass by feature.
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
from utils.plotting_functions import plot_auroc_with_null

BASE_DIR = Path(__file__).parent
MULTICLASS_CONFIG = "config3_trajectory_wt_het"  # Includes Het and WT as separate classes

MULTICLASS_DIR = BASE_DIR / "output" / "cep290_multiclass" / MULTICLASS_CONFIG
BINARY_DIR = BASE_DIR / "output" / "cep290_phenotype"
FIGURES_DIR = BASE_DIR / "output" / "cep290_one_vs_rest_vs_wt" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PHENOTYPES = ['LowToHigh', 'HighToLow', 'Intermediate']
LEFT_LABELS = ['LowToHigh', 'HighToLow', 'Intermediate', 'Het']
RIGHT_LABELS = ['LowToHigh', 'HighToLow', 'Intermediate', 'Het', 'WT']
FEATURES = ['curvature', 'length', 'embedding']

PHENOTYPE_COLORS = {
    'LowToHigh': '#2E7D32',   # Green
    'HighToLow': '#C62828',   # Red
    'Intermediate': '#F57C00',# Orange
    'Het': '#42A5F5',         # Blue
    'WT': '#9E9E9E',          # Gray
}

FEATURE_COLORS = {
    'curvature': '#7E57C2',
    'length': '#43A047',
    'embedding': '#1E88E5',
}


def load_binary_df(phenotype, feature):
    """Load binary phenotype vs WT AUROC."""
    if phenotype == 'Het':
        comparison_dir = "het_vs_wt"
    else:
        comparison_dir = f"{phenotype}_vs_WT".lower()
    csv_path = BINARY_DIR / comparison_dir / f'classification_{feature}.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return prepare_auroc_data(df, significance_threshold=0.01)


def load_multiclass_df(phenotype, feature):
    """Load multiclass OvR AUROC for phenotype vs rest."""
    csv_path = MULTICLASS_DIR / feature / f'ovr_auroc_{phenotype}.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return prepare_auroc_data(df, significance_threshold=0.01)


def plot_phenotype_panel(phenotype):
    """Create a 1x3 panel comparing binary vs multiclass for a phenotype."""
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(18, 5))

    for ax, feature in zip(axes, FEATURES):
        binary_df = load_binary_df(phenotype, feature)
        multiclass_df = load_multiclass_df(phenotype, feature)

        if multiclass_df is None and binary_df is None:
            ax.text(0.5, 0.5, f"No data\n({feature})",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        color = FEATURE_COLORS.get(feature, '#000000')

        if multiclass_df is not None:
            time_col_mc = 'time_bin_center' if 'time_bin_center' in multiclass_df.columns else 'time_bin'
            ax.plot(
                multiclass_df[time_col_mc],
                multiclass_df['auroc_observed'],
                'o-',
                color=color,
                linewidth=2,
                markersize=4,
                label='Multiclass (OvR)'
            )

        if binary_df is not None:
            time_col_bin = 'time_bin_center' if 'time_bin_center' in binary_df.columns else 'time_bin'
            ax.plot(
                binary_df[time_col_bin],
                binary_df['auroc_observed'],
                's--',
                color=color,
                linewidth=2,
                markersize=4,
                alpha=0.8,
                label='Binary (vs WT)'
            )

        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        ax.set_ylabel('AUROC', fontsize=11)
        ax.set_title(feature.capitalize(), fontsize=12)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if feature == 'curvature':
            ax.legend(loc='upper left', fontsize=9)

    plt.suptitle(f'CEP290: {phenotype} (vs WT) vs (vs Rest)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / f'one_vs_rest_vs_wt_{phenotype.lower()}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def write_diff_summary():
    """Write a CSV summary of max AUROC deltas per phenotype/feature."""
    rows = []

    for phenotype in PHENOTYPES:
        for feature in FEATURES:
            binary_df = load_binary_df(phenotype, feature)
            multiclass_df = load_multiclass_df(phenotype, feature)
            if binary_df is None or multiclass_df is None:
                continue

            merged = pd.merge(
                multiclass_df[['time_bin', 'auroc_observed']],
                binary_df[['time_bin', 'auroc_observed']],
                on='time_bin',
                suffixes=('_multiclass', '_binary')
            )
            if merged.empty:
                continue

            max_abs_diff = (merged['auroc_observed_multiclass'] - merged['auroc_observed_binary']).abs().max()
            rows.append({
                'phenotype': phenotype,
                'feature': feature,
                'max_abs_auroc_diff': max_abs_diff
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    output_path = FIGURES_DIR / 'one_vs_rest_vs_wt_diff_summary.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved diff summary: {output_path}")


def plot_binary_vs_ovr_feature_grid():
    """Plot a 3x2 grid: left=binary vs WT, right=multiclass OvR."""
    fig, axes = plt.subplots(len(FEATURES), 2, figsize=(14, 12), sharex='col', sharey='row')

    for row_idx, feature in enumerate(FEATURES):
        ax_bin = axes[row_idx, 0]
        ax_ovr = axes[row_idx, 1]

        for label in LEFT_LABELS:
            color = PHENOTYPE_COLORS.get(label, '#000000')

            binary_df = load_binary_df(label, feature)
            if binary_df is not None:
                plot_auroc_with_null(
                    ax=ax_bin,
                    auroc_df=binary_df,
                    color=color,
                    label=label,
                    style='-',
                    time_col='time_bin_center',
                    show_null_band=True,
                    show_significance=True,
                    sig_05_marker_size=150
                )

        for label in RIGHT_LABELS:
            color = PHENOTYPE_COLORS.get(label, '#000000')

            multiclass_df = load_multiclass_df(label, feature)
            if multiclass_df is not None:
                plot_auroc_with_null(
                    ax=ax_ovr,
                    auroc_df=multiclass_df,
                    color=color,
                    label=label,
                    style='-',
                    time_col='time_bin_center',
                    show_null_band=True,
                    show_significance=True,
                    sig_05_marker_size=150
                )

        for ax in (ax_bin, ax_ovr):
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_ylim(0.3, 1.0)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax_bin.set_ylabel(f'{feature.capitalize()}\nAUROC', fontsize=11)

        if row_idx == len(FEATURES) - 1:
            ax_bin.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
            ax_ovr.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)

        if row_idx == 0:
            ax_bin.set_title('Binary (vs WT)', fontsize=12)
            ax_ovr.set_title('Multiclass (OvR vs Rest)', fontsize=12)
            ax_ovr.scatter([], [], s=150, facecolors='none', edgecolors='black',
                           linewidths=2.5, label='p <= 0.01')
            ax_ovr.legend(loc='upper left', fontsize=9)

    plt.suptitle(
        'CEP290: Binary vs OvR by Feature (Phenotypes)',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout()

    save_path = FIGURES_DIR / 'binary_vs_ovr_feature_grid.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    print("CEP290 One-vs-WT vs One-vs-Rest Plotting")
    print(f"Multiclass config: {MULTICLASS_CONFIG}")

    for phenotype in PHENOTYPES:
        plot_phenotype_panel(phenotype)

    write_diff_summary()
    plot_binary_vs_ovr_feature_grid()
