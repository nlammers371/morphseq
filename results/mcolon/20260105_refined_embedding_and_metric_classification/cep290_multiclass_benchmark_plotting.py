#!/usr/bin/env python
"""CEP290 Multiclass Benchmark Plotting

Overlays multiclass (2-class OvR) vs original binary results for sanity checks.
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

BENCH_OUTPUT_DIR = Path(__file__).parent / "output" / "cep290_multiclass_benchmark"
ORIGINAL_OUTPUT_DIR = Path(__file__).parent / "output" / "cep290_phenotype"
FIGURES_DIR = BENCH_OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ['curvature', 'length', 'embedding']

FEATURE_COLORS = {
    'curvature': '#7E57C2',
    'length': '#43A047',
    'embedding': '#1E88E5',
}


COMPARISONS = {
    'LowToHigh_vs_WT': ('LowToHigh', 'WT'),
    'HighToLow_vs_WT': ('HighToLow', 'WT'),
    'Intermediate_vs_WT': ('Intermediate', 'WT'),
    'LowToHigh_vs_Het': ('LowToHigh', 'Het'),
    'HighToLow_vs_Het': ('HighToLow', 'Het'),
    'Intermediate_vs_Het': ('Intermediate', 'Het'),
    'Het_vs_WT': ('Het', 'WT'),
}


def load_binary_df(comparison_dir, feature):
    """Load original binary classification output if present."""
    csv_path = ORIGINAL_OUTPUT_DIR / comparison_dir / f'classification_{feature}.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return prepare_auroc_data(df, significance_threshold=0.01)


def load_multiclass_df(comparison_dir, feature, positive_label):
    """Load multiclass OvR for the positive class."""
    csv_path = BENCH_OUTPUT_DIR / comparison_dir / feature / f'ovr_auroc_{positive_label}.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None
    return prepare_auroc_data(df, significance_threshold=0.01)


def plot_comparison_panel(comp_name, pos_label, neg_label):
    """Plot a 1x3 panel comparing binary vs multiclass per feature."""
    comparison_dir = comp_name.lower()
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(18, 5))

    for ax, feature in zip(axes, FEATURES):
        binary_df = load_binary_df(comparison_dir, feature)
        multiclass_df = load_multiclass_df(comparison_dir, feature, pos_label)

        if multiclass_df is None:
            ax.text(0.5, 0.5, f"No multiclass data\n({feature})",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        color = FEATURE_COLORS.get(feature, '#000000')
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
                label='Binary'
            )
        else:
            ax.text(0.5, 0.2, "Binary not found",
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)

        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        ax.set_ylabel('AUROC', fontsize=11)
        ax.set_title(f'{feature.capitalize()}', fontsize=12)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if feature == 'curvature':
            ax.legend(loc='upper left', fontsize=9)

    plt.suptitle(f'CEP290 Benchmark: {pos_label} vs {neg_label}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / f'benchmark_{comparison_dir}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def write_diff_summary():
    """Write a CSV summary of max AUROC deltas per comparison/feature."""
    rows = []

    for comp_name, (pos_label, _) in COMPARISONS.items():
        comparison_dir = comp_name.lower()
        for feature in FEATURES:
            binary_df = load_binary_df(comparison_dir, feature)
            multiclass_df = load_multiclass_df(comparison_dir, feature, pos_label)
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
                'comparison': comp_name,
                'feature': feature,
                'max_abs_auroc_diff': max_abs_diff
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    output_path = FIGURES_DIR / 'benchmark_diff_summary.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved diff summary: {output_path}")


if __name__ == '__main__':
    print("CEP290 Multiclass Benchmark Plotting")

    for comp_name, (pos_label, neg_label) in COMPARISONS.items():
        plot_comparison_panel(comp_name, pos_label, neg_label)

    write_diff_summary()
