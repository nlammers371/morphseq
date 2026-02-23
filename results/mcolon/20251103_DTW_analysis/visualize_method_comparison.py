#!/usr/bin/env python3
"""
Generate comparison visualizations for clustering methods.

Creates plots similar to membership_vs_k.png showing how different
clustering methods compare across k values.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR, K_VALUES
import importlib.util

def load_module(name, filepath):
    """Load a module from a file path (handles hyphens in filenames)."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

io_module = load_module("io_module", "io-module.py")
load_data = io_module.load_data
save_plot = io_module.save_plot

import warnings
warnings.filterwarnings('ignore')


def generate_comparison_plots(all_results, OUTPUT_DIR, verbose=True):
    """
    Generate comprehensive comparison visualizations for clustering methods.

    Generates:
    1. Method agreement heatmap (ARI matrix) for each k
    2. Performance comparison across k values (silhouette, core%, bootstrap stability)
    3. Core membership distribution by method and k
    4. Bootstrap stability comparison (box plots)
    """

    methods = all_results[K_VALUES[0]]['methods']
    n_methods = len(methods)

    # ===== 1. METHOD AGREEMENT HEATMAPS (for each k) =====
    if verbose:
        print(f"\nGenerating method agreement heatmaps...")

    for k in K_VALUES:
        ari_matrix = all_results[k]['ari_matrix']

        fig, ax = plt.subplots(figsize=(8, 7), dpi=200)

        # Create heatmap
        im = ax.imshow(ari_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

        # Labels
        method_labels = [m.replace('_', '\n') for m in methods]
        ax.set_xticks(np.arange(n_methods))
        ax.set_yticks(np.arange(n_methods))
        ax.set_xticklabels(method_labels, fontsize=10)
        ax.set_yticklabels(method_labels, fontsize=10)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add values to cells
        for i in range(n_methods):
            for j in range(n_methods):
                text = ax.text(j, i, f'{ari_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=11, fontweight='bold')

        ax.set_title(f'Method Agreement (ARI) - k={k}', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax, label='ARI Score')
        plt.tight_layout()

        # Save
        save_plot(7, f'method_agreement_k{k}', fig, OUTPUT_DIR)
        plt.close(fig)

        if verbose:
            print(f"  ✓ method_agreement_k{k}.png")

    # ===== 2. AGGREGATE COMPARISON VS K (like membership_vs_k.png) =====
    if verbose:
        print(f"\nGenerating aggregate comparison across k values...")

    # Extract metrics for each method and k
    metrics_data = {
        'k': [],
        'method': [],
        'silhouette': [],
        'core_pct': [],
        'bootstrap_ari': []
    }

    for k in K_VALUES:
        for method in methods:
            n_core = all_results[k][method]['n_core']
            n_total = (n_core + all_results[k][method]['n_uncertain'] +
                      all_results[k][method]['n_outlier'])

            metrics_data['k'].append(k)
            metrics_data['method'].append(method)
            metrics_data['silhouette'].append(all_results[k][method]['silhouette'])
            metrics_data['core_pct'].append(100 * n_core / n_total if n_total > 0 else 0)
            metrics_data['bootstrap_ari'].append(all_results[k][method]['bootstrap_ari'])

    df_metrics = pd.DataFrame(metrics_data)

    # Create 3-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

    # Define colors for methods
    colors = {method: plt.cm.Set2(i) for i, method in enumerate(methods)}

    # Panel 1: Silhouette Score
    ax = axes[0]
    for method in methods:
        df_method = df_metrics[df_metrics['method'] == method]
        ax.plot(df_method['k'], df_method['silhouette'], marker='o', linewidth=2.5,
               markersize=8, label=method.replace('_', ' ').title(), color=colors[method])

    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Quality (Silhouette)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9, loc='best')

    # Panel 2: Core Membership %
    ax = axes[1]
    for method in methods:
        df_method = df_metrics[df_metrics['method'] == method]
        ax.plot(df_method['k'], df_method['core_pct'], marker='s', linewidth=2.5,
               markersize=8, label=method.replace('_', ' ').title(), color=colors[method])

    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Core Membership (%)', fontsize=12, fontweight='bold')
    ax.set_title('Membership Confidence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=9, loc='best')

    # Panel 3: Bootstrap Stability (mean ARI)
    ax = axes[2]
    for method in methods:
        df_method = df_metrics[df_metrics['method'] == method]
        ax.plot(df_method['k'], df_method['bootstrap_ari'], marker='^', linewidth=2.5,
               markersize=8, label=method.replace('_', ' ').title(), color=colors[method])

    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bootstrap Mean ARI', fontsize=12, fontweight='bold')
    ax.set_title('Clustering Stability', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9, loc='best')

    fig.suptitle('Clustering Method Comparison Across K Values', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_plot(7, 'method_comparison_vs_k', fig, OUTPUT_DIR)
    plt.close(fig)

    if verbose:
        print(f"  ✓ method_comparison_vs_k.png")

    # ===== 3. CORE MEMBERSHIP DISTRIBUTION =====
    if verbose:
        print(f"\nGenerating core membership distribution plots...")

    for k in K_VALUES:
        methods_list = all_results[k]['methods']
        n_methods = len(methods_list)

        core_counts = []
        uncertain_counts = []
        outlier_counts = []
        method_labels = []

        for method in methods_list:
            core_counts.append(all_results[k][method]['n_core'])
            uncertain_counts.append(all_results[k][method]['n_uncertain'])
            outlier_counts.append(all_results[k][method]['n_outlier'])
            method_labels.append(method.replace('_', '\n').title())

        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

        x = np.arange(n_methods)
        width = 0.6

        bars1 = ax.bar(x, core_counts, width, label='Core', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x, uncertain_counts, width, bottom=core_counts,
                       label='Uncertain', color='#f39c12', alpha=0.8)
        bars3 = ax.bar(x, outlier_counts, width,
                       bottom=np.array(core_counts) + np.array(uncertain_counts),
                       label='Outlier', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Number of Embryos', fontsize=11, fontweight='bold')
        ax.set_title(f'Membership Distribution by Method (k={k})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add count labels on bars
        for i, (core, unc, out) in enumerate(zip(core_counts, uncertain_counts, outlier_counts)):
            ax.text(i, core/2, str(core), ha='center', va='center', fontweight='bold', color='white')
            ax.text(i, core + unc/2, str(unc), ha='center', va='center', fontweight='bold', color='white')
            ax.text(i, core + unc + out/2, str(out), ha='center', va='center', fontweight='bold', color='white')

        plt.tight_layout()

        save_plot(7, f'core_membership_by_method_k{k}', fig, OUTPUT_DIR)
        plt.close(fig)

        if verbose:
            print(f"  ✓ core_membership_by_method_k{k}.png")

    # ===== 4. SUMMARY TABLE =====
    if verbose:
        print(f"\nGenerating summary table...")

    # Create summary table for each k
    for k in K_VALUES:
        methods_list = all_results[k]['methods']

        summary_data = []
        for method in methods_list:
            res = all_results[k][method]
            n_total = res['n_core'] + res['n_uncertain'] + res['n_outlier']
            summary_data.append({
                'Method': method.replace('_', ' ').title(),
                'Silhouette': f"{res['silhouette']:.4f}",
                'Core (%)': f"{100*res['n_core']/n_total:.1f}",
                'Bootstrap ARI': f"{res['bootstrap_ari']:.4f}",
                'Core (n)': res['n_core'],
                'Uncertain (n)': res['n_uncertain'],
                'Outlier (n)': res['n_outlier']
            })

        df_summary = pd.DataFrame(summary_data)

        # Create figure with table
        fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style header
        for i in range(len(df_summary.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(df_summary) + 1):
            for j in range(len(df_summary.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')

        fig.suptitle(f'Method Comparison Summary (k={k})', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_plot(7, f'method_comparison_summary_k{k}', fig, OUTPUT_DIR)
        plt.close(fig)

        if verbose:
            print(f"  ✓ method_comparison_summary_k{k}.png")

    if verbose:
        print(f"\n✓ All comparison visualizations generated!")


def main():
    """Generate all comparison visualizations."""
    print("\nLoading method comparison results...")

    try:
        all_results = load_data(7, 'method_comparison_all_k', OUTPUT_DIR)
    except FileNotFoundError:
        print("Error: method_comparison_all_k.pkl not found!")
        print("Please run compare_clustering_methods.py first.")
        return

    print("✓ Loaded results")

    generate_comparison_plots(all_results, OUTPUT_DIR, verbose=True)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
