"""
AUROC visualization functions.

This module provides plotting functions for visualizing classification
performance over time.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_auroc_over_time(
    df_auc: pd.DataFrame,
    group1: str,
    group2: str,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot observed AUROC vs null distribution over time.

    Parameters
    ----------
    df_auc : pd.DataFrame
        Output from predictive_signal_test.
        Must have columns: time_bin, AUROC_obs, AUROC_null_mean, AUROC_null_std
    group1, group2 : str
        Names of the two groups being compared.
    output_path : str or None
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df_auc["time_bin"], df_auc["AUROC_obs"],
            label="Observed AUROC", color="black", linewidth=2, marker='o')

    ax.fill_between(
        df_auc["time_bin"],
        df_auc["AUROC_null_mean"] - df_auc["AUROC_null_std"],
        df_auc["AUROC_null_mean"] + df_auc["AUROC_null_std"],
        color="gray", alpha=0.3, label="Null ± 1σ"
    )

    ax.axhline(0.5, color="gray", ls="--", linewidth=1.5, alpha=0.7,
               label="Chance level")
    ax.set_xlabel("Predicted stage (hpf bin)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(f"Predictive Signal Test: {group1} vs {group2}",
                fontsize=14, fontweight='bold')
    ax.set_ylim([0.4, 1.0])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_auroc_with_significance(
    df_auc: pd.DataFrame,
    group1: str,
    group2: str,
    alpha: float = 0.05,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot AUROC with significance markers.

    Parameters
    ----------
    df_auc : pd.DataFrame
        Output from predictive_signal_test.
        Must have columns: time_bin, AUROC_obs, AUROC_null_mean,
        AUROC_null_std, pval
    group1, group2 : str
        Names of the two groups being compared.
    alpha : float, default=0.05
        Significance threshold.
    output_path : str or None
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: AUROC over time
    ax1 = axes[0]
    ax1.plot(df_auc["time_bin"], df_auc["AUROC_obs"],
            label="Observed AUROC", color="black", linewidth=2, marker='o')
    ax1.fill_between(
        df_auc["time_bin"],
        df_auc["AUROC_null_mean"] - df_auc["AUROC_null_std"],
        df_auc["AUROC_null_mean"] + df_auc["AUROC_null_std"],
        color="gray", alpha=0.3, label="Null ± 1σ"
    )
    ax1.axhline(0.5, color="gray", ls="--", linewidth=1.5, alpha=0.7)

    # Mark significant bins
    sig_bins = df_auc[df_auc["pval"] < alpha]
    if len(sig_bins) > 0:
        ax1.scatter(sig_bins["time_bin"], sig_bins["AUROC_obs"],
                   color='red', s=150, marker='*', zorder=5,
                   label=f'Significant (p < {alpha})',
                   edgecolors='black', linewidth=1)

    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title(f"Predictive Signal Test: {group1} vs {group2}",
                 fontsize=14, fontweight='bold')
    ax1.set_ylim([0.4, 1.0])
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel 2: P-values over time
    ax2 = axes[1]
    ax2.plot(df_auc["time_bin"], df_auc["pval"],
            'o-', color='steelblue', linewidth=2, label='P-value')
    ax2.axhline(alpha, color='red', ls='--', linewidth=2, label=f'α = {alpha}')
    ax2.set_xlabel("Predicted stage (hpf bin)", fontsize=12)
    ax2.set_ylabel("P-value", fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def summarize_significant_bins(
    df_auc: pd.DataFrame,
    alpha: float = 0.05
) -> dict:
    """
    Summarize significant time bins from AUROC results.

    Parameters
    ----------
    df_auc : pd.DataFrame
        Output from predictive_signal_test
    alpha : float, default=0.05
        Significance threshold

    Returns
    -------
    dict
        Summary including first onset time, number of significant bins, etc.
    """
    sig_bins = df_auc[df_auc["pval"] < alpha].sort_values("time_bin")

    if len(sig_bins) == 0:
        return {
            'has_significant_signal': False,
            'n_significant_bins': 0,
            'first_onset_time': None,
            'first_onset_auroc': None,
            'first_onset_pval': None
        }

    first_sig = sig_bins.iloc[0]

    return {
        'has_significant_signal': True,
        'n_significant_bins': len(sig_bins),
        'first_onset_time': first_sig['time_bin'],
        'first_onset_auroc': first_sig['AUROC_obs'],
        'first_onset_pval': first_sig['pval'],
        'all_significant_times': sig_bins['time_bin'].tolist()
    }
