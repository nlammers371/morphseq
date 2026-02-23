#!/usr/bin/env python
"""
Hybrid Threshold Method: Time-Binned (<48 hpf) + Global (≥48 hpf)

Tests the hybrid approach:
- EARLY (<48 hpf): Time-binned IQR with Gaussian smoothing (σ=1.5)
- LATE (≥48 hpf): Global IQR computed from all WT data ≥48 hpf

Key feature: Ensures bounds "click together" at transition point by
matching the last time-binned value to the global value at 48 hpf.

Author: Generated for CEP290 penetrance analysis
Date: 2026-01-03
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings

from config import (
    METRIC_NAME,
    TIME_COL,
    EMBRYO_COL,
    GENOTYPE_COL,
    CATEGORY_COL,
    IQR_K,
    TIME_BIN_WIDTH,
    WT_GENOTYPE,
    BROAD_CATEGORIES,
    CATEGORY_COLORS,
    GENOTYPE_COLORS,
    FIGURE_DIR as BASE_FIGURE_DIR,
    TABLE_DIR,
    DPI,
)

from data_loading import (
    load_trajectory_data,
    extract_wt_data,
    bin_data_by_time,
)

warnings.filterwarnings('ignore')

# Hybrid cutoff
HYBRID_CUTOFF_HPF = 48.0

# Create hybrid approach subfolder
FIGURE_DIR = BASE_FIGURE_DIR / "hybrid_approach"


# ============================================================================
# Hybrid Threshold Computation
# ============================================================================

def compute_wt_bounds_hybrid(
    wt_df,
    time_bins,
    cutoff_hpf=HYBRID_CUTOFF_HPF,
    metric=METRIC_NAME,
    k=IQR_K,
    smooth_sigma=1.5,
    bin_col='time_bin',
):
    """
    Compute hybrid WT bounds: time-binned for <cutoff, global for ≥cutoff.
    
    Ensures smooth transition at cutoff by using the global bounds directly
    without blending (clean "click").
    
    Parameters
    ----------
    wt_df : pd.DataFrame
        WT embryo data with time_bin column
    time_bins : np.ndarray
        Array of time bin centers
    cutoff_hpf : float
        Transition point from time-binned to global (default 48.0)
    metric : str
        Curvature metric column name
    k : float
        IQR multiplier (default 2.0)
    smooth_sigma : float
        Gaussian smoothing sigma for time_binned portion
    bin_col : str
        Name of time bin column
    
    Returns
    -------
    bounds_df : pd.DataFrame
        DataFrame with columns: [time_bin, low, high, median, n_samples, method]
    """
    results = []
    
    # Split bins
    early_bins = time_bins[time_bins < cutoff_hpf]
    late_bins = time_bins[time_bins >= cutoff_hpf]
    
    print(f"\nHybrid method (cutoff={cutoff_hpf} hpf, k={k}, smooth_sigma={smooth_sigma}):")
    print(f"  Early bins (<{cutoff_hpf} hpf): {len(early_bins)} bins - time-binned IQR")
    print(f"  Late bins (≥{cutoff_hpf} hpf): {len(late_bins)} bins - global IQR")
    
    # -------------------------------------------------------------------------
    # EARLY: Time-binned IQR with smoothing
    # -------------------------------------------------------------------------
    raw_low = []
    raw_high = []
    raw_median = []
    raw_n = []
    
    for time_bin in early_bins:
        bin_df = wt_df[wt_df[bin_col] == time_bin]
        
        if len(bin_df) < 3:
            raw_low.append(np.nan)
            raw_high.append(np.nan)
            raw_median.append(np.nan)
            raw_n.append(len(bin_df))
            continue
        
        values = bin_df[metric].dropna().values
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        raw_low.append(q1 - k * iqr)
        raw_high.append(q3 + k * iqr)
        raw_median.append(np.median(values))
        raw_n.append(len(values))
    
    # Convert to arrays
    raw_low = np.array(raw_low)
    raw_high = np.array(raw_high)
    raw_median = np.array(raw_median)
    raw_n = np.array(raw_n)
    
    # Interpolate NaNs
    valid_mask = ~np.isnan(raw_low)
    if valid_mask.sum() > 2:
        raw_low = np.interp(np.arange(len(raw_low)), 
                           np.where(valid_mask)[0], 
                           raw_low[valid_mask])
        raw_high = np.interp(np.arange(len(raw_high)), 
                            np.where(valid_mask)[0], 
                            raw_high[valid_mask])
        raw_median = np.interp(np.arange(len(raw_median)), 
                              np.where(valid_mask)[0], 
                              raw_median[valid_mask])
    
    # Apply Gaussian smoothing
    smoothed_low = gaussian_filter1d(raw_low, sigma=smooth_sigma)
    smoothed_high = gaussian_filter1d(raw_high, sigma=smooth_sigma)
    smoothed_median = gaussian_filter1d(raw_median, sigma=smooth_sigma)
    
    for i, time_bin in enumerate(early_bins):
        results.append({
            'time_bin': time_bin,
            'low': smoothed_low[i],
            'high': smoothed_high[i],
            'median': smoothed_median[i],
            'n_samples': raw_n[i],
            'method': 'time_binned',
        })
    
    print(f"  Early bounds range: [{smoothed_low.min():.6f}, {smoothed_high.max():.6f}]")
    
    # -------------------------------------------------------------------------
    # LATE: Global IQR from all WT data ≥ cutoff
    # -------------------------------------------------------------------------
    late_wt_df = wt_df[wt_df[bin_col] >= cutoff_hpf]
    late_values = late_wt_df[metric].dropna().values
    
    if len(late_values) > 0:
        q1_global = np.percentile(late_values, 25)
        q3_global = np.percentile(late_values, 75)
        iqr_global = q3_global - q1_global
        
        global_low = q1_global - k * iqr_global
        global_high = q3_global + k * iqr_global
        global_median = np.median(late_values)
        
        print(f"  Global IQR (≥{cutoff_hpf} hpf):")
        print(f"    Pooled {len(late_values):,} WT samples")
        print(f"    Bounds: [{global_low:.6f}, {global_high:.6f}]")
        
        # Report the "click" - difference at transition
        if len(early_bins) > 0:
            last_early_low = smoothed_low[-1]
            last_early_high = smoothed_high[-1]
            print(f"\n  TRANSITION at {cutoff_hpf} hpf:")
            print(f"    Last time-binned: [{last_early_low:.6f}, {last_early_high:.6f}]")
            print(f"    Global bounds:    [{global_low:.6f}, {global_high:.6f}]")
            print(f"    Jump (low):  {global_low - last_early_low:+.6f}")
            print(f"    Jump (high): {global_high - last_early_high:+.6f}")
        
        # Apply global bounds to all late bins
        for time_bin in late_bins:
            bin_df = wt_df[wt_df[bin_col] == time_bin]
            results.append({
                'time_bin': time_bin,
                'low': global_low,
                'high': global_high,
                'median': global_median,
                'n_samples': len(bin_df),
                'method': 'global',
            })
    
    bounds_df = pd.DataFrame(results)
    return bounds_df


def mark_penetrant(df, bounds_df, metric=METRIC_NAME, bin_col='time_bin'):
    """
    Mark frames as penetrant if outside bounds.
    """
    df = df.copy()
    bounds_dict = bounds_df.set_index('time_bin').to_dict('index')
    
    def is_penetrant(row):
        time_bin = row[bin_col]
        if time_bin not in bounds_dict:
            return np.nan
        bounds = bounds_dict[time_bin]
        value = row[metric]
        if pd.isna(value):
            return np.nan
        return 1 if (value < bounds['low'] or value > bounds['high']) else 0
    
    df['penetrant'] = df.apply(is_penetrant, axis=1)
    return df


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_hybrid_bounds(wt_df, bounds_hybrid, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Plot hybrid bounds with clear transition marker.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [2, 1]},
                             sharex=True)
    
    # Top: Bounds with WT scatter
    ax1 = axes[0]
    
    # WT scatter
    ax1.scatter(wt_df[TIME_COL], wt_df[METRIC_NAME],
               s=8, alpha=0.3, c='steelblue', label='WT data', zorder=1)
    
    # Early bounds (time-binned)
    early_bounds = bounds_hybrid[bounds_hybrid['method'] == 'time_binned']
    late_bounds = bounds_hybrid[bounds_hybrid['method'] == 'global']
    
    # Fill early region
    ax1.fill_between(
        early_bounds['time_bin'],
        early_bounds['low'],
        early_bounds['high'],
        alpha=0.3, color='blue', label='Time-binned bounds (<48 hpf)', zorder=2
    )
    ax1.plot(early_bounds['time_bin'], early_bounds['low'], 'b--', linewidth=1.5, alpha=0.7)
    ax1.plot(early_bounds['time_bin'], early_bounds['high'], 'b--', linewidth=1.5, alpha=0.7)
    ax1.plot(early_bounds['time_bin'], early_bounds['median'], 'b-', linewidth=2, alpha=0.8)
    
    # Fill late region
    ax1.fill_between(
        late_bounds['time_bin'],
        late_bounds['low'],
        late_bounds['high'],
        alpha=0.3, color='red', label='Global bounds (≥48 hpf)', zorder=2
    )
    ax1.plot(late_bounds['time_bin'], late_bounds['low'], 'r--', linewidth=1.5, alpha=0.7)
    ax1.plot(late_bounds['time_bin'], late_bounds['high'], 'r--', linewidth=1.5, alpha=0.7)
    ax1.plot(late_bounds['time_bin'], late_bounds['median'], 'r-', linewidth=2, alpha=0.8)
    
    # Transition line
    ax1.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2.5,
               label=f'Hybrid cutoff ({cutoff_hpf} hpf)', zorder=3)
    
    # Background shading
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    ax1.axvspan(xlim[0], cutoff_hpf, alpha=0.05, color='blue', zorder=0)
    ax1.axvspan(cutoff_hpf, xlim[1], alpha=0.05, color='red', zorder=0)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    ax1.set_ylabel(METRIC_NAME, fontsize=12)
    ax1.set_title(f'Hybrid WT Bounds: Time-Binned (σ=1.5) <{cutoff_hpf} hpf | Global ≥{cutoff_hpf} hpf\n(IQR ±{IQR_K})',
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Bottom: Sample counts
    ax2 = axes[1]
    
    n_samples = bounds_hybrid[['time_bin', 'n_samples', 'method']].copy()
    colors = []
    for _, row in n_samples.iterrows():
        if row['n_samples'] < 10:
            colors.append('orange')  # Sparse
        elif row['method'] == 'time_binned':
            colors.append('steelblue')
        else:
            colors.append('indianred')
    
    ax2.bar(n_samples['time_bin'], n_samples['n_samples'],
           width=TIME_BIN_WIDTH * 0.8, color=colors, alpha=0.7)
    ax2.axhline(y=10, color='orange', linestyle='--', linewidth=1.5,
               label='Sparse threshold (N<10)')
    ax2.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2.5)
    
    # Annotate late pooled count
    if len(late_bounds) > 0:
        total_late = bounds_hybrid[bounds_hybrid['method'] == 'global']['n_samples'].sum()
        ax2.annotate(f'Pooled: {total_late:,}',
                    xy=(cutoff_hpf + 10, ax2.get_ylim()[1] * 0.8),
                    fontsize=11, fontweight='bold', color='indianred')
    
    ax2.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax2.set_ylabel('# WT Samples per Bin', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_hybrid_penetrance(wt_marked, df_marked, time_bins, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Plot penetrance over time with hybrid method.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Helper
    def compute_penetrance_by_time(marked_df, group_col=None, group_val=None):
        if group_col and group_val:
            subset = marked_df[marked_df[group_col] == group_val]
        else:
            subset = marked_df
        
        results = []
        for time_bin in sorted(subset['time_bin'].unique()):
            bin_df = subset[subset['time_bin'] == time_bin].dropna(subset=['penetrant'])
            if len(bin_df) == 0:
                continue
            
            n_embryos = bin_df[EMBRYO_COL].nunique()
            penetrant_embryos = bin_df[bin_df['penetrant'] == 1][EMBRYO_COL].nunique()
            penetrance = penetrant_embryos / n_embryos if n_embryos > 0 else 0
            
            results.append({
                'time_bin': time_bin,
                'penetrance': penetrance * 100,
                'n_embryos': n_embryos,
            })
        return pd.DataFrame(results)
    
    # Left: WT penetrance
    ax1 = axes[0]
    wt_pen = compute_penetrance_by_time(wt_marked)
    
    # Color by method region
    early_pen = wt_pen[wt_pen['time_bin'] < cutoff_hpf]
    late_pen = wt_pen[wt_pen['time_bin'] >= cutoff_hpf]
    
    ax1.plot(early_pen['time_bin'], early_pen['penetrance'], 'o-',
            color='steelblue', linewidth=2, markersize=6, label='Time-binned region')
    ax1.plot(late_pen['time_bin'], late_pen['penetrance'], 'o-',
            color='indianred', linewidth=2, markersize=6, label='Global region')
    
    ax1.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Target: 5%')
    ax1.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Warning: 10%')
    ax1.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2, label=f'Cutoff: {cutoff_hpf} hpf')
    
    # Highlight high penetrance
    high_pen = wt_pen[wt_pen['penetrance'] > 10]
    if len(high_pen) > 0:
        ax1.scatter(high_pen['time_bin'], high_pen['penetrance'],
                   s=120, c='red', marker='X', zorder=5, label='High (>10%)')
    
    mean_pen = wt_pen['penetrance'].mean()
    ax1.set_title(f'WT Penetrance (Hybrid Method)\nMean: {mean_pen:.1f}%',
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax1.set_ylabel('WT Penetrance (%)', fontsize=12)
    ax1.set_ylim(0, max(50, wt_pen['penetrance'].max() * 1.1))
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Category penetrance
    ax2 = axes[1]
    for category in BROAD_CATEGORIES:
        cat_pen = compute_penetrance_by_time(df_marked, CATEGORY_COL, category)
        if len(cat_pen) == 0:
            continue
        color = CATEGORY_COLORS.get(category, 'gray')
        ax2.plot(cat_pen['time_bin'], cat_pen['penetrance'], 'o-',
                color=color, linewidth=2, markersize=4, label=category)
    
    ax2.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2)
    
    ax2.set_title('Penetrance by Trajectory Category (Hybrid Method)',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax2.set_ylabel('Penetrance (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_hybrid_scatter(df_marked, bounds_hybrid, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Scatter plot with penetrant markers using hybrid bounds.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot by genotype
    for genotype, color in GENOTYPE_COLORS.items():
        geno_df = df_marked[df_marked[GENOTYPE_COL] == genotype]
        
        # Non-penetrant
        non_pen = geno_df[geno_df['penetrant'] == 0]
        ax.scatter(non_pen[TIME_COL], non_pen[METRIC_NAME],
                  s=15, alpha=0.4, c=color, marker='o',
                  label=f'{genotype.split("_")[1]} (within)')
        
        # Penetrant
        pen = geno_df[geno_df['penetrant'] == 1]
        ax.scatter(pen[TIME_COL], pen[METRIC_NAME],
                  s=40, alpha=0.7, c=color, marker='X', edgecolor='black', linewidth=0.5,
                  label=f'{genotype.split("_")[1]} (outside)')
    
    # Early bounds
    early_bounds = bounds_hybrid[bounds_hybrid['method'] == 'time_binned']
    late_bounds = bounds_hybrid[bounds_hybrid['method'] == 'global']
    
    # Plot early bounds
    ax.plot(early_bounds['time_bin'], early_bounds['low'], 'b--', linewidth=2, alpha=0.8)
    ax.plot(early_bounds['time_bin'], early_bounds['high'], 'b--', linewidth=2, alpha=0.8)
    
    # Plot late bounds (horizontal lines)
    if len(late_bounds) > 0:
        global_low = late_bounds['low'].iloc[0]
        global_high = late_bounds['high'].iloc[0]
        late_times = late_bounds['time_bin'].values
        ax.hlines(y=global_low, xmin=late_times.min(), xmax=late_times.max(),
                 colors='red', linestyles='--', linewidth=2, alpha=0.8)
        ax.hlines(y=global_high, xmin=late_times.min(), xmax=late_times.max(),
                 colors='red', linestyles='--', linewidth=2, alpha=0.8)
    
    # Transition line
    ax.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2.5,
              label=f'Hybrid cutoff ({cutoff_hpf} hpf)')
    
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel(METRIC_NAME, fontsize=12)
    ax.set_title(f'All Genotypes with Hybrid Bounds\n(Time-binned <{cutoff_hpf} hpf | Global ≥{cutoff_hpf} hpf)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_genotype_separate(df_marked, bounds_hybrid, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Plot each genotype separately with hybrid bounds.
    Top row: Scatter with bounds. Bottom row: Penetrance curves.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), 
                             gridspec_kw={'height_ratios': [2, 1]})
    
    genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
    titles = ['Wildtype (WT)', 'Heterozygous (Het)', 'Homozygous (Homo)']
    
    # Helper for penetrance calculation
    def compute_penetrance_by_time(marked_df, genotype):
        subset = marked_df[marked_df[GENOTYPE_COL] == genotype]
        results = []
        for time_bin in sorted(subset['time_bin'].unique()):
            bin_df = subset[subset['time_bin'] == time_bin].dropna(subset=['penetrant'])
            if len(bin_df) == 0:
                continue
            
            n_embryos = bin_df[EMBRYO_COL].nunique()
            penetrant_embryos = bin_df[bin_df['penetrant'] == 1][EMBRYO_COL].nunique()
            penetrance = penetrant_embryos / n_embryos if n_embryos > 0 else 0
            
            results.append({
                'time_bin': time_bin,
                'penetrance': penetrance * 100,
                'n_embryos': n_embryos,
            })
        return pd.DataFrame(results)
    
    for col_idx, (genotype, title) in enumerate(zip(genotypes, titles)):
        # TOP ROW: Scatter with bounds
        ax_top = axes[0, col_idx]
        geno_df = df_marked[df_marked[GENOTYPE_COL] == genotype]
        
        if len(geno_df) == 0:
            ax_top.text(0.5, 0.5, f'No data for {genotype}', 
                       ha='center', va='center', transform=ax_top.transAxes)
            continue
        
        # Plot non-penetrant (within bounds)
        non_pen = geno_df[geno_df['penetrant'] == 0]
        ax_top.scatter(non_pen[TIME_COL], non_pen[METRIC_NAME],
                      s=20, alpha=0.4, c=GENOTYPE_COLORS.get(genotype, 'gray'), 
                      marker='o', label='Within bounds')
        
        # Plot penetrant (outside bounds)
        pen = geno_df[geno_df['penetrant'] == 1]
        ax_top.scatter(pen[TIME_COL], pen[METRIC_NAME],
                      s=50, alpha=0.8, c=GENOTYPE_COLORS.get(genotype, 'gray'), 
                      marker='X', edgecolor='black', linewidth=0.8,
                      label='Outside bounds')
        
        # Plot hybrid bounds
        early_bounds = bounds_hybrid[bounds_hybrid['method'] == 'time_binned']
        late_bounds = bounds_hybrid[bounds_hybrid['method'] == 'global']
        
        ax_top.fill_between(early_bounds['time_bin'], 
                           early_bounds['low'], 
                           early_bounds['high'],
                           alpha=0.15, color='blue', zorder=1)
        ax_top.plot(early_bounds['time_bin'], early_bounds['low'], 
                   'b--', linewidth=1.5, alpha=0.7)
        ax_top.plot(early_bounds['time_bin'], early_bounds['high'], 
                   'b--', linewidth=1.5, alpha=0.7)
        
        if len(late_bounds) > 0:
            ax_top.fill_between(late_bounds['time_bin'],
                               late_bounds['low'],
                               late_bounds['high'],
                               alpha=0.15, color='red', zorder=1)
            ax_top.plot(late_bounds['time_bin'], late_bounds['low'],
                       'r--', linewidth=1.5, alpha=0.7)
            ax_top.plot(late_bounds['time_bin'], late_bounds['high'],
                       'r--', linewidth=1.5, alpha=0.7)
        
        ax_top.axvline(x=cutoff_hpf, color='black', linestyle=':', 
                      linewidth=2, alpha=0.8)
        
        pen_embryos = geno_df[geno_df['penetrant'] == 1][EMBRYO_COL].nunique()
        total_embryos = geno_df[EMBRYO_COL].nunique()
        penetrance = pen_embryos / total_embryos * 100 if total_embryos > 0 else 0
        
        ax_top.set_title(f'{title}\n{total_embryos} embryos | Overall: {penetrance:.1f}%',
                        fontsize=13, fontweight='bold')
        if col_idx == 0:
            ax_top.set_ylabel(METRIC_NAME, fontsize=11)
        
        ax_top.legend(loc='upper right', fontsize=8)
        ax_top.grid(True, alpha=0.3)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        
        # BOTTOM ROW: Penetrance curve
        ax_bot = axes[1, col_idx]
        geno_pen = compute_penetrance_by_time(df_marked, genotype)
        
        if len(geno_pen) > 0:
            color = GENOTYPE_COLORS.get(genotype, 'gray')
            ax_bot.plot(geno_pen['time_bin'], geno_pen['penetrance'], 'o-',
                       color=color, linewidth=3, markersize=7)
        
        ax_bot.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2, alpha=0.8)
        
        ax_bot.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        if col_idx == 0:
            ax_bot.set_ylabel('Penetrance (%)', fontsize=11)
        ax_bot.set_ylim(0, 100)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.spines['top'].set_visible(False)
        ax_bot.spines['right'].set_visible(False)
    
    fig.suptitle(f'Hybrid Bounds Applied to Each Genotype\n(Time-binned <{cutoff_hpf} hpf | Global ≥{cutoff_hpf} hpf)',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_category_separate(df_marked, bounds_hybrid, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Plot each trajectory category separately with hybrid bounds.
    Top row: Scatter with bounds. Bottom row: Penetrance curves.
    """
    n_categories = len(BROAD_CATEGORIES)
    fig, axes = plt.subplots(2, n_categories, figsize=(7*n_categories, 12), 
                             gridspec_kw={'height_ratios': [2, 1]})
    
    # Helper for penetrance calculation
    def compute_penetrance_by_time(marked_df, category):
        subset = marked_df[marked_df[CATEGORY_COL] == category]
        results = []
        for time_bin in sorted(subset['time_bin'].unique()):
            bin_df = subset[subset['time_bin'] == time_bin].dropna(subset=['penetrant'])
            if len(bin_df) == 0:
                continue
            
            n_embryos = bin_df[EMBRYO_COL].nunique()
            penetrant_embryos = bin_df[bin_df['penetrant'] == 1][EMBRYO_COL].nunique()
            penetrance = penetrant_embryos / n_embryos if n_embryos > 0 else 0
            
            results.append({
                'time_bin': time_bin,
                'penetrance': penetrance * 100,
                'n_embryos': n_embryos,
            })
        return pd.DataFrame(results)
    
    for col_idx, category in enumerate(BROAD_CATEGORIES):
        # TOP ROW: Scatter with bounds
        ax_top = axes[0, col_idx]
        cat_df = df_marked[df_marked[CATEGORY_COL] == category]
        
        if len(cat_df) == 0:
            ax_top.text(0.5, 0.5, f'No data for {category}', 
                       ha='center', va='center', transform=ax_top.transAxes)
            continue
        
        # Plot non-penetrant (within bounds)
        non_pen = cat_df[cat_df['penetrant'] == 0]
        ax_top.scatter(non_pen[TIME_COL], non_pen[METRIC_NAME],
                      s=20, alpha=0.4, c=CATEGORY_COLORS.get(category, 'gray'), 
                      marker='o', label='Within bounds')
        
        # Plot penetrant (outside bounds)
        pen = cat_df[cat_df['penetrant'] == 1]
        ax_top.scatter(pen[TIME_COL], pen[METRIC_NAME],
                      s=50, alpha=0.8, c=CATEGORY_COLORS.get(category, 'gray'), 
                      marker='X', edgecolor='black', linewidth=0.8,
                      label='Outside bounds')
        
        # Plot hybrid bounds
        early_bounds = bounds_hybrid[bounds_hybrid['method'] == 'time_binned']
        late_bounds = bounds_hybrid[bounds_hybrid['method'] == 'global']
        
        ax_top.fill_between(early_bounds['time_bin'], 
                           early_bounds['low'], 
                           early_bounds['high'],
                           alpha=0.15, color='blue', zorder=1)
        ax_top.plot(early_bounds['time_bin'], early_bounds['low'], 
                   'b--', linewidth=1.5, alpha=0.7)
        ax_top.plot(early_bounds['time_bin'], early_bounds['high'], 
                   'b--', linewidth=1.5, alpha=0.7)
        
        if len(late_bounds) > 0:
            ax_top.fill_between(late_bounds['time_bin'],
                               late_bounds['low'],
                               late_bounds['high'],
                               alpha=0.15, color='red', zorder=1)
            ax_top.plot(late_bounds['time_bin'], late_bounds['low'],
                       'r--', linewidth=1.5, alpha=0.7)
            ax_top.plot(late_bounds['time_bin'], late_bounds['high'],
                       'r--', linewidth=1.5, alpha=0.7)
        
        ax_top.axvline(x=cutoff_hpf, color='black', linestyle=':', 
                      linewidth=2, alpha=0.8)
        
        pen_embryos = cat_df[cat_df['penetrant'] == 1][EMBRYO_COL].nunique()
        total_embryos = cat_df[EMBRYO_COL].nunique()
        penetrance = pen_embryos / total_embryos * 100 if total_embryos > 0 else 0
        
        ax_top.set_title(f'{category}\n{total_embryos} embryos | Overall: {penetrance:.1f}%',
                        fontsize=13, fontweight='bold')
        if col_idx == 0:
            ax_top.set_ylabel(METRIC_NAME, fontsize=11)
        
        ax_top.legend(loc='upper right', fontsize=8)
        ax_top.grid(True, alpha=0.3)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        
        # BOTTOM ROW: Penetrance curve
        ax_bot = axes[1, col_idx]
        cat_pen = compute_penetrance_by_time(df_marked, category)
        
        if len(cat_pen) > 0:
            color = CATEGORY_COLORS.get(category, 'gray')
            ax_bot.plot(cat_pen['time_bin'], cat_pen['penetrance'], 'o-',
                       color=color, linewidth=3, markersize=7)
        
        ax_bot.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2, alpha=0.8)
        
        ax_bot.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        if col_idx == 0:
            ax_bot.set_ylabel('Penetrance (%)', fontsize=11)
        ax_bot.set_ylim(0, 100)
        ax_bot.grid(True, alpha=0.3)
        ax_bot.spines['top'].set_visible(False)
        ax_bot.spines['right'].set_visible(False)
    
    fig.suptitle(f'Hybrid Bounds Applied to Each Trajectory Category\n(Time-binned <{cutoff_hpf} hpf | Global ≥{cutoff_hpf} hpf)',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig


def plot_penetrance_by_category(df_marked, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Plot penetrance over time for each trajectory category.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    def compute_penetrance_by_time(marked_df, category):
        subset = marked_df[marked_df[CATEGORY_COL] == category]
        results = []
        for time_bin in sorted(subset['time_bin'].unique()):
            bin_df = subset[subset['time_bin'] == time_bin].dropna(subset=['penetrant'])
            if len(bin_df) == 0:
                continue
            
            n_embryos = bin_df[EMBRYO_COL].nunique()
            penetrant_embryos = bin_df[bin_df['penetrant'] == 1][EMBRYO_COL].nunique()
            penetrance = penetrant_embryos / n_embryos if n_embryos > 0 else 0
            
            results.append({
                'time_bin': time_bin,
                'penetrance': penetrance * 100,
                'n_embryos': n_embryos,
            })
        return pd.DataFrame(results)
    
    # Plot each category
    for category in BROAD_CATEGORIES:
        cat_pen = compute_penetrance_by_time(df_marked, category)
        if len(cat_pen) == 0:
            continue
        
        color = CATEGORY_COLORS.get(category, 'gray')
        ax.plot(cat_pen['time_bin'], cat_pen['penetrance'], 'o-',
               color=color, linewidth=2.5, markersize=6, label=category)
    
    # Add transition line
    ax.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2.5,
              label=f'Hybrid cutoff ({cutoff_hpf} hpf)', alpha=0.8)
    
    # Add background shading
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], cutoff_hpf, alpha=0.05, color='blue', zorder=0)
    ax.axvspan(cutoff_hpf, xlim[1], alpha=0.05, color='red', zorder=0)
    ax.set_xlim(xlim)
    
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('Penetrance (%)', fontsize=12)
    ax.set_title(f'Penetrance by Trajectory Category\n(Hybrid Method: Time-binned <{cutoff_hpf} hpf | Global ≥{cutoff_hpf} hpf)',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_penetrance_combined(wt_marked, df_marked, time_bins, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Combined figure: Category penetrance (top) + WT penetrance (bottom left) + Category penetrance curves (bottom right).
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
    
    # Helper
    def compute_penetrance_by_time(marked_df, group_col=None, group_val=None):
        if group_col and group_val:
            subset = marked_df[marked_df[group_col] == group_val]
        else:
            subset = marked_df
        
        results = []
        for time_bin in sorted(subset['time_bin'].unique()):
            bin_df = subset[subset['time_bin'] == time_bin].dropna(subset=['penetrant'])
            if len(bin_df) == 0:
                continue
            
            n_embryos = bin_df[EMBRYO_COL].nunique()
            penetrant_embryos = bin_df[bin_df['penetrant'] == 1][EMBRYO_COL].nunique()
            penetrance = penetrant_embryos / n_embryos if n_embryos > 0 else 0
            
            results.append({
                'time_bin': time_bin,
                'penetrance': penetrance * 100,
                'n_embryos': n_embryos,
            })
        return pd.DataFrame(results)
    
    # TOP: Category penetrance curves (LARGE)
    ax_top = fig.add_subplot(gs[0, :])
    for category in BROAD_CATEGORIES:
        cat_pen = compute_penetrance_by_time(df_marked, CATEGORY_COL, category)
        if len(cat_pen) == 0:
            continue
        color = CATEGORY_COLORS.get(category, 'gray')
        ax_top.plot(cat_pen['time_bin'], cat_pen['penetrance'], 'o-',
                   color=color, linewidth=3, markersize=7, label=category)
    
    ax_top.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2.5,
                  label=f'Hybrid cutoff ({cutoff_hpf} hpf)', alpha=0.8)
    
    xlim = ax_top.get_xlim()
    ax_top.axvspan(xlim[0], cutoff_hpf, alpha=0.05, color='blue', zorder=0)
    ax_top.axvspan(cutoff_hpf, xlim[1], alpha=0.05, color='red', zorder=0)
    ax_top.set_xlim(xlim)
    
    ax_top.set_xlabel('Hours Post Fertilization (hpf)', fontsize=13)
    ax_top.set_ylabel('Penetrance (%)', fontsize=13)
    ax_top.set_title('Penetrance by Trajectory Category (Hybrid Method)',
                    fontsize=15, fontweight='bold')
    ax_top.set_ylim(0, 100)
    ax_top.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=2)
    ax_top.grid(True, alpha=0.3)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    
    # BOTTOM LEFT: WT penetrance
    ax_bl = fig.add_subplot(gs[1, 0])
    wt_pen = compute_penetrance_by_time(wt_marked)
    
    early_pen = wt_pen[wt_pen['time_bin'] < cutoff_hpf]
    late_pen = wt_pen[wt_pen['time_bin'] >= cutoff_hpf]
    
    ax_bl.plot(early_pen['time_bin'], early_pen['penetrance'], 'o-',
              color='steelblue', linewidth=2.5, markersize=6, label='Time-binned region')
    ax_bl.plot(late_pen['time_bin'], late_pen['penetrance'], 'o-',
              color='indianred', linewidth=2.5, markersize=6, label='Global region')
    
    ax_bl.axhline(y=5, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Target: 5%')
    ax_bl.axhline(y=10, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Warning: 10%')
    ax_bl.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2, label=f'Cutoff: {cutoff_hpf} hpf')
    
    high_pen = wt_pen[wt_pen['penetrance'] > 10]
    if len(high_pen) > 0:
        ax_bl.scatter(high_pen['time_bin'], high_pen['penetrance'],
                     s=150, c='red', marker='X', zorder=5, edgecolor='black', linewidth=1)
    
    mean_pen = wt_pen['penetrance'].mean()
    ax_bl.set_title(f'WT Penetrance Quality Control\nMean: {mean_pen:.1f}%',
                   fontsize=13, fontweight='bold')
    ax_bl.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax_bl.set_ylabel('WT Penetrance (%)', fontsize=12)
    ax_bl.set_ylim(0, max(50, wt_pen['penetrance'].max() * 1.1))
    ax_bl.legend(loc='upper right', fontsize=9)
    ax_bl.grid(True, alpha=0.3)
    ax_bl.spines['top'].set_visible(False)
    ax_bl.spines['right'].set_visible(False)
    
    # BOTTOM RIGHT: All categories summary
    ax_br = fig.add_subplot(gs[1, 1])
    
    # Bar chart of mean penetrance per category
    cat_means = []
    cat_labels = []
    cat_colors = []
    for category in BROAD_CATEGORIES:
        cat_pen = compute_penetrance_by_time(df_marked, CATEGORY_COL, category)
        if len(cat_pen) > 0:
            mean_val = cat_pen['penetrance'].mean()
            cat_means.append(mean_val)
            cat_labels.append(category)
            cat_colors.append(CATEGORY_COLORS.get(category, 'gray'))
    
    bars = ax_br.barh(cat_labels, cat_means, color=cat_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, cat_means)):
        ax_br.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax_br.set_xlabel('Mean Penetrance (%)', fontsize=12)
    ax_br.set_title('Mean Penetrance by Category', fontsize=13, fontweight='bold')
    ax_br.set_xlim(0, 100)
    ax_br.grid(True, alpha=0.3, axis='x')
    ax_br.spines['top'].set_visible(False)
    ax_br.spines['right'].set_visible(False)
    
    return fig


def plot_genotype_combined(df_marked, bounds_hybrid, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Combined figure: Individual genotype panels (top) + summary statistics (bottom).
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.25)
    
    genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
    titles = ['Wildtype (WT)', 'Heterozygous (Het)', 'Homozygous (Homo)']
    
    # TOP ROW: Individual genotype scatter plots
    for col_idx, (genotype, title) in enumerate(zip(genotypes, titles)):
        ax = fig.add_subplot(gs[0, col_idx])
        geno_df = df_marked[df_marked[GENOTYPE_COL] == genotype]
        
        if len(geno_df) == 0:
            ax.text(0.5, 0.5, f'No data for {genotype}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot non-penetrant (within bounds)
        non_pen = geno_df[geno_df['penetrant'] == 0]
        ax.scatter(non_pen[TIME_COL], non_pen[METRIC_NAME],
                  s=20, alpha=0.4, c=GENOTYPE_COLORS.get(genotype, 'gray'), 
                  marker='o', label='Within bounds')
        
        # Plot penetrant (outside bounds)
        pen = geno_df[geno_df['penetrant'] == 1]
        ax.scatter(pen[TIME_COL], pen[METRIC_NAME],
                  s=50, alpha=0.8, c=GENOTYPE_COLORS.get(genotype, 'gray'), 
                  marker='X', edgecolor='black', linewidth=0.8,
                  label='Outside bounds')
        
        # Plot hybrid bounds
        early_bounds = bounds_hybrid[bounds_hybrid['method'] == 'time_binned']
        late_bounds = bounds_hybrid[bounds_hybrid['method'] == 'global']
        
        ax.fill_between(early_bounds['time_bin'], 
                       early_bounds['low'], 
                       early_bounds['high'],
                       alpha=0.15, color='blue', zorder=1)
        ax.plot(early_bounds['time_bin'], early_bounds['low'], 
               'b--', linewidth=1.5, alpha=0.7)
        ax.plot(early_bounds['time_bin'], early_bounds['high'], 
               'b--', linewidth=1.5, alpha=0.7)
        
        if len(late_bounds) > 0:
            ax.fill_between(late_bounds['time_bin'],
                           late_bounds['low'],
                           late_bounds['high'],
                           alpha=0.15, color='red', zorder=1)
            ax.plot(late_bounds['time_bin'], late_bounds['low'],
                   'r--', linewidth=1.5, alpha=0.7)
            ax.plot(late_bounds['time_bin'], late_bounds['high'],
                   'r--', linewidth=1.5, alpha=0.7)
        
        ax.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2, alpha=0.8)
        
        pen_embryos = geno_df[geno_df['penetrant'] == 1][EMBRYO_COL].nunique()
        total_embryos = geno_df[EMBRYO_COL].nunique()
        penetrance = pen_embryos / total_embryos * 100 if total_embryos > 0 else 0
        
        ax.set_title(f'{title}\n{total_embryos} embryos | Penetrance: {penetrance:.1f}%',
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(METRIC_NAME, fontsize=11)
        
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # BOTTOM ROW: Summary statistics
    # Left: Penetrance comparison
    ax_bl = fig.add_subplot(gs[1, 0])
    geno_pens = []
    geno_labels = []
    geno_colors = []
    
    for genotype in genotypes:
        geno_df = df_marked[df_marked[GENOTYPE_COL] == genotype]
        if len(geno_df) > 0:
            pen_embryos = geno_df[geno_df['penetrant'] == 1][EMBRYO_COL].nunique()
            total_embryos = geno_df[EMBRYO_COL].nunique()
            penetrance = pen_embryos / total_embryos * 100 if total_embryos > 0 else 0
            geno_pens.append(penetrance)
            geno_labels.append(genotype.split('_')[1])
            geno_colors.append(GENOTYPE_COLORS.get(genotype, 'gray'))
    
    bars = ax_bl.bar(geno_labels, geno_pens, color=geno_colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, geno_pens):
        ax_bl.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%',
                  ha='center', fontsize=12, fontweight='bold')
    
    ax_bl.set_ylabel('Penetrance (%)', fontsize=12)
    ax_bl.set_title('Penetrance by Genotype', fontsize=13, fontweight='bold')
    ax_bl.set_ylim(0, 100)
    ax_bl.grid(True, alpha=0.3, axis='y')
    ax_bl.spines['top'].set_visible(False)
    ax_bl.spines['right'].set_visible(False)
    
    # Middle: Sample sizes
    ax_bm = fig.add_subplot(gs[1, 1])
    geno_ns = []
    for genotype in genotypes:
        geno_df = df_marked[df_marked[GENOTYPE_COL] == genotype]
        geno_ns.append(geno_df[EMBRYO_COL].nunique())
    
    bars = ax_bm.bar(geno_labels, geno_ns, color=geno_colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, geno_ns):
        ax_bm.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val}',
                  ha='center', fontsize=12, fontweight='bold')
    
    ax_bm.set_ylabel('# Embryos', fontsize=12)
    ax_bm.set_title('Sample Size by Genotype', fontsize=13, fontweight='bold')
    ax_bm.grid(True, alpha=0.3, axis='y')
    ax_bm.spines['top'].set_visible(False)
    ax_bm.spines['right'].set_visible(False)
    
    # Right: Penetrant embryo counts
    ax_br = fig.add_subplot(gs[1, 2])
    geno_pen_counts = []
    for genotype in genotypes:
        geno_df = df_marked[df_marked[GENOTYPE_COL] == genotype]
        if len(geno_df) > 0:
            pen_embryos = geno_df[geno_df['penetrant'] == 1][EMBRYO_COL].nunique()
            geno_pen_counts.append(pen_embryos)
        else:
            geno_pen_counts.append(0)
    
    bars = ax_br.bar(geno_labels, geno_pen_counts, color=geno_colors, alpha=0.8, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, geno_pen_counts):
        ax_br.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val}',
                  ha='center', fontsize=12, fontweight='bold')
    
    ax_br.set_ylabel('# Penetrant Embryos', fontsize=12)
    ax_br.set_title('Penetrant Embryos by Genotype', fontsize=13, fontweight='bold')
    ax_br.grid(True, alpha=0.3, axis='y')
    ax_br.spines['top'].set_visible(False)
    ax_br.spines['right'].set_visible(False)
    
    fig.suptitle(f'Genotype Analysis (Hybrid Method: Time-binned <{cutoff_hpf} hpf | Global ≥{cutoff_hpf} hpf)',
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig


def plot_penetrance_status_trajectories(df_marked, bounds_hybrid, cutoff_hpf=HYBRID_CUTOFF_HPF):
    """
    Plot connected trajectories colored by penetrance status.
    Bold line segments when embryo is NOT penetrant.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Plot bounds first
    early_bounds = bounds_hybrid[bounds_hybrid['method'] == 'time_binned']
    late_bounds = bounds_hybrid[bounds_hybrid['method'] == 'global']
    
    # Early bounds
    ax.fill_between(early_bounds['time_bin'], 
                   early_bounds['low'], 
                   early_bounds['high'],
                   alpha=0.1, color='blue', zorder=1)
    ax.plot(early_bounds['time_bin'], early_bounds['low'], 
           'b--', linewidth=1.5, alpha=0.5)
    ax.plot(early_bounds['time_bin'], early_bounds['high'], 
           'b--', linewidth=1.5, alpha=0.5)
    
    # Late bounds
    if len(late_bounds) > 0:
        ax.fill_between(late_bounds['time_bin'],
                       late_bounds['low'],
                       late_bounds['high'],
                       alpha=0.1, color='red', zorder=1)
        ax.plot(late_bounds['time_bin'], late_bounds['low'],
               'r--', linewidth=1.5, alpha=0.5)
        ax.plot(late_bounds['time_bin'], late_bounds['high'],
               'r--', linewidth=1.5, alpha=0.5)
    
    # Transition line
    ax.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=2, alpha=0.5)
    
    # Plot trajectories by category
    for category in BROAD_CATEGORIES:
        cat_df = df_marked[df_marked[CATEGORY_COL] == category].copy()
        if len(cat_df) == 0:
            continue
        
        color = CATEGORY_COLORS.get(category, 'gray')
        
        # Plot each embryo's trajectory
        for embryo_id in cat_df[EMBRYO_COL].unique()[:50]:  # Limit to 50 per category for visibility
            embryo_data = cat_df[cat_df[EMBRYO_COL] == embryo_id].sort_values(TIME_COL)
            
            if len(embryo_data) < 2:
                continue
            
            # Plot segments between consecutive points
            for i in range(len(embryo_data) - 1):
                row1 = embryo_data.iloc[i]
                row2 = embryo_data.iloc[i + 1]
                
                x = [row1[TIME_COL], row2[TIME_COL]]
                y = [row1[METRIC_NAME], row2[METRIC_NAME]]
                
                # Check penetrance status
                is_penetrant = row1['penetrant'] == 1 or row2['penetrant'] == 1
                
                if is_penetrant:
                    # Thin line for penetrant
                    ax.plot(x, y, color=color, alpha=0.3, linewidth=0.8, zorder=2)
                else:
                    # Bold line for non-penetrant
                    ax.plot(x, y, color=color, alpha=0.7, linewidth=2.5, zorder=3)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2.5, alpha=0.7, label='Non-penetrant (within bounds)'),
        Line2D([0], [0], color='gray', linewidth=0.8, alpha=0.3, label='Penetrant (outside bounds)'),
    ]
    for category in BROAD_CATEGORIES:
        color = CATEGORY_COLORS.get(category, 'gray')
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=category))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel(METRIC_NAME, fontsize=12)
    ax.set_title(f'Trajectory Status Over Time (Bold = Non-Penetrant)\n(Hybrid Method: Time-binned <{cutoff_hpf} hpf | Global ≥{cutoff_hpf} hpf)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_transition_zoom(wt_df, bounds_hybrid, cutoff_hpf=HYBRID_CUTOFF_HPF, window=15):
    """
    Zoom into transition region to show the "click".
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Filter to transition window
    t_min = cutoff_hpf - window
    t_max = cutoff_hpf + window
    
    wt_zoom = wt_df[(wt_df[TIME_COL] >= t_min) & (wt_df[TIME_COL] <= t_max)]
    bounds_zoom = bounds_hybrid[(bounds_hybrid['time_bin'] >= t_min) & 
                                (bounds_hybrid['time_bin'] <= t_max)]
    
    # WT scatter
    ax.scatter(wt_zoom[TIME_COL], wt_zoom[METRIC_NAME],
              s=20, alpha=0.5, c='steelblue', label='WT data')
    
    # Early bounds
    early = bounds_zoom[bounds_zoom['method'] == 'time_binned']
    late = bounds_zoom[bounds_zoom['method'] == 'global']
    
    if len(early) > 0:
        ax.fill_between(early['time_bin'], early['low'], early['high'],
                       alpha=0.3, color='blue', label='Time-binned')
        ax.plot(early['time_bin'], early['low'], 'b-', linewidth=2)
        ax.plot(early['time_bin'], early['high'], 'b-', linewidth=2)
    
    if len(late) > 0:
        ax.fill_between(late['time_bin'], late['low'], late['high'],
                       alpha=0.3, color='red', label='Global')
        ax.plot(late['time_bin'], late['low'], 'r-', linewidth=2)
        ax.plot(late['time_bin'], late['high'], 'r-', linewidth=2)
    
    # Transition line with annotation
    ax.axvline(x=cutoff_hpf, color='black', linestyle=':', linewidth=3)
    
    # Annotate the jump
    if len(early) > 0 and len(late) > 0:
        last_early_high = early['high'].iloc[-1]
        first_late_high = late['high'].iloc[0]
        last_early_low = early['low'].iloc[-1]
        first_late_low = late['low'].iloc[0]
        
        # Arrow showing jump
        mid_y = (last_early_high + first_late_high) / 2
        ax.annotate('', xy=(cutoff_hpf + 0.5, first_late_high),
                   xytext=(cutoff_hpf - 0.5, last_early_high),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.annotate(f'Δhigh: {first_late_high - last_early_high:+.4f}',
                   xy=(cutoff_hpf + 1, mid_y), fontsize=10, fontweight='bold')
    
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel(METRIC_NAME, fontsize=12)
    ax.set_title(f'Transition Region ({t_min}-{t_max} hpf): "Click" at {cutoff_hpf} hpf',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


def compute_summary_stats(wt_marked, df_marked):
    """
    Compute summary statistics for hybrid method.
    """
    pen_embryos = wt_marked[wt_marked['penetrant'] == 1][EMBRYO_COL].nunique()
    total_embryos = wt_marked[EMBRYO_COL].nunique()
    wt_penetrance = pen_embryos / total_embryos * 100 if total_embryos > 0 else 0
    
    stats = {
        'WT Penetrance (%)': wt_penetrance,
        'WT Frames Outside': (wt_marked['penetrant'] == 1).sum(),
        'Total WT Frames': len(wt_marked),
        'WT Embryos': total_embryos,
        'WT Embryos Penetrant': pen_embryos,
    }
    
    return pd.Series(stats)


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("HYBRID THRESHOLD METHOD")
    print(f"Time-Binned (σ=1.5) <{HYBRID_CUTOFF_HPF} hpf | Global ≥{HYBRID_CUTOFF_HPF} hpf")
    print("="*70)
    
    # Ensure output directories exist
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_trajectory_data()
    wt_df = extract_wt_data(df)
    df, time_bins = bin_data_by_time(df)
    wt_df, _ = bin_data_by_time(wt_df)
    
    # Compute hybrid bounds
    print("\n" + "-"*50)
    bounds_hybrid = compute_wt_bounds_hybrid(
        wt_df, time_bins, 
        cutoff_hpf=HYBRID_CUTOFF_HPF,
        smooth_sigma=1.5
    )
    
    # Save bounds
    bounds_hybrid.to_csv(TABLE_DIR / "bounds_hybrid_48hpf.csv", index=False)
    print(f"\n✓ Saved bounds to {TABLE_DIR}/bounds_hybrid_48hpf.csv")
    
    # Mark penetrance
    print("\nMarking penetrance...")
    wt_marked = mark_penetrant(wt_df, bounds_hybrid)
    df_marked = mark_penetrant(df, bounds_hybrid)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    stats = compute_summary_stats(wt_marked, df_marked)
    print(stats.to_string())
    stats.to_csv(TABLE_DIR / "hybrid_48hpf_summary.csv")
    
    # Generate figures
    print("\n" + "-"*50)
    print("Generating figures...")
    
    # 1. Hybrid bounds diagnostic
    fig1 = plot_hybrid_bounds(wt_df, bounds_hybrid, HYBRID_CUTOFF_HPF)
    fig1.savefig(FIGURE_DIR / "hybrid_bounds_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig1)
    print(f"  ✓ hybrid_bounds_48hpf.png")
    
    # 2. Penetrance curves
    fig2 = plot_hybrid_penetrance(wt_marked, df_marked, time_bins, HYBRID_CUTOFF_HPF)
    fig2.savefig(FIGURE_DIR / "hybrid_penetrance_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ hybrid_penetrance_48hpf.png")
    
    # 3. Scatter with bounds
    fig3 = plot_hybrid_scatter(df_marked, bounds_hybrid, HYBRID_CUTOFF_HPF)
    fig3.savefig(FIGURE_DIR / "hybrid_scatter_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig3)
    print(f"  ✓ hybrid_scatter_48hpf.png")
    
    # 4. Separate genotype plots
    fig4 = plot_genotype_separate(df_marked, bounds_hybrid, HYBRID_CUTOFF_HPF)
    fig4.savefig(FIGURE_DIR / "hybrid_genotype_separate_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig4)
    print(f"  ✓ hybrid_genotype_separate_48hpf.png")
    
    # 5. Penetrance by category
    fig5 = plot_penetrance_by_category(df_marked, HYBRID_CUTOFF_HPF)
    fig5.savefig(FIGURE_DIR / "hybrid_penetrance_by_category_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig5)
    print(f"  ✓ hybrid_penetrance_by_category_48hpf.png")
    
    # 6. Separate category plots
    fig6 = plot_category_separate(df_marked, bounds_hybrid, HYBRID_CUTOFF_HPF)
    fig6.savefig(FIGURE_DIR / "hybrid_category_separate_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig6)
    print(f"  ✓ hybrid_category_separate_48hpf.png")
    
    # 7. COMBINED: Penetrance analysis (category curves + WT QC + summary)
    fig7 = plot_penetrance_combined(wt_marked, df_marked, time_bins, HYBRID_CUTOFF_HPF)
    fig7.savefig(FIGURE_DIR / "hybrid_penetrance_combined_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig7)
    print(f"  ✓ hybrid_penetrance_combined_48hpf.png")
    
    # 8. COMBINED: Genotype analysis (individual panels + summary stats)
    fig8 = plot_genotype_combined(df_marked, bounds_hybrid, HYBRID_CUTOFF_HPF)
    fig8.savefig(FIGURE_DIR / "hybrid_genotype_combined_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig8)
    print(f"  ✓ hybrid_genotype_combined_48hpf.png")
    
    # 9. Penetrance status trajectories (bold when non-penetrant)
    fig9 = plot_penetrance_status_trajectories(df_marked, bounds_hybrid, HYBRID_CUTOFF_HPF)
    fig9.savefig(FIGURE_DIR / "hybrid_trajectory_status_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig9)
    print(f"  ✓ hybrid_trajectory_status_48hpf.png")
    
    # 10. Transition zoom
    fig10 = plot_transition_zoom(wt_df, bounds_hybrid, HYBRID_CUTOFF_HPF)
    fig10.savefig(FIGURE_DIR / "hybrid_transition_zoom_48hpf.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig10)
    print(f"  ✓ hybrid_transition_zoom_48hpf.png")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}/")
    print(f"  Tables: {TABLE_DIR}/")
    
    wt_pen = stats['WT Penetrance (%)']
    print(f"\n⭐ RESULT: WT Penetrance = {wt_pen:.1f}%")
    if wt_pen <= 5:
        print("   ✓ Within acceptable range (≤5%)")
    elif wt_pen <= 10:
        print("   ⚠️ Slightly elevated (5-10%) - may need adjustment")
    else:
        print("   ❌ Too high (>10%) - threshold method needs revision")


if __name__ == "__main__":
    main()
