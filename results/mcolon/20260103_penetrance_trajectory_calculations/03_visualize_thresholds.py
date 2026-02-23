#!/usr/bin/env python
"""
Threshold Method Comparison: Time-Binned (Smoothed) vs Global IQR

Compares two approaches for defining WT penetrance bounds:
1. TIME-BINNED: IQR computed per 2-hour bin, Gaussian smoothed (sigma=1.5)
2. GLOBAL: Single IQR computed from all WT data

Outputs comparison figures to determine which method produces:
- Lower WT penetrance (should be ~5%)
- Cleaner separation between trajectory groups

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
    FIGURE_DIR,
    TABLE_DIR,
    DPI,
    FIGSIZE_DIAGNOSTIC,
)

from data_loading import (
    load_trajectory_data,
    extract_wt_data,
    bin_data_by_time,
)

warnings.filterwarnings('ignore')


# ============================================================================
# Unified Threshold Computation
# ============================================================================

def compute_wt_bounds(
    wt_df,
    time_bins,
    method='time_binned',
    metric=METRIC_NAME,
    k=IQR_K,
    smooth_sigma=1.5,
    bin_col='time_bin',
):
    """
    Compute WT bounds using specified method.
    
    Parameters
    ----------
    wt_df : pd.DataFrame
        WT embryo data with time_bin column
    time_bins : np.ndarray
        Array of time bin centers
    method : str
        'time_binned' (with Gaussian smoothing) or 'global'
    metric : str
        Curvature metric column name
    k : float
        IQR multiplier (default 2.0 for ~95% coverage)
    smooth_sigma : float
        Gaussian smoothing sigma for time_binned method (default 1.5)
    bin_col : str
        Name of time bin column
    
    Returns
    -------
    bounds_df : pd.DataFrame
        DataFrame with columns: [time_bin, low, high, median, n_samples, method]
        Easy to use for penetrance calculation and plotting
    """
    results = []
    
    if method == 'global':
        # Single IQR from all WT data
        all_values = wt_df[metric].dropna().values
        
        q1 = np.percentile(all_values, 25)
        q3 = np.percentile(all_values, 75)
        iqr = q3 - q1
        
        low = q1 - k * iqr
        high = q3 + k * iqr
        median = np.median(all_values)
        n_total = len(all_values)
        
        print(f"\nGlobal IQR method (k={k}):")
        print(f"  Pooled {n_total:,} WT samples")
        print(f"  Bounds: [{low:.6f}, {high:.6f}]")
        
        # Apply same bounds to all time bins
        for time_bin in time_bins:
            bin_df = wt_df[wt_df[bin_col] == time_bin]
            results.append({
                'time_bin': time_bin,
                'low': low,
                'high': high,
                'median': median,
                'n_samples': len(bin_df),  # Per-bin sample count for diagnostics
                'method': 'global',
            })
    
    elif method == 'time_binned':
        # Per-bin IQR with Gaussian smoothing
        raw_low = []
        raw_high = []
        raw_median = []
        raw_n = []
        valid_bins = []
        
        for time_bin in time_bins:
            bin_df = wt_df[wt_df[bin_col] == time_bin]
            
            if len(bin_df) < 3:  # Need minimum samples
                raw_low.append(np.nan)
                raw_high.append(np.nan)
                raw_median.append(np.nan)
                raw_n.append(len(bin_df))
                valid_bins.append(time_bin)
                continue
            
            values = bin_df[metric].dropna().values
            
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            raw_low.append(q1 - k * iqr)
            raw_high.append(q3 + k * iqr)
            raw_median.append(np.median(values))
            raw_n.append(len(values))
            valid_bins.append(time_bin)
        
        # Convert to arrays
        raw_low = np.array(raw_low)
        raw_high = np.array(raw_high)
        raw_median = np.array(raw_median)
        raw_n = np.array(raw_n)
        
        # Interpolate NaNs before smoothing
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
        
        print(f"\nTime-binned IQR method (k={k}, smooth_sigma={smooth_sigma}):")
        print(f"  {len(valid_bins)} time bins")
        print(f"  Bounds range: [{smoothed_low.min():.6f}, {smoothed_high.max():.6f}]")
        
        for i, time_bin in enumerate(valid_bins):
            results.append({
                'time_bin': time_bin,
                'low': smoothed_low[i],
                'high': smoothed_high[i],
                'median': smoothed_median[i],
                'n_samples': raw_n[i],
                'method': 'time_binned',
            })
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'time_binned' or 'global'")
    
    bounds_df = pd.DataFrame(results)
    return bounds_df


def mark_penetrant(df, bounds_df, metric=METRIC_NAME, bin_col='time_bin'):
    """
    Mark frames as penetrant if outside bounds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to mark
    bounds_df : pd.DataFrame
        Bounds from compute_wt_bounds()
    metric : str
        Metric column name
    bin_col : str
        Time bin column name
    
    Returns
    -------
    df : pd.DataFrame
        Copy with 'penetrant' column added (1 = penetrant, 0 = not)
    """
    df = df.copy()
    
    # Create lookup dict
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

def plot_method_comparison(wt_df, bounds_time_binned, bounds_global, time_bins):
    """
    Side-by-side comparison of threshold methods overlaid on WT data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # Scatter WT data (same for both panels)
    for ax, bounds_df, title in zip(
        axes, 
        [bounds_time_binned, bounds_global],
        ['Time-Binned IQR (Gaussian σ=1.5)', 'Global IQR']
    ):
        # Scatter WT points
        ax.scatter(
            wt_df[TIME_COL], 
            wt_df[METRIC_NAME],
            s=8, alpha=0.3, c='steelblue', label='WT data'
        )
        
        # Plot bounds
        ax.fill_between(
            bounds_df['time_bin'],
            bounds_df['low'],
            bounds_df['high'],
            alpha=0.3, color='red', label=f'IQR ±{IQR_K} bounds'
        )
        ax.plot(bounds_df['time_bin'], bounds_df['low'], 'r--', linewidth=1.5, alpha=0.7)
        ax.plot(bounds_df['time_bin'], bounds_df['high'], 'r--', linewidth=1.5, alpha=0.7)
        ax.plot(bounds_df['time_bin'], bounds_df['median'], 'g-', linewidth=2, alpha=0.7, label='Median')
        
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].set_ylabel(METRIC_NAME, fontsize=12)
    
    fig.suptitle('WT Threshold Method Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_penetrance_by_method(wt_df_tb, wt_df_global, df_tb, df_global, time_bins):
    """
    Compare penetrance rates between methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Helper to compute penetrance by time
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
    
    # Top row: WT penetrance comparison
    for ax, wt_marked, method_name in zip(
        axes[0], 
        [wt_df_tb, wt_df_global], 
        ['Time-Binned', 'Global']
    ):
        wt_pen = compute_penetrance_by_time(wt_marked)
        
        ax.plot(wt_pen['time_bin'], wt_pen['penetrance'], 'o-', 
               color='steelblue', linewidth=2, markersize=5)
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Target: 5%')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Warning: 10%')
        
        # Highlight high penetrance
        high_pen = wt_pen[wt_pen['penetrance'] > 10]
        if len(high_pen) > 0:
            ax.scatter(high_pen['time_bin'], high_pen['penetrance'], 
                      s=100, c='red', marker='X', zorder=5, label='High (>10%)')
        
        mean_pen = wt_pen['penetrance'].mean()
        ax.set_title(f'{method_name}: WT Penetrance\n(Mean: {mean_pen:.1f}%)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Hours Post Fertilization', fontsize=11)
        ax.set_ylabel('WT Penetrance (%)', fontsize=11)
        ax.set_ylim(0, 50)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Bottom row: Category penetrance comparison
    for ax, df_marked, method_name in zip(
        axes[1], 
        [df_tb, df_global], 
        ['Time-Binned', 'Global']
    ):
        for category in BROAD_CATEGORIES:
            cat_pen = compute_penetrance_by_time(df_marked, CATEGORY_COL, category)
            if len(cat_pen) == 0:
                continue
            color = CATEGORY_COLORS.get(category, 'gray')
            ax.plot(cat_pen['time_bin'], cat_pen['penetrance'], 'o-',
                   color=color, linewidth=2, markersize=4, label=category)
        
        ax.set_title(f'{method_name}: Penetrance by Category', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hours Post Fertilization', fontsize=11)
        ax.set_ylabel('Penetrance (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    fig.suptitle('Penetrance Comparison: Time-Binned vs Global IQR', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_scatter_comparison(df_tb, df_global, bounds_tb, bounds_global):
    """
    Scatter plots showing penetrant classification under each method.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    for ax, df_marked, bounds_df, method_name in zip(
        axes,
        [df_tb, df_global],
        [bounds_tb, bounds_global],
        ['Time-Binned (σ=1.5)', 'Global IQR']
    ):
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
        
        # Threshold bounds
        ax.fill_between(
            bounds_df['time_bin'],
            bounds_df['low'],
            bounds_df['high'],
            alpha=0.2, color='gray', label='WT bounds'
        )
        ax.plot(bounds_df['time_bin'], bounds_df['low'], 'k--', linewidth=1.5, alpha=0.7)
        ax.plot(bounds_df['time_bin'], bounds_df['high'], 'k--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].set_ylabel(METRIC_NAME, fontsize=12)
    
    fig.suptitle('Penetrant Classification: Method Comparison', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_bounds_diagnostic(bounds_tb, bounds_global, wt_df):
    """
    Two-panel diagnostic: bounds over time + sample counts.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                             gridspec_kw={'height_ratios': [2, 1]}, 
                             sharex=True)
    
    # Top: Bounds comparison
    ax1 = axes[0]
    
    # Time-binned bounds
    ax1.fill_between(
        bounds_tb['time_bin'],
        bounds_tb['low'],
        bounds_tb['high'],
        alpha=0.25, color='blue', label='Time-binned bounds'
    )
    ax1.plot(bounds_tb['time_bin'], bounds_tb['median'], 'b-', linewidth=2, label='Time-binned median')
    
    # Global bounds
    ax1.fill_between(
        bounds_global['time_bin'],
        bounds_global['low'],
        bounds_global['high'],
        alpha=0.25, color='red', label='Global bounds'
    )
    ax1.plot(bounds_global['time_bin'], bounds_global['median'], 'r-', linewidth=2, label='Global median')
    
    ax1.set_ylabel(METRIC_NAME, fontsize=12)
    ax1.set_title(f'WT Threshold Bounds: Time-Binned (σ=1.5) vs Global IQR (k={IQR_K})', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Bottom: Sample counts
    ax2 = axes[1]
    
    n_samples = bounds_tb[['time_bin', 'n_samples']].copy()
    colors = ['red' if n < 10 else 'steelblue' for n in n_samples['n_samples']]
    
    ax2.bar(n_samples['time_bin'], n_samples['n_samples'], 
           width=TIME_BIN_WIDTH * 0.8, color=colors, alpha=0.7)
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=1.5, 
               label='Sparse threshold (N<10)')
    
    ax2.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax2.set_ylabel('# WT Samples per Bin', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def compute_summary_stats(wt_df_tb, wt_df_global, df_tb, df_global):
    """
    Compute summary statistics for method comparison.
    """
    def wt_penetrance(marked_df):
        pen_embryos = marked_df[marked_df['penetrant'] == 1][EMBRYO_COL].nunique()
        total_embryos = marked_df[EMBRYO_COL].nunique()
        return pen_embryos / total_embryos * 100 if total_embryos > 0 else 0
    
    stats = {
        'Time-Binned': {
            'WT Penetrance (%)': wt_penetrance(wt_df_tb),
            'Total WT Frames Outside': (wt_df_tb['penetrant'] == 1).sum(),
            'Total WT Frames': len(wt_df_tb),
        },
        'Global': {
            'WT Penetrance (%)': wt_penetrance(wt_df_global),
            'Total WT Frames Outside': (wt_df_global['penetrant'] == 1).sum(),
            'Total WT Frames': len(wt_df_global),
        }
    }
    
    return pd.DataFrame(stats).T


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("THRESHOLD METHOD COMPARISON")
    print("Time-Binned (Gaussian σ=1.5) vs Global IQR")
    print("="*70)
    
    # Ensure output directories exist
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_trajectory_data()
    wt_df = extract_wt_data(df)
    df, time_bins = bin_data_by_time(df)
    wt_df, _ = bin_data_by_time(wt_df)
    
    # Compute bounds with both methods
    print("\n" + "-"*50)
    bounds_time_binned = compute_wt_bounds(wt_df, time_bins, method='time_binned', smooth_sigma=1.5)
    bounds_global = compute_wt_bounds(wt_df, time_bins, method='global')
    
    # Save bounds to CSV
    bounds_time_binned.to_csv(TABLE_DIR / "bounds_time_binned.csv", index=False)
    bounds_global.to_csv(TABLE_DIR / "bounds_global.csv", index=False)
    print(f"\n✓ Saved bounds to {TABLE_DIR}/")
    
    # Mark penetrance with both methods
    print("\nMarking penetrance...")
    wt_df_tb = mark_penetrant(wt_df, bounds_time_binned)
    wt_df_global = mark_penetrant(wt_df, bounds_global)
    df_tb = mark_penetrant(df, bounds_time_binned)
    df_global = mark_penetrant(df, bounds_global)
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    stats_df = compute_summary_stats(wt_df_tb, wt_df_global, df_tb, df_global)
    print(stats_df.to_string())
    stats_df.to_csv(TABLE_DIR / "method_comparison_summary.csv")
    
    # Generate figures
    print("\n" + "-"*50)
    print("Generating figures...")
    
    # 1. Method comparison (WT data with bounds)
    fig1 = plot_method_comparison(wt_df, bounds_time_binned, bounds_global, time_bins)
    fig1.savefig(FIGURE_DIR / "threshold_method_comparison.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig1)
    print(f"  ✓ threshold_method_comparison.png")
    
    # 2. Penetrance by method
    fig2 = plot_penetrance_by_method(wt_df_tb, wt_df_global, df_tb, df_global, time_bins)
    fig2.savefig(FIGURE_DIR / "penetrance_method_comparison.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig2)
    print(f"  ✓ penetrance_method_comparison.png")
    
    # 3. Scatter comparison
    fig3 = plot_scatter_comparison(df_tb, df_global, bounds_time_binned, bounds_global)
    fig3.savefig(FIGURE_DIR / "scatter_method_comparison.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig3)
    print(f"  ✓ scatter_method_comparison.png")
    
    # 4. Bounds diagnostic
    fig4 = plot_bounds_diagnostic(bounds_time_binned, bounds_global, wt_df)
    fig4.savefig(FIGURE_DIR / "bounds_diagnostic.png", dpi=DPI, bbox_inches='tight')
    plt.close(fig4)
    print(f"  ✓ bounds_diagnostic.png")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}/")
    print(f"  Tables: {TABLE_DIR}/")
    
    print("\n⭐ RECOMMENDATION:")
    tb_wt_pen = stats_df.loc['Time-Binned', 'WT Penetrance (%)']
    global_wt_pen = stats_df.loc['Global', 'WT Penetrance (%)']
    
    if tb_wt_pen < global_wt_pen:
        print(f"  → Time-binned method produces lower WT penetrance ({tb_wt_pen:.1f}% vs {global_wt_pen:.1f}%)")
        print(f"  → Consider using time-binned if WT penetrance is still too high (>5%)")
    else:
        print(f"  → Global method produces lower WT penetrance ({global_wt_pen:.1f}% vs {tb_wt_pen:.1f}%)")
        print(f"  → Consider using global if WT penetrance is acceptable (<5%)")


if __name__ == "__main__":
    main()
