#!/usr/bin/env python3
"""
Create improved D03 plot with flipped axes and highlight D03 on CV vs 95th percentile plot
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_d03_data():
    """Load the D03 embryo data from the trimmed percentile results."""
    with open('trimmed_percentile_plots/trimmed_percentile_results.json', 'r') as f:
        data = json.load(f)
    
    d03_embryo = data['d03_case_study']
    all_embryos = data['detailed_embryo_data']
    
    return d03_embryo, all_embryos

def create_d03_improved_plot(d03_embryo: dict):
    """Create improved D03 plot with better layout and flipped local variation axis."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    areas = np.array(d03_embryo["areas"])
    frames = np.arange(len(areas))
    local_diffs = d03_embryo.get("local_diffs", [])
    
    # Top plot: Area over time
    ax1.plot(frames, areas, 'b-', linewidth=1.5, alpha=0.8, label='Area')
    
    # Add trend line
    z = np.polyfit(frames, areas, 1)
    p = np.poly1d(z)
    ax1.plot(frames, p(frames), 'r--', alpha=0.8, linewidth=2, label='Linear trend')
    
    # Mark periods
    early_third = len(areas) // 3
    middle_third = 2 * len(areas) // 3
    
    ax1.axvline(early_third, color='orange', linestyle=':', alpha=0.7, label='Period boundaries')
    ax1.axvline(middle_third, color='orange', linestyle=':', alpha=0.7)
    
    ax1.set_title(f'20250305_D03_e01: The Death/Plateau Problem Example\n'
                  f'CV: {d03_embryo["cv"]:.3f} | Local Median: {d03_embryo["local_var_median"]:.3f} | '
                  f'Local 95th: {d03_embryo["local_var_p95"]:.3f}', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_ylabel('Area (pixels)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Local variation over time (FLIPPED AXIS)
    if local_diffs:
        # Create frame indices for local_diffs (accounting for edge effects)
        diff_frames = frames[1:-1]  # Approximate alignment
        if len(local_diffs) == len(diff_frames):
            # FLIPPED: Put 95th percentile on x-axis, frames on y-axis
            ax2.plot(local_diffs, diff_frames, 'g-', linewidth=1.5, alpha=0.8, label='Local variation per frame')
        else:
            # If lengths don't match, interpolate
            diff_frames = np.linspace(0, len(frames)-1, len(local_diffs))
            ax2.plot(local_diffs, diff_frames, 'g-', linewidth=1.5, alpha=0.8, label='Local variation per frame')
        
        # Add threshold lines (now vertical since axis is flipped)
        median_val = d03_embryo["local_var_median"]
        p95_val = d03_embryo["local_var_p95"]
        
        ax2.axvline(median_val, color='blue', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Median: {median_val:.3f} (FAILS)')
        ax2.axvline(p95_val, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'95th percentile: {p95_val:.3f} (SUCCEEDS)')
        
        # Mark periods (now horizontal since axis is flipped)
        ax2.axhline(early_third, color='orange', linestyle=':', alpha=0.7)
        ax2.axhline(middle_third, color='orange', linestyle=':', alpha=0.7)
        
        ax2.set_title('Local Variation vs Frame: Why 95th Percentile Fixes Death/Plateau', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Local Variation (% difference)', fontsize=12)
        ax2.set_ylabel('Frame Number', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text annotations
        ax2.text(0.02, 0.98, 'HIGH VARIATION\n(early frames)\nCaptured by 95th %ile', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                verticalalignment='top', fontsize=11, fontweight='bold')
        
        ax2.text(0.02, 0.02, 'LOW VARIATION\n(death plateau)\nIgnored by 95th %ile', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7),
                verticalalignment='bottom', fontsize=11, fontweight='bold')
        
        # Highlight the key insight
        ax2.text(0.6, 0.5, '95th Percentile = Peak Instability\nMedian = Death Plateau Pollution', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
                fontsize=12, fontweight='bold', ha='center')
    
    plt.tight_layout()
    return fig

def create_cv_vs_p95_with_d03_highlight(all_embryos, d03_embryo):
    """Create CV vs 95th percentile plot with D03 prominently highlighted."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Extract data
    cvs = [e['cv'] for e in all_embryos]
    p95s = [e['local_var_p95'] for e in all_embryos]
    experiments = [e['experiment_id'] for e in all_embryos]
    
    # Plot by experiment
    colors = {'20240418': 'blue', '20250305': 'red'}
    for exp in ['20240418', '20250305']:
        exp_cvs = [cvs[i] for i, e in enumerate(experiments) if e == exp]
        exp_p95s = [p95s[i] for i, e in enumerate(experiments) if e == exp]
        ax.scatter(exp_p95s, exp_cvs, c=colors[exp], label=exp, alpha=0.6, s=50)
    
    # HIGHLIGHT D03 EMBRYO
    d03_cv = d03_embryo['cv']
    d03_p95 = d03_embryo['local_var_p95']
    
    # Big black X for D03
    ax.scatter([d03_p95], [d03_cv], c='black', s=400, marker='X', linewidth=4, 
               label='D03 (death/plateau case)', zorder=10)
    
    # Add a circle around it
    ax.scatter([d03_p95], [d03_cv], c='none', s=800, marker='o', linewidth=3, 
               edgecolor='black', zorder=9)
    
    # Add annotation pointing to D03
    ax.annotate(f'20250305_D03_e01\nCV: {d03_cv:.3f} (HIGH)\nLocal 95th: {d03_p95:.3f} (captures peak variation)\n\nDemonstrates the fix!', 
                xy=(d03_p95, d03_cv), xytext=(d03_p95 + 0.01, d03_cv - 0.05),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
                fontsize=11, fontweight='bold')
    
    # Add threshold lines
    cv_threshold = 0.15
    p95_threshold = 0.0106  # From the analysis
    
    ax.axhline(cv_threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
               label=f'CV threshold ({cv_threshold*100:.0f}%)')
    ax.axvline(p95_threshold, color='blue', linestyle='--', alpha=0.8, linewidth=2,
               label=f'95th %ile threshold ({p95_threshold*100:.1f}%)')
    
    # Labels and formatting
    ax.set_xlabel('Local Variation (95th percentile)', fontsize=14)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=14)
    ax.set_title('CV vs Local 95th Percentile: D03 Shows Why 95th Percentile Works\n'
                 'D03 has high CV (genuine variation) but moderate 95th percentile (captures peak instability)', 
                 fontsize=16, fontweight='bold')
    
    # Add correlation
    correlation = np.corrcoef(cvs, p95s)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8), fontsize=12)
    
    # Add quadrant labels
    ax.text(0.7, 0.05, 'High 95th%ile\nLow CV\n(Local spikes)', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            fontsize=10, ha='center')
    
    ax.text(0.05, 0.8, 'High CV\nLow 95th%ile\n(Biological trends)', transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
            fontsize=10, ha='center')
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.tight_layout()
    return fig

def main():
    """Generate the improved plots with D03 highlighting."""
    
    print("🎯 Creating improved D03 plots with highlighting...")
    
    # Load data
    d03_embryo, all_embryos = load_d03_data()
    
    # Create output directory
    output_dir = Path('d03_highlighted_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("📊 Creating improved D03 death/plateau plot...")
    d03_fig = create_d03_improved_plot(d03_embryo)
    d03_fig.savefig(output_dir / "d03_death_plateau_IMPROVED.png", dpi=300, bbox_inches='tight')
    d03_fig.savefig(output_dir / "d03_death_plateau_IMPROVED.pdf", bbox_inches='tight')
    
    print("📈 Creating CV vs 95th percentile plot with D03 highlighted...")
    cv_p95_fig = create_cv_vs_p95_with_d03_highlight(all_embryos, d03_embryo)
    cv_p95_fig.savefig(output_dir / "cv_vs_p95_with_D03_highlight.png", dpi=300, bbox_inches='tight')
    cv_p95_fig.savefig(output_dir / "cv_vs_p95_with_D03_highlight.pdf", bbox_inches='tight')
    
    print(f"\n✅ Plots saved to: {output_dir}")
    print("📋 Generated files:")
    print(f"  📊 d03_death_plateau_IMPROVED.png - Shows death/plateau problem with flipped axis")
    print(f"  📈 cv_vs_p95_with_D03_highlight.png - CV vs 95th percentile with D03 prominently shown")
    print(f"\n🎯 Key improvements:")
    print(f"  ✅ Flipped local variation axis (95th percentile on x-axis)")
    print(f"  ✅ D03 prominently highlighted on CV vs 95th percentile plot")
    print(f"  ✅ Clear annotations explaining why 95th percentile works")

if __name__ == "__main__":
    main()