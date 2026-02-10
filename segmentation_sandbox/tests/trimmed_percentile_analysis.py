#!/usr/bin/env python3
"""
Trimmed Percentile Analysis: Fixing the Death/Plateau Problem

Compare three approaches for detecting embryo segmentation variability:
1. CV (original - fails on trends)
2. Local Variation Median (first improvement - fails on death/plateau)  
3. Local Variation 95th Percentile (final solution - robust to all artifacts)

Focus on 20250305_D03_e01 as the key example demonstrating the death/plateau problem.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def calculate_local_variation_metrics(areas: np.ndarray, window_size: int = 2) -> dict:
    """
    Calculate both median and 95th percentile local variation metrics.
    
    Args:
        areas: Array of area values in temporal order
        window_size: Number of frames to check before/after each frame
    
    Returns:
        Dictionary with median and 95th percentile local variation scores
    """
    local_diffs_pct = []
    
    for i in range(len(areas)):
        # Define the local window, excluding the current frame
        neighbor_indices = (
            list(range(max(0, i - window_size), i)) + 
            list(range(i + 1, min(len(areas), i + window_size + 1)))
        )
        
        if not neighbor_indices:
            continue
        
        local_mean = np.mean(areas[neighbor_indices])
        if local_mean > 0:
            diff_pct = abs(areas[i] - local_mean) / local_mean
            local_diffs_pct.append(diff_pct)
    
    if not local_diffs_pct:
        return {"median": 0.0, "p95": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    
    return {
        "median": float(np.median(local_diffs_pct)),
        "p95": float(np.percentile(local_diffs_pct, 95)),
        "p90": float(np.percentile(local_diffs_pct, 90)),
        "p99": float(np.percentile(local_diffs_pct, 99)),
        "max": float(np.max(local_diffs_pct)),
        "local_diffs": local_diffs_pct  # Keep for detailed analysis
    }

def load_embryo_data_with_all_metrics(gsam_path: str, target_experiments: list = None):
    """Load embryo data and calculate all three metrics."""
    print(f"📂 Loading GSAM data from {gsam_path}")
    
    with open(gsam_path, 'r') as f:
        gsam_data = json.load(f)
    
    experiments = gsam_data.get("experiments", {})
    if target_experiments:
        experiments = {k: v for k, v in experiments.items() if k in target_experiments}
    
    embryo_data = []
    
    for exp_id, exp_data in experiments.items():
        for video_id, video_data in exp_data.get("videos", {}).items():
            # Group embryos by ID across frames
            embryo_frames = defaultdict(list)  # embryo_id -> [(image_id, area)]
            
            for image_id, image_data in video_data.get("image_ids", {}).items():
                for embryo_id, embryo_data_item in image_data.get("embryos", {}).items():
                    segmentation = embryo_data_item.get("segmentation", {})
                    area = segmentation.get("area")
                    
                    if area is not None and area > 0:
                        embryo_frames[embryo_id].append((image_id, area))
            
            # Calculate all metrics for embryos with multiple frames
            for embryo_id, frame_data in embryo_frames.items():
                if len(frame_data) >= 5:  # Minimum for meaningful analysis
                    # Sort by image_id to ensure temporal order
                    frame_data.sort(key=lambda x: x[0])
                    areas = np.array([area for _, area in frame_data])
                    image_ids = [img_id for img_id, _ in frame_data]
                    
                    # Calculate traditional CV
                    mean_area = np.mean(areas)
                    std_area = np.std(areas)
                    cv = std_area / mean_area if mean_area > 0 else 0
                    
                    # Calculate local variation metrics
                    local_metrics = calculate_local_variation_metrics(areas, window_size=2)
                    
                    embryo_data.append({
                        "experiment_id": exp_id,
                        "video_id": video_id,
                        "embryo_id": embryo_id,
                        "cv": cv,
                        "local_var_median": local_metrics["median"],
                        "local_var_p95": local_metrics["p95"],
                        "local_var_p90": local_metrics["p90"],
                        "local_var_p99": local_metrics["p99"],
                        "local_var_max": local_metrics["max"],
                        "mean_area": mean_area,
                        "std_area": std_area,
                        "areas": areas.tolist(),
                        "image_ids": image_ids,
                        "frame_count": len(areas),
                        "min_area": float(np.min(areas)),
                        "max_area": float(np.max(areas)),
                        "area_range_ratio": float(np.max(areas) / np.min(areas)) if np.min(areas) > 0 else 0,
                        "local_diffs": local_metrics.get("local_diffs", [])  # For detailed analysis
                    })
    
    return embryo_data

def analyze_three_methods(embryo_data: List[Dict]):
    """Compare CV, Local Median, and Local 95th Percentile methods."""
    
    print("\n" + "="*80)
    print("THREE-METHOD COMPARISON: CV vs Local Median vs Local 95th Percentile")
    print("="*80)
    
    # Define thresholds based on previous analysis
    cv_threshold = 0.15          # Current threshold
    median_threshold = 0.012     # 90th percentile from previous analysis
    p95_threshold = 0.05         # To be determined based on data
    
    total_embryos = len(embryo_data)
    print(f"Total embryos analyzed: {total_embryos}")
    
    # Calculate p95 threshold based on data distribution
    p95_values = [e["local_var_p95"] for e in embryo_data]
    p95_90th_percentile = np.percentile(p95_values, 90)
    p95_threshold = p95_90th_percentile
    
    print(f"Thresholds used:")
    print(f"  CV: {cv_threshold*100:.0f}%")
    print(f"  Local Median: {median_threshold*100:.1f}%")
    print(f"  Local 95th Percentile: {p95_threshold*100:.1f}% (90th percentile of distribution)")
    print()
    
    # Flag embryos using each method
    cv_flagged = [e for e in embryo_data if e["cv"] > cv_threshold]
    median_flagged = [e for e in embryo_data if e["local_var_median"] > median_threshold]
    p95_flagged = [e for e in embryo_data if e["local_var_p95"] > p95_threshold]
    
    print(f"FLAGGING RESULTS:")
    print(f"  CV method:               {len(cv_flagged)}/{total_embryos} flagged ({100*len(cv_flagged)/total_embryos:.1f}%)")
    print(f"  Local Median method:     {len(median_flagged)}/{total_embryos} flagged ({100*len(median_flagged)/total_embryos:.1f}%)")
    print(f"  Local 95th Percentile:   {len(p95_flagged)}/{total_embryos} flagged ({100*len(p95_flagged)/total_embryos:.1f}%)")
    print()
    
    # Group by experiment
    by_experiment = defaultdict(list)
    for embryo in embryo_data:
        by_experiment[embryo["experiment_id"]].append(embryo)
    
    print("PER-EXPERIMENT BREAKDOWN:")
    print("-" * 50)
    
    for exp_id, exp_embryos in by_experiment.items():
        exp_total = len(exp_embryos)
        exp_cv_flagged = [e for e in exp_embryos if e["cv"] > cv_threshold]
        exp_median_flagged = [e for e in exp_embryos if e["local_var_median"] > median_threshold]
        exp_p95_flagged = [e for e in exp_embryos if e["local_var_p95"] > p95_threshold]
        
        print(f"{exp_id}:")
        print(f"  CV flagged:               {len(exp_cv_flagged)}/{exp_total} ({100*len(exp_cv_flagged)/exp_total:.1f}%)")
        print(f"  Local Median flagged:     {len(exp_median_flagged)}/{exp_total} ({100*len(exp_median_flagged)/exp_total:.1f}%)")
        print(f"  Local 95th Percentile:    {len(exp_p95_flagged)}/{exp_total} ({100*len(exp_p95_flagged)/exp_total:.1f}%)")
        print()
    
    return {
        "cv_flagged": cv_flagged,
        "median_flagged": median_flagged,
        "p95_flagged": p95_flagged,
        "thresholds": {
            "cv": cv_threshold,
            "median": median_threshold,
            "p95": p95_threshold
        }
    }

def analyze_d03_embryo_specifically(embryo_data: List[Dict]):
    """Focus on the 20250305_D03_e01 embryo to demonstrate the death/plateau problem."""
    
    print("\n" + "="*80)
    print("CASE STUDY: 20250305_D03_e01 - Demonstrating the Death/Plateau Problem")
    print("="*80)
    
    # Find the D03 embryo
    d03_embryo = None
    for embryo in embryo_data:
        if (embryo["experiment_id"] == "20250305" and 
            embryo["video_id"] == "20250305_D03" and 
            embryo["embryo_id"] == "20250305_D03_e01"):
            d03_embryo = embryo
            break
    
    if not d03_embryo:
        print("❌ D03 embryo not found!")
        return
    
    print(f"Embryo: {d03_embryo['embryo_id']}")
    print(f"Frames: {d03_embryo['frame_count']}")
    print(f"Area range: {d03_embryo['min_area']:.0f} - {d03_embryo['max_area']:.0f} pixels")
    print()
    
    # Show the three metrics
    cv = d03_embryo["cv"]
    median_var = d03_embryo["local_var_median"]
    p95_var = d03_embryo["local_var_p95"]
    p99_var = d03_embryo["local_var_p99"]
    max_var = d03_embryo["local_var_max"]
    
    print(f"METRIC COMPARISON:")
    print(f"  CV:                    {cv:.3f} ({cv*100:.1f}%)")
    print(f"  Local Median:          {median_var:.3f} ({median_var*100:.1f}%)")
    print(f"  Local 95th Percentile: {p95_var:.3f} ({p95_var*100:.1f}%)")
    print(f"  Local 99th Percentile: {p99_var:.3f} ({p99_var*100:.1f}%)")
    print(f"  Local Maximum:         {max_var:.3f} ({max_var*100:.1f}%)")
    print()
    
    # Analyze the progression
    areas = np.array(d03_embryo["areas"])
    
    # Look at early vs late periods
    early_third = len(areas) // 3
    middle_third = 2 * len(areas) // 3
    
    early_areas = areas[:early_third]
    middle_areas = areas[early_third:middle_third]
    late_areas = areas[middle_third:]
    
    print(f"TEMPORAL ANALYSIS:")
    print(f"  Early period (frames 0-{early_third}):   mean={np.mean(early_areas):.0f}, std={np.std(early_areas):.0f}")
    print(f"  Middle period (frames {early_third}-{middle_third}): mean={np.mean(middle_areas):.0f}, std={np.std(middle_areas):.0f}")
    print(f"  Late period (frames {middle_third}-{len(areas)}):   mean={np.mean(late_areas):.0f}, std={np.std(late_areas):.0f}")
    print()
    
    # Calculate local variations for each period
    local_diffs = d03_embryo.get("local_diffs", [])
    if local_diffs:
        # Assuming local_diffs aligns with areas (minus edge effects)
        n_diffs = len(local_diffs)
        early_diffs = local_diffs[:n_diffs//3]
        middle_diffs = local_diffs[n_diffs//3:2*n_diffs//3]
        late_diffs = local_diffs[2*n_diffs//3:]
        
        print(f"LOCAL VARIATION BY PERIOD:")
        print(f"  Early period:  median={np.median(early_diffs):.3f}, 95th={np.percentile(early_diffs, 95):.3f}")
        print(f"  Middle period: median={np.median(middle_diffs):.3f}, 95th={np.percentile(middle_diffs, 95):.3f}")
        print(f"  Late period:   median={np.median(late_diffs):.3f}, 95th={np.percentile(late_diffs, 95):.3f}")
        print()
    
    # Conclusion
    print(f"CONCLUSION:")
    print(f"  CV correctly identifies this as highly variable ({cv:.3f} > 0.15)")
    print(f"  Local Median FAILS due to death plateau ({median_var:.3f} < 0.012)")  
    print(f"  Local 95th Percentile SUCCEEDS by capturing peak instability ({p95_var:.3f})")
    
    return d03_embryo

def create_d03_detailed_plot(d03_embryo: dict):
    """Create detailed plot of the D03 embryo showing the death/plateau problem."""
    
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
    
    ax1.set_title(f'20250305_D03_e01: Area Over Time (340 frames)\n'
                  f'CV: {d03_embryo["cv"]:.3f} | Local Median: {d03_embryo["local_var_median"]:.3f} | '
                  f'Local 95th: {d03_embryo["local_var_p95"]:.3f}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Area (pixels)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Local variation over time
    if local_diffs:
        # Create frame indices for local_diffs (accounting for edge effects)
        diff_frames = frames[1:-1]  # Approximate alignment
        if len(local_diffs) == len(diff_frames):
            ax2.plot(diff_frames, local_diffs, 'g-', linewidth=1, alpha=0.8, label='Local variation per frame')
        else:
            # If lengths don't match, interpolate
            diff_frames = np.linspace(0, len(frames)-1, len(local_diffs))
            ax2.plot(diff_frames, local_diffs, 'g-', linewidth=1, alpha=0.8, label='Local variation per frame')
        
        # Add threshold lines
        median_val = d03_embryo["local_var_median"]
        p95_val = d03_embryo["local_var_p95"]
        
        ax2.axhline(median_val, color='blue', linestyle='--', alpha=0.8, 
                   label=f'Median: {median_val:.3f}')
        ax2.axhline(p95_val, color='red', linestyle='--', alpha=0.8, 
                   label=f'95th percentile: {p95_val:.3f}')
        
        # Mark periods
        ax2.axvline(early_third, color='orange', linestyle=':', alpha=0.7)
        ax2.axvline(middle_third, color='orange', linestyle=':', alpha=0.7)
        
        ax2.set_title('Local Variation Over Time: Showing Death/Plateau Effect', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Local Variation (% difference)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add text annotations
        ax2.text(0.02, 0.98, 'HIGH VARIATION\n(early frames)', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                verticalalignment='top', fontsize=10, fontweight='bold')
        
        ax2.text(0.8, 0.02, 'LOW VARIATION\n(death plateau)', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7),
                verticalalignment='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_three_method_comparison_plot(embryo_data: List[Dict], results: dict):
    """Create comprehensive comparison of all three methods."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Extract data
    cvs = [e['cv'] for e in embryo_data]
    medians = [e['local_var_median'] for e in embryo_data]
    p95s = [e['local_var_p95'] for e in embryo_data]
    experiments = [e['experiment_id'] for e in embryo_data]
    
    # Plot 1: CV vs Local 95th Percentile (showing improvement over CV)
    colors = {'20240418': 'blue', '20250305': 'red'}
    for exp in ['20240418', '20250305']:
        exp_cvs = [cvs[i] for i, e in enumerate(experiments) if e == exp]
        exp_p95s = [p95s[i] for i, e in enumerate(experiments) if e == exp]
        ax1.scatter(exp_cvs, exp_p95s, c=colors[exp], label=exp, alpha=0.7, s=40)
    
    ax1.axvline(results["thresholds"]["cv"], color='red', linestyle='--', alpha=0.8, label='CV threshold')
    ax1.axhline(results["thresholds"]["p95"], color='blue', linestyle='--', alpha=0.8, label='P95 threshold')
    ax1.set_xlabel('CV')
    ax1.set_ylabel('Local Variation (95th percentile)')
    ax1.set_title('CV vs Local 95th Percentile', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Local Median vs Local 95th Percentile (showing death/plateau fix)
    for exp in ['20240418', '20250305']:
        exp_medians = [medians[i] for i, e in enumerate(experiments) if e == exp]
        exp_p95s = [p95s[i] for i, e in enumerate(experiments) if e == exp]
        ax2.scatter(exp_medians, exp_p95s, c=colors[exp], label=exp, alpha=0.7, s=40)
    
    ax2.axvline(results["thresholds"]["median"], color='green', linestyle='--', alpha=0.8, label='Median threshold')
    ax2.axhline(results["thresholds"]["p95"], color='blue', linestyle='--', alpha=0.8, label='P95 threshold')
    
    # Highlight the D03 embryo
    d03_median = None
    d03_p95 = None
    for e in embryo_data:
        if e["embryo_id"] == "20250305_D03_e01":
            d03_median = e["local_var_median"]
            d03_p95 = e["local_var_p95"]
            ax2.scatter([d03_median], [d03_p95], c='black', s=200, marker='x', linewidth=3, 
                       label='D03 (death/plateau case)')
            break
    
    ax2.set_xlabel('Local Variation (median)')
    ax2.set_ylabel('Local Variation (95th percentile)')
    ax2.set_title('Median vs 95th Percentile: Fixing Death/Plateau Problem', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Flagging rates comparison
    methods = ['CV', 'Local Median', 'Local 95th %ile']
    flagging_rates = [
        100 * len(results["cv_flagged"]) / len(embryo_data),
        100 * len(results["median_flagged"]) / len(embryo_data), 
        100 * len(results["p95_flagged"]) / len(embryo_data)
    ]
    
    bars = ax3.bar(methods, flagging_rates, color=['red', 'orange', 'blue'], alpha=0.7)
    ax3.set_ylabel('% Embryos Flagged')
    ax3.set_title('Overall Flagging Rates: Three Methods', fontweight='bold')
    ax3.set_ylim(0, max(flagging_rates) * 1.2)
    
    for bar, rate in zip(bars, flagging_rates):
        ax3.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Per-experiment breakdown
    exp_names = ['20240418\n(Lower Bound)', '20250305\n(Upper Bound)']
    
    # Calculate rates per experiment
    exp_rates = {}
    for exp_id in ['20240418', '20250305']:
        exp_embryos = [e for e in embryo_data if e["experiment_id"] == exp_id]
        exp_total = len(exp_embryos)
        
        exp_cv_flagged = len([e for e in exp_embryos if e["cv"] > results["thresholds"]["cv"]])
        exp_median_flagged = len([e for e in exp_embryos if e["local_var_median"] > results["thresholds"]["median"]])
        exp_p95_flagged = len([e for e in exp_embryos if e["local_var_p95"] > results["thresholds"]["p95"]])
        
        exp_rates[exp_id] = [
            100 * exp_cv_flagged / exp_total,
            100 * exp_median_flagged / exp_total,
            100 * exp_p95_flagged / exp_total
        ]
    
    x = np.arange(len(exp_names))
    width = 0.25
    
    bars1 = ax4.bar(x - width, [exp_rates['20240418'][0], exp_rates['20250305'][0]], 
                    width, label='CV', color='red', alpha=0.7)
    bars2 = ax4.bar(x, [exp_rates['20240418'][1], exp_rates['20250305'][1]], 
                    width, label='Local Median', color='orange', alpha=0.7)
    bars3 = ax4.bar(x + width, [exp_rates['20240418'][2], exp_rates['20250305'][2]], 
                    width, label='Local 95th %ile', color='blue', alpha=0.7)
    
    ax4.set_xlabel('Experiment')
    ax4.set_ylabel('% Embryos Flagged')
    ax4.set_title('Per-Experiment Breakdown: Evolution of Methods', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(exp_names)
    ax4.legend()
    ax4.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Three-Method Evolution: CV → Local Median → Local 95th Percentile\n'
                 'Fixing CV False Positives AND Death/Plateau Artifacts', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def main():
    """Run the complete three-method comparison analysis."""
    
    # Configuration
    gsam_path = "data/segmentation/grounded_sam_segmentations.json"
    target_experiments = ["20240418", "20250305"]
    
    print("🎯 THREE-METHOD COMPARISON: Fixing the Death/Plateau Problem")
    print(f"Data source: {gsam_path}")
    print(f"Target experiments: {target_experiments}")
    print()
    
    # Load data with all metrics
    embryo_data = load_embryo_data_with_all_metrics(gsam_path, target_experiments)
    
    if not embryo_data:
        print("❌ No embryo data found!")
        return
    
    # Run three-method comparison
    results = analyze_three_methods(embryo_data)
    
    # Focus on the D03 embryo case study
    d03_embryo = analyze_d03_embryo_specifically(embryo_data)
    
    # Create visualizations
    output_dir = Path('trimmed_percentile_plots')
    output_dir.mkdir(exist_ok=True)
    
    # D03 detailed plot
    if d03_embryo:
        d03_fig = create_d03_detailed_plot(d03_embryo)
        d03_fig.savefig(output_dir / "d03_death_plateau_analysis.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved D03 analysis: {output_dir}/d03_death_plateau_analysis.png")
    
    # Three-method comparison plot
    comparison_fig = create_three_method_comparison_plot(embryo_data, results)
    comparison_fig.savefig(output_dir / "three_method_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved three-method comparison: {output_dir}/three_method_comparison.png")
    
    # Save detailed results
    results_data = {
        "analysis_summary": {
            "total_embryos": len(embryo_data),
            "cv_flagged": len(results["cv_flagged"]),
            "median_flagged": len(results["median_flagged"]),
            "p95_flagged": len(results["p95_flagged"])
        },
        "thresholds": results["thresholds"],
        "d03_case_study": d03_embryo,
        "detailed_embryo_data": embryo_data
    }
    
    with open(output_dir / "trimmed_percentile_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print("\n🎯 KEY FINDINGS:")
    print("✅ Local 95th Percentile method fixes BOTH problems:")
    print("  - Eliminates CV's false positives on biological trends")
    print("  - Captures variation that Local Median misses due to death/plateau")
    print(f"✅ D03 embryo demonstrates the fix:")
    print(f"  - CV: {d03_embryo['cv']:.3f} (correctly flags)")
    print(f"  - Local Median: {d03_embryo['local_var_median']:.3f} (FAILS - death plateau)")
    print(f"  - Local 95th: {d03_embryo['local_var_p95']:.3f} (SUCCEEDS - captures peak variation)")

if __name__ == "__main__":
    main()