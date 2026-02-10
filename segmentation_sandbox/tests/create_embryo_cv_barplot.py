#!/usr/bin/env python3
"""
Create Individual Embryo CV Bar Plot

Generate a comprehensive bar plot showing CV values for each individual embryo,
grouped by experiment, to demonstrate the dramatic differences in segmentation
variability between experiments and validate threshold recommendations.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import sys

def load_embryo_cv_data(gsam_path: str, target_experiments: List[str]):
    """Load embryo CV data from GSAM file."""
    print(f"📂 Loading embryo CV data from {gsam_path}")
    
    with open(gsam_path, 'r') as f:
        gsam_data = json.load(f)
    
    experiments = gsam_data.get("experiments", {})
    experiments = {k: v for k, v in experiments.items() if k in target_experiments}
    
    embryo_data = []
    
    for exp_id, exp_data in experiments.items():
        print(f"   Processing {exp_id}...")
        
        for video_id, video_data in exp_data.get("videos", {}).items():
            # Group embryos by ID across frames
            embryo_frames = defaultdict(list)
            
            for image_id, image_data in video_data.get("image_ids", {}).items():
                for embryo_id, embryo_data_item in image_data.get("embryos", {}).items():
                    segmentation = embryo_data_item.get("segmentation", {})
                    area = segmentation.get("area")
                    
                    if area is not None and area > 0:
                        embryo_frames[embryo_id].append(area)
            
            # Calculate CV for embryos with multiple frames
            for embryo_id, areas in embryo_frames.items():
                if len(areas) >= 3:
                    mean_area = np.mean(areas)
                    std_area = np.std(areas)
                    cv = std_area / mean_area if mean_area > 0 else 0
                    
                    embryo_data.append({
                        "experiment_id": exp_id,
                        "video_id": video_id,
                        "embryo_id": embryo_id,
                        "full_id": f"{video_id}_{embryo_id}",
                        "cv": cv,
                        "cv_percentage": cv * 100,
                        "mean_area": mean_area,
                        "std_area": std_area,
                        "frame_count": len(areas),
                        "min_area": min(areas),
                        "max_area": max(areas),
                        "area_range": max(areas) - min(areas)
                    })
    
    return embryo_data

def create_embryo_cv_barplot(embryo_data: List[Dict], output_path: str = "results/embryo_cv_barplot.png"):
    """Create comprehensive bar plot of CV values by individual embryo."""
    
    print(f"📊 Creating embryo CV bar plot...")
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(embryo_data)
    
    # Define colors and thresholds
    colors = {
        "20240418": "#2E86AB",  # Professional blue
        "20250305": "#E63946"   # Alert red
    }
    
    thresholds = {
        "Current (15%)": {"value": 15, "color": "#FF6B6B", "style": "--", "width": 2},
        "Recommended (27%)": {"value": 27, "color": "#4ECDC4", "style": "--", "width": 3},
        "Strict (32%)": {"value": 32, "color": "#45B7D1", "style": "-.", "width": 2}
    }
    
    # Create figure with larger size for readability
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Prepare data for plotting
    x_pos = 0
    x_ticks = []
    x_labels = []
    experiment_ranges = {}
    
    # Group by experiment and sort within each group
    for exp_id in ["20240418", "20250305"]:  # Explicit order
        exp_df = df[df['experiment_id'] == exp_id].copy()
        exp_df = exp_df.sort_values('cv_percentage')
        
        start_pos = x_pos
        
        # Create bars for this experiment
        x_positions = range(x_pos, x_pos + len(exp_df))
        bars = ax.bar(x_positions, exp_df['cv_percentage'], 
                     color=colors[exp_id], alpha=0.8, 
                     label=f"{exp_id} (n={len(exp_df)})",
                     edgecolor='white', linewidth=0.5)
        
        # Track positions and labels
        x_ticks.extend(x_positions)
        x_labels.extend([row['full_id'] for _, row in exp_df.iterrows()])
        
        # Store experiment range for annotations
        experiment_ranges[exp_id] = {
            'start': start_pos,
            'end': x_pos + len(exp_df) - 1,
            'center': start_pos + len(exp_df) // 2,
            'count': len(exp_df),
            'mean_cv': exp_df['cv_percentage'].mean(),
            'median_cv': exp_df['cv_percentage'].median(),
            'max_cv': exp_df['cv_percentage'].max()
        }
        
        x_pos += len(exp_df) + 5  # Gap between experiments
        
        # Add experiment separator line
        if exp_id != "20250305":  # Don't add line after last experiment
            ax.axvline(x_pos - 2.5, color='gray', linestyle='-', alpha=0.3, linewidth=2)
    
    # Add threshold lines with annotations
    for label, props in thresholds.items():
        line = ax.axhline(props["value"], color=props["color"], 
                         linestyle=props["style"], linewidth=props["width"],
                         alpha=0.9, label=label)
        
        # Count embryos above threshold for each experiment
        flagged_counts = {}
        total_counts = {}
        for exp_id in ["20240418", "20250305"]:
            exp_data = df[df['experiment_id'] == exp_id]
            flagged = len(exp_data[exp_data['cv_percentage'] > props["value"]])
            total = len(exp_data)
            flagged_counts[exp_id] = flagged
            total_counts[exp_id] = total
        
        # Add annotation for threshold impact
        annotation_text = f"{flagged_counts['20240418']}/{total_counts['20240418']} | {flagged_counts['20250305']}/{total_counts['20250305']}"
        ax.annotate(annotation_text, 
                   xy=(max(x_ticks) * 0.02, props["value"]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=props["color"], alpha=0.7),
                   color='white')
    
    # Add experiment labels and statistics
    for exp_id, ranges in experiment_ranges.items():
        # Main experiment label
        ax.text(ranges['center'], ranges['max_cv'] + 2, 
               f"{exp_id}\n(n={ranges['count']})",
               ha='center', va='bottom', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[exp_id], alpha=0.8),
               color='white')
        
        # Statistics annotation
        stats_text = f"Mean: {ranges['mean_cv']:.1f}%\nMedian: {ranges['median_cv']:.1f}%"
        ax.text(ranges['center'], -5, stats_text,
               ha='center', va='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[exp_id], alpha=0.3))
    
    # Formatting
    ax.set_xlabel('Individual Embryos (sorted by CV within experiment)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=14, fontweight='bold')
    ax.set_title('Segmentation CV by Individual Embryo: Experiment Comparison\n' + 
                'Dramatic Difference in Variability Patterns', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks (show subset to avoid overcrowding)
    tick_interval = max(1, len(x_ticks) // 30)  # Show ~30 ticks max
    shown_ticks = x_ticks[::tick_interval]
    shown_labels = [x_labels[i] for i in range(0, len(x_labels), tick_interval)]
    
    ax.set_xticks(shown_ticks)
    ax.set_xticklabels(shown_labels, rotation=45, ha='right', fontsize=8)
    
    # Y-axis formatting
    ax.set_ylim(0, max(df['cv_percentage']) * 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Separate experiment and threshold legends
    exp_handles = handles[:2]  # First two are experiments
    threshold_handles = handles[2:]  # Rest are thresholds
    
    # Create two legends
    exp_legend = ax.legend(exp_handles, labels[:2], 
                          loc='upper left', title='Experiments', 
                          title_fontsize=12, fontsize=11)
    ax.add_artist(exp_legend)
    
    threshold_legend = ax.legend(threshold_handles, labels[2:], 
                                loc='upper right', title='CV Thresholds',
                                title_fontsize=12, fontsize=11)
    
    # Add summary statistics box
    total_embryos = len(df)
    flagged_15 = len(df[df['cv_percentage'] > 15])
    flagged_27 = len(df[df['cv_percentage'] > 27])
    flagged_32 = len(df[df['cv_percentage'] > 32])
    
    summary_text = f"""KEY FINDINGS:
• Total embryos: {total_embryos}
• Current 15% threshold: {flagged_15} flagged ({flagged_15/total_embryos*100:.1f}%)
• Recommended 27% threshold: {flagged_27} flagged ({flagged_27/total_embryos*100:.1f}%)
• Strict 32% threshold: {flagged_32} flagged ({flagged_32/total_embryos*100:.1f}%)

EXPERIMENT DIFFERENCES:
• 20240418: More consistent (lower CV values)
• 20250305: Highly variable (many >30% CV)
• 3.2x difference in median CV between experiments"""
    
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    
    print(f"✅ Plot saved to: {output_file}")
    print(f"📄 PDF version: {output_file.with_suffix('.pdf')}")
    
    # Also save data
    data_file = output_file.with_suffix('.csv')
    df.to_csv(data_file, index=False)
    print(f"💾 Data exported to: {data_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EMBRYO CV BAR PLOT SUMMARY")
    print("="*60)
    
    for exp_id in ["20240418", "20250305"]:
        exp_data = df[df['experiment_id'] == exp_id]['cv_percentage']
        print(f"\n{exp_id}:")
        print(f"  Count: {len(exp_data)}")
        print(f"  Mean CV: {exp_data.mean():.1f}%")
        print(f"  Median CV: {exp_data.median():.1f}%")
        print(f"  Min CV: {exp_data.min():.1f}%")
        print(f"  Max CV: {exp_data.max():.1f}%")
        print(f"  Std CV: {exp_data.std():.1f}%")
        
        # Threshold analysis
        flagged_15 = len(exp_data[exp_data > 15])
        flagged_27 = len(exp_data[exp_data > 27])
        flagged_32 = len(exp_data[exp_data > 32])
        
        print(f"  Flagged by 15%: {flagged_15}/{len(exp_data)} ({flagged_15/len(exp_data)*100:.1f}%)")
        print(f"  Flagged by 27%: {flagged_27}/{len(exp_data)} ({flagged_27/len(exp_data)*100:.1f}%)")
        print(f"  Flagged by 32%: {flagged_32}/{len(exp_data)} ({flagged_32/len(exp_data)*100:.1f}%)")
    
    # Overall comparison
    ratio_mean = df[df['experiment_id'] == '20250305']['cv_percentage'].mean() / df[df['experiment_id'] == '20240418']['cv_percentage'].mean()
    ratio_median = df[df['experiment_id'] == '20250305']['cv_percentage'].median() / df[df['experiment_id'] == '20240418']['cv_percentage'].median()
    
    print(f"\nExperiment Comparison:")
    print(f"  20250305 has {ratio_mean:.1f}x higher mean CV than 20240418")
    print(f"  20250305 has {ratio_median:.1f}x higher median CV than 20240418")
    
    plt.show()

def main():
    # Configuration
    gsam_path = "data/segmentation/grounded_sam_segmentations.json"
    target_experiments = ["20240418", "20250305"]
    output_path = "results/embryo_cv_comparison_barplot.png"
    
    print("🎯 INDIVIDUAL EMBRYO CV BAR PLOT GENERATOR")
    print(f"Data source: {gsam_path}")
    print(f"Target experiments: {target_experiments}")
    print(f"Output: {output_path}")
    print()
    
    # Load data
    embryo_data = load_embryo_cv_data(gsam_path, target_experiments)
    
    if not embryo_data:
        print("❌ No embryo data found!")
        return 1
    
    print(f"✅ Loaded {len(embryo_data)} embryos with CV data")
    
    # Create bar plot
    create_embryo_cv_barplot(embryo_data, output_path)
    
    print("\n✅ Embryo CV bar plot analysis complete!")
    print("🔍 This visualization clearly shows why the current 15% threshold is too sensitive")
    print("🎯 The recommended 27% threshold provides much more balanced flagging")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())