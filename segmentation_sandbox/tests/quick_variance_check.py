#!/usr/bin/env python3
"""
Quick Segmentation Variance Check

Fast analysis that uses pre-computed area data when available,
only decoding masks when necessary.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def quick_variance_analysis(gsam_path: str, target_experiments: list = None):
    """Quick variance analysis using existing area data."""
    print(f"📂 Loading GSAM data from {gsam_path}")
    
    with open(gsam_path, 'r') as f:
        gsam_data = json.load(f)
    
    print("🔍 Extracting variance data (using pre-computed areas when available)...")
    
    experiments = gsam_data.get("experiments", {})
    if target_experiments:
        experiments = {k: v for k, v in experiments.items() if k in target_experiments}
        print(f"🎯 Analyzing experiments: {list(experiments.keys())}")
    
    all_embryo_cvs = []
    experiment_results = {}
    
    for exp_id, exp_data in experiments.items():
        print(f"📁 Processing {exp_id}...")
        
        exp_cvs = []
        
        for video_id, video_data in exp_data.get("videos", {}).items():
            # Group embryos by ID across frames
            embryo_frames = defaultdict(list)  # embryo_id -> [areas]
            
            for image_id, image_data in video_data.get("image_ids", {}).items():
                for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                    # Use pre-computed area if available
                    area = embryo_data.get("area")
                    if area is not None and area > 0:
                        embryo_frames[embryo_id].append(area)
            
            # Calculate CV for each embryo
            for embryo_id, areas in embryo_frames.items():
                if len(areas) >= 3:  # Minimum for CV calculation
                    mean_area = np.mean(areas)
                    std_area = np.std(areas)
                    cv = std_area / mean_area if mean_area > 0 else 0
                    
                    exp_cvs.append(cv)
                    all_embryo_cvs.append(cv)
        
        # Store experiment results
        if exp_cvs:
            experiment_results[exp_id] = {
                "embryo_count": len(exp_cvs),
                "cv_mean": np.mean(exp_cvs),
                "cv_median": np.median(exp_cvs),
                "cv_p90": np.percentile(exp_cvs, 90),
                "cv_p95": np.percentile(exp_cvs, 95),
                "cv_p99": np.percentile(exp_cvs, 99),
                "cv_max": np.max(exp_cvs),
                "flagged_15pct": sum(1 for cv in exp_cvs if cv > 0.15),
                "flagged_20pct": sum(1 for cv in exp_cvs if cv > 0.20),
                "flagged_25pct": sum(1 for cv in exp_cvs if cv > 0.25)
            }
        
        print(f"   ✅ Found {len(exp_cvs)} embryos with 3+ frames")
    
    # Overall statistics
    if all_embryo_cvs:
        print("\n" + "="*60)
        print("QUICK VARIANCE ANALYSIS RESULTS")
        print("="*60)
        print(f"Total embryos analyzed: {len(all_embryo_cvs)}")
        print(f"Overall CV statistics:")
        print(f"  Mean: {np.mean(all_embryo_cvs):.3f}")
        print(f"  Median: {np.median(all_embryo_cvs):.3f}")
        print(f"  90th percentile: {np.percentile(all_embryo_cvs, 90):.3f}")
        print(f"  95th percentile: {np.percentile(all_embryo_cvs, 95):.3f}")
        print(f"  99th percentile: {np.percentile(all_embryo_cvs, 99):.3f}")
        print(f"  Max: {np.max(all_embryo_cvs):.3f}")
        print()
        
        # Current threshold analysis
        flagged_15 = sum(1 for cv in all_embryo_cvs if cv > 0.15)
        flagged_20 = sum(1 for cv in all_embryo_cvs if cv > 0.20)
        flagged_25 = sum(1 for cv in all_embryo_cvs if cv > 0.25)
        
        print("Threshold Analysis:")
        print(f"  15% CV threshold: {flagged_15}/{len(all_embryo_cvs)} embryos flagged ({100*flagged_15/len(all_embryo_cvs):.1f}%)")
        print(f"  20% CV threshold: {flagged_20}/{len(all_embryo_cvs)} embryos flagged ({100*flagged_20/len(all_embryo_cvs):.1f}%)")
        print(f"  25% CV threshold: {flagged_25}/{len(all_embryo_cvs)} embryos flagged ({100*flagged_25/len(all_embryo_cvs):.1f}%)")
        print()
        
        # Per-experiment breakdown
        print("Per-Experiment Results:")
        for exp_id, results in experiment_results.items():
            print(f"  {exp_id}:")
            print(f"    Embryos: {results['embryo_count']}")
            print(f"    Mean CV: {results['cv_mean']:.3f}")
            print(f"    95th percentile: {results['cv_p95']:.3f}")
            flagged_pct = 100 * results['flagged_15pct'] / results['embryo_count']
            print(f"    Flagged by 15%: {results['flagged_15pct']} ({flagged_pct:.1f}%)")
            print()
        
        # Recommendations
        print("RECOMMENDATIONS:")
        p95 = np.percentile(all_embryo_cvs, 95)
        p90 = np.percentile(all_embryo_cvs, 90)
        
        if p95 < 0.15:
            print(f"✅ Current 15% threshold seems reasonable (95th percentile = {p95:.3f})")
        else:
            print(f"⚠️ Current 15% threshold may be too strict!")
            print(f"   95th percentile CV = {p95:.3f}")
            print(f"   Consider threshold around {p95:.3f} (95th percentile)")
            print(f"   Or {p90:.3f} (90th percentile) for stricter detection")
        
        return experiment_results, all_embryo_cvs
    else:
        print("❌ No embryos with sufficient data found!")
        return {}, []

if __name__ == "__main__":
    import sys
    
    gsam_path = "data/segmentation/grounded_sam_segmentations.json"
    target_experiments = ["20240418", "20250305"]
    
    if len(sys.argv) > 1:
        gsam_path = sys.argv[1]
    if len(sys.argv) > 2:
        target_experiments = sys.argv[2].split(",")
    
    quick_variance_analysis(gsam_path, target_experiments)