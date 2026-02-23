#!/usr/bin/env python3
"""
Improved Segmentation Variance Check

Updated to correctly extract area from segmentation.area and handle the temporal structure.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def improved_variance_analysis(gsam_path: str, target_experiments: list = None):
    """Improved variance analysis with correct area extraction."""
    print(f"📂 Loading GSAM data from {gsam_path}")
    
    with open(gsam_path, 'r') as f:
        gsam_data = json.load(f)
    
    print("🔍 Extracting variance data (using segmentation.area)...")
    
    experiments = gsam_data.get("experiments", {})
    if target_experiments:
        experiments = {k: v for k, v in experiments.items() if k in target_experiments}
        print(f"🎯 Analyzing experiments: {list(experiments.keys())}")
    
    all_embryo_cvs = []
    experiment_results = {}
    temporal_stats = {}
    
    for exp_id, exp_data in experiments.items():
        print(f"📁 Processing {exp_id}...")
        
        exp_cvs = []
        videos_with_multiple_frames = 0
        total_videos = 0
        
        for video_id, video_data in exp_data.get("videos", {}).items():
            total_videos += 1
            
            # Group embryos by ID across frames
            embryo_frames = defaultdict(list)  # embryo_id -> [(image_id, area)]
            
            images = video_data.get("image_ids", {})
            if len(images) > 1:
                videos_with_multiple_frames += 1
            
            for image_id, image_data in images.items():
                for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                    # Extract area from segmentation.area
                    segmentation = embryo_data.get("segmentation", {})
                    area = segmentation.get("area")
                    
                    if area is not None and area > 0:
                        embryo_frames[embryo_id].append((image_id, area))
            
            # Calculate CV for embryos with multiple frames
            for embryo_id, frame_data in embryo_frames.items():
                if len(frame_data) >= 3:  # Minimum for CV calculation
                    areas = [area for _, area in frame_data]
                    mean_area = np.mean(areas)
                    std_area = np.std(areas)
                    cv = std_area / mean_area if mean_area > 0 else 0
                    
                    exp_cvs.append(cv)
                    all_embryo_cvs.append(cv)
        
        # Store experiment results
        temporal_stats[exp_id] = {
            "total_videos": total_videos,
            "videos_with_multiple_frames": videos_with_multiple_frames,
            "embryos_with_temporal_data": len(exp_cvs)
        }
        
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
        
        print(f"   ✅ Found {len(exp_cvs)} embryos with 3+ frames from {videos_with_multiple_frames}/{total_videos} multi-frame videos")
    
    # Overall statistics
    print("\n" + "="*60)
    print("IMPROVED VARIANCE ANALYSIS RESULTS")
    print("="*60)
    
    # Temporal structure analysis
    print("TEMPORAL STRUCTURE:")
    for exp_id, stats in temporal_stats.items():
        print(f"  {exp_id}:")
        print(f"    Total videos: {stats['total_videos']}")
        print(f"    Multi-frame videos: {stats['videos_with_multiple_frames']}")
        print(f"    Embryos with temporal data: {stats['embryos_with_temporal_data']}")
    print()
    
    if all_embryo_cvs:
        print(f"Total embryos with temporal data: {len(all_embryo_cvs)}")
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
        
        print("THRESHOLD ANALYSIS:")
        print(f"  15% CV threshold: {flagged_15}/{len(all_embryo_cvs)} embryos flagged ({100*flagged_15/len(all_embryo_cvs):.1f}%)")
        print(f"  20% CV threshold: {flagged_20}/{len(all_embryo_cvs)} embryos flagged ({100*flagged_20/len(all_embryo_cvs):.1f}%)")
        print(f"  25% CV threshold: {flagged_25}/{len(all_embryo_cvs)} embryos flagged ({100*flagged_25/len(all_embryo_cvs):.1f}%)")
        print()
        
        # Per-experiment breakdown
        print("PER-EXPERIMENT RESULTS:")
        for exp_id, results in experiment_results.items():
            print(f"  {exp_id}:")
            print(f"    Embryos: {results['embryo_count']}")
            print(f"    Mean CV: {results['cv_mean']:.3f}")
            print(f"    95th percentile: {results['cv_p95']:.3f}")
            flagged_pct = 100 * results['flagged_15pct'] / results['embryo_count']
            print(f"    Flagged by 15%: {results['flagged_15pct']} ({flagged_pct:.1f}%)")
            print()
        
        # Show some example CV values
        print("EXAMPLE CV VALUES (highest 10):")
        sorted_cvs = sorted(all_embryo_cvs, reverse=True)
        for i, cv in enumerate(sorted_cvs[:10]):
            print(f"  {i+1}. CV = {cv:.3f}")
        print()
        
        # Recommendations
        print("RECOMMENDATIONS:")
        p95 = np.percentile(all_embryo_cvs, 95)
        p90 = np.percentile(all_embryo_cvs, 90)
        median = np.median(all_embryo_cvs)
        
        print(f"Current 15% threshold flags {100*flagged_15/len(all_embryo_cvs):.1f}% of embryos")
        
        if p95 < 0.15:
            print(f"✅ Current 15% threshold seems reasonable (95th percentile = {p95:.3f})")
        else:
            print(f"⚠️ Current 15% threshold may be too strict!")
            print(f"   95th percentile CV = {p95:.3f}")
            print(f"   Consider threshold around {p95:.3f} (95th percentile)")
            print(f"   Or {p90:.3f} (90th percentile) for stricter detection")
        
        print(f"\nMedian CV = {median:.3f} (typical biological variation)")
        
        return experiment_results, all_embryo_cvs, temporal_stats
    else:
        print("❌ No embryos with sufficient temporal data found!")
        print("\nPossible reasons:")
        print("1. Most videos are single-frame (seed frames only)")
        print("2. Area data not available in segmentation")
        print("3. Embryos not tracked across multiple frames")
        return {}, [], temporal_stats

if __name__ == "__main__":
    import sys
    
    gsam_path = "data/segmentation/grounded_sam_segmentations.json"
    target_experiments = ["20240418", "20250305"]
    
    if len(sys.argv) > 1:
        gsam_path = sys.argv[1]
    if len(sys.argv) > 2:
        target_experiments = sys.argv[2].split(",")
    
    improved_variance_analysis(gsam_path, target_experiments)