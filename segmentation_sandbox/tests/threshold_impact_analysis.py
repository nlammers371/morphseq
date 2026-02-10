#!/usr/bin/env python3
"""
Detailed Threshold Impact Analysis

Analyze how different CV thresholds would affect flagging rates and identify
representative embryos for video generation.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def load_embryo_variance_data(gsam_path: str, target_experiments: list = None):
    """Load embryo variance data from GSAM file."""
    print(f"📂 Loading GSAM data from {gsam_path}")
    
    with open(gsam_path, 'r') as f:
        gsam_data = json.load(f)
    
    experiments = gsam_data.get("experiments", {})
    if target_experiments:
        experiments = {k: v for k, v in experiments.items() if k in target_experiments}
    
    embryo_data = []  # List of {embryo_info, cv, areas, video_id}
    
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
            
            # Calculate CV for embryos with multiple frames
            for embryo_id, frame_data in embryo_frames.items():
                if len(frame_data) >= 3:  # Minimum for CV calculation
                    areas = [area for _, area in frame_data]
                    image_ids = [img_id for img_id, _ in frame_data]
                    
                    mean_area = np.mean(areas)
                    std_area = np.std(areas)
                    cv = std_area / mean_area if mean_area > 0 else 0
                    
                    embryo_data.append({
                        "experiment_id": exp_id,
                        "video_id": video_id,
                        "embryo_id": embryo_id,
                        "cv": cv,
                        "mean_area": mean_area,
                        "std_area": std_area,
                        "areas": areas,
                        "image_ids": image_ids,
                        "frame_count": len(areas),
                        "min_area": min(areas),
                        "max_area": max(areas)
                    })
    
    return embryo_data

def analyze_threshold_impact(embryo_data: List[Dict], thresholds: List[float]):
    """Analyze impact of different CV thresholds."""
    
    print("\n" + "="*80)
    print("DETAILED THRESHOLD IMPACT ANALYSIS")
    print("="*80)
    
    total_embryos = len(embryo_data)
    print(f"Total embryos analyzed: {total_embryos}")
    
    # Group by experiment
    by_experiment = defaultdict(list)
    for embryo in embryo_data:
        by_experiment[embryo["experiment_id"]].append(embryo)
    
    print(f"Experiments: {list(by_experiment.keys())}")
    print()
    
    # Analyze each threshold
    for threshold in thresholds:
        print(f"THRESHOLD: {threshold*100:.0f}% CV")
        print("-" * 40)
        
        # Overall statistics
        flagged_overall = [e for e in embryo_data if e["cv"] > threshold]
        flagged_count = len(flagged_overall)
        flagged_pct = 100 * flagged_count / total_embryos
        
        print(f"Overall: {flagged_count}/{total_embryos} embryos flagged ({flagged_pct:.1f}%)")
        
        # Per-experiment breakdown
        for exp_id, exp_embryos in by_experiment.items():
            exp_flagged = [e for e in exp_embryos if e["cv"] > threshold]
            exp_flagged_count = len(exp_flagged)
            exp_total = len(exp_embryos)
            exp_flagged_pct = 100 * exp_flagged_count / exp_total
            
            print(f"  {exp_id}: {exp_flagged_count}/{exp_total} flagged ({exp_flagged_pct:.1f}%)")
        
        print()
    
    return by_experiment

def find_representative_embryos(embryo_data: List[Dict], num_examples: int = 5):
    """Find representative embryos across CV ranges for video generation."""
    
    print("REPRESENTATIVE EMBRYOS FOR VIDEO GENERATION")
    print("="*80)
    
    # Sort embryos by CV
    sorted_embryos = sorted(embryo_data, key=lambda x: x["cv"])
    
    # Define CV ranges and their labels
    cv_ranges = [
        ("normal", 0.0, 0.15, "Normal variation (not flagged by current threshold)"),
        ("borderline", 0.15, 0.25, "Currently flagged but likely normal"),
        ("moderate", 0.25, 0.30, "Moderate variation (borderline cases)"),
        ("high", 0.30, 0.35, "High variation (likely true issues)"),
        ("extreme", 0.35, 1.0, "Extreme variation (definitely problematic)")
    ]
    
    representative_embryos = {}
    
    for category, min_cv, max_cv, description in cv_ranges:
        # Find embryos in this CV range
        range_embryos = [e for e in sorted_embryos if min_cv <= e["cv"] < max_cv]
        
        print(f"\n{category.upper()} ({min_cv*100:.0f}%-{max_cv*100:.0f}% CV)")
        print(f"Description: {description}")
        print(f"Available embryos: {len(range_embryos)}")
        
        if range_embryos:
            # Pick a representative embryo (median CV in range)
            median_idx = len(range_embryos) // 2
            representative = range_embryos[median_idx]
            representative_embryos[category] = representative
            
            print(f"Selected embryo:")
            print(f"  Experiment: {representative['experiment_id']}")
            print(f"  Video: {representative['video_id']}")
            print(f"  Embryo: {representative['embryo_id']}")
            print(f"  CV: {representative['cv']:.3f}")
            print(f"  Mean area: {representative['mean_area']:.1f}")
            print(f"  Frames: {representative['frame_count']}")
            
            # Show area variation
            areas = representative['areas']
            print(f"  Area range: {min(areas):.0f} - {max(areas):.0f} pixels")
            print(f"  Area variation: {representative['std_area']:.1f} ± std")
        else:
            print("  ❌ No embryos found in this range")
    
    return representative_embryos

def generate_threshold_recommendations(embryo_data: List[Dict]):
    """Generate threshold recommendations based on data."""
    
    print("\nTHRESHOLD RECOMMENDATIONS")
    print("="*80)
    
    cvs = [e["cv"] for e in embryo_data]
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 99]
    print("CV Distribution Percentiles:")
    for p in percentiles:
        val = np.percentile(cvs, p)
        print(f"  {p}th percentile: {val:.3f} ({val*100:.1f}%)")
    
    print()
    
    # Current situation
    current_threshold = 0.15
    current_flagged = sum(1 for cv in cvs if cv > current_threshold)
    current_pct = 100 * current_flagged / len(cvs)
    
    print(f"CURRENT SITUATION (15% threshold):")
    print(f"  Flags {current_flagged}/{len(cvs)} embryos ({current_pct:.1f}%)")
    print(f"  This is likely TOO MANY false positives")
    print()
    
    # Recommendations
    p90 = np.percentile(cvs, 90)
    p95 = np.percentile(cvs, 95)
    median = np.median(cvs)
    
    print("RECOMMENDATIONS:")
    print()
    
    print(f"1. CONSERVATIVE (90th percentile): {p90:.3f} ({p90*100:.1f}%)")
    conservative_flagged = sum(1 for cv in cvs if cv > p90)
    conservative_pct = 100 * conservative_flagged / len(cvs)
    print(f"   Would flag: {conservative_flagged}/{len(cvs)} embryos ({conservative_pct:.1f}%)")
    print(f"   Good for: Catching clear outliers while minimizing false positives")
    print()
    
    print(f"2. STRICT (95th percentile): {p95:.3f} ({p95*100:.1f}%)")
    strict_flagged = sum(1 for cv in cvs if cv > p95)
    strict_pct = 100 * strict_flagged / len(cvs)
    print(f"   Would flag: {strict_flagged}/{len(cvs)} embryos ({strict_pct:.1f}%)")
    print(f"   Good for: Only flagging the most extreme cases")
    print()
    
    print(f"3. BIOLOGICAL BASELINE: {median:.3f} ({median*100:.1f}%)")
    print(f"   This is the median - typical biological variation")
    print(f"   Anything much above this (2-3x) is likely a real issue")
    print()
    
    # Specific recommendations
    recommended_threshold = p90
    print(f"RECOMMENDED NEW THRESHOLD: {recommended_threshold:.3f} ({recommended_threshold*100:.1f}%)")
    print(f"Rationale:")
    print(f"  - Reduces false positives from {current_pct:.1f}% to {conservative_pct:.1f}%")
    print(f"  - Still catches embryos with >90th percentile variation")
    print(f"  - Based on actual data distribution, not arbitrary 15%")

def main():
    # Configuration
    gsam_path = "data/segmentation/grounded_sam_segmentations.json"
    target_experiments = ["20240418", "20250305"]
    
    # Thresholds to analyze
    thresholds = [0.15, 0.20, 0.25, 0.27, 0.30, 0.32, 0.35]  # Current + alternatives
    
    print("🎯 THRESHOLD IMPACT ANALYSIS")
    print(f"Data source: {gsam_path}")
    print(f"Target experiments: {target_experiments}")
    print()
    
    # Load data
    embryo_data = load_embryo_variance_data(gsam_path, target_experiments)
    
    if not embryo_data:
        print("❌ No embryo data found!")
        return
    
    # Analyze threshold impact
    by_experiment = analyze_threshold_impact(embryo_data, thresholds)
    
    # Find representative embryos
    representative_embryos = find_representative_embryos(embryo_data)
    
    # Generate recommendations
    generate_threshold_recommendations(embryo_data)
    
    # Save representative embryos for video generation
    output_path = Path("threshold_analysis_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "representative_embryos": representative_embryos,
            "total_embryos_analyzed": len(embryo_data),
            "experiments_analyzed": list(by_experiment.keys()),
            "analysis_timestamp": str(Path(__file__).stat().st_mtime)
        }, f, indent=2)
    
    print(f"\n💾 Representative embryos saved to: {output_path}")
    print("🎬 Use this data to generate threshold comparison videos")

if __name__ == "__main__":
    main()