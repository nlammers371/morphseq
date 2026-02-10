#!/usr/bin/env python3
"""
Analyze Segmentation Variability from GSAM Data

This script extracts area timeseries from the GSAM segmentation file and analyzes
variance to validate and optimize the QC thresholds for segmentation variability detection.

Usage:
    python analyze_segmentation_variance.py --gsam data/segmentation/grounded_sam_segmentations.json
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime

# Optional imports
try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# Add project root to path for imports
import sys
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

# Import mask utils for area calculation
from scripts.utils.mask_utils import decode_mask_rle


class SegmentationVarianceAnalyzer:
    """Analyze segmentation variance from GSAM data."""
    
    def __init__(self, gsam_path: str, target_experiments: List[str] = None):
        """
        Initialize analyzer.
        
        Args:
            gsam_path: Path to GSAM segmentation JSON file
            target_experiments: List of experiment IDs to analyze (default: all)
        """
        self.gsam_path = Path(gsam_path)
        self.target_experiments = target_experiments or []
        
        # Load GSAM data
        print(f"📂 Loading GSAM data from {self.gsam_path}")
        with open(self.gsam_path, 'r') as f:
            self.gsam_data = json.load(f)
        
        print(f"✅ Loaded GSAM data")
        
        # Storage for analysis results
        self.embryo_data = {}  # embryo_id -> {areas, cv, metadata}
        self.experiment_stats = {}  # exp_id -> statistics
        
    def extract_embryo_areas(self) -> Dict:
        """
        Extract area timeseries for all embryos.
        
        Returns:
            Dict of embryo data with area timeseries and metadata
        """
        print("🔍 Extracting embryo area timeseries...")
        
        experiments = self.gsam_data.get("experiments", {})
        total_embryos_processed = 0
        
        # Filter experiments first
        if self.target_experiments:
            experiments = {k: v for k, v in experiments.items() if k in self.target_experiments}
            print(f"🎯 Filtered to {len(experiments)} target experiments: {list(experiments.keys())}")
        else:
            print(f"📊 Processing all {len(experiments)} experiments")
        
        for exp_idx, (exp_id, exp_data) in enumerate(experiments.items()):
            print(f"📁 Processing experiment {exp_idx+1}/{len(experiments)}: {exp_id}")
            
            videos = exp_data.get("videos", {})
            for video_idx, (video_id, video_data) in enumerate(videos.items()):
                if video_idx % 5 == 0:  # Progress every 5 videos
                    print(f"   📹 Video {video_idx+1}/{len(videos)}: {video_id}")
                
                # Group embryos by ID across frames
                embryo_frames = defaultdict(dict)  # embryo_id -> {image_id: area}
                
                for image_id, image_data in video_data.get("image_ids", {}).items():
                    for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                        snip_id = embryo_data.get("snip_id")
                        if not snip_id:
                            continue
                            
                        # Calculate area from segmentation
                        area = embryo_data.get("area")
                        if area is None:
                            segmentation = embryo_data.get("segmentation")
                            if segmentation and segmentation.get("format") in ["rle", "rle_base64"]:
                                try:
                                    mask = decode_mask_rle(segmentation)
                                    area = float(np.sum(mask))
                                except Exception as e:
                                    print(f"⚠️ Error decoding mask for {snip_id}: {e}")
                                    continue
                        
                        if area is not None and area > 0:
                            embryo_frames[embryo_id][image_id] = area
                
                # Process each embryo's timeseries
                for embryo_id, frame_areas in embryo_frames.items():
                    if len(frame_areas) >= 3:  # Minimum frames for CV calculation
                        areas = list(frame_areas.values())
                        image_ids = list(frame_areas.keys())
                        
                        # Calculate statistics
                        mean_area = np.mean(areas)
                        std_area = np.std(areas)
                        cv = std_area / mean_area if mean_area > 0 else 0
                        
                        # Store embryo data
                        embryo_key = f"{exp_id}_{video_id}_{embryo_id}"
                        self.embryo_data[embryo_key] = {
                            "experiment_id": exp_id,
                            "video_id": video_id,
                            "embryo_id": embryo_id,
                            "areas": areas,
                            "image_ids": image_ids,
                            "mean_area": mean_area,
                            "std_area": std_area,
                            "cv": cv,
                            "frame_count": len(areas),
                            "min_area": min(areas),
                            "max_area": max(areas),
                            "area_range": max(areas) - min(areas)
                        }
                        total_embryos_processed += 1
        
        print(f"✅ Processed {total_embryos_processed} embryos across {len(self.embryo_data)} unique embryo timeseries")
        return self.embryo_data
    
    def calculate_experiment_statistics(self) -> Dict:
        """Calculate statistics by experiment."""
        print("📊 Calculating experiment statistics...")
        
        for embryo_key, embryo_info in self.embryo_data.items():
            exp_id = embryo_info["experiment_id"]
            
            if exp_id not in self.experiment_stats:
                self.experiment_stats[exp_id] = {
                    "embryo_count": 0,
                    "cv_values": [],
                    "mean_areas": [],
                    "frame_counts": []
                }
            
            self.experiment_stats[exp_id]["embryo_count"] += 1
            self.experiment_stats[exp_id]["cv_values"].append(embryo_info["cv"])
            self.experiment_stats[exp_id]["mean_areas"].append(embryo_info["mean_area"])
            self.experiment_stats[exp_id]["frame_counts"].append(embryo_info["frame_count"])
        
        # Calculate summary statistics
        for exp_id, stats in self.experiment_stats.items():
            cv_values = np.array(stats["cv_values"])
            stats.update({
                "cv_mean": np.mean(cv_values),
                "cv_median": np.median(cv_values),
                "cv_std": np.std(cv_values),
                "cv_p90": np.percentile(cv_values, 90),
                "cv_p95": np.percentile(cv_values, 95),
                "cv_p99": np.percentile(cv_values, 99),
                "cv_max": np.max(cv_values),
                "embryos_above_15pct": np.sum(cv_values > 0.15),
                "embryos_above_20pct": np.sum(cv_values > 0.20),
                "embryos_above_25pct": np.sum(cv_values > 0.25)
            })
        
        return self.experiment_stats
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report."""
        report = []
        report.append("=" * 60)
        report.append("SEGMENTATION VARIANCE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"GSAM file: {self.gsam_path}")
        report.append(f"Total embryos analyzed: {len(self.embryo_data)}")
        report.append("")
        
        # Overall statistics
        all_cvs = [info["cv"] for info in self.embryo_data.values()]
        if all_cvs:
            report.append("OVERALL CV STATISTICS:")
            report.append(f"  Mean CV: {np.mean(all_cvs):.3f}")
            report.append(f"  Median CV: {np.median(all_cvs):.3f}")
            report.append(f"  90th percentile: {np.percentile(all_cvs, 90):.3f}")
            report.append(f"  95th percentile: {np.percentile(all_cvs, 95):.3f}")
            report.append(f"  99th percentile: {np.percentile(all_cvs, 99):.3f}")
            report.append(f"  Max CV: {np.max(all_cvs):.3f}")
            report.append("")
            
            # Current threshold analysis
            report.append("CURRENT THRESHOLD ANALYSIS (15% CV):")
            flagged_count = sum(1 for cv in all_cvs if cv > 0.15)
            report.append(f"  Embryos flagged: {flagged_count}/{len(all_cvs)} ({100*flagged_count/len(all_cvs):.1f}%)")
            report.append("")
        
        # Per-experiment statistics
        if self.experiment_stats:
            report.append("PER-EXPERIMENT STATISTICS:")
            for exp_id, stats in self.experiment_stats.items():
                report.append(f"  {exp_id}:")
                report.append(f"    Embryos: {stats['embryo_count']}")
                report.append(f"    Mean CV: {stats['cv_mean']:.3f}")
                report.append(f"    Median CV: {stats['cv_median']:.3f}")
                report.append(f"    95th percentile CV: {stats['cv_p95']:.3f}")
                report.append(f"    Flagged by 15% threshold: {stats['embryos_above_15pct']} ({100*stats['embryos_above_15pct']/stats['embryo_count']:.1f}%)")
                report.append("")
        
        return "\n".join(report)
    
    def create_plots(self, output_dir: str = "variance_analysis_plots"):
        """Generate analysis plots."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📊 Creating plots in {output_path}")
        
        # Set up plotting style
        plt.style.use('default')
        if _HAS_SEABORN:
            sns.set_palette("husl")
        
        # 1. CV Distribution histogram
        all_cvs = [info["cv"] for info in self.embryo_data.values()]
        if all_cvs:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_cvs, bins=50, alpha=0.7, density=True)
            ax.axvline(0.15, color='red', linestyle='--', label='Current threshold (15%)')
            ax.axvline(np.percentile(all_cvs, 95), color='orange', linestyle='--', label='95th percentile')
            ax.set_xlabel('Coefficient of Variation (CV)')
            ax.set_ylabel('Density')
            ax.set_title('Distribution of Segmentation CV Values')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / "cv_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Per-experiment CV comparison
        if len(self.experiment_stats) > 1:
            exp_data = []
            for exp_id, stats in self.experiment_stats.items():
                for cv in stats["cv_values"]:
                    exp_data.append({"experiment": exp_id, "cv": cv})
            
            if exp_data:
                df = pd.DataFrame(exp_data)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                if _HAS_SEABORN:
                    sns.boxplot(data=df, x="experiment", y="cv", ax=ax)
                else:
                    # Fallback boxplot without seaborn
                    experiments = df["experiment"].unique()
                    cv_data = [df[df["experiment"] == exp]["cv"].values for exp in experiments]
                    ax.boxplot(cv_data, labels=experiments)
                ax.axhline(0.15, color='red', linestyle='--', label='Current threshold (15%)')
                ax.set_ylabel('Coefficient of Variation (CV)')
                ax.set_title('CV Distribution by Experiment')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_path / "cv_by_experiment.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Area vs CV scatter plot
        areas = [info["mean_area"] for info in self.embryo_data.values()]
        cvs = [info["cv"] for info in self.embryo_data.values()]
        
        if areas and cvs:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(areas, cvs, alpha=0.6)
            ax.axhline(0.15, color='red', linestyle='--', label='Current threshold (15%)')
            ax.set_xlabel('Mean Area (pixels)')
            ax.set_ylabel('Coefficient of Variation (CV)')
            ax.set_title('Mean Area vs CV')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path / "area_vs_cv.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. High variance examples (timeseries plots)
        high_cv_embryos = sorted(self.embryo_data.items(), key=lambda x: x[1]["cv"], reverse=True)[:6]
        
        if high_cv_embryos:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            for i, (embryo_key, embryo_info) in enumerate(high_cv_embryos[:6]):
                ax = axes[i]
                areas = embryo_info["areas"]
                ax.plot(range(len(areas)), areas, 'o-')
                ax.set_title(f'{embryo_info["experiment_id"]}\nCV: {embryo_info["cv"]:.3f}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Area (pixels)')
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('Examples of High CV Embryos')
            plt.tight_layout()
            plt.savefig(output_path / "high_cv_examples.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Plots saved to {output_path}")
    
    def export_data(self, output_dir: str = "variance_analysis_data"):
        """Export raw data for further analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export embryo data as CSV
        embryo_df_data = []
        for embryo_key, info in self.embryo_data.items():
            embryo_df_data.append({
                "embryo_key": embryo_key,
                "experiment_id": info["experiment_id"],
                "video_id": info["video_id"],
                "embryo_id": info["embryo_id"],
                "cv": info["cv"],
                "mean_area": info["mean_area"],
                "std_area": info["std_area"],
                "frame_count": info["frame_count"],
                "min_area": info["min_area"],
                "max_area": info["max_area"],
                "area_range": info["area_range"]
            })
        
        embryo_df = pd.DataFrame(embryo_df_data)
        embryo_df.to_csv(output_path / "embryo_variance_data.csv", index=False)
        
        # Export experiment statistics
        exp_df_data = []
        for exp_id, stats in self.experiment_stats.items():
            exp_df_data.append({
                "experiment_id": exp_id,
                "embryo_count": stats["embryo_count"],
                "cv_mean": stats["cv_mean"],
                "cv_median": stats["cv_median"],
                "cv_std": stats["cv_std"],
                "cv_p90": stats["cv_p90"],
                "cv_p95": stats["cv_p95"],
                "cv_p99": stats["cv_p99"],
                "cv_max": stats["cv_max"],
                "embryos_above_15pct": stats["embryos_above_15pct"],
                "embryos_above_20pct": stats["embryos_above_20pct"],
                "embryos_above_25pct": stats["embryos_above_25pct"]
            })
        
        exp_df = pd.DataFrame(exp_df_data)
        exp_df.to_csv(output_path / "experiment_statistics.csv", index=False)
        
        print(f"✅ Data exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze segmentation variance from GSAM data")
    parser.add_argument("--gsam", required=True, help="Path to GSAM segmentation JSON file")
    parser.add_argument("--experiments", help="Comma-separated list of experiments to analyze (default: all)")
    parser.add_argument("--output-dir", default="variance_analysis", help="Output directory for results")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    
    args = parser.parse_args()
    
    # Parse target experiments
    target_experiments = None
    if args.experiments:
        target_experiments = [exp.strip() for exp in args.experiments.split(",")]
        print(f"🎯 Analyzing specific experiments: {target_experiments}")
    else:
        print("🎯 Analyzing all experiments")
    
    # Initialize analyzer
    analyzer = SegmentationVarianceAnalyzer(args.gsam, target_experiments)
    
    # Run analysis
    analyzer.extract_embryo_areas()
    analyzer.calculate_experiment_statistics()
    
    # Generate outputs
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save report
    report = analyzer.generate_summary_report()
    print("\n" + report)
    with open(output_path / "variance_analysis_report.txt", "w") as f:
        f.write(report)
    
    # Generate plots
    if not args.no_plots:
        analyzer.create_plots(str(output_path / "plots"))
    
    # Export data
    analyzer.export_data(str(output_path / "data"))
    
    print(f"\n✅ Analysis complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()