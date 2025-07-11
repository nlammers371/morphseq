#!/usr/bin/env python3
"""
Stage 4a: Quality Filtering for GroundedDINO Annotations
=======================================================

Simple, focused script to filter GroundedDINO annotations for high quality.
Applies filters step-by-step with tracking.

Core functionality:
1. Load annotations using GroundedDinoAnnotations class
2. Generate confidence histogram
3. Apply confidence threshold filter
4. Remove overlapping detections (IoU filter)
5. Save filtered annotations

Usage:
    python scripts/04a_quality_filtering.py \
      --annotations gdino_annotations.json \
      --output gdino_high_quality_annotations.json \
      --confidence-threshold 0.4
"""


import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
SANDBOX_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SANDBOX_ROOT))

# Import utilities
from scripts.utils.grounded_sam_utils import GroundedDinoAnnotations, calculate_detection_iou


class QualityFilter:
    """
    Simple quality filter that applies filters step-by-step with tracking.
    """
    
    def __init__(self, annotations_manager: GroundedDinoAnnotations, prompt: str = "individual embryo"):
        self.annotations_manager = annotations_manager
        self.prompt = prompt
        self.removal_tracking = []  # Track what was removed by which filter
        
    def analyze_confidence_distribution(self, save_histogram: Optional[str] = None) -> Dict:
        """
        Analyze confidence score distribution and generate histogram.
        
        Args:
            save_histogram: Optional path to save histogram
            
        Returns:
            Statistics dictionary
        """
        print(f"📊 Analyzing confidence distribution for prompt: '{self.prompt}'")
        
        # Collect all confidence scores
        confidence_scores = []
        total_detections = 0
        
        for image_id, image_data in self.annotations_manager.annotations.get("images", {}).items():
            for annotation in image_data.get("annotations", []):
                if annotation.get("prompt") == self.prompt:
                    detections = annotation.get("detections", [])
                    total_detections += len(detections)
                    
                    for detection in detections:
                        confidence_scores.append(detection.get("confidence", 0))
        
        if not confidence_scores:
            print(f"❌ No detections found for prompt: {self.prompt}")
            return {}
        
        # Calculate statistics
        scores_array = np.array(confidence_scores)
        stats = {
            "total_detections": len(confidence_scores),
            "mean": float(np.mean(scores_array)),
            "median": float(np.median(scores_array)),
            "std": float(np.std(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "q25": float(np.percentile(scores_array, 25)),
            "q75": float(np.percentile(scores_array, 75)),
            "q90": float(np.percentile(scores_array, 90)),
            "q95": float(np.percentile(scores_array, 95))
        }
        
        print(f"   Total detections: {stats['total_detections']}")
        print(f"   Mean confidence: {stats['mean']:.3f}")
        print(f"   Median confidence: {stats['median']:.3f}")
        print(f"   90th percentile: {stats['q90']:.3f}")
        print(f"   95th percentile: {stats['q95']:.3f}")
        
        # Generate histogram
        if save_histogram or True:  # Always show histogram
            self._create_histogram(confidence_scores, stats, save_histogram)
        
        return stats
    
    def _create_histogram(self, confidence_scores: List[float], stats: Dict, save_path: Optional[str]):
        """Create and display confidence histogram."""
        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(confidence_scores, bins=50, alpha=0.7, edgecolor='black')
        
        # Add key statistics lines
        plt.axvline(stats["mean"], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.3f}')
        plt.axvline(stats["median"], color='green', linestyle='--', label=f'Median: {stats["median"]:.3f}')
        plt.axvline(0.4, color='orange', linestyle='-', label='Example Threshold: 0.4')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title(f'Confidence Distribution for "{self.prompt}"\nTotal: {stats["total_detections"]} detections')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Histogram saved to: {save_path}")
        
        plt.show()
    
    def apply_confidence_filter(self, threshold: float) -> int:
        """
        Apply confidence threshold filter.
        
        Args:
            threshold: Minimum confidence score to keep
            
        Returns:
            Number of detections removed
        """
        print(f"\n🔍 Filter 1: Applying confidence threshold ≥ {threshold}")
        
        removed_detections = []
        total_removed = 0
        
        for image_id, image_data in self.annotations_manager.annotations.get("images", {}).items():
            for annotation in image_data.get("annotations", []):
                if annotation.get("prompt") == self.prompt:
                    original_detections = annotation.get("detections", [])
                    
                    # Separate kept vs removed detections
                    kept_detections = []
                    removed_from_image = []
                    
                    for detection in original_detections:
                        confidence = detection.get("confidence", 0)
                        if confidence >= threshold:
                            kept_detections.append(detection)
                        else:
                            removed_from_image.append({
                                "detection": detection,
                                "reason": f"confidence {confidence:.3f} < {threshold}"
                            })
                            total_removed += 1
                    
                    # Update annotation
                    annotation["detections"] = kept_detections
                    annotation["num_detections"] = len(kept_detections)
                    
                    # Track removals
                    if removed_from_image:
                        removed_detections.append({
                            "image_id": image_id,
                            "filter": "confidence_threshold",
                            "threshold": threshold,
                            "removed_count": len(removed_from_image),
                            "removed_detections": removed_from_image
                        })
        
        # Store removal tracking
        self.removal_tracking.extend(removed_detections)
        
        print(f"   Removed {total_removed} detections below confidence {threshold}")
        return total_removed
    
    def apply_iou_filter(self, iou_threshold: float = 0.5) -> int:
        """
        Remove overlapping detections using IoU.
        
        Args:
            iou_threshold: IoU threshold for considering detections as overlapping
            
        Returns:
            Number of detections removed
        """
        print(f"\n🔍 Filter 2: Removing overlapping detections (IoU ≥ {iou_threshold})")
        
        removed_detections = []
        total_removed = 0
        
        for image_id, image_data in self.annotations_manager.annotations.get("images", {}).items():
            for annotation in image_data.get("annotations", []):
                if annotation.get("prompt") == self.prompt:
                    detections = annotation.get("detections", [])
                    
                    if len(detections) <= 1:
                        continue  # No overlap possible
                    
                    # Apply IoU-based duplicate removal
                    kept_detections, removed_from_image = self._remove_overlapping_detections(
                        detections, iou_threshold
                    )
                    
                    # Update annotation
                    annotation["detections"] = kept_detections
                    annotation["num_detections"] = len(kept_detections)
                    
                    # Track removals
                    if removed_from_image:
                        total_removed += len(removed_from_image)
                        removed_detections.append({
                            "image_id": image_id,
                            "filter": "iou_overlap",
                            "threshold": iou_threshold,
                            "removed_count": len(removed_from_image),
                            "removed_detections": removed_from_image
                        })
        
        # Store removal tracking
        self.removal_tracking.extend(removed_detections)
        
        print(f"   Removed {total_removed} overlapping detections")
        return total_removed
    
    def _remove_overlapping_detections(self, detections: List[Dict], iou_threshold: float) -> Tuple[List[Dict], List[Dict]]:
        """
        Remove overlapping detections, keeping highest confidence.
        
        Returns:
            Tuple of (kept_detections, removed_detections)
        """
        if len(detections) <= 1:
            return detections, []
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        kept = []
        removed = []
        
        for detection in sorted_detections:
            is_duplicate = False
            
            for kept_detection in kept:
                iou = calculate_detection_iou(detection['box_xywh'], kept_detection['box_xywh'])
                if iou > iou_threshold:
                    is_duplicate = True
                    removed.append({
                        "detection": detection,
                        "reason": f"IoU {iou:.3f} with higher confidence detection"
                    })
                    break
            
            if not is_duplicate:
                kept.append(detection)
        
        return kept, removed
    
    def get_filtering_summary(self) -> Dict:
        """Get summary of all filtering operations."""
        # Count removals by filter type
        filter_stats = {}
        for removal in self.removal_tracking:
            filter_name = removal["filter"]
            if filter_name not in filter_stats:
                filter_stats[filter_name] = {"images": 0, "detections": 0}
            
            filter_stats[filter_name]["images"] += 1
            filter_stats[filter_name]["detections"] += removal["removed_count"]
        
        return {
            "total_filters_applied": len(set(r["filter"] for r in self.removal_tracking)),
            "filter_breakdown": filter_stats,
            "total_removals": sum(r["removed_count"] for r in self.removal_tracking),
            "images_affected": len(set(r["image_id"] for r in self.removal_tracking)),
            "detailed_tracking": self.removal_tracking
        }


def main():
    parser = argparse.ArgumentParser(description="Filter GroundedDINO annotations for high quality")
    parser.add_argument("--annotations", required=True, 
                       help="Path to gdino_annotations.json")
    parser.add_argument("--output", required=True,
                       help="Path to output gdino_high_quality_annotations.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.4,
                       help="Confidence threshold (default: 0.4)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for duplicate removal (default: 0.5)")
    parser.add_argument("--prompt", default="individual embryo",
                       help="Prompt to filter (default: 'individual embryo')")
    parser.add_argument("--save-histogram", 
                       help="Path to save confidence histogram")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Only analyze, don't apply filters")
    
    args = parser.parse_args()
    
    print("🔍 GroundedDINO Quality Filtering")
    print("=" * 40)
    print(f"Input: {args.annotations}")
    print(f"Output: {args.output}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    
    # Load annotations using GroundedDinoAnnotations class
    print(f"\n📁 Loading annotations...")
    annotations_manager = GroundedDinoAnnotations(args.annotations, verbose=True)
    
    # Initialize quality filter
    quality_filter = QualityFilter(annotations_manager, args.prompt)
    
    # Step 1: Analyze confidence distribution
    print(f"\n📊 Step 1: Analyze Confidence Distribution")
    stats = quality_filter.analyze_confidence_distribution(args.save_histogram)
    
    if not stats:
        print("❌ No detections found. Exiting.")
        return
    
    original_count = stats["total_detections"]
    
    if args.analysis_only:
        print("\n✅ Analysis complete (analysis-only mode)")
        return
    
    # Step 2: Apply confidence filter
    print(f"\n🔄 Step 2: Apply Quality Filters")
    confidence_removed = quality_filter.apply_confidence_filter(args.confidence_threshold)
    
    # Step 3: Apply IoU filter  
    iou_removed = quality_filter.apply_iou_filter(args.iou_threshold)
    
    # Step 4: Get summary and save
    print(f"\n📋 Step 3: Generate Summary")
    summary = quality_filter.get_filtering_summary()
    
    # Add summary to annotations
    annotations_manager.annotations["filtering_summary"] = {
        "original_detections": original_count,
        "confidence_threshold": args.confidence_threshold,
        "confidence_removed": confidence_removed,
        "iou_threshold": args.iou_threshold, 
        "iou_removed": iou_removed,
        "total_removed": summary["total_removals"],
        "final_detections": original_count - summary["total_removals"],
        "retention_rate": (original_count - summary["total_removals"]) / original_count if original_count > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save filtered annotations
    print(f"\n💾 Step 4: Save Filtered Annotations")
    annotations_manager.filepath = Path(args.output)  # Update output path
    annotations_manager.save()
    
    # Final summary
    final_count = original_count - summary["total_removals"]
    retention_rate = final_count / original_count * 100 if original_count > 0 else 0
    
    print(f"\n🎯 FILTERING COMPLETE!")
    print(f"=" * 30)
    print(f"Original detections: {original_count}")
    print(f"Final detections: {final_count}")
    print(f"Removed detections: {summary['total_removals']}")
    print(f"Retention rate: {retention_rate:.1f}%")
    print(f"\nFilter breakdown:")
    for filter_name, stats in summary["filter_breakdown"].items():
        print(f"  {filter_name}: {stats['detections']} detections from {stats['images']} images")
    print(f"\n📁 Filtered annotations saved to: {args.output}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/04a_quality_filtering.py \
#   --annotations data/annotation_and_masks/gdino_annotations/gdino_annotations.json \
#   --output data/annotation_and_masks/gdino_annotations/gdino_high_quality_annotations.json \
#   --confidence-threshold 0.4 \
#   --save-histogram confidence_histogram.png

# Analysis only:
# python scripts/04a_quality_filtering.py \
#   --annotations data/annotation_and_masks/gdino_annotations/gdino_annotations.json \
#   --output /tmp/dummy.json \
#   --analysis-only