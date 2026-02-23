#!/usr/bin/env python
"""
05_generate_annotations.py

Final stage of the MorphSeq embryo segmentation pipeline.
Generates COCO format annotations from tracking results.

This script:
1. Loads tracking results from previous stages
2. Converts to COCO format with video metadata
3. Adds temporal information and trajectory IDs
4. Generates final annotation files for analysis
5. Creates summary statistics and QC report

Input:
- Tracking results from stage 4
- Video metadata from stage 1
- Pipeline configuration

Output:
- COCO format annotations with video IDs
- Trajectory summary files
- Final QC report
- Analysis-ready data files

Usage:
    python scripts/05_generate_annotations.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_config, ensure_directory, save_json, load_json,
    setup_pipeline_logging, QCLogger
)


class COCOAnnotationGenerator:
    """
    Generates COCO format annotations with video metadata.
    """
    
    def __init__(self, config):
        """Initialize annotation generator."""
        self.config = config
        self.logger = setup_pipeline_logging(config.config, "coco_generation")
        self.qc_logger = QCLogger(config.get('paths.logs_dir'))
        
        # Get paths
        self.tracking_results_dir = config.get_intermediate_path("tracking_results")
        self.video_metadata_file = Path(config.get('paths.morphseq_well_videos')) / "video_metadata.json"
        self.final_dir = config.get('paths.final_dir')
        
        # Ensure output directory
        ensure_directory(self.final_dir)
        
        # COCO categories from config
        self.categories = config.get_coco_categories()
        
        # Initialize COCO structure
        self.coco_data = {
            "info": {
                "description": "MorphSeq Embryo Tracking Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "MorphSeq Pipeline",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom Research License",
                    "url": ""
                }
            ],
            "categories": self.categories,
            "videos": [],
            "images": [],
            "annotations": [],
            "trajectories": []  # Custom extension for temporal tracking
        }
        
        # Counters for IDs
        self.video_id_counter = 1
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.trajectory_id_counter = 1
    
    def load_video_metadata(self) -> Dict[str, Any]:
        """Load video metadata from stage 1."""
        self.logger.info("Loading video metadata...")
        
        if not self.video_metadata_file.exists():
            self.logger.error(f"Video metadata not found: {self.video_metadata_file}")
            self.qc_logger.add_global_flag(
                "MISSING_VIDEO_METADATA",
                f"Video metadata file not found: {self.video_metadata_file}"
            )
            return {}
        
        return load_json(self.video_metadata_file)
    
    def load_tracking_results(self) -> Dict[str, Any]:
        """Load tracking results from stage 4."""
        self.logger.info("Loading tracking results...")
        
        tracking_file = Path(self.tracking_results_dir) / "embryo_trajectories.json"
        
        if not tracking_file.exists():
            self.logger.error(f"Tracking results not found: {tracking_file}")
            self.qc_logger.add_global_flag(
                "MISSING_TRACKING_RESULTS",
                f"Tracking results file not found: {tracking_file}"
            )
            return {}
        
        return load_json(tracking_file)
    
    def add_video_to_coco(self, video_id: str, video_metadata: Dict[str, Any]) -> int:
        """
        Add video information to COCO format.
        
        Returns:
            Video ID number for referencing
        """
        coco_video_id = self.video_id_counter
        self.video_id_counter += 1
        
        video_entry = {
            "id": coco_video_id,
            "name": video_id,
            "file_name": Path(video_metadata["video_path"]).name,
            "width": video_metadata["resolution"][0],
            "height": video_metadata["resolution"][1],
            "length": video_metadata["valid_frames"],
            "fps": video_metadata["video_fps"],
            "creation_time": video_metadata["creation_time"],
            "source_images": len(video_metadata["source_images"]),
            "jpeg_conversion": video_metadata.get("jpeg_conversion", False),
            "jpeg_quality": video_metadata.get("jpeg_quality", None)
        }
        
        self.coco_data["videos"].append(video_entry)
        return coco_video_id
    
    def add_frame_to_coco(self, video_coco_id: int, frame_idx: int, 
                         video_metadata: Dict[str, Any]) -> int:
        """
        Add frame (image) information to COCO format.
        
        Returns:
            Image ID number for referencing
        """
        coco_image_id = self.image_id_counter
        self.image_id_counter += 1
        
        image_entry = {
            "id": coco_image_id,
            "video_id": video_coco_id,
            "frame_id": frame_idx,
            "width": video_metadata["resolution"][0],
            "height": video_metadata["resolution"][1],
            "file_name": f"frame_{frame_idx:04d}.jpg",  # Virtual frame filename
            "timestamp": frame_idx / video_metadata["video_fps"]  # Time in seconds
        }
        
        self.coco_data["images"].append(image_entry)
        return coco_image_id
    
    def add_detection_to_coco(self, image_id: int, detection: Dict[str, Any], 
                            trajectory_id: Optional[int] = None) -> int:
        """
        Add detection annotation to COCO format.
        
        Returns:
            Annotation ID number
        """
        coco_annotation_id = self.annotation_id_counter
        self.annotation_id_counter += 1
        
        # Convert mask to COCO polygon format if needed
        # For now, using bounding box format
        bbox = detection.get("bbox", [0, 0, 0, 0])  # [x, y, width, height]
        area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
        
        annotation_entry = {
            "id": coco_annotation_id,
            "image_id": image_id,
            "category_id": detection.get("category_id", 1),  # Default to embryo
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "trajectory_id": trajectory_id,  # Custom field for tracking
            "confidence": detection.get("confidence", 1.0),
            "detection_method": detection.get("method", "unknown"),
            # Add mask if available
            "segmentation": detection.get("segmentation", [])
        }
        
        # Add tracking-specific metadata
        if "tracking_metadata" in detection:
            annotation_entry["tracking_metadata"] = detection["tracking_metadata"]
        
        self.coco_data["annotations"].append(annotation_entry)
        return coco_annotation_id
    
    def add_trajectory_to_coco(self, trajectory_data: Dict[str, Any]) -> int:
        """
        Add trajectory information to COCO format.
        
        Returns:
            Trajectory ID number
        """
        coco_trajectory_id = self.trajectory_id_counter
        self.trajectory_id_counter += 1
        
        trajectory_entry = {
            "id": coco_trajectory_id,
            "video_id": trajectory_data.get("video_id"),
            "category_id": trajectory_data.get("category_id", 1),
            "start_frame": trajectory_data.get("start_frame"),
            "end_frame": trajectory_data.get("end_frame"),
            "length": trajectory_data.get("length"),
            "annotations": trajectory_data.get("annotation_ids", []),
            "death_frame": trajectory_data.get("death_frame"),
            "is_complete": trajectory_data.get("is_complete", False),
            "quality_flags": trajectory_data.get("quality_flags", []),
            "centroid_trajectory": trajectory_data.get("centroid_trajectory", []),
            "area_trajectory": trajectory_data.get("area_trajectory", [])
        }
        
        self.coco_data["trajectories"].append(trajectory_entry)
        return coco_trajectory_id
    
    def process_video_tracking(self, video_id: str, video_metadata: Dict[str, Any], 
                             tracking_data: Dict[str, Any]):
        """Process tracking data for a single video."""
        self.logger.info(f"Processing video: {video_id}")
        
        # Add video to COCO
        coco_video_id = self.add_video_to_coco(video_id, video_metadata)
        
        # Process frames and detections
        video_tracking = tracking_data.get(video_id, {})
        frame_detections = video_tracking.get("frame_detections", {})
        trajectories = video_tracking.get("trajectories", {})
        
        # Create frame entries and collect detections
        frame_to_image_id = {}
        
        for frame_idx in range(video_metadata["valid_frames"]):
            image_id = self.add_frame_to_coco(coco_video_id, frame_idx, video_metadata)
            frame_to_image_id[frame_idx] = image_id
            
            # Add detections for this frame
            if str(frame_idx) in frame_detections:
                for detection in frame_detections[str(frame_idx)]:
                    trajectory_id = detection.get("trajectory_id")
                    self.add_detection_to_coco(image_id, detection, trajectory_id)
        
        # Add trajectory information
        for traj_id, traj_data in trajectories.items():
            traj_data["video_id"] = coco_video_id
            self.add_trajectory_to_coco(traj_data)
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset."""
        stats = {
            "dataset_summary": {
                "total_videos": len(self.coco_data["videos"]),
                "total_frames": len(self.coco_data["images"]),
                "total_annotations": len(self.coco_data["annotations"]),
                "total_trajectories": len(self.coco_data["trajectories"])
            },
            "category_breakdown": {},
            "trajectory_statistics": {
                "mean_length": 0,
                "median_length": 0,
                "complete_trajectories": 0,
                "incomplete_trajectories": 0
            },
            "quality_flags": {}
        }
        
        # Category breakdown
        for category in self.categories:
            cat_id = category["id"]
            cat_name = category["name"]
            count = sum(1 for ann in self.coco_data["annotations"] 
                       if ann["category_id"] == cat_id)
            stats["category_breakdown"][cat_name] = count
        
        # Trajectory statistics
        if self.coco_data["trajectories"]:
            lengths = [traj["length"] for traj in self.coco_data["trajectories"] if traj["length"]]
            if lengths:
                stats["trajectory_statistics"]["mean_length"] = np.mean(lengths)
                stats["trajectory_statistics"]["median_length"] = np.median(lengths)
            
            stats["trajectory_statistics"]["complete_trajectories"] = sum(
                1 for traj in self.coco_data["trajectories"] if traj.get("is_complete", False)
            )
            stats["trajectory_statistics"]["incomplete_trajectories"] = (
                len(self.coco_data["trajectories"]) - 
                stats["trajectory_statistics"]["complete_trajectories"]
            )
        
        return stats
    
    def run(self) -> bool:
        """
        Run the COCO annotation generation.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.start_pipeline()
        self.logger.start_stage("COCO Annotation Generation")
        
        try:
            # Load inputs
            video_metadata = self.load_video_metadata()
            tracking_results = self.load_tracking_results()
            
            if not video_metadata or not tracking_results:
                self.logger.error("Missing required input data")
                return False
            
            # Process each video
            for video_id, metadata in video_metadata.items():
                if video_id in tracking_results:
                    self.process_video_tracking(video_id, metadata, tracking_results)
                else:
                    self.logger.warning(f"No tracking results for video: {video_id}")
                    self.qc_logger.add_video_flag(
                        metadata["video_path"], "NO_TRACKING_RESULTS",
                        f"No tracking results found for video {video_id}"
                    )
            
            # Generate summary statistics
            summary_stats = self.generate_summary_statistics()
            
            # Save outputs
            coco_file = Path(self.final_dir) / "embryo_tracking_coco.json"
            save_json(self.coco_data, coco_file)
            
            stats_file = Path(self.final_dir) / "dataset_statistics.json"
            save_json(summary_stats, stats_file)
            
            # Save trajectory CSV for analysis
            self.save_trajectory_csv()
            
            # Save QC report
            self.qc_logger.save_qc_report()
            
            self.logger.end_stage("COCO Annotation Generation",
                                processed_videos=len(video_metadata),
                                total_annotations=len(self.coco_data["annotations"]),
                                total_trajectories=len(self.coco_data["trajectories"]))
            
            self.logger.end_pipeline(
                output_file=str(coco_file),
                total_annotations=len(self.coco_data["annotations"])
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False
    
    def save_trajectory_csv(self):
        """Save trajectory data as CSV for analysis."""
        if not self.coco_data["trajectories"]:
            return
        
        # Convert trajectories to DataFrame
        traj_data = []
        for traj in self.coco_data["trajectories"]:
            traj_data.append({
                "trajectory_id": traj["id"],
                "video_id": traj["video_id"],
                "category_id": traj["category_id"],
                "start_frame": traj["start_frame"],
                "end_frame": traj["end_frame"],
                "length": traj["length"],
                "death_frame": traj.get("death_frame"),
                "is_complete": traj["is_complete"],
                "num_quality_flags": len(traj.get("quality_flags", []))
            })
        
        df = pd.DataFrame(traj_data)
        csv_file = Path(self.final_dir) / "embryo_trajectories.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Saved trajectory CSV: {csv_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate COCO annotations from tracking results")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create and run generator
    generator = COCOAnnotationGenerator(config)
    
    success = generator.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
