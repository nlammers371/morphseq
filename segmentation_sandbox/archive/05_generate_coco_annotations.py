#!/usr/bin/env python
"""
05_generate_coco_annotations.py

Final stage of the MorphSeq embryo segmentation pipeline.
Generates COCO format annotations from tracking results.

This script:
1. Loads tracking results from the tracks directory
2. Converts tracks to COCO format annotations
3. Applies final quality filters and death detection
4. Generates final COCO JSON file with all annotations
5. Creates summary statistics and final QC report

Input:
- Tracking results from tracks/ directory
- Pipeline configuration

Output:
- final/ directory with COCO annotation files
- final_summary.json with pipeline statistics
- Final QC report with all flags consolidated

Usage:
    python scripts/05_generate_coco_annotations.py [--config CONFIG_PATH] [--output_name OUTPUT_NAME]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
from collections import defaultdict

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_config, ensure_directory, save_json, load_json, save_dataframe,
    setup_pipeline_logging, QCLogger
)


class COCOAnnotationGenerator:
    """
    Generates COCO format annotations from tracking results.
    """
    
    def __init__(self, config):
        """Initialize COCO annotation generator."""
        self.config = config
        self.logger = setup_pipeline_logging(config.config, "coco_generation")
        self.qc_logger = QCLogger(config.get('paths.logs_dir'))
        
        # Get configuration parameters
        self.tracks_dir = config.get_intermediate_path('tracks')
        self.videos_dir = config.get('paths.morphseq_well_videos')
        self.coco_categories = config.get_coco_categories()
        
        # Output directory
        self.final_dir = config.get('paths.final_dir')
        ensure_directory(self.final_dir)
        
        # COCO annotation structure
        self.coco_data = {
            'info': {
                'description': 'MorphSeq Embryo Segmentation Dataset',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'MorphSeq Pipeline',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': self.coco_categories
        }
        
        # ID counters
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        
        # Statistics tracking
        self.coco_stats = {
            'start_time': datetime.now().isoformat(),
            'videos_processed': 0,
            'total_images': 0,
            'total_annotations': 0,
            'embryo_annotations': 0,
            'dead_embryo_annotations': 0,
            'tracks_processed': 0,
            'errors': []
        }
    
    def get_video_info(self, video_name: str) -> Optional[Dict[str, Any]]:
        """Get video information from video metadata."""
        try:
            # Load video metadata
            metadata_file = Path(self.videos_dir) / "video_metadata.json"
            if metadata_file.exists():
                metadata = load_json(metadata_file)
                if metadata and video_name in metadata:
                    return metadata[video_name]
            
            # Fallback: get basic info from video file
            import cv2
            video_path = Path(self.videos_dir) / f"{video_name}.mp4"
            if video_path.exists():
                cap = cv2.VideoCapture(str(video_path))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    return {
                        'video_fps': fps,
                        'valid_frames': frame_count,
                        'resolution': [width, height]
                    }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Could not get video info for {video_name}: {e}")
            return None
    
    def convert_track_to_annotations(self, track_data: Dict[str, Any], 
                                   video_name: str, video_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a single track to COCO annotations.
        
        Args:
            track_data: Track trajectory data
            video_name: Name of the video
            video_info: Video metadata
            
        Returns:
            List of COCO annotations
        """
        annotations = []
        
        track_id = track_data['track_id']
        frames = track_data.get('frames', [])
        positions = track_data.get('positions', [])
        is_alive = track_data.get('is_alive', True)
        death_frame = track_data.get('death_frame')
        
        if not frames or not positions:
            return annotations
        
        # Get video dimensions
        width, height = video_info.get('resolution', [1920, 1080])
        
        for i, frame_idx in enumerate(frames):
            if i >= len(positions):
                break
            
            # Determine category
            category_id = 1  # Default: embryo
            if not is_alive and death_frame is not None and frame_idx >= death_frame:
                category_id = 2  # Dead embryo
            
            # Create image entry if not exists
            image_id = f"{video_name}_frame_{frame_idx:06d}"
            
            # Check if image already added
            existing_image = None
            for img in self.coco_data['images']:
                if img['id'] == image_id:
                    existing_image = img
                    break
            
            if not existing_image:
                image_entry = {
                    'id': image_id,
                    'width': width,
                    'height': height,
                    'file_name': f"{video_name}_frame_{frame_idx:06d}.jpg",
                    'video_name': video_name,
                    'frame_index': frame_idx
                }
                self.coco_data['images'].append(image_entry)
                self.coco_stats['total_images'] += 1
            
            # Create annotation
            # Note: This is simplified - in real implementation, would use actual bbox and mask
            pos_x, pos_y = positions[i]
            
            # Convert normalized position to pixel coordinates and create bbox
            center_x = pos_x * width
            center_y = pos_y * height
            
            # Estimated bbox size (would use actual detection bbox in real implementation)
            bbox_width = 50  # Placeholder
            bbox_height = 50  # Placeholder
            
            bbox = [
                max(0, center_x - bbox_width // 2),
                max(0, center_y - bbox_height // 2),
                min(width, bbox_width),
                min(height, bbox_height)
            ]
            
            annotation = {
                'id': self.annotation_id_counter,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
                'track_id': track_id,
                'frame_index': frame_idx,
                'confidence': track_data.get('mean_confidence', 1.0)
            }
            
            # Add segmentation if mask available (placeholder)
            # In real implementation, would include actual mask polygons
            annotation['segmentation'] = []
            
            annotations.append(annotation)
            self.annotation_id_counter += 1
            
            # Update statistics
            if category_id == 1:
                self.coco_stats['embryo_annotations'] += 1
            elif category_id == 2:
                self.coco_stats['dead_embryo_annotations'] += 1
        
        return annotations
    
    def process_video_tracks(self, video_name: str) -> bool:
        """
        Process tracking results for a single video and convert to COCO format.
        
        Args:
            video_name: Name of the video (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Processing COCO annotations for video: {video_name}")
        
        # Load tracking results
        track_file = Path(self.tracks_dir) / f"{video_name}_tracks.json"
        if not track_file.exists():
            self.logger.warning(f"No tracking file found for video: {video_name}")
            return False
        
        track_data = load_json(track_file)
        if not track_data:
            self.logger.error(f"Could not load tracking data: {track_file}")
            return False
        
        try:
            # Get video information
            video_info = self.get_video_info(video_name)
            if not video_info:
                self.logger.warning(f"Could not get video info for: {video_name}")
                video_info = {'resolution': [1920, 1080], 'video_fps': 2.0}  # Default
            
            # Process each track
            tracks = track_data.get('tracks', [])
            video_annotations = []
            
            for track in tracks:
                track_annotations = self.convert_track_to_annotations(track, video_name, video_info)
                video_annotations.extend(track_annotations)
            
            # Add annotations to COCO data
            self.coco_data['annotations'].extend(video_annotations)
            
            # Update statistics
            self.coco_stats['videos_processed'] += 1
            self.coco_stats['tracks_processed'] += len(tracks)
            self.coco_stats['total_annotations'] += len(video_annotations)
            
            self.logger.info(f"Completed COCO conversion: {video_name}")
            self.logger.info(f"  Tracks: {len(tracks)}")
            self.logger.info(f"  Annotations: {len(video_annotations)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing COCO annotations for video {video_name}: {e}")
            self.coco_stats['errors'].append({
                'video_name': video_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def generate_final_summary(self) -> Dict[str, Any]:
        """Generate final pipeline summary."""
        summary = {
            'pipeline_completion_time': datetime.now().isoformat(),
            'coco_statistics': self.coco_stats,
            'dataset_summary': {
                'total_videos': self.coco_stats['videos_processed'],
                'total_images': len(self.coco_data['images']),
                'total_annotations': len(self.coco_data['annotations']),
                'total_tracks': self.coco_stats['tracks_processed'],
                'category_breakdown': {
                    'embryo': self.coco_stats['embryo_annotations'],
                    'dead_embryo': self.coco_stats['dead_embryo_annotations']
                }
            },
            'quality_metrics': {
                'annotations_per_image': len(self.coco_data['annotations']) / max(1, len(self.coco_data['images'])),
                'tracks_per_video': self.coco_stats['tracks_processed'] / max(1, self.coco_stats['videos_processed'])
            }
        }
        
        return summary
    
    def run(self, output_name: str = "morphseq_embryos") -> bool:
        """
        Run the COCO annotation generation pipeline.
        
        Args:
            output_name: Base name for output files
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.start_pipeline()
        self.logger.start_stage("COCO Annotation Generation",
                               tracks_dir=self.tracks_dir,
                               final_dir=self.final_dir,
                               output_name=output_name)
        
        try:
            # Get tracking files
            track_files = list(Path(self.tracks_dir).glob("*_tracks.json"))
            
            if not track_files:
                self.logger.error(f"No tracking files found in {self.tracks_dir}")
                return False
            
            self.logger.info(f"Found {len(track_files)} tracking files to process")
            
            # Process each video
            success_count = 0
            for track_file in track_files:
                video_name = track_file.stem.replace('_tracks', '')
                if self.process_video_tracks(video_name):
                    success_count += 1
            
            if success_count == 0:
                self.logger.error("No videos processed successfully")
                return False
            
            # Save COCO annotations
            coco_file = Path(self.final_dir) / f"{output_name}.json"
            if save_json(self.coco_data, coco_file):
                self.logger.info(f"Saved COCO annotations to: {coco_file}")
            else:
                self.logger.error(f"Failed to save COCO annotations to: {coco_file}")
                return False
            
            # Generate and save final summary
            final_summary = self.generate_final_summary()
            summary_file = Path(self.final_dir) / "final_summary.json"
            save_json(final_summary, summary_file)
            
            # Save consolidated QC report
            self.qc_logger.save_qc_report()
            
            # Create dataset statistics CSV
            if self.coco_data['annotations']:
                import pandas as pd
                
                # Create annotations DataFrame
                annotations_df = pd.DataFrame(self.coco_data['annotations'])
                annotations_csv = Path(self.final_dir) / f"{output_name}_annotations.csv"
                save_dataframe(annotations_df, annotations_csv)
                
                # Create images DataFrame
                images_df = pd.DataFrame(self.coco_data['images'])
                images_csv = Path(self.final_dir) / f"{output_name}_images.csv"
                save_dataframe(images_df, images_csv)
            
            self.logger.end_stage("COCO Annotation Generation",
                                processed_videos=success_count,
                                total_videos=len(track_files),
                                total_annotations=len(self.coco_data['annotations']))
            
            self.logger.end_pipeline(
                videos_processed=success_count,
                total_annotations=len(self.coco_data['annotations']),
                output_file=str(coco_file)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate COCO annotations from tracking results")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output_name', type=str, default="morphseq_embryos",
                       help='Base name for output files')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create and run processor
    processor = COCOAnnotationGenerator(config)
    
    success = processor.run(args.output_name)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
