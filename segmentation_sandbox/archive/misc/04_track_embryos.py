#!/usr/bin/env python
"""
04_track_embryos.py

Fourth stage of the MorphSeq embryo segmentation pipeline.
Tracks embryos across frames using detection and mask information.

This script:
1. Loads mask results from the masks directory
2. Implements first-frame seeding with mode detection and backtracking
3. Tracks embryos across frames using position and appearance similarity
4. Detects embryo death events based on tracking gaps
5. Generates QC flags for tracking quality issues

Input:
- Mask results from masks/ directory
- Pipeline configuration

Output:
- tracks/ directory with per-video tracking files
- tracking_summary.json with overall statistics
- QC flags for tracking quality issues

Usage:
    python scripts/04_track_embryos.py [--config CONFIG_PATH] [--video_pattern PATTERN]
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
from collections import Counter, defaultdict

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_config, ensure_directory, save_json, load_json, save_dataframe,
    setup_pipeline_logging, QCLogger, Detection, EmbryoTracker, 
    analyze_trajectories, detect_tracking_anomalies
)


class EmbryoTrackingProcessor:
    """
    Handles embryo tracking across video frames with improved first-frame seeding.
    """
    
    def __init__(self, config):
        """Initialize embryo tracking processor."""
        self.config = config
        self.logger = setup_pipeline_logging(config.config, "embryo_tracking")
        self.qc_logger = QCLogger(config.get('paths.logs_dir'))
        
        # Get configuration parameters
        self.masks_dir = config.get_intermediate_path('masks')
        self.detection_params = config.get_detection_params()
        self.qc_params = config.get_qc_params()
        
        # Tracking configuration
        tracking_config = {
            'max_tracking_distance': 0.1,  # Normalized distance
            'num_consecutive_undetected': self.detection_params.get('num_consecutive_undetected', 4),
            'max_tracking_gap': self.qc_params.get('max_tracking_gap', 3)
        }
        
        # First-frame seeding parameters
        self.min_consistency_frames = self.detection_params.get('min_consistency_frames', 3)
        self.max_seed_frame = self.detection_params.get('max_seed_frame', 10)
        
        # Output directory
        self.tracks_dir = config.get_intermediate_path('tracks')
        ensure_directory(self.tracks_dir)
        
        # Initialize tracker (will be created per video)
        self.tracker_config = tracking_config
        
        # Statistics tracking
        self.tracking_stats = {
            'start_time': datetime.now().isoformat(),
            'videos_processed': 0,
            'total_tracks': 0,
            'alive_tracks': 0,
            'dead_tracks': 0,
            'tracks_with_anomalies': 0,
            'seeding_failures': 0,
            'seeding_backtracks': 0,
            'errors': []
        }
    
    def detect_embryo_count_mode(self, frame_detections: Dict[int, List], 
                                max_frames: int = None) -> Tuple[int, Dict[int, int]]:
        """
        Detect the mode (most common) number of embryos in early frames.
        
        Args:
            frame_detections: Dictionary of frame_idx -> list of detections
            max_frames: Maximum number of frames to consider for mode detection
            
        Returns:
            Tuple of (mode_count, frame_counts_dict)
        """
        if max_frames is None:
            max_frames = self.max_seed_frame
        
        # Count embryos in each frame
        frame_counts = {}
        for frame_idx in sorted(frame_detections.keys()):
            if frame_idx > max_frames:
                break
            frame_counts[frame_idx] = len(frame_detections[frame_idx])
        
        if not frame_counts:
            return 0, {}
        
        # Find mode
        count_frequency = Counter(frame_counts.values())
        mode_count = count_frequency.most_common(1)[0][0]
        
        self.logger.debug(f"Embryo count mode: {mode_count}")
        self.logger.debug(f"Frame counts: {frame_counts}")
        
        return mode_count, frame_counts
    
    def find_best_seed_frame(self, frame_detections: Dict[int, List], 
                           expected_count: int) -> Optional[int]:
        """
        Find the best frame to use for seeding tracks.
        
        Args:
            frame_detections: Dictionary of frame_idx -> list of detections
            expected_count: Expected number of embryos
            
        Returns:
            Best seed frame index or None if none found
        """
        candidate_frames = []
        
        # Look for frames with expected count in early frames
        for frame_idx in sorted(frame_detections.keys()):
            if frame_idx > self.max_seed_frame:
                break
            
            detection_count = len(frame_detections[frame_idx])
            
            if detection_count == expected_count:
                # Calculate quality score for this frame
                detections = frame_detections[frame_idx]
                
                # Score based on detection confidence and mask quality
                quality_score = 0.0
                for det_dict in detections:
                    quality_score += det_dict.get('confidence', 0.0)
                    quality_score += det_dict.get('quality_score', 0.0)
                
                quality_score /= len(detections) if detections else 1
                
                candidate_frames.append((frame_idx, quality_score))
        
        if not candidate_frames:
            return None
        
        # Return frame with highest quality score
        best_frame = max(candidate_frames, key=lambda x: x[1])[0]
        
        self.logger.debug(f"Selected seed frame: {best_frame}")
        return best_frame
    
    def backtrack_and_seed(self, frame_detections: Dict[int, List], 
                          tracker: EmbryoTracker, seed_frame: int) -> bool:
        """
        Seed tracks from the identified seed frame and backtrack to frame 0.
        
        Args:
            frame_detections: Dictionary of frame_idx -> list of detections
            tracker: EmbryoTracker instance
            seed_frame: Frame to start seeding from
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.debug(f"Backtracking and seeding from frame {seed_frame}")
        
        # Convert detections to Detection objects for seed frame
        seed_detections = []
        for det_dict in frame_detections[seed_frame]:
            detection = Detection(
                bbox=det_dict['bbox'],
                confidence=det_dict['confidence'],
                category_id=det_dict.get('category_id', 1)
            )
            
            # Add mask if available
            if 'mask_properties' in det_dict:
                # Placeholder - in real implementation, would reconstruct mask
                detection.mask_confidence = det_dict.get('quality_score')
            
            seed_detections.append(detection)
        
        # Initialize tracks with seed frame
        tracker.update(seed_frame, seed_detections)
        
        # Backtrack from seed_frame to 0
        for frame_idx in range(seed_frame - 1, -1, -1):
            if frame_idx in frame_detections:
                frame_det_objects = []
                for det_dict in frame_detections[frame_idx]:
                    detection = Detection(
                        bbox=det_dict['bbox'],
                        confidence=det_dict['confidence'],
                        category_id=det_dict.get('category_id', 1)
                    )
                    if 'mask_properties' in det_dict:
                        detection.mask_confidence = det_dict.get('quality_score')
                    frame_det_objects.append(detection)
                
                tracker.update(frame_idx, frame_det_objects)
            else:
                # No detections in this frame
                tracker.update(frame_idx, [])
        
        # Forward track from seed_frame + 1 onwards
        for frame_idx in sorted(frame_detections.keys()):
            if frame_idx <= seed_frame:
                continue
            
            frame_det_objects = []
            for det_dict in frame_detections[frame_idx]:
                detection = Detection(
                    bbox=det_dict['bbox'],
                    confidence=det_dict['confidence'],
                    category_id=det_dict.get('category_id', 1)
                )
                if 'mask_properties' in det_dict:
                    detection.mask_confidence = det_dict.get('quality_score')
                frame_det_objects.append(detection)
            
            tracker.update(frame_idx, frame_det_objects)
        
        return True
    
    def process_video_tracking(self, video_name: str) -> bool:
        """
        Process tracking for a single video.
        
        Args:
            video_name: Name of the video (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Processing tracking for video: {video_name}")
        
        # Load mask results
        mask_file = Path(self.masks_dir) / f"{video_name}_masks.json"
        if not mask_file.exists():
            self.logger.warning(f"No mask file found for video: {video_name}")
            return False
        
        mask_data = load_json(mask_file)
        if not mask_data:
            self.logger.error(f"Could not load mask data: {mask_file}")
            return False
        
        # Check if output already exists
        output_file = Path(self.tracks_dir) / f"{video_name}_tracks.json"
        if output_file.exists() and not self.config.get('processing.overwrite_intermediate', False):
            self.logger.info(f"Tracks already exist, skipping: {output_file}")
            return True
        
        try:
            frame_masks = mask_data.get('frame_masks', {})
            
            if not frame_masks:
                self.logger.warning(f"No frame masks found for video: {video_name}")
                self.qc_logger.add_video_flag(
                    video_name, "NO_FRAME_MASKS",
                    "No frame masks available for tracking"
                )
                return False
            
            # Convert string keys to integers
            frame_detections = {int(k): v for k, v in frame_masks.items()}
            
            # Detect embryo count mode and find best seed frame
            mode_count, frame_counts = self.detect_embryo_count_mode(frame_detections)
            
            if mode_count == 0:
                self.logger.warning(f"No embryos detected in video: {video_name}")
                self.qc_logger.add_video_flag(
                    video_name, "NO_EMBRYOS_DETECTED",
                    "No embryos detected for tracking"
                )
                return False
            
            # Find best seed frame
            seed_frame = self.find_best_seed_frame(frame_detections, mode_count)
            
            if seed_frame is None:
                # Fallback to frame 0 if available
                if 0 in frame_detections:
                    seed_frame = 0
                    self.logger.warning(f"Using fallback seed frame 0 for video: {video_name}")
                    self.tracking_stats['seeding_failures'] += 1
                    self.qc_logger.add_video_flag(
                        video_name, "SEEDING_FALLBACK",
                        f"Could not find good seed frame, using frame 0"
                    )
                else:
                    self.logger.error(f"No suitable seed frame found for video: {video_name}")
                    self.tracking_stats['seeding_failures'] += 1
                    return False
            elif seed_frame > 0:
                self.tracking_stats['seeding_backtracks'] += 1
            
            # Initialize tracker
            tracker = EmbryoTracker(self.tracker_config)
            
            # Perform tracking with backtracking
            success = self.backtrack_and_seed(frame_detections, tracker, seed_frame)
            
            if not success:
                self.logger.error(f"Tracking failed for video: {video_name}")
                return False
            
            # Get tracking results
            all_tracks = tracker.get_all_tracks()
            tracking_summary = tracker.get_tracks_summary()
            
            # Detect tracking anomalies
            anomalies = detect_tracking_anomalies(all_tracks, self.qc_params)
            
            # Quality checks
            if tracking_summary['total_tracks'] == 0:
                self.qc_logger.add_video_flag(
                    video_name, "NO_TRACKS_GENERATED",
                    "No tracks were generated"
                )
            
            if tracking_summary['total_tracks'] > mode_count * 1.5:  # Too many tracks
                self.qc_logger.add_video_flag(
                    video_name, "EXCESSIVE_TRACKS",
                    f"Too many tracks generated: {tracking_summary['total_tracks']} (expected ~{mode_count})",
                    tracks_generated=tracking_summary['total_tracks'],
                    expected_count=mode_count
                )
            
            if tracking_summary['mean_track_length'] < 5:  # Very short tracks
                self.qc_logger.add_video_flag(
                    video_name, "SHORT_TRACKS",
                    f"Mean track length is very short: {tracking_summary['mean_track_length']:.1f}",
                    mean_track_length=tracking_summary['mean_track_length']
                )
            
            # Flag tracks with anomalies
            if anomalies:
                self.tracking_stats['tracks_with_anomalies'] += len(anomalies)
                for anomaly in anomalies:
                    self.qc_logger.add_video_flag(
                        video_name, "TRACKING_ANOMALY",
                        f"Track {anomaly['track_id']} has anomalies: {[a['type'] for a in anomaly['anomalies']]}",
                        track_id=anomaly['track_id'],
                        anomaly_types=[a['type'] for a in anomaly['anomalies']]
                    )
            
            # Generate trajectory analysis
            trajectory_df = analyze_trajectories(all_tracks)
            
            # Save tracking results
            tracking_results = {
                'video_name': video_name,
                'processing_timestamp': datetime.now().isoformat(),
                'base_mask_file': str(mask_file),
                'seeding_info': {
                    'mode_count': mode_count,
                    'seed_frame': seed_frame,
                    'frame_counts': frame_counts,
                    'backtrack_used': seed_frame > 0
                },
                'tracking_summary': tracking_summary,
                'tracking_parameters': self.tracker_config,
                'anomalies': anomalies,
                'tracks': [track.get_trajectory_data() for track in all_tracks]
            }
            
            # Save to file
            if save_json(tracking_results, output_file):
                self.logger.info(f"Saved tracking results to: {output_file}")
            else:
                self.logger.error(f"Failed to save tracking results to: {output_file}")
                return False
            
            # Save trajectory analysis
            trajectory_file = Path(self.tracks_dir) / f"{video_name}_trajectories.csv"
            if save_dataframe(trajectory_df, trajectory_file):
                self.logger.debug(f"Saved trajectory analysis to: {trajectory_file}")
            
            # Update statistics
            self.tracking_stats['videos_processed'] += 1
            self.tracking_stats['total_tracks'] += tracking_summary['total_tracks']
            self.tracking_stats['alive_tracks'] += tracking_summary['alive_tracks']
            self.tracking_stats['dead_tracks'] += tracking_summary['dead_tracks']
            
            self.logger.info(f"Completed tracking: {video_name}")
            self.logger.info(f"  Total tracks: {tracking_summary['total_tracks']}")
            self.logger.info(f"  Alive tracks: {tracking_summary['alive_tracks']}")
            self.logger.info(f"  Dead tracks: {tracking_summary['dead_tracks']}")
            self.logger.info(f"  Seed frame: {seed_frame}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing tracking for video {video_name}: {e}")
            self.tracking_stats['errors'].append({
                'video_name': video_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def run(self, video_pattern: str = "*_masks.json") -> bool:
        """
        Run the embryo tracking pipeline.
        
        Args:
            video_pattern: Pattern to match mask files
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.start_pipeline()
        self.logger.start_stage("Embryo Tracking",
                               masks_dir=self.masks_dir,
                               tracks_dir=self.tracks_dir,
                               seeding_params=f"mode detection, backtrack up to frame {self.max_seed_frame}")
        
        try:
            # Get mask files
            mask_files = list(Path(self.masks_dir).glob(video_pattern))
            
            if not mask_files:
                self.logger.error(f"No mask files found in {self.masks_dir}")
                return False
            
            self.logger.info(f"Found {len(mask_files)} mask files to process")
            
            # Process each video
            success_count = 0
            for mask_file in mask_files:
                video_name = mask_file.stem.replace('_masks', '')
                if self.process_video_tracking(video_name):
                    success_count += 1
            
            # Save summary statistics
            self.tracking_stats['end_time'] = datetime.now().isoformat()
            self.tracking_stats['videos_total'] = len(mask_files)
            self.tracking_stats['videos_successful'] = success_count
            self.tracking_stats['videos_failed'] = len(mask_files) - success_count
            
            summary_file = Path(self.tracks_dir) / "tracking_summary.json"
            save_json(self.tracking_stats, summary_file)
            
            # Save QC report
            self.qc_logger.save_qc_report()
            
            self.logger.end_stage("Embryo Tracking",
                                processed_videos=success_count,
                                total_videos=len(mask_files),
                                total_tracks=self.tracking_stats['total_tracks'])
            
            self.logger.end_pipeline(
                videos_processed=success_count,
                total_tracks=self.tracking_stats['total_tracks']
            )
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Track embryos across video frames")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--video_pattern', type=str, default="*_masks.json",
                       help='Pattern to match mask files')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create and run processor
    processor = EmbryoTrackingProcessor(config)
    
    success = processor.run(args.video_pattern)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
