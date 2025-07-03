#!/usr/bin/env python
"""
02_initial_detection.py

Second stage of the MorphSeq embryo segmentation pipeline.
Performs initial embryo detection using GroundingDINO on all video frames.

This script:
1. Loads videos from morphseq_well_videos directory
2. Runs GroundingDINO detection on each frame
3. Applies initial filtering based on confidence and overlap with embryo masks
4. Saves raw detection results for each video
5. Generates QC flags for detection quality issues

Input:
- Videos from morphseq_well_videos/
- Embryo masks for filtering
- Pipeline configuration

Output:
- detections/ directory with per-video detection files
- detection_summary.json with overall statistics
- QC flags for detection quality issues

Usage:
    python scripts/02_initial_detection.py [--config CONFIG_PATH] [--video_pattern PATTERN]
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import glob

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_config, ensure_directory, save_json, load_json, 
    get_video_files, setup_pipeline_logging, QCLogger,
    Detection, DetectionProcessor, load_mask, calculate_detection_statistics
)


class InitialDetectionProcessor:
    """
    Handles initial embryo detection using GroundingDINO.
    """
    
    def __init__(self, config):
        """Initialize detection processor."""
        self.config = config
        self.logger = setup_pipeline_logging(config.config, "initial_detection")
        self.qc_logger = QCLogger(config.get('paths.logs_dir'))
        
        # Get configuration parameters
        self.videos_dir = config.get('paths.morphseq_well_videos')
        self.embryo_mask_root = config.get_mask_dir('embryo')
        self.detection_params = config.get_detection_params()
        
        # Model configuration
        self.groundingdino_config = config.get_model_config('groundingdino')
        self.text_prompt = self.detection_params.get('text_prompt', 'individual embryo.')
        
        # Output directory
        self.detections_dir = config.get_intermediate_path('detections')
        ensure_directory(self.detections_dir)
        
        # Initialize detection processor
        self.detection_processor = DetectionProcessor(self.detection_params)
        
        # Initialize GroundingDINO (placeholder - actual implementation would load model)
        self.grounding_model = None
        self._load_grounding_model()
        
        # Statistics tracking
        self.detection_stats = {
            'start_time': datetime.now().isoformat(),
            'videos_processed': 0,
            'total_frames': 0,
            'total_detections': 0,
            'videos_with_no_detections': 0,
            'videos_with_flags': 0,
            'errors': []
        }
    
    def _load_grounding_model(self):
        """Load GroundingDINO model."""
        self.logger.info("Loading GroundingDINO model...")
        
        try:
            # This is a placeholder - actual implementation would load GroundingDINO
            # from the configured paths
            config_path = self.groundingdino_config.get('config')
            weights_path = self.groundingdino_config.get('weights')
            
            self.logger.info(f"Model config: {config_path}")
            self.logger.info(f"Model weights: {weights_path}")
            
            # Check if model files exist
            if not Path(config_path).exists():
                self.logger.error(f"GroundingDINO config not found: {config_path}")
                self.qc_logger.add_global_flag(
                    "MISSING_MODEL_CONFIG",
                    f"GroundingDINO config not found: {config_path}"
                )
                return False
            
            if not Path(weights_path).exists():
                self.logger.error(f"GroundingDINO weights not found: {weights_path}")
                self.qc_logger.add_global_flag(
                    "MISSING_MODEL_WEIGHTS",
                    f"GroundingDINO weights not found: {weights_path}"
                )
                return False
            
            # Placeholder for actual model loading
            # self.grounding_model = load_groundingdino_model(config_path, weights_path)
            self.logger.info("GroundingDINO model loaded successfully (placeholder)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading GroundingDINO model: {e}")
            self.detection_stats['errors'].append({
                'stage': 'model_loading',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def _run_grounding_detection(self, image: np.ndarray) -> List[Detection]:
        """
        Run GroundingDINO detection on image.
        
        Args:
            image: Input image
            
        Returns:
            List of Detection objects
        """
        # This is a placeholder implementation
        # Actual implementation would run GroundingDINO inference
        
        if self.grounding_model is None:
            return []
        
        try:
            # Placeholder: Generate some dummy detections for testing
            # In real implementation, this would call GroundingDINO
            detections = []
            
            # Simulate detection results
            if np.random.random() > 0.1:  # 90% chance of detection
                num_detections = np.random.randint(1, 4)  # 1-3 detections per frame
                
                for i in range(num_detections):
                    # Random bbox in normalized coordinates
                    x1 = np.random.uniform(0.1, 0.6)
                    y1 = np.random.uniform(0.1, 0.6)
                    x2 = x1 + np.random.uniform(0.1, 0.3)
                    y2 = y1 + np.random.uniform(0.1, 0.3)
                    
                    # Ensure bbox is within image
                    x2 = min(x2, 0.9)
                    y2 = min(y2, 0.9)
                    
                    confidence = np.random.uniform(0.3, 0.9)
                    
                    detection = Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=confidence,
                        category_id=1
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in GroundingDINO detection: {e}")
            return []
    
    def load_embryo_mask_for_video(self, video_path: str) -> Optional[np.ndarray]:
        """
        Load embryo mask for video filtering.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Embryo mask or None if not found
        """
        video_name = Path(video_path).stem
        
        # Look for corresponding mask file
        # This is a placeholder - implement based on actual mask naming convention
        mask_patterns = [
            f"{video_name}_embryo_mask.npy",
            f"{video_name}_mask.npy",
            f"{video_name}.npy"
        ]
        
        for pattern in mask_patterns:
            mask_path = Path(self.embryo_mask_root) / pattern
            if mask_path.exists():
                self.logger.debug(f"Loading embryo mask: {mask_path}")
                return load_mask(mask_path)
        
        # Try with different extensions
        for ext in ['.png', '.tif', '.tiff']:
            for pattern in [f"{video_name}_embryo_mask", f"{video_name}_mask", video_name]:
                mask_path = Path(self.embryo_mask_root) / f"{pattern}{ext}"
                if mask_path.exists():
                    self.logger.debug(f"Loading embryo mask: {mask_path}")
                    return load_mask(mask_path)
        
        self.logger.warning(f"No embryo mask found for video: {video_name}")
        return None
    
    def process_video_detections(self, video_path: str) -> bool:
        """
        Process detections for a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if successful, False otherwise
        """
        video_name = Path(video_path).stem
        self.logger.info(f"Processing detections for video: {video_name}")
        
        # Check if output already exists
        output_file = Path(self.detections_dir) / f"{video_name}_detections.json"
        if output_file.exists() and not self.config.get('processing.overwrite_intermediate', False):
            self.logger.info(f"Detections already exist, skipping: {output_file}")
            return True
        
        try:
            # Load embryo mask for filtering
            embryo_mask = self.load_embryo_mask_for_video(video_path)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return False
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"Video properties: {total_frames} frames, {fps} fps, {width}x{height}")
            
            # Process each frame
            frame_detections = {}
            video_detection_count = 0
            frames_with_detections = 0
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                raw_detections = self._run_grounding_detection(frame)
                
                # Apply filtering
                filtered_detections = self.detection_processor.process_detections(
                    raw_detections, embryo_mask
                )
                
                # Store detections
                if filtered_detections:
                    frame_detections[frame_idx] = [det.to_dict() for det in filtered_detections]
                    video_detection_count += len(filtered_detections)
                    frames_with_detections += 1
                
                # Log frame-level statistics
                if self.logger.logger.isEnabledFor(self.logger.logger.getEffectiveLevel()):
                    detection_stats = calculate_detection_statistics(filtered_detections)
                    self.logger.log_detection_results(frame_idx, len(filtered_detections), detection_stats)
                
                frame_idx += 1
            
            cap.release()
            
            # Video-level quality checks
            detection_rate = frames_with_detections / total_frames if total_frames > 0 else 0
            avg_detections_per_frame = video_detection_count / total_frames if total_frames > 0 else 0
            
            # Flag videos with quality issues
            if detection_rate < 0.5:  # Less than 50% of frames have detections
                self.qc_logger.add_video_flag(
                    video_path, "LOW_DETECTION_RATE",
                    f"Low detection rate: {detection_rate:.2%}",
                    detection_rate=detection_rate,
                    frames_with_detections=frames_with_detections,
                    total_frames=total_frames
                )
            
            if video_detection_count == 0:
                self.qc_logger.add_video_flag(
                    video_path, "NO_DETECTIONS",
                    "No detections found in entire video"
                )
                self.detection_stats['videos_with_no_detections'] += 1
            
            if avg_detections_per_frame > 5:  # Too many detections might indicate noise
                self.qc_logger.add_video_flag(
                    video_path, "HIGH_DETECTION_COUNT",
                    f"High average detections per frame: {avg_detections_per_frame:.1f}",
                    avg_detections_per_frame=avg_detections_per_frame
                )
            
            # Save detection results
            video_results = {
                'video_path': video_path,
                'video_name': video_name,
                'processing_timestamp': datetime.now().isoformat(),
                'video_properties': {
                    'total_frames': total_frames,
                    'fps': fps,
                    'resolution': [width, height]
                },
                'detection_summary': {
                    'total_detections': video_detection_count,
                    'frames_with_detections': frames_with_detections,
                    'detection_rate': detection_rate,
                    'avg_detections_per_frame': avg_detections_per_frame
                },
                'detection_parameters': self.detection_params,
                'embryo_mask_used': embryo_mask is not None,
                'frame_detections': frame_detections
            }
            
            # Save to file
            if save_json(video_results, output_file):
                self.logger.info(f"Saved detections to: {output_file}")
            else:
                self.logger.error(f"Failed to save detections to: {output_file}")
                return False
            
            # Update statistics
            self.detection_stats['videos_processed'] += 1
            self.detection_stats['total_frames'] += total_frames
            self.detection_stats['total_detections'] += video_detection_count
            
            self.logger.info(f"Completed processing: {video_name}")
            self.logger.info(f"  Total detections: {video_detection_count}")
            self.logger.info(f"  Detection rate: {detection_rate:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            self.detection_stats['errors'].append({
                'video_path': video_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def run(self, video_pattern: str = "*.mp4") -> bool:
        """
        Run the initial detection pipeline.
        
        Args:
            video_pattern: Pattern to match video files
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.start_pipeline()
        self.logger.start_stage("Initial Detection",
                               videos_dir=self.videos_dir,
                               detections_dir=self.detections_dir,
                               text_prompt=self.text_prompt)
        
        try:
            # Get video files
            video_files = get_video_files(self.videos_dir)
            
            if not video_files:
                self.logger.error(f"No video files found in {self.videos_dir}")
                return False
            
            self.logger.info(f"Found {len(video_files)} video files to process")
            
            # Process each video
            success_count = 0
            for video_path in video_files:
                if self.process_video_detections(str(video_path)):
                    success_count += 1
            
            # Save summary statistics
            self.detection_stats['end_time'] = datetime.now().isoformat()
            self.detection_stats['videos_total'] = len(video_files)
            self.detection_stats['videos_successful'] = success_count
            self.detection_stats['videos_failed'] = len(video_files) - success_count
            
            summary_file = Path(self.detections_dir) / "detection_summary.json"
            save_json(self.detection_stats, summary_file)
            
            # Save QC report
            self.qc_logger.save_qc_report()
            
            self.logger.end_stage("Initial Detection",
                                processed_videos=success_count,
                                total_videos=len(video_files),
                                total_detections=self.detection_stats['total_detections'])
            
            self.logger.end_pipeline(
                videos_processed=success_count,
                total_detections=self.detection_stats['total_detections']
            )
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run initial embryo detection")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--video_pattern', type=str, default="*.mp4",
                       help='Pattern to match video files')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Create and run processor
    processor = InitialDetectionProcessor(config)
    
    success = processor.run(args.video_pattern)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
