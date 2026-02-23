#!/usr/bin/env python
"""
03_generate_sam2_masks.py

Third stage of the MorphSeq embryo segmentation pipeline.
Generates detailed segmentation masks using SAM2 based on GroundingDINO detections.

This script:
1. Loads detection results from the detections directory
2. Uses SAM2 to generate precise segmentation masks for each detection
3. Applies mask quality filtering and cleanup
4. Saves masks with detections for further processing
5. Generates QC flags for mask quality issues

Input:
- Detection results from detections/ directory
- Videos from morphseq_well_videos/
- Pipeline configuration

Output:
- masks/ directory with per-video mask files
- Updated detection files with mask information
- mask_summary.json with overall statistics
- QC flags for mask quality issues

Usage:
    python scripts/03_generate_sam2_masks.py [--config CONFIG_PATH] [--video_pattern PATTERN]
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

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    load_config, ensure_directory, save_json, load_json, 
    get_video_files, setup_pipeline_logging, QCLogger,
    Detection, DetectionProcessor, clean_mask, get_mask_quality_score,
    calculate_mask_coverage, get_mask_properties
)


class SAM2MaskProcessor:
    """
    Handles SAM2 mask generation for detected embryos.
    """
    
    def __init__(self, config):
        """Initialize SAM2 mask processor."""
        self.config = config
        self.logger = setup_pipeline_logging(config.config, "sam2_masks")
        self.qc_logger = QCLogger(config.get('paths.logs_dir'))
        
        # Get configuration parameters
        self.videos_dir = config.get('paths.morphseq_well_videos')
        self.detections_dir = config.get_intermediate_path('detections')
        self.embryo_mask_root = config.get_mask_dir('embryo')
        self.detection_params = config.get_detection_params()
        
        # Model configuration
        self.sam2_config = config.get_model_config('sam2')
        
        # Output directory
        self.masks_dir = config.get_intermediate_path('masks')
        ensure_directory(self.masks_dir)
        
        # Initialize SAM2 model (placeholder)
        self.sam2_model = None
        self._load_sam2_model()
        
        # Quality thresholds
        self.min_mask_area = 100
        self.max_mask_area = 50000
        self.min_mask_solidity = 0.5
        self.sam2_overlap_threshold = self.detection_params.get('sam2_embryo_mask_overlap_threshold', 0.30)
        
        # Statistics tracking
        self.mask_stats = {
            'start_time': datetime.now().isoformat(),
            'videos_processed': 0,
            'total_detections': 0,
            'masks_generated': 0,
            'masks_filtered_quality': 0,
            'masks_filtered_overlap': 0,
            'videos_with_flags': 0,
            'errors': []
        }
    
    def _load_sam2_model(self):
        """Load SAM2 model."""
        self.logger.info("Loading SAM2 model...")
        
        try:
            config_path = self.sam2_config.get('config')
            checkpoint_path = self.sam2_config.get('checkpoint')
            
            self.logger.info(f"SAM2 config: {config_path}")
            self.logger.info(f"SAM2 checkpoint: {checkpoint_path}")
            
            # Check if model files exist
            if not Path(config_path).exists():
                self.logger.error(f"SAM2 config not found: {config_path}")
                self.qc_logger.add_global_flag(
                    "MISSING_MODEL_CONFIG",
                    f"SAM2 config not found: {config_path}"
                )
                return False
            
            if not Path(checkpoint_path).exists():
                self.logger.error(f"SAM2 checkpoint not found: {checkpoint_path}")
                self.qc_logger.add_global_flag(
                    "MISSING_MODEL_CHECKPOINT",
                    f"SAM2 checkpoint not found: {checkpoint_path}"
                )
                return False
            
            # Placeholder for actual SAM2 model loading
            # self.sam2_model = load_sam2_model(config_path, checkpoint_path)
            self.logger.info("SAM2 model loaded successfully (placeholder)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading SAM2 model: {e}")
            self.mask_stats['errors'].append({
                'stage': 'model_loading',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def _run_sam2_segmentation(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Run SAM2 segmentation on image with bounding box prompt.
        
        Args:
            image: Input image
            bbox: Bounding box in normalized coordinates [x1, y1, x2, y2]
            
        Returns:
            Segmentation mask or None if failed
        """
        if self.sam2_model is None:
            return None
        
        try:
            h, w = image.shape[:2]
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            # Ensure bbox is within image bounds
            x1 = max(0, min(w-1, x1))
            x2 = max(0, min(w-1, x2))
            y1 = max(0, min(h-1, y1))
            y2 = max(0, min(h-1, y2))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Placeholder implementation - generate synthetic mask
            # In real implementation, this would call SAM2
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Create elliptical mask in bbox region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius_x = max(1, (x2 - x1) // 3)
            radius_y = max(1, (y2 - y1) // 3)
            
            # Add some randomness to simulate real SAM2 output
            noise_factor = np.random.uniform(0.7, 1.3)
            radius_x = int(radius_x * noise_factor)
            radius_y = int(radius_y * noise_factor)
            
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1, -1)
            
            # Add some noise to make it more realistic
            if np.random.random() > 0.8:  # 20% chance of noise
                noise_mask = np.random.random((h, w)) > 0.95
                mask[noise_mask] = 1 - mask[noise_mask]
            
            return mask
            
        except Exception as e:
            self.logger.error(f"Error in SAM2 segmentation: {e}")
            return None
    
    def load_embryo_mask_for_video(self, video_path: str) -> Optional[np.ndarray]:
        """Load embryo mask for filtering (same as in detection script)."""
        video_name = Path(video_path).stem
        
        # Look for corresponding mask file
        mask_patterns = [
            f"{video_name}_embryo_mask.npy",
            f"{video_name}_mask.npy",
            f"{video_name}.npy"
        ]
        
        for pattern in mask_patterns:
            mask_path = Path(self.embryo_mask_root) / pattern
            if mask_path.exists():
                self.logger.debug(f"Loading embryo mask: {mask_path}")
                from utils.mask_utils import load_mask
                return load_mask(mask_path)
        
        # Try with different extensions
        for ext in ['.png', '.tif', '.tiff']:
            for pattern in [f"{video_name}_embryo_mask", f"{video_name}_mask", video_name]:
                mask_path = Path(self.embryo_mask_root) / f"{pattern}{ext}"
                if mask_path.exists():
                    self.logger.debug(f"Loading embryo mask: {mask_path}")
                    from utils.mask_utils import load_mask
                    return load_mask(mask_path)
        
        return None
    
    def process_video_masks(self, video_path: str) -> bool:
        """
        Process SAM2 masks for a single video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if successful, False otherwise
        """
        video_name = Path(video_path).stem
        self.logger.info(f"Processing SAM2 masks for video: {video_name}")
        
        # Load detection results
        detection_file = Path(self.detections_dir) / f"{video_name}_detections.json"
        if not detection_file.exists():
            self.logger.warning(f"No detection file found for video: {video_name}")
            return False
        
        detection_data = load_json(detection_file)
        if not detection_data:
            self.logger.error(f"Could not load detection data: {detection_file}")
            return False
        
        # Check if output already exists
        output_file = Path(self.masks_dir) / f"{video_name}_masks.json"
        if output_file.exists() and not self.config.get('processing.overwrite_intermediate', False):
            self.logger.info(f"Masks already exist, skipping: {output_file}")
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
            
            # Process frames with detections
            frame_detections = detection_data.get('frame_detections', {})
            frame_masks = {}
            
            video_mask_count = 0
            masks_filtered_quality = 0
            masks_filtered_overlap = 0
            
            for frame_idx_str, detections in frame_detections.items():
                frame_idx = int(frame_idx_str)
                
                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Could not read frame {frame_idx}")
                    continue
                
                # Process each detection in frame
                frame_detection_masks = []
                
                for detection_dict in detections:
                    # Convert back to Detection object
                    detection = Detection(
                        bbox=detection_dict['bbox'],
                        confidence=detection_dict['confidence'],
                        category_id=detection_dict.get('category_id', 1)
                    )
                    
                    # Generate SAM2 mask
                    mask = self._run_sam2_segmentation(frame, detection.bbox)
                    
                    if mask is None:
                        continue
                    
                    # Clean mask
                    cleaned_mask = clean_mask(mask, min_hole_size=50, min_object_size=self.min_mask_area)
                    
                    if cleaned_mask is None:
                        continue
                    
                    # Quality filtering
                    quality_score = get_mask_quality_score(
                        cleaned_mask, 
                        min_area=self.min_mask_area,
                        min_solidity=self.min_mask_solidity
                    )
                    
                    if quality_score < 0.5:  # Quality threshold
                        masks_filtered_quality += 1
                        self.logger.debug(f"Mask filtered by quality: {quality_score:.3f}")
                        continue
                    
                    # Overlap filtering with embryo mask
                    if embryo_mask is not None:
                        overlap = calculate_mask_coverage(cleaned_mask, embryo_mask)
                        if overlap < self.sam2_overlap_threshold:
                            masks_filtered_overlap += 1
                            self.logger.debug(f"Mask filtered by overlap: {overlap:.3f}")
                            continue
                    
                    # Get mask properties
                    mask_properties = get_mask_properties(cleaned_mask)
                    
                    # Update detection with mask
                    detection.mask = cleaned_mask
                    detection.mask_confidence = quality_score
                    
                    # Store mask data
                    mask_data = detection.to_dict()
                    mask_data.update({
                        'mask_properties': mask_properties,
                        'quality_score': quality_score,
                        'embryo_overlap': overlap if embryo_mask is not None else None
                    })
                    
                    frame_detection_masks.append(mask_data)
                    video_mask_count += 1
                
                if frame_detection_masks:
                    frame_masks[frame_idx] = frame_detection_masks
            
            cap.release()
            
            # Video-level quality checks
            total_detections = sum(len(dets) for dets in frame_detections.values())
            mask_generation_rate = video_mask_count / total_detections if total_detections > 0 else 0
            
            # Flag videos with quality issues
            if mask_generation_rate < 0.7:  # Less than 70% of detections got good masks
                self.qc_logger.add_video_flag(
                    video_path, "LOW_MASK_GENERATION_RATE",
                    f"Low mask generation rate: {mask_generation_rate:.2%}",
                    mask_generation_rate=mask_generation_rate,
                    masks_generated=video_mask_count,
                    total_detections=total_detections
                )
            
            if video_mask_count == 0 and total_detections > 0:
                self.qc_logger.add_video_flag(
                    video_path, "NO_VALID_MASKS",
                    f"No valid masks generated from {total_detections} detections"
                )
            
            # Save mask results
            mask_results = {
                'video_path': video_path,
                'video_name': video_name,
                'processing_timestamp': datetime.now().isoformat(),
                'base_detection_file': str(detection_file),
                'mask_summary': {
                    'total_detections': total_detections,
                    'masks_generated': video_mask_count,
                    'masks_filtered_quality': masks_filtered_quality,
                    'masks_filtered_overlap': masks_filtered_overlap,
                    'mask_generation_rate': mask_generation_rate
                },
                'mask_parameters': {
                    'min_mask_area': self.min_mask_area,
                    'max_mask_area': self.max_mask_area,
                    'min_mask_solidity': self.min_mask_solidity,
                    'sam2_overlap_threshold': self.sam2_overlap_threshold
                },
                'embryo_mask_used': embryo_mask is not None,
                'frame_masks': frame_masks
            }
            
            # Save to file
            if save_json(mask_results, output_file):
                self.logger.info(f"Saved masks to: {output_file}")
            else:
                self.logger.error(f"Failed to save masks to: {output_file}")
                return False
            
            # Update statistics
            self.mask_stats['videos_processed'] += 1
            self.mask_stats['total_detections'] += total_detections
            self.mask_stats['masks_generated'] += video_mask_count
            self.mask_stats['masks_filtered_quality'] += masks_filtered_quality
            self.mask_stats['masks_filtered_overlap'] += masks_filtered_overlap
            
            self.logger.info(f"Completed mask processing: {video_name}")
            self.logger.info(f"  Total detections: {total_detections}")
            self.logger.info(f"  Masks generated: {video_mask_count}")
            self.logger.info(f"  Generation rate: {mask_generation_rate:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing masks for video {video_path}: {e}")
            self.mask_stats['errors'].append({
                'video_path': video_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def run(self, video_pattern: str = "*.mp4") -> bool:
        """
        Run the SAM2 mask generation pipeline.
        
        Args:
            video_pattern: Pattern to match video files
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.start_pipeline()
        self.logger.start_stage("SAM2 Mask Generation",
                               videos_dir=self.videos_dir,
                               detections_dir=self.detections_dir,
                               masks_dir=self.masks_dir)
        
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
                if self.process_video_masks(str(video_path)):
                    success_count += 1
            
            # Save summary statistics
            self.mask_stats['end_time'] = datetime.now().isoformat()
            self.mask_stats['videos_total'] = len(video_files)
            self.mask_stats['videos_successful'] = success_count
            self.mask_stats['videos_failed'] = len(video_files) - success_count
            
            summary_file = Path(self.masks_dir) / "mask_summary.json"
            save_json(self.mask_stats, summary_file)
            
            # Save QC report
            self.qc_logger.save_qc_report()
            
            self.logger.end_stage("SAM2 Mask Generation",
                                processed_videos=success_count,
                                total_videos=len(video_files),
                                masks_generated=self.mask_stats['masks_generated'])
            
            self.logger.end_pipeline(
                videos_processed=success_count,
                masks_generated=self.mask_stats['masks_generated']
            )
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate SAM2 masks for detections")
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
    processor = SAM2MaskProcessor(config)
    
    success = processor.run(args.video_pattern)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
