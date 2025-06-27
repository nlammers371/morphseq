#!/usr/bin/env python
"""
MorphSeq Embryo Segmentation Pipeline Utilities

This package contains utility modules for the segmentation pipeline:
- config_utils: Configuration loading and management
- file_utils: File I/O and path management
- mask_utils: Mask processing and analysis
- detection_utils: Detection processing and filtering
- tracking_utils: Embryo tracking across frames
- logging_utils: Structured logging and QC reporting
"""

from .config_utils import PipelineConfig, load_config
from .file_utils import (
    ensure_directory, save_json, load_json, save_pickle, load_pickle,
    save_dataframe, load_dataframe, get_video_files, get_image_files
)
from .mask_utils import (
    load_mask, calculate_bbox_mask_overlap, calculate_mask_overlap,
    calculate_mask_coverage, filter_masks_by_size, clean_mask,
    get_mask_properties, compare_masks
)
from .detection_utils import (
    Detection, DetectionProcessor, convert_groundingdino_output,
    convert_sam2_output, draw_detections, calculate_detection_statistics
)
from .tracking_utils import (
    EmbryoTrack, EmbryoTracker, analyze_trajectories, detect_tracking_anomalies
)
from .logging_utils import (
    PipelineLogger, QCLogger, setup_pipeline_logging
)

__version__ = "1.0.0"
__author__ = "MorphSeq Pipeline Team"

# Expose main classes and functions
__all__ = [
    # Config utilities
    'PipelineConfig', 'load_config',
    
    # File utilities
    'ensure_directory', 'save_json', 'load_json', 'save_pickle', 'load_pickle',
    'save_dataframe', 'load_dataframe', 'get_video_files', 'get_image_files',
    
    # Mask utilities
    'load_mask', 'calculate_bbox_mask_overlap', 'calculate_mask_overlap',
    'calculate_mask_coverage', 'filter_masks_by_size', 'clean_mask',
    'get_mask_properties', 'compare_masks',
    
    # Detection utilities
    'Detection', 'DetectionProcessor', 'convert_groundingdino_output',
    'convert_sam2_output', 'draw_detections', 'calculate_detection_statistics',
    
    # Tracking utilities
    'EmbryoTrack', 'EmbryoTracker', 'analyze_trajectories', 'detect_tracking_anomalies',
    
    # Logging utilities
    'PipelineLogger', 'QCLogger', 'setup_pipeline_logging'
]
