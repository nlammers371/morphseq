"""
Detection and Segmentation Module (Module 2)
===========================================

This module provides object detection and segmentation capabilities:
- GroundingDINO detection with annotation management
- SAM2 video segmentation with entity tracking
- Quality control utilities (planned)
- Mask export utilities (planned)

Key classes:
- GroundedDinoAnnotations: Detection annotation management with metadata integration
- GroundedSamAnnotations: SAM2 video segmentation with entity validation
"""

from .grounded_dino_utils import (
    GroundedDinoAnnotations,
    load_groundingdino_model,
    load_config,
    get_model_metadata,
    calculate_detection_iou,
    run_inference,
    visualize_detections,
    gdino_inference_with_visualization
)

from .sam2_utils import (
    GroundedSamAnnotations,
    load_sam2_model,
    create_snip_id,
    convert_sam2_mask_to_rle,
    convert_sam2_mask_to_polygon,
    extract_bbox_from_mask,
    run_sam2_propagation,
    run_bidirectional_propagation,
    process_single_video_from_annotations
)

__all__ = [
    # GroundingDINO utilities
    'GroundedDinoAnnotations',
    'load_groundingdino_model', 
    'load_config',
    'get_model_metadata',
    'calculate_detection_iou',
    'run_inference',
    'visualize_detections',
    'gdino_inference_with_visualization',
    
    # SAM2 utilities
    'GroundedSamAnnotations',
    'load_sam2_model',
    'create_snip_id',
    'convert_sam2_mask_to_rle',
    'convert_sam2_mask_to_polygon',
    'extract_bbox_from_mask',
    'run_sam2_propagation',
    'run_bidirectional_propagation',
    'process_single_video_from_annotations'
]
