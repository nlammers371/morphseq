"""
GroundedSAM2 Segmentation Pipeline

Primary embryo segmentation pipeline combining:
- GroundingDINO: Zero-shot object detection for seed frames
- SAM2: Temporal mask propagation for video tracking

Modules:
    - frame_organization_for_sam2: Organize frames for SAM2 video processing
    - gdino_detection: GroundingDINO embryo detection
    - propagation: SAM2 mask propagation (forward/bidirectional)
    - mask_export: Export masks as labeled PNG images
    - csv_formatter: Flatten JSON results to CSV format
"""

from .frame_organization_for_sam2 import (
    organize_frames_for_sam2,
    cleanup_frame_directory,
    sam2_frame_context,
    prepare_bidirectional_propagation,
    remap_frame_indices,
    merge_bidirectional_results,
)

from .gdino_detection import (
    load_groundingdino_model,
    detect_embryos,
    filter_detections,
    select_seed_frame,
    convert_boxes_to_sam2_format,
    calculate_iou,
)

from .propagation import (
    load_sam2_model,
    propagate_forward,
    propagate_bidirectional,
    decode_sam2_masks,
    calculate_bbox_from_mask,
    encode_mask_to_rle,
    save_propagation_results,
)

from .mask_export import (
    create_labeled_mask_image,
    export_frame_masks,
    export_all_masks,
    load_labeled_mask,
    extract_individual_masks,
    visualize_masks,
    get_mask_statistics,
)

from .csv_formatter import (
    flatten_sam2_json_to_csv,
    add_well_metadata,
    encode_rle_for_csv,
    validate_csv_schema,
    load_sam2_json,
    export_sam2_to_csv,
    get_csv_summary,
    REQUIRED_CSV_COLUMNS,
)

__all__ = [
    # Frame organization
    "organize_frames_for_sam2",
    "cleanup_frame_directory",
    "sam2_frame_context",
    "prepare_bidirectional_propagation",
    "remap_frame_indices",
    "merge_bidirectional_results",
    # GroundingDINO detection
    "load_groundingdino_model",
    "detect_embryos",
    "filter_detections",
    "select_seed_frame",
    "convert_boxes_to_sam2_format",
    "calculate_iou",
    # SAM2 propagation
    "load_sam2_model",
    "propagate_forward",
    "propagate_bidirectional",
    "decode_sam2_masks",
    "calculate_bbox_from_mask",
    "encode_mask_to_rle",
    "save_propagation_results",
    # Mask export
    "create_labeled_mask_image",
    "export_frame_masks",
    "export_all_masks",
    "load_labeled_mask",
    "extract_individual_masks",
    "visualize_masks",
    "get_mask_statistics",
    # CSV formatting
    "flatten_sam2_json_to_csv",
    "add_well_metadata",
    "encode_rle_for_csv",
    "validate_csv_schema",
    "load_sam2_json",
    "export_sam2_to_csv",
    "get_csv_summary",
    "REQUIRED_CSV_COLUMNS",
]
