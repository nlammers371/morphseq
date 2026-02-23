#!/usr/bin/env python3
"""
Segmentation Quality QC Module

Validates SAM2 mask quality by detecting common segmentation issues:
- Edge contact: Masks touching image boundaries (incomplete embryos)
- Discontinuous masks: Multiple disconnected components (tracking errors)
- Overlapping masks: Masks from different embryos overlapping (ID confusion)

This module operates on segmentation_tracking.csv and outputs segmentation_quality_qc.csv.

Authors: Wave 6 Implementation
Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
import warnings
import json

# Optional imports for discontinuous mask detection
try:
    from pycocotools import mask as mask_utils
    from skimage.measure import label
    _HAS_IMAGE_LIBS = True
except ImportError:
    _HAS_IMAGE_LIBS = False
    warnings.warn("skimage/pycocotools not available - discontinuous mask check disabled")


def decode_mask_rle(rle_data: Union[dict, str]) -> np.ndarray:
    """
    Decode RLE mask to binary array.

    Handles both dict and JSON string formats.

    Args:
        rle_data: RLE data as dict (pycocotools format) or JSON string

    Returns:
        Binary mask as np.ndarray
    """
    if not _HAS_IMAGE_LIBS:
        raise ImportError("pycocotools required for mask decoding")

    # Handle JSON string format (from csv_formatter)
    if isinstance(rle_data, str):
        if rle_data.strip() == "":
            raise ValueError("Empty RLE string")
        try:
            rle_data = json.loads(rle_data)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in RLE string: {rle_data}")

    # Handle dict format
    if isinstance(rle_data, dict):
        if 'counts' in rle_data and 'size' in rle_data:
            # Already in pycocotools format
            mask = mask_utils.decode(rle_data)
        else:
            raise ValueError(f"Unknown RLE format: {rle_data.keys()}")
    else:
        raise TypeError(f"Expected dict or str, got {type(rle_data)}")

    return mask.astype(bool)


def check_mask_on_edge(mask: np.ndarray, margin_pixels: int = 2) -> Dict[str, bool]:
    """
    Check if mask touches image edges within margin.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask array
    margin_pixels : int, default 2
        Safety margin in pixels from edge

    Returns
    -------
    dict
        Boolean flags for each edge: {'top', 'bottom', 'left', 'right'}
    """
    height, width = mask.shape

    touches_edges = {
        'top': bool(np.any(mask[:margin_pixels, :])),
        'bottom': bool(np.any(mask[-margin_pixels:, :])),
        'left': bool(np.any(mask[:, :margin_pixels])),
        'right': bool(np.any(mask[:, -margin_pixels:]))
    }

    return touches_edges


def check_discontinuous_mask(mask: np.ndarray, min_component_fraction: float = 0.05) -> Dict[str, Any]:
    """
    Check if mask has multiple disconnected components.

    Only flags masks with multiple SIGNIFICANT components (>5% of largest component).
    This filters out noise pixels while catching real tracking errors.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask array
    min_component_fraction : float, default 0.05
        Minimum size relative to largest component (5%)

    Returns
    -------
    dict
        - num_components: Total number of components
        - num_significant: Number of components >5% of largest
        - is_discontinuous: True if >1 significant component
        - component_areas: List of all component areas
    """
    if not _HAS_IMAGE_LIBS:
        return {
            'num_components': None,
            'num_significant': None,
            'is_discontinuous': False,
            'component_areas': None,
            'error': 'skimage not available'
        }

    # Label connected components
    labeled_mask = label(mask)
    num_components = int(np.max(labeled_mask))

    if num_components <= 1:
        return {
            'num_components': num_components,
            'num_significant': num_components,
            'is_discontinuous': False,
            'component_areas': [int(np.sum(mask))] if num_components == 1 else []
        }

    # Calculate component areas
    component_areas = [int(np.sum(labeled_mask == i)) for i in range(1, num_components + 1)]
    largest_area = max(component_areas)
    min_significant_area = largest_area * min_component_fraction

    # Count significant components
    num_significant = sum(1 for area in component_areas if area > min_significant_area)

    return {
        'num_components': num_components,
        'num_significant': num_significant,
        'is_discontinuous': num_significant > 1,
        'component_areas': component_areas,
        'largest_component_area': largest_area,
        'smallest_component_area': min(component_areas)
    }


def check_overlapping_masks_per_image(image_masks: Dict[str, np.ndarray], iou_threshold: float = 0.1) -> list:
    """
    Check for overlapping masks within a single image.

    Parameters
    ----------
    image_masks : dict
        {snip_id: mask_array} for all masks in one image
    iou_threshold : float, default 0.1
        IoU threshold for flagging (10% overlap)

    Returns
    -------
    list
        List of overlap dicts with snip_id pairs and IoU values
    """
    overlaps = []
    snip_ids = list(image_masks.keys())

    if len(snip_ids) < 2:
        return overlaps

    # Check all pairs
    for i in range(len(snip_ids)):
        for j in range(i + 1, len(snip_ids)):
            snip_id1 = snip_ids[i]
            snip_id2 = snip_ids[j]
            mask1 = image_masks[snip_id1]
            mask2 = image_masks[snip_id2]

            # Calculate IoU
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1 | mask2)

            if union > 0:
                iou = intersection / union

                if iou > iou_threshold:
                    overlaps.append({
                        'snip_id1': snip_id1,
                        'snip_id2': snip_id2,
                        'iou': float(iou),
                        'intersection_pixels': int(intersection),
                        'union_pixels': int(union)
                    })

    return overlaps


def compute_segmentation_quality_qc(
    segmentation_df: pd.DataFrame,
    margin_pixels: int = 2,
    iou_threshold: float = 0.1,
    min_component_fraction: float = 0.05
) -> pd.DataFrame:
    """
    Compute SAM2 segmentation quality flags.

    Generates per-snip flags for:
    - edge_flag: Mask touches image boundary
    - discontinuous_mask_flag: Mask has multiple disconnected parts
    - overlapping_mask_flag: Mask overlaps with another embryo

    Parameters
    ----------
    segmentation_df : pd.DataFrame
        segmentation_tracking.csv with columns:
        [snip_id, embryo_id, image_id, mask_rle, ...]
    margin_pixels : int, default 2
        Edge detection margin in pixels
    iou_threshold : float, default 0.1
        IoU threshold for overlap detection (10%)
    min_component_fraction : float, default 0.05
        Minimum component size for discontinuous detection (5%)

    Returns
    -------
    pd.DataFrame
        QC flags with columns:
        [snip_id, embryo_id, image_id, edge_flag, discontinuous_mask_flag,
         overlapping_mask_flag, mask_quality_flag]

    Notes
    -----
    - mask_quality_flag = edge_flag OR discontinuous_mask_flag OR overlapping_mask_flag
    - Requires mask_rle column with RLE-encoded masks
    - Discontinuous check requires skimage (gracefully disabled if missing)
    """
    print(f"🔍 Computing segmentation quality QC for {len(segmentation_df)} snips...")

    # Initialize output dataframe
    qc_df = segmentation_df[['snip_id', 'embryo_id', 'image_id']].copy()
    qc_df['edge_flag'] = False
    qc_df['discontinuous_mask_flag'] = False
    qc_df['overlapping_mask_flag'] = False

    # Check required columns
    if 'mask_rle' not in segmentation_df.columns:
        print("⚠️  Warning: mask_rle column not found, skipping segmentation QC")
        qc_df['mask_quality_flag'] = False
        return qc_df

    # Process each snip
    edge_count = 0
    discontinuous_count = 0

    for idx, row in segmentation_df.iterrows():
        snip_id = row['snip_id']
        mask_rle = row['mask_rle']

        if pd.isna(mask_rle):
            continue

        try:
            # Decode mask
            mask = decode_mask_rle(mask_rle)

            # Check edge contact
            edge_result = check_mask_on_edge(mask, margin_pixels)
            if any(edge_result.values()):
                qc_df.loc[qc_df['snip_id'] == snip_id, 'edge_flag'] = True
                edge_count += 1

            # Check discontinuous mask
            if _HAS_IMAGE_LIBS:
                disc_result = check_discontinuous_mask(mask, min_component_fraction)
                if disc_result['is_discontinuous']:
                    qc_df.loc[qc_df['snip_id'] == snip_id, 'discontinuous_mask_flag'] = True
                    discontinuous_count += 1

        except Exception as e:
            print(f"⚠️  Warning: Failed to process snip {snip_id}: {e}")
            continue

    # Check overlapping masks (per-image)
    overlap_count = 0
    for image_id in segmentation_df['image_id'].unique():
        image_snips = segmentation_df[segmentation_df['image_id'] == image_id]

        if len(image_snips) < 2:
            continue

        # Decode all masks for this image
        image_masks = {}
        for idx, row in image_snips.iterrows():
            snip_id = row['snip_id']
            mask_rle = row['mask_rle']

            if pd.isna(mask_rle):
                continue

            try:
                image_masks[snip_id] = decode_mask_rle(mask_rle)
            except Exception:
                continue

        # Check for overlaps
        overlaps = check_overlapping_masks_per_image(image_masks, iou_threshold)

        for overlap in overlaps:
            snip_id1 = overlap['snip_id1']
            snip_id2 = overlap['snip_id2']
            qc_df.loc[qc_df['snip_id'].isin([snip_id1, snip_id2]), 'overlapping_mask_flag'] = True
            overlap_count += 2  # Both snips flagged

    # Compute composite mask_quality_flag
    qc_df['mask_quality_flag'] = (
        qc_df['edge_flag'] |
        qc_df['discontinuous_mask_flag'] |
        qc_df['overlapping_mask_flag']
    )

    # Summary
    total_flagged = qc_df['mask_quality_flag'].sum()
    print(f"✅ Segmentation QC complete:")
    print(f"   Edge contact: {edge_count} snips")
    print(f"   Discontinuous: {discontinuous_count} snips")
    print(f"   Overlapping: {overlap_count//2} pairs ({overlap_count} snips)")
    print(f"   Total flagged: {total_flagged} snips ({100*total_flagged/len(qc_df):.1f}%)")

    if not _HAS_IMAGE_LIBS:
        print(f"   ⚠️  Discontinuous check disabled (skimage not available)")

    return qc_df


def main():
    """Example usage and testing."""
    print("Segmentation Quality QC Module")
    print("=" * 50)
    print("Validates SAM2 mask quality:")
    print("- edge_flag: Masks touching boundaries")
    print("- discontinuous_mask_flag: Multiple components")
    print("- overlapping_mask_flag: Mask overlaps between embryos")
    print("\nUsage:")
    print("  from quality_control.segmentation_qc import compute_segmentation_quality_qc")
    print("  qc_df = compute_segmentation_quality_qc(segmentation_tracking_df)")


if __name__ == "__main__":
    main()
