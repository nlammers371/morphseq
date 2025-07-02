#!/usr/bin/env python
"""
Mask processing utilities for the MorphSeq embryo segmentation pipeline.
Handles mask loading, filtering, overlap calculations, and quality assessment.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology


def load_mask(mask_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load mask from file.
    
    Args:
        mask_path: Path to mask file
        
    Returns:
        Mask array or None if error
    """
    try:
        mask_path = Path(mask_path)
        if not mask_path.exists():
            return None
        
        # Load mask (handle different formats)
        if mask_path.suffix.lower() in ['.npy']:
            mask = np.load(mask_path)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Ensure binary mask
        if mask is not None:
            mask = (mask > 0).astype(np.uint8)
        
        return mask
    except Exception as e:
        print(f"Error loading mask from {mask_path}: {e}")
        return None


def calculate_bbox_mask_overlap(bbox: List[float], mask: np.ndarray) -> float:
    """
    Calculate overlap between bounding box and mask.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2] in normalized coordinates
        mask: Binary mask array
        
    Returns:
        Overlap ratio (bbox area overlapping with mask / total bbox area)
    """
    if mask is None:
        return 0.0
    
    h, w = mask.shape
    
    # Convert normalized coordinates to pixel coordinates
    x1 = int(bbox[0] * w)
    y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w)
    y2 = int(bbox[3] * h)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h-1, y2))
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Extract bbox region from mask
    bbox_mask = mask[y1:y2, x1:x2]
    
    # Calculate overlap
    bbox_area = (x2 - x1) * (y2 - y1)
    overlap_area = np.sum(bbox_mask)
    
    return overlap_area / bbox_area if bbox_area > 0 else 0.0


def calculate_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate overlap between two masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Overlap ratio (intersection / union)
    """
    if mask1 is None or mask2 is None:
        return 0.0
    
    # Ensure same size
    if mask1.shape != mask2.shape:
        return 0.0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0


def calculate_mask_coverage(detection_mask: np.ndarray, reference_mask: np.ndarray) -> float:
    """
    Calculate how much of detection mask overlaps with reference mask.
    
    Args:
        detection_mask: Detection mask (SAM2 output)
        reference_mask: Reference mask (embryo/yolk mask)
        
    Returns:
        Coverage ratio (overlapping detection area / total detection area)
    """
    if detection_mask is None or reference_mask is None:
        return 0.0
    
    if detection_mask.shape != reference_mask.shape:
        return 0.0
    
    detection_area = np.sum(detection_mask)
    if detection_area == 0:
        return 0.0
    
    overlap_area = np.sum(np.logical_and(detection_mask, reference_mask))
    return overlap_area / detection_area


def filter_masks_by_size(masks: List[np.ndarray], 
                        min_area: int = 100, 
                        max_area: Optional[int] = None) -> List[np.ndarray]:
    """
    Filter masks by area size.
    
    Args:
        masks: List of binary masks
        min_area: Minimum area threshold
        max_area: Maximum area threshold (None for no limit)
        
    Returns:
        Filtered list of masks
    """
    filtered_masks = []
    
    for mask in masks:
        if mask is None:
            continue
        
        area = np.sum(mask)
        if area >= min_area:
            if max_area is None or area <= max_area:
                filtered_masks.append(mask)
    
    return filtered_masks


def clean_mask(mask: np.ndarray, 
              min_hole_size: int = 50,
              min_object_size: int = 100) -> np.ndarray:
    """
    Clean mask by removing small holes and objects.
    
    Args:
        mask: Binary mask
        min_hole_size: Minimum hole size to fill
        min_object_size: Minimum object size to keep
        
    Returns:
        Cleaned mask
    """
    if mask is None:
        return None
    
    # Fill small holes
    mask_filled = morphology.remove_small_holes(
        mask.astype(bool), area_threshold=min_hole_size
    ).astype(np.uint8)
    
    # Remove small objects
    mask_cleaned = morphology.remove_small_objects(
        mask_filled.astype(bool), min_size=min_object_size
    ).astype(np.uint8)
    
    return mask_cleaned


def get_mask_properties(mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate properties of a binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Dictionary of mask properties
    """
    if mask is None:
        return {}
    
    properties = {}
    
    # Basic properties
    properties['area'] = np.sum(mask)
    properties['perimeter'] = 0
    properties['centroid_x'] = 0
    properties['centroid_y'] = 0
    properties['bbox_area'] = 0
    properties['solidity'] = 0
    properties['eccentricity'] = 0
    
    if properties['area'] > 0:
        # Use skimage regionprops for detailed analysis
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)
        
        if regions:
            # Take the largest region
            largest_region = max(regions, key=lambda r: r.area)
            
            properties['perimeter'] = largest_region.perimeter
            properties['centroid_x'] = largest_region.centroid[1]  # x is column
            properties['centroid_y'] = largest_region.centroid[0]  # y is row
            properties['bbox_area'] = largest_region.bbox_area
            properties['solidity'] = largest_region.solidity
            properties['eccentricity'] = largest_region.eccentricity
    
    return properties


def compare_masks(mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, float]:
    """
    Compare two masks and return similarity metrics.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        Dictionary of comparison metrics
    """
    metrics = {
        'iou': 0.0,
        'dice': 0.0,
        'coverage_1_by_2': 0.0,
        'coverage_2_by_1': 0.0,
        'area_ratio': 0.0
    }
    
    if mask1 is None or mask2 is None:
        return metrics
    
    if mask1.shape != mask2.shape:
        return metrics
    
    # Calculate areas
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    
    if area1 == 0 and area2 == 0:
        metrics['iou'] = 1.0
        metrics['dice'] = 1.0
        return metrics
    
    if area1 == 0 or area2 == 0:
        return metrics
    
    # Calculate intersection and union
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    
    # IoU (Intersection over Union)
    metrics['iou'] = intersection / union if union > 0 else 0.0
    
    # Dice coefficient
    metrics['dice'] = 2 * intersection / (area1 + area2)
    
    # Coverage metrics
    metrics['coverage_1_by_2'] = intersection / area1
    metrics['coverage_2_by_1'] = intersection / area2
    
    # Area ratio
    metrics['area_ratio'] = min(area1, area2) / max(area1, area2)
    
    return metrics


def create_overlay_image(image: np.ndarray, mask: np.ndarray, 
                        color: Tuple[int, int, int] = (255, 0, 0),
                        alpha: float = 0.3) -> np.ndarray:
    """
    Create overlay of mask on image.
    
    Args:
        image: Input image (grayscale or RGB)
        mask: Binary mask
        color: Overlay color (R, G, B)
        alpha: Transparency of overlay
        
    Returns:
        Overlay image
    """
    if image is None or mask is None:
        return image
    
    # Ensure image is RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 0] = color
    
    # Blend images
    overlay = cv2.addWeighted(image_rgb, 1-alpha, colored_mask, alpha, 0)
    
    return overlay


def extract_mask_from_coco_annotation(annotation: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Extract mask from COCO annotation.
    
    Args:
        annotation: COCO annotation dictionary
        image_shape: (height, width) of image
        
    Returns:
        Binary mask
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    if 'segmentation' in annotation:
        segmentation = annotation['segmentation']
        
        if isinstance(segmentation, list):
            # Polygon format
            for polygon in segmentation:
                polygon = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [polygon], 1)
        else:
            # RLE format (not implemented yet)
            print("RLE segmentation format not implemented")
    
    return mask


def get_mask_quality_score(mask: np.ndarray, 
                          min_area: int = 100,
                          max_holes: int = 5,
                          min_solidity: float = 0.7) -> float:
    """
    Calculate quality score for a mask.
    
    Args:
        mask: Binary mask
        min_area: Minimum acceptable area
        max_holes: Maximum acceptable number of holes
        min_solidity: Minimum acceptable solidity
        
    Returns:
        Quality score (0-1, higher is better)
    """
    if mask is None:
        return 0.0
    
    score = 1.0
    
    # Check area
    area = np.sum(mask)
    if area < min_area:
        score *= area / min_area
    
    # Check solidity and holes
    if area > 0:
        properties = get_mask_properties(mask)
        solidity = properties.get('solidity', 0)
        
        if solidity < min_solidity:
            score *= solidity / min_solidity
        
        # Count holes (approximate)
        filled_mask = ndimage.binary_fill_holes(mask)
        holes = np.sum(filled_mask) - area
        if holes > max_holes:
            score *= max_holes / holes
    
    return max(0.0, min(1.0, score))


# Example usage and testing
if __name__ == "__main__":
    # Test mask utilities with synthetic data
    print("Testing mask utilities...")
    
    # Create test masks
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[20:80, 20:80] = 1
    
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[30:90, 30:90] = 1
    
    # Test overlap calculation
    overlap = calculate_mask_overlap(mask1, mask2)
    print(f"Mask overlap: {overlap:.3f}")
    
    # Test mask properties
    properties = get_mask_properties(mask1)
    print(f"Mask1 area: {properties['area']}")
    
    # Test comparison
    comparison = compare_masks(mask1, mask2)
    print(f"IoU: {comparison['iou']:.3f}, Dice: {comparison['dice']:.3f}")
    
    # Test quality score
    quality = get_mask_quality_score(mask1)
    print(f"Quality score: {quality:.3f}")
    
    print("Mask utilities test completed")
