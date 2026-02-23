"""
Simple mask utilities for segmentation encoding/decoding and manipulation.
Handles RLE, polygon formats and consolidates bbox/area calculations.

This module provides a centralized, simple approach to mask operations
without the complexity of the previous sam2_utils implementation.
"""

import numpy as np
import cv2
from typing import Dict, List, Union, Tuple
import base64


def encode_mask_rle(binary_mask: np.ndarray) -> Dict:
    """
    Simple RLE encoding using pycocotools - the clean way.
    
    Args:
        binary_mask: Binary mask as numpy array
        
    Returns:
        Dict with 'counts' (base64 string), 'size' (2D list)
    """
    try:
        from pycocotools import mask as mask_utils
        import base64
    except ImportError:
        raise ImportError("pycocotools required for RLE encoding")
    
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Ensure binary mask is 2D
    if binary_mask.ndim > 2:
        binary_mask = binary_mask.squeeze()
        if binary_mask.ndim > 2:
            binary_mask = binary_mask[0]  # Take first channel if still >2D
    
    # Standard pycocotools encoding - keep it simple!
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    
    # Handle unusual case where pycocotools returns a list
    if isinstance(rle, list):
        if len(rle) > 0:
            rle = rle[0]  # Take first RLE
        else:
            raise ValueError("pycocotools returned empty list for RLE encoding")
    
    # Convert bytes to base64 string for JSON serialization
    if isinstance(rle['counts'], bytes):
        rle['counts'] = base64.b64encode(rle['counts']).decode('utf-8')
    
    return rle


def decode_mask_rle(rle_data: Dict) -> np.ndarray:
    """
    Decode RLE data handling various formats we might encounter.
    Supports both new (bytes) and old (base64 strings) formats.
    
    Args:
        rle_data: Dict with 'counts' and 'size' fields
        
    Returns:
        Binary mask as numpy array
        
    Raises:
        ValueError: If RLE data is corrupted and can't be decoded
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        raise ImportError("pycocotools required for RLE decoding")
    
    # Make a copy to avoid modifying original
    rle_copy = rle_data.copy()
    
    # Handle the current broken format (base64 encoded strings)
    if isinstance(rle_copy['counts'], str):
        try:
            # Try base64 decode for existing data
            rle_copy['counts'] = base64.b64decode(rle_copy['counts'])
        except Exception as e:
            raise ValueError(f"Failed to decode base64 RLE counts: {e}")
    
    # Fix size array if needed (remove extra dimensions)
    if 'size' in rle_copy and len(rle_copy['size']) > 2:
        original_size = rle_copy['size']
        rle_copy['size'] = rle_copy['size'][-2:]  # Take last 2 dimensions (H, W)
        print(f"Fixed size array: {original_size} -> {rle_copy['size']}")
    
    # Attempt decoding with loud error on failure
    try:
        return mask_utils.decode(rle_copy)
    except Exception as e:
        raise ValueError(f"Failed to decode RLE mask: {e}. RLE data: counts type={type(rle_copy['counts'])}, size={rle_copy['size']}")


def mask_to_bbox(binary_mask: np.ndarray) -> List[float]:
    """
    Extract normalized bounding box from binary mask in xyxy format.
    
    Args:
        binary_mask: Binary mask as numpy array
        
    Returns:
        List of [x_min, y_min, x_max, y_max] normalized to [0,1]
    """
    # Ensure binary mask is 2D
    if binary_mask.ndim > 2:
        binary_mask = binary_mask.squeeze()
        if binary_mask.ndim > 2:
            binary_mask = binary_mask[0]  # Take first channel
    
    y_indices, x_indices = np.where(binary_mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
    
    h, w = binary_mask.shape
    return [x_min / w, y_min / h, x_max / w, y_max / h]


def mask_area(binary_mask: np.ndarray) -> float:
    """
    Calculate mask area (number of pixels).
    
    Args:
        binary_mask: Binary mask as numpy array
        
    Returns:
        Area in pixels as float
    """
    return float(np.sum(binary_mask > 0))


def mask_to_polygon(binary_mask: np.ndarray) -> List[List[float]]:
    """
    Convert binary mask to polygon format.
    
    Args:
        binary_mask: Binary mask as numpy array
        
    Returns:
        List of polygons, each polygon is a list of [x,y,x,y,...] coordinates
    """
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Valid polygon needs at least 3 points
            polygon = contour.flatten().astype(float).tolist()
            polygons.append(polygon)
    
    return polygons


def polygon_to_mask(polygons: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    Convert polygon to binary mask.
    
    Args:
        polygons: List of polygons, each polygon is [x,y,x,y,...] coordinates
        height: Height of output mask
        width: Width of output mask
        
    Returns:
        Binary mask as numpy array
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for polygon in polygons:
        if len(polygon) >= 6:  # At least 3 points (x,y pairs)
            points = np.array(polygon).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
    
    return mask


def encode_mask_rle_full_info(binary_mask: np.ndarray, format: str = "rle") -> Dict:
    """
    Create complete segmentation object with RLE + derived info (area, bbox).
    This is the new self-contained format.
    
    Args:
        binary_mask: Binary mask as numpy array
        format: Format type ("rle" or "polygon")
        
    Returns:
        Complete segmentation dict with format, counts/polygons, size, area, bbox
    """
    if format == "rle":
        # Get RLE encoding
        rle = encode_mask_rle(binary_mask)
        
        # Create complete segmentation object
        segmentation = {
            "counts": rle["counts"],
            "size": rle["size"],
            # Use canonical format name 'rle' for compatibility with sam2_utils
            # counts may be base64-encoded bytes/strings depending on encoder
            "format": "rle",
            "area": mask_area(binary_mask),
            "bbox": mask_to_bbox(binary_mask)
        }
        
    elif format == "polygon":
        polygons = mask_to_polygon(binary_mask)
        
        segmentation = {
            "polygons": polygons,
            "size": list(binary_mask.shape),
            "format": "polygon", 
            "area": mask_area(binary_mask),
            "bbox": mask_to_bbox(binary_mask)
        }
        
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'rle' or 'polygon'")
    
    return segmentation


def validate_rle_format(rle_data: Dict) -> bool:
    """
    Simple validation to check if RLE data has required fields.
    No strict validation - just basic sanity check.
    
    Args:
        rle_data: RLE data dictionary
        
    Returns:
        True if basic format is valid
    """
    required_keys = ['counts', 'size']
    return all(key in rle_data for key in required_keys)


def get_segmentation_format(segmentation: Dict, embryo_data: Dict = None) -> str:
    """
    Get segmentation format from either new location (segmentation.format) 
    or old location (embryo_data.segmentation_format) for backward compatibility.
    
    Args:
        segmentation: Segmentation dict
        embryo_data: Optional embryo data dict for backward compatibility
        
    Returns:
        Format string ("rle", "polygon", or "unknown")
    """
    # Check new location first, then fall back to old field for backward compatibility
    format_str = segmentation.get("format") or segmentation.get("segmentation_format")
    if not format_str and embryo_data:
        # older records stored format on embryo_data.segmentation_format
        format_str = embryo_data.get("segmentation_format")

    if not format_str:
        return "unknown"

    # Normalize common variants to canonical values used across the codebase
    f = str(format_str).lower()
    if "rle" in f:
        return "rle"
    if "polygon" in f or "polygons" in f:
        return "polygon"

    # Unknown/other - return lowercased raw string for visibility
    return f
