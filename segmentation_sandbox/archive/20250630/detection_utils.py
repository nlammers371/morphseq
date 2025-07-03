#!/usr/bin/env python
"""
Detection utilities for the MorphSeq embryo segmentation pipeline.
Handles GroundingDINO detection, SAM2 segmentation, and detection filtering.
"""

import os
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
from dataclasses import dataclass
from .mask_utils import calculate_bbox_mask_overlap, calculate_mask_coverage


@dataclass
class Detection:
    """Data class for detection results."""
    bbox: List[float]  # [x1, y1, x2, y2] normalized
    confidence: float
    mask: Optional[np.ndarray] = None
    mask_confidence: Optional[float] = None
    category_id: int = 1  # Default to embryo
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        result = {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'category_id': self.category_id
        }
        if self.mask_confidence is not None:
            result['mask_confidence'] = self.mask_confidence
        return result


class DetectionProcessor:
    """Handles detection processing and filtering."""
    
    def __init__(self, config_params: Dict[str, Any]):
        """
        Initialize detection processor.
        
        Args:
            config_params: Detection configuration parameters
        """
        self.config = config_params
        self.box_threshold = config_params.get('box_threshold', 0.46)
        self.text_threshold = config_params.get('text_threshold', 0.2)
        self.bbox_mask_overlap_threshold = config_params.get('bbox_mask_overlap_threshold', 0.10)
        self.sam2_embryo_mask_overlap_threshold = config_params.get('sam2_embryo_mask_overlap_threshold', 0.30)
        self.iou_threshold = config_params.get('iou_threshold', 0.1)
        
    def filter_detections_by_confidence(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter detections by confidence threshold.
        
        Args:
            detections: List of detection objects
            
        Returns:
            Filtered detections
        """
        return [det for det in detections if det.confidence >= self.box_threshold]
    
    def filter_detections_by_mask_overlap(self, detections: List[Detection], 
                                        reference_mask: np.ndarray) -> List[Detection]:
        """
        Filter detections based on overlap with reference mask.
        
        Args:
            detections: List of detection objects
            reference_mask: Reference mask (embryo mask)
            
        Returns:
            Filtered detections
        """
        if reference_mask is None:
            return detections
        
        filtered_detections = []
        
        for detection in detections:
            # Check bbox overlap with reference mask
            bbox_overlap = calculate_bbox_mask_overlap(detection.bbox, reference_mask)
            
            if bbox_overlap >= self.bbox_mask_overlap_threshold:
                # If detection has a mask, check mask overlap too
                if detection.mask is not None:
                    mask_coverage = calculate_mask_coverage(detection.mask, reference_mask)
                    if mask_coverage >= self.sam2_embryo_mask_overlap_threshold:
                        filtered_detections.append(detection)
                else:
                    # No mask, just use bbox overlap
                    filtered_detections.append(detection)
        
        return filtered_detections
    
    def apply_non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """
        Apply non-maximum suppression to remove overlapping detections.
        
        Args:
            detections: List of detection objects
            
        Returns:
            Filtered detections after NMS
        """
        if len(detections) <= 1:
            return detections
        
        # Convert to format for NMS
        boxes = []
        scores = []
        
        for detection in detections:
            bbox = detection.bbox
            # Convert normalized coordinates to pixel coordinates (assuming some reference size)
            # For NMS, we'll use normalized coordinates directly
            boxes.append(bbox)
            scores.append(detection.confidence)
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Calculate IoU matrix
        ious = self._calculate_iou_matrix(boxes)
        
        # Apply NMS
        keep_indices = self._nms_indices(scores, ious, self.iou_threshold)
        
        return [detections[i] for i in keep_indices]
    
    def _calculate_iou_matrix(self, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU matrix for boxes."""
        n = len(boxes)
        ious = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                iou = self._calculate_bbox_iou(boxes[i], boxes[j])
                ious[i, j] = iou
                ious[j, i] = iou
        
        return ious
    
    def _calculate_bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _nms_indices(self, scores: np.ndarray, ious: np.ndarray, threshold: float) -> List[int]:
        """Get indices to keep after NMS."""
        keep = []
        indices = np.argsort(scores)[::-1]  # Sort by score descending
        
        while len(indices) > 0:
            # Keep the highest scoring detection
            keep.append(indices[0])
            
            if len(indices) == 1:
                break
            
            # Remove detections with high IoU
            remaining = []
            for i in indices[1:]:
                if ious[indices[0], i] < threshold:
                    remaining.append(i)
            
            indices = np.array(remaining)
        
        return keep
    
    def process_detections(self, detections: List[Detection], 
                         reference_mask: Optional[np.ndarray] = None) -> List[Detection]:
        """
        Apply full detection processing pipeline.
        
        Args:
            detections: Raw detections
            reference_mask: Reference mask for filtering
            
        Returns:
            Processed detections
        """
        # Filter by confidence
        filtered_detections = self.filter_detections_by_confidence(detections)
        
        # Filter by mask overlap if reference mask provided
        if reference_mask is not None:
            filtered_detections = self.filter_detections_by_mask_overlap(
                filtered_detections, reference_mask
            )
        
        # Apply NMS
        final_detections = self.apply_non_max_suppression(filtered_detections)
        
        return final_detections


def convert_groundingdino_output(boxes: torch.Tensor, logits: torch.Tensor, 
                               phrases: List[str]) -> List[Detection]:
    """
    Convert GroundingDINO output to Detection objects.
    
    Args:
        boxes: Detection boxes tensor
        logits: Detection logits tensor
        phrases: Detection phrases
        
    Returns:
        List of Detection objects
    """
    detections = []
    
    if len(boxes) == 0:
        return detections
    
    # Convert tensors to numpy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    
    for i, (box, logit) in enumerate(zip(boxes, logits)):
        # Convert box format if needed (ensure [x1, y1, x2, y2])
        bbox = box.tolist()
        confidence = float(logit)
        
        detection = Detection(
            bbox=bbox,
            confidence=confidence,
            category_id=1  # Embryo category
        )
        detections.append(detection)
    
    return detections


def convert_sam2_output(masks: Union[torch.Tensor, np.ndarray], 
                       scores: Union[torch.Tensor, np.ndarray],
                       boxes: Union[torch.Tensor, np.ndarray]) -> List[Detection]:
    """
    Convert SAM2 output to Detection objects.
    
    Args:
        masks: SAM2 mask predictions
        scores: SAM2 mask scores
        boxes: Input boxes for SAM2
        
    Returns:
        List of Detection objects with masks
    """
    detections = []
    
    # Convert tensors to numpy
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    for i, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
        # Ensure mask is binary
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        detection = Detection(
            bbox=box.tolist(),
            confidence=1.0,  # Use SAM2 score for mask confidence
            mask=mask_binary,
            mask_confidence=float(score),
            category_id=1
        )
        detections.append(detection)
    
    return detections


def draw_detections(image: np.ndarray, detections: List[Detection], 
                   draw_masks: bool = True, draw_boxes: bool = True) -> np.ndarray:
    """
    Draw detections on image for visualization.
    
    Args:
        image: Input image
        detections: List of detections to draw
        draw_masks: Whether to draw masks
        draw_boxes: Whether to draw bounding boxes
        
    Returns:
        Image with detections drawn
    """
    if image is None:
        return None
    
    result_image = image.copy()
    h, w = image.shape[:2]
    
    # Ensure image is RGB
    if len(result_image.shape) == 2:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
    
    for i, detection in enumerate(detections):
        # Draw mask
        if draw_masks and detection.mask is not None:
            mask = detection.mask
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h))
            
            # Create colored overlay
            color = plt.cm.tab10(i % 10)[:3]  # Get color from colormap
            color = tuple(int(c * 255) for c in color)
            
            overlay = result_image.copy()
            overlay[mask > 0] = color
            result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
        
        # Draw bounding box
        if draw_boxes:
            bbox = detection.bbox
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            # Draw rectangle
            color = (0, 255, 0)  # Green for boxes
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            conf_text = f"{detection.confidence:.2f}"
            cv2.putText(result_image, conf_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result_image


def calculate_detection_statistics(detections: List[Detection]) -> Dict[str, Any]:
    """
    Calculate statistics for a list of detections.
    
    Args:
        detections: List of detections
        
    Returns:
        Statistics dictionary
    """
    if not detections:
        return {
            'count': 0,
            'mean_confidence': 0.0,
            'std_confidence': 0.0,
            'min_confidence': 0.0,
            'max_confidence': 0.0,
            'mean_mask_confidence': 0.0,
            'masks_available': 0
        }
    
    confidences = [det.confidence for det in detections]
    mask_confidences = [det.mask_confidence for det in detections 
                       if det.mask_confidence is not None]
    masks_count = sum(1 for det in detections if det.mask is not None)
    
    stats = {
        'count': len(detections),
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'masks_available': masks_count
    }
    
    if mask_confidences:
        stats['mean_mask_confidence'] = np.mean(mask_confidences)
        stats['std_mask_confidence'] = np.std(mask_confidences)
    else:
        stats['mean_mask_confidence'] = 0.0
        stats['std_mask_confidence'] = 0.0
    
    return stats


# Example usage and testing
if __name__ == "__main__":
    # Test detection utilities
    print("Testing detection utilities...")
    
    # Create sample detections
    detections = [
        Detection([0.1, 0.1, 0.3, 0.3], 0.8),
        Detection([0.2, 0.2, 0.4, 0.4], 0.9),
        Detection([0.5, 0.5, 0.7, 0.7], 0.7),
    ]
    
    # Test detection processor
    config = {
        'box_threshold': 0.5,
        'iou_threshold': 0.3
    }
    
    processor = DetectionProcessor(config)
    
    # Test confidence filtering
    filtered = processor.filter_detections_by_confidence(detections)
    print(f"Detections after confidence filtering: {len(filtered)}")
    
    # Test NMS
    nms_filtered = processor.apply_non_max_suppression(detections)
    print(f"Detections after NMS: {len(nms_filtered)}")
    
    # Test statistics
    stats = calculate_detection_statistics(detections)
    print(f"Detection statistics: count={stats['count']}, mean_conf={stats['mean_confidence']:.3f}")
    
    print("Detection utilities test completed")
