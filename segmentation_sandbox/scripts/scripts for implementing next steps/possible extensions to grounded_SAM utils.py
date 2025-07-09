#!/usr/bin/env python3
"""
Extensions for grounded_sam_utils.py
====================================

Additional utility functions for high-quality annotation filtering and SAM2 integration.
Add these functions to your existing grounded_sam_utils.py file.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from collections import Counter, defaultdict
import torch
from datetime import datetime
import json

# Add these functions to your grounded_sam_utils.py file:

def calculate_detection_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate IoU between two bounding boxes in xywh format.
    
    Args:
        box1: [x, y, w, h] format
        box2: [x, y, w, h] format
        
    Returns:
        IoU value between 0 and 1
    """
    # Convert to xyxy format
    x1_min, y1_min = box1[0], box1[1]
    x1_max, y1_max = box1[0] + box1[2], box1[1] + box1[3]
    
    x2_min, y2_min = box2[0], box2[1]
    x2_max, y2_max = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Calculate union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def remove_overlapping_detections(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Remove overlapping detections, keeping the one with highest confidence.
    
    Args:
        detections: List of detection dictionaries with 'box_xywh' and 'confidence' keys
        iou_threshold: IoU threshold for considering detections as overlapping
        
    Returns:
        Filtered list of detections
    """
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    for i, det in enumerate(sorted_detections):
        is_duplicate = False
        for kept_det in keep:
            iou = calculate_detection_iou(det['box_xywh'], kept_det['box_xywh'])
            if iou > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep.append(det)
    
    return keep


def filter_annotations_by_confidence(annotations: Dict, prompt: str = "individual embryo", 
                                   confidence_threshold: float = 0.5) -> Dict:
    """
    Filter annotations by confidence threshold for a specific prompt.
    
    Args:
        annotations: Annotation dictionary from GroundedDinoAnnotations
        prompt: Prompt to filter for
        confidence_threshold: Minimum confidence score
        
    Returns:
        Filtered annotations dictionary
    """
    filtered_annotations = {
        "file_info": annotations.get("file_info", {}),
        "images": {}
    }
    
    total_original = 0
    total_filtered = 0
    
    for image_id, image_data in annotations.get("images", {}).items():
        filtered_image_annotations = []
        
        for annotation in image_data.get("annotations", []):
            if annotation.get("prompt") == prompt:
                # Filter detections by confidence
                original_detections = annotation.get("detections", [])
                total_original += len(original_detections)
                
                filtered_detections = [
                    det for det in original_detections 
                    if det.get("confidence", 0) >= confidence_threshold
                ]
                
                if filtered_detections:
                    # Remove overlapping detections
                    filtered_detections = remove_overlapping_detections(filtered_detections)
                    total_filtered += len(filtered_detections)
                    
                    # Create new annotation with filtered detections
                    filtered_annotation = annotation.copy()
                    filtered_annotation["detections"] = filtered_detections
                    filtered_annotation["num_detections"] = len(filtered_detections)
                    filtered_annotation["filtering_applied"] = {
                        "confidence_threshold": confidence_threshold,
                        "iou_threshold": 0.5,
                        "original_count": len(original_detections),
                        "filtered_count": len(filtered_detections),
                        "filter_timestamp": datetime.now().isoformat()
                    }
                    
                    filtered_image_annotations.append(filtered_annotation)
            else:
                # Keep annotations for other prompts unchanged
                filtered_image_annotations.append(annotation)
        
        if filtered_image_annotations:
            filtered_annotations["images"][image_id] = {
                "annotations": filtered_image_annotations
            }
    
    # Add filtering summary
    filtered_annotations["filtering_summary"] = {
        "total_original_detections": total_original,
        "total_filtered_detections": total_filtered,
        "retention_rate": total_filtered / total_original if total_original > 0 else 0,
        "confidence_threshold": confidence_threshold,
        "filter_timestamp": datetime.now().isoformat()
    }
    
    print(f"Filtering complete: {total_original} → {total_filtered} detections "
          f"({total_filtered/total_original*100:.1f}% retained)")
    
    return filtered_annotations


def generate_quality_histogram(annotations: Dict, prompt: str = "individual embryo", 
                             save_path: Optional[str] = None,
                             metadata_path: Optional[str] = None,
                             experiment_ids: Optional[List[str]] = None,
                             video_ids: Optional[List[str]] = None,
                             image_ids: Optional[List[str]] = None) -> Dict:
    """
    Generate histogram of confidence scores for quality analysis with optional filtering.
    
    Args:
        annotations: Annotation dictionary
        prompt: Prompt to analyze
        save_path: Optional path to save histogram plot
        metadata_path: Path to experiment metadata (required for experiment/video filtering)
        experiment_ids: Optional list of experiment IDs to analyze
        video_ids: Optional list of video IDs to analyze  
        image_ids: Optional list of specific image IDs to analyze
        
    Returns:
        Dictionary with histogram statistics
    """
    from scripts.utils.experiment_metadata_utils import load_experiment_metadata, list_image_ids_for_video
    
    # Determine which images to analyze (reuse the helper function)
    if metadata_path and (experiment_ids or video_ids):
        # Convert experiment_ids and video_ids to image_ids
        target_image_ids = set()
        
        if image_ids:
            target_image_ids.update(image_ids)
        
        if experiment_ids:
            metadata = load_experiment_metadata(metadata_path)
            for exp_id in experiment_ids:
                exp_data = metadata.get("experiments", {}).get(exp_id, {})
                for video_id, video_data in exp_data.get("videos", {}).items():
                    video_image_ids = video_data.get("image_ids", [])
                    target_image_ids.update(video_image_ids)
        
        if video_ids:
            for video_id in video_ids:
                video_image_ids = list_image_ids_for_video(video_id, metadata_path)
                target_image_ids.update(video_image_ids)
    elif image_ids:
        target_image_ids = set(image_ids)
    else:
        target_image_ids = None
    
    # Collect confidence scores from target images only
    confidence_scores = []
    
    for image_id, image_data in annotations.get("images", {}).items():
        # Skip images not in our target set
        if target_image_ids and image_id not in target_image_ids:
            continue
            
        for annotation in image_data.get("annotations", []):
            if annotation.get("prompt") == prompt:
                for detection in annotation.get("detections", []):
                    confidence_scores.append(detection.get("confidence", 0))
    
    if not confidence_scores:
        print(f"No detections found for prompt: {prompt} in filtered images")
        return {}
    
    # Calculate statistics (rest of function remains the same)
    scores_array = np.array(confidence_scores)
    stats = {
        "total_detections": len(confidence_scores),
        "mean_confidence": float(np.mean(scores_array)),
        "median_confidence": float(np.median(scores_array)),
        "std_confidence": float(np.std(scores_array)),
        "min_confidence": float(np.min(scores_array)),
        "max_confidence": float(np.max(scores_array)),
        "percentiles": {
            "25th": float(np.percentile(scores_array, 25)),
            "75th": float(np.percentile(scores_array, 75)),
            "90th": float(np.percentile(scores_array, 90)),
            "95th": float(np.percentile(scores_array, 95))
        }
    }
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(confidence_scores, bins=50, alpha=0.7, edgecolor='black')
    
    # Add vertical lines for key statistics
    plt.axvline(stats["mean_confidence"], color='red', linestyle='--', 
                label=f'Mean: {stats["mean_confidence"]:.3f}')
    plt.axvline(stats["median_confidence"], color='green', linestyle='--', 
                label=f'Median: {stats["median_confidence"]:.3f}')
    plt.axvline(0.5, color='orange', linestyle='-', 
                label='Potential Threshold: 0.5')
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    
    # Create title with filter info
    title = f'Confidence Score Distribution for "{prompt}"\nTotal Detections: {stats["total_detections"]}'
    if experiment_ids:
        title += f'\nExperiments: {", ".join(experiment_ids[:3])}{"..." if len(experiment_ids) > 3 else ""}'
    elif video_ids:
        title += f'\nVideos: {", ".join(video_ids[:3])}{"..." if len(video_ids) > 3 else ""}'
    elif image_ids:
        title += f'\nImages: {len(image_ids)} selected'
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    textstr = f'''Statistics:
Mean: {stats["mean_confidence"]:.3f}
Median: {stats["median_confidence"]:.3f}
Std: {stats["std_confidence"]:.3f}
95th %ile: {stats["percentiles"]["95th"]:.3f}'''
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    plt.show()
    
    return stats


def group_annotations_by_video(annotations: Dict, metadata_path: Union[str, Path], 
                              prompt: str = "individual embryo") -> Dict[str, Dict]:
    """
    Group annotations by video_id using experiment metadata.
    
    Args:
        annotations: Annotation dictionary
        metadata_path: Path to experiment_metadata.json
        prompt: Prompt to group by
        
    Returns:
        Dictionary mapping video_id to grouped annotations
    """
    from scripts.utils.experiment_metadata_utils import load_experiment_metadata
    
    # Load metadata
    metadata = load_experiment_metadata(metadata_path)
    
    # Create mapping from image_id to video_id
    image_to_video = {}
    for exp_id, exp_data in metadata.get("experiments", {}).items():
        for video_id, video_data in exp_data.get("videos", {}).items():
            for image_id in video_data.get("image_ids", []):
                image_to_video[image_id] = video_id
    
    # Group annotations by video
    video_annotations = defaultdict(lambda: defaultdict(list))
    
    for image_id, image_data in annotations.get("images", {}).items():
        video_id = image_to_video.get(image_id)
        if not video_id:
            print(f"Warning: No video_id found for image_id: {image_id}")
            continue
            
        for annotation in image_data.get("annotations", []):
            if annotation.get("prompt") == prompt:
                video_annotations[video_id][image_id] = annotation.get("detections", [])
    
    return dict(video_annotations)


def find_seed_frame(video_annotations: Dict[str, List], metadata_path: Union[str, Path], 
                   video_id: str, first_frames_percent: float = 0.2) -> Tuple[str, Dict]:
    """
    Find the optimal seed frame for SAM2 processing.
    
    Args:
        video_annotations: Annotations grouped by video
        metadata_path: Path to experiment metadata
        video_id: Video ID to process
        first_frames_percent: Percentage of first frames to consider
        
    Returns:
        Tuple of (seed_frame_image_id, seed_frame_info)
    """
    from scripts.utils.experiment_metadata_utils import (
        load_experiment_metadata, list_image_ids_for_video
    )
    
    # Get all image_ids for this video in correct order
    all_image_ids = list_image_ids_for_video(video_id, metadata_path)
    if not all_image_ids:
        raise ValueError(f"No images found for video_id: {video_id}")
    
    # Consider only first N% of frames
    num_first_frames = max(1, int(len(all_image_ids) * first_frames_percent))
    first_frame_ids = all_image_ids[:num_first_frames]
    
    # Count detections in first frames
    detection_counts = []
    frame_detection_info = {}
    
    for image_id in first_frame_ids:
        detections = video_annotations.get(video_id, {}).get(image_id, [])
        count = len(detections)
        detection_counts.append(count)
        frame_detection_info[image_id] = {
            "detection_count": count,
            "detections": detections
        }
    
    if not detection_counts:
        raise ValueError(f"No detections found in first frames for video: {video_id}")
    
    # Find mode (most common count)
    count_frequency = Counter(detection_counts)
    mode_count = count_frequency.most_common(1)[0][0]
    
    # Find earliest frame with mode count
    seed_frame_id = None
    for image_id in first_frame_ids:
        if frame_detection_info[image_id]["detection_count"] == mode_count:
            seed_frame_id = image_id
            break
    
    if seed_frame_id is None:
        raise ValueError(f"Could not find seed frame for video: {video_id}")
    
    # Calculate seed frame index
    seed_frame_index = all_image_ids.index(seed_frame_id)
    
    seed_info = {
        "seed_frame_image_id": seed_frame_id,
        "seed_frame_index": seed_frame_index,
        "mode_detection_count": mode_count,
        "total_video_frames": len(all_image_ids),
        "frames_analyzed": num_first_frames,
        "detection_counts": detection_counts,
        "count_frequency": dict(count_frequency)
    }
    
    print(f"Video {video_id}: Selected seed frame {seed_frame_id} "
          f"(index {seed_frame_index}) with {mode_count} detections")
    
    return seed_frame_id, seed_info


def assign_embryo_ids(video_id: str, num_embryos: int) -> List[str]:
    """
    Generate embryo IDs for a video.
    
    Args:
        video_id: Video identifier
        num_embryos: Number of embryos to assign IDs for
        
    Returns:
        List of embryo IDs
    """
    embryo_ids = []
    for i in range(1, num_embryos + 1):
        if i < 10:
            embryo_id = f"{video_id}_e0{i}"
        else:
            embryo_id = f"{video_id}_e{i:02d}"
        embryo_ids.append(embryo_id)
    
    return embryo_ids


def convert_sam2_mask_to_rle(mask: np.ndarray) -> Dict:
    """
    Convert SAM2 binary mask to RLE format (much more compact than polygons).
    
    Args:
        mask: Binary mask array
        
    Returns:
        RLE dictionary in COCO format
    """
    try:
        import pycocotools.mask as mask_utils
        # Ensure mask is in correct format (uint8, Fortran order)
        mask_fortran = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_utils.encode(mask_fortran)
        
        # Convert bytes to string for JSON serialization
        if isinstance(rle['counts'], bytes):
            rle['counts'] = rle['counts'].decode('utf-8')
        
        return rle
    except ImportError:
        # Fallback to manual RLE encoding
        return _manual_rle_encode(mask)


def _manual_rle_encode(binary_mask: np.ndarray) -> Dict:
    """Manual RLE encoding fallback when pycocotools not available."""
    h, w = binary_mask.shape
    mask_flat = binary_mask.flatten(order='F')  # Fortran order (column-major)
    
    # Find run lengths
    diff = np.diff(np.concatenate(([0], mask_flat, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Encode as alternating zeros and ones
    counts = []
    pos = 0
    
    for start, end in zip(starts, ends):
        # Add zeros before this run
        if start > pos:
            counts.append(start - pos)
        else:
            counts.append(0)
        
        # Add the run of ones
        counts.append(end - start)
        pos = end
    
    # Add remaining zeros
    if pos < len(mask_flat):
        counts.append(len(mask_flat) - pos)
    
    return {
        'size': [h, w],
        'counts': counts
    }


def convert_sam2_mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    Convert SAM2 binary mask to polygon segmentation.
    
    Note: This is kept for compatibility, but RLE format is recommended
    for better compression and storage efficiency.
    
    Args:
        mask: Binary mask array
        
    Returns:
        List of polygons in COCO format
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Simplify contour
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) >= 3:
            # Flatten and convert to list
            polygon = approx.flatten().tolist()
            if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                polygons.append(polygon)
    
    return polygons


def extract_bbox_from_mask(mask: np.ndarray) -> List[float]:
    """
    Extract bounding box from binary mask in xywh format.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Bounding box as [x, y, width, height]
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [0, 0, 0, 0]
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return [float(x), float(y), float(w), float(h)]


# Extension to GroundedDinoAnnotations class
class GroundedSamAnnotations(GroundedDinoAnnotations):
    """
    Extended version of GroundedDinoAnnotations with SAM2 integration for combined GroundedSAM workflow.
    """
    
    def add_sam2_results(self, image_id: str, embryo_results: Dict[str, Dict]):
        """
        Add SAM2 segmentation results to annotations.
        
        Args:
            image_id: Image identifier
            embryo_results: Dictionary mapping embryo_id to segmentation results
        """
        if image_id not in self.annotations["images"]:
            self.annotations["images"][image_id] = {"annotations": []}
        
        # Find existing embryo annotation or create new one
        existing_annotation = None
        for ann in self.annotations["images"][image_id]["annotations"]:
            if ann.get("prompt") == "individual embryo":
                existing_annotation = ann
                break
        
        if existing_annotation is None:
            # Create new annotation entry
            existing_annotation = {
                "annotation_id": f"ann_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
                "prompt": "individual embryo",
                "timestamp": datetime.now().isoformat(),
                "detections": [],
                "sam2_results": {}
            }
            self.annotations["images"][image_id]["annotations"].append(existing_annotation)
        
        # Add SAM2 results
        if "sam2_results" not in existing_annotation:
            existing_annotation["sam2_results"] = {}
        
        for embryo_id, result in embryo_results.items():
            existing_annotation["sam2_results"][embryo_id] = {
                "embryo_id": embryo_id,
                "segmentation": result.get("segmentation", []),
                "bbox": result.get("bbox", [0, 0, 0, 0]),
                "area": result.get("area", 0),
                "mask_confidence": result.get("mask_confidence", 0.0),
                "timestamp": datetime.now().isoformat()
            }
        
        self._unsaved_changes = True
    
    def add_video_processing_metadata(self, video_id: str, seed_info: Dict, 
                                    embryo_ids: List[str], processing_stats: Dict):
        """
        Add video-level processing metadata.
        
        Args:
            video_id: Video identifier
            seed_info: Seed frame information
            embryo_ids: List of assigned embryo IDs
            processing_stats: Processing statistics
        """
        if "video_metadata" not in self.annotations:
            self.annotations["video_metadata"] = {}
        
        self.annotations["video_metadata"][video_id] = {
            "video_id": video_id,
            "seed_frame_info": seed_info,
            "embryo_ids": embryo_ids,
            "num_embryos": len(embryo_ids),
            "processing_stats": processing_stats,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        self._unsaved_changes = True