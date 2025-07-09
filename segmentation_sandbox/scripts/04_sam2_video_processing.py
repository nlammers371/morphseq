ß#!/usr/bin/env python3
"""
Stage 4: SAM2 Video Processing for Embryo Segmentation
=====================================================

This script implements Block 4 of the embryo segmentation pipeline:
- Uses high-quality GroundedDINO annotations to select seed frames
- Propagates embryo masks across video frames using SAM2
- Assigns unique embryo IDs and tracks them across frames
- Handles bidirectional propagation when seed frame is not the first frame

Features:
- Video grouping and seed frame selection
- SAM2 video predictor integration
- Bidirectional mask propagation
- Embryo ID assignment and tracking
- Quality control and validation
- Comprehensive metadata tracking

Usage:
    python scripts/04_sam2_video_processing.py \
      --config /path/to/pipeline_config.yaml \
      --metadata /path/to/experiment_metadata.json \
      --annotations /path/to/gdino_high_quality_annotations.json \
      --output /path/to/grounded_sam_annotations.json
"""

import os
import sys
import json
import yaml
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict
import cv2
import tempfile
import shutil

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
SANDBOX_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SANDBOX_ROOT))

# Add SAM2 to path
SAM2_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/image_segmentation/sam2")
sys.path.append(str(SAM2_ROOT))

# Import utilities
from scripts.utils.grounded_sam_utils import load_config, GroundedDinoAnnotations
from scripts.utils.experiment_metadata_utils import (
    load_experiment_metadata, get_image_id_paths, list_image_ids_for_video
)

# Import SAM2
try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAM2 not available: {e}")
    SAM2_AVAILABLE = False


def load_sam2_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """
    Load SAM2 video predictor model.
    
    Args:
        config_path: Path to SAM2 config file
        checkpoint_path: Path to SAM2 checkpoint
        device: Device to load model on
        
    Returns:
        SAM2 video predictor instance
    """
    print(f"🔧 Loading SAM2 model...")
    print(f"   Config: {config_path}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Device: {device}")
    
    try:
        from sam2.build_sam import build_sam2_video_predictor
        
        predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)
        print(f"✅ SAM2 model loaded successfully")
        return predictor
        
    except ImportError as e:
        raise ImportError(f"Failed to import SAM2. Make sure it's installed: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM2 model: {e}")


def prepare_video_frames(video_id: str, metadata: Dict) -> Tuple[Path, List[str], Dict[str, int], Dict]:
    """
    Prepare video information for SAM2 processing using experiment metadata.
    
    This function consolidates video preparation logic, providing all necessary outputs
    for SAM2 processing including directory path, image IDs, frame mapping, and metadata.
    
    Args:
        video_id: Video identifier
        metadata: Experiment metadata dictionary
        
    Returns:
        Tuple of (video_directory, image_ids, image_id_to_frame_index_mapping, video_metadata)
    """
    # Get video metadata from experiment metadata
    video_info = None
    for exp_id, exp_data in metadata.get("experiments", {}).items():
        for vid_id, vid_data in exp_data.get("videos", {}).items():
            if vid_id == video_id:
                video_info = vid_data
                break
        if video_info:
            break
    
    if not video_info:
        raise ValueError(f"Video {video_id} not found in experiment metadata")
    
    # Get the processed images directory from metadata
    processed_jpg_images_dir = Path(video_info["processed_jpg_images_dir"])
    if not processed_jpg_images_dir.exists():
        raise FileNotFoundError(f"Processed images directory not found: {processed_jpg_images_dir}")
    
    # Get all image IDs for this video in correct order
    image_ids = video_info.get("image_ids", [])
    if not image_ids:
        raise ValueError(f"No images found for video_id: {video_id}")
    
    # Verify that the images exist in the processed directory
    missing_images = []
    for image_id in image_ids:
        image_path = processed_jpg_images_dir / f"{image_id}.jpg"
        if not image_path.exists():
            missing_images.append(image_id)
    
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} images in {processed_jpg_images_dir}: {missing_images[:5]}...")
    
    # Create mapping from image_id to frame index (SAM2 uses indices)
    image_id_to_frame_idx = {image_id: idx for idx, image_id in enumerate(image_ids)}
    
    print(f"📁 Video {video_id}: {len(image_ids)} frames in {processed_jpg_images_dir}")
    print(f"🎬 SAM2 video directory: {processed_jpg_images_dir}")
    print(f"   Frame range: 0 to {len(image_ids)-1}")
    
    return processed_jpg_images_dir, image_ids, image_id_to_frame_idx, video_info


def convert_sam2_mask_to_rle(binary_mask: np.ndarray) -> Dict:
    """Convert SAM2 binary mask to RLE format for compact storage."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        # Fallback to simple dict if pycocotools not available
        print("Warning: pycocotools not available, using simple mask storage")
        return {
            'format': 'simple_mask',
            'size': binary_mask.shape,
            'data': binary_mask.flatten().tolist()
        }
    
    # Convert to uint8 if needed
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Convert to Fortran order for COCO tools
    binary_mask_fortran = np.asfortranarray(binary_mask)
    
    # Encode to RLE
    rle = mask_utils.encode(binary_mask_fortran)
    
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle


def convert_sam2_mask_to_polygon(binary_mask: np.ndarray) -> List[List[float]]:
    """Convert SAM2 binary mask to polygon format."""
    import cv2
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Flatten contour points and convert to list
        if len(contour) >= 3:  # Need at least 3 points for a polygon
            polygon = contour.flatten().astype(float).tolist()
            polygons.append(polygon)
    
    return polygons


def extract_bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    """Extract bounding box from binary mask in normalized xywh format."""
    y_indices, x_indices = np.where(binary_mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Convert to normalized coordinates
    h, w = binary_mask.shape
    bbox_norm = [
        x_min / w,
        y_min / h,
        (x_max - x_min) / w,
        (y_max - y_min) / h
    ]
    
    return bbox_norm


def group_annotations_by_video(annotations: Dict, metadata: Dict) -> Dict[str, Dict]:
    """
    Group high-quality annotations by video_id.
    
    Args:
        annotations: High-quality annotations dictionary
        metadata: Experiment metadata
        
    Returns:
        Dictionary mapping video_id to image_id -> detections
    """
    print("🔗 Grouping annotations by video...")
    
    video_annotations = defaultdict(dict)
    
    # Get all image_ids from annotations
    for image_id, image_data in annotations.get("images", {}).items():
        # Extract video_id from image_id (format: experiment_well_frame)
        parts = image_id.split('_')
        if len(parts) >= 3:
            video_id = '_'.join(parts[:2])  # experiment_well
            
            # Extract individual embryo detections
            detections = []
            for annotation in image_data.get('annotations', []):
                if annotation.get('prompt') == 'individual embryo':
                    detections.extend(annotation.get('detections', []))
            
            if detections:
                video_annotations[video_id][image_id] = detections
    
    print(f"📊 Found annotations for {len(video_annotations)} videos")
    for video_id, image_annotations in video_annotations.items():
        print(f"  {video_id}: {len(image_annotations)} images with detections")
    
    return dict(video_annotations)


def find_seed_frame(video_annotations: Dict, metadata: Dict, video_id: str) -> Tuple[str, Dict]:
    """
    Find the optimal seed frame for a video.
    
    Args:
        video_annotations: Annotations grouped by video
        metadata: Experiment metadata
        video_id: Video identifier
        
    Returns:
        Tuple of (seed_frame_id, seed_info)
    """
    # Get all image_ids for this video from metadata
    video_info = None
    for exp_id, exp_data in metadata.get("experiments", {}).items():
        for vid_id, vid_data in exp_data.get("videos", {}).items():
            if vid_id == video_id:
                video_info = vid_data
                break
        if video_info:
            break
    
    if not video_info:
        raise ValueError(f"Video {video_id} not found in metadata")
    
    all_image_ids = video_info.get("image_ids", [])
    if not all_image_ids:
        raise ValueError(f"No image_ids found for video {video_id}")
    
    # Consider first 20% of frames
    first_20_percent = max(1, len(all_image_ids) // 5)
    early_frames = all_image_ids[:first_20_percent]
    
    # Count detections in early frames
    detection_counts = []
    frame_detection_info = {}
    
    video_ann = video_annotations.get(video_id, {})
    
    for image_id in early_frames:
        if image_id in video_ann:
            detections = video_ann[image_id]
            count = len(detections)
        else:
            detections = []
            count = 0
        
        detection_counts.append(count)
        frame_detection_info[image_id] = {
            'count': count,
            'detections': detections
        }
    
    if not detection_counts or max(detection_counts) == 0:
        raise ValueError(f"No detections found in early frames for {video_id}")
    
    # Find mode of detection counts
    count_freq = Counter(detection_counts)
    mode_count = count_freq.most_common(1)[0][0]
    
    # Find earliest frame with mode count
    seed_frame = None
    for image_id in early_frames:
        if frame_detection_info[image_id]['count'] == mode_count and mode_count > 0:
            seed_frame = image_id
            break
    
    if not seed_frame:
        raise ValueError(f"No suitable seed frame found for {video_id}")
    
    seed_info = {
        'video_id': video_id,
        'seed_frame': seed_frame,
        'num_embryos': mode_count,
        'detections': frame_detection_info[seed_frame]['detections'],
        'is_first_frame': seed_frame == all_image_ids[0],
        'all_frames': all_image_ids,
        'seed_frame_index': all_image_ids.index(seed_frame)
    }
    
    return seed_frame, seed_info

@ agent these sam2 functions should be moved to grounded sam utils and should have clear naming where sam2 and groundino is meant for the procesing
def assign_embryo_ids(video_id: str, num_embryos: int) -> List[str]:
    """
    Generate unique embryo IDs for a video.
    
    Args:
        video_id: Video identifier
        num_embryos: Number of embryos in the video
        
    Returns:
        List of embryo IDs
    """
    embryo_ids = []
    for i in range(num_embryos):
        if num_embryos >= 10:
            embryo_id = f"{video_id}_e{i+1:02d}"  # e01, e02, ..., e10, e11, ...
        else:
            embryo_id = f"{video_id}_e{i+1:01d}"  # e1, e2, ...
        embryo_ids.append(embryo_id)
    return embryo_ids


def run_sam2_propagation(predictor, video_dir: Path, seed_frame_idx: int, 
                        seed_detections: List[Dict], embryo_ids: List[str],
                        image_ids: List[str], segmentation_format: str = 'rle') -> Dict:
    """
    Run SAM2 propagation from seed frame using the actual processed images directory.
    
    Args:
        predictor: SAM2 video predictor
        video_dir: Directory containing processed video frames (from metadata)
        seed_frame_idx: Index of seed frame
        seed_detections: Detections from seed frame
        embryo_ids: Assigned embryo IDs
        image_ids: Ordered list of image IDs corresponding to frame indices
        segmentation_format: 'rle' (recommended) or 'polygon' for segmentation storage
        
    Returns:
        Dictionary mapping image_ids to embryo segmentation results
    """
    print(f"🔄 Running SAM2 propagation from frame {seed_frame_idx}...")
    print(f"   Video directory: {video_dir}")
    print(f"   Seed frame image_id: {image_ids[seed_frame_idx]}")
    
    # Initialize SAM2 inference state
    inference_state = predictor.init_state(video_path=str(video_dir))
    predictor.reset_state(inference_state)
    
    # Add bounding boxes from seed frame detections
    for embryo_idx, (detection, embryo_id) in enumerate(zip(seed_detections, embryo_ids)):
        # Convert xywh to xyxy format for SAM2
        x, y, w, h = detection["box_xywh"]
        bbox_xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)
        
        # Add box to SAM2
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=seed_frame_idx,
            obj_id=embryo_idx + 1,  # SAM2 object IDs start from 1
            box=bbox_xyxy
        )
        
        print(f"   Added embryo {embryo_id} (SAM2 obj_id: {embryo_idx + 1})")
    
    # Propagate through video
    video_segments = {}
    print(f"   Propagating through {len(image_ids)} video frames...")
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # Map frame index back to image_id
        if out_frame_idx < len(image_ids):
            image_id = image_ids[out_frame_idx]
            frame_results = {}
            
            for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                if obj_id <= len(embryo_ids):  # Valid embryo ID
                    embryo_id = embryo_ids[obj_id - 1]  # Convert back to 0-based indexing
                    
                    # Convert mask logits to binary mask
                    binary_mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                    
                    # Extract segmentation in specified format
                    if segmentation_format == 'rle':
                        segmentation = convert_sam2_mask_to_rle(binary_mask)
                    elif segmentation_format == 'polygon':
                        segmentation = convert_sam2_mask_to_polygon(binary_mask)
                    else:
                        raise ValueError(f"Unknown segmentation_format: {segmentation_format}")
                    
                    bbox = extract_bbox_from_mask(binary_mask)
                    area = float(np.sum(binary_mask))
                    
                    # Calculate mask confidence (mean of positive logits)
                    positive_logits = mask_logits[0][mask_logits[0] > 0]
                    mask_confidence = float(torch.mean(positive_logits)) if len(positive_logits) > 0 else 0.0
                    
                    frame_results[embryo_id] = {
                        "segmentation": segmentation,
                        "segmentation_format": segmentation_format,
                        "bbox": bbox,
                        "area": area,
                        "mask_confidence": mask_confidence
                    }
            
            video_segments[image_id] = frame_results
    
    print(f"✅ SAM2 propagation complete for {len(video_segments)} frames")
    return video_segments


def run_bidirectional_propagation(predictor, video_dir: Path, seed_frame_idx: int,
                                 seed_detections: List[Dict], embryo_ids: List[str],
                                 image_ids: List[str], segmentation_format: str = 'rle') -> Dict:
    """
    Run bidirectional SAM2 propagation when seed frame is not the first frame.
    
    Args:
        predictor: SAM2 video predictor
        video_dir: Directory containing processed video frames (from metadata)
        seed_frame_idx: Index of seed frame
        seed_detections: Detections from seed frame
        embryo_ids: Assigned embryo IDs
        image_ids: Ordered list of image IDs corresponding to frame indices
        segmentation_format: 'rle' (recommended) or 'polygon' for segmentation storage
        
    Returns:
        Combined segmentation results from both directions
    """
    print(f"🔄 Running bidirectional SAM2 propagation...")
    print(f"   Seed frame: {seed_frame_idx} ({image_ids[seed_frame_idx]})")
    print(f"   Total frames: {len(image_ids)}")
    
    # Forward propagation (seed to end) - use original video directory
    print("   🔜 Forward propagation (seed → end)")
    forward_results = run_sam2_propagation(predictor, video_dir, seed_frame_idx, 
                                          seed_detections, embryo_ids, image_ids, segmentation_format)
    
    # Backward propagation (seed to beginning)
    print("   🔙 Backward propagation (seed → beginning)")
    
    # Create temporary directory with properly ordered frames
    with tempfile.TemporaryDirectory() as temp_dir_str:
        backward_video_dir = Path(temp_dir_str) / "backward_frames"
        backward_video_dir.mkdir(parents=True)
        
        # Create frames to reverse: from seed_frame_idx down to 0
        frames_to_reverse = list(range(seed_frame_idx + 1))  # [0, 1, 2, ..., seed_frame_idx]
        frames_to_reverse.reverse()  # [seed_frame_idx, seed_frame_idx-1, ..., 1, 0]
        
        print(f"   📁 Reordering {len(frames_to_reverse)} frames for backward propagation")
        
        # Create sequentially named symlinks so SAM2 processes them in correct order
        backward_image_ids = []
        for new_idx, original_idx in enumerate(frames_to_reverse):
            original_image_id = image_ids[original_idx]
            backward_image_ids.append(original_image_id)
            
            # Source: original image file
            src_frame = video_dir / f"{original_image_id}.jpg"
            
            # Destination: sequential numbering (000000.jpg, 000001.jpg, ...)
            # This ensures SAM2 processes them in the order we want
            dst_frame = backward_video_dir / f"{new_idx:06d}.jpg"
            
            if src_frame.exists() and not dst_frame.exists():
                try:
                    dst_frame.symlink_to(src_frame.absolute())
                except OSError:
                    # Fallback to copying if symlink fails
                    shutil.copy2(src_frame, dst_frame)
        
        print(f"   🎯 Seed frame ({image_ids[seed_frame_idx]}) is now at index 0")
        
        # Run backward propagation with properly ordered frames
        backward_results = {}
        
        # Initialize SAM2 for backward directory
        inference_state = predictor.init_state(video_path=str(backward_video_dir))
        predictor.reset_state(inference_state)
        
        # Add bounding boxes from seed frame (now at index 0 in backward directory)
        print(f"   🎯 Adding {len(seed_detections)} embryo detections at frame 0 (seed)")
        for embryo_idx, (detection, embryo_id) in enumerate(zip(seed_detections, embryo_ids)):
            x, y, w, h = detection["box_xywh"]
            bbox_xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)
            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,  # Seed frame is now at index 0
                obj_id=embryo_idx + 1,
                box=bbox_xyxy
            )
            print(f"      Added {embryo_id} (SAM2 obj_id: {embryo_idx + 1})")
        
        # Propagate through backward video
        print(f"   🔄 Propagating through {len(backward_image_ids)} frames in backward order")
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # Map SAM2 frame index back to original image_id
            if out_frame_idx < len(backward_image_ids):
                original_image_id = backward_image_ids[out_frame_idx]
                frame_results = {}
                
                for obj_id, mask_logits in zip(out_obj_ids, out_mask_logits):
                    if obj_id <= len(embryo_ids):
                        embryo_id = embryo_ids[obj_id - 1]
                        
                        # Convert mask and extract features
                        binary_mask = (mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8)
                        
                        if segmentation_format == 'rle':
                            segmentation = convert_sam2_mask_to_rle(binary_mask)
                        else:
                            segmentation = convert_sam2_mask_to_polygon(binary_mask)
                        
                        bbox = extract_bbox_from_mask(binary_mask)
                        area = float(np.sum(binary_mask))
                        
                        positive_logits = mask_logits[0][mask_logits[0] > 0]
                        mask_confidence = float(torch.mean(positive_logits)) if len(positive_logits) > 0 else 0.0
                        
                        frame_results[embryo_id] = {
                            "segmentation": segmentation,
                            "segmentation_format": segmentation_format,
                            "bbox": bbox,
                            "area": area,
                            "mask_confidence": mask_confidence
                        }
                
                backward_results[original_image_id] = frame_results
    
    # Stitch results together
    print("   🧵 Stitching bidirectional results...")
    combined_results = {}
    
    # Add forward results (from seed frame onwards)
    for image_id, frame_results in forward_results.items():
        combined_results[image_id] = frame_results
    
    # Add backward results (before seed frame, avoiding duplicates)
    seed_image_id = image_ids[seed_frame_idx]
    for image_id, frame_results in backward_results.items():
        if image_id != seed_image_id:  # Skip seed frame (already in forward results)
            combined_results[image_id] = frame_results
    
    print(f"✅ Bidirectional propagation complete: {len(combined_results)} frames")
    return combined_results


def process_single_video(video_id: str, video_annotations: Dict, metadata: Dict,
                        predictor, processing_stats: Dict, segmentation_format: str = 'rle') -> Tuple[Dict, Dict]:
    """
    Process a single video with SAM2 segmentation using experiment metadata.
    
    Args:
        video_id: Video identifier
        video_annotations: Annotations for this video (image_id -> detections)
        metadata: Experiment metadata
        predictor: SAM2 video predictor
        processing_stats: Dictionary to update with processing statistics
        segmentation_format: 'rle' (recommended) or 'polygon' for segmentation storage
        
    Returns:
        Tuple of (sam2_results, video_metadata)
    """
    print(f"\n🎬 Processing video: {video_id}")
    
    try:
        # Get video directory and frame mapping from experiment metadata
        video_dir, image_ids, image_id_to_frame_idx, video_info = prepare_video_frames(video_id, metadata)
        
        # Find seed frame using the video annotations
        seed_frame_id, seed_info = find_seed_frame(
            {video_id: video_annotations}, metadata, video_id
        )
        
        # Get seed frame index and detections
        seed_frame_idx = image_id_to_frame_idx[seed_frame_id]
        seed_detections = video_annotations[seed_frame_id]
        
        # Assign embryo IDs
        num_embryos = len(seed_detections)
        embryo_ids = assign_embryo_ids(video_id, num_embryos)
        
        print(f"   📍 Seed frame: {seed_frame_id} (index {seed_frame_idx})")
        print(f"   🧬 Embryos: {num_embryos} ({', '.join(embryo_ids)})")
        print(f"   📁 Using processed images: {video_dir}")
        
        if seed_frame_idx != 0:
            print(f"   ⚠️  Seed frame is not first frame - will use bidirectional propagation")
        
        # Verify that the seed frame image exists
        seed_image_path = video_dir / f"{seed_frame_id}.jpg"
        if not seed_image_path.exists():
            raise FileNotFoundError(f"Seed frame image not found: {seed_image_path}")
        
        # Run SAM2 propagation
        if seed_frame_idx == 0:
            # Simple forward propagation (seed frame is first frame)
            sam2_results = run_sam2_propagation(
                predictor, video_dir, seed_frame_idx, seed_detections, embryo_ids, image_ids, segmentation_format
            )
        else:
            # Bidirectional propagation (seed frame is not first frame)
            sam2_results = run_bidirectional_propagation(
                predictor, video_dir, seed_frame_idx, seed_detections, embryo_ids, image_ids, segmentation_format
            )
        
        # Update processing statistics
        processing_stats["videos_processed"] += 1
        processing_stats["total_frames_processed"] += len(sam2_results)
        processing_stats["total_embryos_tracked"] += num_embryos
        
        if seed_frame_idx != 0:
            processing_stats["videos_with_non_first_seed"] += 1
        
        # Create video metadata
        video_metadata = {
            "video_id": video_id,
            "seed_info": seed_info,
            "embryo_ids": embryo_ids,
            "num_embryos": num_embryos,
            "frames_processed": len(sam2_results),
            "processed_jpg_images_dir": str(video_dir),
            "requires_bidirectional_propagation": seed_frame_idx != 0,
            "processing_timestamp": datetime.now().isoformat(),
            "sam2_success": True
        }
        
        print(f"   ✅ Processed {len(sam2_results)} frames with {num_embryos} embryos")
        
        return sam2_results, video_metadata
        
    except Exception as e:
        print(f"   ❌ Error processing video {video_id}: {e}")
        processing_stats["videos_failed"] += 1
        
        # Return empty results with error info
        error_metadata = {
            "video_id": video_id,
            "sam2_success": False,
            "error_message": str(e),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return {}, error_metadata


def main():
    parser = argparse.ArgumentParser(description="SAM2 video processing for embryo segmentation")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--annotations", required=True, help="Path to gdino_high_quality_annotations.json")
    parser.add_argument("--output", required=True, help="Path to output grounded_sam_annotations.json")
    parser.add_argument("--sam2-config", help="Path to SAM2 config file")
    parser.add_argument("--sam2-checkpoint", help="Path to SAM2 checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--segmentation-format", default="rle", choices=["rle", "polygon"],
                       help="Format for storing segmentation masks (rle is much more compact)")
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Maximum number of videos to process (for testing)")
    parser.add_argument("--video-ids", nargs="+", default=None,
                       help="Specific video IDs to process")
    
    args = parser.parse_args()
    
    print("🎬 SAM2 Video Processing for Embryo Segmentation")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Metadata: {args.metadata}")
    print(f"Annotations: {args.annotations}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Segmentation format: {args.segmentation_format}")
    
    if args.segmentation_format == 'rle':
        print("📦 RLE format provides much better compression than polygons")
    
    # Load configuration and metadata
    print("\n📁 Loading configuration and metadata...")
    config = load_config(args.config)
    metadata = load_experiment_metadata(args.metadata)
    
    # Get SAM2 model paths
    sam2_config_path = args.sam2_config
    sam2_checkpoint_path = args.sam2_checkpoint
    
    if not sam2_config_path:
        sam2_config_path = config.get('models', {}).get('sam2', {}).get('config_path', 'sam2_hiera_l.yaml')
    if not sam2_checkpoint_path:
        sam2_checkpoint_path = config.get('models', {}).get('sam2', {}).get('checkpoint_path', 'sam2_hiera_large.pt')
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load SAM2 model
    predictor = load_sam2_model(sam2_config_path, sam2_checkpoint_path, device)
    
    # Load high-quality annotations
    print("📁 Loading high-quality annotations...")
    with open(args.annotations, 'r') as f:
        annotations = json.load(f)
    
    print(f"   Found annotations for {len(annotations.get('images', {}))} images")
    
    # Group annotations by video
    video_annotations = group_annotations_by_video(annotations, metadata)
    
    # Filter videos if specified
    if args.video_ids:
        video_annotations = {vid: ann for vid, ann in video_annotations.items() 
                           if vid in args.video_ids}
        print(f"   Filtered to {len(video_annotations)} specified videos")
    
    if args.max_videos:
        video_ids = list(video_annotations.keys())[:args.max_videos]
        video_annotations = {vid: video_annotations[vid] for vid in video_ids}
        print(f"   Limited to {len(video_annotations)} videos for testing")
    
    # Processing statistics
    processing_stats = {
        "videos_processed": 0,
        "videos_failed": 0,
        "videos_with_non_first_seed": 0,
        "total_frames_processed": 0,
        "total_embryos_tracked": 0,
        "start_time": datetime.now().isoformat()
    }
    
    # Process each video
    print(f"\n🔄 Processing {len(video_annotations)} videos...")
    
    all_results = {
        'processing_metadata': {
            'timestamp': datetime.now().isoformat(),
            'script_version': '04_sam2_video_processing.py',
            'sam2_config': sam2_config_path,
            'sam2_checkpoint': sam2_checkpoint_path,
            'segmentation_format': args.segmentation_format,
            'total_videos': len(video_annotations),
            'processing_stats': processing_stats
        },
        'videos': {}
    }
    
    for video_idx, (video_id, video_ann) in enumerate(video_annotations.items(), 1):
        print(f"\n{'='*20} Video {video_idx}/{len(video_annotations)} {'='*20}")
        
        # Process video
        sam2_results, video_metadata = process_single_video(
            video_id, video_ann, metadata, predictor, processing_stats, args.segmentation_format
        )
        
        # Store results
        if sam2_results:
            all_results['videos'][video_id] = {
                'metadata': video_metadata,
                'segmentations': sam2_results
            }
        else:
            all_results['videos'][video_id] = {
                'metadata': video_metadata,
                'segmentations': {}
            }
        
        # Periodic save every 5 videos
        if video_idx % 5 == 0:
            print(f"💾 Intermediate save after {video_idx} videos...")
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    # Final save
    print(f"\n💾 Final save...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Update final statistics
    processing_stats["end_time"] = datetime.now().isoformat()
    processing_stats["total_videos"] = len(video_annotations)
    all_results['processing_metadata']['processing_stats'] = processing_stats
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print(f"\n🎯 Processing Complete!")
    print(f"=" * 30)
    print(f"Videos processed: {processing_stats['videos_processed']}")
    print(f"Videos failed: {processing_stats['videos_failed']}")
    print(f"Success rate: {processing_stats['videos_processed']/len(video_annotations)*100:.1f}%")
    print(f"Frames processed: {processing_stats['total_frames_processed']}")
    print(f"Embryos tracked: {processing_stats['total_embryos_tracked']}")
    print(f"Videos with non-first seed: {processing_stats['videos_with_non_first_seed']}")
    
    print(f"\n📁 Results saved to: {args.output}")
    print("Ready for downstream processing!")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/04_sam2_video_processing.py \
#   --config /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/configs/pipeline_config.yaml \
#   --metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json \
#   --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_high_quality_annotations.json \
#   --output /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_results/grounded_sam_annotations.json
