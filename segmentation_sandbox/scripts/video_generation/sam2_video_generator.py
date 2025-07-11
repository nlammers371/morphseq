#!/usr/bin/env python3
"""
SAM2 Video Generator from Annotations
=====================================

This script generates videos from SAM2 annotations by:
1. Loading segmentation masks from grounded_sam_annotations.json
2. Overlaying masks on original images with embryo tracking
3. Creating videos with color-coded embryos

Usage:
    python sam2_video_generator.py --annotations path/to/grounded_sam_annotations.json --output output_dir/ --max-videos 5
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

warnings.filterwarnings("ignore")


def load_annotations(annotations_path: str) -> Dict:
    """Load SAM2 annotations from JSON file."""
    print(f"📁 Loading annotations from: {annotations_path}")
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded annotations:")
    print(f"   - Experiments: {len(data.get('experiment_ids', []))}")
    print(f"   - Videos: {len(data.get('video_ids', []))}")
    print(f"   - Embryos: {len(data.get('embryo_ids', []))}")
    
    return data


def decode_rle_mask(rle: Dict, height: int, width: int) -> np.ndarray:
    """Decode RLE format mask to binary array."""
    try:
        from pycocotools import mask as mask_utils
        
        # Handle both dictionary and direct RLE formats
        if isinstance(rle, dict):
            if 'counts' in rle and 'size' in rle:
                # Standard COCO RLE format
                if isinstance(rle['counts'], str):
                    rle_copy = {'counts': rle['counts'].encode('utf-8'), 'size': rle['size']}
                else:
                    rle_copy = rle
                binary_mask = mask_utils.decode(rle_copy)
            else:
                # Handle simple mask format
                if 'data' in rle:
                    binary_mask = np.array(rle['data']).reshape(rle['size'])
                else:
                    raise ValueError("Unknown RLE format")
        else:
            raise ValueError("RLE must be a dictionary")
            
        return binary_mask.astype(np.uint8)
        
    except ImportError:
        # Fallback if pycocotools not available
        print("Warning: pycocotools not available, using fallback decoding")
        if isinstance(rle, dict) and 'data' in rle:
            return np.array(rle['data']).reshape(height, width).astype(np.uint8)
        else:
            raise ValueError("Cannot decode RLE without pycocotools")


def generate_embryo_colors(num_embryos: int) -> List[Tuple[int, int, int]]:
    """Generate distinct colors for each embryo."""
    # Use matplotlib's colormap for distinct colors
    if num_embryos <= 10:
        cmap = plt.cm.get_cmap('tab10')
    else:
        cmap = plt.cm.get_cmap('tab20')
    
    colors = []
    for i in range(num_embryos):
        color = cmap(i / max(num_embryos - 1, 1))[:3]  # Get RGB, ignore alpha
        # Convert to BGR for OpenCV and scale to 0-255
        bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        colors.append(bgr_color)
    
    return colors


def overlay_masks_on_image(image: np.ndarray, masks: Dict[str, Dict], 
                          embryo_colors: Dict[str, Tuple[int, int, int]],
                          alpha: float = 0.5) -> np.ndarray:
    """
    Overlay segmentation masks on image with embryo-specific colors.
    
    Args:
        image: Original image
        masks: Dictionary of embryo_id -> mask data
        embryo_colors: Dictionary of embryo_id -> BGR color
        alpha: Transparency for overlay (0-1)
        
    Returns:
        Image with overlaid masks
    """
    overlay = image.copy()
    mask_overlay = np.zeros_like(image)
    
    for embryo_id, mask_data in masks.items():
        if embryo_id not in embryo_colors:
            continue
            
        # Decode mask
        segmentation = mask_data.get('segmentation', {})
        if not segmentation:
            continue
            
        try:
            binary_mask = decode_rle_mask(segmentation, image.shape[0], image.shape[1])
            
            # Apply color to mask
            color = embryo_colors[embryo_id]
            mask_overlay[binary_mask > 0] = color
            
            # Draw bbox if available
            if 'bbox' in mask_data:
                bbox = mask_data['bbox']  # Normalized xywh format
                h, w = image.shape[:2]
                cx, cy, bw, bh = bbox
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                
                # Draw rectangle
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Add embryo ID label
                label = embryo_id.split('_')[-1]  # Get just the "e1", "e2" part
                cv2.putText(overlay, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        except Exception as e:
            print(f"Warning: Failed to decode mask for {embryo_id}: {e}")
            continue
    
    # Blend mask overlay with original image
    mask_areas = np.any(mask_overlay > 0, axis=2)
    overlay[mask_areas] = cv2.addWeighted(image[mask_areas], 1-alpha, 
                                          mask_overlay[mask_areas], alpha, 0)
    
    return overlay


def create_video_from_annotations(video_data: Dict, output_path: str, 
                                 fps: int = 10, show_info: bool = True) -> bool:
    """
    Create a video from SAM2 annotations for a single video.
    
    Args:
        video_data: Video annotation data
        output_path: Output video file path
        fps: Frames per second for output video
        show_info: Whether to show info overlay
        
    Returns:
        Success status
    """
    video_id = video_data['video_id']
    print(f"\n🎬 Creating video for: {video_id}")
    
    # Get embryo information
    embryo_ids = video_data.get('embryo_ids', [])
    num_embryos = video_data.get('num_embryos', 0)
    
    if num_embryos == 0:
        print(f"   ⚠️  No embryos found in {video_id}")
        return False
    
    # Generate colors for embryos
    embryo_colors = {}
    colors = generate_embryo_colors(num_embryos)
    for i, embryo_id in enumerate(embryo_ids):
        embryo_colors[embryo_id] = colors[i]
    
    # Get image data
    images_data = video_data.get('images', {})
    if not images_data:
        print(f"   ⚠️  No image data found for {video_id}")
        return False
    
    # Sort images by frame index
    sorted_images = sorted(images_data.items(), 
                          key=lambda x: x[1].get('frame_index', 0))
    
    # Get first image to determine video dimensions
    first_image_id = sorted_images[0][0]
    
    # Try to find the image directory from metadata
    # Assuming images are in a standard location relative to annotations
    possible_dirs = [
        Path("data/raw_data_organized/processed_jpg_images"),
        Path("data/processed_jpg_images"),
        Path(".").glob(f"**/*{video_id}*"),
    ]
    
    image_dir = None
    for pdir in possible_dirs:
        if isinstance(pdir, Path) and pdir.exists():
            test_path = pdir / f"{first_image_id}.jpg"
            if test_path.exists():
                image_dir = pdir
                break
        elif hasattr(pdir, '__iter__'):  # glob result
            for path in pdir:
                if path.is_dir():
                    test_path = path / f"{first_image_id}.jpg"
                    if test_path.exists():
                        image_dir = path
                        break
    
    if image_dir is None:
        print(f"   ❌ Could not find image directory for {video_id}")
        print(f"   Tried to find: {first_image_id}.jpg")
        return False
    
    print(f"   📁 Found images in: {image_dir}")
    
    # Load first image to get dimensions
    first_image_path = image_dir / f"{first_image_id}.jpg"
    first_image = cv2.imread(str(first_image_path))
    if first_image is None:
        print(f"   ❌ Could not load first image: {first_image_path}")
        return False
    
    height, width = first_image.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    print(f"   🔄 Processing {len(sorted_images)} frames...")
    
    for image_id, image_info in tqdm(sorted_images, desc="Frames"):
        # Load image
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            print(f"   ⚠️  Missing image: {image_path}")
            continue
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   ⚠️  Failed to load: {image_path}")
            continue
        
        # Get embryo masks for this frame
        embryo_masks = image_info.get('embryos', {})
        
        # Overlay masks
        if embryo_masks:
            image_with_masks = overlay_masks_on_image(image, embryo_masks, embryo_colors)
        else:
            image_with_masks = image
        
        # Add info overlay if requested
        if show_info:
            # Add frame info
            frame_idx = image_info.get('frame_index', -1)
            is_seed = image_info.get('is_seed_frame', False)
            
            info_text = f"Frame: {frame_idx}"
            if is_seed:
                info_text += " (SEED)"
            
            cv2.putText(image_with_masks, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add embryo count
            embryo_count_text = f"Embryos: {len(embryo_masks)}/{num_embryos}"
            cv2.putText(image_with_masks, embryo_count_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add legend for embryos
            y_offset = 90
            for embryo_id, color in embryo_colors.items():
                label = embryo_id.split('_')[-1]
                cv2.putText(image_with_masks, label, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 25
        
        # Write frame to video
        out.write(image_with_masks)
    
    # Release video writer
    out.release()
    
    print(f"   ✅ Video saved: {output_path}")
    return True


def process_videos(annotations: Dict, output_dir: str, max_videos: Optional[int] = None,
                  fps: int = 10, show_info: bool = True) -> Dict:
    """
    Process multiple videos from annotations.
    
    Args:
        annotations: SAM2 annotations data
        output_dir: Output directory for videos
        max_videos: Maximum number of videos to process
        fps: Frames per second for output videos
        show_info: Whether to show info overlay
        
    Returns:
        Processing statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all videos
    all_videos = []
    for exp_data in annotations.get('experiments', {}).values():
        for video_id, video_data in exp_data.get('videos', {}).items():
            if video_data.get('sam2_success', False):
                all_videos.append((video_id, video_data))
    
    print(f"\n📊 Found {len(all_videos)} successfully processed videos")
    
    # Apply max_videos limit
    if max_videos:
        all_videos = all_videos[:max_videos]
        print(f"   Limiting to {max_videos} videos")
    
    # Process statistics
    stats = {
        'total_videos': len(all_videos),
        'videos_created': 0,
        'videos_failed': 0,
        'start_time': datetime.now()
    }
    
    # Process each video
    for i, (video_id, video_data) in enumerate(all_videos, 1):
        print(f"\n{'='*15} Video {i}/{len(all_videos)} {'='*15}")
        
        # Create output filename
        output_filename = f"{video_id}_tracked.mp4"
        output_video_path = str(output_path / output_filename)
        
        try:
            success = create_video_from_annotations(video_data, output_video_path, 
                                                   fps=fps, show_info=show_info)
            if success:
                stats['videos_created'] += 1
            else:
                stats['videos_failed'] += 1
                
        except Exception as e:
            print(f"   ❌ Error processing {video_id}: {e}")
            stats['videos_failed'] += 1
    
    # Final statistics
    stats['end_time'] = datetime.now()
    stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
    
    print(f"\n🎯 Processing Complete!")
    print(f"   Videos created: {stats['videos_created']}")
    print(f"   Videos failed: {stats['videos_failed']}")
    print(f"   Total time: {stats['duration']:.1f} seconds")
    print(f"   Output directory: {output_path}")
    
    return stats


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Generate videos from SAM2 annotations')
    parser.add_argument('--annotations', type=str, 
                       default='data/annotation_and_masks/sam2_annotations/grounded_sam_annotations.json',
                       help='Path to grounded_sam_annotations.json')
    parser.add_argument('--output', type=str, default='output_videos/',
                       help='Output directory for videos')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='Maximum number of videos to process')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for output videos')
    parser.add_argument('--no-info', action='store_true',
                       help='Disable info overlay on videos')
    parser.add_argument('--test', action='store_true',
                       help='Run test case with 5 videos')
    
    args = parser.parse_args()
    
    # Override for test case
    if args.test:
        args.max_videos = 5
        print("🧪 Running test case with max 5 videos")
    
    # Load annotations
    try:
        annotations = load_annotations(args.annotations)
    except Exception as e:
        print(f"❌ Failed to load annotations: {e}")
        return 1
    
    # Process videos
    stats = process_videos(annotations, args.output, 
                          max_videos=args.max_videos,
                          fps=args.fps,
                          show_info=not args.no_info)
    
    return 0 if stats['videos_created'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())