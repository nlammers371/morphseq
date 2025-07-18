#!/usr/bin/env python3
"""
SAM2 Video Generator - Simple & Streamlined
===========================================

Creates tracking videos from SAM2 annotations with color-coded embryos.
All bounding boxes are expected in box_xyxy format (normalized coordinates).

Usage:
    python sam2_video_generator.py annotations.json output_dir/ --max-videos 5
"""

import json
import argparse
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add utils to path for experiment metadata utilities
SANDBOX_ROOT = Path(__file__).parent.parent.parent
if str(SANDBOX_ROOT) not in sys.path:
    sys.path.append(str(SANDBOX_ROOT))


def load_annotations(path: str) -> Dict:
    """Load SAM2 annotations."""
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"📁 Loaded {len(data.get('video_ids', []))} videos")
    return data


def decode_rle_mask(rle: Dict) -> Optional[np.ndarray]:
    """Decode RLE mask to binary array."""
    try:
        from pycocotools import mask as mask_utils
        if isinstance(rle['counts'], str):
            rle = {'counts': rle['counts'].encode('utf-8'), 'size': rle['size']}
        return mask_utils.decode(rle).astype(np.uint8)
    except (ImportError, Exception):
        # Fallback without pycocotools or on decode error
        try:
            return np.array(rle['data']).reshape(rle['size']).astype(np.uint8)
        except Exception:
            return None


def get_embryo_colors(num_embryos: int) -> List[Tuple[int, int, int]]:
    """Generate distinct BGR colors for embryos."""
    cmap = plt.cm.get_cmap('tab10' if num_embryos <= 10 else 'tab20')
    colors = []
    for i in range(num_embryos):
        rgb = cmap(i / max(num_embryos - 1, 1))[:3]
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    return colors


def overlay_embryos(image: np.ndarray, embryos: Dict, colors: Dict, alpha: float = 0.5) -> np.ndarray:
    """Overlay embryo masks and bboxes on image. Expects box_xyxy format."""
    result = image.copy()
    h, w = image.shape[:2]
    
    for embryo_id, data in embryos.items():
        if embryo_id not in colors:
            continue
        color = colors[embryo_id]
        
        # Draw mask
        if 'segmentation' in data:
            mask = decode_rle_mask(data['segmentation'])
            if mask is not None:
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0] = color
                result = cv2.addWeighted(result, 1-alpha, colored_mask, alpha, 0)
        
        # Draw bbox (box_xyxy format - normalized coordinates)
        if 'bbox' in data:
            x1, y1, x2, y2 = data['bbox']
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = embryo_id.split('_')[-1]  # "e1", "e2", etc.
            cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result


def find_image_dir(video_id: str, first_image_id: str) -> Path:
    """Find image directory for a video using experiment metadata."""
    try:
        from scripts.utils.experiment_metadata_utils import get_image_id_paths
        # Try to get image path using metadata utils
        img_path = get_image_id_paths(first_image_id, 
                                      SANDBOX_ROOT / "data/raw_data_organized/experiment_metadata.json")
        return img_path.parent
    except Exception as e:
        # Fallback: search common locations
        search_paths = [
            SANDBOX_ROOT / "data/raw_data_organized" / video_id.split('_')[0] / "images" / video_id,
            SANDBOX_ROOT / f"data/**/images/{video_id}",
        ]
        
        for path in search_paths:
            if path.exists() and (path / f"{first_image_id}.jpg").exists():
                return path
        
        raise FileNotFoundError(f"Could not find images for {video_id}. "
                               f"Metadata error: {e}")  


def create_video(video_id: str, video_data: Dict, output_path: str, fps: int = 10) -> bool:
    """Create tracking video for one video with improved error handling."""
    print(f"🎬 Creating video: {video_id}")
    
    # Get embryo info and colors
    embryo_ids = video_data.get('embryo_ids', [])
    if not embryo_ids:
        print(f"   ❌ No embryos found")
        return False
    
    colors = get_embryo_colors(len(embryo_ids))
    embryo_colors = {eid: colors[i] for i, eid in enumerate(embryo_ids)}
    
    # Get sorted images - ensure proper temporal ordering
    images_data = video_data.get('images', {})
    if not images_data:
        print(f"   ❌ No image data")
        return False
    
    # Sort by frame_index first, then by image_id as fallback to ensure temporal order
    sorted_images = sorted(images_data.items(), 
                          key=lambda x: (x[1].get('frame_index', 999999), x[0]))
    
    if len(sorted_images) == 0:
        print(f"   ❌ No images found after sorting")
        return False
    
    first_image_id = sorted_images[0][0]
    
    # Verify frame ordering
    frame_indices = [img_info.get('frame_index', -1) for _, img_info in sorted_images]
    if frame_indices != sorted(frame_indices):
        print(f"   ⚠️  Frame indices not in sequential order, relying on image_id sorting")
    else:
        print(f"   ✅ Frames properly ordered: {len(sorted_images)} frames from index {frame_indices[0]} to {frame_indices[-1]}")
    
    # Additional check: ensure the first image has frame index 0 or lowest available
    first_frame_idx = sorted_images[0][1].get('frame_index', -1)
    min_frame_idx = min(frame_indices)
    if first_frame_idx == min_frame_idx and min_frame_idx >= 0:
        print(f"   ✅ First frame verification: frame {first_frame_idx} is correctly positioned first")
    else:
        print(f"   ⚠️  First frame issue: frame {first_frame_idx} is first, expected {min_frame_idx} (min available)")
    
    # Find image directory
    try:
        image_dir = find_image_dir(video_id, first_image_id)
        print(f"   📁 Found images: {image_dir}")
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return False
    
    # Load first image for dimensions
    first_img_path = image_dir / f"{first_image_id}.jpg"
    first_img = cv2.imread(str(first_img_path))
    if first_img is None:
        print(f"   ❌ Cannot load first image: {first_img_path}")
        return False
    # Create video writer
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        print(f"   ❌ Failed to create video writer")
        return False
    
    try:
        # Process frames
        frames_processed = 0
        for image_id, image_info in tqdm(sorted_images, desc="Processing frames"):
            img_path = image_dir / f"{image_id}.jpg"
            if not img_path.exists():
                print(f"   ❓ Image not found, skipping: {img_path}")
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"   ❓ Failed to read image, skipping: {img_path}")
                continue
            
            # Overlay embryos (using box_xyxy format)
            embryos = image_info.get('embryos', {})
            if embryos:
                img = overlay_embryos(img, embryos, embryo_colors)
            
            # Add frame info
            frame_idx = image_info.get('frame_index', -1)
            is_seed = image_info.get('is_seed_frame', False)
            info = f"Frame {frame_idx}" + (" (SEED)" if is_seed else "")
            cv2.putText(img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(img)
            frames_processed += 1
        
        print(f"   📹 Processed {frames_processed} frames")
        
    finally:
        out.release()
    
    print(f"   ✅ Saved: {output_path}")
    return True


def process_videos(annotations: Dict, output_dir: str, max_videos: Optional[int] = None, fps: int = 10, show_info: bool = True):
    """Process multiple videos from annotations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect successful videos
    videos = []
    for exp_data in annotations.get('experiments', {}).values():
        for vid_id, vid_data in exp_data.get('videos', {}).items():
            if vid_data.get('sam2_success', False):
                videos.append((vid_id, vid_data))
    
    # Limit number of videos
    if max_videos:
        videos = videos[:max_videos]
    
    print(f"🔄 Processing {len(videos)} videos...")
    
    # Process each video
    for vid_id, vid_data in videos:
        output_file = str(output_path / f"{vid_id}_tracked.mp4")
        create_video(vid_id, vid_data, output_file, fps)
    
    print("✅ All videos processed.")


def test_video_generation():
    """Test video generation with 5 videos."""
    print("🧪 SAM2 Video Generation Test")
    print("=" * 50)

    # Configuration
    annotations_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_ft_annotations.json"
    output_dir = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/test_videos"
    max_videos = 5
    fps = 10
    show_info = True

    print(f"📋 Test Configuration:")
    print(f"   Annotations: {annotations_path}")
    print(f"   Output dir: {output_dir}")
    print(f"   Max videos: {max_videos}")
    print(f"   FPS: {fps}")
    print(f"   Show info: {show_info}")

    # Check if annotations file exists
    if not Path(annotations_path).exists():
        print(f"Error: Annotations file not found at {annotations_path}")
        sys.exit(1)

    # Load annotations
    annotations = load_annotations(annotations_path)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process videos
    process_videos(annotations, output_dir, max_videos, fps, show_info)

    print("✅ Video generation test completed successfully!")


def main():
    """Main function with improved video selection."""
    parser = argparse.ArgumentParser(description='Generate SAM2 tracking videos')
    parser.add_argument('annotations', help='Path to grounded_sam_annotations.json')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--max-videos', type=int, help='Max videos to process')
    parser.add_argument('--fps', type=int, default=10, help='Video FPS')
    
    args = parser.parse_args()
    
    # Load annotations
    annotations = load_annotations(args.annotations)
    
    # Get all successful videos
    videos = []
    for exp_data in annotations.get('experiments', {}).values():
        for vid_id, vid_data in exp_data.get('videos', {}).items():
            if vid_data.get('sam2_success', False):
                videos.append((vid_id, vid_data))
    
    if not videos:
        print("❌ No successful SAM2 videos found")
        return
    
    if args.max_videos:
        videos = videos[:args.max_videos]
    
    print(f"🔄 Processing {len(videos)} videos...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process videos
    success_count = 0
    for vid_id, vid_data in videos:
        output_path = str(output_dir / f"{vid_id}_tracked.mp4")
        if create_video(vid_id, vid_data, output_path, args.fps):
            success_count += 1
    
    print(f"🎯 Created {success_count}/{len(videos)} videos in {output_dir}")
    
    if success_count == 0:
        print("❌ No videos were successfully created")
    elif success_count < len(videos):
        print(f"⚠️  {len(videos) - success_count} videos failed to create")


if __name__ == "__main__":
    test_video_generation()