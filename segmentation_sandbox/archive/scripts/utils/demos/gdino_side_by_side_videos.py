import sys
sys.path.append("/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/")

# Robust video creation from image directory
import glob
def create_video_from_image_dir(images_dir, output_video, fps=8):
    """
    Create a video from all .jpg files in images_dir, ensuring even dimensions and skipping unreadable frames.
    """
    img_paths = sorted(glob.glob(str(Path(images_dir) / '*.jpg')))
    if not img_paths:
        raise RuntimeError(f"No .jpg files found in {images_dir}")

    # Read the first frame to get dimensions
    frame = cv2.imread(img_paths[0])
    if frame is None:
        raise RuntimeError(f"Cannot read {img_paths[0]}")
    h, w = frame.shape[:2]

    # Ensure even dims for codecs
    if h % 2 or w % 2:
        h, w = h - (h % 2), w - (w % 2)
        frame = frame[:h, :w]

    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open video writer")

    # Write each frame
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: skipping unreadable {img_path}")
            continue
        img = img[:h, :w]  # crop if necessary
        writer.write(img)

    writer.release()
    print(f"✅ Video saved to {output_video}")

#!/usr/bin/env python3
"""
Simple GroundedDINO Video Generator - Separate Videos
===================================================
Creates individual videos for base and finetuned models.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

# Import utilities
from scripts.utils.grounded_sam_utils import GroundedDinoAnnotations
from scripts.utils.experiment_metadata_utils import ExperimentMetadata, get_image_id_paths

# Side‐by‐side helper
from pathlib import Path
from typing import List

def create_side_by_side_video(
    video_id: str,
    base_dir: Path,
    ft_dir: Path,
    output_path: str,
    frame_ids: List[str],
    fps: int = 10
) -> bool:
    """
    Reads each frame from `base_dir` and `ft_dir` (same filenames),
    concatenates them horizontally, and writes to output_path.
    Returns True if successful.
    """
    import cv2
    from tqdm import tqdm

    writer = None

    for frame_id in tqdm(frame_ids, desc=f"Building SxS {video_id}"):
        base_frame = base_dir / f"{frame_id}.jpg"
        ft_frame   = ft_dir   / f"{frame_id}.jpg"

        img1 = cv2.imread(str(base_frame))
        img2 = cv2.imread(str(ft_frame))
        if img1 is None or img2 is None:
            print(f"   ❌ Missing frame: {frame_id}")
            return False

        # Resize to match dimensions
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        combined = cv2.hconcat([img1, img2])

        if writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                print("   ❌ Cannot open video writer")
                return False

        writer.write(combined)

    writer.release()
    print(f"   ✅ Side-by-side saved: {output_path}")
    return True

# Configuration
SANDBOX_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox")
DEFAULT_BASE_ANNOTATIONS = SANDBOX_ROOT / "data/annotation_and_masks/gdino_annotations/gdino_annotations.json"
DEFAULT_FT_ANNOTATIONS = SANDBOX_ROOT / "data/annotation_and_masks/gdino_annotations/gdino_annotations_finetuned.json"
DEFAULT_OUTPUT_DIR = SANDBOX_ROOT / "data/visualization_output/20250716"
DEFAULT_METADATA_PATH = SANDBOX_ROOT / "data/raw_data_organized/experiment_metadata.json"

# Colors for bounding boxes
COLORS = [
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Yellow
    (0, 255, 0),      # Lime
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Sky blue
    (255, 0, 128),    # Pink
]

def draw_bboxes_on_image(image, detections, model_name=""):
    """Draw bounding boxes on image with confidence scores."""
    result = image.copy()
    height, width = image.shape[:2]
    
    # Calculate sizes
    line_thickness = max(2, width // 400)
    font_scale = max(0.6, width / 2000)
    font_thickness = max(1, line_thickness // 2) * 2
    
    for i, det in enumerate(detections):
        box = det.get("box_xyxy", [])
        if len(box) != 4:
            continue
            
        conf = det.get("confidence", 0.0)
        phrase = det.get("phrase", "object")
        color = COLORS[i % len(COLORS)]
        
        # Convert coordinates
        if all(coord <= 1.0 for coord in box):
            # Normalized coordinates
            x1, y1, x2, y2 = int(box[0]*width), int(box[1]*height), int(box[2]*width), int(box[3]*height)
        else:
            # Pixel coordinates
            x1, y1, x2, y2 = map(int, box)
        
        # Ensure bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width-1, x2), min(height-1, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Draw semi-transparent bbox
        overlay = result.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, line_thickness * 2)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Draw label
        label = f"{phrase}: {conf:.3f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Position text
        text_y = y1 - 5 if y1 - th - 10 > 0 else y1 + th + 5
        text_x = max(0, min(x1, width - tw))
        
        # Draw background
        overlay = result.copy()
        cv2.rectangle(overlay, (text_x - 3, text_y - th - 5), (text_x + tw + 3, text_y + 5), color, -1)
        cv2.addWeighted(overlay, 0.8, result, 0.2, 0, result)
        
        # Draw text
        cv2.putText(result, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    # Add model name
    if model_name:
        model_label = f"{model_name} ({len(detections)} detections)"
        cv2.putText(result, model_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return result

def get_detections_for_image(image_id, ann_loader, prompt):
    """Get detections for an image."""
    annotations = ann_loader.get_annotations_for_image(image_id)
    for ann in annotations:
        if ann.get("prompt") == prompt:
            return ann.get("detections", [])
    return []
def create_video_from_frames_simple(frames_dir, output_path, fps=8):
    """
    Create video from frames by testing multiple codec/container/strategy combinations
    to maximize QuickTime compatibility and detect corrupted (green) outputs.
    """
    import glob

    # Collect frames
    img_array = []
    for filename in sorted(glob.glob(str(frames_dir / "*.jpg"))):
        img = cv2.imread(filename)
        if img is None:
            print(f"Warning: Could not read {filename}")
            continue

        h, w = img.shape[:2]
        # Ensure even dimensions
        if h % 2 != 0: h -= 1
        if w % 2 != 0: w -= 1
        if img.shape[:2] != (h, w):
            img = img[:h, :w]
        img_array.append(img)

    if not img_array:
        print(f"No images found in {frames_dir}")
        return False

    height, width = img_array[0].shape[:2]
    size = (width, height)
    print(f"Creating video from {len(img_array)} frames at size {size}")

    # Codec and container combinations to test
    codec_combinations = [
        ('mp4v', '.mp4'), ('MP4V', '.mp4'), ('avc1', '.mp4'),
        ('XVID', '.avi'), ('MJPG', '.avi'),
    ]
    # Frame writing strategies
    strategies = [
        'direct', 'black_first', 'duplicate_first', 'fade_in', 'buffer_prime'
    ]

    for codec, ext in codec_combinations:
        for strat in strategies:
            test_path = output_path.with_suffix(ext)
            print(f"Testing {codec}{ext} with strategy {strat}")
            try:
                fourcc = 0 if codec == '\x00\x00\x00\x00' else cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(test_path), fourcc, fps, size)
                if not out.isOpened():
                    print(f"  ❌ Cannot open writer for codec {codec}")
                    continue

                # Build frame list per strategy
                to_write = list(img_array)
                if strat == 'black_first':
                    black = np.zeros_like(img_array[0])
                    to_write.insert(0, black)
                elif strat == 'duplicate_first':
                    to_write.insert(0, img_array[0].copy())
                elif strat == 'fade_in':
                    for i in range(5):
                        alpha = i/5.0
                        fade = (img_array[0] * alpha).astype(np.uint8)
                        to_write.insert(i, fade)
                elif strat == 'buffer_prime':
                    dummy = np.zeros_like(img_array[0]) + 128
                    out.write(dummy)

                # Write frames
                for frame in to_write:
                    out.write(frame)
                out.release()

                # Verify first frame
                if not test_path.exists() or test_path.stat().st_size < 1000:
                    print(f"  ❌ Invalid file {test_path}")
                    continue
                cap = cv2.VideoCapture(str(test_path))
                if not cap.isOpened():
                    print(f"  ❌ Cannot open {test_path} for readback")
                    continue
                ret, frame = cap.read()
                cap.release()
                if not ret or frame is None:
                    print(f"  ❌ Cannot read back first frame")
                    continue

                # Detect green/solid error
                mean_bgr = frame.mean(axis=(0,1))
                std = frame.std()
                if std < 10 or (mean_bgr[1] > mean_bgr[0]+50 and mean_bgr[1] > mean_bgr[2]+50 and mean_bgr[1] > 200):
                    print(f"  ❌ Green/corrupt {test_path}")
                    continue

                # Success
                print(f"  ✅ Successful: {test_path}")
                return True

            except Exception as e:
                print(f"  ❌ Error with {codec}{ext} / {strat}: {e}")
                continue

    print("❌ All codec/strategy combos failed")
    return False


def create_merged_frames(base_frames_dir, ft_frames_dir, merged_frames_dir, frame_ids):
    """Create side-by-side merged frames with even dimensions."""
    merged_frames_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    for frame_id in tqdm(frame_ids, desc="Merging frames"):
        base_frame_path = base_frames_dir / f"{frame_id}.jpg"
        ft_frame_path = ft_frames_dir / f"{frame_id}.jpg"
        
        if not base_frame_path.exists() or not ft_frame_path.exists():
            continue
        
        base_img = cv2.imread(str(base_frame_path))
        ft_img = cv2.imread(str(ft_frame_path))
        
        if base_img is None or ft_img is None:
            continue
        
        # Ensure same dimensions
        if base_img.shape != ft_img.shape:
            ft_img = cv2.resize(ft_img, (base_img.shape[1], base_img.shape[0]))
        
        # Merge side by side
        merged = cv2.hconcat([base_img, ft_img])
        
        # CRITICAL FIX: Ensure dimensions are even for H.264 codec
        h, w = merged.shape[:2]
        if h % 2 != 0:
            h = h - 1
        if w % 2 != 0:
            w = w - 1
        
        # Resize to even dimensions if needed
        if merged.shape[:2] != (h, w):
            merged = merged[:h, :w]  # Crop to even dimensions
            print(f"   📐 Adjusted dimensions to {w}x{h} for codec compatibility")
        
        # Save merged frame
        merged_path = merged_frames_dir / f"{frame_id}.jpg"
        if cv2.imwrite(str(merged_path), merged):
            successful += 1
    
    return successful

def process_video(video_id, base_ann, ft_ann, metadata, output_dir, prompt, fps, create_individual=True, create_merged=True):
    """Process a single video."""
    print(f"\n🎬 Processing video: {video_id}")
    
    # Get common frame IDs
    base_ids = [img_id for img_id in base_ann.get_all_image_ids() if img_id.startswith(video_id + "_")]
    ft_ids = [img_id for img_id in ft_ann.get_all_image_ids() if img_id.startswith(video_id + "_")]
    common_ids = sorted(list(set(base_ids) & set(ft_ids)))
    
    if not common_ids:
        print(f"❌ No common frames for {video_id}")
        return False
    
    print(f"📊 Found {len(common_ids)} frames")
    
    # Create output directories
    video_dir = output_dir / video_id
    base_frames_dir = video_dir / "base_frames"
    ft_frames_dir = video_dir / "ft_frames"
    merged_frames_dir = video_dir / "merged_frames"
    
    base_frames_dir.mkdir(parents=True, exist_ok=True)
    ft_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each frame
    successful_frames = []
    print("🖼️ Rendering frames...")
    
    for image_id in tqdm(common_ids, desc="Processing frames"):
        try:
            # Get image path
            image_paths = get_image_id_paths([image_id], metadata)
            if not image_paths:
                continue
            image_path = image_paths[0]
            
            # Load original image
            original_img = cv2.imread(str(image_path))
            if original_img is None:
                continue
            
            # Get detections
            base_detections = get_detections_for_image(image_id, base_ann, prompt)
            ft_detections = get_detections_for_image(image_id, ft_ann, prompt)
            
            # Draw bboxes
            base_img = draw_bboxes_on_image(original_img, base_detections, "Base Model")
            ft_img = draw_bboxes_on_image(original_img, ft_detections, "Finetuned Model")
            
            # Save frames
            base_output = base_frames_dir / f"{image_id}.jpg"
            ft_output = ft_frames_dir / f"{image_id}.jpg"
            
            if cv2.imwrite(str(base_output), base_img) and cv2.imwrite(str(ft_output), ft_img):
                successful_frames.append(image_id)
                
        except Exception as e:
            print(f"⚠️ Error processing {image_id}: {e}")
            continue
    
    if not successful_frames:
        print(f"❌ No frames processed for {video_id}")
        return False
    
    print(f"✅ Processed {len(successful_frames)} frames")
    
    success_count = 0
    
    # Create individual videos
    if create_individual:
        print("🎥 Creating individual videos...")
        # Base video
        base_video_path = output_dir / f"{video_id}_gdino_base.mp4"
        try:
            create_video_from_image_dir(base_frames_dir, base_video_path, fps)
            print(f"✅ Created: {base_video_path}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to create base video: {e}")
        # Finetuned video
        ft_video_path = output_dir / f"{video_id}_gdino_ft.mp4"
        try:
            create_video_from_image_dir(ft_frames_dir, ft_video_path, fps)
            print(f"✅ Created: {ft_video_path}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to create finetuned video: {e}")

    # Create side-by-side video
    if create_merged:
        print("🎥 Creating side-by-side video...")
        sxs_path = output_dir / f"{video_id}_gdino_side_by_side.mp4"
        # First, create merged frames
        merged_frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in successful_frames:
            base_frame = base_frames_dir / f"{frame_id}.jpg"
            ft_frame = ft_frames_dir / f"{frame_id}.jpg"
            if not base_frame.exists() or not ft_frame.exists():
                continue
            img1 = cv2.imread(str(base_frame))
            img2 = cv2.imread(str(ft_frame))
            if img1 is None or img2 is None:
                continue
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            merged = cv2.hconcat([img1, img2])
            h, w = merged.shape[:2]
            if h % 2 or w % 2:
                h, w = h - (h % 2), w - (w % 2)
                merged = merged[:h, :w]
            merged_path = merged_frames_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(merged_path), merged)
        try:
            create_video_from_image_dir(merged_frames_dir, sxs_path, fps)
            print(f"✅ Created: {sxs_path}")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to create side-by-side for {video_id}: {e}")
    
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="GroundedDINO video generator - separate and merged")
    parser.add_argument("--base_annotations", type=Path, default=DEFAULT_BASE_ANNOTATIONS)
    parser.add_argument("--finetuned_annotations", type=Path, default=DEFAULT_FT_ANNOTATIONS)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--prompt", type=str, default="individual embryo")
    parser.add_argument("--video_ids", nargs='+', 
                       default=["20230525_A04", "20250624_chem02_35C_T00_1216_B06", "20250305_G04", "20250305_G03"])
    parser.add_argument("--individual-only", action="store_true", help="Create only individual videos")
    parser.add_argument("--merged-only", action="store_true", help="Create only merged video")
    
    args = parser.parse_args()
    
    # Validate files
    for path, name in [(args.base_annotations, "base annotations"), 
                      (args.finetuned_annotations, "finetuned annotations"),
                      (args.metadata, "experiment metadata")]:
        if not path.exists():
            print(f"❌ {name} file not found: {path}")
            return 1
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine what to create
    create_individual = not args.merged_only
    create_merged = not args.individual_only
    
    # Load annotations and metadata
    print("🔄 Loading data...")
    base_ann = GroundedDinoAnnotations(args.base_annotations, verbose=False)
    ft_ann = GroundedDinoAnnotations(args.finetuned_annotations, verbose=False)
    metadata = ExperimentMetadata(args.metadata, verbose=False)
    
    # Process videos
    successful = 0
    for video_id in args.video_ids:
        if process_video(video_id, base_ann, ft_ann, metadata, args.output_dir, args.prompt, 
                        args.fps, create_individual, create_merged):
            successful += 1
    
    print(f"\n📈 Successfully processed {successful}/{len(args.video_ids)} videos")
    print(f"📁 All outputs saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    exit(main())