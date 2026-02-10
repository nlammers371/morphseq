#!/usr/bin/env python3
"""
Debug script to check if SAM2 masks are being decoded correctly.
Creates binary mask images to visually verify mask decoding.
"""

import json
import numpy as np
import cv2
from pathlib import Path
import sys

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.mask_utils import decode_mask_rle

def debug_mask_decoding():
    """Debug mask decoding by saving binary images."""
    
    # Paths
    results_json = Path("data/segmentation/grounded_sam_segmentations.json")
    debug_dir = Path("temp/debug_masks")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Target video
    experiment_id = "20250612_30hpf_ctrl_atf6"
    video_id = "20250612_30hpf_ctrl_atf6_A01"
    
    print(f"🔍 Debugging mask decoding for {video_id}")
    print(f"📁 Results file: {results_json}")
    print(f"💾 Debug output: {debug_dir}")
    
    # Load SAM2 results
    try:
        with open(results_json, 'r') as f:
            sam2_data = json.load(f)
        print(f"✅ Loaded results JSON")
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        return 1
    
    # Navigate to video data
    experiments = sam2_data.get("experiments", {})
    if experiment_id not in experiments:
        print(f"❌ Experiment {experiment_id} not found")
        return 1
        
    exp_data = experiments[experiment_id]
    videos = exp_data.get("videos", {})
    if video_id not in videos:
        print(f"❌ Video {video_id} not found")
        return 1
        
    video_data = videos[video_id]
    images = video_data.get("images", {})
    
    print(f"📊 Found {len(images)} images with data")
    
    # Process first few images with embryo data
    processed_count = 0
    max_debug_images = 5
    
    for image_id, image_data in images.items():
        if processed_count >= max_debug_images:
            break
            
        embryos = image_data.get("embryos", {})
        if not embryos:
            continue
            
        print(f"\n🖼️ Processing {image_id}")
        print(f"   Found {len(embryos)} embryos")
        
        # Create debug image for this frame
        debug_image_dir = debug_dir / image_id
        debug_image_dir.mkdir(exist_ok=True)
        
        for embryo_id, embryo_data in embryos.items():
            print(f"   🔬 Processing embryo {embryo_id}")
            
            # Get segmentation data
            segmentation = embryo_data.get("segmentation")
            if not segmentation:
                print(f"      ❌ No segmentation data")
                continue
                
            # Print raw segmentation info
            print(f"      📋 Segmentation type: {type(segmentation)}")
            if isinstance(segmentation, dict):
                print(f"      📋 Keys: {list(segmentation.keys())}")
                if 'size' in segmentation:
                    print(f"      📋 Size: {segmentation['size']}")
                if 'counts' in segmentation:
                    counts_type = type(segmentation['counts'])
                    if isinstance(segmentation['counts'], str):
                        counts_len = len(segmentation['counts'])
                        print(f"      📋 Counts: {counts_type}, length={counts_len}")
                    else:
                        print(f"      📋 Counts: {counts_type}")
            
            # Try to decode mask
            try:
                # Handle nested format
                if isinstance(segmentation, dict):
                    format_type = segmentation.get("format", "rle")
                    if format_type in ["rle", "rle_base64"]:
                        mask = decode_mask_rle(segmentation)
                    else:
                        print(f"      ⚠️ Unsupported format: {format_type}")
                        continue
                else:
                    # Legacy format
                    mask = decode_mask_rle(segmentation)
                    
                if mask is None:
                    print(f"      ❌ Decoded mask is None")
                    continue
                    
                print(f"      ✅ Decoded mask: shape={mask.shape}, dtype={mask.dtype}")
                print(f"      📊 Mask stats: min={mask.min()}, max={mask.max()}, sum={mask.sum()}")
                
                # Save binary mask as image
                binary_mask_path = debug_image_dir / f"{embryo_id}_binary_mask.png"
                
                # Convert to 0-255 for visualization
                mask_viz = (mask * 255).astype(np.uint8)
                cv2.imwrite(str(binary_mask_path), mask_viz)
                print(f"      💾 Saved binary mask: {binary_mask_path}")
                
                # Also save a colored version for easier viewing
                colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                colored_mask[mask > 0] = [0, 255, 0]  # Green mask
                colored_mask_path = debug_image_dir / f"{embryo_id}_colored_mask.png"
                cv2.imwrite(str(colored_mask_path), colored_mask)
                print(f"      💾 Saved colored mask: {colored_mask_path}")
                
                # Get bbox for comparison
                bbox = embryo_data.get("bbox", [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    mask_area = int(np.sum(mask))
                    bbox_area = w * h
                    fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0.0
                    print(f"      📐 BBox: {bbox}")
                    print(f"      📏 Mask area: {mask_area}, BBox area: {bbox_area}, Fill ratio: {fill_ratio:.3f}")
                    
                    # Create bbox visualization
                    bbox_viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                    cv2.rectangle(bbox_viz, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                    bbox_viz_path = debug_image_dir / f"{embryo_id}_bbox.png"
                    cv2.imwrite(str(bbox_viz_path), bbox_viz)
                    print(f"      💾 Saved bbox visualization: {bbox_viz_path}")
                    
                    # Create combined visualization
                    combined = colored_mask.copy()
                    cv2.rectangle(combined, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                    combined_path = debug_image_dir / f"{embryo_id}_combined.png"
                    cv2.imwrite(str(combined_path), combined)
                    print(f"      💾 Saved combined visualization: {combined_path}")
                
            except Exception as e:
                print(f"      ❌ Failed to decode mask: {e}")
                continue
                
        processed_count += 1
    
    print(f"\n✅ Debug complete. Check {debug_dir} for mask visualizations")
    print(f"📊 Processed {processed_count} images")
    
    return 0

if __name__ == "__main__":
    sys.exit(debug_mask_decoding())
