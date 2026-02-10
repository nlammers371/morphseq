#!/usr/bin/env python3
"""
Quick script to visualize a specific mask from grounded_sam_annotations.json
"""

import json
import cv2
import numpy as np
from pathlib import Path
import sys

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.mask_utils import decode_mask_rle

def visualize_mask_from_json():
    """Load and visualize a specific mask from the JSON file."""
    
    # Paths
    json_path = Path("archive/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json")
    output_dir = Path("temp/mask_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target identifiers
    experiment_id = "20250612_30hpf_ctrl_atf6"
    video_id = "20250612_30hpf_ctrl_atf6_A01"
    image_id = "20250612_30hpf_ctrl_atf6_A01_0000"  # First frame (correct format)
    embryo_id = "20250612_30hpf_ctrl_atf6_A01_e01"
    
    print(f"🔍 Loading mask data from {json_path}")
    print(f"🎯 Target: {embryo_id}")
    
    # Load JSON
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded JSON file")
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return
    
    # Navigate to the specific mask
    try:
        experiments = data.get("experiments", {})
        if experiment_id not in experiments:
            print(f"❌ Experiment {experiment_id} not found")
            available_exps = list(experiments.keys())
            print(f"Available experiments: {available_exps[:5]}...")
            return
            
        exp_data = experiments[experiment_id]
        videos = exp_data.get("videos", {})
        if video_id not in videos:
            print(f"❌ Video {video_id} not found")
            available_videos = list(videos.keys())
            print(f"Available videos: {available_videos[:5]}...")
            return
            
        video_data = videos[video_id]
        images = video_data.get("images", {})
        if image_id not in images:
            print(f"❌ Image {image_id} not found")
            available_images = list(images.keys())
            print(f"Available images: {available_images[:5]}...")
            return
            
        image_data = images[image_id]
        embryos = image_data.get("embryos", {})
        if embryo_id not in embryos:
            print(f"❌ Embryo {embryo_id} not found")
            available_embryos = list(embryos.keys())
            print(f"Available embryos: {available_embryos}")
            return
            
        embryo_data = embryos[embryo_id]
        print(f"✅ Found embryo data")
        
    except Exception as e:
        print(f"❌ Error navigating JSON structure: {e}")
        return
    
    # Extract segmentation data
    segmentation = embryo_data.get("segmentation")
    if not segmentation:
        print(f"❌ No segmentation data found")
        return
        
    bbox = embryo_data.get("bbox", [])
    area = embryo_data.get("area", 0)
    confidence = embryo_data.get("mask_confidence", 0)
    
    print(f"📊 Mask info:")
    print(f"   Format: {embryo_data.get('segmentation_format', 'unknown')}")
    print(f"   Size: {segmentation.get('size', 'unknown')}")
    print(f"   Area: {area}")
    print(f"   Confidence: {confidence}")
    print(f"   BBox: {bbox}")
    
    # Decode the mask
    try:
        print(f"🔄 Decoding RLE mask...")
        
        # Try direct pycocotools decoding
        try:
            from pycocotools import mask as mask_utils
            import base64
            
            counts_raw = segmentation.get('counts', '')
            size = segmentation.get('size', [])
            
            print(f"🔍 RLE info: counts_len={len(counts_raw)}, size={size}")
            
            # Fix base64 padding
            missing_padding = len(counts_raw) % 4
            if missing_padding:
                counts_raw += '=' * (4 - missing_padding)
                print(f"🔧 Added {4 - missing_padding} padding chars")
            
            # Decode base64 to bytes
            counts_bytes = base64.b64decode(counts_raw)
            print(f"✅ Base64 decoded: {len(counts_bytes)} bytes")
            
            # Create RLE dict for pycocotools
            rle_dict = {
                'size': size,
                'counts': counts_bytes
            }
            
            # Decode with pycocotools
            mask = mask_utils.decode(rle_dict)
            print(f"✅ Decoded mask: shape={mask.shape}, dtype={mask.dtype}")
            print(f"📊 Mask stats: min={mask.min()}, max={mask.max()}, sum={mask.sum()}")
            
        except Exception as e:
            print(f"❌ Direct pycocotools failed: {e}")
            
            # Fallback: try our mask_utils
            try:
                mask = decode_mask_rle(segmentation)
                print(f"✅ Fallback decode successful")
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return
        
    except Exception as e:
        print(f"❌ Failed to decode mask: {e}")
        return
    
    # Create visualizations
    try:
        # 1. Binary mask (black/white)
        binary_viz = (mask * 255).astype(np.uint8)
        binary_path = output_dir / f"{embryo_id}_binary.png"
        cv2.imwrite(str(binary_path), binary_viz)
        print(f"💾 Saved binary mask: {binary_path}")
        
        # 2. Colored mask (green on black)
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask > 0] = [0, 255, 0]  # Green
        colored_path = output_dir / f"{embryo_id}_colored.png"
        cv2.imwrite(str(colored_path), colored_mask)
        print(f"💾 Saved colored mask: {colored_path}")
        
        # 3. Mask with bounding box
        if len(bbox) == 4:
            bbox_viz = colored_mask.copy()
            h, w = mask.shape[:2]
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)
            
            cv2.rectangle(bbox_viz, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue bbox
            
            # Add text
            cv2.putText(bbox_viz, f"{embryo_id}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(bbox_viz, f"Area: {int(area)}", (x1, y2+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bbox_viz, f"Conf: {confidence:.2f}", (x1, y2+50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            bbox_path = output_dir / f"{embryo_id}_with_bbox.png"
            cv2.imwrite(str(bbox_path), bbox_viz)
            print(f"💾 Saved mask with bbox: {bbox_path}")
        
        print(f"\n✅ Visualization complete!")
        print(f"📁 Output directory: {output_dir}")
        
        # Calculate some mask properties
        mask_area_px = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        fill_percentage = (mask_area_px / total_pixels) * 100
        
        if len(bbox) == 4:
            bbox_area_px = (bbox[2] - bbox[0]) * w * (bbox[3] - bbox[1]) * h
            bbox_fill_ratio = mask_area_px / bbox_area_px if bbox_area_px > 0 else 0
            print(f"📏 Mask area: {mask_area_px} pixels ({fill_percentage:.2f}% of image)")
            print(f"📦 BBox fill ratio: {bbox_fill_ratio:.2f}")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return

if __name__ == "__main__":
    visualize_mask_from_json()
