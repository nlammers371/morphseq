#!/usr/bin/env python3
"""
Test with manual debugging of the small mask check to see where it fails.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from detection_segmentation.gsam_quality_control import GSAMQualityControl
from pycocotools import mask as mask_utils
import numpy as np

def debug_small_mask_method():
    """Manually step through the small mask check method."""
    
    test_file = "test_gsam_violations.json"
    
    print("🔍 DEBUGGING check_small_masks METHOD")
    print("="*60)
    
    # Initialize QC
    qc = GSAMQualityControl(test_file, verbose=False)
    
    # Get all entities (like process_all=True would do)
    entities = qc.get_all_entities_to_process()
    print(f"Entities: {entities}")
    
    # Manually implement the small mask check logic
    target_experiments = set(entities.get("experiment_ids", []))
    target_videos = set(entities.get("video_ids", []))
    target_images = set(entities.get("image_ids", []))
    target_snips = set(entities.get("snip_ids", []))
    
    # This mimics the logic in check_small_masks
    process_all = not any([target_experiments, target_videos, target_images, target_snips])
    
    print(f"process_all: {process_all}")
    print(f"target_experiments: {len(target_experiments)} items")
    print(f"target_videos: {len(target_videos)} items")
    print(f"target_images: {len(target_images)} items")
    print(f"target_snips: {len(target_snips)} items")
    
    pct_threshold = 0.001  # 0.1%
    flag_count = 0
    processed_count = 0
    
    experiments_items = list(qc.gsam_data.get("experiments", {}).items())
    print(f"\nExperiments to iterate: {len(experiments_items)}")
    
    for exp_id, exp_data in experiments_items:
        print(f"\n📁 Processing experiment: {exp_id}")
        
        # Skip experiment if not in target list (unless processing all)
        if not process_all and target_experiments and exp_id not in target_experiments:
            print(f"   ❌ Skipping experiment {exp_id} - not in target list")
            continue
        else:
            print(f"   ✅ Processing experiment {exp_id}")
        
        for video_id, video_data in exp_data.get("videos", {}).items():
            print(f"   📹 Processing video: {video_id}")
            
            # Skip video if not in target list (unless processing all)
            if not process_all and target_videos and video_id not in target_videos:
                print(f"      ❌ Skipping video {video_id} - not in target list")
                continue
            else:
                print(f"      ✅ Processing video {video_id}")
                
            for image_id, image_data in video_data.get("images", {}).items():
                print(f"      🖼️  Processing image: {image_id}")
                
                # Skip image if not in target list (unless processing all)
                if not process_all and target_images and image_id not in target_images:
                    print(f"         ❌ Skipping image {image_id} - not in target list")
                    continue
                else:
                    print(f"         ✅ Processing image {image_id}")
                    
                for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                    snip_id = embryo_data.get("snip_id")
                    if not snip_id:
                        print(f"            ❌ No snip_id for embryo {embryo_id}")
                        continue
                    
                    print(f"            🧬 Processing embryo {embryo_id}, snip: {snip_id}")
                    
                    # Skip snip if not in target list (unless processing all)
                    if not process_all and target_snips and snip_id not in target_snips:
                        print(f"               ❌ Skipping snip {snip_id} - not in target list")
                        continue
                    else:
                        print(f"               ✅ Processing snip {snip_id}")
                    
                    processed_count += 1
                    
                    segmentation = embryo_data.get("segmentation")
                    if segmentation and segmentation.get("format") == "rle":
                        print(f"               📊 RLE segmentation found")
                        try:
                            mask = mask_utils.decode(segmentation)
                            mask_area = np.sum(mask)
                            total_area = mask.shape[0] * mask.shape[1]
                            pct_area = mask_area / total_area
                            
                            print(f"               📊 Area: {mask_area}/{total_area} = {pct_area:.6f}")
                            
                            if pct_area < pct_threshold and mask_area > 0:
                                print(f"               ✅ VIOLATION DETECTED!")
                                flag_count += 1
                            else:
                                print(f"               ⭕ No violation")
                                
                        except Exception as e:
                            print(f"               ❌ Error decoding mask: {e}")
                    else:
                        print(f"               ❌ No RLE segmentation (format: {segmentation.get('format') if segmentation else 'None'})")
    
    print(f"\n📊 Final results:")
    print(f"   Processed embryos: {processed_count}")
    print(f"   Violations found: {flag_count}")

if __name__ == "__main__":
    debug_small_mask_method()
