#!/usr/bin/env python3
"""
Debug the specific QC check logic to see why violations aren't found.
"""

import sys
import json
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from detection_segmentation.gsam_quality_control import GSAMQualityControl
from pycocotools import mask as mask_utils
import numpy as np

def debug_small_mask_check():
    """Debug why the small mask check isn't finding violations."""
    
    test_file = "test_gsam_violations.json"
    
    print("🔍 DEBUGGING SMALL MASK CHECK")
    print("="*60)
    
    # Initialize QC
    qc = GSAMQualityControl(test_file, verbose=True)
    
    # Get expanded entities
    target_entities = {
        "experiment_ids": ["20240411_test"],
        "video_ids": [],
        "image_ids": [],
        "snip_ids": []
    }
    
    entities = qc._expand_target_entities(target_entities)
    print(f"Expanded entities: {len(entities['snip_ids'])} snips")
    print(f"Sample snips: {entities['snip_ids'][:5]}")
    
    # Manually check what should be a small mask (e04)
    target_snips = set(entities['snip_ids'])
    pct_threshold = 0.001  # 0.1% threshold
    
    print(f"\n🔍 Manual small mask detection (threshold: {pct_threshold}):")
    
    violations_found = []
    
    for exp_id, exp_data in qc.gsam_data.get("experiments", {}).items():
        for video_id, video_data in exp_data.get("videos", {}).items():
            for image_id, image_data in video_data.get("images", {}).items():
                for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                    snip_id = embryo_data.get("snip_id")
                    if not snip_id:
                        continue
                    
                    # Check if this snip is in our target list
                    if snip_id not in target_snips:
                        print(f"   ❌ Skipping {snip_id} - not in target list")
                        continue
                    
                    segmentation = embryo_data.get("segmentation")
                    if segmentation and segmentation.get("format") == "rle":
                        try:
                            mask = mask_utils.decode(segmentation)
                            mask_area = np.sum(mask)
                            total_area = mask.shape[0] * mask.shape[1]
                            pct_area = mask_area / total_area
                            
                            print(f"   📊 {snip_id} ({embryo_id} in {image_id}):")
                            print(f"      Mask area: {mask_area} pixels")
                            print(f"      Total area: {total_area} pixels")
                            print(f"      Percentage: {pct_area:.6f} ({pct_area*100:.4f}%)")
                            print(f"      Threshold: {pct_threshold:.6f} ({pct_threshold*100:.4f}%)")
                            
                            if pct_area < pct_threshold and mask_area > 0:
                                print(f"      ✅ VIOLATION DETECTED!")
                                violations_found.append(snip_id)
                            else:
                                print(f"      ⭕ No violation")
                            
                        except Exception as e:
                            print(f"   ❌ Error processing {snip_id}: {e}")
    
    print(f"\n📊 Manual scan results:")
    print(f"   Violations found: {len(violations_found)}")
    print(f"   Violated snips: {violations_found}")
    
    # Now run the actual check and compare
    print(f"\n🔍 Running actual check_small_masks:")
    qc.check_small_masks("debug", entities)
    
    # Check flags
    small_mask_flags = qc.get_flags_by_type("SMALL_MASK")
    print(f"   QC method found: {len(small_mask_flags)} violations")
    
    if small_mask_flags:
        for flag in small_mask_flags:
            print(f"      {flag['snip_id']}: {flag['area_percentage']:.6f}")

def debug_large_mask_check():
    """Debug the large mask check."""
    
    test_file = "test_gsam_violations.json"
    
    print("\n🔍 DEBUGGING LARGE MASK CHECK")
    print("="*60)
    
    # Initialize QC
    qc = GSAMQualityControl(test_file, verbose=False)
    
    # Get expanded entities
    target_entities = {
        "experiment_ids": ["20240411_test"],
        "video_ids": [],
        "image_ids": [],
        "snip_ids": []
    }
    
    entities = qc._expand_target_entities(target_entities)
    target_snips = set(entities['snip_ids'])
    pct_threshold = 0.15  # 15% threshold
    
    print(f"Manual large mask detection (threshold: {pct_threshold}):")
    
    violations_found = []
    
    for exp_id, exp_data in qc.gsam_data.get("experiments", {}).items():
        for video_id, video_data in exp_data.get("videos", {}).items():
            for image_id, image_data in video_data.get("images", {}).items():
                for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                    snip_id = embryo_data.get("snip_id")
                    if not snip_id or snip_id not in target_snips:
                        continue
                    
                    segmentation = embryo_data.get("segmentation")
                    if segmentation and segmentation.get("format") == "rle":
                        try:
                            mask = mask_utils.decode(segmentation)
                            mask_area = np.sum(mask)
                            total_area = mask.shape[0] * mask.shape[1]
                            pct_area = mask_area / total_area
                            
                            print(f"   📊 {snip_id} ({embryo_id}):")
                            print(f"      Percentage: {pct_area:.4f} ({pct_area*100:.2f}%)")
                            
                            if pct_area > pct_threshold:
                                print(f"      ✅ LARGE MASK VIOLATION!")
                                violations_found.append(snip_id)
                            
                        except Exception as e:
                            print(f"   ❌ Error: {e}")
    
    print(f"\n   Manual violations: {violations_found}")
    
    # Run actual check
    qc.check_large_masks("debug", entities)
    large_mask_flags = qc.get_flags_by_type("LARGE_MASK")
    print(f"   QC method found: {len(large_mask_flags)} violations")

if __name__ == "__main__":
    debug_small_mask_check()
    debug_large_mask_check()
