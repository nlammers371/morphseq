#!/usr/bin/env python3
"""
Inspect GSAM data structure to understand what's available.
"""

import json
from pathlib import Path

def inspect_gsam_structure(gsam_path: str):
    """Inspect the structure of GSAM data."""
    print(f"📂 Loading GSAM data from {gsam_path}")
    
    with open(gsam_path, 'r') as f:
        gsam_data = json.load(f)
    
    print("🔍 Analyzing data structure...")
    
    # Top-level structure
    print(f"\nTop-level keys: {list(gsam_data.keys())}")
    
    # Experiments
    experiments = gsam_data.get("experiments", {})
    print(f"\nFound {len(experiments)} experiments:")
    
    exp_list = list(experiments.keys())
    for i, exp_id in enumerate(exp_list[:10]):  # Show first 10
        print(f"  {i+1}. {exp_id}")
    if len(exp_list) > 10:
        print(f"  ... and {len(exp_list) - 10} more")
    
    # Check for target experiments
    target_experiments = ["20240418", "20250305"]
    print(f"\nChecking for target experiments: {target_experiments}")
    for exp_id in target_experiments:
        if exp_id in experiments:
            print(f"✅ Found {exp_id}")
        else:
            print(f"❌ Missing {exp_id}")
            # Find similar experiment IDs
            similar = [e for e in exp_list if exp_id in e]
            if similar:
                print(f"   Similar IDs found: {similar[:5]}")
    
    # Detailed structure for first experiment
    if experiments:
        first_exp_id = list(experiments.keys())[0]
        first_exp_data = experiments[first_exp_id]
        
        print(f"\nDetailed structure for first experiment '{first_exp_id}':")
        print(f"  Keys: {list(first_exp_data.keys())}")
        
        videos = first_exp_data.get("videos", {})
        print(f"  Videos: {len(videos)}")
        
        if videos:
            first_video_id = list(videos.keys())[0]
            first_video_data = videos[first_video_id]
            
            print(f"  First video '{first_video_id}' structure:")
            print(f"    Keys: {list(first_video_data.keys())}")
            
            images = first_video_data.get("image_ids", {})
            print(f"    Images: {len(images)}")
            
            if images:
                first_image_id = list(images.keys())[0]
                first_image_data = images[first_image_id]
                
                print(f"    First image '{first_image_id}' structure:")
                print(f"      Keys: {list(first_image_data.keys())}")
                
                embryos = first_image_data.get("embryos", {})
                print(f"      Embryos: {len(embryos)}")
                
                if embryos:
                    first_embryo_id = list(embryos.keys())[0]
                    first_embryo_data = embryos[first_embryo_id]
                    
                    print(f"      First embryo '{first_embryo_id}' structure:")
                    print(f"        Keys: {list(first_embryo_data.keys())}")
                    
                    # Check for area data
                    area = first_embryo_data.get("area")
                    print(f"        Area: {area} (type: {type(area)})")
                    
                    # Check segmentation format
                    segmentation = first_embryo_data.get("segmentation")
                    if segmentation:
                        print(f"        Segmentation format: {segmentation.get('format', 'unknown')}")
                        print(f"        Segmentation keys: {list(segmentation.keys())}")

if __name__ == "__main__":
    import sys
    
    gsam_path = "data/segmentation/grounded_sam_segmentations.json"
    if len(sys.argv) > 1:
        gsam_path = sys.argv[1]
    
    inspect_gsam_structure(gsam_path)