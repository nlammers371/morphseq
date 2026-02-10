#!/usr/bin/env python3
"""
Test script to verify hierarchical processing is actually working by adding debug prints.
"""

import json
import sys
from pathlib import Path

# Add script paths
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def test_with_debug_output():
    """Test QC with debug output to confirm it's processing child entities."""
    
    from detection_segmentation.gsam_quality_control import GSAMQualityControl
    
    gsam_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/segmentation/grounded_sam_segmentations.json"
    
    print("🔧 Testing QC with debug output for hierarchical processing...")
    
    # Initialize QC
    qc = GSAMQualityControl(gsam_path, verbose=True)
    
    # Target just one experiment to make output manageable
    target_entities = {
        "experiment_ids": ["20231206"],  # Just one experiment
        "video_ids": [],
        "image_ids": [], 
        "snip_ids": []
    }
    
    print(f"\n📊 Target entities: {target_entities}")
    
    # Add debug prints to the segmentation variability check temporarily
    print("\n🔍 Testing segmentation variability check with debug output...")
    
    # Manual implementation of part of check_segmentation_variability with debug
    experiments_items = list(qc.gsam_data.get("experiments", {}).items())
    
    video_count = 0
    image_count = 0 
    snip_count = 0
    
    for exp_id, exp_data in experiments_items:
        exp_targeted = qc._should_process_experiment(exp_id, target_entities)
        print(f"   Experiment {exp_id}: exp_targeted={exp_targeted}")
        
        if not exp_targeted:
            continue
            
        videos = exp_data.get("videos", {})
        print(f"      Found {len(videos)} videos in experiment {exp_id}")
        
        for video_id, video_data in videos.items():
            video_targeted = qc._should_process_video(video_id, target_entities, exp_targeted)
            if video_targeted:
                video_count += 1
                
                images = video_data.get("images", {})
                
                for image_id, image_data in images.items():
                    image_targeted = qc._should_process_image(image_id, target_entities, exp_targeted or video_targeted)
                    if image_targeted:
                        image_count += 1
                        
                        embryos = image_data.get("embryos", {})
                        for embryo_id, embryo_data in embryos.items():
                            snip_id = embryo_data.get("snip_id")
                            if snip_id:
                                snip_targeted = qc._should_process_snip(snip_id, target_entities, exp_targeted or video_targeted or image_targeted)
                                if snip_targeted:
                                    snip_count += 1
                                    
                        # Show sample details for first few images
                        if image_count <= 5:
                            print(f"         Image {image_id}: {len(embryos)} embryos, image_targeted={image_targeted}")
                
                # Show sample details for first few videos
                if video_count <= 3:
                    print(f"      Video {video_id}: {len(images)} images, video_targeted={video_targeted}")
                    
    print(f"\n📊 Processing summary for experiment 20231206:")
    print(f"   Videos to be processed: {video_count}")
    print(f"   Images to be processed: {image_count}")
    print(f"   Snips to be processed: {snip_count}")
    
    # Now run the actual QC to confirm 
    print(f"\n🚀 Running actual QC check...")
    qc.run_all_checks(
        author="debug_hierarchical",
        target_entities=target_entities,
        save_in_place=False  # Don't save
    )
    
    # Print summary
    qc.print_summary()

if __name__ == "__main__":
    test_with_debug_output()
