#!/usr/bin/env python3
"""
Debug script to verify hierarchical entity processing is working correctly.
"""

import json
import sys
from pathlib import Path

# Add script paths
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def test_hierarchical_processing():
    """Test that hierarchical processing works correctly."""
    
    # Import the QC class
    from detection_segmentation.gsam_quality_control import GSAMQualityControl
    
    gsam_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/segmentation/grounded_sam_segmentations.json"
    
    print("🔧 Testing hierarchical entity processing...")
    
    # Initialize QC
    qc = GSAMQualityControl(gsam_path, verbose=True)
    
    # Define target entities (experiment level targeting)
    target_entities = {
        "experiment_ids": ["20231206"],  # Just one experiment
        "video_ids": [],
        "image_ids": [], 
        "snip_ids": []
    }
    
    print(f"\n📊 Target entities: {target_entities}")
    
    # Test the hierarchical filtering functions directly
    print("\n🔍 Testing hierarchical filtering functions:")
    
    # Should process experiment 20231206
    should_process = qc._should_process_experiment("20231206", target_entities)
    print(f"   _should_process_experiment('20231206'): {should_process}")
    
    # Should NOT process experiment 20240418
    should_process = qc._should_process_experiment("20240418", target_entities)
    print(f"   _should_process_experiment('20240418'): {should_process}")
    
    # Should process video (inherits from experiment)
    should_process = qc._should_process_video("20231206_D06", target_entities, exp_targeted=True)
    print(f"   _should_process_video('20231206_D06', exp_targeted=True): {should_process}")
    
    # Should NOT process video (no inheritance)
    should_process = qc._should_process_video("20231206_D06", target_entities, exp_targeted=False)
    print(f"   _should_process_video('20231206_D06', exp_targeted=False): {should_process}")
    
    # Test manual iteration through the data structure
    print("\n🔍 Manual test of experiment 20231206 processing:")
    
    experiments = qc.gsam_data.get("experiments", {})
    video_count = 0
    image_count = 0
    snip_count = 0
    
    for exp_id, exp_data in experiments.items():
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
                                    
                        # Only show details for first few images
                        if image_count <= 3:
                            print(f"         Image {image_id}: {len(embryos)} embryos, image_targeted={image_targeted}")
                
                # Only show details for first few videos  
                if video_count <= 3:
                    print(f"      Video {video_id}: {len(images)} images, video_targeted={video_targeted}")
    
    print(f"\n📊 Manual count results for experiment 20231206:")
    print(f"   Videos that would be processed: {video_count}")
    print(f"   Images that would be processed: {image_count}")
    print(f"   Snips that would be processed: {snip_count}")

if __name__ == "__main__":
    test_hierarchical_processing()
