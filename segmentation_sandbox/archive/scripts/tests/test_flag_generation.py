#!/usr/bin/env python3
"""
Test with aggressive thresholds to confirm flags are being generated.
"""

import json
import sys
from pathlib import Path

# Add script paths
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

def test_flag_generation():
    """Test with aggressive thresholds to ensure flags are generated."""
    
    from detection_segmentation.gsam_quality_control import GSAMQualityControl
    
    gsam_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/segmentation/grounded_sam_segmentations.json"
    
    print("🔧 Testing flag generation with aggressive thresholds...")
    
    # Initialize QC
    qc = GSAMQualityControl(gsam_path, verbose=True)
    
    # Target just one experiment, but limit to first video for speed
    with open(gsam_path, 'r') as f:
        data = json.load(f)
    
    # Get first video ID from experiment 20231206
    exp_20231206 = data["experiments"]["20231206"]
    first_video_id = list(exp_20231206["videos"].keys())[0]
    
    target_entities = {
        "experiment_ids": [],
        "video_ids": [first_video_id],  # Target just one video
        "image_ids": [], 
        "snip_ids": []
    }
    
    print(f"\n📊 Target entities: {target_entities}")
    print(f"   This should process video {first_video_id} and all its children")
    
    # Run QC
    print(f"\n🚀 Running QC with video-level targeting...")
    qc.run_all_checks(
        author="debug_video_test",
        target_entities=target_entities,
        save_in_place=False
    )
    
    # Print summary
    qc.print_summary()
    
    # Show flag details if any were generated
    flag_summary = qc.get_flags_summary()
    if flag_summary["total_flags"] > 0:
        print(f"\n📝 Flag details:")
        for flag_type, count in flag_summary["flag_categories"].items():
            if count > 0:
                flags = qc.get_flags_by_type(flag_type)
                print(f"   {flag_type}: {count} instances")
                for flag in flags[:3]:  # Show first 3
                    print(f"      {flag}")

if __name__ == "__main__":
    test_flag_generation()
