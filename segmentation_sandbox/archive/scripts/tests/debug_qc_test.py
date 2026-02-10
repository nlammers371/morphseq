#!/usr/bin/env python3
"""
Simple test to debug QC entity processing issues.
"""

import sys
import json
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from detection_segmentation.gsam_quality_control import GSAMQualityControl

def test_entity_targeting():
    """Test the entity targeting logic."""
    
    # Load the test file
    test_file = "test_gsam_violations.json"
    
    if not Path(test_file).exists():
        print(f"❌ Test file {test_file} not found. Run create_dummy_gsam.py first.")
        return
    
    print(f"🔍 Testing entity targeting with {test_file}")
    
    # Initialize QC
    qc = GSAMQualityControl(test_file, verbose=True)
    
    print("\n" + "="*60)
    print("TESTING ENTITY TARGETING")
    print("="*60)
    
    # Test 1: Get all entities
    print("\n1. Testing get_all_entities_to_process():")
    all_entities = qc.get_all_entities_to_process()
    for entity_type, entities in all_entities.items():
        print(f"   {entity_type}: {len(entities)} entities")
        if entities:
            print(f"      Sample: {entities[:3]}")
    
    # Test 2: Get entities and run QC
    print("\n2. Testing run_all_checks with all entities:")
    all_entities = qc.get_all_entities_to_process()
    
    # Debug: Print what entities we're targeting
    print("\n   🎯 Entities to target:")
    for entity_type, entities in all_entities.items():
        print(f"      {entity_type}: {len(entities)} entities")
        if entities:
            print(f"         First few: {entities[:3]}")
    
    qc.run_all_checks(
        author="debug_test", 
        target_entities=all_entities,
        save_in_place=False
    )
    
    # Check results
    summary = qc.get_flags_summary()
    print(f"\n   Results: {summary['total_flags']} total flags")
    for flag_type, count in summary['flag_categories'].items():
        if count > 0:
            print(f"      {flag_type}: {count}")
    
    # Test 3: Test with empty targets (should process nothing)
    print("\n3. Testing run_all_checks with empty targets (should process nothing):")
    empty_entities = {
        "experiment_ids": [],
        "video_ids": [],
        "image_ids": [],
        "snip_ids": []
    }
    
    # Reset QC state for clean test
    qc2 = GSAMQualityControl(test_file, verbose=False)
    qc2.run_all_checks(
        author="debug_test_empty",
        target_entities=empty_entities,
        save_in_place=False
    )
    
    summary2 = qc2.get_flags_summary()
    print(f"   Results: {summary2['total_flags']} total flags (should be 0)")
    
    # Test 4: Test process_all=True (should process all regardless of targets)
    print("\n4. Testing run_all_checks with process_all=True:")
    qc3 = GSAMQualityControl(test_file, verbose=False)
    qc3.run_all_checks(
        author="debug_test_all",
        process_all=True,
        save_in_place=False
    )
    
    summary3 = qc3.get_flags_summary()
    print(f"   Results: {summary3['total_flags']} total flags")
    for flag_type, count in summary3['flag_categories'].items():
        if count > 0:
            print(f"      {flag_type}: {count}")

def test_manual_entity_extraction():
    """Manually extract entities to verify data structure."""
    
    test_file = "test_gsam_violations.json"
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    print("\n" + "="*60)
    print("MANUAL ENTITY EXTRACTION")
    print("="*60)
    
    experiments = []
    videos = []
    images = []
    snips = []
    
    for exp_id, exp_data in data.get("experiments", {}).items():
        experiments.append(exp_id)
        print(f"📁 Experiment: {exp_id}")
        
        for video_id, video_data in exp_data.get("videos", {}).items():
            videos.append(video_id)
            print(f"   📹 Video: {video_id}")
            
            for image_id, image_data in video_data.get("images", {}).items():
                images.append(image_id)
                embryo_count = len(image_data.get("embryos", {}))
                
                for embryo_id, embryo_data in image_data.get("embryos", {}).items():
                    snip_id = embryo_data.get("snip_id")
                    if snip_id:
                        snips.append(snip_id)
                
                print(f"      🖼️  Image: {image_id} ({embryo_count} embryos)")
    
    print(f"\nCounts:")
    print(f"   Experiments: {len(experiments)}")
    print(f"   Videos: {len(videos)}")
    print(f"   Images: {len(images)}")
    print(f"   Snips: {len(snips)}")
    print(f"   Sample snips: {snips[:5]}")

if __name__ == "__main__":
    print("🧪 QC DEBUG TEST")
    print("="*60)
    
    test_manual_entity_extraction()
    test_entity_targeting()
