#!/usr/bin/env python3

"""
Debug Entity Expansion
=======================
Test if the _expand_target_entities method is working correctly.
"""

import sys
sys.path.append('scripts/detection_segmentation')
sys.path.append('scripts/utils')

from gsam_quality_control import GSAMQualityControl

def test_entity_expansion():
    """Test the _expand_target_entities method."""
    
    # Load test data
    qc = GSAMQualityControl("test_gsam_violations.json", verbose=True)
    
    print("\n" + "="*60)
    print("TESTING ENTITY EXPANSION")
    print("="*60)
    
    # Test 1: Experiment-only input (should expand to all sub-entities)
    experiment_only = {
        "experiment_ids": ["20240411_test"],
        "video_ids": [],
        "image_ids": [],
        "snip_ids": []
    }
    
    print(f"\n🧪 Test 1: Experiment-only input")
    print(f"   Input: {experiment_only}")
    
    expanded = qc._expand_target_entities(experiment_only)
    
    print(f"   Output:")
    print(f"     Experiments: {len(expanded['experiment_ids'])} - {expanded['experiment_ids'][:3]}...")
    print(f"     Videos: {len(expanded['video_ids'])} - {expanded['video_ids'][:3]}...")
    print(f"     Images: {len(expanded['image_ids'])} - {expanded['image_ids'][:3]}...")
    print(f"     Snips: {len(expanded['snip_ids'])} - {expanded['snip_ids'][:5]}...")
    
    # Test 2: Empty input (should expand to all new entities)
    empty_input = {
        "experiment_ids": [],
        "video_ids": [],
        "image_ids": [],
        "snip_ids": []
    }
    
    print(f"\n🧪 Test 2: Empty input (should use new entities)")
    print(f"   Input: {empty_input}")
    
    expanded_empty = qc._expand_target_entities(empty_input)
    
    print(f"   Output:")
    print(f"     Experiments: {len(expanded_empty['experiment_ids'])} - {expanded_empty['experiment_ids']}")
    print(f"     Videos: {len(expanded_empty['video_ids'])} - {expanded_empty['video_ids']}")
    print(f"     Images: {len(expanded_empty['image_ids'])} - {expanded_empty['image_ids'][:3]}...")
    print(f"     Snips: {len(expanded_empty['snip_ids'])} - {expanded_empty['snip_ids'][:5]}...")
    
    # Test 3: Specific snip list
    specific_snips = {
        "experiment_ids": [],
        "video_ids": [],
        "image_ids": [],
        "snip_ids": ["20240411_test_H01_e01_t0000", "20240411_test_H01_e02_t0000"]
    }
    
    print(f"\n🧪 Test 3: Specific snips")
    print(f"   Input: {specific_snips}")
    
    expanded_specific = qc._expand_target_entities(specific_snips)
    
    print(f"   Output:")
    print(f"     Experiments: {len(expanded_specific['experiment_ids'])} - {expanded_specific['experiment_ids']}")
    print(f"     Videos: {len(expanded_specific['video_ids'])} - {expanded_specific['video_ids']}")
    print(f"     Images: {len(expanded_specific['image_ids'])} - {expanded_specific['image_ids']}")
    print(f"     Snips: {len(expanded_specific['snip_ids'])} - {expanded_specific['snip_ids']}")


if __name__ == "__main__":
    test_entity_expansion()
