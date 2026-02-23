#!/usr/bin/env python3

"""
Debug Entity Processing Flow
============================
Test what entities are actually being passed to the QC check methods.
"""

import sys
sys.path.append('scripts/detection_segmentation')
sys.path.append('scripts/utils')

from gsam_quality_control import GSAMQualityControl

def debug_entity_processing():
    """Debug the entity processing flow."""
    
    # Load test data
    qc = GSAMQualityControl("test_gsam_violations.json", verbose=True)
    
    print("\n" + "="*60)
    print("DEBUGGING ENTITY PROCESSING FLOW")
    print("="*60)
    
    # Test get_all_entities_to_process
    print(f"\n🧪 Testing get_all_entities_to_process():")
    all_entities = qc.get_all_entities_to_process()
    print(f"   Experiments: {len(all_entities['experiment_ids'])} - {all_entities['experiment_ids']}")
    print(f"   Videos: {len(all_entities['video_ids'])} - {all_entities['video_ids']}")
    print(f"   Images: {len(all_entities['image_ids'])} - {all_entities['image_ids'][:3]}...")
    print(f"   Snips: {len(all_entities['snip_ids'])} - {all_entities['snip_ids'][:5]}...")
    
    # Test get_new_entities_to_process
    print(f"\n🧪 Testing get_new_entities_to_process():")
    new_entities = qc.get_new_entities_to_process()
    print(f"   Experiments: {len(new_entities['experiment_ids'])} - {new_entities['experiment_ids']}")
    print(f"   Videos: {len(new_entities['video_ids'])} - {new_entities['video_ids']}")
    print(f"   Images: {len(new_entities['image_ids'])} - {new_entities['image_ids'][:3]}...")
    print(f"   Snips: {len(new_entities['snip_ids'])} - {new_entities['snip_ids'][:5]}...")
    
    # Simulate the run_all_checks logic for process_all=True
    print(f"\n🧪 Simulating run_all_checks(process_all=True) logic:")
    target_entities = None
    process_all = True
    
    if target_entities and process_all:
        entities_to_process = qc._expand_target_entities(target_entities)
        print("   Branch: target_entities + process_all")
    elif target_entities:
        entities_to_process = qc._expand_target_entities(target_entities)
        print("   Branch: target_entities only")
    elif process_all:
        entities_to_process = qc.get_all_entities_to_process()
        print("   Branch: process_all only")
    else:
        entities_to_process = qc.get_new_entities_to_process()
        print("   Branch: default (new entities)")
    
    print(f"   Result:")
    print(f"     Experiments: {len(entities_to_process['experiment_ids'])} - {entities_to_process['experiment_ids']}")
    print(f"     Videos: {len(entities_to_process['video_ids'])} - {entities_to_process['video_ids']}")
    print(f"     Images: {len(entities_to_process['image_ids'])} - {entities_to_process['image_ids'][:3]}...")
    print(f"     Snips: {len(entities_to_process['snip_ids'])} - {entities_to_process['snip_ids'][:5]}...")
    
    # Now test the check_segmentation_variability method directly with these entities
    print(f"\n🧪 Testing check_segmentation_variability directly:")
    
    # Add some debug prints to the method temporarily
    import time
    
    if len(entities_to_process['snip_ids']) > 0:
        print(f"   Calling check_segmentation_variability with {len(entities_to_process['snip_ids'])} snips...")
        t0 = time.time()
        qc.check_segmentation_variability("debug_test", entities_to_process)
        elapsed = time.time() - t0
        print(f"   Completed in {elapsed:.2f}s")
        
        # Check flags
        flags = qc.gsam_data.get("flags", {})
        total_flags = 0
        for entity_type in ["by_experiment", "by_video", "by_image", "by_snip", "by_embryo"]:
            if entity_type in flags:
                for entity_id, entity_flags in flags[entity_type].items():
                    for flag_type, flag_instances in entity_flags.items():
                        total_flags += len(flag_instances)
                        print(f"   Found {len(flag_instances)} {flag_type} flags for {entity_id}")
        
        print(f"   Total flags created: {total_flags}")
    else:
        print("   No snips to process!")


if __name__ == "__main__":
    debug_entity_processing()
