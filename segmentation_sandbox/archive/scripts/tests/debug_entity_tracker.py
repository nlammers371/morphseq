#!/usr/bin/env python3
"""
Debug the EntityIDTracker extraction to see why it returns 0 entities.
"""

import sys
import json
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from detection_segmentation.gsam_quality_control import GSAMQualityControl
from utils.entity_id_tracker import EntityIDTracker

def debug_entity_extraction():
    """Debug the EntityIDTracker to see why it's not finding entities."""
    
    test_file = "test_gsam_violations.json"
    
    print("🔍 DEBUGGING ENTITY EXTRACTION")
    print("="*60)
    
    # Load data directly
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    print("1. Direct data inspection:")
    print(f"   Experiments in data: {list(data.get('experiments', {}).keys())}")
    
    # Test EntityIDTracker directly
    print("\n2. EntityIDTracker.extract_entities():")
    entities = EntityIDTracker.extract_entities(data)
    print(f"   Result: {entities}")
    for entity_type, entity_list in entities.items():
        print(f"   {entity_type}: {len(entity_list)} entities")
        if entity_list:
            print(f"      Sample: {entity_list[:3]}")
    
    # Test within QC class
    print("\n3. Within GSAMQualityControl:")
    qc = GSAMQualityControl(test_file, verbose=False)
    
    all_entities = qc.get_all_entities_to_process()
    print(f"   get_all_entities_to_process(): {all_entities}")
    for entity_type, entity_list in all_entities.items():
        print(f"   {entity_type}: {len(entity_list)} entities")
    
    new_entities = qc.get_new_entities_to_process()
    print(f"   get_new_entities_to_process(): {new_entities}")
    for entity_type, entity_list in new_entities.items():
        print(f"   {entity_type}: {len(entity_list)} entities")
    
    # Check processed entity tracking
    print(f"\n4. Processed entity tracking:")
    print(f"   processed_snip_ids: {len(qc.processed_snip_ids)}")
    print(f"   new_snip_ids: {len(qc.new_snip_ids)}")
    print(f"   Sample processed: {list(qc.processed_snip_ids)[:5] if qc.processed_snip_ids else 'None'}")
    print(f"   Sample new: {list(qc.new_snip_ids)[:5] if qc.new_snip_ids else 'None'}")

if __name__ == "__main__":
    debug_entity_extraction()
