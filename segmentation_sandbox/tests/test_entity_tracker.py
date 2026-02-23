#!/usr/bin/env python3
"""
Debug EntityIDTracker functionality
"""
import sys
import json
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_entity_tracker():
    """Test EntityIDTracker with our data"""
    from scripts.utils.entity_id_tracker import EntityIDTracker
    
    # Test data
    sam_data = {
        "experiments": {
            "test_exp_001": {
                "videos": {
                    "test_exp_001_A01": {
                        "embryo_ids": ["test_exp_001_A01_e01", "test_exp_001_A01_e02"],
                        "images": {
                            "test_exp_001_A01_ch00_t0100": {
                                "embryos": {
                                    "test_exp_001_A01_e01": {"snip_id": "test_exp_001_A01_e01_s0100"},
                                    "test_exp_001_A01_e02": {"snip_id": "test_exp_001_A01_e02_s0100"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    print("🔍 Testing EntityIDTracker...")
    print(f"Input data keys: {list(sam_data.keys())}")
    print(f"Experiments: {list(sam_data['experiments'].keys())}")
    
    # Extract entities
    entities = EntityIDTracker.extract_entities(sam_data)
    print(f"Extracted entities: {entities}")
    
    # Get counts
    counts = EntityIDTracker.get_counts(entities)
    print(f"Entity counts: {counts}")
    
    return True

if __name__ == "__main__":
    test_entity_tracker()