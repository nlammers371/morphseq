#!/usr/bin/env python3
"""
Test if generate_overview is being called correctly.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from detection_segmentation.gsam_quality_control import GSAMQualityControl

def test_overview_generation():
    """Test if overview generation works correctly."""
    
    test_file = "test_gsam_violations.json"
    
    print("🔍 TESTING OVERVIEW GENERATION")
    print("="*60)
    
    # Initialize QC
    qc = GSAMQualityControl(test_file, verbose=False)
    
    print("1. Add a test flag manually:")
    flag_data = {
        "snip_id": "20240411_test_H01_e04_s0000",
        "test": True,
        "timestamp": "2025-08-13T06:00:00"
    }
    
    qc._add_flag("TEST_FLAG", flag_data, "snip", "20240411_test_H01_e04_s0000")
    print("   Flag added")
    
    print("\n2. Before generate_overview:")
    summary = qc.get_flags_summary()
    print(f"   Total flags: {summary['total_flags']}")
    print(f"   Flag categories: {summary['flag_categories']}")
    
    print("\n3. Call generate_overview manually:")
    entities = qc.get_all_entities_to_process()
    qc.generate_overview(entities)
    print("   Overview generated")
    
    print("\n4. After generate_overview:")
    summary = qc.get_flags_summary()
    print(f"   Total flags: {summary['total_flags']}")
    print(f"   Flag categories: {summary['flag_categories']}")
    
    print("\n5. Test _count_flags_in_hierarchy directly:")
    flag_counts = qc._count_flags_in_hierarchy()
    print(f"   Direct flag counts: {flag_counts}")
    
    print("\n6. Check flag_overview in data:")
    overview = qc.gsam_data["flags"].get("flag_overview", {})
    print(f"   Overview: {overview}")

if __name__ == "__main__":
    test_overview_generation()
