#!/usr/bin/env python3
"""
Test the _add_flag method directly to see if it works.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from detection_segmentation.gsam_quality_control import GSAMQualityControl

def test_add_flag():
    """Test the _add_flag method directly."""
    
    test_file = "test_gsam_violations.json"
    
    print("🔍 TESTING _add_flag METHOD")
    print("="*60)
    
    # Initialize QC
    qc = GSAMQualityControl(test_file, verbose=False)
    
    print("1. Initial state:")
    summary = qc.get_flags_summary()
    print(f"   Total flags: {summary['total_flags']}")
    
    print("\n2. Adding a test flag:")
    flag_data = {
        "snip_id": "20240411_test_H01_e04_s0000",
        "test": True,
        "timestamp": "2025-08-13T06:00:00"
    }
    
    qc._add_flag("TEST_FLAG", flag_data, "snip", "20240411_test_H01_e04_s0000")
    print("   Flag added")
    
    print("\n3. Check if flag was added:")
    summary = qc.get_flags_summary()
    print(f"   Total flags: {summary['total_flags']}")
    print(f"   Flag categories: {summary['flag_categories']}")
    
    # Check specific flags
    test_flags = qc.get_flags_by_type("TEST_FLAG")
    print(f"   TEST_FLAG flags: {len(test_flags)}")
    if test_flags:
        print(f"   First flag: {test_flags[0]}")
    
    print("\n4. Check flags structure directly:")
    flags_section = qc.gsam_data.get("flags", {})
    by_snip = flags_section.get("by_snip", {})
    snip_flags = by_snip.get("20240411_test_H01_e04_s0000", {})
    print(f"   Snip flags: {snip_flags}")

if __name__ == "__main__":
    test_add_flag()
