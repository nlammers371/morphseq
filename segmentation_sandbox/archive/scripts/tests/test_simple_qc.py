#!/usr/bin/env python3
"""
Simple test of QC with process_all=True to bypass filtering.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from detection_segmentation.gsam_quality_control import GSAMQualityControl

def test_simple_qc():
    """Test QC with process_all=True to bypass all filtering."""
    
    test_file = "test_gsam_violations.json"
    
    print("🔍 SIMPLE QC TEST WITH PROCESS_ALL=TRUE")
    print("="*60)
    
    # Initialize QC
    qc = GSAMQualityControl(test_file, verbose=True)
    
    print("\n1. Running with process_all=True (should find violations):")
    qc.run_all_checks(
        author="debug_simple",
        process_all=True,  # This should bypass all filtering
        save_in_place=False
    )
    
    # Check results
    summary = qc.get_flags_summary()
    print(f"\n   📊 Results: {summary['total_flags']} total flags")
    
    for flag_type, count in summary['flag_categories'].items():
        if count > 0:
            print(f"      ✅ {flag_type}: {count}")
            # Show sample flags
            flags = qc.get_flags_by_type(flag_type)
            if flags:
                print(f"         Sample: {flags[0]['snip_id']}")

if __name__ == "__main__":
    test_simple_qc()
