#!/usr/bin/env python3
"""Test script to check for verbose parameter issues."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from scripts.detection_segmentation.sam2_utils import find_seed_frame_from_video_annotations
    
    # Test the function
    test_annotations = {
        "20231206_A01_t0000": [{"confidence": 0.8}],
        "20231206_A01_t0030": [{"confidence": 0.9}]
    }
    
    result = find_seed_frame_from_video_annotations(test_annotations, "20231206_A01", verbose=True)
    print(f"✅ Function test passed: {result[0]}")
    
except Exception as e:
    print(f"❌ Error testing function: {e}")
    import traceback
    traceback.print_exc()
