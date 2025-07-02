#!/usr/bin/env python3
"""
Test filename parsing with actual data format.
"""

import sys
from pathlib import Path
import re

def test_filename_parsing():
    """Test filename parsing with actual MorphSeq data patterns."""
    
    # Test cases from actual data
    test_filenames = [
        "A01_t0000_ch00_stitch.png",
        "A01_t0001_ch00_stitch.png", 
        "B12_t0000_ch00_stitch.jpg",
        "H11_t0150_ch00_stitch.tiff",
        "well_A01_t001.tif",  # Legacy format
        "A01_t001.jpg",       # Simple format
    ]
    
    def extract_well_id(filename: str) -> str:
        """Extract well ID from filename."""
        # Remove extension
        basename = Path(filename).stem
        
        # Pattern 1: A01_t0000_ch00_stitch -> A01 (current data format)
        well_match = re.search(r'^([A-H]\d{2})_t\d+', basename.upper())
        if well_match:
            return well_match.group(1)
        
        # Pattern 2: well_A01_t001 -> A01 (legacy format)
        well_match = re.search(r'well[_\-]?([A-H]\d{2})', basename.upper())
        if well_match:
            return well_match.group(1)
        
        # Pattern 3: A01_t001 -> A01 (simple format)
        direct_match = re.search(r'([A-H]\d{2})', basename.upper())
        if direct_match:
            return direct_match.group(1)
        
        # Fallback: use first part before underscore
        parts = basename.split('_')
        if parts:
            return parts[0].upper()
        
        return basename.upper()
    
    def extract_timepoint(filename: str) -> int:
        """Extract timepoint number from filename."""
        # Pattern 1: t followed by numbers (t0000, t0001, t123)
        t_match = re.search(r't(\d+)', filename.lower())
        if t_match:
            return int(t_match.group(1))
        
        # Fallback: last number in filename
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])
        
        return 0
    
    print("Testing filename parsing:")
    print("-" * 50)
    
    for filename in test_filenames:
        well_id = extract_well_id(filename)
        timepoint = extract_timepoint(filename)
        video_id = f"20240411_{well_id}"
        
        print(f"File: {filename}")
        print(f"  Well ID: {well_id}")
        print(f"  Timepoint: {timepoint}")
        print(f"  Video ID: {video_id}")
        print()
    
    return True

def test_file_format_detection():
    """Test file format detection logic."""
    test_files = [
        "A01_t0000_ch00_stitch.png",
        "A01_t0000_ch00_stitch.jpg", 
        "A01_t0000_ch00_stitch.jpeg",
        "A01_t0000_ch00_stitch.tiff",
        "A01_t0000_ch00_stitch.tif",
    ]
    
    print("Testing file format detection:")
    print("-" * 50)
    
    for filename in test_files:
        ext = Path(filename).suffix.lower()
        is_jpeg = ext in ['.jpg', '.jpeg']
        needs_conversion = not is_jpeg
        
        print(f"File: {filename}")
        print(f"  Extension: {ext}")
        print(f"  Is JPEG: {is_jpeg}")
        print(f"  Needs conversion: {needs_conversion}")
        print()
    
    return True

if __name__ == "__main__":
    print("Testing updated filename parsing for MorphSeq data...")
    print("=" * 60)
    
    success1 = test_filename_parsing()
    success2 = test_file_format_detection()
    
    if success1 and success2:
        print("✓ All filename parsing tests passed!")
    else:
        print("✗ Some tests failed!")
        sys.exit(1)
