#!/usr/bin/env python3
"""
Test filename parsing with actual 20240411 data.
"""

import os
import re
from pathlib import Path

def extract_well_id(filename: str) -> str:
    """Extract well ID from filename using the proven parsing logic."""
    # Remove extension - handle both with and without extension
    name, ext = os.path.splitext(filename)
    
    # Split by underscore: ['A01', 't0000', 'ch00', 'stitch']
    parts = name.split("_")
    
    if len(parts) < 2:
        return None
        
    # First part should be the well ID (A01, B12, etc.)
    well = parts[0]
    
    # Validate it looks like a well ID (letter + 2 digits)
    if re.match(r'^[A-H]\d{2}$', well.upper()):
        return well.upper()
        
    return None

def extract_timepoint(filename: str) -> int:
    """Extract timepoint number from filename using proven parsing logic."""
    # Remove extension
    name, ext = os.path.splitext(filename)
    
    # Split by underscore: ['A01', 't0000', 'ch00', 'stitch'] 
    parts = name.split("_")
    
    if len(parts) < 2:
        return 0
        
    # Second part should be timepoint (t0000, t0001, etc.)
    time_str = parts[1]
    
    if time_str.startswith('t'):
        time_str = time_str[1:]  # Remove 't' prefix: '0000'
        try:
            return int(time_str)
        except ValueError:
            pass
    
    # Fallback: look for any t followed by digits
    t_match = re.search(r't(\d+)', filename.lower())
    if t_match:
        return int(t_match.group(1))
    
    return 0

def test_filename_parsing():
    """Test filename parsing with actual data."""
    
    # Test directory
    data_dir = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images/20240411")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return False
    
    # Get first 10 files for testing
    jpg_files = list(data_dir.glob("*.jpg"))[:10]
    
    if not jpg_files:
        print("No JPG files found!")
        return False
    
    print(f"Testing filename parsing with {len(jpg_files)} sample files from 20240411:")
    print()
    
    well_counts = {}
    timepoint_ranges = {}
    
    for file_path in jpg_files:
        filename = file_path.name
        well_id = extract_well_id(filename)
        timepoint = extract_timepoint(filename)
        
        print(f"File: {filename}")
        print(f"  Well ID: {well_id}")
        print(f"  Timepoint: {timepoint}")
        print()
        
        if well_id:
            if well_id not in well_counts:
                well_counts[well_id] = 0
                timepoint_ranges[well_id] = []
            well_counts[well_id] += 1
            timepoint_ranges[well_id].append(timepoint)
    
    print("Summary:")
    print(f"  Wells found: {list(well_counts.keys())}")
    for well, count in well_counts.items():
        tp_range = timepoint_ranges[well]
        print(f"  {well}: {count} files, timepoints {min(tp_range)}-{max(tp_range)}")
    
    return True

if __name__ == "__main__":
    success = test_filename_parsing()
    if success:
        print("\n✓ Filename parsing test completed!")
    else:
        print("\n✗ Filename parsing test failed!")
