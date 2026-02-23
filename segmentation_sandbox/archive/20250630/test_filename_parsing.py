#!/usr/bin/env python
"""
Test filename parsing for the video preparation script.
"""

import re

def extract_video_id(filename: str) -> str:
    """Extract video identifier from filename (e.g., well_A01 from well_A01_t001.tif)."""
    # Remove file extension
    base_name = filename.split('.')[0]
    
    # Common patterns to extract well ID
    # Pattern 1: well_A01_t001 -> well_A01
    # Pattern 2: well_A01_timepoint_001 -> well_A01
    parts = base_name.split('_')
    
    if len(parts) >= 2:
        # Look for well pattern (well_A01, well_B12, etc.)
        if parts[0].lower() == 'well' and len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"  # e.g., "well_A01"
        # Alternative: first two parts
        return f"{parts[0]}_{parts[1]}"
    
    # Fallback to full filename without extension
    return base_name

def extract_timepoint(filename: str) -> int:
    """Extract timepoint number from filename."""
    # Look for timepoint patterns: t001, timepoint_001, frame_001, etc.
    
    # Pattern 1: t followed by numbers (t001, t123)
    t_match = re.search(r't(\d+)', filename.lower())
    if t_match:
        return int(t_match.group(1))
    
    # Pattern 2: timepoint followed by numbers
    tp_match = re.search(r'timepoint[_\-]?(\d+)', filename.lower())
    if tp_match:
        return int(tp_match.group(1))
    
    # Pattern 3: frame followed by numbers
    frame_match = re.search(r'frame[_\-]?(\d+)', filename.lower())
    if frame_match:
        return int(frame_match.group(1))
    
    # Fallback: last number in filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])
    
    return 0

def test_filename_parsing():
    """Test the filename parsing methods."""
    
    # Test cases for filename parsing
    test_files = [
        "well_A01_t001.tif",
        "well_A01_t002.tif", 
        "well_B12_timepoint_001.tif",
        "well_C05_frame_123.tif",
        "plate1_well_A01_t001.tif",
        "experiment_well_A01_001.tif"
    ]
    
    print("Testing video ID extraction:")
    for filename in test_files:
        video_id = extract_video_id(filename)
        timepoint = extract_timepoint(filename)
        print(f"  {filename:30} -> video_id: {video_id:15} timepoint: {timepoint}")
    
    print("\nTesting experiment + video ID combination:")
    experiment_ids = ["20241215", "20241220"]
    for exp_id in experiment_ids:
        for filename in test_files[:3]:  # Just test first few
            video_id = extract_video_id(filename)
            full_id = f"{exp_id}_{video_id}"
            print(f"  {exp_id} + {filename:20} -> {full_id}")

if __name__ == "__main__":
    test_filename_parsing()
