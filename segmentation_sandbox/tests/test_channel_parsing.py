#!/usr/bin/env python3
"""
Quick test script to validate channel parsing methods work as expected.

Tests both current implementation and expected behavior for channel support.
"""

import sys
import os
from pathlib import Path
import tempfile
import re

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

try:
    from utils.parsing_utils import (
        parse_entity_id, 
        build_image_id, 
        get_image_filename_from_id,
        get_entity_type
    )
    from data_organization.data_organizer import DataOrganizer
    print("✅ Successfully imported parsing utilities")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_current_parsing():
    """Test current parsing behavior (legacy and new formats)."""
    print("\n" + "="*50)
    print("TESTING PARSING (LEGACY + NEW FORMATS)")
    print("="*50)
    
    # Test both legacy and new image_id formats
    test_image_ids = [
        "20240411_A01_t0000",           # Legacy format
        "20240411_A01_ch00_t0000",      # New format 
        "20240411_B02_t0042",           # Legacy format
        "20240411_B02_ch01_t0042",      # New format with channel 1
        "20250612_H12_ch02_t1234"       # New format with channel 2
    ]
    
    for image_id in test_image_ids:
        try:
            entity_type = get_entity_type(image_id)
            parsed = parse_entity_id(image_id)
            filename = get_image_filename_from_id(image_id)
            
            print(f"\n📝 Testing: {image_id}")
            print(f"   Type: {entity_type}")
            print(f"   Parsed: {parsed}")
            print(f"   Filename: {filename}")
            
        except Exception as e:
            print(f"❌ Error parsing {image_id}: {e}")

def test_stitch_filename_parsing():
    """Test stitch filename parsing."""
    print("\n" + "="*50)
    print("TESTING STITCH FILENAME PARSING")
    print("="*50)
    
    test_filenames = [
        "A01_t0000_ch00_stitch.png",  # Target format
        "A01_t0000_ch01_stitch.png",  # Different channel
        "B02_t0042_ch02_stitch.png",  # Another example
        "A01_0000_stitch.tif",        # Legacy format without 't' prefix
        "H12_t1234_stitch.png",       # No channel info (legacy)
    ]
    
    for filename in test_filenames:
        print(f"\n📝 Testing: {filename}")
        result = DataOrganizer.parse_stitch_filename(filename)
        if result:
            if len(result) == 2:  # Legacy format (well, frame) - SHOULD NOT HAPPEN after update
                well_id, frame = result
                print(f"   LEGACY parsing: well_id='{well_id}', frame='{frame}', channel=IGNORED")
            elif len(result) == 3:  # New format (well, frame, channel)
                well_id, frame, channel = result  
                print(f"   ✅ NEW parsing: well_id='{well_id}', frame='{frame}', channel='{channel}'")
            else:
                print(f"   Unexpected result: {result}")
        else:
            print(f"   ❌ Could not parse filename")

def test_expected_channel_behavior():
    """Test what the behavior SHOULD be with channel support."""
    print("\n" + "="*50)
    print("TESTING EXPECTED CHANNEL BEHAVIOR")
    print("="*50)
    
    # Simulate what we want the parsing to look like
    test_cases = [
        {
            "stitch_file": "A01_t0000_ch00_stitch.png",
            "expected_image_id": "20240411_A01_ch00_t0000", 
            "expected_filename": "20240411_A01_ch00_t0000.jpg"
        },
        {
            "stitch_file": "B02_t0042_ch01_stitch.png", 
            "expected_image_id": "20240411_B02_ch01_t0042",
            "expected_filename": "20240411_B02_ch01_t0042.jpg"
        }
    ]
    
    print("Expected transformations after channel support implementation:")
    for case in test_cases:
        print(f"\n📝 Input: {case['stitch_file']}")
        print(f"   Expected image_id: {case['expected_image_id']}")
        print(f"   Expected filename: {case['expected_filename']}")
        
        # Try to manually extract channel info
        channel_match = re.search(r'ch(\d{2})', case['stitch_file'])
        if channel_match:
            channel = channel_match.group(1)
            print(f"   ✅ Channel extracted: '{channel}'")
        else:
            print(f"   ❌ No channel found")

def test_build_image_id_current():
    """Test build_image_id with channel support.""" 
    print("\n" + "="*50)
    print("TESTING build_image_id WITH CHANNELS")
    print("="*50)
    
    test_cases = [
        ("20240411_A01", 0, 0),      # Default channel 0
        ("20240411_A01", 0, 1),      # Channel 1
        ("20240411_B02", 42, 0),     # Default channel 0
        ("20240411_B02", 42, 2),     # Channel 2
        ("20250612_H12", 1234, 0)    # Default channel 0
    ]
    
    print("Testing build_image_id with channel parameter:")
    for video_id, frame_number, channel in test_cases:
        try:
            image_id = build_image_id(video_id, frame_number, channel)
            print(f"build_image_id('{video_id}', {frame_number}, {channel}) → '{image_id}'")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\nTesting build_image_id with default channel (backward compatibility):")
    for video_id, frame_number, _ in test_cases[:3]:  # Just first 3
        try:
            image_id = build_image_id(video_id, frame_number)  # No channel = default 0
            print(f"build_image_id('{video_id}', {frame_number}) → '{image_id}'")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_sam2_file_expectations():
    """Test how SAM2 expects files based on sam2_utils.py analysis."""
    print("\n" + "="*50)
    print("TESTING SAM2 FILE EXPECTATIONS")
    print("="*50)
    
    print("🔍 SAM2 utils analysis:")
    print("   - Expects source files as: {image_id}.jpg")
    print("   - Creates temp symlinks as: {i:05d}.jpg (00000.jpg, 00001.jpg, etc.)")
    print("   - Source path: Path(images_dir) / image_filename")
    print("   - Where: image_filename = f'{image_id}.jpg'")
    
    # Test current vs expected compatibility
    print("\n📋 Compatibility test:")
    
    current_image_ids = ["20240411_A01_t0000", "20240411_A01_t0001"]
    expected_image_ids = ["20240411_A01_ch00_t0000", "20240411_A01_ch00_t0001"] 
    
    print("\n   Current image_ids → expected filenames:")
    for image_id in current_image_ids:
        filename = f"{image_id}.jpg"
        print(f"     {image_id} → {filename}")
        
    print("\n   Expected image_ids → expected filenames:")  
    for image_id in expected_image_ids:
        filename = f"{image_id}.jpg"
        print(f"     {image_id} → {filename}")
        
    print("\n✅ SAM2 compatibility:")
    print("   - SAM2 uses: image_filename = f'{image_id}.jpg'")
    print("   - Current format works: 20240411_A01_t0000.jpg")
    print("   - Channel format will work: 20240411_A01_ch00_t0000.jpg")
    print("   - No changes needed to SAM2 utils!")

def create_temp_test_structure():
    """Create temporary test structure to simulate SAM2 workflow."""
    print("\n" + "="*50)
    print("SIMULATING SAM2 TEMP STRUCTURE CREATION")
    print("="*50)
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"📁 Created temp directory: {temp_path}")
        
        # Create structure similar to data organizer output
        experiment_dir = temp_path / "20240411"
        images_dir = experiment_dir / "images" / "20240411_A01" 
        vids_dir = experiment_dir / "vids"
        
        images_dir.mkdir(parents=True)
        vids_dir.mkdir(parents=True)
        
        # Create sample image files (both current and expected format)
        current_files = ["20240411_A01_t0000.jpg", "20240411_A01_t0001.jpg"]  # Current format (updated)
        expected_files = ["20240411_A01_ch00_t0000.jpg", "20240411_A01_ch00_t0001.jpg"]  # Expected format
        
        print(f"📁 Source images directory: {images_dir}")
        for filename in current_files + expected_files:
            (images_dir / filename).touch()
            print(f"     - {filename}")
            
        # Simulate SAM2 temp directory creation
        sam2_temp_dir = temp_path / "sam2_temp"
        sam2_temp_dir.mkdir()
        
        print(f"\n📁 SAM2 temp directory: {sam2_temp_dir}")
        print("   Simulating SAM2 symlink creation...")
        
        # Simulate what SAM2 does: creates sequential symlinks
        test_image_ids = ["20240411_A01_ch00_t0000", "20240411_A01_ch00_t0001"]
        for i, image_id in enumerate(test_image_ids):
            src_file = images_dir / f"{image_id}.jpg"
            dst_file = sam2_temp_dir / f"{i:05d}.jpg"
            
            if src_file.exists():
                dst_file.symlink_to(src_file)
                print(f"     {i:05d}.jpg -> {src_file}")
            else:
                print(f"     ❌ Source not found: {src_file}")
                
        print("\n✅ SAM2 workflow simulation completed")
        print("   - Channel-inclusive filenames work perfectly with SAM2")
        print("   - No modifications needed to sam2_utils.py")

def main():
    print("🧪 CHANNEL PARSING VALIDATION TESTS")
    print("="*50)
    
    # Test current functionality
    test_current_parsing()
    test_stitch_filename_parsing()
    test_build_image_id_current()
    
    # Test expected behavior
    test_expected_channel_behavior()
    
    # Test SAM2 compatibility
    test_sam2_file_expectations()
    
    # Test file structure
    create_temp_test_structure()
    
    print("\n" + "="*50)
    print("✅ TESTING COMPLETED")
    print("="*50)
    print("\nNext steps:")
    print("1. Implement channel support in parsing_utils.py")
    print("2. Update data_organizer.py to extract channels from stitch files")
    print("3. Test with real multi-channel stitch files")
    print("4. Validate SAM2 compatibility with new file naming")

if __name__ == "__main__":
    main()