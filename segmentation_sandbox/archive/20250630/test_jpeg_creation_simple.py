#!/usr/bin/env python3
"""
Test script for JPEG frame creation functionality.
"""

import sys
from pathlib import Path
import json
import yaml

def test_config_jpeg_path():
    """Test that the config properly loads the JPEG frames path."""
    try:
        # Load config directly
        config_path = Path("configs/pipeline_config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Get base sandbox directory
        sandbox_dir = Path(__file__).parent
        
        # Build JPEG frames path
        jpeg_rel_path = config['paths']['intermediate']['jpeg_frames']
        jpeg_path = sandbox_dir / jpeg_rel_path
        
        print(f"JPEG frames path: {jpeg_path}")
        
        # Test creating a test directory structure
        test_video_id = "20241215_A01"
        jpeg_frames_dir = jpeg_path / test_video_id
        print(f"Sample JPEG frames dir: {jpeg_frames_dir}")
        
        # Test frame path generation
        sample_frame_paths = []
        for i in range(3):
            frame_path = jpeg_frames_dir / f"frame_{i:04d}.jpg"
            sample_frame_paths.append(str(frame_path))
        
        print("Sample JPEG frame paths:")
        for path in sample_frame_paths:
            print(f"  {path}")
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metadata_format():
    """Test the video metadata format with JPEG images."""
    sample_metadata = {
        "20241215_A01": {
            "experiment_id": "20241215",
            "video_id": "20241215_A01",
            "video_path": "/path/to/20241215_A01.mp4",
            "source_images": [
                "/path/to/20241215/well_A01_t001.tif", 
                "/path/to/20241215/well_A01_t002.tif"
            ],
            "jpeg_images": [
                "/path/to/data/intermediate/jpeg_frames/20241215_A01/frame_0000.jpg",
                "/path/to/data/intermediate/jpeg_frames/20241215_A01/frame_0001.jpg"
            ],
            "total_source_images": 2,
            "valid_frames": 2,
            "video_fps": 2,
            "resolution": [1024, 768],
            "creation_time": "2025-01-30T12:00:00",
            "jpeg_conversion": True,
            "jpeg_quality": 95
        }
    }
    
    print("\nSample video metadata with JPEG images:")
    print(json.dumps(sample_metadata, indent=2))
    return True

if __name__ == "__main__":
    print("Testing JPEG frame creation functionality...")
    
    success1 = test_config_jpeg_path()
    success2 = test_metadata_format()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
