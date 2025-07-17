#!/usr/bin/env python3
"""
Test script for SAM2 video generation
=====================================

This script tests the video generation functionality with a maximum of 5 videos.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add the project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Assuming the script is 3 levels deep
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure script directory is in sys.path
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
print("hello")
# Import the video generator functions
try:
    from sam2_video_generator import load_annotations, process_videos, create_video_from_annotations
except ImportError:
    print("Error: Could not import sam2_video_generator.py")
    print("Please ensure sam2_video_generator.py is in the same directory or in Python path")
    sys.exit(1)


def test_video_generation():
    """Test video generation with 5 videos."""
    
    print("🧪 SAM2 Video Generation Test")
    print("=" * 50)
    
    # Configuration
    annotations_path = "data/annotation_and_masks/sam2_annotations/grounded_sam_annotations.json"
    output_dir = "test_output_videos"
    max_videos = 5
    fps = 10
    show_info = True
    
    print(f"📋 Test Configuration:")
    print(f"   Annotations: {annotations_path}")
    print(f"   Output dir: {output_dir}")
    print(f"   Max videos: {max_videos}")
    print(f"   FPS: {fps}")
    print(f"   Show info: {show_info}")
    
    # Check if annotations file exists
    if not Path(annotations_path).exists():
        print(f"Error: Annotations file not found at {annotations_path}")
        sys.exit(1)
    
    # Load annotations
    annotations = load_annotations(annotations_path)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process videos
    process_videos(annotations, output_dir, max_videos, fps, show_info)
    
    print("✅ Video generation test completed successfully!")

if __name__ == "__main__":
    test_video_generation()