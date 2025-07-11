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

# Import the video generator functions
# Note: This assumes the sam2_video_generator.py is in the same directory
# or accessible via Python path
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
        print(f"\n❌ Annotations file not found: {annotations_path}")
        print("Please ensure the file exists at the specified path")
        return False
    
    # Load annotations
    print(f"\n1️⃣ Loading annotations...")
    try:
        annotations = load_annotations(annotations_path)
        
        # Print some statistics
        print(f"\n📊 Annotation Statistics:")
        print(f"   Total experiments: {len(annotations.get('experiment_ids', []))}")
        print(f"   Total videos: {len(annotations.get('video_ids', []))}")
        print(f"   Total embryos tracked: {len(annotations.get('embryo_ids', []))}")
        
        # Show sample videos
        sample_videos = []
        for exp_data in annotations.get('experiments', {}).values():
            for video_id, video_data in exp_data.get('videos', {}).items():
                if video_data.get('sam2_success', False):
                    sample_videos.append({
                        'video_id': video_id,
                        'num_embryos': video_data.get('num_embryos', 0),
                        'frames': video_data.get('frames_processed', 0)
                    })
                    if len(sample_videos) >= 5:
                        break
            if len(sample_videos) >= 5:
                break
        
        print(f"\n📹 Sample videos to process:")
        for i, video_info in enumerate(sample_videos, 1):
            print(f"   {i}. {video_info['video_id']} - {video_info['num_embryos']} embryos, {video_info['frames']} frames")
        
    except Exception as e:
        print(f"❌ Failed to load annotations: {e}")
        return False
    
    # Create output directory
    print(f"\n2️⃣ Creating output directory...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Output directory: {output_path.absolute()}")
    
    # Process videos
    print(f"\n3️⃣ Processing videos...")
    try:
        stats = process_videos(
            annotations=annotations,
            output_dir=output_dir,
            max_videos=max_videos,
            fps=fps,
            show_info=show_info
        )
        
        # Print results
        print(f"\n✅ Test completed successfully!")
        print(f"   Videos created: {stats['videos_created']}")
        print(f"   Videos failed: {stats['videos_failed']}")
        print(f"   Processing time: {stats['duration']:.1f} seconds")
        
        # List created videos
        created_videos = list(output_path.glob("*.mp4"))
        if created_videos:
            print(f"\n📹 Created videos:")
            for video_path in created_videos[:5]:
                file_size = video_path.stat().st_size / (1024 * 1024)  # MB
                print(f"   - {video_path.name} ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    dependencies = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV (cv2)',
        'tqdm': 'tqdm',
        'matplotlib': 'Matplotlib'
    }
    
    optional_deps = {
        'pycocotools': 'pycocotools (for RLE decoding)'
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required dependencies
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - REQUIRED")
            missing_required.append(name)
    
    # Check optional dependencies
    for module, name in optional_deps.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️  {name} - Optional")
            missing_optional.append(name)
    
    if missing_required:
        print(f"\n❌ Missing required dependencies: {', '.join(missing_required)}")
        print("Install with: pip install numpy opencv-python tqdm matplotlib")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies: {', '.join(missing_optional)}")
        print("Install with: pip install pycocotools")
        print("(Video generation will still work with fallback methods)")
    
    return True


def main():
    """Main test function."""
    print("🎬 SAM2 Video Generator Test Script")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running the test")
        return 1
    
    # Run test
    success = test_video_generation()
    
    if success:
        print("\n✅ All tests passed!")
        print("\nTo run with different parameters, use:")
        print("  python sam2_video_generator.py --annotations <path> --output <dir> --max-videos 10 --fps 15")
        print("\nOr for a quick test:")
        print("  python sam2_video_generator.py --test")
        return 0
    else:
        print("\n❌ Test failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())