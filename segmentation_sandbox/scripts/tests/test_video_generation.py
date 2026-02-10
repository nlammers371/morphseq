#!/usr/bin/env python3
"""
Test the new video generation utilities.

Tests both foundation video creation and enhanced video overlays.
"""

import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.video_generation import VideoGenerator, OverlayManager, VideoConfig, COLORBLIND_PALETTE
import tempfile
import numpy as np
import cv2

def create_test_images(temp_dir, count=5):
    """Create test JPEG images for video generation."""
    jpeg_paths = []
    
    for i in range(count):
        # Create a simple test image with safe color values
        base_color = min(255, 50 + i * 30)  # Ensure values stay within uint8 range
        img = np.ones((480, 640, 3), dtype=np.uint8) * base_color  # Gradient
        
        # Add some content
        cv2.circle(img, (320 + i*20, 240), 50, (255, 255, 255), -1)
        cv2.putText(img, f"Frame {i:04d}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save as JPEG
        jpeg_path = Path(temp_dir) / f"{i:04d}.jpg"
        cv2.imwrite(str(jpeg_path), img)
        jpeg_paths.append(jpeg_path)
        
    return jpeg_paths

def test_foundation_video():
    """Test foundation video creation."""
    print("🧪 Testing foundation video creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images
        jpeg_paths = create_test_images(temp_dir, count=10)
        
        # Create foundation video
        video_generator = VideoGenerator(VideoConfig.fast_generation())
        video_path = Path(temp_dir) / "test_foundation.mp4"
        
        success = video_generator.create_foundation_video(
            jpeg_paths=jpeg_paths,
            video_path=video_path,
            video_id="20240101_A01",
            verbose=True
        )
        
        if success and video_path.exists():
            print("✅ Foundation video test passed!")
            print(f"   📁 Video created: {video_path}")
            print(f"   📏 Size: {video_path.stat().st_size} bytes")
        else:
            print("❌ Foundation video test failed!")
            
        return success

def test_enhanced_video():
    """Test enhanced video with overlays."""
    print("\n🧪 Testing enhanced video with overlays...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test images and foundation video first
        jpeg_paths = create_test_images(temp_dir, count=5)
        
        video_generator = VideoGenerator(VideoConfig.fast_generation())
        foundation_path = Path(temp_dir) / "foundation.mp4"
        
        # Create foundation
        success = video_generator.create_foundation_video(
            jpeg_paths=jpeg_paths,
            video_path=foundation_path,
            video_id="20240101_A01",
            verbose=False
        )
        
        if not success:
            print("❌ Could not create foundation video for enhanced test")
            return False
            
        # Create test overlay data
        overlay_dict = {
            "20240101_A01_t0000": [
                {"bbox": [100, 100, 150, 150], "confidence": 0.95, "label": "embryo1"}
            ],
            "20240101_A01_t0001": [
                {"bbox": [120, 110, 140, 140], "confidence": 0.87, "label": "embryo1"},
                {"bbox": [300, 200, 100, 100], "confidence": 0.76, "label": "embryo2"}
            ],
            "20240101_A01_t0002": [
                {"bbox": [140, 120, 130, 130], "confidence": 0.91, "label": "embryo1"}
            ]
        }
        
        # Create enhanced video
        enhanced_path = Path(temp_dir) / "enhanced_detections.mp4"
        success = video_generator.create_enhanced_video(
            foundation_video_path=foundation_path,
            output_video_path=enhanced_path,
            overlay_dict=overlay_dict,
            overlay_type="detection",
            verbose=True
        )
        
        if success and enhanced_path.exists():
            print("✅ Enhanced video test passed!")
            print(f"   📁 Enhanced video: {enhanced_path}")
            print(f"   📏 Size: {enhanced_path.stat().st_size} bytes")
        else:
            print("❌ Enhanced video test failed!")
            
        return success

def test_overlay_manager():
    """Test overlay manager directly."""
    print("\n🧪 Testing overlay manager...")
    
    # Create test frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    overlay_manager = OverlayManager(VideoConfig())
    
    # Test detection overlay
    detections = [
        {"bbox": [100, 100, 150, 150], "confidence": 0.95, "label": "embryo1"},
        {"bbox": [300, 200, 120, 120], "confidence": 0.87, "label": "embryo2"}
    ]
    
    frame_with_detections = overlay_manager.add_overlay(frame, detections, "detection")
    
    # Test metadata overlay
    metadata = {
        "phenotype": "normal",
        "treatment": "DMSO", 
        "stage": "blastula"
    }
    
    frame_with_metadata = overlay_manager.add_overlay(frame_with_detections, metadata, "metadata")
    
    # Test QC flags
    qc_flags = ["BLUR", "LOW_CONTRAST"]
    final_frame = overlay_manager.add_overlay(frame_with_metadata, qc_flags, "qc_flags")
    
    print("✅ Overlay manager test completed!")
    print(f"   🎨 Applied detection, metadata, and QC flag overlays")
    print(f"   🌈 Colors: {list(COLORBLIND_PALETTE.keys())}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 Testing MorphSeq Video Generation Utilities")
    print("=" * 50)
    
    tests = [
        test_foundation_video,
        test_enhanced_video, 
        test_overlay_manager
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   ✅ Passed: {sum(results)}/{len(results)}")
    print(f"   ❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Video generation utility is ready.")
    else:
        print("\n⚠️  Some tests failed. Check implementation.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
