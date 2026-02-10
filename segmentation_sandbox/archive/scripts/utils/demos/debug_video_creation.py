#!/usr/bin/env python3
"""
Debug script to identify video creation issues.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List

def debug_video_creation_step_by_step(video_id: str, base_frames_dir: Path, ft_frames_dir: Path, 
                                    output_dir: Path, frame_ids: List[str], fps: int = 8):
    """
    Step-by-step debugging of video creation process.
    """
    print(f"🔍 DEBUGGING VIDEO CREATION FOR: {video_id}")
    print("=" * 60)
    
    # Check frame directories and files
    print(f"📁 Base frames directory: {base_frames_dir}")
    print(f"📁 Finetuned frames directory: {ft_frames_dir}")
    print(f"📊 Expected frame count: {len(frame_ids)}")
    
    # Check if directories exist
    if not base_frames_dir.exists():
        print(f"❌ Base frames directory does not exist!")
        return
    if not ft_frames_dir.exists():
        print(f"❌ Finetuned frames directory does not exist!")
        return
    
    # Check individual frames
    base_frames = list(base_frames_dir.glob("*.jpg"))
    ft_frames = list(ft_frames_dir.glob("*.jpg"))
    print(f"📊 Base frames found: {len(base_frames)}")
    print(f"📊 Finetuned frames found: {len(ft_frames)}")
    
    # Test loading a few frames
    test_frame_ids = frame_ids[:3]  # Test first 3 frames
    for i, frame_id in enumerate(test_frame_ids):
        print(f"\n🖼️ Testing frame {i+1}: {frame_id}")
        
        base_path = base_frames_dir / f"{frame_id}.jpg"
        ft_path = ft_frames_dir / f"{frame_id}.jpg"
        
        # Check if files exist
        print(f"   Base exists: {base_path.exists()}")
        print(f"   FT exists: {ft_path.exists()}")
        
        if base_path.exists() and ft_path.exists():
            # Load and inspect frames
            base_img = cv2.imread(str(base_path))
            ft_img = cv2.imread(str(ft_path))
            
            if base_img is not None and ft_img is not None:
                print(f"   Base shape: {base_img.shape}")
                print(f"   FT shape: {ft_img.shape}")
                print(f"   Base dtype: {base_img.dtype}")
                print(f"   FT dtype: {ft_img.dtype}")
                
                # Check for unusual values
                print(f"   Base min/max: {base_img.min()}/{base_img.max()}")
                print(f"   FT min/max: {ft_img.min()}/{ft_img.max()}")
                
                # Check color channels
                print(f"   Base BGR mean: {base_img.mean(axis=(0,1))}")
                print(f"   FT BGR mean: {ft_img.mean(axis=(0,1))}")
            else:
                print(f"   ❌ Failed to load frames")
    
    print(f"\n🎬 Testing video creation with simpler approach...")
    
    # Test creating video with our own simple implementation
    test_video_path = output_dir / f"{video_id}_debug_test.mp4"
    create_debug_video(base_frames_dir, ft_frames_dir, test_video_path, frame_ids[:10], fps)

def create_debug_video(base_frames_dir: Path, ft_frames_dir: Path, 
                      output_path: Path, frame_ids: List[str], fps: int = 8):
    """
    Simple, clean video creation for debugging.
    """
    print(f"🎥 Creating debug video: {output_path}")
    
    # Load first frame to get dimensions
    first_frame_id = frame_ids[0]
    base_path = base_frames_dir / f"{first_frame_id}.jpg"
    ft_path = ft_frames_dir / f"{first_frame_id}.jpg"
    
    base_img = cv2.imread(str(base_path))
    ft_img = cv2.imread(str(ft_path))
    
    if base_img is None or ft_img is None:
        print(f"❌ Cannot load first frames for dimension calculation")
        return False
    
    height, width = base_img.shape[:2]
    
    # Create side-by-side canvas
    canvas_width = width * 2
    canvas_height = height
    
    print(f"📐 Frame dimensions: {width}x{height}")
    print(f"📐 Video dimensions: {canvas_width}x{canvas_height}")
    
    # Setup video writer with different codec options
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try multiple codec options
    codec_options = [
        ('mp4v', 'MP4V'),  # More compatible
        ('XVID', 'XVID'),  # Alternative
        ('avc1', 'H264'),  # H264
    ]
    
    for codec_fourcc, codec_name in codec_options:
        print(f"\n🔧 Trying codec: {codec_name}")
        test_path = output_path.with_name(f"{output_path.stem}_{codec_name.lower()}.mp4")
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
            video_writer = cv2.VideoWriter(str(test_path), fourcc, fps, (canvas_width, canvas_height))
            
            if not video_writer.isOpened():
                print(f"   ❌ Failed to open video writer for {codec_name}")
                continue
            
            frames_written = 0
            for frame_id in frame_ids:
                # Load frames
                base_path = base_frames_dir / f"{frame_id}.jpg"
                ft_path = ft_frames_dir / f"{frame_id}.jpg"
                
                base_img = cv2.imread(str(base_path))
                ft_img = cv2.imread(str(ft_path))
                
                if base_img is None or ft_img is None:
                    continue
                
                # Ensure frames are correct size
                base_img = cv2.resize(base_img, (width, height))
                ft_img = cv2.resize(ft_img, (width, height))
                
                # Create side-by-side frame
                side_by_side = np.hstack([base_img, ft_img])
                
                # Add separator line
                cv2.line(side_by_side, (width, 0), (width, height), (255, 255, 255), 2)
                
                # Add labels
                cv2.putText(side_by_side, "Base Model", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(side_by_side, "Finetuned Model", (width + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Write frame
                video_writer.write(side_by_side)
                frames_written += 1
            
            video_writer.release()
            
            if frames_written > 0:
                print(f"   ✅ Successfully created {codec_name} video with {frames_written} frames")
                print(f"   📁 Saved: {test_path}")
                
                # Test if video can be read back
                test_cap = cv2.VideoCapture(str(test_path))
                if test_cap.isOpened():
                    ret, test_frame = test_cap.read()
                    if ret:
                        print(f"   ✅ Video verification: Can read back frames")
                        print(f"   📐 Read frame shape: {test_frame.shape}")
                        print(f"   🎨 Read frame color stats: min={test_frame.min()}, max={test_frame.max()}")
                    else:
                        print(f"   ❌ Video verification: Cannot read frames")
                    test_cap.release()
                else:
                    print(f"   ❌ Video verification: Cannot open for reading")
            else:
                print(f"   ❌ No frames written for {codec_name}")
                
        except Exception as e:
            print(f"   ❌ Error with {codec_name}: {e}")
    
    return True

def replace_create_side_by_side_video():
    """
    Replacement function for the problematic create_side_by_side_video.
    """
    return """
def create_side_by_side_video_fixed(video_id: str, base_dir: Path, ft_dir: Path, 
                                   output_path: str, frame_ids: List[str], fps: int = 8):
    '''
    Fixed side-by-side video creation function.
    '''
    output_path = Path(output_path)
    print(f"🎬 Creating side-by-side video: {output_path}")
    
    if not frame_ids:
        print(f"❌ No frame IDs provided")
        return False
    
    # Load first frame to get dimensions
    first_frame_id = frame_ids[0]
    base_path = base_dir / f"{first_frame_id}.jpg"
    ft_path = ft_dir / f"{first_frame_id}.jpg"
    
    base_img = cv2.imread(str(base_path))
    ft_img = cv2.imread(str(ft_path))
    
    if base_img is None or ft_img is None:
        print(f"❌ Cannot load reference frames")
        return False
    
    height, width = base_img.shape[:2]
    canvas_width = width * 2
    canvas_height = height
    
    # Use MP4V codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_width, canvas_height))
    
    if not video_writer.isOpened():
        print(f"❌ Failed to open video writer")
        return False
    
    frames_written = 0
    for frame_id in sorted(frame_ids):
        base_path = base_dir / f"{frame_id}.jpg"
        ft_path = ft_dir / f"{frame_id}.jpg"
        
        if not base_path.exists() or not ft_path.exists():
            continue
            
        base_img = cv2.imread(str(base_path))
        ft_img = cv2.imread(str(ft_path))
        
        if base_img is None or ft_img is None:
            continue
        
        # Ensure consistent dimensions
        base_img = cv2.resize(base_img, (width, height))
        ft_img = cv2.resize(ft_img, (width, height))
        
        # Create side-by-side
        side_by_side = np.hstack([base_img, ft_img])
        
        # Add separator and labels
        cv2.line(side_by_side, (width, 0), (width, height), (255, 255, 255), 3)
        
        # Write frame
        video_writer.write(side_by_side)
        frames_written += 1
    
    video_writer.release()
    
    print(f"✅ Created video with {frames_written} frames")
    return frames_written > 0
    """

# Quick test function you can run
def quick_test(video_id: str = "20240418_H02"):
    """Quick test of video creation."""
    from pathlib import Path
    
    base_dir = Path(f"/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/visualization_output/20250716/{video_id}")
    
    debug_video_creation_step_by_step(
        video_id=video_id,
        base_frames_dir=base_dir / "base_frames",
        ft_frames_dir=base_dir / "ft_frames", 
        output_dir=base_dir.parent,
        frame_ids=[f"{video_id}_{i:04d}" for i in range(5)],  # Test first 5 frames
        fps=8
    )

if __name__ == "__main__":
    quick_test()