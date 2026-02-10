#!/usr/bin/env python3
"""
Create SAM2 evaluation video for quality control.
"""

from pathlib import Path
import sys

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.video_generation.video_generator import VideoGenerator

def main():
    # Paths
    results_json = Path("data/segmentation/grounded_sam_segmentations.json")
    output_dir = Path("results/sam2_eval_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target video
    experiment_id = "20250612_30hpf_ctrl_atf6"
    video_id = "20250612_30hpf_ctrl_atf6_A01"
    output_video = output_dir / f"{video_id}_sam2_eval.mp4"
    
    print(f"🎬 Creating SAM2 evaluation video")
    print(f"📁 Input: {results_json}")
    print(f"🎯 Target: {experiment_id} / {video_id}")
    print(f"💾 Output: {output_video}")
    
    # Create video generator
    vg = VideoGenerator()
    
    # Generate evaluation video
    success = vg.create_sam2_eval_video_from_results(
        results_json_path=results_json,
        experiment_id=experiment_id,
        video_id=video_id,
        output_video_path=output_video,
        show_bbox=True,
        show_mask=True,
        show_metrics=True,
        verbose=True
    )
    
    if success:
        print(f"✅ Video saved: {output_video}")
        print(f"🔍 Open with: vlc {output_video}")
    else:
        print("❌ Video generation failed")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
