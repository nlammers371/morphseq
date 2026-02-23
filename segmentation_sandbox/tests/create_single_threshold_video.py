#!/usr/bin/env python3
"""
Create Single Threshold Video
Generate the missing 20250305 32% threshold video
"""

import sys
from pathlib import Path

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.video_generation.video_generator import VideoGenerator

def main():
    # Missing video details (from previous analysis)
    gsam_path = "data/segmentation/grounded_sam_segmentations.json"
    experiment_id = "20250305"
    video_id = "20250305_E03"
    embryo_id = "20250305_E03_e01"
    cv_pct = 32.9
    output_dir = Path("results/threshold_validation_videos")
    
    output_video = output_dir / f"20250305_threshold_32.0pct_cv_{cv_pct:.1f}pct_{embryo_id}.mp4"
    
    print(f"🎯 Creating missing 20250305 32% threshold video")
    print(f"   Experiment: {experiment_id}")
    print(f"   Video: {video_id}")
    print(f"   Embryo: {embryo_id}")
    print(f"   CV: {cv_pct:.1f}%")
    print(f"   Output: {output_video}")
    
    # Create video generator
    vg = VideoGenerator()
    
    # Generate video
    success = vg.create_sam2_eval_video_from_results(
        results_json_path=gsam_path,
        experiment_id=experiment_id,
        video_id=video_id,
        output_video_path=output_video,
        show_bbox=True,
        show_mask=True,
        show_metrics=True,
        verbose=False
    )
    
    if success:
        print(f"✅ Success: {output_video}")
        
        # Create summary file
        import json
        summary_file = output_video.with_suffix('.json')
        summary_data = {
            "threshold_level": "32%",
            "experiment_id": experiment_id,
            "embryo_id": embryo_id,
            "actual_cv": f"{cv_pct:.1f}%",
            "video_id": video_id,
            "threshold_interpretation": {
                "would_be_flagged_27pct": True,
                "would_be_flagged_32pct": True,
                "flagging_level": "high_priority"
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        return 0
    else:
        print(f"❌ Failed to create video")
        return 1

if __name__ == "__main__":
    sys.exit(main())