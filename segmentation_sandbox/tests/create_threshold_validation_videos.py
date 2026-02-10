#!/usr/bin/env python3
"""
Create Threshold Validation Videos

Generate videos from embryos at specific CV threshold levels (27% and 32%)
from each experiment to provide visual validation of threshold choices.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.video_generation.video_generator import VideoGenerator

def find_threshold_embryos(gsam_path: str, target_experiments: List[str], target_thresholds: List[float]):
    """Find embryos at specific CV threshold levels."""
    print(f"📂 Loading GSAM data from {gsam_path}")
    
    with open(gsam_path, 'r') as f:
        gsam_data = json.load(f)
    
    experiments = gsam_data.get("experiments", {})
    experiments = {k: v for k, v in experiments.items() if k in target_experiments}
    
    embryo_data = []
    
    for exp_id, exp_data in experiments.items():
        print(f"   Processing {exp_id}...")
        
        for video_id, video_data in exp_data.get("videos", {}).items():
            # Group embryos by ID across frames
            embryo_frames = defaultdict(list)
            
            for image_id, image_data in video_data.get("image_ids", {}).items():
                for embryo_id, embryo_data_item in image_data.get("embryos", {}).items():
                    segmentation = embryo_data_item.get("segmentation", {})
                    area = segmentation.get("area")
                    
                    if area is not None and area > 0:
                        embryo_frames[embryo_id].append(area)
            
            # Calculate CV for embryos with multiple frames
            for embryo_id, areas in embryo_frames.items():
                if len(areas) >= 3:
                    import numpy as np
                    mean_area = np.mean(areas)
                    std_area = np.std(areas)
                    cv = std_area / mean_area if mean_area > 0 else 0
                    
                    embryo_data.append({
                        "experiment_id": exp_id,
                        "video_id": video_id,
                        "embryo_id": embryo_id,
                        "cv": cv,
                        "cv_percentage": cv * 100,
                        "mean_area": mean_area,
                        "std_area": std_area,
                        "frame_count": len(areas),
                        "min_area": min(areas),
                        "max_area": max(areas)
                    })
    
    # Find embryos near target thresholds
    threshold_embryos = {}
    
    for threshold in target_thresholds:
        threshold_embryos[threshold] = {}
        
        for exp_id in target_experiments:
            exp_embryos = [e for e in embryo_data if e["experiment_id"] == exp_id]
            
            # Find embryo closest to threshold (above it)
            above_threshold = [e for e in exp_embryos if e["cv_percentage"] >= threshold]
            
            if above_threshold:
                # Sort by proximity to threshold and pick closest
                closest = min(above_threshold, key=lambda x: abs(x["cv_percentage"] - threshold))
                threshold_embryos[threshold][exp_id] = closest
                print(f"   Found {exp_id} embryo at {threshold}% threshold: {closest['embryo_id']} (CV: {closest['cv_percentage']:.1f}%)")
            else:
                print(f"   ⚠️ No {exp_id} embryos found above {threshold}% threshold")
    
    return threshold_embryos

def create_threshold_validation_videos(gsam_path: str, threshold_embryos: Dict, output_dir: str):
    """Create videos for threshold validation."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 Creating threshold validation videos in {output_path}")
    
    # Create video generator
    vg = VideoGenerator()
    
    videos_created = []
    
    for threshold, exp_embryos in threshold_embryos.items():
        for exp_id, embryo_info in exp_embryos.items():
            experiment_id = embryo_info["experiment_id"]
            video_id = embryo_info["video_id"]
            embryo_id = embryo_info["embryo_id"]
            cv_value = embryo_info["cv"]
            cv_pct = embryo_info["cv_percentage"]
            
            # Create descriptive output filename
            output_video = output_path / f"{exp_id}_threshold_{threshold}pct_cv_{cv_pct:.1f}pct_{embryo_id}.mp4"
            
            print(f"🎯 Creating video for {exp_id} at {threshold}% threshold")
            print(f"   Embryo: {embryo_id} (actual CV: {cv_pct:.1f}%)")
            print(f"   Video: {video_id}")
            print(f"   Output: {output_video.name}")
            
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
                print(f"   ✅ Success: {output_video}")
                videos_created.append({
                    "threshold": threshold,
                    "experiment": exp_id,
                    "embryo_id": embryo_id,
                    "cv_percentage": cv_pct,
                    "video_path": str(output_video),
                    "comparison_group": f"{threshold}pct_threshold"
                })
                
                # Create summary file
                summary_file = output_video.with_suffix('.json')
                summary_data = {
                    "threshold_level": f"{threshold}%",
                    "experiment_id": exp_id,
                    "embryo_id": embryo_id,
                    "actual_cv": f"{cv_pct:.1f}%",
                    "video_id": video_id,
                    "embryo_stats": {
                        "mean_area": embryo_info["mean_area"],
                        "std_area": embryo_info["std_area"],
                        "frame_count": embryo_info["frame_count"],
                        "area_range": [embryo_info["min_area"], embryo_info["max_area"]]
                    },
                    "threshold_interpretation": {
                        "would_be_flagged_27pct": bool(cv_pct >= 27),
                        "would_be_flagged_32pct": bool(cv_pct >= 32),
                        "flagging_level": (
                            "high_priority" if cv_pct >= 32 else
                            "moderate_priority" if cv_pct >= 27 else
                            "low_priority"
                        )
                    }
                }
                
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                
            else:
                print(f"   ❌ Failed to create video")
            
            print()
    
    return videos_created

def main():
    # Configuration
    gsam_path = "data/segmentation/grounded_sam_segmentations.json"
    target_experiments = ["20240418", "20250305"]
    target_thresholds = [27.0, 32.0]  # Key threshold levels for validation
    output_dir = "results/threshold_validation_videos"
    
    print("🎯 THRESHOLD VALIDATION VIDEO GENERATOR")
    print(f"Data source: {gsam_path}")
    print(f"Target experiments: {target_experiments}")
    print(f"Target thresholds: {target_thresholds}%")
    print(f"Output directory: {output_dir}")
    print()
    
    # Find embryos at target threshold levels
    threshold_embryos = find_threshold_embryos(gsam_path, target_experiments, target_thresholds)
    
    if not any(threshold_embryos.values()):
        print("❌ No suitable embryos found at target thresholds!")
        return 1
    
    # Create videos
    videos_created = create_threshold_validation_videos(gsam_path, threshold_embryos, output_dir)
    
    if not videos_created:
        print("❌ No videos were created successfully")
        return 1
    
    # Final summary
    print("🎬 THRESHOLD VALIDATION VIDEOS CREATED")
    print("=" * 50)
    
    for threshold in target_thresholds:
        print(f"\n{threshold}% Threshold Videos:")
        threshold_videos = [v for v in videos_created if v["threshold"] == threshold]
        
        for video in threshold_videos:
            print(f"  📹 {video['experiment']}: {video['embryo_id']} (CV: {video['cv_percentage']:.1f}%)")
            print(f"      Path: {video['video_path']}")
    
    print(f"\n🎯 Video Comparison Guide:")
    print(f"📊 Use these videos to validate threshold choices:")
    print(f"   • 27% threshold videos show 'borderline' cases")
    print(f"   • 32% threshold videos show 'clear problems'")
    print(f"   • Compare experiments to see technical differences")
    
    print(f"\n🔍 View videos with: vlc <video_path>")
    print(f"📁 All videos in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())