#!/usr/bin/env python3
"""
Generate evaluation videos for top-ranked embryos by percentile analysis.
"""

import subprocess
import sys
from pathlib import Path

def run_video_generation(experiment_id, video_id, embryo_id, percentile_type, rank, percentile_value):
    """Generate a single evaluation video."""
    
    output_dir = Path("results/percentile_threshold_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_name = f"{percentile_type}_{experiment_id}_{embryo_id}_rank{rank}_p{percentile_value:.3f}.mp4"
    output_path = output_dir / output_name
    
    # Command to run the video generation
    cmd = [
        sys.executable, 
        "tests/create_sam2_eval_video.py",
        "--experiment", experiment_id,
        "--video", video_id,
        "--output", str(output_path)
    ]
    
    print(f"🎬 Generating: {output_name}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ Success: {output_name}")
            return True
        else:
            print(f"❌ Failed: {output_name}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {output_name}")
        return False
    except Exception as e:
        print(f"💥 Exception: {output_name} - {e}")
        return False

def main():
    """Generate all videos for ranked embryos."""
    
    print("🎯 Generating evaluation videos for top-ranked embryos...")
    
    # Video generation tasks (from ranking analysis)
    video_tasks = [
        ("20240418", "20240418_B11", "20240418_B11_e02", "80th_90th", 1, 0.1918365737363945),
        ("20240418", "20240418_C04", "20240418_C04_e01", "80th_90th", 2, 0.15344567263427872),
        ("20250305", "20250305_A03", "20250305_A03_e01", "80th_90th", 1, 0.07806048343555608),
        ("20250305", "20250305_E04", "20250305_E04_e01", "80th_90th", 2, 0.06773517902793538),
        ("20240418", "20240418_F11", "20240418_F11_e01", "95th", 1, 0.31360733619093195),
        ("20240418", "20240418_C04", "20240418_C04_e01", "95th", 2, 0.23905496188024908),
        ("20250305", "20250305_A03", "20250305_A03_e01", "95th", 1, 0.14357822443456558),
        ("20250305", "20250305_C03", "20250305_C03_e01", "95th", 2, 0.11843695560340078),
    ]
    
    successful = 0
    total = len(video_tasks)
    
    for experiment_id, video_id, embryo_id, percentile_type, rank, percentile_value in video_tasks:
        success = run_video_generation(experiment_id, video_id, embryo_id, percentile_type, rank, percentile_value)
        if success:
            successful += 1
    
    print(f"\n📊 Summary: {successful}/{total} videos generated successfully")
    
    if successful > 0:
        print(f"📁 Videos saved to: results/percentile_threshold_videos/")
        print(f"🔍 Use vlc or other video player to review the results")

if __name__ == "__main__":
    main()
