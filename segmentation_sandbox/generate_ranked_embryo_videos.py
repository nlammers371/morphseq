#!/usr/bin/env python3
"""
Generate evaluation videos for top-ranked embryos by percentile analysis.
"""

import json
import sys
from pathlib import Path

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.video_generation.video_generator import VideoGenerator

def load_rankings():
    """Load the embryo rankings from the analysis."""
    with open('percentile_threshold_analysis/embryo_rankings.json', 'r') as f:
        return json.load(f)

def generate_video_for_embryo(embryo_info, percentile_type, rank, vg):
    """Generate a single evaluation video for an embryo."""
    
    experiment_id = embryo_info['video_id'].split('_')[0] + '_' + embryo_info['video_id'].split('_')[1]  # Extract experiment from video_id
    video_id = embryo_info['video_id']
    embryo_id = embryo_info['embryo_id']
    
    output_dir = Path("results/percentile_threshold_videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create descriptive filename
    percentile_val = embryo_info.get('local_var_p95' if percentile_type == '95th' else 'local_var_p90', 0)
    output_name = f"{percentile_type}_{experiment_id}_{embryo_id}_rank{rank}_p{percentile_val:.3f}.mp4"
    output_path = output_dir / output_name
    
    print(f"🎬 Generating: {output_name}")
    print(f"   Experiment: {experiment_id}")
    print(f"   Video: {video_id}")
    print(f"   Embryo: {embryo_id}")
    print(f"   CV: {embryo_info['cv']:.3f}, Percentile: {percentile_val:.3f}")
    
    try:
        # Generate evaluation video
        success = vg.create_sam2_eval_video_from_results(
            results_json_path=Path("data/segmentation/grounded_sam_segmentations.json"),
            experiment_id=experiment_id,
            video_id=video_id,
            output_video_path=output_path,
            show_bbox=True,
            show_mask=True,
            show_metrics=True,
            verbose=False
        )
        
        if success:
            print(f"✅ Success: {output_name}")
            return True
        else:
            print(f"❌ Failed: {output_name}")
            return False
            
    except Exception as e:
        print(f"💥 Exception: {output_name} - {e}")
        return False

def main():
    """Generate all videos for ranked embryos."""
    
    print("🎯 Generating evaluation videos for top-ranked embryos...")
    print("="*60)
    
    # Load rankings
    rankings = load_rankings()
    
    # Create video generator
    vg = VideoGenerator()
    
    successful = 0
    total = 0
    
    for percentile_type, exp_data in rankings.items():
        print(f"\n📊 {percentile_type} percentile embryos:")
        
        for exp_id, embryos in exp_data.items():
            print(f"\n🧪 {exp_id}:")
            
            for rank, embryo_info in enumerate(embryos, 1):
                total += 1
                success = generate_video_for_embryo(embryo_info, percentile_type, rank, vg)
                if success:
                    successful += 1
                print()  # Add spacing
    
    print(f"\n📊 Summary: {successful}/{total} videos generated successfully")
    
    if successful > 0:
        print(f"\n📁 Videos saved to: results/percentile_threshold_videos/")
        print(f"🔍 Use vlc or other video player to review the results")
        
        # Create summary report
        create_video_summary_report(rankings, successful, total)
    
    return 0 if successful == total else 1

def create_video_summary_report(rankings, successful, total):
    """Create a summary report of generated videos."""
    
    report_content = f"""# Percentile Threshold Video Analysis Report

## Summary
- **Total videos generated**: {successful}/{total}
- **Success rate**: {100*successful/total:.1f}%
- **Output directory**: results/percentile_threshold_videos/

## Ranking Analysis Results

### Purpose
Visual assessment of embryos ranked by different percentile thresholds to determine optimal thresholds for segmentation quality control.

### Embryo Rankings by Percentile

"""
    
    for percentile_type, exp_data in rankings.items():
        report_content += f"\n#### {percentile_type} Percentile\n\n"
        
        for exp_id, embryos in exp_data.items():
            report_content += f"**{exp_id}**:\n"
            
            for rank, embryo_info in enumerate(embryos, 1):
                percentile_val = embryo_info.get('local_var_p95' if percentile_type == '95th' else 'local_var_p90', 0)
                video_name = f"{percentile_type}_{exp_id}_{embryo_info['embryo_id']}_rank{rank}_p{percentile_val:.3f}.mp4"
                
                report_content += f"- **Rank {rank}**: {embryo_info['embryo_id']}\n"
                report_content += f"  - CV: {embryo_info['cv']:.3f}\n"
                report_content += f"  - {percentile_type} percentile: {percentile_val:.3f}\n"
                report_content += f"  - Frames: {embryo_info['frame_count']}\n"
                report_content += f"  - Video: `{video_name}`\n\n"
    
    report_content += f"""
## Analysis Instructions

1. **Review videos** in order of ranking within each percentile category
2. **Compare** how different percentile thresholds capture different types of variation
3. **Look for**:
   - Genuine segmentation instability vs. biological patterns
   - Death/plateau effects in longer videos (20250305)
   - Frame-to-frame consistency issues
4. **Determine** which percentile threshold best captures quality issues while avoiding false positives

## Expected Findings

- **80th/90th percentile**: May catch more subtle issues but risk false positives
- **95th percentile**: Should capture clear segmentation problems while being robust to biological trends
- **20240418 vs 20250305**: Lower bound vs upper bound conditions should show different patterns

## Next Steps

Based on visual assessment, select optimal percentile threshold for production QC implementation.
"""
    
    # Save report
    report_path = Path("results/percentile_threshold_videos/VIDEO_ANALYSIS_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"📄 Analysis report saved: {report_path}")

if __name__ == "__main__":
    sys.exit(main())