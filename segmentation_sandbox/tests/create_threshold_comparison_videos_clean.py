#!/usr/bin/env python3
"""
Create Threshold Comparison Videos

Generate videos for representative embryos at different CV threshold levels
to visually validate the optimal threshold for segmentation variability QC.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List

# Add scripts to path
SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.video_generation.video_generator import VideoGenerator

def load_representative_embryos(results_path: str = "threshold_analysis_results.json") -> Dict:
    """Load representative embryos from threshold analysis."""
    results_file = Path(results_path)
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("🔧 Run threshold_impact_analysis.py first to generate representative embryos")
        return {}
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results.get("representative_embryos", {})

def create_threshold_videos(gsam_path: str, output_dir: str, representative_embryos: Dict, 
                          categories: List[str] = None):
    """Create videos for representative embryos."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default to all categories if none specified
    if categories is None:
        categories = list(representative_embryos.keys())
    
    print(f"🎬 Creating threshold comparison videos")
    print(f"📁 GSAM data: {gsam_path}")
    print(f"📁 Output directory: {output_path}")
    print(f"🎯 Categories: {categories}")
    print()
    
    # Create video generator
    vg = VideoGenerator()
    
    # Generate videos for each category
    success_count = 0
    total_count = len(categories)
    
    for category in categories:
        if category not in representative_embryos:
            print(f"⚠️ No representative embryo found for category: {category}")
            continue
        
        embryo_info = representative_embryos[category]
        experiment_id = embryo_info["experiment_id"]
        video_id = embryo_info["video_id"]
        embryo_id = embryo_info["embryo_id"]
        cv_value = embryo_info["cv"]
        
        # Create category-specific output directory
        category_dir = output_path / f"{category}_cv_{cv_value*100:.1f}pct"
        category_dir.mkdir(exist_ok=True)
        
        output_video = category_dir / f"{video_id}_{embryo_id}_cv_{cv_value*100:.1f}pct.mp4"
        
        print(f"🎯 Creating {category.upper()} video")
        print(f"   Experiment: {experiment_id}")
        print(f"   Video: {video_id}")
        print(f"   Embryo: {embryo_id}")
        print(f"   CV: {cv_value:.3f} ({cv_value*100:.1f}%)")
        print(f"   Output: {output_video}")
        
        # Generate video with enhanced metadata
        success = vg.create_sam2_eval_video_from_results(
            results_json_path=gsam_path,
            experiment_id=experiment_id,
            video_id=video_id,
            output_video_path=output_video,
            show_bbox=True,
            show_mask=True,
            show_metrics=True,
            verbose=False  # Reduce verbosity for batch processing
        )
        
        if success:
            print(f"   ✅ Success: {output_video}")
            success_count += 1
            
            # Create a summary file for this video
            summary_file = category_dir / f"{video_id}_{embryo_id}_summary.json"
            summary_data = {
                "category": category,
                "cv_value": cv_value,
                "cv_percentage": f"{cv_value*100:.1f}%",
                "experiment_id": experiment_id,
                "video_id": video_id,
                "embryo_id": embryo_id,
                "mean_area": embryo_info.get("mean_area"),
                "frame_count": embryo_info.get("frame_count"),
                "area_range": [embryo_info.get("min_area"), embryo_info.get("max_area")],
                "video_path": str(output_video),
                "threshold_analysis": {
                    "current_15pct": "flagged" if cv_value > 0.15 else "not_flagged",
                    "recommended_27pct": "flagged" if cv_value > 0.27 else "not_flagged",
                    "strict_32pct": "flagged" if cv_value > 0.32 else "not_flagged"
                }
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
        else:
            print(f"   ❌ Failed to create video")
        
        print()
    
    # Create overall summary
    print(f"📊 Summary: {success_count}/{total_count} videos created successfully")
    
    if success_count > 0:
        print(f"\n🎬 Video locations:")
        for category in categories:
            if category in representative_embryos:
                cv_value = representative_embryos[category]["cv"]
                category_dir = output_path / f"{category}_cv_{cv_value*100:.1f}pct"
                if category_dir.exists():
                    videos = list(category_dir.glob("*.mp4"))
                    if videos:
                        print(f"   {category.upper()}: {videos[0]}")
        
        print(f"\n🔍 View videos with: vlc <video_path>")
        print(f"📁 All videos in: {output_path}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(
        description="Create threshold comparison videos for segmentation variability analysis"
    )
    
    parser.add_argument(
        "--gsam",
        default="data/segmentation/grounded_sam_segmentations.json",
        help="Path to GSAM segmentation JSON file"
    )
    parser.add_argument(
        "--output",
        default="results/threshold_comparison_videos",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--results",
        default="threshold_analysis_results.json",
        help="Path to threshold analysis results file"
    )
    parser.add_argument(
        "--categories",
        help="Comma-separated list of categories to process (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    gsam_path = Path(args.gsam)
    if not gsam_path.exists():
        print(f"❌ GSAM file not found: {gsam_path}")
        return 1
    
    # Load representative embryos
    representative_embryos = load_representative_embryos(args.results)
    if not representative_embryos:
        return 1
    
    print(f"📋 Found representative embryos for {len(representative_embryos)} categories:")
    for category, info in representative_embryos.items():
        cv_pct = info["cv"] * 100
        print(f"   {category}: CV = {cv_pct:.1f}% ({info['experiment_id']}/{info['video_id']}/{info['embryo_id']})")
    print()
    
    # Parse categories
    categories = None
    if args.categories:
        categories = [cat.strip() for cat in args.categories.split(",")]
        
        # Validate categories
        invalid_categories = [cat for cat in categories if cat not in representative_embryos]
        if invalid_categories:
            print(f"❌ Invalid categories: {invalid_categories}")
            print(f"Available categories: {list(representative_embryos.keys())}")
            return 1
    
    # Create videos
    success_count = create_threshold_videos(
        gsam_path=str(gsam_path),
        output_dir=args.output,
        representative_embryos=representative_embryos,
        categories=categories
    )
    
    if success_count == 0:
        print("❌ No videos were created successfully")
        return 1
    
    print("\n✅ Threshold comparison videos created successfully!")
    print("🔍 Review the videos to validate the optimal CV threshold for production use.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())