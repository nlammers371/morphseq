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
            verbose=False,  # Reduce verbosity for batch processing
            target_embryo_id=embryo_id  # Focus on specific embryo if supported
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
    
    return success_count\n\ndef main():\n    parser = argparse.ArgumentParser(\n        description=\"Create threshold comparison videos for segmentation variability analysis\",\n        formatter_class=argparse.RawDescriptionHelpFormatter,\n        epilog=\"\"\"\nExamples:\n  # Create videos for all threshold categories\n  python create_threshold_comparison_videos.py\n  \n  # Create videos for specific categories only\n  python create_threshold_comparison_videos.py --categories normal,borderline,extreme\n  \n  # Use custom paths\n  python create_threshold_comparison_videos.py \\\n    --gsam data/segmentation/grounded_sam_segmentations.json \\\n    --output results/threshold_videos \\\n    --results threshold_analysis_results.json\n        \"\"\"\n    )\n    \n    parser.add_argument(\n        \"--gsam\",\n        default=\"data/segmentation/grounded_sam_segmentations.json\",\n        help=\"Path to GSAM segmentation JSON file\"\n    )\n    parser.add_argument(\n        \"--output\",\n        default=\"results/threshold_comparison_videos\",\n        help=\"Output directory for videos\"\n    )\n    parser.add_argument(\n        \"--results\",\n        default=\"threshold_analysis_results.json\",\n        help=\"Path to threshold analysis results file\"\n    )\n    parser.add_argument(\n        \"--categories\",\n        help=\"Comma-separated list of categories to process (default: all)\"\n    )\n    \n    args = parser.parse_args()\n    \n    # Validate input files\n    gsam_path = Path(args.gsam)\n    if not gsam_path.exists():\n        print(f\"❌ GSAM file not found: {gsam_path}\")\n        return 1\n    \n    # Load representative embryos\n    representative_embryos = load_representative_embryos(args.results)\n    if not representative_embryos:\n        return 1\n    \n    print(f\"📋 Found representative embryos for {len(representative_embryos)} categories:\")\n    for category, info in representative_embryos.items():\n        cv_pct = info[\"cv\"] * 100\n        print(f\"   {category}: CV = {cv_pct:.1f}% ({info['experiment_id']}/{info['video_id']}/{info['embryo_id']})\")\n    print()\n    \n    # Parse categories\n    categories = None\n    if args.categories:\n        categories = [cat.strip() for cat in args.categories.split(\",\")]\n        \n        # Validate categories\n        invalid_categories = [cat for cat in categories if cat not in representative_embryos]\n        if invalid_categories:\n            print(f\"❌ Invalid categories: {invalid_categories}\")\n            print(f\"Available categories: {list(representative_embryos.keys())}\")\n            return 1\n    \n    # Create videos\n    success_count = create_threshold_videos(\n        gsam_path=str(gsam_path),\n        output_dir=args.output,\n        representative_embryos=representative_embryos,\n        categories=categories\n    )\n    \n    if success_count == 0:\n        print(\"❌ No videos were created successfully\")\n        return 1\n    \n    print(\"\\n✅ Threshold comparison videos created successfully!\")\n    print(\"🔍 Review the videos to validate the optimal CV threshold for production use.\")\n    \n    return 0\n\nif __name__ == \"__main__\":\n    sys.exit(main())