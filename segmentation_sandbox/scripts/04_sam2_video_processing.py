#!/usr/bin/env python3
"""
Stage 4: SAM2 Video Processing for Embryo Segmentation
=====================================================

This script implements Block 4 of the embryo segmentation pipeline:
- Uses high-quality GroundedDINO annotations to select seed frames
- Propagates embryo masks across video frames using SAM2
- Assigns unique embryo IDs and tracks them across frames
- Handles bidirectional propagation when seed frame is not the first frame

The heavy lifting is done by the GroundedSamAnnotations class in sam2_utils.py.
This script provides a command-line interface for the SAM2 video processing pipeline.

Features:
- High-quality annotation processing
- Automatic seed frame selection
- SAM2 video predictor integration
- Bidirectional mask propagation
- Comprehensive metadata tracking
- Structured output generation

Usage:
    python scripts/04_sam2_video_processing.py \
      --config /path/to/pipeline_config.yaml \
      --metadata /path/to/experiment_metadata.json \
      --annotations /path/to/gdino_high_quality_annotations.json \
      --output /path/to/grounded_sam_annotations.json
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
SANDBOX_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SANDBOX_ROOT / "scripts/utils"))
sys.path.append(str(SANDBOX_ROOT))

# Import utilities
print(sys.path)
from scripts.utils.sam2_utils import load_config
from scripts.utils.sam2_utils import GroundedSamAnnotations


def main():
    parser = argparse.ArgumentParser(description="SAM2 video processing for embryo segmentation")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--annotations", required=True, help="Path to gdino_high_quality_annotations.json (seed annotations) for prompting sam2")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json file")
    parser.add_argument("--output", required=True, help="Path to existing sam annotations json OR output grounded_sam_annotations.json")
    
    # SAM2 model configuration
    parser.add_argument("--sam2-config", help="Path to SAM2 config file (overrides config file)")
    parser.add_argument("--sam2-checkpoint", help="Path to SAM2 checkpoint (overrides config file)")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    
    # Processing configuration
    parser.add_argument("--target-prompt", default="individual embryo", 
                       help="Target prompt for embryo detection (default: 'individual embryo')")
    parser.add_argument("--segmentation-format", default="rle", choices=["rle", "polygon"],
                       help="Format for storing segmentation masks (rle is much more compact)")
    
    # Processing limits (for testing)
    parser.add_argument("--max-videos", type=int, default=None,
                       help="Maximum number of videos to process (for testing)")
    parser.add_argument("--video-ids", nargs="+", default=None,
                       help="Specific video IDs to process")
    
    # Output options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save-interval", type=int, default=5,
                       help="Save results every N videos (default: 5)")
    
    args = parser.parse_args()
    
    print("🎬 SAM2 Video Processing for Embryo Segmentation")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Metadata: {args.metadata}")
    print(f"Seed annotations: {args.annotations}")
    print(f"Output: {args.output}")
    print(f"Target prompt: '{args.target_prompt}'")
    print(f"Device: {args.device}")
    print(f"Segmentation format: {args.segmentation_format}")
    
    if args.segmentation_format == 'rle':
        print("📦 RLE format provides much better compression than polygons")
    
    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print(f"⚠️  Requested device '{args.device}' not available, using '{device}'")
    
    # Load configuration
    print("\n📁 Loading configuration...")
    config = load_config(args.config)
    
    # Get SAM2 model paths from config or arguments
    sam2_config_path = args.sam2_config
    sam2_checkpoint_path = args.sam2_checkpoint
    
    if not sam2_config_path:
        # Use 'config' key from pipeline_config.yaml for SAM2
        sam2_config_path = config.get('models', {}).get('sam2', {}).get('config')
        if not sam2_config_path:
            raise ValueError("SAM2 config path not found in config file or arguments")
    
    if not sam2_checkpoint_path:
        # Use 'checkpoint' key from pipeline_config.yaml for SAM2
        sam2_checkpoint_path = config.get('models', {}).get('sam2', {}).get('checkpoint')
        if not sam2_checkpoint_path:
            raise ValueError("SAM2 checkpoint path not found in config file or arguments")
    
    print(f"🔧 SAM2 Configuration:")
    print(f"   Config: {sam2_config_path}")
    print(f"   Checkpoint: {sam2_checkpoint_path}")
    
    # Initialize GroundedSamAnnotations
    print("\n🚀 Initializing GroundedSamAnnotations...")
    
    grounded_sam = GroundedSamAnnotations(
        filepath=args.output,
        seed_annotations_path=args.annotations,
        experiment_metadata_path=args.metadata,  # FIXED: Added required metadata path
        sam2_config=sam2_config_path,
        sam2_checkpoint=sam2_checkpoint_path,
        device=device,
        target_prompt=args.target_prompt,
        segmentation_format=args.segmentation_format,
        verbose=args.verbose or True  # Always show some output
    )
    
    # Check processing status
    print("\n📊 Checking processing status...")
    
    # Get experiment_ids from video_ids if specified
    experiment_ids = None
    if args.video_ids:
        experiment_ids = list(set(vid.split('_')[0] for vid in args.video_ids))
        print(f"   Derived experiment IDs from video IDs: {experiment_ids}")
    
    # Get missing videos
    missing_videos = grounded_sam.get_missing_videos(
        video_ids=args.video_ids,
        experiment_ids=experiment_ids
    )
    
    # Debugging: Print contents of the annotation file
    print("\n🔍 Debugging: Loading annotation file...")
    try:
        with open(args.annotations, 'r') as f:
            import json
            annotations_data = json.load(f)
            print(f"Annotation file structure keys: {list(annotations_data.keys())}")
            
            # Check for high quality annotations
            hq_annotations = annotations_data.get("high_quality_annotations", {})
            if hq_annotations:
                print(f"High quality annotations found for {len(hq_annotations)} experiments")
                for exp_id, exp_data in list(hq_annotations.items())[:3]:  # Show first 3
                    prompt = exp_data.get("prompt", "unknown")
                    filtered_count = len(exp_data.get("filtered", {}))
                    print(f"  {exp_id}: prompt='{prompt}', filtered={filtered_count} images")
            else:
                print("No high_quality_annotations section found")
                
    except Exception as e:
        print(f"Error reading annotation file: {e}")
        return

    # Debugging: Print output of get_missing_videos
    print("\n🔍 Debugging: Output of get_missing_videos...")
    print(f"Missing videos: {missing_videos}")
    
    if args.max_videos:
        missing_videos = missing_videos[:args.max_videos]
        print(f"   Limited to {args.max_videos} videos for testing")
    
    if len(missing_videos) == 0:
        print("✅ No missing videos to process!")
        print("🎉 All videos already processed or no annotations available.")
        return
    
    print(f"📊 Will process {len(missing_videos)} missing videos")
    
    # Process missing videos using the automated method
    print(f"\n🔄 Starting automated video processing...")
    
    results = grounded_sam.process_missing_annotations(
        video_ids=args.video_ids,
        experiment_ids=experiment_ids,
        max_videos=args.max_videos,
        auto_save_interval=args.save_interval,
        overwrite=False
    )
    
    # Print final summary
    print(f"\n🎯 Processing Complete!")
    print(f"=" * 30)
    
    summary = grounded_sam.get_summary()
    print(f"Videos processed: {summary.get('videos_processed', 0)}")
    print(f"Videos failed: {summary.get('videos_failed', 0)}")
    
    total_attempted = len(missing_videos)
    success_rate = (summary.get('videos_processed', 0) / total_attempted * 100) if total_attempted > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    print(f"Frames processed: {summary.get('total_frames_processed', 0)}")
    print(f"Embryos tracked: {summary.get('total_embryos_tracked', 0)}")
    
    # Check for bidirectional propagation usage
    videos_with_non_first_seed = 0
    for exp_data in grounded_sam.results.get("experiments", {}).values():
        for video_data in exp_data.get("videos", {}).values():
            if video_data.get("requires_bidirectional_propagation", False):
                videos_with_non_first_seed += 1
    
    if videos_with_non_first_seed > 0:
        print(f"Videos with non-first seed: {videos_with_non_first_seed}")
    
    print(f"\n📁 Results saved to: {args.output}")
    
    # Print final structure summary
    grounded_sam.print_summary()
    
    print("\n🎉 Ready for downstream processing!")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/04_sam2_video_processing.py \
#   --config /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/configs/pipeline_config.yaml \
#   --metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json \
#   --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations_finetuned.json \
#   --output /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json \
#   --target-prompt "individual embryo" \
#   --segmentation-format rle \
#   --verbose

# For testing with limited videos:
# python scripts/04_sam2_video_processing.py \
#   --config /path/to/config.yaml \
#   --metadata /path/to/experiment_metadata.json \
#   --annotations /path/to/annotations.json \
#   --output /path/to/output.json \
#   --max-videos 5 \
#   --verbose

# For specific videos:
# python scripts/04_sam2_video_processing.py \
#   --config /path/to/config.yaml \
#   --metadata /path/to/experiment_metadata.json \
#   --annotations /path/to/annotations.json \
#   --output /path/to/output.json \
#   --video-ids 20240411_A01 20240411_A02 \
#   --verbose

# Process missing videos only (recommended):
# python scripts/04_sam2_video_processing.py \
#   --config /path/to/config.yaml \
#   --metadata /path/to/experiment_metadata.json \
#   --annotations /path/to/high_quality_annotations.json \
#   --output /path/to/output.json \
#   --save-interval 3 \
#   --verbose