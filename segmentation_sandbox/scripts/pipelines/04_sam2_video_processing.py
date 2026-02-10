#!/usr/bin/env python3
"""
SAM2 Video Processing Pipeline Script
=====================================

Process GroundedDINO high-quality annotations with SAM2 video segmentation.
Uses the refactored SAM2 utilities with modular integration.

Usage:
    python 04_sam2_video_processing.py \
        --config configs/pipeline_config.yaml \
        --metadata data/raw_data_organized/experiment_metadata.json \
        --annotations data/detections/gdino_detections.json \
        --output data/segmentation/grounded_sam_segmentations.json \
        --entities_to_process "20240411,20250612" \
        --target-prompt "individual embryo" \
        --segmentation-format rle \
        --verbose
"""

import argparse
import sys
from pathlib import Path
import yaml
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.detection_segmentation.sam2_utils import GroundedSamAnnotations, load_sam2_model
from scripts.utils.parsing_utils import group_by


def load_config(config_path: Path) -> dict:
    """Load pipeline configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_entities_list(entities_str: str) -> list:
    """Parse comma-separated entities list."""
    if not entities_str:
        return None
    return [e.strip() for e in entities_str.split(',') if e.strip()]


def main():
    parser = argparse.ArgumentParser(description="SAM2 Video Processing Pipeline")
    parser.add_argument("--config", required=True, help="Pipeline configuration YAML file")
    parser.add_argument("--metadata", required=True, help="Experiment metadata JSON file")
    parser.add_argument("--annotations", required=True, help="GroundedDINO annotations JSON file")
    parser.add_argument("--output", required=True, help="Output SAM2 annotations JSON file")
    parser.add_argument("--entities_to_process", help="Comma-separated list of experiment IDs to process")
    parser.add_argument("--target-prompt", default="individual embryo", help="Target prompt for annotations")
    parser.add_argument("--segmentation-format", choices=["rle", "polygon"], default="rle", help="Segmentation output format")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for SAM2 model")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process")
    parser.add_argument("--save-interval", type=int, default=5, help="Auto-save interval (videos)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed videos")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    config_path = Path(args.config)
    metadata_path = Path(args.metadata)
    annotations_path = Path(args.annotations)
    output_path = Path(args.output)
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
        
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        sys.exit(1)
        
    if not annotations_path.exists():
        print(f"❌ Annotations file not found: {annotations_path}")
        sys.exit(1)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse entities
    entities_to_process = parse_entities_list(args.entities_to_process)
    
    if args.verbose:
        print("🎬 SAM2 Video Processing Pipeline")
        print("=" * 50)
        print(f"📋 Config: {config_path}")
        print(f"📊 Metadata: {metadata_path}")
        print(f"🎯 Annotations: {annotations_path}")
        print(f"💾 Output: {output_path}")
        print(f"🎭 Target prompt: '{args.target_prompt}'")
        print(f"📦 Format: {args.segmentation_format}")
        print(f"🖥️  Device: {args.device}")
        if entities_to_process:
            print(f"🔬 Processing entities: {entities_to_process}")
        else:
            print(f"🔬 Processing all available entities")
        print()
    
    try:
        # Load configuration
        config = load_config(config_path)
        sam2_config = config.get("models", {}).get("sam2", {})
        
        if args.verbose:
            print("🔧 Loading SAM2 configuration...")
        
        # Get SAM2 model paths from config
        sam2_model_config = sam2_config.get("config")
        sam2_checkpoint = sam2_config.get("checkpoint")
        
        if not sam2_model_config or not sam2_checkpoint:
            print("❌ SAM2 model configuration missing from config file")
            print("   Required: models.sam2.config and models.sam2.checkpoint")
            print(f"   Found sam2_config keys: {list(sam2_config.keys()) if sam2_config else 'None'}")
            sys.exit(1)
        
        # Initialize GroundedSamAnnotations
        if args.verbose:
            print("🎬 Initializing GroundedSamAnnotations...")
            
        gsam = GroundedSamAnnotations(
            filepath=output_path,
            seed_annotations_path=annotations_path,
            experiment_metadata_path=metadata_path,
            target_prompt=args.target_prompt,
            segmentation_format=args.segmentation_format,
            device=args.device,
            verbose=args.verbose
        )
        
        # Set SAM2 model paths and load model
        if args.verbose:
            print("🔧 Loading SAM2 model...")
            
        gsam.set_sam2_model_paths(sam2_model_config, sam2_checkpoint)
        
        # Get videos to process
        video_groups = gsam.group_annotations_by_video()
        available_videos = list(video_groups.keys())
        
        if entities_to_process:
            # Filter videos by experiment IDs
            filtered_videos = []
            for video_id in available_videos:
                if any(video_id.startswith(exp_id) for exp_id in entities_to_process):
                    filtered_videos.append(video_id)
            available_videos = filtered_videos
        
        if args.max_videos:
            available_videos = available_videos[:args.max_videos]
        
        if args.verbose:
            print(f"🎯 Found {len(available_videos)} videos to process")
            if available_videos:
                print("📹 Videos:")
                for video_id in available_videos[:5]:  # Show first 5
                    video_images = video_groups.get(video_id, {})
                    print(f"   - {video_id}: {len(video_images)} images")
                if len(available_videos) > 5:
                    print(f"   ... and {len(available_videos) - 5} more videos")
            print()
        
        if not available_videos:
            print("⚠️ No videos found to process")
            if entities_to_process:
                print(f"   Check that entities {entities_to_process} have high-quality annotations")
            sys.exit(0)
        
        # Process videos
        start_time = datetime.now()
        
        processing_stats = gsam.process_missing_annotations(
            video_ids=available_videos,
            max_videos=args.max_videos,
            auto_save_interval=args.save_interval,
            overwrite=args.overwrite
        )
        
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        # Final summary
        if args.verbose:
            print("\n" + "=" * 50)
            print("✅ SAM2 Video Processing Complete!")
            print(f"⏱️  Processing time: {processing_time}")
            print(f"📊 Videos processed: {processing_stats['processed']}")
            print(f"❌ Errors: {processing_stats['errors']}")
            print(f"💾 Output saved to: {output_path}")
            
            # Show final statistics
            summary = gsam.get_summary()
            print(f"\n📈 Final Statistics:")
            print(f"   🧪 Experiments: {summary['total_experiments']}")
            print(f"   🖼️  Images: {summary['total_images']}")
            print(f"   🎭 Snips: {summary['total_snips']}")
            print(f"   📦 Format: {summary['segmentation_format']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
