#!/usr/bin/env python3
"""
Pipeline Script 4: SAM2 Video Segmentation

Run SAM2 video segmentation using GroundedDINO detection annotations.
Processes all experiments by default.
"""

import argparse
import sys
import json
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from metadata.experiment_metadata import ExperimentMetadata

def main():
    parser = argparse.ArgumentParser(
        description="Run SAM2 video segmentation for MorphSeq pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all detections for segmentation
  python 04_sam2_segmentation.py --metadata experiment_metadata.json \\
    --annotations detections.json --output segmentations.json
  
  # Process specific experiments only
  python 04_sam2_segmentation.py --metadata experiment_metadata.json \\
    --annotations detections.json --output segmentations.json \\
    --experiments "20240506,20250703_chem3_28C_T00_1325"
  
  # Custom segmentation parameters
  python 04_sam2_segmentation.py --metadata experiment_metadata.json \\
    --annotations detections.json --output segmentations.json \\
    --propagation-frames 5 --temporal-window 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--metadata", 
        required=True, 
        help="Path to experiment_metadata.json from step 1"
    )
    parser.add_argument(
        "--annotations", 
        required=True, 
        help="Path to detection annotations JSON from step 3"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output path for segmentation annotations JSON"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config", 
        help="Pipeline config YAML file (for SAM2 model paths)"
    )
    parser.add_argument(
        "--experiments", 
        help="Comma-separated experiment IDs to process (default: all)"
    )
    parser.add_argument(
        "--propagation-frames", 
        type=int, 
        default=10, 
        help="Number of frames to propagate per sequence (default: 10)"
    )
    parser.add_argument(
        "--temporal-window", 
        type=int, 
        default=5, 
        help="Temporal window for SAM2 tracking (default: 5)"
    )
    parser.add_argument(
        "--confidence-threshold", 
        type=float, 
        default=0.3, 
        help="Minimum confidence for using detections (default: 0.3)"
    )
    parser.add_argument(
        "--max-objects-per-frame", 
        type=int, 
        default=20, 
        help="Maximum objects to track per frame (default: 20)"
    )
    parser.add_argument(
        "--save-interval", 
        type=int, 
        default=50, 
        help="Auto-save every N frames (default: 50)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be processed without running segmentation"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    metadata_path = Path(args.metadata).resolve()
    annotations_path = Path(args.annotations).resolve()
    output_path = Path(args.output).resolve()
    
    if not metadata_path.exists():
        print(f"❌ Error: Metadata file does not exist: {metadata_path}")
        sys.exit(1)
    
    if not annotations_path.exists():
        print(f"❌ Error: Annotations file does not exist: {annotations_path}")
        sys.exit(1)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("📋 Loading experiment metadata...")
    try:
        meta = ExperimentMetadata(str(metadata_path))
        print(f"✅ Loaded metadata with {len(meta.metadata['experiments'])} experiments")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        sys.exit(1)
    
    # Load detection annotations
    print("🔍 Loading detection annotations...")
    try:
        with open(annotations_path, 'r') as f:
            detection_data = json.load(f)
        
        annotations = detection_data.get('high_quality_annotations', detection_data.get('annotations', {}))
        print(f"📸 Found annotations for {len(annotations)} images")
        
        if len(annotations) == 0:
            print("⚠️  No annotations found - check detection results")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        sys.exit(1)
    
    # Parse experiment filter
    experiment_ids = None
    if args.experiments:
        experiment_ids = [e.strip() for e in args.experiments.split(",")]
        print(f"📌 Processing specific experiments: {experiment_ids}")
    else:
        print("📌 Processing ALL experiments in metadata")
    
    # Group annotations by video (SAM2 handles this internally)
    print("🎬 Analyzing annotations for video sequences...")
    total_annotations = len(annotations)
    if total_annotations == 0:
        print("⚠️  No annotations found - check detection results")
        sys.exit(1)
    
    print(f"📸 Found {total_annotations} annotated images ready for segmentation")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Segmentation plan:")
        print(f"  📸 Detection annotations: {len(annotations)}")
        print(f"  🕰️  Propagation frames: {args.propagation_frames}")
        print(f"  🎯 Confidence threshold: {args.confidence_threshold}")
        print(f"  📊 Max objects per frame: {args.max_objects_per_frame}")
        print(f"  💾 Output: {output_path}")
        
        # Show sample annotation info
        sample_images = list(annotations.keys())[:5]
        print(f"\n📸 Sample annotated images:")
        for image_id in sample_images:
            detections = len(annotations[image_id])
            print(f"  - {image_id}: {detections} detections")
        
        if len(annotations) > 5:
            print(f"  ... and {len(annotations) - 5} more")
        
        return
    
    print("🚀 Starting SAM2 video segmentation...")
    print(f"🕰️  Propagation frames: {args.propagation_frames}")
    print(f"🎯 Confidence threshold: {args.confidence_threshold}")
    print(f"📊 Max objects per frame: {args.max_objects_per_frame}")
    
    try:
        # Import and use actual SAM2 segmentation
        from detection_segmentation.sam2_utils import GroundedSamAnnotations
        
        print("🎬 Initializing SAM2 video segmentation...")
        
        # Initialize GroundedSamAnnotations with proper parameters
        gsam = GroundedSamAnnotations(
            filepath=output_path,
            seed_annotations_path=annotations_path,
            experiment_metadata_path=metadata_path,
            target_prompt="individual embryo",
            segmentation_format="rle",
            verbose=args.verbose
        )
        
        # Set SAM2 model paths if config provided
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            sam2_config = config.get('sam2', {}).get('config_path')
            sam2_checkpoint = config.get('sam2', {}).get('checkpoint_path')
            
            if sam2_config and sam2_checkpoint:
                gsam.set_sam2_model_paths(sam2_config, sam2_checkpoint)
                print(f"✅ SAM2 model paths set from config")
        
        # Get missing videos to process
        missing_videos = gsam.get_missing_videos(experiment_ids=experiment_ids)
        
        if not missing_videos:
            print("✅ All videos already processed!")
            return
        
        print(f"🎬 Processing {len(missing_videos)} missing videos...")
        
        # Process missing annotations with specified parameters
        results = gsam.process_missing_annotations(
            experiment_ids=experiment_ids,
            auto_save_interval=args.save_interval,
            overwrite=False
        )
        
        # Final save
        gsam.save()
        
        print(f"✅ SAM2 segmentation complete!")
        print(f"📄 Results saved to: {output_path}")
        print(f"📋 Processed {results.get('videos_processed', 0)} videos")
        print(f"🧬 Tracked {results.get('total_embryos_tracked', 0)} embryos")
        print(f"� Processed {results.get('total_frames_processed', 0)} frames")
        
    except ImportError as e:
        print(f"❌ Cannot import SAM2 utilities: {e}")
        print("� Make sure SAM2 utils are properly implemented in detection_segmentation/")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during segmentation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
