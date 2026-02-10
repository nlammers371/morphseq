#!/usr/bin/env python3
"""
Stage 3: GroundedDINO Detection with Quality Filtering (Modern Module 2 Implementation)
======================================================================================

This script uses the new Module 2 GroundingDINO implementation to generate annotations
with high-quality filtering using our modular pipeline utilities.

Features:
- Uses Module 2 GroundedDinoAnnotations class
- Integrates with ExperimentMetadata for efficient image discovery
- Generates high-quality annotations with confidence and IoU filtering
- Entity tracking and validation using Module 0/1 utilities
- Atomic saves with backup functionality
"""

import os
import sys
import json
import yaml
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
SANDBOX_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(SANDBOX_ROOT))

# Import Module 2 utilities
from scripts.detection_segmentation.grounded_dino_utils import (
    load_config, load_groundingdino_model, GroundedDinoAnnotations
)
from scripts.metadata.experiment_metadata import ExperimentMetadata


def main():
    parser = argparse.ArgumentParser(description="Generate GroundedDINO detections with quality filtering (Module 2)")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--annotations", required=True, help="Path to output annotations JSON")
    
    # Quality filtering arguments
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                       help="Confidence threshold for filtering (default: 0.45)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for duplicate removal (default: 0.5)")
    parser.add_argument("--skip-filtering", action="store_true",
                       help="Skip quality filtering step")
    
    # Model configuration
    parser.add_argument("--weights", default=None,
                       help="Path to model weights (overrides config)")
    parser.add_argument("--prompt", default="individual embryo",
                       help="Detection prompt (default: individual embryo)")
    
    # Processing options
    parser.add_argument("--entities_to_process", default=None,
                       help="Comma-separated list of entities to process (experiments, videos, or images)")
    parser.add_argument("--max-images", type=int, default=None,
                       help="Maximum number of images to process (for testing)")
    parser.add_argument("--auto-save-interval", type=int, default=100,
                       help="Auto-save every N images")
    
    args = parser.parse_args()

    print("🚀 Starting GroundingDINO Detection with Quality Filtering (Module 2)")
    print("=" * 70)

    # EARLY CHECK: Initialize annotations manager first to see if work is needed
    print("🔍 Quick check: Do we need to do any work?")
    annotations = GroundedDinoAnnotations(args.annotations, verbose=False, metadata_path=args.metadata)
    
    # Quick check for existing HQ annotations - but only do this if we have specific entities
    # Otherwise, we need to do the full check after loading metadata
    if args.entities_to_process and not args.skip_filtering:
        print(f"⚡ Quick check: Do HQ annotations exist for specific entities?")
        
        # Do a very quick check for the specific entities
        basic_missing = annotations.get_missing_annotations(
            prompts=[args.prompt],  # Convert string to list
            experiment_ids=args.entities_to_process.split(',') if args.entities_to_process else None,
            consider_different_if_different_weights=True,
        )
        
        if not basic_missing or not basic_missing.get(args.prompt, []):
            print("⚡ No new work needed - all annotations exist for specified entities!")
            annotations.print_summary()
            print("✅ GroundingDINO detection generation completed!")
            return
        else:
            print(f"⚡ Found {len(basic_missing.get(args.prompt, []))} images needing work")
    
    # If no specific entities provided, we'll do the full check after loading metadata

    # If we get here, we need to do work, so load everything properly
    print("🔧 Loading full pipeline components...")

    # Load config and setup device
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")

    # Override weights if specified
    if args.weights:
        config["models"]["groundingdino"]["weights"] = args.weights
        print(f"🔄 Using custom weights: {Path(args.weights).name}")

    # Load experiment metadata
    metadata_manager = ExperimentMetadata(args.metadata, verbose=True)
    print(f"📊 Loaded metadata: {metadata_manager}")

    # Get target images
    if args.entities_to_process:
        # Parse comma-separated entities and let parsing utilities figure out what they are
        entity_list = [entity.strip() for entity in args.entities_to_process.split(',')]
        print(f"🎯 Processing entities: {entity_list}")
        
        target_images = []
        experiments_found = []
        videos_found = []
        images_found = []
        
        for entity in entity_list:
            # Try to determine entity type using metadata manager
            if entity in metadata_manager.list_experiments():
                # It's an experiment
                exp_images = metadata_manager.list_images(experiment_id=entity)
                target_images.extend(exp_images)
                experiments_found.append(entity)
                print(f"   📁 Experiment {entity}: {len(exp_images)} images")
            else:
                # Check if it's a video by looking through all experiments
                video_found = False
                for exp_id in metadata_manager.list_experiments():
                    if entity in metadata_manager.list_videos(exp_id):
                        video_images = metadata_manager.list_images(exp_id, entity)
                        target_images.extend(video_images)
                        videos_found.append(entity)
                        print(f"   🎬 Video {entity}: {len(video_images)} images")
                        video_found = True
                        break
                
                if not video_found:
                    # Assume it's an image ID
                    if entity in metadata_manager.list_images():
                        target_images.append(entity)
                        images_found.append(entity)
                        print(f"   🖼️ Image {entity}")
                    else:
                        print(f"   ⚠️ Entity '{entity}' not found in metadata")
        
        print(f"📊 Found: {len(experiments_found)} experiments, {len(videos_found)} videos, {len(images_found)} images")
        # Set the parsed entities for later use
        parsed_experiment_ids = experiments_found if experiments_found else None
        parsed_video_ids = videos_found if videos_found else None
    else:
        target_images = metadata_manager.list_images()
        print(f"📊 Processing all images: {len(target_images)}")
        parsed_experiment_ids = None
        parsed_video_ids = None

    if args.max_images and len(target_images) > args.max_images:
        target_images = target_images[:args.max_images]
        print(f"🔢 Limited to first {args.max_images} images for testing")

    # =================================================================================
    # PHASE 1: GroundingDINO Detection
    # =================================================================================
    print("\n" + "="*60)
    print(f"🔍 PHASE 1: GroundingDINO Detection ('{args.prompt}')")
    print("="*60)

    # Use the annotations manager we already created in the early check
    annotations.verbose = True  # Turn on verbose output for the actual processing phase
    print(f"💾 Annotations will be saved to: {args.annotations}")

    # Check what needs to be processed BEFORE loading the model
    print("🔍 Checking for missing annotations...")
    missing_images = annotations.get_missing_annotations(
        prompts=[args.prompt],  # Convert string to list
        experiment_ids=parsed_experiment_ids,
        video_ids=parsed_video_ids,
        image_ids=target_images if not parsed_experiment_ids and not parsed_video_ids else None,
        consider_different_if_different_weights=True,
    )
    
    new_work_done = False  # Track if we actually processed anything new
    
    # Extract the actual missing image list for our prompt
    missing_image_list = missing_images.get(args.prompt, [])
    
    if not missing_image_list:
        print("✅ No missing annotations found - all images already processed!")
        print("✅ Phase 1 complete: No detection needed")
    else:
        print(f"📊 Found {len(missing_image_list)} images needing annotation")
        
        try:
            # Only load model if there's work to do
            print("🔄 Loading GroundingDINO model...")
            model = load_groundingdino_model(config, device=device)
            print("✅ Model loaded successfully")

            # Process missing annotations
            print(f"🔄 Processing annotations for prompt: '{args.prompt}'")
            results = annotations.process_missing_annotations(
                model=model,
                prompts=args.prompt,  # Keep as string - process_missing_annotations handles conversion
                experiment_ids=parsed_experiment_ids,
                video_ids=parsed_video_ids,
                image_ids=target_images if not parsed_experiment_ids and not parsed_video_ids else None,
                auto_save_interval=args.auto_save_interval,
                store_image_source=False,
                show_anno=False,
                overwrite=False,
                consider_different_if_different_weights=True,
            )
            
            print(f"✅ Processed {len(results)} images")
            annotations.save()
            print("✅ Phase 1 complete: Detection finished")
            new_work_done = len(results) > 0  # Track if we actually did work

        except Exception as e:
            print(f"❌ Error in Phase 1 (detection): {e}")
            import traceback
            traceback.print_exc()
            return

    # Skip filtering if requested
    if args.skip_filtering:
        print("\n⏭️ Skipping quality filtering as requested.")
        annotations.print_summary()
        print("✅ GroundingDINO detection generation completed!")
        return

    # Skip filtering if no new work was done and HQ annotations already exist
    if not new_work_done:
        existing_hq = annotations.annotations.get("high_quality_annotations", {})
        # Check if HQ annotations exist for the target prompt (proper structure check)
        hq_exists = False
        if existing_hq:
            for exp_data in existing_hq.values():
                if exp_data.get('prompt') == args.prompt and exp_data.get('filtered'):
                    hq_exists = True
                    break
        
        if hq_exists:
            print(f"\n⏭️ Skipping quality filtering - no new work done and HQ annotations already exist for '{args.prompt}'.")
            annotations.print_summary()
            print("✅ GroundingDINO detection generation completed!")
            return

    # =================================================================================
    # PHASE 2: High-Quality Filtering
    # =================================================================================
    print("\n" + "="*60)
    print("🎯 PHASE 2: High-Quality Filtering")
    print("="*60)
    
    try:
        # Get all image IDs that have annotations for the prompt
        annotated_images = annotations.get_annotated_image_ids(prompt=args.prompt)
        print(f"Found {len(annotated_images)} images with '{args.prompt}' annotations")
        
        if annotated_images:
            print(f"🎯 Generating high-quality annotations...")
            print(f"   Confidence threshold: {args.confidence_threshold}")
            print(f"   IoU threshold: {args.iou_threshold}")
            
            result = annotations.generate_high_quality_annotations(
                image_ids=annotated_images,
                prompt=args.prompt,
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                overwrite=True,
                save_to_self=True
            )
            
            # Print filtering results
            stats = result["statistics"]
            print(f"📊 Filtering Summary:")
            print(f"   Original detections: {stats['original_detections']}")
            print(f"   Confidence removed: {stats['confidence_removed']}")
            print(f"   IoU removed: {stats['iou_removed']}")
            print(f"   Final detections: {stats['final_detections']}")
            print(f"   Retention rate: {stats['retention_rate']:.1%}")
            print(f"   Final images: {stats['final_images']}")
            print(f"   Experiments processed: {stats['experiments_processed']}")
        else:
            print("⚠️ No annotations found for filtering")
        
        annotations.save()
        print("✅ Phase 2 complete: Quality filtering finished")
        
    except Exception as e:
        print(f"❌ Error in Phase 2 (filtering): {e}")
        import traceback
        traceback.print_exc()

    # =================================================================================
    # FINAL SUMMARY
    # =================================================================================
    print("\n" + "="*60)
    print("📊 FINAL PIPELINE SUMMARY")
    print("="*60)
    
    annotations.print_summary()
    
    print(f"\n📁 OUTPUT:")
    print(f"   Annotations file: {args.annotations}")
    print(f"   Prompt processed: '{args.prompt}'")
    print(f"   Quality filtering: {'Applied' if not args.skip_filtering else 'Skipped'}")
    
    if not args.skip_filtering:
        hq_data = annotations.annotations.get("high_quality_annotations", {})
        if hq_data:
            total_hq_images = sum(len(exp_data.get("filtered", {})) for exp_data in hq_data.values())
            total_hq_detections = sum(
                sum(len(dets) for dets in exp_data.get("filtered", {}).values()) 
                for exp_data in hq_data.values()
            )
            print(f"   High-quality results: {len(hq_data)} experiments, {total_hq_images} images, {total_hq_detections} detections")
    
    print(f"\n✅ ANNOTATION GENERATION COMPLETE!")
    print(f"🎯 Ready for downstream processing (SAM2, etc.)")


if __name__ == "__main__":
    main()


# Example usage:
# python scripts/pipelines/03_gdino_detection.py \
#   --config configs/pipeline_config.yaml \
#   --metadata data/raw_data_organized/experiment_metadata.json \
#   --annotations data/annotation_and_masks/gdino_annotations/gdino_annotations_modern.json \
#   --confidence-threshold 0.45 \
#   --iou-threshold 0.5 \
#   --experiment-ids 20250612_30hpf_ctrl_atf6 \
#   --max-images 50
