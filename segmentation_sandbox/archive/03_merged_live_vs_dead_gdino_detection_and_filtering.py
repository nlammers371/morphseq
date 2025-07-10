#!/usr/bin/env python3
"""
Stage 3: Merged GroundedDINO Detection and Quality Filtering
===========================================================

This script combines individual embryo detection with live/dead classification:
1. Phase 0: Individual embryo detection using base model
2. Phase 1: Live/dead detection using finetuned model  
3. Phase 2: Quality filtering for individual embryo annotations
4. Phase 3: Quality filtering for live/dead annotations (lower thresholds)

Features:
- Uses base model for individual embryo detection
- Uses finetuned model for live/dead classification
- Applies appropriate quality filtering to each annotation type
- Saves annotations to separate files for different model types
- Integrated high-quality annotation filtering with configurable thresholds
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
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
SANDBOX_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SANDBOX_ROOT))

# Import GroundedDINO utilities
from scripts.utils.grounded_sam_utils import (
    load_config, load_groundingdino_model, GroundedDinoAnnotations,
    calculate_detection_iou
)
from scripts.utils.experiment_metadata_utils import load_experiment_metadata, get_image_id_paths


def main():
    parser = argparse.ArgumentParser(description="Generate GroundedDINO detections and apply quality filtering")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--base-annotations", required=True, help="Path to base model annotations JSON")
    parser.add_argument("--finetuned-annotations", required=True, help="Path to finetuned model annotations JSON")
    
    # Quality filtering arguments
    parser.add_argument("--embryo-confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for individual embryo filtering (default: 0.5)")
    parser.add_argument("--embryo-iou-threshold", type=float, default=0.5,
                       help="IoU threshold for individual embryo duplicate removal (default: 0.5)")
    parser.add_argument("--live-dead-confidence-threshold", type=float, default=0.3,
                       help="Confidence threshold for live/dead filtering (default: 0.3 - lower)")
    parser.add_argument("--live-dead-iou-threshold", type=float, default=0.4,
                       help="IoU threshold for live/dead duplicate removal (default: 0.4 - lower)")
    parser.add_argument("--skip-filtering", action="store_true",
                       help="Skip quality filtering step")
    
    # Model configuration
    parser.add_argument("--finetuned-weights", default=None,
                       help="Path to finetuned model weights (overrides config)")
    
    args = parser.parse_args()

    # Set default finetuned weights if not provided
    if args.finetuned_weights is None:
        args.finetuned_weights = (
            "/net/trapnell/vol1/home/mdcolon/proj/"
            "image_segmentation/Open-GroundingDino/"
            "finetune_output/finetune_output_run_nick_masks_20250308/"
            "checkpoint_best_regular.pth"
        )

    print("🚀 Starting Merged GroundedDINO Detection and Quality Filtering Pipeline")
    print("=" * 80)
    
    # Load config and setup device
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Load experiment metadata
    metadata = load_experiment_metadata(args.metadata)
    image_ids = metadata.get("image_ids", [])
    print(f"📊 Total images in metadata: {len(image_ids)}")

    # =================================================================================
    # PHASE 0: Individual Embryo Detection (Base Model)
    # =================================================================================
    print("\n" + "="*60)
    print("🔍 PHASE 0: Individual Embryo Detection (Base Model)")
    print("="*60)
    
    # Initialize base model annotations manager
    base_annotations = GroundedDinoAnnotations(args.base_annotations, verbose=True)
    base_annotations.set_metadata_path(args.metadata)
    
    print(f"💾 Base annotations will be saved to: {args.base_annotations}")
    
    try:
        # Load base model
        base_model = load_groundingdino_model(config, device=device)
        print("✅ Base model loaded successfully")
        
        # Process individual embryo detection
        base_annotations.process_missing_annotations(
            model=base_model,
            prompts="individual embryo",
            auto_save_interval=100,
            store_image_source=False,
            show_anno=False,
            overwrite=False
        )
        base_annotations.save()
        
        print("✅ Phase 0 complete: Individual embryo detection finished")
        
    except Exception as e:
        print(f"❌ Error in Phase 0 (individual embryo): {e}")
        return

    # =================================================================================
    # PHASE 1: Live/Dead Detection (Finetuned Model)
    # =================================================================================
    print("\n" + "="*60)
    print("🔍 PHASE 1: Live/Dead Detection (Finetuned Model)")
    print("="*60)
    
    # Initialize finetuned model annotations manager
    ft_annotations = GroundedDinoAnnotations(args.finetuned_annotations, verbose=True)
    ft_annotations.set_metadata_path(args.metadata)
    
    print(f"💾 Finetuned annotations will be saved to: {args.finetuned_annotations}")
    
    try:
        # Update config to use finetuned weights
        original_weights = config["models"]["groundingdino"]["weights"]
        config["models"]["groundingdino"]["weights"] = args.finetuned_weights
        print(f"🔄 Switching to finetuned weights: {Path(args.finetuned_weights).name}")
        
        # Load finetuned model
        ft_model = load_groundingdino_model(config, device=device)
        print("✅ Finetuned model loaded successfully")
        
        # Process live and dead detection with lower thresholds
        ft_annotations.process_missing_annotations(
            model=ft_model,
            prompts=["live", "dead"],
            box_threshold=0.15,  # Lower threshold for finetuned model
            text_threshold=0.01,
            auto_save_interval=100,
            store_image_source=False,
            show_anno=False,
            overwrite=False,
            consider_different_if_different_weights=True
        )
        ft_annotations.save()
        
        print("✅ Phase 1 complete: Live/dead detection finished")
        
        # Restore original weights in config
        config["models"]["groundingdino"]["weights"] = original_weights
        
    except Exception as e:
        print(f"❌ Error in Phase 1 (live/dead detection): {e}")
        return

    # Skip filtering if requested
    if args.skip_filtering:
        print("\n⏭️ Skipping quality filtering as requested.")
        print("✅ GroundedDINO detection generation completed!")
        return

    # =================================================================================
    # PHASE 2: Quality Filtering for Individual Embryo Annotations
    # =================================================================================
    print("\n" + "="*60)
    print("🎯 PHASE 2: Quality Filtering for Individual Embryo Annotations")
    print("="*60)
    
    try:
        # Get all image IDs that have individual embryo annotations
        embryo_image_ids = []
        for image_id, image_data in base_annotations.annotations.get("images", {}).items():
            for annotation in image_data.get("annotations", []):
                if annotation.get("prompt") == "individual embryo":
                    embryo_image_ids.append(image_id)
                    break
        
        print(f"Found {len(embryo_image_ids)} images with 'individual embryo' annotations")
        
        if embryo_image_ids:
            print(f"🎯 Applying quality filtering to individual embryo annotations...")
            print(f"   Confidence threshold: {args.embryo_confidence_threshold}")
            print(f"   IoU threshold: {args.embryo_iou_threshold}")
            
            result = base_annotations.generate_high_quality_annotations(
                image_ids=embryo_image_ids,
                prompt="individual embryo",
                confidence_threshold=args.embryo_confidence_threshold,
                iou_threshold=args.embryo_iou_threshold,
                overwrite=True,
                save_to_self=True
            )
            
            base_annotations.save()
            
            # Print filtering results
            stats = result["statistics"]
            print(f"\n📊 Individual Embryo Filtering Summary:")
            print(f"   Original detections: {stats['original_detections']}")
            print(f"   Confidence removed: {stats['confidence_removed']}")
            print(f"   IoU removed: {stats['iou_removed']}")
            print(f"   Final detections: {stats['final_detections']}")
            print(f"   Retention rate: {stats['retention_rate']:.1%}")
            print(f"   Final images: {stats['final_images']}")
            
            print("✅ Phase 2 complete: Individual embryo quality filtering finished")
        else:
            print("⚠️ No individual embryo annotations found for filtering")
            
    except Exception as e:
        print(f"❌ Error in Phase 2 (embryo filtering): {e}")

    # =================================================================================
    # PHASE 3: Quality Filtering for Live/Dead Annotations (Lower Thresholds)
    # =================================================================================
    print("\n" + "="*60)
    print("🎯 PHASE 3: Quality Filtering for Live/Dead Annotations")
    print("="*60)
    
    try:
        # Process both live and dead prompts
        for prompt in ["live", "dead"]:
            # Get all image IDs that have this prompt's annotations
            prompt_image_ids = []
            for image_id, image_data in ft_annotations.annotations.get("images", {}).items():
                for annotation in image_data.get("annotations", []):
                    if annotation.get("prompt") == prompt:
                        prompt_image_ids.append(image_id)
                        break
            
            print(f"\nFound {len(prompt_image_ids)} images with '{prompt}' annotations")
            
            if prompt_image_ids:
                print(f"🎯 Applying quality filtering to '{prompt}' annotations...")
                print(f"   Confidence threshold: {args.live_dead_confidence_threshold}")
                print(f"   IoU threshold: {args.live_dead_iou_threshold}")
                
                result = ft_annotations.generate_high_quality_annotations(
                    image_ids=prompt_image_ids,
                    prompt=prompt,
                    confidence_threshold=args.live_dead_confidence_threshold,
                    iou_threshold=args.live_dead_iou_threshold,
                    overwrite=True,
                    save_to_self=True
                )
                
                # Print filtering results
                stats = result["statistics"]
                print(f"\n📊 '{prompt.title()}' Filtering Summary:")
                print(f"   Original detections: {stats['original_detections']}")
                print(f"   Confidence removed: {stats['confidence_removed']}")
                print(f"   IoU removed: {stats['iou_removed']}")
                print(f"   Final detections: {stats['final_detections']}")
                print(f"   Retention rate: {stats['retention_rate']:.1%}")
                print(f"   Final images: {stats['final_images']}")
            else:
                print(f"⚠️ No '{prompt}' annotations found for filtering")
        
        ft_annotations.save()
        print("✅ Phase 3 complete: Live/dead quality filtering finished")
        
    except Exception as e:
        print(f"❌ Error in Phase 3 (live/dead filtering): {e}")

    # =================================================================================
    # FINAL SUMMARY
    # =================================================================================
    print("\n" + "="*60)
    print("📊 FINAL PIPELINE SUMMARY")
    print("="*60)
    
    print(f"\n📁 Base model annotations (individual embryo): {args.base_annotations}")
    base_annotations.print_processing_summary(["individual embryo"])
    
    print(f"\n📁 Finetuned model annotations (live/dead): {args.finetuned_annotations}")
    ft_annotations.print_processing_summary(["live", "dead"], consider_different_if_different_weights=True)
    
    print(f"\n✅ ALL PROCESSING COMPLETE!")
    print(f"🎯 Individual embryo annotations: {args.base_annotations}")
    print(f"🎯 Live/dead annotations: {args.finetuned_annotations}")
    print(f"📊 Quality filtering applied with appropriate thresholds for each annotation type")


if __name__ == "__main__":
    main()

# Example usage:
# python3 03_merged_gdino_detection_and_filtering.py \
#   --config /path/to/pipeline_config.yaml \
#   --metadata /path/to/experiment_metadata.json \
#   --base-annotations /path/to/gdino_annotations.json \
#   --finetuned-annotations /path/to/gdino_annotations_finetuned_live_vs_dead.json \
#   --embryo-confidence-threshold 0.4 \
#   --embryo-iou-threshold 0.5 \
#   --live-dead-confidence-threshold 0.1 \
#   --live-dead-iou-threshold 0.4