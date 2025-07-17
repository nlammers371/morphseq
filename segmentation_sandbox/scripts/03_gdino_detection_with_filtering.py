#!/usr/bin/env python3
"""
Stage 3: Dual GroundedDINO Detection with Quality Filtering
==========================================================

This script generates two separate annotation files:
1. Base model annotations (individual embryo detection)
2. Finetuned model annotations (individual embryo detection)

Both models predict "individual embryo" for comparison purposes.
Then applies high-quality filtering to both annotation files.

Features:
- Uses base model for individual embryo detection
- Uses finetuned model for individual embryo detection
- Generates high-quality annotations for both files
- Saves annotations to separate files for model comparison
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

    parser = argparse.ArgumentParser(description="Generate GroundedDINO detections with quality filtering (finetuned model by default)")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--finetuned-annotations", required=True, help="Path to finetuned model annotations JSON")
    # Quality filtering arguments
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                       help="Confidence threshold for filtering (default: 0.45)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for duplicate removal (default: 0.5)")
    parser.add_argument("--skip-filtering", action="store_true",
                       help="Skip quality filtering step")
    # Model configuration
    parser.add_argument("--weights", default=None,
                       help="Path to model weights (default: finetuned weights)")
    args = parser.parse_args()

    # Use finetuned weights as default if not specified
    if not args.weights:
        args.weights = (
            "/net/trapnell/vol1/home/mdcolon/proj/"
            "image_segmentation/Open-GroundingDino/"
            "finetune_output/finetune_output_run_nick_masks_20250308/"
            "checkpoint_best_regular.pth"
        )

    print("🚀 Starting GroundedDINO Detection with Quality Filtering (Finetuned Model)")
    print("=" * 70)

    # Load config and setup device
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")

    # Load experiment metadata
    metadata = load_experiment_metadata(args.metadata)
    image_ids = metadata.get("image_ids", [])
    print(f"📊 Total images in metadata: {len(image_ids)}")

    # =================================================================================
    # PHASE 1: Finetuned Model Detection (Individual Embryo)
    # =================================================================================
    print("\n" + "="*60)
    print("🔍 PHASE 1: Finetuned Model Detection (Individual Embryo)")
    print("="*60)

    # Initialize finetuned model annotations manager
    ft_annotations = GroundedDinoAnnotations(args.finetuned_annotations, verbose=True)
    ft_annotations.set_metadata_path(args.metadata)

    print(f"💾 Finetuned annotations will be saved to: {args.finetuned_annotations}")

    try:
        # Update config to use finetuned weights
        original_weights = config["models"]["groundingdino"]["weights"]
        config["models"]["groundingdino"]["weights"] = args.weights
        print(f"🔄 Using weights: {Path(args.weights).name}")

        # Load model
        ft_model = load_groundingdino_model(config, device=device)
        print("✅ Model loaded successfully")

        # Process individual embryo detection with finetuned model
        ft_annotations.process_missing_annotations(
            model=ft_model,
            prompts="individual embryo",
            auto_save_interval=100,
            store_image_source=False,
            show_anno=False,
            overwrite=False,
            consider_different_if_different_weights=True
        )
        ft_annotations.save()

        print("✅ Phase 1 complete: Finetuned model detection finished")

        # Restore original weights in config
        config["models"]["groundingdino"]["weights"] = original_weights

    except Exception as e:
        print(f"❌ Error in Phase 1 (finetuned model): {e}")
        return

    # Skip filtering if requested
    if args.skip_filtering:
        print("\n⏭️ Skipping quality filtering as requested.")
        print("✅ GroundedDINO detection generation completed!")
        return

    # =================================================================================
    # PHASE 2: High-Quality Filtering for Base Model Annotations
    # =================================================================================
    print("\n" + "="*60)
    print("🎯 PHASE 2: High-Quality Filtering for Base Model Annotations")
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
            print(f"🎯 Generating high-quality annotations for individual embryo...")
            print(f"   Confidence threshold: {args.confidence_threshold}")
            print(f"   IoU threshold: {args.iou_threshold}")
            
            result = base_annotations.generate_high_quality_annotations(
                image_ids=embryo_image_ids,
                prompt="individual embryo",
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                overwrite=True,
                save_to_self=True
            )
            
            base_annotations.save()
            
            # Print filtering results
            stats = result["statistics"]
            print(f"\n📊 Base Model Filtering Summary:")
            print(f"   Original detections: {stats['original_detections']}")
            print(f"   Confidence removed: {stats['confidence_removed']}")
            print(f"   IoU removed: {stats['iou_removed']}")
            print(f"   Final detections: {stats['final_detections']}")
            print(f"   Retention rate: {stats['retention_rate']:.1%}")
            print(f"   Final images: {stats['final_images']}")
            
            print("✅ Phase 2 complete: Base model quality filtering finished")
        else:
            print("⚠️ No individual embryo annotations found for filtering")
            
    except Exception as e:
        print(f"❌ Error in Phase 2 (base model filtering): {e}")

    # =================================================================================
    # PHASE 3: High-Quality Filtering for Finetuned Model Annotations
    # =================================================================================
    print("\n" + "="*60)
    print("🎯 PHASE 3: High-Quality Filtering for Finetuned Model Annotations")
    print("="*60)
    
    try:
        # Get all image IDs that have individual embryo annotations from finetuned model
        ft_embryo_image_ids = []
        for image_id, image_data in ft_annotations.annotations.get("images", {}).items():
            for annotation in image_data.get("annotations", []):
                if annotation.get("prompt") == "individual embryo":
                    ft_embryo_image_ids.append(image_id)
                    break
        
        print(f"Found {len(ft_embryo_image_ids)} images with 'individual embryo' annotations from finetuned model")
        
        if ft_embryo_image_ids:
            print(f"🎯 Generating high-quality annotations for finetuned model...")
            print(f"   Confidence threshold: {args.confidence_threshold}")
            print(f"   IoU threshold: {args.iou_threshold}")
            
            result = ft_annotations.generate_high_quality_annotations(
                image_ids=ft_embryo_image_ids,
                prompt="individual embryo",
                confidence_threshold=args.confidence_threshold,
                iou_threshold=args.iou_threshold,
                overwrite=True,
                save_to_self=True
            )
            
            # Print filtering results
            stats = result["statistics"]
            print(f"📊 Finetuned Model Filtering Summary:")
            print(f"   Original detections: {stats['original_detections']}")
            print(f"   Confidence removed: {stats['confidence_removed']}")
            print(f"   IoU removed: {stats['iou_removed']}")
            print(f"   Final detections: {stats['final_detections']}")
            print(f"   Retention rate: {stats['retention_rate']:.1%}")
            print(f"   Final images: {stats['final_images']}")
        else:
            print("⚠️ No individual embryo annotations found for filtering")
        
        ft_annotations.save()
        print("✅ Phase 3 complete: Finetuned model quality filtering finished")
        
    except Exception as e:
        print(f"❌ Error in Phase 3 (finetuned model filtering): {e}")

    # =================================================================================
    # FINAL SUMMARY
    # =================================================================================
    print("\n" + "="*60)
    print("📊 FINAL PIPELINE SUMMARY")
    print("="*60)
    
    print(f"\n📁 BASE MODEL ANNOTATIONS:")
    print(f"   File: {args.base_annotations}")
    
    # Additional base model stats
    base_total_images = len([img_id for img_id, img_data in base_annotations.annotations.get("images", {}).items() 
                           if any(ann.get("prompt") == "individual embryo" for ann in img_data.get("annotations", []))])
    base_total_detections = sum([ann.get("num_detections", 0) 
                               for img_data in base_annotations.annotations.get("images", {}).values()
                               for ann in img_data.get("annotations", [])
                               if ann.get("prompt") == "individual embryo"])
    base_hq_detections = sum([ann.get("num_detections", 0) 
                            for img_data in base_annotations.annotations.get("images", {}).values()
                            for ann in img_data.get("annotations", [])
                            if ann.get("prompt") == "individual embryo" and ann.get("annotation_type") == "high_quality"])
    
    print(f"   📊 Base Model Stats:")
    print(f"      • Images with detections: {base_total_images}")
    print(f"      • Total detections: {base_total_detections}")
    print(f"      • High-quality detections: {base_hq_detections}")
    if base_total_detections > 0:
        print(f"      • High-quality retention rate: {base_hq_detections/base_total_detections:.1%}")
    
    print(f"\n📁 FINETUNED MODEL ANNOTATIONS:")
    print(f"   File: {args.finetuned_annotations}")
    
    # Additional finetuned model stats
    ft_total_images = len([img_id for img_id, img_data in ft_annotations.annotations.get("images", {}).items() 
                         if any(ann.get("prompt") == "individual embryo" for ann in img_data.get("annotations", []))])
    ft_total_detections = sum([ann.get("num_detections", 0) 
                             for img_data in ft_annotations.annotations.get("images", {}).values()
                             for ann in img_data.get("annotations", [])
                             if ann.get("prompt") == "individual embryo"])
    ft_hq_detections = sum([ann.get("num_detections", 0) 
                          for img_data in ft_annotations.annotations.get("images", {}).values()
                          for ann in img_data.get("annotations", [])
                          if ann.get("prompt") == "individual embryo" and ann.get("annotation_type") == "high_quality"])
    
    print(f"   📊 Finetuned Model Stats:")
    print(f"      • Images with detections: {ft_total_images}")
    print(f"      • Total detections: {ft_total_detections}")
    print(f"      • High-quality detections: {ft_hq_detections}")
    if ft_total_detections > 0:
        print(f"      • High-quality retention rate: {ft_hq_detections/ft_total_detections:.1%}")
    
    # Comparison summary
    print(f"\n🔍 MODEL COMPARISON SUMMARY:")
    print(f"   📈 Detection Counts:")
    print(f"      • Base model: {base_total_detections} total, {base_hq_detections} high-quality")
    print(f"      • Finetuned model: {ft_total_detections} total, {ft_hq_detections} high-quality")
    
    if base_total_detections > 0 and ft_total_detections > 0:
        detection_ratio = ft_total_detections / base_total_detections
        hq_ratio = ft_hq_detections / base_hq_detections if base_hq_detections > 0 else 0
        print(f"   📊 Ratios (Finetuned/Base):")
        print(f"      • Total detections: {detection_ratio:.2f}x")
        if hq_ratio > 0:
            print(f"      • High-quality detections: {hq_ratio:.2f}x")
    
    print(f"\n✅ DUAL ANNOTATION GENERATION COMPLETE!")
    print(f"🎯 Base model file (individual embryo): {args.base_annotations}")
    print(f"🎯 Finetuned model file (individual embryo): {args.finetuned_annotations}")
    print(f"📊 High-quality annotations generated for both files")
    print(f"🔍 Ready for model comparison analysis!")


if __name__ == "__main__":
    main()

## Example usage:
# python3 03_gdino_detection_with_filtering.py \
#   --config /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/configs/pipeline_config.yaml \
#   --metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json \
#   --finetuned-annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations_finetuned.json \
#   --confidence-threshold 0.45 \
#   --iou-threshold 0.5
# (Finetuned weights will be used by default unless --weights is specified)

# Both models will predict "individual embryo" for comparison purposes