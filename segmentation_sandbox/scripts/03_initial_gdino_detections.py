#!/usr/bin/env python3
"""
Stage 3: GroundedDINO Detection Generation
=========================================

This script loads the GroundedDINO model and generates detections for embryo images.
It processes all images in the experiment_metadata.json file and saves detections
in a structured format for downstream SAM2 processing.

Features:
- Loads GroundedDINO model from pipeline configuration
- Processes images incrementally (skips already processed ones)
- Saves detections with confidence scores and prompts
- Tracks model checkpoints and processing metadata
- Integrated high-quality annotation filtering
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GroundedDINO detections for embryo images")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--metadata", required=True, help="Path to experiment_metadata.json")
    parser.add_argument("--annotations", required=True, help="Path to output annotations JSON")
    parser.add_argument("--prompts", nargs="+", required=True, help="List of text prompts")
    
    # Quality filtering arguments
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for filtering (default: 0.5)")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                       help="IoU threshold for duplicate removal (default: 0.5)")
    parser.add_argument("--skip-filtering", action="store_true",
                       help="Skip quality filtering step")
    
    args = parser.parse_args()

    # Load model
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_groundingdino_model(config, device=device)
    print("Model loaded successfully")
    
    # Load experiment metadata
    metadata = load_experiment_metadata(args.metadata)
    image_ids = metadata.get("image_ids", [])

    # Initialize annotations manager
    annotations = GroundedDinoAnnotations(args.annotations)
    annotations.set_metadata_path(args.metadata)  # Set metadata path for annotations manager

    # Save any new annotations
    annotations.process_missing_annotations(model, args.prompts, auto_save_interval=100, store_image_source=False)
    annotations.save()  # Persist annotations to disk

    if args.skip_filtering:
        print("Skipping quality filtering as requested.")
        print("✅ GroundedDINO detection generation completed!")
        sys.exit(0)

    print("Initial GroundedDINO detections completed. Now applying quality filtering...")

    # =================================================================================
    # Block 3: High Quality Annotations Filtering (Using New Methods)
    # =================================================================================
    
    print("\n=== Block 3: Quality Filtering for GroundedDINO Annotations ===")
    
    # Get all image IDs that have annotations for the target prompt
    target_prompt = args.prompts
    all_image_ids = []
    
    for image_id, image_data in annotations.annotations.get("images", {}).items():
        for annotation in image_data.get("annotations", []):
            if annotation.get("prompt") == target_prompt:
                all_image_ids.append(image_id)
                break
    
    print(f"Found {len(all_image_ids)} images with '{target_prompt}' annotations")
    
    if not all_image_ids:
        print("❌ No individual embryo annotations found. Exiting.")
        sys.exit(1)

    # Generate high-quality annotations using the new method
    print(f"\n🎯 Generating high-quality annotations...")
    print(f"   Confidence threshold: {args.confidence_threshold}")
    print(f"   IoU threshold: {args.iou_threshold}")
    
    for prompt in args.prompts:
        print(f"\nGenerating high quality annotations for : {prompt}")
        result = annotations.generate_high_quality_annotations(
            image_ids=all_image_ids,
            prompt=target_prompt,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            overwrite=True,
            save_to_self=True
        )

          # Save the updated annotations
        annotations.save()
        
        # Print filtering results
        stats = result["statistics"]
        print(f"\n🎯 Filtering Summary:")
        print(f"   Original detections: {stats['original_detections']}")
        print(f"   Confidence removed: {stats['confidence_removed']}")
        print(f"   IoU removed: {stats['iou_removed']}")
        print(f"   Final detections: {stats['final_detections']}")
        print(f"   Retention rate: {stats['retention_rate']:.1%}")
        print(f"   Final images: {stats['final_images']}")
        print(f"   Experiments processed: {stats['experiments_processed']}")
        
        # # Export high-quality annotations to separate file
        # high_quality_path = Path(args.annotations).parent / "gdino_high_quality_annotations.json"
        # annotations.export_high_quality_annotations(high_quality_path)
        
        # print(f"✅ High-quality annotations saved to: {high_quality_path}")
        # print(f"\n✅ Block 3 (Quality Filtering) completed successfully!")
        # print(f"📁 Next step: Use high-quality annotations for SAM2 processing in Block 4")

# Example usage:
# python3 /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/03_initial_gdino_detections.py --config /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/configs/pipeline_config.yaml --metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations.json --prompts "individual embryo"