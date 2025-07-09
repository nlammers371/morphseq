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
from scripts.utils.grounded_sam_utils import load_config, load_groundingdino_model, GroundedDinoAnnotations
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
    print(device)
    model = load_groundingdino_model(config, device=device)
    print("Model loaded successfullyand fixed memory issue ")
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
    # Block 3: High Quality Annotations Filtering  
    # =================================================================================
    
    print("\n=== Block 3: Quality Filtering for GroundedDINO Annotations ===")
    
   # Load the generated annotations
    with open(args.annotations, 'r') as f:
        all_annotations = json.load(f)
    
    # Extract all "individual embryo" detections and collect confidence scores
    individual_embryo_detections = []
    
    print("Extracting individual embryo detections...")
    for image_id, data in all_annotations.get("images", {}).items():
        if 'annotations' in data:
            for annotation in data['annotations']:
                if annotation.get('prompt') == 'individual embryo':
                    for detection in annotation.get('detections', []):
                        if detection.get('phrase') == 'individual embryo':
                        individual_embryo_detections.append({
                            'image_id': image_id,
                            'detection': detection,
                            'annotation_id': annotation.get('annotation_id'),
                            'confidence': detection.get('confidence', 0)
                        })
    
    print(f"Found {len(individual_embryo_detections)} individual embryo detections")
    
    # Gather confidences directly from detections
    confidences = [d['confidence'] for d in individual_embryo_detections]
    if not confidences:
        print("❌ No individual embryo detections found. Exiting.")
        sys.exit(1)
    

    

    
    # Print confidence statistics
    print("📈 Confidence Statistics:")
    print(f"   Total detections: {len(confidences)}")
    print(f"   Mean: {mean_conf:.3f}")
    print(f"   Median: {median_conf:.3f}")
    print(f"   Min: {np.min(confidences):.3f}")
    print(f"   Max: {np.max(confidences):.3f}")
    print(f"   Q90: {np.percentile(confidences, 90):.3f}")
    print(f"   Q95: {np.percentile(confidences, 95):.3f}")
    
    # Apply confidence threshold filter
    confidence_threshold = args.confidence_threshold
    print(f"\n🔍 Applying confidence threshold filter (>= {confidence_threshold})...")
    
    high_confidence_detections = [
        det for det in individual_embryo_detections 
        if det['detection']['confidence'] >= confidence_threshold
    ]
    
    removed_by_confidence = len(individual_embryo_detections) - len(high_confidence_detections)
    print(f"   Removed {removed_by_confidence} low-confidence detections")
    print(f"   Retained {len(high_confidence_detections)} high-confidence detections")
    
    # Group detections by image_id for IoU filtering
    print(f"\n🔍 Applying IoU-based duplicate removal...")
    detections_by_image = defaultdict(list)
    for det in high_confidence_detections:
        detections_by_image[det['image_id']].append(det)
    
    # Apply IoU-based non-maximum suppression
    iou_threshold = args.iou_threshold
    filtered_detections = []
    total_iou_removed = 0
    
    for image_id, detections in detections_by_image.items():
        if len(detections) <= 1:
            filtered_detections.extend(detections)
            continue
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['detection']['confidence'], reverse=True)
        
        keep_detections = []
        for i, det in enumerate(detections):
            should_keep = True
            for kept_det in keep_detections:
                iou = calculate_detection_iou(
                    det['detection']['box_xywh'],
                    kept_det['detection']['box_xywh']
                )
                if iou > iou_threshold:
                    should_keep = False
                    total_iou_removed += 1
                    break
            
            if should_keep:
                keep_detections.append(det)
        
        filtered_detections.extend(keep_detections)
    
    print(f"   Removed {total_iou_removed} overlapping detections")
    print(f"   Final high-quality detections: {len(filtered_detections)}")
    
    # Create high quality annotations JSON structure
    print(f"\n💾 Creating high-quality annotations file...")
    high_quality_annotations = {"images": {}}
    
    # Group filtered detections back by image_id
    filtered_by_image = defaultdict(list)
    for det in filtered_detections:
        filtered_by_image[det['image_id']].append(det)
    
    # Build the high-quality annotations structure
    for image_id, detections in filtered_by_image.items():
        if image_id not in high_quality_annotations["images"]:
            high_quality_annotations["images"][image_id] = {"annotations": []}
        
        # Group detections by annotation_id
        detections_by_annotation = defaultdict(list)
        for det in detections:
            detections_by_annotation[det['annotation_id']].append(det['detection'])
        
        # Create filtered annotations
        for annotation_id, detection_list in detections_by_annotation.items():
            # Find the original annotation to copy metadata
            original_annotation = None
            for annotation in all_annotations["images"][image_id]['annotations']:
                if annotation.get('annotation_id') == annotation_id:
                    original_annotation = annotation
                    break
            
            if original_annotation:
                # Create filtered annotation
                filtered_annotation = original_annotation.copy()
                filtered_annotation['detections'] = detection_list
                filtered_annotation['num_detections'] = len(detection_list)
                high_quality_annotations["images"][image_id]['annotations'].append(filtered_annotation)
    
    # Add filtering metadata
    high_quality_annotations["filtering_metadata"] = {
        "original_detections": len(individual_embryo_detections),
        "confidence_threshold": confidence_threshold,
        "confidence_removed": removed_by_confidence,
        "iou_threshold": iou_threshold,
        "iou_removed": total_iou_removed,
        "final_detections": len(filtered_detections),
        "retention_rate": len(filtered_detections) / len(individual_embryo_detections) if len(individual_embryo_detections) > 0 else 0,
        "processing_timestamp": datetime.now().isoformat(),
        "script_version": "03_initial_gdino_detections.py"
    }
    
    # Save high quality annotations
    high_quality_path = Path(args.annotations).parent / "gdino_high_quality_annotations.json"
    with open(high_quality_path, 'w') as f:
        json.dump(high_quality_annotations, f, indent=2)
    
    print(f"✅ High-quality annotations saved to: {high_quality_path}")
    
    # Print summary
    retention_rate = (len(filtered_detections) / len(individual_embryo_detections)) * 100
    print(f"\n🎯 Filtering Summary:")
    print(f"   Original detections: {len(individual_embryo_detections)}")
    print(f"   After confidence filter: {len(high_confidence_detections)}")
    print(f"   After IoU filter: {len(filtered_detections)}")
    print(f"   Retention rate: {retention_rate:.1f}%")
    print(f"   Images with high-quality detections: {len(filtered_by_image)}")
    
    print(f"\n✅ Block 3 (Quality Filtering) completed successfully!")
    print(f"📁 Next step: Use '{high_quality_path}' for SAM2 processing in Block 4")



# python3 /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/03_initial_gdino_detections.py --config /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/configs/pipeline_config.yaml --metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations.json --prompts "individual embryo"