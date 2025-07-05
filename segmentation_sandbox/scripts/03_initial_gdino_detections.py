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

    # Determine which images need processingf
    processed = set(annotations.get_all_image_ids())
    to_process = [img for img in image_ids if img not in processed]
    # Save any new annotations
    annotations.process_missing_annotations(model, args.prompts,auto_save_interval= 100,store_image_source=False )
    annotations.save()  # Persist annotations to disk




# python3 /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/03_initial_gdino_detections.py --config /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/configs/pipeline_config.yaml --metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/experiment_metadata.json --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/gdino_annotations/gdino_annotations.json --prompts "individual embryo"