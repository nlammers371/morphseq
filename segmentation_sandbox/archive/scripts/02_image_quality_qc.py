#!/usr/bin/env python
"""
02_image_quality_qc.py

Performs automated image quality assessment following the proper MorphSeq QC workflow:

WORKFLOW: Initialize → Manual → Automatic → Done

1. INITIALIZE: Adds all images from metadata to QC JSON (this script does this first)
2. MANUAL: Human experts can manually review and flag images (external process)  
3. AUTOMATIC: This script performs algorithmic QC on remaining unflagged images
4. DONE: Complete QC dataset ready for pipeline

Usage:
    python scripts/02_image_quality_qc.py \
        --data_dir /path/to/raw_data_organized \
        --experiments_to_process 20240411,20240412

QC Philosophy:
    - Images default to no flags = assumed good quality
    - Only flag images with actual detected problems
    - Manual annotations always take precedence over automatic

NEW JSON STRUCTURE:
==================
The QC system now uses a hierarchical JSON format in experiment_data_qc.json:

{
    "valid_qc_flag_categories": {
        "experiment_level": {"POOR_IMAGING_CONDITIONS": "Description", ...},
        "video_level": {"DRY_WELL": "Description", "FOCUS_DRIFT": "Description", ...},
        "image_level": {"BLUR": "Description", "DARK": "Description", ...},
        "embryo_level": {"DEAD_EMBRYO": "Description", ...}
    },
    "experiments": {
        "20241215": {
            "flags": ["POOR_IMAGING_CONDITIONS"],
            "authors": ["mcolon"],
            "notes": ["Manual review notes"],
            "videos": {
                "20241215_A01": {
                    "flags": ["DRY_WELL"],
                    "authors": ["mcolon"],
                    "notes": ["Well dried out after frame 50"],
                    "images": {
                        "20241215_A01_t001": {
                            "flags": ["BLUR"],
                            "authors": ["automatic"],
                            "notes": ["blur_score=45 < threshold=100"]
                        }
                    },
                    "embryos": {
                        "20241215_A01_t001_e01": {
                            "flags": ["DEAD_EMBRYO"],
                            "authors": ["expert_reviewer"],
                            "notes": ["No movement detected"]
                        }
                    }
                }
            }
        }
    }
}

FUNCTION BREAKDOWN:
===================

📋 INITIALIZATION FUNCTIONS:
    - initialize_qc_structure_from_metadata()
        Purpose: Populates QC JSON with ALL experiments/videos/images from metadata
        Input: experiment_metadata.json file
        Output: QC JSON with hierarchical structure (no flags by default)
        Philosophy: Every entity starts unflagged (assumed good quality)

🏷️ QC FLAGGING FUNCTIONS:
    - add_qc_flag()
        Purpose: Adds QC flags at any level (experiment/video/image/embryo)
        Input: level, entity_id, qc_flag, author, notes, parent_ids
        Output: Updated QC JSON with flags only for problem entities
        Philosophy: Only flag when there's an actual quality issue
    
    - flag_image(), flag_video(), flag_experiment(), flag_embryo()
        Purpose: Convenience functions for flagging specific entity types
        Input: entity_id, qc_flag, author, notes
        Output: Updated QC JSON

📊 QC DATA MANAGEMENT:
    - load_qc_data()
        Purpose: Loads current QC JSON state
        Returns: Dict with hierarchical QC structure
    - get_qc_flags()
        Purpose: Gets QC flags for specific entity
        Returns: Dict with flags, authors, notes for entity
    - get_qc_summary()
        Purpose: Summarizes QC flags across all levels
        Returns: Dict with flag counts by level

🔍 IMAGE ANALYSIS FUNCTIONS:
    - calculate_image_metrics()
        Purpose: Analyzes single image for quality issues (blur, corruption)
        Input: Image file path
        Output: Quality metrics + flag recommendation (None if good)
    - process_single_image()
        Purpose: QC wrapper for single image processing
        Returns: (image_id, qc_flag) or (image_id, None) if no problems
    - process_experiment_qc()
        Purpose: Batch processes all images in an experiment
        Workflow: Load existing QC → Analyze unflagged images → Flag problems only

🔄 WORKFLOW ORCHESTRATION:
    - main()
        Purpose: Orchestrates full QC workflow (Initialize → Automatic QC)
        Steps: 1) Initialize QC JSON from metadata
               2) Run automatic analysis on unflagged images  
               3) Flag only images with detected problems
               4) Generate summary report

🎯 KEY DESIGN PRINCIPLES:
    - Hierarchical structure mirrors experiment_metadata.json organization
    - Images default to no flags = assumed good quality
    - Only flag entities with actual detected problems
    - Manual annotations always take precedence over automatic
    - Batch processing for efficiency with parallel workers
    - Clear separation: initialization vs. analysis vs. flagging
    - Preserve existing manual QC decisions (never overwrite)
    - Multi-level QC: experiment, video, image, and embryo levels
    - Author tracking for manual vs automatic annotations

This script respects existing manual annotations and only processes images
that haven't been manually reviewed. The philosophy is "innocent until proven guilty" -
entities are assumed to be good quality unless specifically flagged.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Import from scripts directory
from experiment_data_qc_utils import (
    initialize_qc_structure_from_metadata,  # CRITICAL: Initialize QC JSON from metadata first
    load_qc_data,
    get_qc_flags,
    flag_image,
    get_qc_summary,
    parse_image_id,
    VALID_QC_FLAG_CATEGORIES
)

# QC thresholds (can be adjusted based on your data)
QC_THRESHOLDS = {
    'blur_threshold': 100,      # Variance of Laplacian below this = BLUR
}

def calculate_image_metrics(image_path: Path) -> Dict:
    """
    Calculate quality metrics for an image.
    Returns dict with 'qc_flag' (None if good quality) and 'metrics'.
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {'qc_flag': 'CORRUPT', 'metrics': {}}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate blur score
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        metrics = {
            'blur_score': float(blur_score),
            'width': image.shape[1],
            'height': image.shape[0]
        }
        
        # Determine QC flag - only flag if there's a problem
        if blur_score < QC_THRESHOLDS['blur_threshold']:
            qc_flag = 'BLUR'
        else:
            qc_flag = None  # No flag needed - image is good quality
        
        return {'qc_flag': qc_flag, 'metrics': metrics}
        
    except Exception as e:
        return {'qc_flag': 'CORRUPT', 'metrics': {'error': str(e)}}

def process_single_image(image_id: str, images_dir: Path, quality_control_dir: Path, overwrite: bool) -> Tuple[str, Optional[str]]:
    """
    Process a single image for QC. Returns (image_id, qc_flag) or (image_id, None) if no flag needed.
    """
    # Check if image already has QC flags (and we're not overwriting)
    if not overwrite:
        try:
            parent_ids = parse_image_id(image_id)
            existing_flags = get_qc_flags(quality_control_dir, "image", image_id, parent_ids)
            if existing_flags["flags"]:  # Has existing flags
                return image_id, None  # Skip processing
        except (ValueError, KeyError):
            pass  # Continue with processing
    
    # Find the JPEG file
    jpeg_path = images_dir / f"{image_id}.jpg"
    if not jpeg_path.exists():
        return image_id, None
    
    # Calculate QC metrics
    qc_result = calculate_image_metrics(jpeg_path)
    
    # Return flag if there's a problem, None if good quality
    return image_id, qc_result['qc_flag']

def process_experiment_qc(
    experiment_id: str,
    raw_data_dir: Path,
    quality_control_dir: Path,
    metadata: Dict,
    overwrite: bool = False,
    verbose: bool = True,
    workers: int = 4
) -> Tuple[int, int]:
    """Process QC for all images in an experiment. Returns (processed_count, skipped_count)."""
    
    if experiment_id not in metadata['experiments']:
        print(f"Warning: Experiment {experiment_id} not found in metadata")
        return 0, 0
    
    experiment_data = metadata['experiments'][experiment_id]
    if verbose:
        print(f"\nProcessing QC for experiment: {experiment_id}")
    
    processed_count = 0
    skipped_count = 0
    
    for video_id, video_data in experiment_data['videos'].items():
        well_id = video_data['well_id']
        image_ids = video_data.get('image_ids', [])
        
        if verbose:
            print(f"  Processing video {video_id}: {len(image_ids)} images")
        
        images_dir = Path(video_data['processed_jpg_images_dir'])
        
        # Filter images that need processing
        images_to_process = []
        for image_id in image_ids:
            # Check if image already has flags
            if not overwrite:
                try:
                    parent_ids = parse_image_id(image_id)
                    existing_flags = get_qc_flags(quality_control_dir, "image", image_id, parent_ids)
                    if existing_flags["flags"]:
                        skipped_count += 1
                        continue
                except (ValueError, KeyError):
                    pass  # Continue with processing
            
            images_to_process.append(image_id)
        
        if not images_to_process:
            if verbose:
                print(f"    All images already processed, skipping.")
            continue
        
        if verbose:
            print(f"    Processing {len(images_to_process)} images with {workers} workers")
        
        # Process images in parallel
        qc_flags_to_add = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(process_single_image, image_id, images_dir, quality_control_dir, overwrite): image_id
                for image_id in images_to_process
            }
            
            # Collect results with progress bar
            for future in tqdm(as_completed(future_to_image), 
                             total=len(future_to_image), 
                             desc=f"QC {video_id}", 
                             disable=not verbose, 
                             leave=False):
                try:
                    image_id, qc_flag = future.result()
                    if qc_flag is not None:  # Only add if there's actually a problem
                        qc_flags_to_add.append((image_id, qc_flag))
                except Exception as e:
                    image_id = future_to_image[future]
                    print(f"Warning: Failed to process {image_id}: {e}")
        
        # Add all QC flags in batch
        for image_id, qc_flag in qc_flags_to_add:
            try:
                flag_image(
                    quality_control_dir=quality_control_dir,
                    image_id=image_id,
                    qc_flag=qc_flag,
                    author='automatic',
                    notes=f"Automatic QC: {qc_flag}",
                    overwrite=overwrite
                )
                processed_count += 1
            except Exception as e:
                print(f"Warning: Failed to flag {image_id}: {e}")
                continue
    
    if verbose:
        print(f"  Processed {processed_count} images, skipped {skipped_count}")
    
    return processed_count, skipped_count

def main():
    parser = argparse.ArgumentParser(
        description="Automated image quality assessment for MorphSeq pipeline using hierarchical JSON structure",
        epilog="""
Examples:
  # Process all experiments
  python 02_image_quality_qc.py \\
    --raw_data_dir /path/to/data/raw_data_organized \\
    --quality_control_dir /path/to/data/quality_control

  # Process specific experiments  
  python 02_image_quality_qc.py \\
    --raw_data_dir /path/to/data/raw_data_organized \\
    --quality_control_dir /path/to/data/quality_control \\
    --experiments_to_process 20241215,20241216
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw_data_dir", type=str, required=True,
        help="Path to raw_data_organized directory (contains experiment_metadata.json)"
    )
    parser.add_argument(
        "--quality_control_dir", type=str, required=True,
        help="Path to quality_control directory (where experiment_data_qc.json will be stored)"
    )
    parser.add_argument(
        "--experiments_to_process", type=str,
        help="Comma-separated list of experiment IDs to process. If not provided, processes all."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process images that already have QC flags"
    )
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true',
        help="Enable detailed output"
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help="Number of parallel workers for processing (default: 4)"
    )
    parser.set_defaults(verbose=True)
    
    args = parser.parse_args()
    
    raw_data_dir = Path(args.raw_data_dir)
    quality_control_dir = Path(args.quality_control_dir)
    
    if not raw_data_dir.exists():
        print(f"Error: Raw data directory not found: {raw_data_dir}")
        return 1
    
    # Load metadata
    metadata_path = raw_data_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        print("Run 01_prepare_videos.py first to generate metadata")
        return 1

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Ensure QC directory exists
    quality_control_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: INITIALIZE QC STRUCTURE FROM METADATA (CRITICAL FIRST STEP)
    print("🔄 Step 1: Initializing hierarchical QC structure from metadata...")
    print("   → Scanning experiment_metadata.json for all experiments/videos/images")
    print("   → Creating hierarchical JSON structure mirroring metadata organization")
    print("   → Adding new entries to experiment_data_qc.json")
    
    # Initialize QC structure to ensure ALL entities from metadata are tracked
    qc_data_before = load_qc_data(quality_control_dir)
    existing_experiments = len(qc_data_before.get("experiments", {}))
    
    qc_data_after = initialize_qc_structure_from_metadata(
        quality_control_dir=quality_control_dir,
        experiment_metadata_path=metadata_path,
        overwrite=False  # Preserve existing manual annotations
    )
    
    new_experiments = len(qc_data_after.get("experiments", {})) - existing_experiments
    print(f"✅ QC structure initialization complete!")
    print(f"   📈 Total experiments now tracked: {len(qc_data_after.get('experiments', {}))}")
    print(f"   ➕ New experiments added from metadata: {new_experiments}")
    print(f"   💾 Existing QC annotations preserved")
    
    if new_experiments > 0:
        print("   📝 New entities are ready for manual/automatic QC")
    else:
        print("   ✅ All entities already in QC structure")
    
    # STEP 2: AUTOMATIC QC PROCESSING
    print(f"\n🤖 Step 2: Automatic image-level QC processing...")
    print("   → Processing images algorithmically (blur detection, etc.)")
    print("   → Only flagging images that fail quality checks")
    print("   → Manual annotations take precedence over automatic")
    
    # Determine experiments to process
    if args.experiments_to_process:
        experiments_to_process = [e.strip() for e in args.experiments_to_process.split(',')]
    else:
        experiments_to_process = list(metadata['experiments'].keys())
    
    print(f"Processing QC for {len(experiments_to_process)} experiments")
    
    # Process each experiment with automatic QC
    total_processed = 0
    total_skipped = 0
    for experiment_id in experiments_to_process:
        processed, skipped = process_experiment_qc(
            experiment_id, raw_data_dir, quality_control_dir, metadata, 
            args.overwrite, args.verbose, args.workers
        )
        total_processed += processed
        total_skipped += skipped
    
    # STEP 3: SUMMARY AND COMPLETION
    print(f"\n🎯 Step 3: QC Processing Complete!")
    print(f"   📊 Total images processed: {total_processed}")
    print(f"   ⏭️  Total images skipped (already flagged): {total_skipped}")
    
    # Load final QC data for summary
    qc_summary = get_qc_summary(quality_control_dir)
    print(f"\nQC processing complete. Summary by level:")
    
    for level, flags in qc_summary.items():
        if flags:
            print(f"\n{level.replace('_', ' ').title()}:")
            for flag, count in flags.items():
                print(f"  {flag}: {count}")
    
    return 0
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
