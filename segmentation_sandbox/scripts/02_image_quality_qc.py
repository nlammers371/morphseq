#!/usr/bin/env python
"""
02_image_quality_qc.py

Performs automated image quality assessment following the proper MorphSeq QC workflow:

WORKFLOW: Initialize → Manual → Automatic → Done

1. INITIALIZE: Adds all images from metadata to QC CSV (this script does this first)
2. MANUAL: Human experts can manually review and flag images (external process)  
3. AUTOMATIC: This script performs algorithmic QC on remaining unflagged images
4. DONE: Complete QC dataset ready for pipeline

Usage:
    python scripts/02_image_quality_qc.py \
        --data_dir /path/to/raw_data_organized \
        --experiments_to_process 20240411,20240412

QC Philosophy:
    - Images default to None (no flag) = assumed good quality
    - Only flag images with actual detected problems
    - Manual annotations always take precedence over automatic

QC Flags:
    - None: Good quality image (default, no flag needed)
    - BLUR: Image is blurry (low variance of Laplacian)
    - CORRUPT: Cannot read/process image
    - DARK: Image is too dark
    - BRIGHT: Image is overexposed
    - MANUAL_REJECT: Human reviewer flagged as problematic

FUNCTION BREAKDOWN:
===================

📋 INITIALIZATION FUNCTIONS:
    - initialize_qc_file()
        Purpose: Populates QC CSV with ALL images from experiment metadata
        Input: experiment_metadata.json file
        Output: QC CSV with image entries (qc_flag=None by default)
        Philosophy: Every image starts as None (assumed good quality)

🏷️ QC FLAGGING FUNCTIONS:
    - flag_qc()
        Purpose: Adds/updates QC flags for specific problematic images
        Input: image_ids, qc_flag, annotator (manual/automatic)
        Output: Updated QC CSV with flags only for problem images
        Philosophy: Only flag when there's an actual quality issue

📊 QC DATA MANAGEMENT:
    - load_qc_data()
        Purpose: Loads current QC CSV state
        Returns: DataFrame with all QC records
    - check_existing_qc()
        Purpose: Checks if images already have QC annotations
        Returns: Dict mapping image_id to existing flags (or None)

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
        Steps: 1) Initialize QC CSV from metadata
               2) Run automatic analysis on unflagged images  
               3) Flag only images with detected problems
               4) Generate summary report

🎯 KEY DESIGN PRINCIPLES:
    - Images default to None (no flag) = assumed good quality
    - Only flag images with actual detected problems
    - Manual annotations always take precedence over automatic
    - Batch processing for efficiency with parallel workers
    - Clear separation: initialization vs. analysis vs. flagging
    - Preserve existing manual QC decisions (never overwrite)

This script respects existing manual annotations and only processes images
that haven't been manually reviewed. The philosophy is "innocent until proven guilty" -
images are assumed to be good quality unless specifically flagged.
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'image_quality_qc_utils'))
from image_quality_qc_utils import (
    initialize_qc_file,  # CRITICAL: Initialize QC CSV from metadata first
    get_qc_csv_path, 
    load_qc_data, 
    check_existing_qc, 
    flag_qc, 
    QC_FLAGS
)

# QC thresholds (can be adjusted based on your data)
QC_THRESHOLDS = {
    'blur_threshold': 100,      # Variance of Laplacian below this = BLUR
}

def calculate_image_metrics(image_path: Path) -> Dict:
    """Calculate quality metrics for an image."""
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

def process_single_image(image_id: str, images_dir: Path, data_dir: Path, overwrite: bool) -> Tuple[str, Optional[str]]:
    """
    Process a single image for QC. Returns (image_id, qc_flag) or (image_id, None) if no flag needed.
    """
    # Find the JPEG file
    jpeg_path = images_dir / f"{image_id}.jpg"
    if not jpeg_path.exists():
        return image_id, None
    
    # Calculate QC metrics
    qc_result = calculate_image_metrics(jpeg_path)
    
    # Skip if QC result indicates passing quality
    if qc_result['qc_flag'] is None:
        return image_id, None
    
    return image_id, qc_result['qc_flag']

def process_experiment_qc(
    experiment_id: str,
    data_dir: Path,
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
    
    # Load QC data once for the entire experiment
    qc_df = load_qc_data(data_dir)
    existing_image_ids = set(qc_df['image_id'].tolist()) if len(qc_df) > 0 else set()
    
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
            if not overwrite and image_id in existing_image_ids:
                skipped_count += 1
                continue
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
                executor.submit(process_single_image, image_id, images_dir, data_dir, overwrite): image_id
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
                    if qc_flag is not None:
                        qc_flags_to_add.append((image_id, qc_flag))
                except Exception as e:
                    image_id = future_to_image[future]
                    print(f"Warning: Failed to process {image_id}: {e}")
        
        # Add all QC flags in batch
        for image_id, qc_flag in qc_flags_to_add:
            try:
                flag_qc(
                    data_dir=data_dir,
                    image_ids=[image_id],
                    qc_flag=qc_flag,
                    annotator='automatic',
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
        description="Automated image quality assessment for MorphSeq pipeline"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to raw_data_organized directory"
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
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    # Load metadata
    metadata_path = data_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        print("Run 01_prepare_videos.py first to generate metadata")
        return 1
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Ensure QC directory exists
    qc_dir = data_dir / "quality_control"
    qc_dir.mkdir(exist_ok=True)
    
    # STEP 1: INITIALIZE QC CSV FROM METADATA (CRITICAL FIRST STEP)
    print("🔄 Step 1: Initializing QC system from metadata...")
    print("   → Scanning experiment_metadata.json for all images")
    print("   → Detecting new images not yet in QC system")
    print("   → Adding new entries to image_quality_qc.csv")
    
    # Initialize QC file to ensure ALL images from metadata are tracked
    qc_df_before = load_qc_data(data_dir)
    existing_count = len(qc_df_before)
    
    qc_df_after = initialize_qc_file(
        data_dir=data_dir,
        experiment_metadata_path=metadata_path,
        overwrite=False  # Preserve existing manual annotations
    )
    
    new_images_added = len(qc_df_after) - existing_count
    print(f"✅ QC initialization complete!")
    print(f"   📈 Total images now tracked: {len(qc_df_after)}")
    print(f"   ➕ New images added from metadata: {new_images_added}")
    print(f"   💾 Existing QC annotations preserved: {existing_count}")
    
    if new_images_added > 0:
        print("   📝 These new images are ready for manual/automatic QC")
    else:
        print("   ✅ All images already in image quality QC file")
    
    # STEP 2: AUTOMATIC QC PROCESSING
    print(f"\n🤖 Step 2: Automatic QC processing...")
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
            experiment_id, data_dir, metadata, 
            args.overwrite, args.verbose, args.workers
        )
        total_processed += processed
        total_skipped += skipped
    
    # STEP 3: SUMMARY AND COMPLETION
    print(f"\n🎯 Step 3: QC Processing Complete!")
    print(f"   📊 Total images processed: {total_processed}")
    print(f"   ⏭️  Total images skipped (already flagged): {total_skipped}")
    
    # Load final QC data for summary
    qc_df = load_qc_data(data_dir)
    print(f"\nQC processing complete. Total QC records: {len(qc_df)}")
    
    # Print summary
    if len(qc_df) > 0:
        print("\nQC Summary:")
        print(qc_df['qc_flag'].value_counts())
        print(f"\nAnnotator breakdown:")
        print(qc_df['annotator'].value_counts())
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
