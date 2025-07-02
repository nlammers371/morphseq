#!/usr/bin/env python
"""
02_image_quality_qc.py

Performs automated image quality assessment on processed images and maintains
a shared CSV file with QC flags. Manual QC annotations can be added to the same file.

Usage:
    python scripts/02_image_quality_qc.py \
        --data_dir /path/to/raw_data_organized \
        --experiments_to_process 20240411,20240412

QC Flags:
    - PASS: Good quality image
    - BLUR: Image is blurry (low variance of Laplacian)
    - DARK: Image is too dark (low mean brightness)
    - BRIGHT: Image is oversaturated (high mean brightness)
    - LOW_CONTRAST: Poor contrast (low standard deviation)
    - CORRUPT: Cannot read/process image
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
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from image_qc_utils import load_qc_data, save_qc_data, check_existing_qc, auto_qc, QC_FLAGS

# QC thresholds (can be adjusted based on your data)
QC_THRESHOLDS = {
    'blur_threshold': 100,      # Variance of Laplacian below this = BLUR
    'dark_threshold': 30,       # Mean brightness below this = DARK
    'bright_threshold': 220,    # Mean brightness above this = BRIGHT
    'contrast_threshold': 15,   # Std dev below this = LOW_CONTRAST
}

def calculate_image_metrics(image_path: Path) -> Dict:
    """Calculate quality metrics for an image."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {'qc_flag': 'CORRUPT', 'metrics': {}}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = gray.mean()
        contrast = gray.std()
        
        metrics = {
            'blur_score': float(blur_score),
            'mean_brightness': float(mean_brightness),
            'contrast': float(contrast),
            'width': image.shape[1],
            'height': image.shape[0]
        }
        
        # Determine QC flag
        if blur_score < QC_THRESHOLDS['blur_threshold']:
            qc_flag = 'BLUR'
        elif mean_brightness < QC_THRESHOLDS['dark_threshold']:
            qc_flag = 'DARK'
        elif mean_brightness > QC_THRESHOLDS['bright_threshold']:
            qc_flag = 'BRIGHT'
        elif contrast < QC_THRESHOLDS['contrast_threshold']:
            qc_flag = 'LOW_CONTRAST'
        else:
            qc_flag = 'PASS'
        
        return {'qc_flag': qc_flag, 'metrics': metrics}
        
    except Exception as e:
        return {'qc_flag': 'CORRUPT', 'metrics': {'error': str(e)}}

def process_experiment_qc(
    experiment_id: str,
    data_dir: Path,
    metadata: Dict,
    qc_df: pd.DataFrame,
    overwrite: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """Process QC for all images in an experiment."""
    
    if experiment_id not in metadata['experiments']:
        print(f"Warning: Experiment {experiment_id} not found in metadata")
        return qc_df
    
    experiment_data = metadata['experiments'][experiment_id]
    if verbose:
        print(f"\nProcessing QC for experiment: {experiment_id}")
    
    new_records = []
    processed_count = 0
    skipped_count = 0
    
    for video_id, video_data in experiment_data['videos'].items():
        well_id = video_data['well_id']
        image_ids = video_data.get('image_ids', [])
        
        if verbose:
            print(f"  Processing video {video_id}: {len(image_ids)} images")
        
        images_dir = Path(video_data['processed_jpg_images_dir'])
        
        for image_id in tqdm(image_ids, desc=f"QC {video_id}", disable=not verbose):
            # Check if already processed
            if not overwrite and image_id in qc_df['image_id'].values:
                skipped_count += 1
                continue
            
            # Extract timepoint from image_id (e.g., "20240411_A01_0000" -> "0000")
            timepoint = image_id.split('_')[-1]
            
            # Find the JPEG file
            jpeg_path = images_dir / f"{image_id}.jpg"
            if not jpeg_path.exists():
                print(f"Warning: JPEG not found: {jpeg_path}")
                continue
            
            # Calculate QC metrics
            qc_result = calculate_image_metrics(jpeg_path)
            
            # Create record
            record = {
                'image_id': image_id,
                'experiment_id': experiment_id,
                'video_id': video_id,
                'well_id': well_id,
                'timepoint': timepoint,
                'qc_flag': qc_result['qc_flag'],
                'annotator': 'automatic',
                'qc_timestamp': datetime.now().isoformat(),
                'notes': ''
            }
            
            # Add metrics
            record.update(qc_result['metrics'])
            new_records.append(record)
            processed_count += 1
    
    if new_records:
        new_df = pd.DataFrame(new_records)
        qc_df = pd.concat([qc_df, new_df], ignore_index=True)
        
    if verbose:
        print(f"  Processed {processed_count} images, skipped {skipped_count}")
    
    return qc_df

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
    
    # Load or create QC CSV
    qc_csv_path = data_dir / "image_quality_qc.csv"
    qc_df = load_or_create_qc_csv(qc_csv_path)
    
    # Determine experiments to process
    if args.experiments_to_process:
        experiments_to_process = [e.strip() for e in args.experiments_to_process.split(',')]
    else:
        experiments_to_process = list(metadata['experiments'].keys())
    
    print(f"Processing QC for {len(experiments_to_process)} experiments")
    
    # Process each experiment
    for experiment_id in experiments_to_process:
        qc_df = process_experiment_qc(
            experiment_id, data_dir, metadata, qc_df, 
            args.overwrite, args.verbose
        )
    
    # Save updated QC data
    qc_df.to_csv(qc_csv_path, index=False)
    print(f"\nSaved QC data to: {qc_csv_path}")
    
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
