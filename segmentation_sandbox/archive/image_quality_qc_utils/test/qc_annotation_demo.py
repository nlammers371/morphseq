#!/usr/bin/env python
"""
QC Annotation Demo - Manual and Automatic Examples

This script demonstrates how to use the image quality control utilities
for both manual and automatic annotations. Each section is designed to be
run as a separate block in a Jupyter notebook.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Setup - Add the parent directory to path for imports
current_dir = Path(__file__).parent.resolve() if '__file__' in globals() else Path.cwd()
utils_dir = current_dir.parent if 'test' in str(current_dir) else current_dir
sys.path.insert(0, str(utils_dir))

from image_quality_qc_utils import (
    load_qc_data, save_qc_data, flag_qc, remove_qc, 
    get_qc_summary, get_flagged_images, get_unflagged_images,
    get_images_by_flag, get_images_by_annotator, QC_FLAGS,
    manual_qc, auto_qc
)

# =============================================================================
# BLOCK 1: Setup and Initialize Test Data
# =============================================================================
def block1_setup():
    """Initialize test QC data for demonstration."""
    print("=== BLOCK 1: Setup Test Data ===")
    
    # Create a test QC DataFrame with sample data
    test_data = [
        {
            'experiment_id': '20250101',
            'video_id': '20250101_A01',
            'image_id': '20250101_A01_t001',
            'qc_flag': None,
            'notes': None,
            'annotator': None
        },
        {
            'experiment_id': '20250101',
            'video_id': '20250101_A01',
            'image_id': '20250101_A01_t002',
            'qc_flag': None,
            'notes': None,
            'annotator': None
        },
        {
            'experiment_id': '20250101',
            'video_id': '20250101_B01',
            'image_id': '20250101_B01_t001',
            'qc_flag': None,
            'notes': None,
            'annotator': None
        },
        {
            'experiment_id': '20250102',
            'video_id': '20250102_A01',
            'image_id': '20250102_A01_t001',
            'qc_flag': None,
            'notes': None,
            'annotator': None
        }
    ]
    
    qc_df = pd.DataFrame(test_data)
    
    # Save to test directory
    test_dir = Path('test')
    test_dir.mkdir(exist_ok=True)
    qc_csv_path = test_dir / 'demo_qc.csv'
    qc_df.to_csv(qc_csv_path, index=False)
    
    print(f"Created test QC file with {len(qc_df)} images at: {qc_csv_path}")
    print("Initial QC data:")
    print(qc_df.to_string(index=False))
    
    return qc_df, qc_csv_path

# =============================================================================
# BLOCK 2: Manual QC Annotations
# =============================================================================
def block2_manual_qc(qc_csv_path):
    """Demonstrate manual QC flagging."""
    print("\n=== BLOCK 2: Manual QC Annotations ===")
    
    # Load the QC data by creating a fake data_dir that points to our test CSV
    qc_df = pd.read_csv(qc_csv_path)
    
    print("Available QC flags:")
    for flag, description in QC_FLAGS.items():
        print(f"  {flag}: {description}")
    
    # Example 1: Flag a single image as BLUR
    print("\n1. Flagging image as BLUR...")
    test_record = {
        'experiment_id': '20250101',
        'video_id': '20250101_A01',
        'image_id': '20250101_A01_t001',
        'qc_flag': 'BLUR',
        'notes': 'Manual review - image appears blurry',
        'annotator': 'mcolon'
    }
    
    # Add the manual flag
    mask = qc_df['image_id'] == test_record['image_id']
    qc_df.loc[mask, 'qc_flag'] = test_record['qc_flag']
    qc_df.loc[mask, 'notes'] = test_record['notes']
    qc_df.loc[mask, 'annotator'] = test_record['annotator']
    
    # Example 2: Flag multiple images by video
    print("\n2. Flagging frames in a video as DARK...")
    video_frames = ['20250101_A01_t002']
    for image_id in video_frames:
        mask = qc_df['image_id'] == image_id
        qc_df.loc[mask, 'qc_flag'] = 'DARK'
        qc_df.loc[mask, 'notes'] = 'Video frames too dark for analysis'
        qc_df.loc[mask, 'annotator'] = 'mcolon'
    
    # Save updated data
    qc_df.to_csv(qc_csv_path, index=False)
    
    print("Updated QC data after manual annotations:")
    flagged_data = qc_df.dropna(subset=['qc_flag'])
    print(flagged_data.to_string(index=False))
    
    return qc_df

# =============================================================================
# BLOCK 3: Automatic QC Annotations
# =============================================================================
def block3_automatic_qc(qc_csv_path):
    """Demonstrate automatic QC flagging."""
    print("\n=== BLOCK 3: Automatic QC Annotations ===")
    
    qc_df = pd.read_csv(qc_csv_path)
    
    # Simulate automatic blur detection
    print("1. Simulating automatic blur detection...")
    
    # Flag an image automatically based on blur metric
    auto_blur_image = '20250101_B01_t001'
    mask = qc_df['image_id'] == auto_blur_image
    qc_df.loc[mask, 'qc_flag'] = 'BLUR'
    qc_df.loc[mask, 'notes'] = 'Automatic: Laplacian variance < threshold (blur_threshold=100)'
    qc_df.loc[mask, 'annotator'] = 'automatic'
    
    # Simulate automatic brightness detection
    print("2. Simulating automatic brightness detection...")
    auto_dark_image = '20250102_A01_t001'
    mask = qc_df['image_id'] == auto_dark_image
    qc_df.loc[mask, 'qc_flag'] = 'DARK'
    qc_df.loc[mask, 'notes'] = 'Automatic: Mean brightness < threshold (brightness_threshold=50)'
    qc_df.loc[mask, 'annotator'] = 'automatic'
    
    # Save updated data
    qc_df.to_csv(qc_csv_path, index=False)
    
    print("Updated QC data after automatic annotations:")
    auto_flagged = qc_df[qc_df['annotator'] == 'automatic']
    print(auto_flagged.to_string(index=False))
    
    return qc_df

# =============================================================================
# BLOCK 4: Analyzing QC Data
# =============================================================================
def block4_analyze_qc(qc_csv_path):
    """Demonstrate QC data analysis functions."""
    print("\n=== BLOCK 4: Analyzing QC Data ===")
    
    qc_df = pd.read_csv(qc_csv_path)
    
    # Get flagged images
    flagged_images = get_flagged_images(qc_df)
    print(f"Flagged images (excluding PASS): {len(flagged_images)}")
    print(f"  {flagged_images}")
    
    # Get unflagged images
    unflagged_images = get_unflagged_images(qc_df)
    print(f"\nUnflagged images: {len(unflagged_images)}")
    print(f"  {unflagged_images}")
    
    # Get images by specific flag
    blur_images = get_images_by_flag(qc_df, 'BLUR')
    print(f"\nImages flagged as BLUR: {len(blur_images)}")
    print(f"  {blur_images}")
    
    # Get images by annotator
    manual_images = get_images_by_annotator(qc_df, 'mcolon')
    auto_images = get_images_by_annotator(qc_df, 'automatic')
    print(f"\nManually flagged images: {len(manual_images)}")
    print(f"  {manual_images}")
    print(f"\nAutomatically flagged images: {len(auto_images)}")
    print(f"  {auto_images}")
    
    # QC summary
    print("\nQC Summary by flag:")
    flag_counts = qc_df['qc_flag'].value_counts()
    print(flag_counts)
    
    print("\nQC Summary by annotator:")
    annotator_counts = qc_df['annotator'].value_counts()
    print(annotator_counts)
    
    return qc_df

# =============================================================================
# BLOCK 5: Removing and Updating QC Flags
# =============================================================================
def block5_remove_update_qc(qc_csv_path):
    """Demonstrate removing and updating QC flags."""
    print("\n=== BLOCK 5: Removing and Updating QC Flags ===")
    
    qc_df = pd.read_csv(qc_csv_path)
    
    print("Current flagged images:")
    flagged_before = qc_df.dropna(subset=['qc_flag'])
    print(flagged_before[['image_id', 'qc_flag', 'annotator']].to_string(index=False))
    
    # Remove a QC flag
    print("\n1. Removing QC flag from one image...")
    image_to_clear = '20250101_A01_t002'
    mask = qc_df['image_id'] == image_to_clear
    qc_df.loc[mask, 'qc_flag'] = None
    qc_df.loc[mask, 'notes'] = None
    qc_df.loc[mask, 'annotator'] = None
    
    # Update an existing flag
    print("2. Updating an existing QC flag...")
    image_to_update = '20250101_A01_t001'
    mask = qc_df['image_id'] == image_to_update
    qc_df.loc[mask, 'qc_flag'] = 'OUT_OF_FOCUS'
    qc_df.loc[mask, 'notes'] = 'Manual review - changed from BLUR to OUT_OF_FOCUS'
    qc_df.loc[mask, 'annotator'] = 'mcolon'
    
    # Save updated data
    qc_df.to_csv(qc_csv_path, index=False)
    
    print("\nUpdated flagged images:")
    flagged_after = qc_df.dropna(subset=['qc_flag'])
    print(flagged_after[['image_id', 'qc_flag', 'annotator']].to_string(index=False))
    
    return qc_df

# =============================================================================
# BLOCK 6: Final Summary and Cleanup
# =============================================================================
def block6_summary(qc_csv_path):
    """Final summary of all QC operations."""
    print("\n=== BLOCK 6: Final Summary ===")
    
    qc_df = pd.read_csv(qc_csv_path)
    
    print("Final QC data:")
    print(qc_df.to_string(index=False))
    
    print(f"\nTotal images: {len(qc_df)}")
    print(f"Flagged images: {len(qc_df.dropna(subset=['qc_flag']))}")
    print(f"Unflagged images: {len(qc_df[qc_df['qc_flag'].isna()])}")
    
    print("\nFlag distribution:")
    flag_dist = qc_df['qc_flag'].value_counts()
    print(flag_dist)
    
    print("\nAnnotator distribution:")
    annotator_dist = qc_df['annotator'].value_counts()
    print(annotator_dist)
    
    return qc_df

# =============================================================================
# Main execution (for script mode)
# =============================================================================
if __name__ == "__main__":
    print("QC Annotation Demo - Running all blocks")
    
    # Run all blocks in sequence
    qc_df, qc_csv_path = block1_setup()
    qc_df = block2_manual_qc(qc_csv_path)
    qc_df = block3_automatic_qc(qc_csv_path)
    qc_df = block4_analyze_qc(qc_csv_path)
    qc_df = block5_remove_update_qc(qc_csv_path)
    qc_df = block6_summary(qc_csv_path)
    
    print(f"\nDemo complete! Check the results in: {qc_csv_path}")
