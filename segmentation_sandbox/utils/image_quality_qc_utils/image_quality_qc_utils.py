#!/usr/bin/env python
"""
image_qc_utils.py

⚠️  DEPRECATED LEGACY MODULE ⚠️

This CSV-based QC system has been replaced with a hierarchical JSON-based system.

🔄 MIGRATION GUIDE:
===================

OLD (this file):                    NEW (use instead):
------------------                   -------------------
image_quality_qc.csv                experiment_data_qc.json  
utils/image_quality_qc_utils/        scripts/experiment_data_qc_utils.py
QC_FLAGS                             VALID_QC_FLAG_CATEGORIES
validate_qc_flag(flag)               validate_qc_flag(flag, level, qc_data)
flag_qc(image_ids, flag, ...)        flag_image(image_id, flag, author, ...)

🏗️  NEW FEATURES:
- Multi-level QC (experiment/video/image/embryo)
- Author tracking with timestamps
- Hierarchical organization mirroring metadata
- Better validation and error handling
- Single JSON file instead of CSV

📍 NEW LOCATION: scripts/experiment_data_qc_utils.py

For compatibility, basic functions are provided below, but new code should use
the new hierarchical system in scripts/experiment_data_qc_utils.py
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import warnings

# Issue deprecation warning
warnings.warn(
    "image_quality_qc_utils is deprecated. Use scripts/experiment_data_qc_utils.py instead. "
    "See migration guide in module docstring.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy QC flags for backward compatibility (image-level only)
QC_FLAGS = {
    'BLUR': 'Image is blurry (low variance of Laplacian)',
    'DARK': 'Image is too dark (low mean brightness)',
    'BRIGHT': 'Image is oversaturated (high mean brightness)', 
    'LOW_CONTRAST': 'Poor contrast (low standard deviation)',
    'CORRUPT': 'Cannot read/process image',
    'ARTIFACT': 'Contains artifacts or technical issues',
    'OUT_OF_FOCUS': 'Image is out of focus',
    'DEBRIS': 'Contains debris affecting analysis',
    'MANUAL_REJECT': 'Manual fail flag (general quality issue)',
    'FAIL': 'General failure flag'
}

def get_qc_csv_path(data_dir: Union[str, Path]) -> Path:
    """LEGACY: Get path to QC CSV file. Use experiment_data_qc.json instead."""
    data_parent = Path(data_dir).parent
    return data_parent / "quality_control" / "image_quality_qc.csv"

def validate_qc_flag(qc_flag: str) -> None:
    """LEGACY: Validate QC flag. Use experiment_data_qc_utils.validate_qc_flag(flag, level, qc_data) instead."""
    if qc_flag not in QC_FLAGS:
        valid_flags = list(QC_FLAGS.keys())
        raise ValueError(f"Invalid QC flag '{qc_flag}'. Valid flags: {valid_flags}")

def load_qc_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    """LEGACY: Load QC data as DataFrame. Use experiment_data_qc_utils.load_qc_data() instead."""
    qc_csv_path = get_qc_csv_path(data_dir)
    
    if qc_csv_path.exists():
        df = pd.read_csv(qc_csv_path)
        print(f"Loaded legacy QC data: {len(df)} records")
    else:
        df = pd.DataFrame(columns=[
            'experiment_id', 'video_id', 'image_id', 'qc_flag', 'notes', 'annotator'
        ])
        print("Created empty legacy QC DataFrame")
    
    return df

def save_qc_data(qc_df: pd.DataFrame, data_dir: Union[str, Path]) -> None:
    """LEGACY: Save QC DataFrame to CSV. Use experiment_data_qc_utils.save_qc_data() instead."""
    qc_csv_path = get_qc_csv_path(data_dir)
    qc_csv_path.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_csv(qc_csv_path, index=False)
    print(f"Saved legacy QC data to: {qc_csv_path}")

def parse_image_id(image_id: str) -> tuple:
    """LEGACY: Parse image_id. Use experiment_data_qc_utils.parse_image_id() instead."""
    parts = image_id.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid image_id format: {image_id}")
    
    experiment_id = parts[0]
    well_id = parts[1] 
    timepoint = parts[2]
    video_id = f"{experiment_id}_{well_id}"
    
    return experiment_id, video_id, well_id, timepoint

# Minimal legacy functions for critical compatibility
def flag_qc(data_dir: Union[str, Path], image_ids: List[str], qc_flag: str, 
           annotator: str = 'manual', notes: str = '', overwrite: bool = False) -> pd.DataFrame:
    """LEGACY: Flag images. Use experiment_data_qc_utils.flag_image() instead."""
    print("⚠️  WARNING: Using deprecated flag_qc(). Migrate to experiment_data_qc_utils.flag_image()")
    
    validate_qc_flag(qc_flag)
    qc_df = load_qc_data(data_dir)
    
    new_records = []
    for image_id in image_ids:
        exp_id, vid_id, well_id, timepoint = parse_image_id(image_id)
        
        # Check for existing records
        mask = (qc_df['image_id'] == image_id)
        if mask.any() and not overwrite:
            continue
        
        # Remove existing if overwriting
        if overwrite:
            qc_df = qc_df[~mask]
        
        new_records.append({
            'experiment_id': exp_id,
            'video_id': vid_id,
            'image_id': image_id,
            'qc_flag': qc_flag,
            'notes': notes,
            'annotator': annotator
        })
    
    if new_records:
        new_df = pd.DataFrame(new_records)
        qc_df = pd.concat([qc_df, new_df], ignore_index=True)
    
    save_qc_data(qc_df, data_dir)
    return qc_df

def get_qc_summary(data_dir: Union[str, Path]) -> Dict:
    """LEGACY: Get QC summary. Use experiment_data_qc_utils.get_qc_summary() instead."""
    qc_df = load_qc_data(data_dir)
    
    if len(qc_df) == 0:
        return {"total_images": 0, "qc_flags": {}, "annotators": {}}
    
    return {
        "total_images": len(qc_df),
        "qc_flags": qc_df['qc_flag'].value_counts().to_dict(),
        "annotators": qc_df['annotator'].value_counts().to_dict(),
        "experiments": qc_df['experiment_id'].nunique(),
        "videos": qc_df['video_id'].nunique()
    }

def check_existing_qc(data_dir: Union[str, Path], image_ids: List[str]) -> Dict[str, str]:
    """LEGACY: Check existing QC. Use experiment_data_qc_utils.get_qc_flags() instead."""
    qc_df = load_qc_data(data_dir)
    
    result = {}
    for image_id in image_ids:
        existing = qc_df[qc_df['image_id'] == image_id]
        if len(existing) > 0:
            result[image_id] = existing.iloc[0]['qc_flag']
        else:
            result[image_id] = ""
    
    return result

# Additional legacy compatibility functions
def initialize_qc_file(data_dir: Union[str, Path], experiment_metadata_path: Optional[Union[str, Path]] = None, overwrite: bool = False) -> pd.DataFrame:
    """LEGACY: Use experiment_data_qc_utils.initialize_qc_structure_from_metadata() instead."""
    print("⚠️  WARNING: Using deprecated initialize_qc_file(). Migrate to initialize_qc_structure_from_metadata()")
    
    if experiment_metadata_path is None:
        experiment_metadata_path = Path(data_dir) / "experiment_metadata.json"
    
    with open(experiment_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    all_images = []
    for exp_id, exp_data in metadata['experiments'].items():
        for video_id, video_data in exp_data['videos'].items():
            for image_id in video_data.get('image_ids', []):
                all_images.append({
                    'experiment_id': exp_id,
                    'video_id': video_id,
                    'image_id': image_id,
                    'qc_flag': None,
                    'notes': None,
                    'annotator': None
                })
    
    qc_df = pd.DataFrame(all_images)
    save_qc_data(qc_df, data_dir)
    return qc_df

# Additional convenience functions for compatibility
def get_flagged_images(qc_df: pd.DataFrame, exclude_none: bool = True) -> List[str]:
    """LEGACY: Get flagged images."""
    if exclude_none:
        flagged = qc_df.dropna(subset=['qc_flag'])
    else:
        flagged = qc_df.copy()
    return flagged['image_id'].tolist()

def get_unflagged_images(qc_df: pd.DataFrame) -> List[str]:
    """LEGACY: Get unflagged images."""
    unflagged = qc_df[qc_df['qc_flag'].isna()]
    return unflagged['image_id'].tolist()

def manual_qc(data_dir: Union[str, Path], annotator: str, **kwargs) -> pd.DataFrame:
    """LEGACY: Manual QC wrapper."""
    return flag_qc(data_dir=data_dir, annotator=annotator, **kwargs)

def auto_qc(data_dir: Union[str, Path], **kwargs) -> pd.DataFrame:
    """LEGACY: Auto QC wrapper."""
    return flag_qc(data_dir=data_dir, annotator='auto', **kwargs)
    """
    Load or create QC CSV file with correct column structure.
    
    Args:
        qc_csv_path: Path to the QC CSV file
        
    Returns:
        DataFrame with QC data
    """
    qc_csv_path = Path(qc_csv_path)
    
    if qc_csv_path.exists():
        df = pd.read_csv(qc_csv_path)
        print(f"Loaded existing QC data: {len(df)} records")
    else:
        # Create with correct column structure matching specifications
        df = pd.DataFrame(columns=[
            'experiment_id', 'video_id', 'image_id', 'qc_flag', 'notes', 'annotator'
        ])
        # Create directory if it doesn't exist
        qc_csv_path.parent.mkdir(parents=True, exist_ok=True)
        print("Created new QC DataFrame")
    
    return df

def load_qc_data(data_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load the QC CSV file. Creates a new empty DataFrame if file doesn't exist.
    
    Args:
        data_dir: Directory containing the QC CSV file
        
    Returns:
        DataFrame with QC data
    """
    qc_csv_path = get_qc_csv_path(data_dir)
    
    if qc_csv_path.exists():
        df = pd.read_csv(qc_csv_path)
        print(f"Loaded existing QC data: {len(df)} records")
    else:
        df = pd.DataFrame(columns=[
            'experiment_id', 'video_id', 'image_id', 'qc_flag', 'notes', 'annotator'
        ])
        print("Created new QC DataFrame")
    
    return df

def save_qc_data(qc_df: pd.DataFrame, data_dir: Union[str, Path]) -> None:
    """
    Save QC DataFrame to CSV file.
    
    Args:
        qc_df: QC DataFrame to save
        data_dir: Directory to save the QC CSV file
    """
    qc_csv_path = get_qc_csv_path(data_dir)
    qc_csv_path.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_csv(qc_csv_path, index=False)
    print(f"Saved QC data to: {qc_csv_path}")

def parse_image_id(image_id: str) -> tuple:
    """
    Parse image_id to extract experiment_id, video_id, well_id, and timepoint.
    
    Expected format: "YYYYMMDD_WELL_tXXX" (e.g., "20241215_A01_t001")
    
    Args:
        image_id: Image ID string
        
    Returns:
        tuple: (experiment_id, video_id, well_id, timepoint)
    """
    parts = image_id.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid image_id format: {image_id}")
    
    experiment_id = parts[0]
    well_id = parts[1] 
    timepoint = parts[2]
    video_id = f"{experiment_id}_{well_id}"
    
    return experiment_id, video_id, well_id, timepoint

def validate_qc_flag(qc_flag: str) -> None:
    """
    Validate that QC flag is one of the standard types.
    Note: None is allowed in the CSV but not for explicit flagging operations.
    
    Args:
        qc_flag: QC flag to validate
        
    Raises:
        ValueError: If flag is not valid
    """
    if qc_flag not in QC_FLAGS:
        valid_flags = list(QC_FLAGS.keys())
        raise ValueError(f"Invalid QC flag '{qc_flag}'. Valid flags: {valid_flags}")

def flag_qc(
    data_dir: Union[str, Path],
    image_ids: Optional[List[str]] = None,
    video_id: Optional[str] = None,
    frames: Optional[List[str]] = None,
    qc_flag: str = 'FAIL',
    annotator: str = '',
    notes: str = '',
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Add QC flags for specified problematic images.
    
    This function is for FLAGGING images with quality issues.
    Images without flags (qc_flag=None) are assumed to be good quality.
    
    Args:
        data_dir: Directory containing QC CSV
        image_ids: List of image IDs to flag (alternative to video_id + frames)
        video_id: Video ID (use with frames parameter)
        frames: List of frame/timepoint identifiers (use with video_id)
        qc_flag: QC flag to assign (must be in QC_FLAGS, no None allowed here)
        annotator: Who is adding the flag (required)
        notes: Optional notes about the QC decision
        overwrite: Whether to overwrite existing QC entries
        
    Returns:
        Updated QC DataFrame
        
    Raises:
        ValueError: For invalid inputs or duplicate entries without overwrite
    """
    if not annotator:
        raise ValueError("Annotator must be provided")
    
    validate_qc_flag(qc_flag)
    
    # Load existing QC data
    qc_df = load_qc_data(data_dir)
    
    # Build list of images to process
    to_process = []
    
    if image_ids:
        for image_id in image_ids:
            exp_id, vid_id, well_id, timepoint = parse_image_id(image_id)
            to_process.append((exp_id, vid_id, well_id, timepoint, image_id))
    elif video_id and frames:
        # Extract experiment_id from video_id (first part before underscore)
        exp_id = video_id.split('_')[0]
        well_id = '_'.join(video_id.split('_')[1:])  # Handle multi-part well IDs
        for frame in frames:
            image_id = f"{video_id}_{frame}"
            to_process.append((exp_id, video_id, well_id, frame, image_id))
    else:
        raise ValueError("Must provide either image_ids OR (video_id + frames)")
    
    # Process each image
    new_records = []
    updated_count = 0
    
    for exp_id, vid_id, well_id, timepoint, image_id in to_process:
        # Check if record already exists
        mask = (qc_df['image_id'] == image_id)
        existing_records = qc_df[mask]
        
        if len(existing_records) > 0:
            if not overwrite:
                raise ValueError(f"QC record exists for {image_id}; use overwrite=True to replace")
            else:
                # Remove existing records
                qc_df = qc_df[~mask]
                updated_count += 1
        
        # Create new record
        record = {
            'experiment_id': exp_id,
            'video_id': vid_id,
            'image_id': image_id,
            'qc_flag': qc_flag,
            'notes': notes,
            'annotator': annotator
        }
        new_records.append(record)
    
    # Add new records
    if new_records:
        new_df = pd.DataFrame(new_records)
        qc_df = pd.concat([qc_df, new_df], ignore_index=True)
    
    # Save updated data
    save_qc_data(qc_df, data_dir)
    
    print(f"Added QC flags for {len(new_records)} images")
    if updated_count > 0:
        print(f"Updated {updated_count} existing records")
    
    return qc_df

def remove_qc(
    data_dir: Union[str, Path],
    image_ids: Optional[List[str]] = None,
    video_id: Optional[str] = None,
    frames: Optional[List[str]] = None,
    annotator: str = ''
) -> pd.DataFrame:
    """
    Remove QC flags for specified images.
    
    Args:
        data_dir: Directory containing QC CSV
        image_ids: List of image IDs to remove flags from
        video_id: Video ID (use with frames parameter)
        frames: List of frame/timepoint identifiers (use with video_id)
        annotator: Who is removing the flags (required for logging)
        
    Returns:
        Updated QC DataFrame
    """
    if not annotator:
        raise ValueError("Annotator must be provided for QC removal")
    
    # Load existing QC data
    qc_df = load_qc_data(data_dir)
    
    # Build list of images to process
    to_remove = []
    
    if image_ids:
        to_remove = image_ids
    elif video_id and frames:
        for frame in frames:
            image_id = f"{video_id}_{frame}"
            to_remove.append(image_id)
    else:
        raise ValueError("Must provide either image_ids OR (video_id + frames)")
    
    # Remove records
    initial_count = len(qc_df)
    qc_df = qc_df[~qc_df['image_id'].isin(to_remove)]
    removed_count = initial_count - len(qc_df)
    
    # Save updated data
    save_qc_data(qc_df, data_dir)
    
    print(f"Removed QC flags for {removed_count} images")
    
    return qc_df

def get_qc_summary(data_dir: Union[str, Path]) -> Dict:
    """
    Get summary statistics for QC data.
    
    Args:
        data_dir: Directory containing QC CSV
        
    Returns:
        Dictionary with QC statistics
    """
    qc_df = load_qc_data(data_dir)
    
    if len(qc_df) == 0:
        return {"total_images": 0, "qc_flags": {}, "annotators": {}}
    
    summary = {
        "total_images": len(qc_df),
        "qc_flags": qc_df['qc_flag'].value_counts().to_dict(),
        "annotators": qc_df['annotator'].value_counts().to_dict(),
        "experiments": qc_df['experiment_id'].nunique(),
        "videos": qc_df['video_id'].nunique()
    }
    
    return summary

def check_existing_qc(
    data_dir: Union[str, Path],
    image_ids: List[str]
) -> Dict[str, str]:
    """
    Check which images already have QC flags.
    
    Args:
        data_dir: Directory containing QC CSV
        image_ids: List of image IDs to check
        
    Returns:
        Dictionary mapping image_id to existing qc_flag (empty if no flag)
    """
    qc_df = load_qc_data(data_dir)
    
    result = {}
    for image_id in image_ids:
        existing = qc_df[qc_df['image_id'] == image_id]
        if len(existing) > 0:
            result[image_id] = existing.iloc[0]['qc_flag']
        else:
            result[image_id] = ""
    
    return result

def generate_comprehensive_qc(
    data_dir: Union[str, Path],
    experiment_metadata_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Generate comprehensive QC file with all images.
    Images default to None (no flag) unless they have specific quality issues.

    QC file naming updated to use folder prefix for consistency: image_quality_qc.comprehensive.csv
    """
    if experiment_metadata_path is None:
        experiment_metadata_path = Path(data_dir) / "experiment_metadata.json"
    
    # Load experiment metadata
    with open(experiment_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load existing QC data
    qc_df = load_qc_data(data_dir)
    
    # Build comprehensive list of all images
    all_images = []
    for exp_id, exp_data in metadata['experiments'].items():
        for video_id, video_data in exp_data['videos'].items():
            for image_id in video_data.get('image_ids', []):
                all_images.append({
                    'experiment_id': exp_id,
                    'video_id': video_id,
                    'image_id': image_id,
                    'qc_flag': None,  # Default to None (no flag = good quality)
                    'notes': None,    # No notes by default
                    'annotator': None # No annotator by default
                })
    
    # Create comprehensive DataFrame
    comprehensive_df = pd.DataFrame(all_images)
    
    # Update with actual QC flags (only for images that have been flagged)
    for _, qc_row in qc_df.iterrows():
        mask = comprehensive_df['image_id'] == qc_row['image_id']
        if mask.any():
            comprehensive_df.loc[mask, 'qc_flag'] = qc_row['qc_flag']
            comprehensive_df.loc[mask, 'annotator'] = qc_row['annotator']
            comprehensive_df.loc[mask, 'notes'] = qc_row['notes']
    
    # Save comprehensive QC file
    comprehensive_path = Path(data_dir).parent / "quality_control" / "comprehensive_image_qc.csv"
    comprehensive_df.to_csv(comprehensive_path, index=False)
    print(f"Saved comprehensive QC data to: {comprehensive_path}")
    
    return comprehensive_df

def initialize_qc_file(
    data_dir: Union[str, Path],
    experiment_metadata_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Initialize or update the QC CSV file from experiment metadata.
    Creates entries for all images with empty qc_flag, notes, and annotator fields.
    
    Args:
        data_dir: Directory containing QC CSV
        experiment_metadata_path: Path to experiment metadata JSON file
        overwrite: Whether to overwrite existing QC file
        
    Returns:
        Initialized QC DataFrame
    """
    if experiment_metadata_path is None:
        experiment_metadata_path = Path(data_dir) / "experiment_metadata.json"
    
    if not Path(experiment_metadata_path).exists():
        raise FileNotFoundError(f"Experiment metadata file not found: {experiment_metadata_path}")
    
    # Check if QC file already exists
    qc_csv_path = get_qc_csv_path(data_dir)
    if qc_csv_path.exists() and not overwrite:
        existing_df = load_qc_data(data_dir)
        if len(existing_df) > 0:
            print(f"QC file already exists with {len(existing_df)} entries. Use overwrite=True to recreate.")
            return existing_df
    
    # Load experiment metadata
    with open(experiment_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("Initializing QC file from experiment metadata...")
    
    # Build list of all images
    all_images = []
    for exp_id, exp_data in metadata['experiments'].items():
        for video_id, video_data in exp_data['videos'].items():
            for image_id in video_data.get('image_ids', []):
                all_images.append({
                    'experiment_id': exp_id,
                    'video_id': video_id,
                    'image_id': image_id,
                    'qc_flag': None,  # No QC flag initially
                    'notes': None,    # No notes initially
                    'annotator': None # No annotator initially
                })
    
    # Create DataFrame
    qc_df = pd.DataFrame(all_images)
    
    # Save to CSV
    save_qc_data(qc_df, data_dir)
    
    print(f"Initialized QC file with {len(qc_df)} images from {len(metadata['experiments'])} experiments")
    return qc_df

# Convenience functions for analyzing QC data
def get_flagged_images(qc_df: pd.DataFrame, exclude_none: bool = True) -> List[str]:
    """
    Get list of image IDs that have QC flags.
    
    Args:
        qc_df: QC DataFrame
        exclude_none: Whether to exclude images with no flag (None/NaN)
        
    Returns:
        List of image IDs with QC flags
    """
    if exclude_none:
        flagged = qc_df.dropna(subset=['qc_flag'])
    else:
        flagged = qc_df.copy()
    
    return flagged['image_id'].tolist()

def get_unflagged_images(qc_df: pd.DataFrame) -> List[str]:
    """
    Get list of image IDs that do not have QC flags (qc_flag is None/NaN).
    
    Args:
        qc_df: QC DataFrame
        
    Returns:
        List of image IDs without QC flags
    """
    unflagged = qc_df[qc_df['qc_flag'].isna()]
    return unflagged['image_id'].tolist()

def get_all_qc_data(qc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get all QC data (convenience function for consistency).
    
    Args:
        qc_df: QC DataFrame
        
    Returns:
        The same QC DataFrame
    """
    return qc_df

def get_images_by_flag(qc_df: pd.DataFrame, qc_flag: str) -> List[str]:
    """
    Get list of image IDs with a specific QC flag.
    
    Args:
        qc_df: QC DataFrame
        qc_flag: QC flag to filter by
        
    Returns:
        List of image IDs with the specified flag
    """
    filtered = qc_df[qc_df['qc_flag'] == qc_flag]
    return filtered['image_id'].tolist()

def get_images_by_annotator(qc_df: pd.DataFrame, annotator: str) -> List[str]:
    """
    Get list of image IDs flagged by a specific annotator.
    
    Args:
        qc_df: QC DataFrame
        annotator: Annotator name to filter by
        
    Returns:
        List of image IDs flagged by the specified annotator
    """
    filtered = qc_df[qc_df['annotator'] == annotator]
    return filtered['image_id'].tolist()

# Convenience wrappers for manual and automatic QC
def manual_qc(data_dir: Union[str, Path], annotator: str, **kwargs) -> pd.DataFrame:
    """Wrapper for flag_qc with manual annotator specified."""
    return flag_qc(data_dir=data_dir, annotator=annotator, **kwargs)

def auto_qc(data_dir: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Wrapper for flag_qc with automatic annotator."""
    return flag_qc(data_dir=data_dir, annotator='auto', **kwargs)
