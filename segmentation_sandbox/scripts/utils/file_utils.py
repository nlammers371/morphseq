#!/usr/bin/env python
"""
File and directory utilities for the MorphSeq embryo segmentation pipeline.
Handles file I/O, path management, and data serialization.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_copy_file(src: Union[str, Path], dst: Union[str, Path], 
                   overwrite: bool = False) -> bool:
    """
    Safely copy a file with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if successful, False otherwise
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        print(f"Source file does not exist: {src_path}")
        return False
    
    if dst_path.exists() and not overwrite:
        print(f"Destination file exists and overwrite=False: {dst_path}")
        return False
    
    try:
        # Ensure destination directory exists
        ensure_directory(dst_path.parent)
        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"Error copying {src_path} to {dst_path}: {e}")
        return False


def save_json(data: Dict[str, Any], filepath: Union[str, Path], 
              indent: int = 2) -> bool:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False


def load_json(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file with error handling.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data or None if error
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"JSON file does not exist: {filepath}")
            return None
            
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None


def save_pickle(data: Any, filepath: Union[str, Path]) -> bool:
    """
    Save data to pickle file with error handling.
    
    Args:
        data: Data to save
        filepath: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving pickle to {filepath}: {e}")
        return False


def load_pickle(filepath: Union[str, Path]) -> Optional[Any]:
    """
    Load data from pickle file with error handling.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data or None if error
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Pickle file does not exist: {filepath}")
            return None
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle from {filepath}: {e}")
        return None


def save_dataframe(df: pd.DataFrame, filepath: Union[str, Path], 
                   format: str = 'csv') -> bool:
    """
    Save DataFrame to file with error handling.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
        format: File format ('csv', 'parquet', 'excel')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format.lower() == 'excel':
            df.to_excel(filepath, index=False)
        else:
            print(f"Unsupported format: {format}")
            return False
            
        return True
    except Exception as e:
        print(f"Error saving DataFrame to {filepath}: {e}")
        return False


def load_dataframe(filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from file with error handling.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded DataFrame or None if error
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"DataFrame file does not exist: {filepath}")
            return None
        
        suffix = filepath.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(filepath)
        elif suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        else:
            print(f"Unsupported file format: {suffix}")
            return None
            
    except Exception as e:
        print(f"Error loading DataFrame from {filepath}: {e}")
        return None


def get_video_files(directory: Union[str, Path], 
                   extensions: List[str] = ['.mp4', '.avi', '.mov']) -> List[Path]:
    """
    Get list of video files in directory.
    
    Args:
        directory: Directory to search
        extensions: Video file extensions to look for
        
    Returns:
        List of video file paths
    """
    directory = Path(directory)
    video_files = []
    
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return video_files
    
    for ext in extensions:
        video_files.extend(directory.glob(f"*{ext}"))
        video_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(video_files)


def get_image_files(directory: Union[str, Path],
                   extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']) -> List[Path]:
    """
    Get list of image files in directory.
    
    Args:
        directory: Directory to search
        extensions: Image file extensions to look for
        
    Returns:
        List of image file paths
    """
    directory = Path(directory)
    image_files = []
    
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return image_files
    
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def create_timestamp_filename(base_name: str, extension: str = '.json') -> str:
    """
    Create filename with timestamp.
    
    Args:
        base_name: Base filename
        extension: File extension
        
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def backup_file(filepath: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> bool:
    """
    Create backup of existing file.
    
    Args:
        filepath: File to backup
        backup_dir: Directory for backup (default: same dir with .backup suffix)
        
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return True  # Nothing to backup
    
    if backup_dir is None:
        backup_path = filepath.with_suffix(filepath.suffix + '.backup')
    else:
        backup_dir = Path(backup_dir)
        ensure_directory(backup_dir)
        backup_path = backup_dir / filepath.name
    
    return safe_copy_file(filepath, backup_path, overwrite=True)


def clean_old_files(directory: Union[str, Path], pattern: str = "*.backup", 
                   max_age_days: int = 7) -> int:
    """
    Clean old files matching pattern.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        max_age_days: Maximum age in days
        
    Returns:
        Number of files cleaned
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    current_time = datetime.now().timestamp()
    max_age_seconds = max_age_days * 24 * 3600
    
    files_cleaned = 0
    for file_path in directory.glob(pattern):
        try:
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                files_cleaned += 1
        except Exception as e:
            print(f"Error cleaning file {file_path}: {e}")
    
    return files_cleaned


def validate_file_structure(required_files: List[Union[str, Path]], 
                          base_dir: Optional[Union[str, Path]] = None) -> Dict[str, bool]:
    """
    Validate that required files exist.
    
    Args:
        required_files: List of required file paths
        base_dir: Base directory for relative paths
        
    Returns:
        Dictionary mapping file paths to existence status
    """
    validation_results = {}
    base_dir = Path(base_dir) if base_dir else Path.cwd()
    
    for file_path in required_files:
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = base_dir / file_path
        
        validation_results[str(file_path)] = file_path.exists()
    
    return validation_results


# Example usage and testing
if __name__ == "__main__":
    # Test file utilities
    test_dir = Path("test_file_utils")
    ensure_directory(test_dir)
    
    # Test JSON save/load
    test_data = {"test": "data", "timestamp": datetime.now()}
    json_file = test_dir / "test.json"
    
    if save_json(test_data, json_file):
        loaded_data = load_json(json_file)
        print(f"JSON test: {loaded_data is not None}")
    
    # Test file listing
    print(f"Files in current dir: {len(get_image_files('.'))}")
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    print("File utilities test completed")
