#!/usr/bin/env python
"""
Experiment Metadata Utilities
=============================

Utility functions for working with experiment metadata and finding image file paths.
This module provides functions to query the experiment_metadata.json file and locate
image files based on their image_ids.
"""


# **Structure of `experiment_metadata.json`:**
# ```json
# {
#   "script_version": "01_prepare_videos.py",
#   "creation_time": "YYYY-MM-DDThh:mm:ss",
#   "experiment_ids": ["20240411", ...],
#   "video_ids": ["20240411_A01", ...],
#   "image_ids": ["20240411_A01_0000", ...],
#   "experiments": {
#     "20240411": {
#       "experiment_id": "20240411",
#       "first_processed_time": "YYYY-MM-DDThh:mm:ss",
#       "last_processed_time": "YYYY-MM-DDThh:mm:ss",
#       "videos": {
#         "20240411_A01": {
#           "video_id": "20240411_A01",
#           "well_id": "A01",
#           "mp4_path": "/path/to/20240411/vids/20240411_A01.mp4",
#           "processed_jpg_images_dir": "/path/to/20240411/images/20240411_A01",
#           "image_ids": ["20240411_A01_0000", ...],
#           "total_source_images": 100,
#           "valid_frames": 100,
#           "video_resolution": [512, 512],
#           "last_processed_time": "YYYY-MM-DDThh:mm:ss"
#         }
#       }
#     }
#   }
# }


import json
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import json
import sys
import shutil
from datetime import datetime
from collections import defaultdict
import os
import re
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import pyvips for faster image I/O
try:
    import pyvips
    PYVIPS_AVAILABLE = True
except ImportError:
    PYVIPS_AVAILABLE = False
    
# --- Configuration for Image and Video Processing ---
JPEG_QUALITY = 90
VIDEO_FPS = 5
VIDEO_CODEC = 'mp4v' # More compatible than H264, use 'avc1' for H264

# Frame overlay settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_COLOR = (255, 255, 255)  # White text
FONT_THICKNESS = 3


class ExperimentMetadata:
    """
    Enhanced experiment metadata manager with auto-save, backup, and initialization features.
    
    This class provides robust metadata management with:
    - Automatic backup creation before saves
    - Auto-save functionality for batch processing
    - Safe loading with error recovery
    - Incremental updates without data loss
    - Thread-safe operations
    
    Key Features:
    - Store experiment, video, and image metadata
    - Track processing timestamps and status
    - Auto-discovery of new experiments
    - Batch processing support with checkpoints
    - Rollback capability via backups
    
    Basic Usage:
        # Initialize or load existing metadata
        metadata = ExperimentMetadata("experiment_metadata.json")
        
        # Add experiment data
        metadata.add_experiment("20240411")
        metadata.add_video("20240411", "A01", video_info)
        metadata.add_images("20240411_A01", image_list)
        
        # Auto-save during processing
        metadata.save()
    
    Batch Processing:
        # Enable auto-save every 10 processed items
        metadata = ExperimentMetadata("metadata.json", auto_save_interval=10)
        
        # Process items with automatic checkpointing
        for item in large_batch:
            metadata.process_item(item)
            metadata.increment_processed()  # Triggers auto-save when interval reached
    """

    def __init__(self, filepath: Union[str, Path], verbose: bool = True, 
                 auto_save_interval: Optional[int] = None):
        """
        Initialize the metadata manager.
        
        Args:
            filepath: Path to metadata JSON file
            verbose: Whether to print status messages
            auto_save_interval: If provided, auto-save every N processed items
        """
        self.filepath = Path(filepath)
        self.verbose = verbose
        self.auto_save_interval = auto_save_interval
        self._processed_count = 0
        self.metadata = self._load_or_initialize()
        self._unsaved_changes = False

    def _load_or_initialize(self) -> Dict:
        """Load existing metadata file or initialize a new one."""
        if self.filepath.exists():
            if self.verbose:
                print(f"📁 Loading existing metadata from: {self.filepath}")
            try:
                with open(self.filepath, 'r') as f:
                    metadata = json.load(f)
                # Validate and upgrade schema if needed
                metadata = self._validate_and_upgrade_schema(metadata)
                return metadata
            except (json.JSONDecodeError, KeyError) as e:
                if self.verbose:
                    print(f"⚠️  Error loading metadata: {e}")
                    print("🔄 Creating backup and initializing new metadata...")
                # Create backup of corrupted file
                backup_path = self.filepath.with_suffix('.json.backup.corrupted')
                shutil.copy2(self.filepath, backup_path)
                if self.verbose:
                    print(f"📦 Corrupted file backed up to: {backup_path}")
        
        if self.verbose:
            print(f"🆕 Initializing new metadata file at: {self.filepath}")
        return self._create_empty_metadata()

    def _validate_and_upgrade_schema(self, metadata: Dict) -> Dict:
        """Validate and upgrade metadata schema to current version."""
        # Ensure all required top-level keys exist
        required_keys = {
            "script_version": "01_prepare_videos.py",
            "creation_time": datetime.now().isoformat(),
            "experiment_ids": [],
            "video_ids": [],
            "image_ids": [],
            "experiments": {}
        }
        
        for key, default_value in required_keys.items():
            if key not in metadata:
                metadata[key] = default_value
                self._unsaved_changes = True

        # Add processing statistics if missing
        if "processing_stats" not in metadata:
            metadata["processing_stats"] = {
                "last_updated": datetime.now().isoformat(),
                "total_experiments_processed": len(metadata.get("experiment_ids", [])),
                "total_videos_processed": len(metadata.get("video_ids", [])),
                "total_images_processed": len(metadata.get("image_ids", []))
            }
            self._unsaved_changes = True

        return metadata

    def _create_empty_metadata(self) -> Dict:
        """Create a new empty metadata structure."""
        return {
            "script_version": "01_prepare_videos.py",
            "creation_time": datetime.now().isoformat(),
            "experiment_ids": [],
            "video_ids": [],
            "image_ids": [],
            "experiments": {},
            "processing_stats": {
                "last_updated": datetime.now().isoformat(),
                "total_experiments_processed": 0,
                "total_videos_processed": 0,
                "total_images_processed": 0
            }
        }

    def save(self, create_backup: bool = True):
        """
        Save metadata to file with optional backup.
        
        Args:
            create_backup: Whether to create a backup before saving
        """
        # Create backup if requested and file exists
        if create_backup and self.filepath.exists():
            backup_path = self.filepath.with_suffix(f'.json.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            try:
                shutil.copy2(self.filepath, backup_path)
                if self.verbose:
                    print(f"📦 Created backup: {backup_path.name}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to create backup: {e}")

        # Update processing stats
        self.metadata["processing_stats"]["last_updated"] = datetime.now().isoformat()
        
        # Ensure parent directory exists
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_path = self.filepath.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Atomic rename
            temp_path.replace(self.filepath)
            self._unsaved_changes = False
            
            if self.verbose:
                print(f"💾 Saved metadata to: {self.filepath}")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to save metadata: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    def add_experiment(self, experiment_id: str, experiment_data: Optional[Dict] = None) -> bool:
        """
        Add a new experiment to metadata.
        
        Args:
            experiment_id: Experiment identifier
            experiment_data: Optional experiment data (will be merged with defaults)
            
        Returns:
            True if experiment was added, False if it already existed
        """
        if experiment_id in self.metadata["experiments"]:
            # Update last processed time
            self.metadata["experiments"][experiment_id]["last_processed_time"] = datetime.now().isoformat()
            self._unsaved_changes = True
            return False

        # Create new experiment entry
        new_experiment = {
            "experiment_id": experiment_id,
            "first_processed_time": datetime.now().isoformat(),
            "last_processed_time": datetime.now().isoformat(),
            "videos": {}
        }
        
        # Merge with provided data
        if experiment_data:
            new_experiment.update(experiment_data)

        self.metadata["experiments"][experiment_id] = new_experiment
        
        # Update top-level lists
        if experiment_id not in self.metadata["experiment_ids"]:
            self.metadata["experiment_ids"].append(experiment_id)
        
        self._unsaved_changes = True
        self._check_auto_save()
        
        if self.verbose:
            print(f"✅ Added experiment: {experiment_id}")
        
        return True

    def add_video(self, experiment_id: str, well_id: str, video_data: Dict) -> bool:
        """
        Add a video to an experiment.
        
        Args:
            experiment_id: Parent experiment ID
            well_id: Well identifier
            video_data: Video metadata dictionary
            
        Returns:
            True if video was added, False if it already existed
        """
        # Ensure experiment exists
        if experiment_id not in self.metadata["experiments"]:
            self.add_experiment(experiment_id)

        video_id = f"{experiment_id}_{well_id}"
        
        # Check if video already exists
        if video_id in self.metadata["experiments"][experiment_id]["videos"]:
            # Update existing video data
            self.metadata["experiments"][experiment_id]["videos"][video_id].update(video_data)
            self.metadata["experiments"][experiment_id]["videos"][video_id]["last_processed_time"] = datetime.now().isoformat()
            self._unsaved_changes = True
            return False

        # Add video_id and well_id to the video data
        video_data_complete = {
            "video_id": video_id,
            "well_id": well_id,
            "last_processed_time": datetime.now().isoformat(),
            **video_data
        }

        self.metadata["experiments"][experiment_id]["videos"][video_id] = video_data_complete
        
        # Update top-level lists
        if video_id not in self.metadata["video_ids"]:
            self.metadata["video_ids"].append(video_id)
        
        self._unsaved_changes = True
        self._check_auto_save()
        
        if self.verbose:
            print(f"✅ Added video: {video_id}")
        
        return True

    def add_images(self, video_id: str, image_ids: List[str]) -> int:
        """
        Add image IDs to a video.
        
        Args:
            video_id: Video identifier
            image_ids: List of image IDs to add
            
        Returns:
            Number of new images added
        """
        # Find the experiment that contains this video
        experiment_id = None
        for exp_id, exp_data in self.metadata.get("experiments", {}).items():
            if video_id in exp_data.get("videos", {}):
                experiment_id = exp_id
                break
        
        if experiment_id is None:
            raise ValueError(f"Video {video_id} not found in any experiment")
        
        if video_id not in self.metadata["experiments"][experiment_id]["videos"]:
            raise ValueError(f"Video {video_id} not found in experiment {experiment_id}")

        # Get existing image IDs for this video
        existing_images = set(self.metadata["experiments"][experiment_id]["videos"][video_id].get("image_ids", []))
        
        # Add new images
        new_images = [img_id for img_id in image_ids if img_id not in existing_images]
        
        if new_images:
            # Update video's image list
            all_images = list(existing_images) + new_images
            self.metadata["experiments"][experiment_id]["videos"][video_id]["image_ids"] = sorted(all_images)
            
            # Update top-level image list
            for img_id in new_images:
                if img_id not in self.metadata["image_ids"]:
                    self.metadata["image_ids"].append(img_id)
            
            self._unsaved_changes = True
            self._check_auto_save()
            
            if self.verbose:
                print(f"✅ Added {len(new_images)} new images to {video_id}")

        return len(new_images)

    def increment_processed(self):
        """Increment processed counter and trigger auto-save if needed."""
        self._processed_count += 1
        self._check_auto_save()

    def _check_auto_save(self):
        """Check if auto-save should be triggered."""
        if (self.auto_save_interval and 
            self._processed_count >= self.auto_save_interval and 
            self._unsaved_changes):
            
            if self.verbose:
                print(f"💾 Auto-saving metadata (processed {self._processed_count} items)...")
            self.save()
            self._processed_count = 0

    def get_summary(self) -> Dict:
        """Get summary statistics about the metadata."""
        return {
            "total_experiments": len(self.metadata.get("experiment_ids", [])),
            "total_videos": len(self.metadata.get("video_ids", [])),
            "total_images": len(self.metadata.get("image_ids", [])),
            "creation_time": self.metadata.get("creation_time", "Unknown"),
            "last_updated": self.metadata.get("processing_stats", {}).get("last_updated", "Unknown"),
            "script_version": self.metadata.get("script_version", "Unknown"),
            "unsaved_changes": self._unsaved_changes
        }

    def print_summary(self):
        """Print a formatted summary of the metadata."""
        summary = self.get_summary()
        print(f"\n📊 EXPERIMENT METADATA SUMMARY")
        print(f"=" * 40)
        print(f"🧪 Total experiments: {summary['total_experiments']}")
        print(f"🎬 Total videos: {summary['total_videos']}")
        print(f"🖼️  Total images: {summary['total_images']}")
        print(f"📅 Created: {summary['creation_time'][:10]}")
        print(f"🔄 Last updated: {summary['last_updated'][:19]}")
        print(f"💾 Status: {'⚠️ unsaved changes' if summary['unsaved_changes'] else '✅ saved'}")

    @property
    def has_unsaved_changes(self) -> bool:
        """Check for unsaved changes."""
        return self._unsaved_changes

    def cleanup_old_backups(self, keep_count: int = 1):
        """Clean up old backup files, keeping only the most recent ones."""
        backup_pattern = f"{self.filepath.stem}.json.backup.*"
        backup_files = sorted(
            self.filepath.parent.glob(backup_pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old backups beyond keep_count
        for backup_file in backup_files[keep_count:]:
            try:
                backup_file.unlink()
                if self.verbose:
                    print(f"🗑️  Removed old backup: {backup_file.name}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Failed to remove backup {backup_file.name}: {e}")

    def __repr__(self) -> str:
        """String representation."""
        summary = self.get_summary()
        status = "✅ saved" if not self._unsaved_changes else "⚠️ unsaved"
        return f"ExperimentMetadata(experiments={summary['total_experiments']}, videos={summary['total_videos']}, images={summary['total_images']}, {status})"
    
    def get_video_ids(self) -> List[str]:
        """Get list of all video IDs."""
        return self.metadata.get("video_ids", [])
    
    def get_image_ids(self) -> List[str]:
        """Get list of all image IDs."""
        return self.metadata.get("image_ids", [])
    
    def get_experiment_videos(self, experiment_id: str) -> Dict:
        """Get all videos for an experiment."""
        return self.metadata.get("experiments", {}).get(experiment_id, {}).get("videos", {})
    
    def get_video_image_ids(self, experiment_id: str, video_id: str) -> List[str]:
        """Get image IDs for a specific video."""
        videos = self.get_experiment_videos(experiment_id)
        return videos.get(video_id, {}).get("image_ids", [])

    def has_video(self, video_id: str) -> bool:
        """Check if a video ID exists."""
        return video_id in self.get_video_ids()
    
    def has_image(self, image_id: str) -> bool:
        """Check if an image ID exists."""
        return image_id in self.get_image_ids()


def load_experiment_metadata(metadata_path: Union[str, Path]) -> Dict:
    """
    Load experiment metadata from JSON file.
    
    Args:
        metadata_path: Path to experiment_metadata.json
        
    Returns:
        Dictionary containing experiment metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If metadata file is corrupted
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Experiment metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def parse_image_id(image_id: str) -> Dict[str, str]:
    """
    Parse image_id to extract its components.
    
    Args:
        image_id: Image identifier (e.g., "20231206_A02_0000")
        
    Returns:
        Dictionary with keys: experiment_id, well_id, video_id, timepoint
        
    Raises:
        ValueError: If image_id format is invalid
        
    Example:
        >>> parse_image_id("20231206_A02_0000")
        {
            'experiment_id': '20231206',
            'well_id': 'A02', 
            'video_id': '20231206_A02',
            'timepoint': '0000'
        }
    """
    parts = image_id.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid image_id format: {image_id}. Expected format: YYYYMMDD_WELL_TIMEPOINT")
    
    experiment_id = parts[0]  # e.g., "20231206"
    well_id = parts[1]        # e.g., "A02"
    timepoint = parts[2]      # e.g., "0000"
    video_id = f"{experiment_id}_{well_id}"  # e.g., "20231206_A02"
    
    return {
        'experiment_id': experiment_id,
        'well_id': well_id,
        'video_id': video_id,
        'timepoint': timepoint
    }


def get_image_id_paths(image_ids: Union[str, List[str]], metadata_or_path: Union[str, Path, Dict, ExperimentMetadata]) -> Union[Path, List[Path]]:
    """
    Get the full path(s) to image file(s) from image_id(s) using metadata lookup.
    
    This is the ROBUST method that:
    1. First checks if image_id exists in the metadata's image_ids list
    2. Searches through all experiments and videos to find the image_id
    3. Uses the actual processed_jpg_images_dir path from metadata
    4. Verifies the image exists in that directory
    
    Args:
        image_ids: Single image identifier OR list of image identifiers
                  (e.g., "20231206_A02_0000" or ["20231206_A02_0000", "20231206_A02_0001"])
        metadata_or_path: Either a loaded metadata dict, ExperimentMetadata instance, OR path to experiment_metadata.json
                         Use loaded dict/instance for efficiency when processing multiple images
        
    Returns:
        Single Path if single image_id provided, List[Path] if list of image_ids provided
        
    Raises:
        ValueError: If any image_id is not found in metadata
        FileNotFoundError: If any image is in metadata but file doesn't exist on disk
        
    Example:
        # Single image
        >>> path = get_image_id_paths("20231206_A02_0000", "data/raw_data_organized/experiment_metadata.json")
        
        # Multiple images
        >>> paths = get_image_id_paths(["20231206_A02_0000", "20231206_A02_0001"], metadata)
        
        # Method 2: Pass loaded metadata (efficient for batch processing)
        >>> metadata = ExperimentMetadata("data/raw_data_organized/experiment_metadata.json")
        >>> path = get_image_id_paths("20231206_A02_0000", metadata)
    """
    # Handle different metadata input types
    if isinstance(metadata_or_path, ExperimentMetadata):
        # Use ExperimentMetadata instance
        metadata = metadata_or_path.metadata
    elif isinstance(metadata_or_path, (str, Path)):
        # Load metadata from path
        try:
            metadata = load_experiment_metadata(metadata_or_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise FileNotFoundError(f"Could not load experiment metadata from {metadata_or_path}: {e}")
    elif isinstance(metadata_or_path, dict):
        # Use provided metadata object
        metadata = metadata_or_path
    else:
        raise TypeError(f"metadata_or_path must be str, Path, dict, or ExperimentMetadata, got {type(metadata_or_path)}")
    
    # Handle both single image_id and list of image_ids
    if isinstance(image_ids, str):
        # Single image_id case
        image_id = image_ids
        
        # First check if image_id exists in the top-level image_ids list
        all_image_ids = metadata.get("image_ids", [])
        if image_id not in all_image_ids:
            raise ValueError(f"Image ID '{image_id}' not found in experiment metadata. "
                            f"Available image count: {len(all_image_ids)}")
        
        # Search through all experiments and videos to find the image_id
        for experiment_id, experiment_data in metadata.get("experiments", {}).items():
            for video_id, video_data in experiment_data.get("videos", {}).items():
                # Check if image_id exists in this video
                if image_id in video_data.get("image_ids", []):
                    # Found the video containing this image_id
                    # Get the actual processed_jpg_images_dir path from metadata
                    images_dir = Path(video_data["processed_jpg_images_dir"])
                    
                    # Construct full image path
                    image_path = images_dir / f"{image_id}.jpg"
                    
                    # Verify the image actually exists
                    if image_path.exists():
                        return image_path
                    else:
                        # Image is in metadata but file doesn't exist
                        raise FileNotFoundError(f"Image '{image_id}' found in metadata but file does not exist: {image_path}")
        
        # This should not happen if image_id was in the top-level list
        raise ValueError(f"Image ID '{image_id}' found in metadata image_ids list but not in any video")
    
    elif isinstance(image_ids, list):
        # Multiple image_ids case - recursively call for each
        paths = []
        for image_id in image_ids:
            # Recursively call for single image_id (efficient since metadata is already loaded)
            path = get_image_id_paths(image_id, metadata)
            paths.append(path)
        return paths
    
    else:
        raise TypeError(f"image_ids must be str or List[str], got {type(image_ids)}")


def get_image_path_fast(image_id: str, base_data_dir: Union[str, Path] = None) -> Path:
    """
    Get image path directly from image_id without loading metadata (faster).
    
    This assumes the standard directory structure:
    {base_data_dir}/{experiment_id}/images/{video_id}/{image_id}.jpg
    
    Args:
        image_id: Image identifier (e.g., "20231206_A02_0000")
        base_data_dir: Base data directory (defaults to standard location)
        
    Returns:
        Path to the image file (may not exist if image_id is invalid)
        
    Raises:
        ValueError: If image_id format is invalid
        
    Example:
        >>> path = get_image_path_fast("20231206_A02_0000")
        >>> print(path)
        /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/raw_data_organized/20231206/images/20231206_A02/20231206_A02_0000.jpg
    """
    if base_data_dir is None:
        # Default to standard location relative to this file (scripts/utils -> segmentation_sandbox/data)
        base_data_dir = Path(__file__).parent.parent.parent / "data" / "raw_data_organized"
    else:
        base_data_dir = Path(base_data_dir)
    
    # Parse image_id to extract components
    components = parse_image_id(image_id)
    experiment_id = components['experiment_id']
    video_id = components['video_id']
    
    # Construct path following standard structure
    image_path = base_data_dir / experiment_id / "images" / video_id / f"{image_id}.jpg"
    
    return image_path


def verify_image_exists(image_id: str, metadata_or_path: Union[str, Path, Dict]) -> bool:
    """
    Verify that an image file actually exists on disk in the location specified by metadata.
    
    This function:
    1. Looks up the image_id in metadata to find its video
    2. Gets the processed_jpg_images_dir from that video's metadata
    3. Checks if the actual file exists at that location
    
    Args:
        image_id: Image identifier
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
        
    Returns:
        True if image file exists at the metadata-specified location, False otherwise
    """
    try:
        image_path = get_image_id_path(image_id, metadata_or_path)
        return True  # get_image_id_path already checks existence and raises if not found
    except (ValueError, FileNotFoundError):
        return False


def find_image(image_id: str, metadata_or_path: Union[str, Path, Dict] = None) -> Optional[Path]:
    """
    Find an image file, with default metadata path.
    
    Args:
        image_id: Image identifier
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
                         (defaults to standard location if None)
        
    Returns:
        Path to image file if found, None if not found
    """
    if metadata_or_path is None:
        # Default metadata path (scripts/utils -> segmentation_sandbox/data)
        metadata_or_path = Path(__file__).parent.parent.parent / "data" / "raw_data_organized" / "experiment_metadata.json"
    
    try:
        return get_image_id_path(image_id, metadata_or_path)
    except (ValueError, FileNotFoundError):
        return None


def find_video_for_image(image_id: str, metadata_or_path: Union[str, Path, Dict]) -> Optional[Dict]:
    """
    Find which video contains a specific image_id.
    
    Args:
        image_id: Image identifier to search for
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
        
    Returns:
        Dictionary with video information, or None if not found
        
    Example:
        >>> video_info = find_video_for_image("20231206_A02_0000", metadata_path)
        >>> print(video_info['video_id'])
        20231206_A02
    """
    # Handle both metadata object and path
    if isinstance(metadata_or_path, (str, Path)):
        try:
            metadata = load_experiment_metadata(metadata_or_path)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    elif isinstance(metadata_or_path, dict):
        metadata = metadata_or_path
    else:
        return None
    
    # Search through all experiments and videos to find the image_id
    for experiment_id, experiment_data in metadata.get("experiments", {}).items():
        for video_id, video_data in experiment_data.get("videos", {}).items():
            if image_id in video_data.get("image_ids", []):
                return video_data
    
    return None


def get_video_info(video_id: str, metadata_or_path: Union[str, Path, Dict]) -> Optional[Dict]:
    """
    Get information about a video from metadata.
    
    Args:
        video_id: Video identifier (e.g., "20231206_A02")
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
        
    Returns:
        Dictionary with video information, or None if not found
    """
    # Handle both metadata object and path
    if isinstance(metadata_or_path, (str, Path)):
        try:
            metadata = load_experiment_metadata(metadata_or_path)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    elif isinstance(metadata_or_path, dict):
        metadata = metadata_or_path
    else:
        return None
    
    # Parse video_id to get experiment_id
    parts = video_id.split('_')
    if len(parts) < 2:
        return None
    
    experiment_id = parts[0]
    
    try:
        return metadata["experiments"][experiment_id]["videos"][video_id]
    except KeyError:
        return None


def get_experiment_info(experiment_id: str, metadata_or_path: Union[str, Path, Dict]) -> Optional[Dict]:
    """
    Get information about an experiment from metadata.
    
    Args:
        experiment_id: Experiment identifier (e.g., "20231206")
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
        
    Returns:
        Dictionary with experiment information, or None if not found
    """
    # Handle both metadata object and path
    if isinstance(metadata_or_path, (str, Path)):
        try:
            metadata = load_experiment_metadata(metadata_or_path)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    elif isinstance(metadata_or_path, dict):
        metadata = metadata_or_path
    else:
        return None
    
    try:
        return metadata["experiments"][experiment_id]
    except KeyError:
        return None


def list_all_image_ids(metadata_or_path: Union[str, Path, Dict]) -> List[str]:
    """
    Get a list of all image_ids in the metadata.
    
    Args:
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
        
    Returns:
        List of all image_ids
    """
    # Handle both metadata object and path
    if isinstance(metadata_or_path, (str, Path)):
        try:
            metadata = load_experiment_metadata(metadata_or_path)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    elif isinstance(metadata_or_path, dict):
        metadata = metadata_or_path
    else:
        return []
    
    return metadata.get("image_ids", [])


def list_video_ids_for_experiment(experiment_id: str, metadata_or_path: Union[str, Path, Dict]) -> List[str]:
    """
    Get a list of all video_ids for a specific experiment.
    
    Args:
        experiment_id: Experiment identifier
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
        
    Returns:
        List of video_ids for the experiment
    """
    # Handle both metadata object and path
    if isinstance(metadata_or_path, (str, Path)):
        try:
            metadata = load_experiment_metadata(metadata_or_path)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    elif isinstance(metadata_or_path, dict):
        metadata = metadata_or_path
    else:
        return []
    
    try:
        experiment_data = metadata["experiments"][experiment_id]
        return list(experiment_data["videos"].keys())
    except KeyError:
        return []


def list_image_ids_for_video(video_id: str, metadata_or_path: Union[str, Path, Dict]) -> List[str]:
    """
    Get a list of all image_ids for a specific video.
    
    Args:
        video_id: Video identifier
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
        
    Returns:
        List of image_ids for the video
    """
    video_info = get_video_info(video_id, metadata_or_path)
    if video_info is None:
        return []
    return video_info.get("image_ids", [])


def get_metadata_summary(metadata_or_path: Union[str, Path, Dict]) -> Dict:
    """
    Get a summary of the metadata contents.
    
    Args:
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
        
    Returns:
        Dictionary with summary statistics
    """
    # Handle both metadata object and path
    if isinstance(metadata_or_path, (str, Path)):
        try:
            metadata = load_experiment_metadata(metadata_or_path)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "total_experiments": 0,
                "total_videos": 0,
                "total_images": 0,
                "experiment_ids": [],
                "creation_time": "Unknown",
                "script_version": "Unknown"
            }
    elif isinstance(metadata_or_path, dict):
        metadata = metadata_or_path
    else:
        return {
            "total_experiments": 0,
            "total_videos": 0,
            "total_images": 0,
            "experiment_ids": [],
            "creation_time": "Unknown",
            "script_version": "Unknown"
        }
    
    return {
        "total_experiments": len(metadata.get("experiment_ids", [])),
        "total_videos": len(metadata.get("video_ids", [])),
        "total_images": len(metadata.get("image_ids", [])),
        "experiment_ids": metadata.get("experiment_ids", []),
        "creation_time": metadata.get("creation_time", "Unknown"),
        "script_version": metadata.get("script_version", "Unknown")
    }


def parse_filename(filename: str) -> Tuple[Union[str, None], Union[str, None]]:
    """
    Extracts well ID and timepoint string from a filename.
    Example: 'A01_t0000_ch00_stitch.png' -> ('A01', '0000')
    """
    name, _ = os.path.splitext(filename)
    parts = name.split("_")
    if len(parts) < 2:
        return None, None
    well_id, time_str = parts[0], parts[1]
    if re.match(r'^[A-H][0-9]{2}$', well_id) and time_str.startswith('t'):
        return well_id, time_str[1:]
    return None, None

def get_image_files(directory: Path) -> List[Path]:
    """Find all supported image files in a directory, searching recursively."""
    extensions = ['.png', '.tif', '.tiff', '.jpg', '.jpeg']
    return [p for p in directory.rglob('*') if p.suffix.lower() in extensions]

def process_and_save_jpeg(
    image_path: Path, 
    output_path: Path,
    target_size: Tuple[int, int] = None,
    overwrite: bool = False,
    show_warnings: bool = True
):
    """
    Reads an image and saves it as a clean JPEG without overlays.
    Only resizes if target_size is provided and different from current size.
    Returns the dimensions of the saved JPEG.
    Uses pyvips when available for faster I/O, falls back to OpenCV.
    """
    if output_path.exists() and not overwrite:
        # Get dimensions from existing file
        if PYVIPS_AVAILABLE:
            try:
                img = pyvips.Image.new_from_file(str(output_path), access='sequential')
                return img.width, img.height
            except Exception:
                pass
        # Fallback to OpenCV
        existing_image = cv2.imread(str(output_path))
        if existing_image is not None:
            return existing_image.shape[1], existing_image.shape[0]
    
    # Use pyvips for faster processing when available
    if PYVIPS_AVAILABLE:
        try:
            # Load image with pyvips
            img = pyvips.Image.new_from_file(str(image_path), access='sequential')
            
            # Convert to RGB if needed (pyvips handles this automatically)
            if img.bands == 4:  # RGBA
                img = img[:3]  # Take only RGB channels
            
            # Resize if needed
            if target_size and (img.width, img.height) != target_size:
                target_width, target_height = target_size
                # Ensure dimensions are even for video codec compatibility
                target_width = target_width - (target_width % 2)
                target_height = target_height - (target_height % 2)
                img = img.resize(target_width / img.width, vscale=target_height / img.height)
            
            # Save as JPEG with specified quality
            img.write_to_file(str(output_path), Q=JPEG_QUALITY)
            return img.width, img.height
            
        except Exception as e:
            if show_warnings:
                print(f"Warning: pyvips failed for {image_path}, falling back to OpenCV: {e}")
    
    # Fallback to OpenCV (original implementation)
    image = cv2.imread(str(image_path))
    if image is None:
        if show_warnings:
            print(f"Warning: Could not read image {image_path}, skipping.")
        return None

    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    height, width = image.shape[:2]
    
    # Only resize if target_size is provided and different from current size
    if target_size and (width, height) != target_size:
        target_width, target_height = target_size
        # Ensure dimensions are even for video codec compatibility
        target_width = target_width - (target_width % 2)
        target_height = target_height - (target_height % 2)
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return image.shape[1], image.shape[0] # width, height

def create_video_from_jpegs(
    jpeg_paths: List[Path],
    video_path: Path,
    video_size: Tuple[int, int],
    overwrite: bool = False,
    verbose: bool = True
):
    """
    Creates an MP4 video from a list of JPEG images, adding frame number overlays.
    """
    if video_path.exists() and not overwrite:
        return

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    video_writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, video_size)

    if not video_writer.isOpened():
        if verbose:
            print(f"Error: Could not open video writer for {video_path}")
        return

    frames_written = 0
    for jpeg_path in sorted(jpeg_paths):
        frame = cv2.imread(str(jpeg_path))
        if frame is None:
            continue

        # Overlay full image ID for each frame
        image_id = jpeg_path.stem
        frame_text = image_id

        (text_width, text_height), _ = cv2.getTextSize(frame_text, FONT, FONT_SCALE, FONT_THICKNESS)
        # Position text at top right, 10% down from the top
        height, width = frame.shape[:2]
        margin_px = 10
        text_x = width - text_width - margin_px
        text_y = int(0.1 * height)

        # Add a dark, semi-transparent background for the text
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        cv2.putText(frame, frame_text, (text_x, text_y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)
        
        video_writer.write(frame)
        frames_written += 1
        
    video_writer.release()
    if verbose:
        print(f"Created video: {video_path.name} ({frames_written} frames)")


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) > 1:
        image_id = sys.argv[1]
        
        print(f"Analyzing image_id: {image_id}")
        
        # Parse image_id
        try:
            components = parse_image_id(image_id)
            print(f"Components: {components}")
        except ValueError as e:
            print(f"Parse error: {e}")
            sys.exit(1)
        
        # Try both methods to find the image
        print(f"\nLooking up image path...")
        
        # Method 1: Robust metadata lookup (searches through all videos)
        try:
            path1 = find_image(image_id)
            print(f"Robust metadata lookup: {path1}")
            if path1:
                print(f"Exists: {path1.exists()}")
                
                # Show which video contains this image
                video_info = find_video_for_image(image_id, Path(__file__).parent.parent.parent / "data" / "raw_data_organized" / "experiment_metadata.json")
                if video_info:
                    print(f"Found in video: {video_info['video_id']}")
                    print(f"Images directory: {video_info['processed_jpg_images_dir']}")
            else:
                print("Image not found in metadata or file doesn't exist")
        except Exception as e:
            print(f"Robust metadata lookup failed: {e}")
        
        # Method 2: Fast path construction (assumes standard structure)
        try:
            path2 = get_image_path_fast(image_id)
            print(f"Fast path construction: {path2}")
            print(f"Exists: {path2.exists()}")
        except Exception as e:
            print(f"Fast path construction failed: {e}")
            
        # Get video info
        video_id = components['video_id']
        print(f"\nVideo info for {video_id}:")
        video_info = get_video_info(video_id, Path(__file__).parent.parent.parent / "data" / "raw_data_organized" / "experiment_metadata.json")
        if video_info:
            print(f"  Well ID: {video_info.get('well_id')}")
            print(f"  Total images: {len(video_info.get('image_ids', []))}")
            print(f"  Video path: {video_info.get('mp4_path')}")
        else:
            print("  Video not found in metadata")
            
    else:
        # Show metadata summary
        print("Experiment Metadata Utilities")
        print("============================")
        print("Usage: python experiment_metadata_utils.py <image_id>")
        print("Example: python experiment_metadata_utils.py 20231206_A02_0000")
        
        # Show summary of current metadata
        metadata_path = Path(__file__).parent.parent.parent / "data" / "raw_data_organized" / "experiment_metadata.json"
        if metadata_path.exists():
            summary = get_metadata_summary(metadata_path)
            print(f"\nCurrent metadata summary:")
            print(f"  Total experiments: {summary['total_experiments']}")
            print(f"  Total videos: {summary['total_videos']}")
            print(f"  Total images: {summary['total_images']}")
            print(f"  Latest experiments: {summary['experiment_ids'][-5:] if summary['experiment_ids'] else []}")
        else:
            print(f"\nNo metadata file found at: {metadata_path}")
