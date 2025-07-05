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
from typing import Optional, Union, Dict, List


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


def get_image_id_paths(image_ids: Union[str, List[str]], metadata_or_path: Union[str, Path, Dict]) -> Union[Path, List[Path]]:
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
        metadata_or_path: Either a loaded metadata dict OR path to experiment_metadata.json
                         Use loaded dict for efficiency when processing multiple images
        
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
        >>> metadata = load_experiment_metadata("data/raw_data_organized/experiment_metadata.json")
        >>> path = get_image_id_paths("20231206_A02_0000", metadata)
    """
    # Handle both metadata object and path
    if isinstance(metadata_or_path, (str, Path)):
        # Load metadata from path
        try:
            metadata = load_experiment_metadata(metadata_or_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise FileNotFoundError(f"Could not load experiment metadata from {metadata_or_path}: {e}")
    elif isinstance(metadata_or_path, dict):
        # Use provided metadata object
        metadata = metadata_or_path
    else:
        raise TypeError(f"metadata_or_path must be str, Path, or dict, got {type(metadata_or_path)}")
    
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
