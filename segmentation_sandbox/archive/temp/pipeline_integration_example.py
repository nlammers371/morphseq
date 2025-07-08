#!/usr/bin/env python
"""
Example Pipeline Script using Experiment Metadata Utilities
==========================================================

This script demonstrates how to integrate the experiment metadata utilities
into a real image processing pipeline.
"""

import sys
from pathlib import Path

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from experiment_metadata_utils import (
    load_experiment_metadata,
    get_image_id_path,
    find_video_for_image,
    get_metadata_summary,
    list_image_ids_for_video
)


def process_images_from_video(video_id: str, metadata_path: Path, max_images: int = 10):
    """
    Example function that processes all images from a specific video.
    
    Args:
        video_id: Video identifier (e.g., "20241023_A01")
        metadata_path: Path to experiment metadata
        max_images: Maximum number of images to process
    """
    print(f"\nProcessing images from video: {video_id}")
    
    # Load metadata once for efficiency
    try:
        metadata = load_experiment_metadata(metadata_path)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    # Get all image_ids for this video
    image_ids = list_image_ids_for_video(video_id, metadata)
    if not image_ids:
        print(f"No images found for video {video_id}")
        return
    
    print(f"Found {len(image_ids)} images in video")
    
    # Process up to max_images
    processed_count = 0
    error_count = 0
    
    for image_id in image_ids[:max_images]:
        try:
            # Get robust path using loaded metadata
            image_path = get_image_id_path(image_id, metadata)
            
            # Simulate processing
            if image_path.exists():
                # Here you would do actual image processing
                # For demo, just check file size
                file_size = image_path.stat().st_size
                print(f"  ✓ {image_id}: {file_size:,} bytes")
                processed_count += 1
            else:
                print(f"  ✗ {image_id}: File missing")
                error_count += 1
                
        except Exception as e:
            print(f"  ✗ {image_id}: Error - {e}")
            error_count += 1
    
    print(f"Processed: {processed_count}, Errors: {error_count}")


def process_images_by_list(image_ids: list, metadata_path: Path):
    """
    Example function that processes a specific list of image_ids.
    
    Args:
        image_ids: List of image identifiers
        metadata_path: Path to experiment metadata
    """
    print(f"\nProcessing {len(image_ids)} specific images...")
    
    # Load metadata once for efficiency
    try:
        metadata = load_experiment_metadata(metadata_path)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return
    
    results = []
    
    for image_id in image_ids:
        try:
            # Robust path lookup
            image_path = get_image_id_path(image_id, metadata)
            
            # Get video information
            video_info = find_video_for_image(image_id, metadata)
            
            result = {
                'image_id': image_id,
                'path': image_path,
                'exists': image_path.exists(),
                'video_id': video_info['video_id'] if video_info else None,
                'well_id': video_info['well_id'] if video_info else None
            }
            results.append(result)
            
            print(f"  ✓ {image_id}: {result['video_id']} ({result['well_id']})")
            
        except ValueError as e:
            print(f"  ✗ {image_id}: Not in metadata - {e}")
            results.append({
                'image_id': image_id,
                'path': None,
                'exists': False,
                'error': str(e)
            })
        except FileNotFoundError as e:
            print(f"  ✗ {image_id}: File missing - {e}")
            results.append({
                'image_id': image_id,
                'path': None,
                'exists': False,
                'error': str(e)
            })
    
    return results


def validate_image_batch(image_ids: list, metadata_path: Path):
    """
    Validate that a batch of image_ids all exist and are accessible.
    
    Args:
        image_ids: List of image identifiers to validate
        metadata_path: Path to experiment metadata
        
    Returns:
        Tuple of (valid_ids, invalid_ids)
    """
    print(f"\nValidating {len(image_ids)} images...")
    
    # Load metadata once
    try:
        metadata = load_experiment_metadata(metadata_path)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return [], image_ids
    
    valid_ids = []
    invalid_ids = []
    
    for image_id in image_ids:
        try:
            image_path = get_image_id_path(image_id, metadata)
            if image_path.exists():
                valid_ids.append(image_id)
            else:
                invalid_ids.append(image_id)
        except:
            invalid_ids.append(image_id)
    
    print(f"Valid: {len(valid_ids)}, Invalid: {len(invalid_ids)}")
    return valid_ids, invalid_ids


def main():
    """Main function demonstrating pipeline usage."""
    
    # Set up paths
    metadata_path = Path(__file__).parent.parent.parent / "data" / "raw_data_organized" / "experiment_metadata.json"
    
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return
    
    print("Experiment Metadata Utilities - Pipeline Integration Demo")
    print("=" * 60)
    
    # Show metadata summary
    summary = get_metadata_summary(metadata_path)
    print(f"Dataset summary:")
    print(f"  Experiments: {summary['total_experiments']}")
    print(f"  Videos: {summary['total_videos']}")
    print(f"  Images: {summary['total_images']}")
    
    # Example 1: Process images from a specific video
    process_images_from_video("20241023_A01", metadata_path, max_images=5)
    
    # Example 2: Process specific image_ids
    specific_images = [
        "20241023_A01_0000",
        "20241023_A01_0001", 
        "20241023_A01_0002",
        "20241023_A01_9999",  # This one shouldn't exist
        "20999999_Z99_0000"   # This one definitely doesn't exist
    ]
    results = process_images_by_list(specific_images, metadata_path)
    
    # Example 3: Validate a batch of images
    test_batch = [
        "20241023_A01_0000",
        "20241023_A01_0001",
        "20241023_A01_0002",
        "20241023_A01_0003",
        "20241023_A01_0004"
    ]
    valid_ids, invalid_ids = validate_image_batch(test_batch, metadata_path)
    
    if valid_ids:
        print(f"\nValid images ready for processing: {len(valid_ids)}")
        # Here you would continue with your actual image processing pipeline
        print("Ready to proceed with image processing pipeline...")
    else:
        print("No valid images found - check your image_ids")


if __name__ == "__main__":
    main()
