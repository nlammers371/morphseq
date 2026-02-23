#!/usr/bin/env python
"""
01_prepare_videos.py

Refactored for simplicity and robustness. This script processes raw stitched 
images from one or more experiments, organizes them by well (video_id), creates 
clean JPEG image sequences, and generates a single downscaled video with 
frame number overlays for each well.

This script is designed to be incremental and reproducible. It maintains a single,
cumulative JSON metadata file, tracking all processed experiments, videos, and 
images. Already-processed images are skipped on subsequent runs unless the 
--overwrite flag is specified.

Expected Input Structure:
    directory_with_experiments/
    ├── 20240411/
    │   ├── A01_t0000_ch00_stitch.png
    │   └── ...
    └── 20240412/
        └── ...

Output Structure:
    output_parent_dir/
    └── raw_data_organized/
        ├── experiment_metadata.json  (Cumulative metadata for all runs)
        └── 20240411/
            ├── vids/
            │   └── 20240411_A01.mp4
            └── images/
                └── 20240411_A01/
                    └── 20240411_A01_0000.jpg

Usage:
    # Process specific experiments, creating/updating the metadata
    python scripts/01_prepare_videos.py \
        --directory_with_experiments /path/to/stitched_images \
        --output_parent_dir /path/to/top_level_output \
        --experiments_to_process 20240411,20240412

    # Process all new experiments found in the source directory
    python scripts/01_prepare_videos.py \
        --directory_with_experiments /path/to/stitched_images \
        --output_parent_dir /path/to/top_level_output
"""

import os
import argparse
import cv2
from pathlib import Path
import re
from tqdm import tqdm
from collections import defaultdict
import shutil
import json
from datetime import datetime
from typing import Union, Tuple, List, Optional, Dict

# global checkpoint variables
test_videos_written = 0
TEST_METADATA_PATH = None

# --- Configuration for Image and Video Processing ---
MAX_DIMENSION = 512  # Max width or height for both JPEGs and video
JPEG_QUALITY = 90
VIDEO_FPS = 5
VIDEO_CODEC = 'mp4v' # More compatible than H264, use 'avc1' for H264

# Frame overlay settings
ADD_FRAME_NUMBERS = True
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0  # Increased for better readability
FONT_COLOR = (255, 255, 255)  # White text
FONT_THICKNESS = 3  # Increased thickness for visibility

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
    """
    if output_path.exists() and not overwrite:
        # To get dimensions, we have to read the existing image
        existing_image = cv2.imread(str(output_path))
        if existing_image is not None:
            return existing_image.shape[1], existing_image.shape[0]
        # If we can't read it, we'll just overwrite it
    
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

        # Extract full image ID for overlay
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

def process_experiment(
    experiment_dir: Path,
    output_dir: Path,
    metadata: Dict,
    overwrite: bool = False,
    verbose: bool = True
) -> bool:
    """
    Processes a single experiment directory, updating the metadata object.
    Returns True if processing was successful, False otherwise.
    """
    if not experiment_dir.is_dir():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        return False

    experiment_id = experiment_dir.name
    if verbose:
        print(f"\n{'='*20}")
        print(f"Processing experiment: {experiment_id}")
        print(f"{'='*20}")

    # Add experiment to metadata if it's new
    if experiment_id not in metadata['experiments']:
        metadata['experiments'][experiment_id] = {
            "experiment_id": experiment_id,
            "first_processed_time": datetime.now().isoformat(),
            "videos": {}
        }
        metadata['experiment_ids'].append(experiment_id)

    metadata['experiments'][experiment_id]['last_processed_time'] = datetime.now().isoformat()
    
    experiment_output_dir = output_dir / experiment_id
    
    # Create dedicated folders for videos and image frames
    vids_dir = experiment_output_dir / "vids"
    images_dir = experiment_output_dir / "images"
    vids_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output videos will be in: {vids_dir}")
        print(f"Output images will be in: {images_dir}")

    if verbose:
        print("Step 1: Discovering image files...")
    all_images = get_image_files(experiment_dir)
    if not all_images:
        print(f"Warning: No image files found in {experiment_dir}. Skipping.")
        return False
    if verbose:
        print(f"Found {len(all_images)} total image files")

    if verbose:
        print("Step 2: Grouping images by well ID...")
    well_groups = defaultdict(list)
    for img_path in all_images:
        well_id, _ = parse_filename(img_path.name)
        if well_id:
            well_groups[well_id].append(img_path)
    
    if not well_groups:
        print(f"Warning: No images with valid well IDs found in {experiment_dir}. Skipping.")
        return False
        
    if verbose:
        print(f"Successfully grouped into {len(well_groups)} wells.")

    # Process only first 3 wells for testing
    wells_to_process = list(well_groups.items())[2:5]
    if verbose:
        print(f"Step 3: Testing on the first 3 wells: {[i[0] for i in wells_to_process]}")

    # Track warnings to limit verbosity
    warning_count = 0
    max_warnings = 5

    if verbose:
        print("Step 4: Processing wells...")
    
    for well_id, image_paths in tqdm(wells_to_process, desc=f"Processing {experiment_id}", disable=not verbose):
        if verbose:
            print(f"\n  Processing well: {well_id} ({len(image_paths)} images)")
        
        video_id = f"{experiment_id}_{well_id}"
        if verbose:
            print(f"  Video ID: {video_id}")

        # Skip well if already processed (only skip video creation when not overwriting)
        if not overwrite and video_id in metadata['video_ids']:
            if verbose:
                print(f"  Skipping already processed well/video: {video_id}")
            continue

        video_frame_dir = images_dir / video_id
        video_frame_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"  Created directory: {video_frame_dir}")

        processed_jpeg_paths = []
        image_dimensions = []
        
        # Get existing image IDs for this video, if any
        image_ids = metadata['experiments'][experiment_id].get('videos', {}).get(video_id, {}).get('image_ids', [])

        if verbose:
            print(f"  Converting {len(image_paths)} images to JPEG (skipping existing)...")
        
        new_images_processed = 0
        for image_path in sorted(image_paths):
            _, time_str = parse_filename(image_path.name)
            if not time_str:
                continue
            
            image_id = f"{video_id}_{time_str}"
            jpeg_path = video_frame_dir / f"{image_id}.jpg"

            # --- Incremental Processing Check ---
            if image_id in metadata['image_ids'] and not overwrite:
                # If we are not overwriting, we still need the path and dimensions
                if jpeg_path.exists():
                    processed_jpeg_paths.append(jpeg_path)
                    # This is slow, but necessary if we need to check for resizes
                    img = cv2.imread(str(jpeg_path))
                    if img is not None:
                        image_dimensions.append((img.shape[1], img.shape[0]))
                continue

            show_warnings = warning_count < max_warnings
            dims = process_and_save_jpeg(image_path, jpeg_path, None, overwrite, show_warnings)
            
            if dims is None and warning_count < max_warnings:
                warning_count += 1
            elif dims:
                new_images_processed += 1
                processed_jpeg_paths.append(jpeg_path)
                image_dimensions.append(dims)
                if image_id not in metadata['image_ids']:
                    metadata['image_ids'].append(image_id)
                if image_id not in image_ids:
                    image_ids.append(image_id)

        if verbose and not overwrite:
            print(f"  Skipped {len(image_paths) - new_images_processed} existing images. Processed {new_images_processed} new images.")

        if image_dimensions:
            unique_dimensions = list(set(image_dimensions))
            
            if len(unique_dimensions) == 1:
                final_video_size = unique_dimensions[0]
                if verbose:
                    print(f"  All images have consistent dimensions: {final_video_size}")
            else:
                if verbose:
                    print(f"  Found {len(unique_dimensions)} different image dimensions: {unique_dimensions}")
                
                dimension_counts = {dim: image_dimensions.count(dim) for dim in unique_dimensions}
                target_size = max(dimension_counts, key=dimension_counts.get)
                
                if verbose:
                    print(f"  Most common dimension: {target_size} ({dimension_counts[target_size]}/{len(image_dimensions)} images)")
                    print(f"  Standardizing all images to: {target_size}")
                
                images_resized = 0
                # We need to find the original source images to resize from scratch
                source_map = {f"{video_id}_{parse_filename(p.name)[1]}": p for p in image_paths if parse_filename(p.name)[1]}

                for i, (jpeg_path, dims) in enumerate(zip(processed_jpeg_paths, image_dimensions)):
                    if dims != target_size:
                        # Find the original source image to re-process
                        base_id = jpeg_path.stem
                        original_path = source_map.get(base_id)
                        
                        if original_path:
                            process_and_save_jpeg(original_path, jpeg_path, target_size, True, False)
                            images_resized += 1
                
                if verbose:
                    print(f"  Resized {images_resized} images to match target dimensions")
                final_video_size = target_size
        else:
            final_video_size = None

        if verbose:
            print(f"  Successfully processed/validated {len(processed_jpeg_paths)} images for this well.")
        
        video_path = None
        if processed_jpeg_paths and final_video_size:
            if verbose:
                print(f"  Creating video from {len(processed_jpeg_paths)} frames...")
            video_path = vids_dir / f"{video_id}.mp4"
            create_video_from_jpegs(
                processed_jpeg_paths,
                video_path,
                final_video_size,
                overwrite,
                verbose
            )
        elif verbose:
            print(f"  Warning: No valid frames for video creation in well {well_id}")

        # Update metadata for the video
        if video_id not in metadata['video_ids']:
            metadata['video_ids'].append(video_id)

        # --- checkpoint logic ---
        global test_videos_written, TEST_METADATA_PATH
        test_videos_written += 1
        if test_videos_written % 100 == 0:
            with open(TEST_METADATA_PATH, 'w') as f:
                json.dump(metadata, f, indent=2)
            if verbose:
                print(f"[checkpoint] Saved metadata after {test_videos_written} videos")

        metadata['experiments'][experiment_id]['videos'][video_id] = {
            "video_id": video_id,
            "well_id": well_id,
            "mp4_path": str(video_path) if video_path else None,
            "processed_jpg_images_dir": str(video_frame_dir),
            "image_ids": sorted(image_ids),
            "total_source_images": len(image_paths),
            "valid_frames": len(processed_jpeg_paths),
            "video_resolution": list(final_video_size) if final_video_size else None,
            "last_processed_time": datetime.now().isoformat()
        }

    if warning_count >= max_warnings:
        print(f"\n... and {warning_count - max_warnings} more image reading warnings (suppressed)")
    
    if verbose:
        print(f"\nFinished processing experiment {experiment_id}.")
    
    return True

def load_or_initialize_metadata(path: Path) -> Dict:
    """Loads metadata from a JSON file or returns a new structure."""
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Basic validation
                if 'experiments' in data and 'image_ids' in data:
                    print(f"Loaded existing metadata from {path}")
                    return data
        except (json.JSONDecodeError, TypeError):
            print(f"Warning: Could not parse existing metadata file at {path}. Starting fresh.")
    
    print("Initializing new metadata file.")
    return {
        "script_version": "01_prepare_videos.py",
        "creation_time": datetime.now().isoformat(),
        "experiment_ids": [],
        "video_ids": [],
        "image_ids": [],
        "experiments": {}
    }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Incrementally convert raw stitched images into JPEG sequences and summary videos.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--directory_with_experiments", type=str, required=True,
        help="Path to the parent directory containing one or more experiment folders (e.g., .../stitched_images)."
    )
    parser.add_argument(
        "--output_parent_dir", type=str, required=True,
        help="Path to the parent output directory. A 'raw_data_organized' subfolder will be created here."
    )
    parser.add_argument(
        "--experiments_to_process", type=str,
        help="Comma-separated list of specific experiment folder names to process. If not provided, script will prompt to process all."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="If set, re-process and overwrite all existing images and videos."
    )
    parser.add_argument(
        '--verbose', dest='verbose', action='store_true',
        help="Enable detailed print statements (default: on)."
    )
    parser.add_argument(
        '--no-verbose', dest='verbose', action='store_false',
        help="Disable detailed print statements."
    )
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    experiments_dir = Path(args.directory_with_experiments)
    # All outputs will go into a 'raw_data_organized' sub-directory
    output_dir = Path(args.output_parent_dir) / "raw_data_organized"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not experiments_dir.is_dir():
        print(f"Error: Directory with experiments not found: {experiments_dir}")
        return 1

    # Load existing metadata or create a new one
    metadata_path = output_dir / "experiment_metadata.json"
    metadata = load_or_initialize_metadata(metadata_path)
    # set global path for checkpoints
    global TEST_METADATA_PATH
    TEST_METADATA_PATH = metadata_path

    # First, generate a list of all available experiment directories
    all_available_experiments = sorted([d for d in experiments_dir.iterdir() if d.is_dir()])
    if not all_available_experiments:
        print(f"No experiment directories found in {experiments_dir}")
        return 0
    
    available_experiment_names = [exp.name for exp in all_available_experiments]
    
    if args.verbose:
        print(f"Found {len(all_available_experiments)} experiment directories: {available_experiment_names}")

    # Determine which experiments to process
    experiments_to_process = []
    if args.experiments_to_process:
        # Process only specified experiments - filter the master list
        requested_names = [name.strip() for name in args.experiments_to_process.split(',')]
        for name in requested_names:
            if name not in available_experiment_names:
                print(f"Error: Specified experiment '{name}' not found in {experiments_dir}")
                print(f"Available experiments: {available_experiment_names}")
                return 1
            # Find the corresponding Path object
            exp_path = experiments_dir / name
            experiments_to_process.append(exp_path)
        
        if args.verbose:
            print(f"Processing {len(experiments_to_process)} specified experiments: {[p.name for p in experiments_to_process]}")
    else:
        # Prompt user to process all found experiments
        print("Found the following experiment directories:")
        for exp in all_available_experiments:
            print(f"  - {exp.name}")
        
        try:
            answer = input(f"\nDo you want to process all {len(all_available_experiments)} experiments? (y/n): ").lower()
            if answer == 'y':
                experiments_to_process = all_available_experiments
            else:
                print("Aborting.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nAborting.")
            return 0

    if not experiments_to_process:
        print("No experiments selected for processing.")
        return 0

    print(f"\nStarting batch processing for {len(experiments_to_process)} experiment(s).")
    print(f"Output directory: {output_dir}")
    print(f"Metadata file: {metadata_path}")
    
    successful_count = 0
    for exp_dir in experiments_to_process:
        success = process_experiment(exp_dir, output_dir, metadata, args.overwrite, args.verbose)
        if success:
            successful_count += 1

    # Save the updated metadata once at the very end
    if args.verbose:
        print(f"\nSaving updated metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBatch processing complete. Successfully processed {successful_count}/{len(experiments_to_process)} experiments.")
    print(f"Total experiments in metadata: {len(metadata['experiment_ids'])}")
    print(f"Total videos in metadata: {len(metadata['video_ids'])}")
    print(f"Total images in metadata: {len(metadata['image_ids'])}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
