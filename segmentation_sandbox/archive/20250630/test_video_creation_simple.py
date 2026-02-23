#!/usr/bin/env python3
"""
Minimal video preparation script for testing 20240411 data.
Avoids heavy imports that cause library conflicts.
"""

import os
import sys
import json
import yaml
import cv2
from pathlib import Path
from datetime import datetime
import re
from typing import List, Dict

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = "configs/pipeline_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from directory."""
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)

def extract_well_id(filename: str) -> str:
    """Extract well ID from filename like A01_t0000_ch00_stitch.png -> A01"""
    basename = Path(filename).stem
    
    # Pattern 1: A01_t0000_ch00_stitch -> A01 (current data format)
    well_match = re.search(r'^([A-H]\d{2})_t\d+', basename.upper())
    if well_match:
        return well_match.group(1)
    
    # Pattern 2: well_A01_t001 -> A01 (legacy format)
    well_match = re.search(r'well[_\-]?([A-H]\d{2})', basename.upper())
    if well_match:
        return well_match.group(1)
    
    # Pattern 3: A01_t001 -> A01 (simple format)
    direct_match = re.search(r'([A-H]\d{2})', basename.upper())
    if direct_match:
        return direct_match.group(1)
    
    # Fallback: use first part before underscore
    parts = basename.split('_')
    if parts:
        return parts[0].upper()
    
    return basename.upper()

def extract_timepoint(filename: str) -> int:
    """Extract timepoint number from filename."""
    # Pattern 1: t followed by numbers (t0000, t0001, t123)
    t_match = re.search(r't(\d+)', filename.lower())
    if t_match:
        return int(t_match.group(1))
    
    # Pattern 2: timepoint followed by numbers
    tp_match = re.search(r'timepoint[_\-]?(\d+)', filename.lower())
    if tp_match:
        return int(tp_match.group(1))
    
    # Pattern 3: frame followed by numbers
    frame_match = re.search(r'frame[_\-]?(\d+)', filename.lower())
    if frame_match:
        return int(frame_match.group(1))
    
    # Fallback: last number in filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])
    
    return 0

def analyze_experiment_directory(exp_dir: Path) -> Dict:
    """Analyze the structure of an experiment directory."""
    print(f"\nAnalyzing experiment directory: {exp_dir.name}")
    
    # Get all image files
    image_files = get_image_files(exp_dir)
    print(f"Total image files: {len(image_files)}")
    
    if not image_files:
        return {"experiment_id": exp_dir.name, "sequences": {}, "total_files": 0}
    
    # Show file format distribution
    extensions = {}
    for img in image_files:
        ext = img.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1
    
    print("File format distribution:")
    for ext, count in extensions.items():
        print(f"  {ext}: {count} files")
    
    # Group by well
    sequences = {}
    for image_path in image_files:
        well_id = extract_well_id(image_path.name)
        if well_id not in sequences:
            sequences[well_id] = []
        sequences[well_id].append(image_path)
    
    # Sort each sequence by timepoint
    for well_id in sequences:
        sequences[well_id] = sorted(sequences[well_id], 
                                  key=lambda x: extract_timepoint(x.name))
    
    print(f"Number of wells found: {len(sequences)}")
    
    # Show sample wells and their frame counts
    sample_wells = list(sequences.keys())[:5]
    print("Sample wells:")
    for well in sample_wells:
        frames = len(sequences[well])
        first_file = sequences[well][0].name if sequences[well] else "None"
        last_file = sequences[well][-1].name if sequences[well] else "None"
        print(f"  {well}: {frames} frames ({first_file} -> {last_file})")
    
    return {
        "experiment_id": exp_dir.name,
        "sequences": sequences,
        "total_files": len(image_files),
        "extensions": extensions,
        "wells": list(sequences.keys())
    }

def test_single_video_creation(sequences: Dict, well_id: str, output_dir: Path, experiment_id: str):
    """Test creating a single video for one well."""
    if well_id not in sequences:
        print(f"Well {well_id} not found in sequences")
        return False
    
    image_paths = sequences[well_id]
    video_id = f"{experiment_id}_{well_id}"
    
    print(f"\nTesting video creation for {video_id}")
    print(f"Number of frames: {len(image_paths)}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_id}.mp4"
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        print(f"Could not read first image: {image_paths[0]}")
        return False
    
    height, width = first_image.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 2
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Could not open video writer")
        return False
    
    # Process images
    valid_frames = 0
    print("Processing frames...")
    
    for i, image_path in enumerate(image_paths):
        if i % 10 == 0:  # Progress every 10 frames
            print(f"  Frame {i+1}/{len(image_paths)}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Warning: Could not read frame {i}: {image_path}")
            continue
        
        # Resize if needed
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        
        video_writer.write(image)
        valid_frames += 1
    
    video_writer.release()
    
    print(f"✓ Video created: {output_video_path}")
    print(f"✓ Valid frames: {valid_frames}/{len(image_paths)}")
    
    # Create metadata
    metadata = {
        video_id: {
            "experiment_id": experiment_id,
            "video_id": video_id,
            "video_path": str(output_video_path),
            "source_images": [str(p) for p in image_paths],
            "total_source_images": len(image_paths),
            "valid_frames": valid_frames,
            "video_fps": fps,
            "resolution": [width, height],
            "creation_time": datetime.now().isoformat(),
            "jpeg_conversion": False,
            "jpeg_quality": None
        }
    }
    
    # Save metadata
    metadata_file = output_dir / "video_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved: {metadata_file}")
    
    return True

def main():
    """Main function to test on 20240411 data."""
    print("=== Testing Video Preparation on 20240411 Data ===")
    
    # Load config
    try:
        config = load_config()
        morphseq_data_dir = Path(config['paths']['morphseq_data_dir'])
        stitched_images_dir = morphseq_data_dir / config['paths']['stitched_images_dir']
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1
    
    # Test directory
    test_exp_dir = stitched_images_dir / "20240411"
    
    if not test_exp_dir.exists():
        print(f"Test directory does not exist: {test_exp_dir}")
        return 1
    
    # Analyze the directory
    analysis = analyze_experiment_directory(test_exp_dir)
    
    if analysis["total_files"] == 0:
        print("No image files found!")
        return 1
    
    # Output directory
    output_dir = Path("data/intermediate/morphseq_well_videos")
    
    # Test creating a video for the first well
    if analysis["wells"]:
        test_well = analysis["wells"][0]  # Use first well for testing
        print(f"\nTesting with well: {test_well}")
        
        success = test_single_video_creation(
            analysis["sequences"], 
            test_well, 
            output_dir, 
            analysis["experiment_id"]
        )
        
        if success:
            print("\n✓ Single video test successful!")
            print(f"\nTo process all wells, you can modify this script or run:")
            print(f"  python scripts/01_prepare_videos.py --config configs/pipeline_config.yaml")
        else:
            print("\n✗ Single video test failed!")
            return 1
    else:
        print("No wells found in analysis!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
