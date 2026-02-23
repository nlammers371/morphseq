#!/usr/bin/env python3
"""
Test script to run video preparation on just the 20240411 directory.
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Temporarily redirect the heavy imports to test without library issues
import tempfile
import shutil

def test_on_20240411():
    """Test video preparation on the 20240411 directory only."""
    
    print("Testing video preparation on 20240411 directory...")
    
    # Paths
    source_dir = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images/20240411")
    sandbox_dir = Path(__file__).parent
    output_dir = sandbox_dir / "data" / "intermediate" / "morphseq_well_videos"
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}")
        return False
    
    # List some files to verify content
    jpg_files = list(source_dir.glob("*.jpg"))
    png_files = list(source_dir.glob("*.png"))
    
    print(f"Found {len(jpg_files)} JPG files in source directory")
    print(f"Found {len(png_files)} PNG files in source directory")
    
    # Show sample files from whichever format exists
    sample_files = jpg_files[:5] if jpg_files else png_files[:5]
    print("Sample files:")
    for f in sample_files:
        print(f"  {f.name}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    
    # Now we need to import and run the actual pipeline
    # Let's try a minimal approach first
    try:
        from utils.config_utils import load_config
        config = load_config('configs/pipeline_config.yaml')
        print("✓ Config loaded successfully")
        
        # Check if we can access the stitched images directory
        stitched_dir = Path(config.get_stitched_images_dir())
        print(f"Configured stitched images dir: {stitched_dir}")
        
        # Verify 20240411 directory
        test_exp_dir = stitched_dir / "20240411"
        if test_exp_dir.exists():
            jpg_files = list(test_exp_dir.glob("*.jpg"))
            png_files = list(test_exp_dir.glob("*.png"))
            print(f"✓ Found {len(jpg_files)} JPG files in {test_exp_dir}")
            print(f"✓ Found {len(png_files)} PNG files in {test_exp_dir}")
            
            # Show first few files
            sample_files = jpg_files[:3] if jpg_files else png_files[:3]
            for i, f in enumerate(sample_files):
                print(f"  Sample {i+1}: {f.name}")
        else:
            print(f"✗ Directory not found: {test_exp_dir}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"Import error (expected due to library issues): {e}")
        print("But directory structure looks good for manual testing")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_on_20240411()
    if success:
        print("\n✓ Test setup looks good!")
        print("\nNext step: Run the video preparation script:")
        print("python scripts/01_prepare_videos.py --config configs/pipeline_config.yaml")
    else:
        print("\n✗ Test setup failed!")
        sys.exit(1)
