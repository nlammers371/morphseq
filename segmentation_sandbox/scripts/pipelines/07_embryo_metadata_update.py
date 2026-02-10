#!/usr/bin/env python3
"""
Pipeline Script 7: Embryo Metadata Update

Initialize or update biological annotations (phenotypes, genotypes, treatments)
based on SAM2 segmentation results.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Optional imports with fallbacks
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# Pipeline imports
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.base_file_handler import BaseFileHandler

def create_data_directory(base_path: Path) -> Path:
    """Create data/embryo_metadata directory if it doesn't exist."""
    metadata_dir = base_path / "data" / "embryo_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir

def get_default_output_path(sam2_path: Path, output_dir: Path) -> Path:
    """Generate default output path based on SAM2 filename."""
    base_name = sam2_path.stem.replace('_sam2', '').replace('_annotations', '')
    return output_dir / f"{base_name}_biology.json"

def extract_frame_number(image_id: str) -> int:
    """Extract frame number from image_id like '20240418_A01_t0100' -> 100."""
    if '_t' in image_id:
        frame_part = image_id.split('_t')[-1]
        return int(frame_part)
    else:
        raise ValueError(f"Cannot extract frame number from image_id: {image_id}")

def create_embryo_structure(embryo_id: str) -> Dict:
    """Create empty embryo structure for annotations."""
    # Extract experiment and video IDs from embryo_id
    parts = embryo_id.split('_')
    if len(parts) >= 3:
        experiment_id = parts[0]
        video_id = '_'.join(parts[:2])
    else:
        experiment_id = "unknown"
        video_id = "unknown"
    
    return {
        "embryo_id": embryo_id,
        "experiment_id": experiment_id,
        "video_id": video_id,
        "genotype": None,
        "treatments": [],
        "snips": {}
    }

def create_snip_structure(snip_id: str, frame_number: int) -> Dict:
    """Create empty snip structure for frame-level annotations."""
    return {
        "snip_id": snip_id,
        "frame_number": frame_number,
        "phenotypes": [],
        "flags": []
    }

def extract_embryos_from_sam2(sam2_data: Dict) -> Dict[str, Dict]:
    """
    Extract embryo structure from SAM2 data.
    
    Returns:
        Dict mapping embryo_id to embryo structure with all snips
    """
    embryos = {}
    
    # Scan through experiments -> videos -> images -> embryos
    for exp_id, exp_data in sam2_data.get("experiments", {}).items():
        for video_id, video_data in exp_data.get("videos", {}).items():
            for image_id, image_data in video_data.get("images", {}).items():
                try:
                    frame_num = extract_frame_number(image_id)
                except ValueError:
                    print(f"Warning: Skipping image with invalid frame format: {image_id}")
                    continue
                
                for embryo_id in image_data.get("embryos", {}):
                    # Create embryo structure if first time seeing this embryo
                    if embryo_id not in embryos:
                        embryos[embryo_id] = create_embryo_structure(embryo_id)
                    
                    # Add snip for this frame
                    snip_id = f"{embryo_id}_s{frame_num:04d}"
                    embryos[embryo_id]["snips"][snip_id] = create_snip_structure(snip_id, frame_num)
    
    return embryos

def create_metadata_structure(sam2_path: Path, sam2_data: Dict) -> Dict:
    """Create the complete metadata structure from SAM2 data."""
    embryos = extract_embryos_from_sam2(sam2_data)
    
    return {
        "metadata": {
            "source_sam2": str(sam2_path),
            "created": datetime.now().isoformat(),
            "version": "simplified_v1"
        },
        "embryos": embryos
    }

def update_existing_metadata(existing_data: Dict, sam2_data: Dict) -> Dict:
    """
    Update existing metadata with new embryos from SAM2.
    Preserves existing annotations while adding new embryos/snips.
    """
    # Extract new embryos from SAM2
    new_embryos = extract_embryos_from_sam2(sam2_data)
    
    # Update metadata timestamp
    existing_data["metadata"]["updated"] = datetime.now().isoformat()
    
    # Merge embryos
    for embryo_id, embryo_structure in new_embryos.items():
        if embryo_id not in existing_data["embryos"]:
            # New embryo - add complete structure
            existing_data["embryos"][embryo_id] = embryo_structure
            print(f"Added new embryo: {embryo_id}")
        else:
            # Existing embryo - only add new snips
            existing_snips = existing_data["embryos"][embryo_id].get("snips", {})
            new_snips = embryo_structure.get("snips", {})
            
            added_count = 0
            for snip_id, snip_data in new_snips.items():
                if snip_id not in existing_snips:
                    existing_snips[snip_id] = snip_data
                    added_count += 1
            
            if added_count > 0:
                print(f"Added {added_count} new snips to embryo: {embryo_id}")
    
    return existing_data

def main():
    parser = argparse.ArgumentParser(
        description="Initialize or update embryo metadata from SAM2 annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize new metadata file
  python 07_embryo_metadata_update.py sam2_annotations.json
  
  # Update existing metadata with new embryos/frames
  python 07_embryo_metadata_update.py sam2_annotations.json --output existing_biology.json
  
  # Specify custom output directory
  python 07_embryo_metadata_update.py sam2_annotations.json --output-dir /custom/path
        """
    )
    
    parser.add_argument(
        "sam2_file",
        type=Path,
        help="Path to SAM2 annotations JSON file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for biology annotations (default: auto-generated in data/embryo_metadata/)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory for data/embryo_metadata/ (default: current directory)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.sam2_file.exists():
        print(f"Error: SAM2 file not found: {args.sam2_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_data_directory(args.output_dir)
    
    # Determine output path
    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = output_dir / output_path
    else:
        output_path = get_default_output_path(args.sam2_file, output_dir)
    
    print(f"SAM2 input: {args.sam2_file}")
    print(f"Output path: {output_path}")
    
    # Check if output exists
    if output_path.exists() and not args.force and not args.dry_run:
        print(f"Output file exists: {output_path}")
        print("Use --force to overwrite or --dry-run to preview")
        sys.exit(1)
    
    try:
        # Load SAM2 data
        print("Loading SAM2 data...")
        file_handler = BaseFileHandler(args.sam2_file)
        sam2_data = file_handler.load_json()
        
        if args.dry_run:
            print("\nDRY RUN MODE - No files will be written")
        
        # Determine operation mode
        if output_path.exists() and not args.force:
            # Update existing metadata
            print("Updating existing metadata...")
            existing_handler = BaseFileHandler(output_path)
            existing_data = existing_handler.load_json()
            result_data = update_existing_metadata(existing_data, sam2_data)
            operation = "update"
        else:
            # Create new metadata
            print("Creating new metadata structure...")
            result_data = create_metadata_structure(args.sam2_file, sam2_data)
            operation = "create"
        
        # Count embryos and snips
        embryo_count = len(result_data["embryos"])
        total_snips = sum(len(embryo["snips"]) for embryo in result_data["embryos"].values())
        
        print(f"\nResult summary:")
        print(f"Operation: {operation}")
        print(f"Embryos: {embryo_count}")
        print(f"Total snips: {total_snips}")
        
        if not args.dry_run:
            # Save result
            output_handler = BaseFileHandler(output_path)
            output_handler.save_json(result_data)
            print(f"\nSaved metadata to: {output_path}")
        else:
            print(f"\nWould save metadata to: {output_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()