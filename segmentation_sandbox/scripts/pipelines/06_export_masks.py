#!/usr/bin/env python3
"""
Export SAM2 masks as labeled images using SimpleMaskExporter.

Creates labeled PNG images where pixel value corresponds to embryo number.
Supports CRUD operations to avoid re-exporting existing masks.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.simple_mask_exporter import SimpleMaskExporter
from scripts.utils.parsing_utils import extract_experiment_id

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export SAM2 masks as labeled images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--sam2-annotations", 
        type=str, 
        required=True,
        help="Path to SAM2 annotations JSON file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output directory for exported masks"
    )
    
    # Optional arguments - entities filtering removed since input is per-experiment
    # parser.add_argument(
    #     "--entities-to-process", 
    #     type=str,
    #     help="Comma-separated list of experiment IDs to process (default: all)"
    # )
    parser.add_argument(
        "--export-format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "tiff"],
        help="Export format for mask images"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported masks"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without actually exporting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main export function."""
    args = parse_args()
    
    # Validate paths
    sam2_path = Path(args.sam2_annotations)
    if not sam2_path.exists():
        print(f"❌ SAM2 annotations file not found: {sam2_path}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    
    # Experiment filtering removed - input is already per-experiment
    # experiment_ids = None
    # if args.entities_to_process:
    #     experiment_ids = [exp.strip() for exp in args.entities_to_process.split(",")]
    #     if args.verbose:
    #         print(f"📋 Processing specific experiments: {experiment_ids}")
    
    # Initialize exporter
    if args.verbose:
        print(f"🚀 Initializing SimpleMaskExporter...")
        print(f"   📁 SAM2 annotations: {sam2_path}")
        print(f"   📁 Output directory: {output_dir}")
        print(f"   🎨 Export format: {args.export_format}")
    
    try:
        exporter = SimpleMaskExporter(
            sam2_path=sam2_path,
            output_dir=output_dir,
            format=args.export_format
        )
    except Exception as e:
        print(f"❌ Failed to initialize exporter: {e}")
        sys.exit(1)
    
    # Get export status
    try:
        status = exporter.get_export_status()
        
        if args.verbose:
            print(f"📊 Export status:")
            print(f"   Total images: {status['total_images']}")
            print(f"   Already exported: {status['exported_images']}")
            print(f"   Missing/new: {status['missing_images']}")
            print(f"   Available experiments: {status['available_experiments']}")
        
        # Experiment filtering removed - input is already per-experiment
        # All experiments in the SAM2 file will be processed
        
    except Exception as e:
        print(f"❌ Failed to get export status: {e}")
        sys.exit(1)
    
    # Dry run mode
    if args.dry_run:
        print(f"🧪 DRY RUN MODE - No files will be exported")
        
        if args.overwrite:
            print(f"   Would export ALL {status['total_images']} images (overwrite mode)")
        else:
            print(f"   Would export {status['missing_images']} missing images")
            
        # Experiment filtering removed - processing all experiments in per-experiment file
            
        print(f"   Output format: {args.export_format}")
        print(f"   Output directory: {output_dir}")
        return
    
    # Perform export
    try:
        if args.verbose:
            action = "overwriting all" if args.overwrite else "exporting missing"
            print(f"🔄 Starting export ({action} masks)...")
        
        exported = exporter.process_missing_masks(
            experiment_ids=None,  # Process all experiments in per-experiment file
            overwrite=args.overwrite
        )
        
        # Report results
        if exported:
            print(f"✅ Export completed successfully!")
            print(f"   📊 {len(exported)} images exported")
            
            if args.verbose:
                # Group by experiment for reporting
                by_experiment = {}
                for image_id in exported.keys():
                    exp_id = extract_experiment_id(image_id)
                    by_experiment.setdefault(exp_id, []).append(image_id)
                
                for exp_id, images in by_experiment.items():
                    print(f"   📁 {exp_id}: {len(images)} images")
            
            # Final status
            final_status = exporter.get_export_status()
            print(f"   📈 Total exported: {final_status['exported_images']}/{final_status['total_images']} images")
            
        else:
            print(f"ℹ️  No new masks to export")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()