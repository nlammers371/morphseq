#!/usr/bin/env python
"""
manual_image_quality_qc.py

Command-line script for manual image quality control flags in the MorphSeq pipeline.
Allows flagging or removing QC annotations for images.

Usage Examples:
    # Flag specific images as FAIL
    python scripts/manual_image_quality_qc.py flag \
        --data_dir /path/to/data \
        --image_ids 20241215_A01_t001,20241215_A01_t002 \
        --qc_flag FAIL \
        --annotator nlammers \
        --notes "Embryo out of focus"

    # Flag all frames in a video as BLUR
    python scripts/manual_image_quality_qc.py flag \
        --data_dir /path/to/data \
        --video_id 20241215_A01 \
        --frames t001,t002,t003 \
        --qc_flag BLUR \
        --annotator mcolon

    # Remove QC flags from specific images
    python scripts/manual_image_quality_qc.py remove \
        --data_dir /path/to/data \
        --image_ids 20241215_A01_t001 \
        --annotator nlammers

    # Check QC status of images
    python scripts/manual_image_quality_qc.py check \
        --data_dir /path/to/data \
        --image_ids 20241215_A01_t001,20241215_A01_t002
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'image_quality_qc_utils'))
from image_qc_utils import (
    flag_qc, remove_qc, check_existing_qc, get_qc_summary, 
    QC_FLAGS, load_qc_data, generate_comprehensive_qc
)

def validate_inputs(args):
    """Validate command line arguments."""
    if args.mode in ['flag', 'remove']:
        # Must have either image_ids OR (video_id + frames)
        has_image_ids = args.image_ids is not None
        has_video_frames = args.video_id is not None and args.frames is not None
        
        if not (has_image_ids or has_video_frames):
            raise ValueError("Must provide either --image_ids OR (--video_id + --frames)")
        
        if has_image_ids and has_video_frames:
            raise ValueError("Cannot provide both --image_ids AND (--video_id + --frames)")
        
        # Annotator is required
        if not args.annotator:
            raise ValueError("--annotator is required for flag and remove operations")
        
        # QC flag is required for flagging
        if args.mode == 'flag' and not args.qc_flag:
            raise ValueError("--qc_flag is required for flag operation")
        
        # Validate QC flag
        if args.mode == 'flag' and args.qc_flag not in QC_FLAGS:
            valid_flags = list(QC_FLAGS.keys())
            raise ValueError(f"Invalid QC flag '{args.qc_flag}'. Valid flags: {valid_flags}")

def parse_list_arg(arg_string: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string into list."""
    if arg_string is None:
        return None
    return [item.strip() for item in arg_string.split(',')]

def flag_images(args):
    """Flag images with QC annotations."""
    image_ids = parse_list_arg(args.image_ids)
    frames = parse_list_arg(args.frames)
    
    print(f"Flagging images with QC flag: {args.qc_flag}")
    print(f"Annotator: {args.annotator}")
    if args.notes:
        print(f"Notes: {args.notes}")
    
    try:
        qc_df = flag_qc(
            data_dir=args.data_dir,
            image_ids=image_ids,
            video_id=args.video_id,
            frames=frames,
            qc_flag=args.qc_flag,
            annotator=args.annotator,
            notes=args.notes or '',
            overwrite=args.overwrite
        )
        
        print("QC flags successfully added!")
        
        # Show summary
        summary = get_qc_summary(args.data_dir)
        print(f"\nQC Summary: {summary['total_images']} total images")
        print("Flag counts:", summary['qc_flags'])
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def remove_flags(args):
    """Remove QC flags from images."""
    image_ids = parse_list_arg(args.image_ids)
    frames = parse_list_arg(args.frames)
    
    print(f"Removing QC flags from images")
    print(f"Annotator: {args.annotator}")
    
    try:
        qc_df = remove_qc(
            data_dir=args.data_dir,
            image_ids=image_ids,
            video_id=args.video_id,
            frames=frames,
            annotator=args.annotator
        )
        
        print("QC flags successfully removed!")
        
        # Show summary
        summary = get_qc_summary(args.data_dir)
        print(f"\nQC Summary: {summary['total_images']} total images")
        print("Flag counts:", summary['qc_flags'])
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def check_qc_status(args):
    """Check QC status of specific images."""
    image_ids = parse_list_arg(args.image_ids)
    
    if not image_ids:
        print("Error: --image_ids required for check operation")
        return 1
    
    try:
        qc_status = check_existing_qc(args.data_dir, image_ids)
        
        print("QC Status:")
        for image_id, qc_flag in qc_status.items():
            status = qc_flag if qc_flag else "No QC flag"
            print(f"  {image_id}: {status}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def show_summary(args):
    """Show QC summary for the dataset."""
    try:
        summary = get_qc_summary(args.data_dir)
        
        print("=== QC Summary ===")
        print(f"Total images with QC flags: {summary['total_images']}")
        print(f"Experiments: {summary['experiments']}")
        print(f"Videos: {summary['videos']}")
        
        print("\nQC Flag Distribution:")
        for flag, count in summary['qc_flags'].items():
            description = QC_FLAGS.get(flag, "Unknown flag")
            print(f"  {flag}: {count} ({description})")
        
        print("\nAnnotator Distribution:")
        for annotator, count in summary['annotators'].items():
            print(f"  {annotator}: {count}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def generate_comprehensive(args):
    """Generate comprehensive QC file with all images."""
    try:
        metadata_path = Path(args.data_dir) / "experiment_metadata.json"
        if not metadata_path.exists():
            print(f"Error: Metadata file not found: {metadata_path}")
            print("Run 01_prepare_videos.py first to generate metadata")
            return 1
        
        print("Generating comprehensive QC file...")
        comprehensive_df = generate_comprehensive_qc(args.data_dir, metadata_path)
        
        print(f"Generated comprehensive QC file with {len(comprehensive_df)} images")
        
        # Show breakdown
        pass_count = (comprehensive_df['qc_flag'] == 'PASS').sum()
        fail_count = len(comprehensive_df) - pass_count
        print(f"  PASS: {pass_count}")
        print(f"  Other flags: {fail_count}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Manual image quality control for MorphSeq pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory containing quality_control folder"
    )
    
    # Flag command
    flag_parser = subparsers.add_parser('flag', parents=[parent_parser], help='Add QC flags to images')
    flag_parser.add_argument(
        "--image_ids", type=str,
        help="Comma-separated list of image IDs to flag"
    )
    flag_parser.add_argument(
        "--video_id", type=str,
        help="Video ID (use with --frames)"
    )
    flag_parser.add_argument(
        "--frames", type=str,
        help="Comma-separated list of frame/timepoint identifiers (use with --video_id)"
    )
    flag_parser.add_argument(
        "--qc_flag", type=str, required=True,
        choices=list(QC_FLAGS.keys()),
        help="QC flag to assign"
    )
    flag_parser.add_argument(
        "--annotator", type=str, required=True,
        help="Annotator name (e.g., 'nlammers', 'mcolon')"
    )
    flag_parser.add_argument(
        "--notes", type=str,
        help="Optional notes about the QC decision"
    )
    flag_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing QC entries"
    )
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', parents=[parent_parser], help='Remove QC flags from images')
    remove_parser.add_argument(
        "--image_ids", type=str,
        help="Comma-separated list of image IDs to remove flags from"
    )
    remove_parser.add_argument(
        "--video_id", type=str,
        help="Video ID (use with --frames)"
    )
    remove_parser.add_argument(
        "--frames", type=str,
        help="Comma-separated list of frame/timepoint identifiers (use with --video_id)"
    )
    remove_parser.add_argument(
        "--annotator", type=str, required=True,
        help="Annotator name for logging removal"
    )
    
    # Check command
    check_parser = subparsers.add_parser('check', parents=[parent_parser], help='Check QC status of images')
    check_parser.add_argument(
        "--image_ids", type=str, required=True,
        help="Comma-separated list of image IDs to check"
    )
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', parents=[parent_parser], help='Show QC summary')
    
    # Comprehensive command
    comprehensive_parser = subparsers.add_parser(
        'comprehensive', parents=[parent_parser], 
        help='Generate comprehensive QC file with all images'
    )
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return 1
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    try:
        # Validate inputs
        validate_inputs(args)
        
        # Execute based on mode
        if args.mode == 'flag':
            return flag_images(args)
        elif args.mode == 'remove':
            return remove_flags(args)
        elif args.mode == 'check':
            return check_qc_status(args)
        elif args.mode == 'summary':
            return show_summary(args)
        elif args.mode == 'comprehensive':
            return generate_comprehensive(args)
        else:
            print(f"Unknown mode: {args.mode}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
