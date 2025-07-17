#!/usr/bin/env python3
"""
Export embryo masks from SAM2 annotations as labeled images
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
UTILS_DIR = SCRIPT_DIR / "utils"
sys.path.insert(0, str(UTILS_DIR))
print("sys.path:", sys.path)

from mask_export_utils import EmbryoMaskExporter

def main():
    parser = argparse.ArgumentParser(description="Export embryo masks as labeled images")
    parser.add_argument("--annotations", required=True, help="Path to grounded_sam_annotations.json")
    parser.add_argument("--output", required=True, help="Output directory for mask images")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], 
                       help="Output format (default: jpg, recommended: png for label masks)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    print("🎭 Embryo Mask Export")
    print("=" * 40)
    if args.format == "jpg":
        print("⚠️  Warning: Using JPEG format for label masks may introduce compression artifacts.")
        print("   Consider using --format png for exact pixel value preservation.")
    # Initialize exporter
    exporter = EmbryoMaskExporter(
        sam2_annotations_path=args.annotations,
        output_base_dir=args.output,
        verbose=args.verbose,
        output_format=args.format
    )
    # Export all masks
    export_paths = exporter.export_all_masks(max_workers=args.workers)
    print(f"\n✅ Exported {len(export_paths)} mask images")
    print(f"📁 Output directory: {args.output}")

if __name__ == "__main__":
    main()


# python /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/scripts/05_export_embryo_masks.py --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_ft_annotations_test.json --output /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/jpg_masks --format jpg --workers 4 --verbose