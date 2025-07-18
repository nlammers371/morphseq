#!/usr/bin/env python3
"""
Export embryo masks from SAM2 annotations as labeled images
Also transfers QC flags from SAM annotations to EmbryoMetadata using GSAM ID matching
"""

import argparse
from pathlib import Path
import sys
import json

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
print("SCRIPT_DIR:", SCRIPT_DIR)
UTILS_DIR = SCRIPT_DIR / "utils"
sys.path.insert(0, str(UTILS_DIR))
print("sys.path:", sys.path)

from mask_export_utils import EmbryoMaskExporter
from embryo_metada_dev_instruction.embryo_metadata_refactored import EmbryoMetadata

def transfer_qc_flags_via_gsam_id(sam_annotations_path: Path, embryo_metadata_path: Path, verbose: bool = True):
    """
    Transfer QC flags from SAM annotations to EmbryoMetadata using GSAM ID matching.
    """
    # Load SAM annotations
    with open(sam_annotations_path, 'r') as f:
        sam_data = json.load(f)
    
    # Check if QC analysis exists in SAM annotations
    qc_analysis = sam_data.get("qc_analysis")
    if not qc_analysis:
        if verbose:
            print("ℹ️  No QC analysis found in SAM annotations")
        return 0
    
    sam_gsam_id = qc_analysis.get("gsam_annotation_id")
    qc_flags = qc_analysis.get("qc_flags", {})
    
    if not sam_gsam_id or not qc_flags:
        if verbose:
            print("ℹ️  No GSAM ID or QC flags found in SAM annotations")
        return 0
    
    # Load EmbryoMetadata
    embryo_metadata = EmbryoMetadata(
        sam_annotation_path=str(sam_annotations_path),
        embryo_metadata_path=str(embryo_metadata_path),
        verbose=verbose
    )
    
    # Check GSAM ID match
    metadata_gsam_id = embryo_metadata.results.get("source", {}).get("gsam_annotation_id")
    
    if sam_gsam_id != metadata_gsam_id:
        if verbose:
            print(f"⚠️  GSAM ID mismatch: SAM={sam_gsam_id}, Metadata={metadata_gsam_id}")
        return 0
    
    # Transfer QC flags to EmbryoMetadata
    flags_transferred = 0
    for entity_id, flag_list in qc_flags.items():
        for flag_dict in flag_list:
            flag = flag_dict.get("flag")
            author = flag_dict.get("author")
            details = flag_dict.get("details", "")
            
            embryo_metadata.add_flag(entity_id, flag, details, author)
            flags_transferred += 1
    
    # Save updated metadata
    embryo_metadata.save()
    
    if verbose:
        print(f"✅ Transferred {flags_transferred} QC flags from SAM annotations to EmbryoMetadata via GSAM ID {sam_gsam_id}")
    
    return flags_transferred

def main():
    parser = argparse.ArgumentParser(description="Export embryo masks as labeled images and transfer QC flags")
    parser.add_argument("--annotations", required=True, help="Path to grounded_sam_annotations.json")
    parser.add_argument("--embryo_metadata", required=True, help="Path to embryo_metadata.json")
    parser.add_argument("--output", required=True, help="Output directory for mask images")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], 
                       help="Output format (default: jpg, recommended: png for label masks)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🎭 Embryo Mask Export & QC Flag Transfer")
    print("=" * 50)
    
    if args.format == "jpg":
        print("⚠️  Warning: Using JPEG format for label masks may introduce compression artifacts.")
        print("   Consider using --format png for exact pixel value preservation.")
    
    # Step 1: Export masks
    print("\n📦 Step 1: Exporting embryo masks...")
    exporter = EmbryoMaskExporter(
        sam2_annotations_path=args.annotations,
        output_base_dir=args.output,
        verbose=args.verbose,
        output_format=args.format
    )
    
    export_paths = exporter.export_all_masks(max_workers=args.workers)
    print(f"✅ Exported {len(export_paths)} mask images")
    
    # Step 2: Transfer QC flags via GSAM ID matching
    print("\n🔄 Step 2: Transferring QC flags via GSAM ID...")
    try:
        flags_transferred = transfer_qc_flags_via_gsam_id(
            sam_annotations_path=Path(args.annotations),
            embryo_metadata_path=Path(args.embryo_metadata),
            verbose=args.verbose
        )
        
        if flags_transferred > 0:
            print(f"✅ Successfully transferred {flags_transferred} QC flags")
        else:
            print("ℹ️  No QC flags to transfer")
            
    except Exception as e:
        print(f"❌ QC flag transfer failed: {e}")
    
    print(f"\n📁 Output directory: {args.output}")
    print(f"📊 Updated metadata: {args.embryo_metadata}")

if __name__ == "__main__":
    main()


# Example usage:
# python scripts/05_export_embryo_masks.py \
#   --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/sam2_annotations/grounded_sam_annotations_finetuned.json \
#   --embryo_metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/embryo_metadata/embryo_metadata_finetuned.json \
#   --output  /net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox/data/annotation_and_masks/jpg_masks/grounded_sam_finetuned \
#   --format jpg \
#   --workers 16 \
#   --verbose
