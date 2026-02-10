#!/usr/bin/env python3
"""
Pipeline Script 5: Analysis and Export

Export embryo masks and perform final analysis.
Processes all segmentation results by default.
"""

import argparse
import sys
import json
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from metadata.experiment_metadata import ExperimentMetadata

def main():
    parser = argparse.ArgumentParser(
        description="Export masks and analyze results for MorphSeq pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all segmentation results
  python 05_analysis_export.py --metadata experiment_metadata.json \\
    --segmentations segmentations.json --output /path/to/export
  
  # Export specific experiments only
  python 05_analysis_export.py --metadata experiment_metadata.json \\
    --segmentations segmentations.json --output /path/to/export \\
    --experiments "20240506,20250703_chem3_28C_T00_1325"
  
  # Custom export format
  python 05_analysis_export.py --metadata experiment_metadata.json \\
    --segmentations segmentations.json --output /path/to/export \\
    --format jpg --workers 16
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--metadata", 
        required=True, 
        help="Path to experiment_metadata.json from step 1"
    )
    parser.add_argument(
        "--segmentations", 
        required=True, 
        help="Path to segmentation annotations JSON from step 4"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output directory for exported masks and analysis"
    )
    
    # Optional arguments
    parser.add_argument(
        "--embryo-metadata", 
        help="Path to embryo metadata JSON file"
    )
    parser.add_argument(
        "--experiments", 
        help="Comma-separated experiment IDs to process (default: all)"
    )
    parser.add_argument(
        "--format", 
        choices=["jpg", "png", "both"], 
        default="jpg", 
        help="Export format for masks (default: jpg)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=8, 
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--quality-threshold", 
        type=float, 
        default=0.5, 
        help="Minimum quality score for exported masks (default: 0.5)"
    )
    parser.add_argument(
        "--min-mask-area", 
        type=int, 
        default=100, 
        help="Minimum mask area in pixels (default: 100)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be exported without creating files"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    metadata_path = Path(args.metadata).resolve()
    segmentations_path = Path(args.segmentations).resolve()
    output_dir = Path(args.output).resolve()
    
    if not metadata_path.exists():
        print(f"❌ Error: Metadata file does not exist: {metadata_path}")
        sys.exit(1)
    
    if not segmentations_path.exists():
        print(f"❌ Error: Segmentations file does not exist: {segmentations_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("📋 Loading experiment metadata...")
    try:
        meta = ExperimentMetadata(str(metadata_path))
        print(f"✅ Loaded metadata with {len(meta.metadata['experiments'])} experiments")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        sys.exit(1)
    
    # Load segmentation results
    print("🎭 Loading segmentation results...")
    try:
        with open(segmentations_path, 'r') as f:
            segmentation_data = json.load(f)
        
        segmentations = segmentation_data.get('segmentations', {})
        tracking_results = segmentation_data.get('tracking_results', {})
        print(f"🎭 Found segmentations for {len(segmentations)} images")
        print(f"🔗 Found tracking for {len(tracking_results)} sequences")
        
        if len(segmentations) == 0:
            print("⚠️  No segmentations found - check segmentation results")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error loading segmentations: {e}")
        sys.exit(1)
    
    # Parse experiment filter
    experiment_ids = None
    if args.experiments:
        experiment_ids = [e.strip() for e in args.experiments.split(",")]
        print(f"📌 Processing specific experiments: {experiment_ids}")
    else:
        print("📌 Processing ALL experiments in metadata")
    
    # Load embryo metadata if provided
    embryo_metadata = None
    if args.embryo_metadata:
        embryo_path = Path(args.embryo_metadata).resolve()
        if embryo_path.exists():
            try:
                with open(embryo_path, 'r') as f:
                    embryo_metadata = json.load(f)
                print(f"🧬 Loaded embryo metadata: {len(embryo_metadata.get('embryos', {}))} embryos")
            except Exception as e:
                print(f"⚠️  Warning: Could not load embryo metadata: {e}")
        else:
            print(f"⚠️  Warning: Embryo metadata file not found: {embryo_path}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Export plan:")
        print(f"  🎭 Segmentations: {len(segmentations)}")
        print(f"  🔗 Tracking results: {len(tracking_results)}")
        print(f"  🧬 Embryo metadata: {'Yes' if embryo_metadata else 'No'}")
        print(f"  📸 Export format: {args.format}")
        print(f"  👥 Workers: {args.workers}")
        print(f"  📊 Quality threshold: {args.quality_threshold}")
        print(f"  📏 Min mask area: {args.min_mask_area}")
        print(f"  💾 Output: {output_dir}")
        
        # Show sample of what would be exported
        print(f"\n📸 Sample exports (first 3):")
        for i, (image_id, seg_data) in enumerate(list(segmentations.items())[:3]):
            masks = seg_data.get('masks', [])
            print(f"  {i+1}. {image_id}")
            print(f"     🎭 {len(masks)} masks")
        
        if len(segmentations) > 3:
            print(f"  ... and {len(segmentations) - 3} more images")
        
        return
    
    print("🚀 Starting mask export and analysis...")
    print(f"📸 Export format: {args.format}")
    print(f"👥 Workers: {args.workers}")
    print(f"📊 Quality threshold: {args.quality_threshold}")
    print(f"📏 Min mask area: {args.min_mask_area}")
    
    try:
        # TODO: Import and use actual mask exporter
        # For now, create placeholder
        print("⚠️  Mask export and analysis not yet implemented")
        print("📝 This script will call the export logic once Module 4 is complete")
        
        # Create placeholder export structure
        from datetime import datetime
        
        # Create export directories
        (output_dir / "masks").mkdir(exist_ok=True)
        (output_dir / "analysis").mkdir(exist_ok=True)
        
        # Create placeholder analysis file
        analysis_results = {
            "file_info": {
                "creation_time": datetime.now().isoformat(),
                "script_version": "05_analysis_export.py",
                "metadata_source": str(metadata_path),
                "segmentation_source": str(segmentations_path),
                "processing_parameters": {
                    "format": args.format,
                    "workers": args.workers,
                    "quality_threshold": args.quality_threshold,
                    "min_mask_area": args.min_mask_area
                },
                "processing_status": "placeholder"
            },
            "export_summary": {
                "total_images": len(segmentations),
                "exported_masks": 0,
                "total_embryos": 0,
                "quality_metrics": {}
            },
            "exported_files": []
        }
        
        analysis_path = output_dir / "analysis" / "export_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"📄 Created placeholder analysis: {analysis_path}")
        print(f"📁 Created export directories: {output_dir}")
        print(f"✅ Export and analysis complete!")
        print(f"🎉 Pipeline finished! Check {output_dir} for results")
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
