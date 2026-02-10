#!/usr/bin/env python3
"""
Pipeline Script 1: Prepare Videos and Create Metadata

Organizes raw stitched images into standard structure and creates videos.
This script processes all experiments in a directory by default.
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent  # Go up to scripts/ directory
sys.path.insert(0, str(SCRIPTS_DIR))

def check_entity_tracking_complete(metadata_path):
    """Check if entity tracking is properly initialized in metadata."""
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if entity_tracker section exists
        if 'entity_tracker' not in metadata:
            return False
            
        tracker = metadata['entity_tracker']
        
        # Check if it has required fields
        required_fields = ['entities', 'last_updated']
        if not all(field in tracker for field in required_fields):
            return False
            
        # Check if entities are populated
        entities = tracker.get('entities', {})
        if not entities.get('experiments') and metadata.get('experiments'):
            return False
            
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Organize raw data and create videos for MorphSeq pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all experiments in directory using actual paths (creates videos and metadata)
  python 01_prepare_videos.py \\
    --directory_with_experiments /net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images \\
    --output_parent_dir data \\
    --workers 8 \\
    --verbose
  
  # Process specific experiments only (good for testing with real experiment names)
  python 01_prepare_videos.py \\
    --directory_with_experiments /net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images \\
    --output_parent_dir data \\
    --experiments_to_process "20231206,20240418,20250612_30hpf_ctrl_atf6" \\
    --verbose

  python 01_prepare_videos.py \\
    --directory_with_experiments /net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images \\
    --output_parent_dir data \\
    --experiments_to_process "20250612_30hpf_ctrl_atf6" \\
    --verbose
  
  # Dry-run to see what would be processed (shows all 90 available experiments)
  python 01_prepare_videos.py \\
    --directory_with_experiments /net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images \\
    --output_parent_dir data \\
    --dry-run \\
    --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--directory_with_experiments", 
        required=True, 
        help="Root directory containing stitched images organized by experiment"
    )
    parser.add_argument(
        "--output_parent_dir", 
        required=True, 
        help="Output directory (will create raw_data_organized/ subdirectory)"
    )
    
        # Optional arguments
    parser.add_argument(
        "--experiments_to_process", 
        help="Comma-separated list of experiment names to process (default: all)"
    )
    parser.add_argument(
        "--entities_to_process", 
        help="Comma-separated list of entities to process (experiments, videos, or images)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=8, 
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--save-interval", 
        type=int, 
        default=1, 
        help="Auto-save metadata every N experiments (default: 1, use 0 to disable)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be processed without actually doing it"
    )
    
    args = parser.parse_args()
    
        # Convert paths to Path objects
    input_dir = Path(args.directory_with_experiments).resolve()
    output_dir = Path(args.output_parent_dir).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        print(f"❌ Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Parse experiment list (support both argument formats)
    experiment_names = None
    if args.entities_to_process:
        # New unified format - assume all entities are experiments for this step
        experiment_names = [e.strip() for e in args.entities_to_process.split(",")]
        if args.verbose:
            print(f"📋 Will process specific entities (as experiments): {experiment_names}")
    elif args.experiments_to_process:
        # Legacy format for backward compatibility
        experiment_names = [e.strip() for e in args.experiments_to_process.split(",")]
        if args.verbose:
            print(f"📋 Will process specific experiments: {experiment_names}")
    else:
        if args.verbose:
            print("📋 Will process ALL experiments found in input directory")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dry_run:
        print("🔍 DRY RUN - Analyzing processing status...")
        
        # Find all potential experiments
        potential_experiments = []
        for item in input_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                potential_experiments.append(item.name)
        
        print(f"📁 Found {len(potential_experiments)} potential experiments in input directory")
        
        # Check existing metadata
        metadata_path = output_dir / "raw_data_organized" / "experiment_metadata.json"
        existing_experiments = set()
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
                existing_experiments = set(existing_metadata.get('experiments', {}).keys())
                print(f"📋 Found existing metadata with {len(existing_experiments)} processed experiments")
            except Exception as e:
                print(f"⚠️  Could not read existing metadata: {e}")
        else:
            print(f"📋 No existing metadata found - all experiments will be new")
        
        # Determine what will be processed
        if experiment_names:
            requested_experiments = set(experiment_names)
            available_experiments = set(potential_experiments)
            
            # Check what's available vs requested
            missing_experiments = requested_experiments - available_experiments
            valid_experiments = requested_experiments & available_experiments
            
            # Check what's already processed vs new - but also check for new files within existing experiments
            fully_processed = set()
            needs_update = set()
            completely_new = set()
            
            for exp in valid_experiments:
                if exp in existing_experiments:
                    # Experiment exists, but check if there are new files
                    input_exp_dir = input_dir / exp
                    if input_exp_dir.exists():
                        # Count files in input directory
                        input_files = []
                        for pattern in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]:
                            input_files.extend(input_exp_dir.glob(f"**/{pattern}"))
                        
                        # Get processed files from metadata
                        try:
                            exp_metadata = existing_metadata['experiments'][exp]
                            processed_images = []
                            for video_data in exp_metadata.get('videos', {}).values():
                                processed_images.extend(video_data.get('image_ids', []))
                            
                            if len(input_files) > len(processed_images):
                                needs_update.add(exp)
                                if args.verbose:
                                    print(f"   📊 {exp}: {len(input_files)} input files, {len(processed_images)} processed")
                            else:
                                fully_processed.add(exp)
                        except Exception as e:
                            print(f"⚠️  Could not analyze {exp}: {e}")
                            needs_update.add(exp)
                    else:
                        fully_processed.add(exp)  # Input dir missing, consider processed
                else:
                    completely_new.add(exp)
            
            print(f"\n📌 Requested specific experiments: {len(experiment_names)}")
            
            if missing_experiments:
                print(f"❌ Not found in input directory: {len(missing_experiments)}")
                for exp in sorted(missing_experiments):
                    print(f"   - {exp}")
            
            if fully_processed:
                print(f"✅ Fully processed (no new files): {len(fully_processed)}")
                for exp in sorted(fully_processed):
                    print(f"   - {exp}")
            
            if needs_update:
                print(f"🔄 Needs update (new files detected): {len(needs_update)}")
                for exp in sorted(needs_update):
                    print(f"   - {exp}")
            
            if completely_new:
                print(f"🆕 Completely new: {len(completely_new)}")
                for exp in sorted(completely_new):
                    print(f"   - {exp}")
            
            if not needs_update and not completely_new:
                print(f"✨ No new files to process!")
                
        else:
            # Processing all experiments - check each one for new files
            completely_new = set(potential_experiments) - existing_experiments
            needs_update = set()
            fully_processed = set()
            
            for exp in existing_experiments:
                input_exp_dir = input_dir / exp
                if input_exp_dir.exists():
                    # Count files in input directory
                    input_files = []
                    for pattern in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]:
                        input_files.extend(input_exp_dir.glob(f"**/{pattern}"))
                    
                    # Get processed files from metadata
                    try:
                        exp_metadata = existing_metadata['experiments'][exp]
                        processed_images = []
                        for video_data in exp_metadata.get('videos', {}).values():
                            processed_images.extend(video_data.get('image_ids', []))
                        
                        if len(input_files) > len(processed_images):
                            needs_update.add(exp)
                        else:
                            fully_processed.add(exp)
                    except Exception as e:
                        print(f"⚠️  Could not analyze {exp}: {e}")
                        needs_update.add(exp)
                else:
                    fully_processed.add(exp)  # Input dir missing, consider processed
            
            print(f"\n📌 Processing mode: ALL experiments")
            print(f"✅ Fully processed: {len(fully_processed)}")
            print(f"🔄 Need updates (new files): {len(needs_update)}")
            print(f"🆕 Completely new: {len(completely_new)}")
            
            if args.verbose:
                if needs_update:
                    print(f"\n🔄 Experiments needing updates:")
                    for exp in sorted(needs_update):
                        print(f"   - {exp}")
                
                if completely_new:
                    print(f"\n🆕 New experiments to process:")
                    for exp in sorted(completely_new):
                        print(f"   - {exp}")
                
                if fully_processed:
                    print(f"\n✅ Fully processed experiments:")
                    for exp in sorted(fully_processed):
                        print(f"   - {exp}")
        
        print(f"\n📁 Output directory: {output_dir}")
        print("👆 Use --verbose to see detailed experiment lists")
        return
    
    # TODO: Import the actual data organizer when it's available
    # For now, create a placeholder but check what's already processed
    print("🚀 Starting data organization...")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"👥 Workers: {args.workers}")
    
    # Check existing metadata to see what's already processed
    metadata_path = output_dir / "raw_data_organized" / "experiment_metadata.json"
    existing_experiments = set()
    existing_metadata = {}
    if metadata_path.exists():
        try:
            import json
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
            existing_experiments = set(existing_metadata.get('experiments', {}).keys())
            print(f"📋 Found {len(existing_experiments)} already processed experiments")
        except Exception as e:
            print(f"⚠️  Could not read existing metadata: {e}")
    
    # Determine what actually needs processing (check for new files within experiments)
    if experiment_names:
        requested_experiments = set(experiment_names)
        
        # Check each requested experiment for new files
        fully_processed = set()
        needs_processing = set()
        
        for exp in requested_experiments:
            input_exp_dir = input_dir / exp
            if not input_exp_dir.exists():
                print(f"❌ Experiment directory not found: {exp}")
                continue
                
            if exp in existing_experiments:
                # Count files in input directory
                input_files = []
                for pattern in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]:
                    input_files.extend(input_exp_dir.glob(f"**/{pattern}"))
                
                # Get processed files from metadata
                try:
                    exp_metadata = existing_metadata['experiments'][exp]
                    processed_images = []
                    for video_data in exp_metadata.get('videos', {}).values():
                        processed_images.extend(video_data.get('image_ids', []))
                    
                    # Check both file completeness AND entity tracking
                    files_complete = len(input_files) <= len(processed_images)
                    entity_tracking_complete = check_entity_tracking_complete(metadata_path)
                    
                    if not files_complete:
                        needs_processing.add(exp)
                        print(f"🔄 {exp}: {len(input_files)} input files, {len(processed_images)} processed - needs update")
                    elif not entity_tracking_complete:
                        needs_processing.add(exp)
                        print(f"🔄 {exp}: Missing entity tracking - needs update")
                    else:
                        fully_processed.add(exp)
                        print(f"✅ {exp}: {len(processed_images)} files already processed with entity tracking")
                except Exception as e:
                    print(f"⚠️  Could not analyze {exp}: {e} - will reprocess")
                    needs_processing.add(exp)
            else:
                needs_processing.add(exp)
                input_files = []
                for pattern in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]:
                    input_files.extend(input_exp_dir.glob(f"**/{pattern}"))
                print(f"🆕 {exp}: {len(input_files)} files - completely new")
        
        if fully_processed:
            print(f"✅ Skipping {len(fully_processed)} fully processed experiments")
        
        if needs_processing:
            print(f"🆕 Processing {len(needs_processing)} experiments with new/missing data:")
            for exp in sorted(needs_processing):
                print(f"   - {exp}")
        else:
            print(f"✨ All requested experiments are fully processed!")
            print(f"📋 Next step: Run 03_gdino_detection.py with --metadata {metadata_path}")
            return
    else:
        print(f"📌 Processing ALL experiments (will analyze each for new files)")
        needs_processing = set()
        
        # Check all available experiments
        for item in input_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                exp = item.name
                if exp in existing_experiments:
                    # Check for new files
                    input_files = []
                    for pattern in ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]:
                        input_files.extend(item.glob(f"**/{pattern}"))
                    
                    try:
                        exp_metadata = existing_metadata['experiments'][exp]
                        processed_images = []
                        for video_data in exp_metadata.get('videos', {}).values():
                            processed_images.extend(video_data.get('image_ids', []))
                        
                        if len(input_files) > len(processed_images):
                            needs_processing.add(exp)
                    except Exception:
                        needs_processing.add(exp)
                else:
                    needs_processing.add(exp)
        
        print(f"🔄 Found {len(needs_processing)} experiments needing processing")
    
    try:
        # Import and use the actual data organizer
        sys.path.insert(0, str(SCRIPTS_DIR))  # Ensure scripts directory is in path
        from data_organization.data_organizer import DataOrganizer

        # Determine which experiments to process
        experiments_to_process = list(needs_processing) if needs_processing else experiment_names

        print(f"🚀 Processing {len(experiments_to_process) if experiments_to_process else 'ALL'} experiments...")

        DataOrganizer.process_experiments(
            source_dir=input_dir,
            output_dir=output_dir,
            experiment_names=experiments_to_process,
            verbose=args.verbose,
            overwrite=False
        )

        # DataOrganizer automatically creates/updates the metadata file
        metadata_path = output_dir / "raw_data_organized" / "experiment_metadata.json"

        print(f"✅ Data organization complete!")
        print(f"📋 Next step: Run 03_gdino_detection.py with --metadata {metadata_path}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
