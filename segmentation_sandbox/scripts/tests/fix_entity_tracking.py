#!/usr/bin/env python3
"""
Script to validate and fix entity tracking in existing metadata.
This addresses the pipeline weakness where experiments are considered "processed"
even when entity tracking is missing.
"""

import sys
from pathlib import Path
import json

# Add the scripts directory to the path to import our modules
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.data_organization.data_organizer import DataOrganizer

def main():
    """Main function to validate and fix entity tracking."""
    
    # Paths
    data_dir = Path(__file__).parent / "data"
    raw_data_dir = data_dir / "raw_data_organized"
    metadata_path = raw_data_dir / "experiment_metadata.json"
    
    print("🔍 Validating entity tracking in existing metadata...")
    print(f"📁 Data directory: {raw_data_dir}")
    print(f"📄 Metadata file: {metadata_path}")
    
    # Check if metadata file exists
    if not metadata_path.exists():
        print("❌ No metadata file found - run the pipeline first")
        return 1
        
    # Load existing metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata with {len(metadata.get('experiments', {}))} experiments")
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return 1
        
    # Validate entity tracking
    is_complete = DataOrganizer.validate_entity_tracking_completeness(metadata, verbose=True)
    
    if is_complete:
        print("✅ Entity tracking is already complete!")
        return 0
    else:
        print("📋 Entity tracking is incomplete - fixing...")
        
        # Force entity tracking update using DataOrganizer
        print("🔄 Re-scanning organized data and adding entity tracking...")
        
        # Use DataOrganizer to process with overwrite=False but force entity tracking
        try:
            DataOrganizer.process_experiments(
                source_dir="/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images",
                output_dir=str(data_dir),
                experiment_names=None,  # Process all
                verbose=True,
                overwrite=False  # Don't reprocess files, just fix metadata
            )
            print("✅ Entity tracking fix completed!")
            return 0
        except Exception as e:
            print(f"❌ Failed to fix entity tracking: {e}")
            return 1

if __name__ == "__main__":
    exit(main())
