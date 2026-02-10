#!/usr/bin/env python3
"""
Test backup functionality for ExperimentMetadata
"""

import sys
from pathlib import Path

# Add project root to path
SANDBOX_ROOT = Path(__file__).parent
sys.path.append(str(SANDBOX_ROOT))

from scripts.metadata.experiment_metadata import ExperimentMetadata

def main():
    metadata_path = SANDBOX_ROOT / "data" / "raw_data_organized" / "experiment_metadata.json"
    
    print(f"Testing backup functionality for: {metadata_path}")
    print(f"File exists: {metadata_path.exists()}")
    
    if metadata_path.exists():
        # Load metadata
        metadata_manager = ExperimentMetadata(metadata_path, verbose=True)
        print(f"Loaded metadata: {metadata_manager}")
        
        # Add a dummy experiment to trigger a save
        test_exp_id = "test_backup_experiment"
        print(f"\nAdding test experiment: {test_exp_id}")
        metadata_manager.add_experiment(test_exp_id, test_field="backup_test")
        
        # Force a save
        print("Forcing save to test backup creation...")
        metadata_manager.save(force=True)
        
        # Remove the test experiment
        if test_exp_id in metadata_manager.metadata["experiments"]:
            del metadata_manager.metadata["experiments"][test_exp_id]
            # Update entity tracking
            metadata_manager.metadata["entity_tracking"]["experiments"] = [
                exp for exp in metadata_manager.metadata["entity_tracking"]["experiments"] 
                if exp != test_exp_id
            ]
            print(f"Removed test experiment: {test_exp_id}")
            metadata_manager.save(force=True)
        
        # Check for backup files
        print("\nChecking for backup files...")
        backup_pattern = f"{metadata_path.stem}.backup.*{metadata_path.suffix}"
        backup_files = list(metadata_path.parent.glob(backup_pattern))
        
        if backup_files:
            print(f"✅ Found {len(backup_files)} backup files:")
            for backup in sorted(backup_files):
                print(f"   - {backup.name}")
        else:
            print("❌ No backup files found")
    else:
        print("❌ Metadata file does not exist")

if __name__ == "__main__":
    main()
