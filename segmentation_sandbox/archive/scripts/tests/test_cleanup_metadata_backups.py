#!/usr/bin/env python3
"""
Test the updated BaseFileHandler backup cleanup
"""

import sys
from pathlib import Path

# Add project root to path
SANDBOX_ROOT = Path(__file__).parent
sys.path.append(str(SANDBOX_ROOT))

from scripts.utils.base_file_handler import BaseFileHandler

def main():
    metadata_path = SANDBOX_ROOT / "data" / "raw_data_organized" / "experiment_metadata.json"
    
    print(f"Testing backup cleanup for: {metadata_path}")
    
    # Create a handler and manually trigger backup cleanup
    handler = BaseFileHandler(metadata_path, verbose=True)
    print("Manually triggering backup cleanup...")
    handler.cleanup_backups(keep_count=1)
    
    # Check remaining backups
    print("\nChecking remaining backup files...")
    backup_pattern = f"{metadata_path.stem}.backup.*{metadata_path.suffix}"
    backup_files = list(metadata_path.parent.glob(backup_pattern))
    
    if backup_files:
        print(f"Found {len(backup_files)} backup files:")
        for backup in sorted(backup_files):
            print(f"   - {backup.name}")
    else:
        print("No backup files found")

if __name__ == "__main__":
    main()
