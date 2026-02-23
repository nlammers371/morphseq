#!/usr/bin/env python3
"""
Cleanup old backup files using the BaseFileHandler backup management
"""

import sys
from pathlib import Path

# Add project root to path
SANDBOX_ROOT = Path(__file__).parent
sys.path.append(str(SANDBOX_ROOT))

from scripts.utils.base_file_handler import BaseFileHandler

def main():
    # Clean up gdino_detections backups
    gdino_annotations_path = SANDBOX_ROOT / "data" / "detections" / "gdino_detections.json"
    if gdino_annotations_path.exists():
        print("🧹 Cleaning up GroundingDINO annotations backups...")
        handler = BaseFileHandler(gdino_annotations_path, verbose=True)
        handler.cleanup_backups(keep_count=1)
    
    # Clean up any other backup files in data directory
    for backup_file in SANDBOX_ROOT.glob("data/**/*.backup.*"):
        # Extract the original file path
        parts = backup_file.name.split('.backup.')
        if len(parts) == 2:
            original_name = parts[0] + backup_file.suffix.split('.')[-1]
            original_path = backup_file.parent / (original_name + '.json')
            
            if original_path.exists():
                print(f"\n🧹 Cleaning up backups for {original_path.name}...")
                handler = BaseFileHandler(original_path, verbose=True)
                handler.cleanup_backups(keep_count=1)
    
    print("\n✅ Backup cleanup completed!")

if __name__ == "__main__":
    main()
