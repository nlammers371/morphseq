"""
Test script for DataOrganizer using example experiment '20231206'.
This will run the DataOrganizer on a single experiment and print results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_organization.data_organizer import DataOrganizer
import json
# Set up paths
ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/segmentation_sandbox")
STITCHED_DIR = Path("/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/stitched_FF_images/")
DATA_DIR = ROOT / "data"
EXPERIMENT = "20250703_chem3_28C_T00_1325"

# Clean output directory
if DATA_DIR.exists():
    import shutil
    shutil.rmtree(DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Run DataOrganizer on the example experiment
print(f"Running DataOrganizer on experiment: {EXPERIMENT}")

# First run - should process everything
print("\n🚀 First run: Processing experiment...")
DataOrganizer.process_experiments(
    source_dir=STITCHED_DIR,
    output_dir=DATA_DIR,
    experiment_names=[EXPERIMENT],
    verbose=True,
    overwrite=False
)

# Test with overwrite=False (should skip on second run)
print("\n🔄 Testing autosave functionality (second run with overwrite=False)...")

# Run again without overwrite - should skip existing
DataOrganizer.process_experiments(
    source_dir=STITCHED_DIR,
    output_dir=DATA_DIR,
    experiment_names=[EXPERIMENT],
    verbose=True,
    overwrite=False  # Should skip existing
)

print("\n🔄 Testing overwrite functionality...")

# Run again with overwrite=True - should reprocess
DataOrganizer.process_experiments(
    source_dir=STITCHED_DIR,
    output_dir=DATA_DIR, 
    experiment_names=[EXPERIMENT],
    verbose=True,
    overwrite=True  # Should reprocess existing
)

# Check for metadata output
metadata_path = DATA_DIR / "raw_data_organized" / "experiment_metadata.json"
if metadata_path.exists():
    with open(metadata_path) as f:
        metadata = json.load(f)
    print("\nExperiment metadata created:")
    print(json.dumps(metadata, indent=2)[:2000])  # Print first 2000 chars
else:
    print("[ERROR] experiment_metadata.json not found!")
