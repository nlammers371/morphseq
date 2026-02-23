import json
import subprocess
import sys
from pathlib import Path
import shutil

# Paths
base_dir = Path(__file__).parent
script_path = base_dir.parents[1] / "scripts" / "pipelines" / "07_embryo_metadata_update.py"

# Clean up from previous runs
if (base_dir / "data").exists():
    shutil.rmtree(base_dir / "data")

# Build minimal SAM2 dataset
sam2_initial = {
    "experiments": {
        "exp1": {
            "videos": {
                "exp1_A01": {
                    "images": {
                        "exp1_A01_t0001": {"embryos": {"exp1_A01_e0001": {}}}
                    }
                }
            }
        }
    }
}

sam2_updated = {
    "experiments": {
        "exp1": {
            "videos": {
                "exp1_A01": {
                    "images": {
                        "exp1_A01_t0001": {"embryos": {"exp1_A01_e0001": {}}},
                        "exp1_A01_t0002": {"embryos": {"exp1_A01_e0001": {}}}
                    }
                }
            }
        }
    }
}

# Write initial SAM2 file
sam2_path1 = base_dir / "sam2_annotations.json"
with open(sam2_path1, "w") as f:
    json.dump(sam2_initial, f)

# Run script to create metadata
create = subprocess.run([
    sys.executable,
    str(script_path),
    str(sam2_path1),
    "--output-dir",
    str(base_dir),
], capture_output=True, text=True)

print("First run (create):")
print(create.stdout, create.stderr, sep="")

# Determine metadata path
metadata_path = base_dir / "data" / "embryo_metadata" / "sam2_biology.json"

# Write updated SAM2 file with new frame
sam2_path2 = base_dir / "sam2_annotations_v2.json"
with open(sam2_path2, "w") as f:
    json.dump(sam2_updated, f)

# Attempt to update existing metadata without --force
update = subprocess.run([
    sys.executable,
    str(script_path),
    str(sam2_path2),
    "--output",
    metadata_path.name,
    "--output-dir",
    str(base_dir),
], capture_output=True, text=True)

print("Second run (attempt update without --force):")
print(update.stdout, update.stderr, sep="")

