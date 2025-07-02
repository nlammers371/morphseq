# Test script for Image Quality Control utilities
"""
This script tests the core functions of the image quality control utilities
and saves output to the `test` subfolder. You can follow these steps in a notebook.
"""
import sys
import os
from pathlib import Path
import pandas as pd

# Add the parent directory (image_quality_qc_utils) to the path
current_dir = Path(__file__).parent.resolve()
utils_dir = current_dir.parent
sys.path.insert(0, str(utils_dir))

# Now import directly from the module in the parent directory
from image_quality_qc_utils import load_or_create_qc_csv, QC_FLAGS

# Define paths
qc_csv_path = current_dir / 'test_image_quality_qc.csv'
print(f"QC CSV path: {qc_csv_path}")

# Load or create the QC DataFrame
qc_df = load_or_create_qc_csv(qc_csv_path)
print(f"Initial QC records: {len(qc_df)}")

# Append a test record using pd.concat (append is deprecated)
test_record = pd.DataFrame([{
    'experiment_id': '20250101',
    'video_id': '20250101_A01',
    'image_id': '20250101_A01_t001',
    'qc_flag': 'PASS',
    'notes': 'auto test record',
    'annotator': 'test_script'
}])
qc_df = pd.concat([qc_df, test_record], ignore_index=True)
print(f"After append QC records: {len(qc_df)}")

# Save to CSV
qc_df.to_csv(qc_csv_path, index=False)
print(f"Saved updated QC data to {qc_csv_path}")
