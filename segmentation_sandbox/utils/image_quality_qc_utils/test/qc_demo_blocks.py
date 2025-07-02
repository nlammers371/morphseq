#!/usr/bin/env python
"""
QC Demo - Manual and Automatic Annotations
==========================================

This script demonstrates how to use the image quality control utilities
for both manual and automatic QC flagging. Each block can be run separately
in a Jupyter notebook.

Run each block sequentially to see the full workflow.
"""

# =============================================================================
# BLOCK 1: Setup and Imports
# =============================================================================
import sys
import os
from pathlib import Path
import pandas as pd

# Add the parent directory to path for imports
current_dir = Path(__file__).parent.resolve() if '__file__' in locals() else Path.cwd()
utils_dir = current_dir.parent
sys.path.insert(0, str(utils_dir))

# Import QC utilities
from image_quality_qc_utils import (
    load_qc_data, save_qc_data, flag_qc, remove_qc,
    get_flagged_images, get_unflagged_images, 
    get_qc_summary, QC_FLAGS
)

print("=== BLOCK 1: Setup Complete ===")
print(f"Available QC flags: {list(QC_FLAGS.keys())}")
print(f"Working directory: {current_dir}")

# =============================================================================
# BLOCK 2: Initialize Test Data Directory and QC File
# =============================================================================
# Create a mock data directory structure for testing
test_data_dir = current_dir / "mock_data"
test_data_dir.mkdir(exist_ok=True)

# Create quality_control subdirectory
qc_dir = test_data_dir / "quality_control"
qc_dir.mkdir(exist_ok=True)

print("=== BLOCK 2: Test Data Directory Setup ===")
print(f"Test data directory: {test_data_dir}")
print(f"QC directory: {qc_dir}")

# Load or create initial QC data
qc_df = load_qc_data(test_data_dir)
print(f"Initial QC records: {len(qc_df)}")

# =============================================================================
# BLOCK 3: Add Sample Images to QC System
# =============================================================================
# Create some sample image records (simulating images from different experiments)
sample_images = [
    {"experiment_id": "20241215", "video_id": "20241215_A01", "image_id": "20241215_A01_t001"},
    {"experiment_id": "20241215", "video_id": "20241215_A01", "image_id": "20241215_A01_t002"},
    {"experiment_id": "20241215", "video_id": "20241215_A01", "image_id": "20241215_A01_t003"},
    {"experiment_id": "20241215", "video_id": "20241215_B01", "image_id": "20241215_B01_t001"},
    {"experiment_id": "20241216", "video_id": "20241216_A01", "image_id": "20241216_A01_t001"},
    {"experiment_id": "20241216", "video_id": "20241216_A01", "image_id": "20241216_A01_t002"},
]

# Add sample images to QC DataFrame if not already present
for img in sample_images:
    if img["image_id"] not in qc_df["image_id"].values:
        new_row = pd.DataFrame([{
            "experiment_id": img["experiment_id"],
            "video_id": img["video_id"], 
            "image_id": img["image_id"],
            "qc_flag": None,
            "notes": None,
            "annotator": None
        }])
        qc_df = pd.concat([qc_df, new_row], ignore_index=True)

# Save the updated DataFrame
save_qc_data(qc_df, test_data_dir)

print("=== BLOCK 3: Sample Images Added ===")
print(f"Total records after adding samples: {len(qc_df)}")
print("\nSample images:")
print(qc_df[["experiment_id", "video_id", "image_id"]].head())

# =============================================================================
# BLOCK 4: Manual QC Flagging - Individual Images
# =============================================================================
print("\n=== BLOCK 4: Manual QC Flagging ===")

# Flag individual images manually
manual_flags = [
    ("20241215_A01_t002", "BLUR", "mcolon", "Image appears blurry under inspection"),
    ("20241215_B01_t001", "DARK", "mcolon", "Image is too dark to see embryo clearly"),
    ("20241216_A01_t001", "ARTIFACT", "nlammers", "Debris visible in well"),
]

for image_id, flag, annotator, notes in manual_flags:
    qc_df = flag_qc(
        data_dir=test_data_dir,
        image_ids=[image_id],
        qc_flag=flag,
        annotator=annotator,
        notes=notes,
        overwrite=True
    )
    print(f"Flagged {image_id} as {flag} by {annotator}")

# Show current status
print(f"\nRecords after manual flagging: {len(qc_df)}")
flagged = qc_df[qc_df["qc_flag"].notna()]
print(f"Flagged images: {len(flagged)}")
if len(flagged) > 0:
    print("\nFlagged images summary:")
    print(flagged[["image_id", "qc_flag", "annotator", "notes"]])

# =============================================================================
# BLOCK 5: Manual QC Flagging - By Video and Frames
# =============================================================================
print("\n=== BLOCK 5: Video-Based Flagging ===")

# Flag multiple frames in a video at once
qc_df = flag_qc(
    data_dir=test_data_dir,
    video_id="20241216_A01",
    frames=["t002"],  # Flag frame t002 in this video
    qc_flag="OUT_OF_FOCUS",
    annotator="mcolon",
    notes="Entire sequence appears out of focus",
    overwrite=True
)

print("Flagged frame t002 in video 20241216_A01 as OUT_OF_FOCUS")

# Show updated status
flagged = qc_df[qc_df["qc_flag"].notna()]
print(f"\nTotal flagged images: {len(flagged)}")

# =============================================================================
# BLOCK 6: Automatic QC Flagging (Simulated)
# =============================================================================
print("\n=== BLOCK 6: Automatic QC Flagging ===")

# Simulate automatic QC results (in real usage, this would come from image analysis)
automatic_flags = [
    ("20241215_A01_t001", "PASS", "Quality metrics within normal range"),
    ("20241215_A01_t003", "PASS", "Quality metrics within normal range"),
]

for image_id, flag, notes in automatic_flags:
    qc_df = flag_qc(
        data_dir=test_data_dir,
        image_ids=[image_id],
        qc_flag=flag,
        annotator="automatic",  # Mark as automatic
        notes=notes,
        overwrite=True
    )
    print(f"Automatically flagged {image_id} as {flag}")

print(f"\nTotal records after automatic flagging: {len(qc_df)}")

# =============================================================================
# BLOCK 7: QC Data Analysis and Summary
# =============================================================================
print("\n=== BLOCK 7: QC Analysis ===")

# Get summary statistics
summary = get_qc_summary(test_data_dir)
print("QC Summary:")
for key, value in summary.items():
    print(f"  {key}: {value}")

# Get flagged and unflagged images
flagged_images = get_flagged_images(qc_df)
unflagged_images = get_unflagged_images(qc_df)

print(f"\nFlagged images: {len(flagged_images)}")
print(f"Unflagged images: {len(unflagged_images)}")

# Show breakdown by flag type
flag_counts = qc_df[qc_df["qc_flag"].notna()]["qc_flag"].value_counts()
print("\nFlag type breakdown:")
print(flag_counts)

# Show breakdown by annotator
annotator_counts = qc_df[qc_df["annotator"].notna()]["annotator"].value_counts()
print("\nAnnotator breakdown:")
print(annotator_counts)

# =============================================================================
# BLOCK 8: Removing QC Flags
# =============================================================================
print("\n=== BLOCK 8: Removing QC Flags ===")

# Remove a QC flag if it was added incorrectly
print("Before removal:")
specific_image = qc_df[qc_df["image_id"] == "20241215_B01_t001"]
print(specific_image[["image_id", "qc_flag", "annotator", "notes"]])

qc_df = remove_qc(
    data_dir=test_data_dir,
    image_ids=["20241215_B01_t001"],
    annotator="mcolon"
)

print("\nAfter removal:")
specific_image = qc_df[qc_df["image_id"] == "20241215_B01_t001"]
print(specific_image[["image_id", "qc_flag", "annotator", "notes"]])

# =============================================================================
# BLOCK 9: Final Data Export and Cleanup
# =============================================================================
print("\n=== BLOCK 9: Final Results ===")

# Show final QC data
print("Final QC data:")
print(qc_df)

# Save final results
final_csv = current_dir / "final_qc_demo_results.csv"
qc_df.to_csv(final_csv, index=False)
print(f"\nFinal results saved to: {final_csv}")

# Show file locations
qc_csv_path = qc_dir / "image_quality_qc.csv"
print(f"QC data saved to: {qc_csv_path}")

print("\n=== DEMO COMPLETE ===")
print("You can now examine the generated CSV files to see the QC data structure.")
