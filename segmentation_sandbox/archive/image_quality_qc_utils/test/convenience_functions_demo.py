#!/usr/bin/env python
"""
QC Convenience Functions Demo
============================

This script demonstrates how to use the convenient manual_qc() and auto_qc() 
wrapper functions, which are the recommended way to flag images for quality control.

These functions simplify the workflow by automatically setting the annotator field.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add the parent directory to path for imports
current_dir = Path(__file__).parent.resolve() if '__file__' in locals() else Path.cwd()
utils_dir = current_dir.parent
sys.path.insert(0, str(utils_dir))

# Import QC utilities - focus on the convenience functions
from image_quality_qc_utils import (
    load_qc_data, save_qc_data, 
    get_qc_summary, get_flagged_images, get_unflagged_images,
    get_images_by_flag, get_images_by_annotator, QC_FLAGS,
    manual_qc, auto_qc  # ← These are the star functions!
)

print("=== QC Convenience Functions Demo ===")
print(f"Available QC flags: {list(QC_FLAGS.keys())}")

# =============================================================================
# SETUP: Create Test Data
# =============================================================================
print("\n=== SETUP: Creating Test Data ===")

# Create test data directory
test_data_dir = current_dir / "convenience_demo_data"
test_data_dir.mkdir(exist_ok=True)

# Initialize with some sample images
sample_images = [
    {"experiment_id": "20250101", "video_id": "20250101_A01", "image_id": "20250101_A01_t001"},
    {"experiment_id": "20250101", "video_id": "20250101_A01", "image_id": "20250101_A01_t002"},
    {"experiment_id": "20250101", "video_id": "20250101_A01", "image_id": "20250101_A01_t003"},
    {"experiment_id": "20250101", "video_id": "20250101_B01", "image_id": "20250101_B01_t001"},
    {"experiment_id": "20250102", "video_id": "20250102_A01", "image_id": "20250102_A01_t001"},
]

# Load or create QC data
qc_df = load_qc_data(test_data_dir)

# Add sample images if they don't exist
for img in sample_images:
    if len(qc_df) == 0 or img["image_id"] not in qc_df["image_id"].values:
        new_row = pd.DataFrame([{
            "experiment_id": img["experiment_id"],
            "video_id": img["video_id"], 
            "image_id": img["image_id"],
            "qc_flag": None,
            "notes": None,
            "annotator": None
        }])
        qc_df = pd.concat([qc_df, new_row], ignore_index=True)

save_qc_data(qc_df, test_data_dir)
print(f"Setup complete: {len(qc_df)} images ready for QC")

# =============================================================================
# DEMO 1: Manual QC using manual_qc() - The Easy Way!
# =============================================================================
print("\n=== DEMO 1: Manual QC with manual_qc() ===")
print("The manual_qc() function automatically sets annotator to your name.")
print("Just specify your name once, then flag images easily!\n")

# Example 1: Flag a single image as BLUR
print("Example 1: Flagging a blurry image")
qc_df = manual_qc(
    data_dir=test_data_dir,
    annotator="mcolon",  # Your name here
    image_ids=["20250101_A01_t001"],
    qc_flag="BLUR",
    notes="Manual inspection - image appears out of focus",
    overwrite=True
)
print("✓ Flagged 20250101_A01_t001 as BLUR\n")

# Example 2: Flag multiple images with quality issues
print("Example 2: Flagging multiple images at once")
qc_df = manual_qc(
    data_dir=test_data_dir,
    annotator="mcolon",
    image_ids=["20250101_A01_t002", "20250101_B01_t001"],
    qc_flag="DARK",
    notes="Too dark to see embryo features clearly",
    overwrite=True
)
print("✓ Flagged 2 images as DARK\n")

# Example 3: Flag by video and frame numbers
print("Example 3: Flagging specific frames in a video")
qc_df = manual_qc(
    data_dir=test_data_dir,
    annotator="nlammers",  # Different annotator
    video_id="20250101_A01",
    frames=["t003"],
    qc_flag="ARTIFACT",
    notes="Debris visible in well affecting analysis"
)
print("✓ Flagged frame t003 in video 20250101_A01 as ARTIFACT\n")

print("Manual QC Summary:")
manual_images = get_images_by_annotator(qc_df, "mcolon")
print(f"  mcolon flagged: {len(manual_images)} images")
nlammers_images = get_images_by_annotator(qc_df, "nlammers") 
print(f"  nlammers flagged: {len(nlammers_images)} images")

# =============================================================================
# DEMO 2: Automatic QC using auto_qc() - Even Easier!
# =============================================================================
print("\n=== DEMO 2: Automatic QC with auto_qc() ===")
print("The auto_qc() function automatically sets annotator to 'automatic'.")
print("Perfect for algorithmic quality control!\n")

# Example 1: Automatic blur detection result
print("Example 1: Automatic blur detection")
qc_df = auto_qc(
    data_dir=test_data_dir,
    image_ids=["20250102_A01_t001"],
    qc_flag="BLUR",
    notes="Automatic: Laplacian variance = 45.2 < threshold (100)",
    overwrite=True
)
print("✓ Automatically flagged 20250102_A01_t001 as BLUR\n")

# Example 2: Automatic PASS flags for good images
print("Example 2: Automatic PASS flags for quality images")
# In real usage, you might process a batch of images like this:
good_images = ["20250101_A01_t001"]  # Re-evaluate the first image
qc_df = auto_qc(
    data_dir=test_data_dir,
    image_ids=good_images,
    qc_flag="PASS",
    notes="Automatic: All quality metrics within normal range",
    overwrite=True
)
print("✓ Automatically flagged good images as PASS\n")

print("Automatic QC Summary:")
auto_images = get_images_by_annotator(qc_df, "automatic")
print(f"  Automatic flags: {len(auto_images)} images")

# =============================================================================
# DEMO 3: Analyzing Results - Mix of Manual and Automatic
# =============================================================================
print("\n=== DEMO 3: Analyzing Mixed QC Results ===")

# Get overall summary
summary = get_qc_summary(test_data_dir)
print("Overall QC Summary:")
print(f"  Total images: {summary['total_images']}")
print(f"  QC flags: {summary['qc_flags']}")
print(f"  Annotators: {summary['annotators']}")

# Show breakdown by QC flag type
print("\nBreakdown by QC Flag:")
for flag in ['PASS', 'BLUR', 'DARK', 'ARTIFACT']:
    flag_images = get_images_by_flag(qc_df, flag)
    if flag_images:
        print(f"  {flag}: {len(flag_images)} images - {flag_images}")

# Show manual vs automatic flagging
print("\nManual vs Automatic Flagging:")
manual_flagged = get_images_by_annotator(qc_df, "mcolon")
auto_flagged = get_images_by_annotator(qc_df, "automatic")
print(f"  Manual (mcolon): {len(manual_flagged)} images")
print(f"  Automatic: {len(auto_flagged)} images")

# =============================================================================
# DEMO 4: Advanced Usage Patterns
# =============================================================================
print("\n=== DEMO 4: Advanced Usage Patterns ===")

print("Pattern 1: Batch manual review")
print("# Review and flag multiple images with same issue")
print("manual_qc(data_dir, annotator='reviewer_name',")
print("          image_ids=['img1', 'img2', 'img3'],")
print("          qc_flag='BLUR', notes='Batch review - all blurry')")

print("\nPattern 2: Automatic quality pipeline")
print("# Process all images in a directory automatically")
print("for image_path in image_directory.glob('*.tif'):")
print("    quality_score = analyze_image(image_path)")
print("    if quality_score < threshold:")
print("        auto_qc(data_dir, image_ids=[image_id],")
print("                qc_flag='BLUR', notes=f'Score: {quality_score}')")

print("\nPattern 3: Video-based flagging")
print("# Flag entire video sequences")
print("manual_qc(data_dir, annotator='user',")
print("          video_id='20250101_A01', frames=['t001', 't002'],")
print("          qc_flag='OUT_OF_FOCUS', notes='Entire sequence')")

# =============================================================================
# FINAL RESULTS
# =============================================================================
print("\n=== FINAL RESULTS ===")

# Show the final QC data
print("Final QC Data:")
display_df = qc_df[qc_df['qc_flag'].notna()][['image_id', 'qc_flag', 'annotator', 'notes']]
print(display_df.to_string(index=False))

# Save results
output_file = current_dir / "convenience_demo_results.csv"
qc_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

print("\n=== KEY TAKEAWAYS ===")
print("1. Use manual_qc() for human review - just specify your name once")
print("2. Use auto_qc() for algorithmic flagging - annotator set automatically")
print("3. Both functions support individual images, lists, or video+frames")
print("4. Mix manual and automatic QC in the same workflow")
print("5. All data goes to the same CSV file for unified analysis")

print("\n🎉 Demo complete! Try the convenience functions in your own workflow.")
