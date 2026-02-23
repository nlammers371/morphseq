#!/usr/bin/env python3
"""
Diagnostic script to identify why embryo snips are missing for Build03A processing.

This script investigates the 25 embryos from experiment 20250305 that have 100% NaN
in their latent embeddings, indicating their snip images were never generated.

Outputs saved to: /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251013/nan_investigation/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Setup paths
MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
MORPHSEQ_PLAYGROUND = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
RESULTS_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251013")
OUTPUT_DIR = RESULTS_DIR / "nan_investigation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("MISSING SNIPS DIAGNOSTIC SCRIPT")
print("="*80)
print(f"\nMorphseq root: {MORPHSEQ_ROOT}")
print(f"Morphseq playground: {MORPHSEQ_PLAYGROUND}")
print(f"Output directory: {OUTPUT_DIR}")

# ==============================================================================
# 1. Load NaN statistics and identify problem embryos
# ==============================================================================
print("\n" + "="*80)
print("1. LOADING NaN STATISTICS")
print("="*80)

nan_stats_path = RESULTS_DIR / "data" / "nan_stats_by_embryo.csv"
nan_stats_df = pd.read_csv(nan_stats_path)

# Filter to experiment 20250305 with 100% NaN
exp_20250305_df = nan_stats_df[
    (nan_stats_df['experiment'] == '20250305') &
    (nan_stats_df['pct_nan'] == 100.0)
].copy()

print(f"\nTotal embryos in NaN stats: {len(nan_stats_df)}")
print(f"Embryos from 20250305: {len(nan_stats_df[nan_stats_df['experiment'] == '20250305'])}")
print(f"Embryos from 20250305 with 100% NaN: {len(exp_20250305_df)}")

# Also check partial NaN embryos
partial_nan_df = nan_stats_df[
    (nan_stats_df['experiment'] == '20250305') &
    (nan_stats_df['pct_nan'] > 0) &
    (nan_stats_df['pct_nan'] < 100)
].copy()
print(f"Embryos from 20250305 with partial NaN: {len(partial_nan_df)}")

# Combine for investigation
problem_embryos_df = pd.concat([exp_20250305_df, partial_nan_df], ignore_index=True)
print(f"\nTotal problem embryos to investigate: {len(problem_embryos_df)}")

# ==============================================================================
# 2. Check SAM2 metadata CSV
# ==============================================================================
print("\n" + "="*80)
print("2. CHECKING SAM2 METADATA CSV")
print("="*80)

sam2_csv_path = MORPHSEQ_PLAYGROUND / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20250305.csv"
print(f"\nSAM2 CSV path: {sam2_csv_path}")
print(f"SAM2 CSV exists: {sam2_csv_path.exists()}")

# Initialize diagnosis columns
problem_embryos_df['in_sam2_csv'] = False
problem_embryos_df['sam2_row_count'] = 0
problem_embryos_df['has_exported_mask_path'] = False
problem_embryos_df['has_image_id'] = False
problem_embryos_df['has_time_info'] = False
problem_embryos_df['duplicate_snip_ids'] = False

if sam2_csv_path.exists():
    sam2_df = pd.read_csv(sam2_csv_path, dtype={'experiment_id': str})
    print(f"SAM2 CSV loaded: {len(sam2_df)} total rows")

    # Check each problem embryo
    for idx, row in problem_embryos_df.iterrows():
        embryo_id = row['embryo_id']

        # Find rows for this embryo
        embryo_rows = sam2_df[sam2_df['embryo_id'] == embryo_id]

        problem_embryos_df.at[idx, 'in_sam2_csv'] = len(embryo_rows) > 0
        problem_embryos_df.at[idx, 'sam2_row_count'] = len(embryo_rows)

        if len(embryo_rows) > 0:
            # Check for required columns
            problem_embryos_df.at[idx, 'has_exported_mask_path'] = (
                'exported_mask_path' in embryo_rows.columns and
                embryo_rows['exported_mask_path'].notna().any()
            )
            problem_embryos_df.at[idx, 'has_image_id'] = (
                'image_id' in embryo_rows.columns and
                embryo_rows['image_id'].notna().any()
            )

            # Check for time information
            has_time = False
            for col in ['time_int', 'time_string', 'time_key', 'frame_index']:
                if col in embryo_rows.columns and embryo_rows[col].notna().any():
                    has_time = True
                    break
            problem_embryos_df.at[idx, 'has_time_info'] = has_time

            # Check for duplicate snip_ids
            if 'snip_id' in embryo_rows.columns:
                duplicate_count = embryo_rows['snip_id'].duplicated().sum()
                problem_embryos_df.at[idx, 'duplicate_snip_ids'] = duplicate_count > 0

    # Summary
    print(f"\nEmbryos found in SAM2 CSV: {problem_embryos_df['in_sam2_csv'].sum()}")
    print(f"Embryos with exported_mask_path: {problem_embryos_df['has_exported_mask_path'].sum()}")
    print(f"Embryos with image_id: {problem_embryos_df['has_image_id'].sum()}")
    print(f"Embryos with time info: {problem_embryos_df['has_time_info'].sum()}")
    print(f"Embryos with duplicate snip_ids: {problem_embryos_df['duplicate_snip_ids'].sum()}")
else:
    print("\n⚠️ WARNING: SAM2 CSV not found!")
    sam2_df = pd.DataFrame()

# ==============================================================================
# 3. Check SAM2 mask file existence
# ==============================================================================
print("\n" + "="*80)
print("3. CHECKING SAM2 MASK FILES")
print("="*80)

problem_embryos_df['mask_files_exist'] = False
problem_embryos_df['mask_files_checked'] = 0
problem_embryos_df['mask_files_missing'] = 0

missing_masks = []

if sam2_csv_path.exists() and not sam2_df.empty:
    # Check environment variable for mask directory override
    base_override = os.environ.get("MORPHSEQ_SANDBOX_MASKS_DIR")
    if base_override:
        mask_base = Path(base_override)
        print(f"\nUsing MORPHSEQ_SANDBOX_MASKS_DIR: {mask_base}")
    else:
        mask_base = MORPHSEQ_PLAYGROUND / "sam2_pipeline_files" / "exported_masks"
        print(f"\nUsing default mask path: {mask_base}")

    mask_dir = mask_base / "20250305" / "masks"
    print(f"Checking masks in: {mask_dir}")
    print(f"Mask directory exists: {mask_dir.exists()}")

    for idx, row in problem_embryos_df.iterrows():
        embryo_id = row['embryo_id']
        embryo_rows = sam2_df[sam2_df['embryo_id'] == embryo_id]

        if len(embryo_rows) > 0 and 'exported_mask_path' in embryo_rows.columns:
            mask_paths = embryo_rows['exported_mask_path'].dropna().unique()

            checked = 0
            missing = 0
            all_exist = True

            for mask_filename in mask_paths:
                checked += 1
                mask_path = mask_dir / mask_filename

                if not mask_path.exists():
                    missing += 1
                    all_exist = False
                    missing_masks.append({
                        'embryo_id': embryo_id,
                        'mask_filename': mask_filename,
                        'expected_path': str(mask_path)
                    })

            problem_embryos_df.at[idx, 'mask_files_checked'] = checked
            problem_embryos_df.at[idx, 'mask_files_missing'] = missing
            problem_embryos_df.at[idx, 'mask_files_exist'] = all_exist

    print(f"\nTotal mask files checked: {problem_embryos_df['mask_files_checked'].sum()}")
    print(f"Total mask files missing: {problem_embryos_df['mask_files_missing'].sum()}")
    print(f"Embryos with all masks existing: {problem_embryos_df['mask_files_exist'].sum()}")

# ==============================================================================
# 4. Check full-frame (FF) image existence
# ==============================================================================
print("\n" + "="*80)
print("4. CHECKING FULL-FRAME IMAGES")
print("="*80)

problem_embryos_df['ff_images_exist'] = False
problem_embryos_df['ff_images_checked'] = 0
problem_embryos_df['ff_images_missing'] = 0
problem_embryos_df['ff_format_found'] = ''  # 'new', 'legacy', 'raw_path', or 'none'

missing_ff_images = []

ff_dir = MORPHSEQ_PLAYGROUND / "built_image_data" / "stitched_FF_images" / "20250305"
print(f"\nChecking FF images in: {ff_dir}")
print(f"FF directory exists: {ff_dir.exists()}")

if sam2_csv_path.exists() and not sam2_df.empty and ff_dir.exists():
    # OPTIMIZATION: Pre-build a set of all FF image basenames for fast lookup
    print("Building FF image index...")
    ff_files = list(ff_dir.glob("*.jpg")) + list(ff_dir.glob("*.png"))
    ff_basenames = {f.stem for f in ff_files}
    print(f"Found {len(ff_basenames)} FF images")

    for idx, row in problem_embryos_df.iterrows():
        embryo_id = row['embryo_id']
        embryo_rows = sam2_df[sam2_df['embryo_id'] == embryo_id]

        if len(embryo_rows) == 0:
            continue

        checked = 0
        missing = 0
        format_found = set()

        for _, emb_row in embryo_rows.iterrows():
            checked += 1
            found_any = False

            # Try new format: check if image_id exists in our set
            if 'image_id' in emb_row and pd.notna(emb_row['image_id']):
                image_id = emb_row['image_id']
                # Check exact match or prefix match
                if image_id in ff_basenames or any(b.startswith(image_id) for b in ff_basenames):
                    format_found.add('new')
                    found_any = True

            # Try legacy format: {well}_t{time}
            if not found_any and 'well' in emb_row:
                well = emb_row['well'] if pd.notna(emb_row.get('well')) else None
                time_int = None
                for col in ['time_int', 'time_key', 'frame_index']:
                    if col in emb_row and pd.notna(emb_row[col]):
                        try:
                            time_int = int(emb_row[col])
                            break
                        except (ValueError, TypeError):
                            pass

                if well and time_int is not None:
                    legacy_stub = f"{well}_t{time_int:04d}"
                    # Check if any basename starts with this pattern
                    if any(b.startswith(legacy_stub) for b in ff_basenames):
                        format_found.add('legacy')
                        found_any = True

            if not found_any:
                missing += 1
                missing_ff_images.append({
                    'embryo_id': embryo_id,
                    'image_id': emb_row.get('image_id', 'N/A'),
                    'well': emb_row.get('well', 'N/A'),
                    'time_int': emb_row.get('time_int', 'N/A')
                })

        problem_embryos_df.at[idx, 'ff_images_checked'] = checked
        problem_embryos_df.at[idx, 'ff_images_missing'] = missing
        problem_embryos_df.at[idx, 'ff_images_exist'] = missing == 0
        problem_embryos_df.at[idx, 'ff_format_found'] = ','.join(sorted(format_found)) if format_found else 'none'

    print(f"\nTotal FF images checked: {problem_embryos_df['ff_images_checked'].sum()}")
    print(f"Total FF images missing: {problem_embryos_df['ff_images_missing'].sum()}")
    print(f"Embryos with all FF images existing: {problem_embryos_df['ff_images_exist'].sum()}")
    print(f"\nFormat distribution:")
    print(problem_embryos_df['ff_format_found'].value_counts())

# ==============================================================================
# 5. Check if snips already exist
# ==============================================================================
print("\n" + "="*80)
print("5. CHECKING EXISTING SNIP FILES")
print("="*80)

problem_embryos_df['snips_exist'] = False
problem_embryos_df['snips_found'] = 0
problem_embryos_df['snips_expected'] = 0

snip_dir = MORPHSEQ_PLAYGROUND / "training_data" / "bf_embryo_snips" / "20250305"
print(f"\nChecking snips in: {snip_dir}")
print(f"Snip directory exists: {snip_dir.exists()}")

if snip_dir.exists():
    # Get all existing snip files
    existing_snips = set([f.stem for f in snip_dir.glob("*.jpg")])
    print(f"Total existing snip files: {len(existing_snips)}")

    if sam2_csv_path.exists() and not sam2_df.empty:
        for idx, row in problem_embryos_df.iterrows():
            embryo_id = row['embryo_id']
            embryo_rows = sam2_df[sam2_df['embryo_id'] == embryo_id]

            if len(embryo_rows) > 0 and 'snip_id' in embryo_rows.columns:
                expected_snips = set(embryo_rows['snip_id'].dropna().unique())
                found_snips = expected_snips & existing_snips

                problem_embryos_df.at[idx, 'snips_expected'] = len(expected_snips)
                problem_embryos_df.at[idx, 'snips_found'] = len(found_snips)
                problem_embryos_df.at[idx, 'snips_exist'] = len(found_snips) == len(expected_snips)

        print(f"\nEmbryos with all snips existing: {problem_embryos_df['snips_exist'].sum()}")
        print(f"Total snips expected: {problem_embryos_df['snips_expected'].sum()}")
        print(f"Total snips found: {problem_embryos_df['snips_found'].sum()}")
        print(f"Total snips missing: {problem_embryos_df['snips_expected'].sum() - problem_embryos_df['snips_found'].sum()}")

# ==============================================================================
# 6. Generate diagnostic summary
# ==============================================================================
print("\n" + "="*80)
print("6. GENERATING DIAGNOSTIC SUMMARY")
print("="*80)

# Add failure mode column
def determine_failure_mode(row):
    """Determine the primary failure mode for this embryo."""
    if not row['in_sam2_csv']:
        return 'NOT_IN_SAM2_CSV'
    elif not row['has_exported_mask_path']:
        return 'NO_MASK_PATH_IN_CSV'
    elif not row['mask_files_exist']:
        return 'MASK_FILE_MISSING'
    elif not row['has_image_id']:
        return 'NO_IMAGE_ID'
    elif not row['ff_images_exist']:
        return 'FF_IMAGE_MISSING'
    elif row['duplicate_snip_ids']:
        return 'DUPLICATE_SNIP_IDS'
    elif row['snips_exist']:
        return 'SNIPS_EXIST_BUILD06_ISSUE'
    else:
        return 'PROCESSING_INCOMPLETE'

problem_embryos_df['failure_mode'] = problem_embryos_df.apply(determine_failure_mode, axis=1)

# Print summary
print("\nFailure Mode Summary:")
print(problem_embryos_df['failure_mode'].value_counts().to_string())

# ==============================================================================
# 7. Save outputs
# ==============================================================================
print("\n" + "="*80)
print("7. SAVING DIAGNOSTIC OUTPUTS")
print("="*80)

# Save detailed diagnosis CSV
diagnosis_csv = OUTPUT_DIR / "missing_snips_diagnosis.csv"
problem_embryos_df.to_csv(diagnosis_csv, index=False)
print(f"\n✓ Saved detailed diagnosis: {diagnosis_csv}")

# Save missing masks list
if missing_masks:
    missing_masks_df = pd.DataFrame(missing_masks)
    missing_masks_txt = OUTPUT_DIR / "missing_masks_list.txt"
    with open(missing_masks_txt, 'w') as f:
        f.write(f"Missing SAM2 Mask Files ({len(missing_masks)} total)\n")
        f.write("="*80 + "\n\n")
        for _, row in missing_masks_df.iterrows():
            f.write(f"Embryo: {row['embryo_id']}\n")
            f.write(f"  File: {row['mask_filename']}\n")
            f.write(f"  Path: {row['expected_path']}\n\n")
    print(f"✓ Saved missing masks list: {missing_masks_txt}")

# Save missing FF images list
if missing_ff_images:
    missing_ff_df = pd.DataFrame(missing_ff_images)
    missing_ff_txt = OUTPUT_DIR / "missing_ff_images_list.txt"
    with open(missing_ff_txt, 'w') as f:
        f.write(f"Missing Full-Frame Images ({len(missing_ff_images)} total)\n")
        f.write("="*80 + "\n\n")
        for _, row in missing_ff_df.iterrows():
            f.write(f"Embryo: {row['embryo_id']}\n")
            f.write(f"  Image ID: {row['image_id']}\n")
            f.write(f"  Well: {row['well']}\n")
            f.write(f"  Time: {row['time_int']}\n\n")
    print(f"✓ Saved missing FF images list: {missing_ff_txt}")

# Save summary report
summary_txt = OUTPUT_DIR / "diagnostic_summary.txt"
with open(summary_txt, 'w') as f:
    f.write("MISSING SNIPS DIAGNOSTIC SUMMARY\n")
    f.write("="*80 + "\n\n")

    f.write(f"Total problem embryos investigated: {len(problem_embryos_df)}\n")
    f.write(f"  - With 100% NaN: {len(exp_20250305_df)}\n")
    f.write(f"  - With partial NaN: {len(partial_nan_df)}\n\n")

    f.write("FAILURE MODE BREAKDOWN:\n")
    f.write("-"*80 + "\n")
    for mode, count in problem_embryos_df['failure_mode'].value_counts().items():
        f.write(f"  {mode}: {count}\n")
    f.write("\n")

    f.write("SAM2 METADATA CHECK:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Embryos in SAM2 CSV: {problem_embryos_df['in_sam2_csv'].sum()}\n")
    f.write(f"  Embryos with mask path: {problem_embryos_df['has_exported_mask_path'].sum()}\n")
    f.write(f"  Embryos with image ID: {problem_embryos_df['has_image_id'].sum()}\n")
    f.write(f"  Embryos with time info: {problem_embryos_df['has_time_info'].sum()}\n")
    f.write(f"  Embryos with duplicates: {problem_embryos_df['duplicate_snip_ids'].sum()}\n\n")

    f.write("FILE EXISTENCE CHECK:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Mask files checked: {problem_embryos_df['mask_files_checked'].sum()}\n")
    f.write(f"  Mask files missing: {problem_embryos_df['mask_files_missing'].sum()}\n")
    f.write(f"  FF images checked: {problem_embryos_df['ff_images_checked'].sum()}\n")
    f.write(f"  FF images missing: {problem_embryos_df['ff_images_missing'].sum()}\n")
    f.write(f"  Snips expected: {problem_embryos_df['snips_expected'].sum()}\n")
    f.write(f"  Snips found: {problem_embryos_df['snips_found'].sum()}\n")
    f.write(f"  Snips missing: {problem_embryos_df['snips_expected'].sum() - problem_embryos_df['snips_found'].sum()}\n\n")

    f.write("NEXT STEPS:\n")
    f.write("-"*80 + "\n")

    # Provide actionable recommendations based on failure modes
    failure_counts = problem_embryos_df['failure_mode'].value_counts()

    if 'FF_IMAGE_MISSING' in failure_counts and failure_counts['FF_IMAGE_MISSING'] > 0:
        f.write(f"\n1. FF_IMAGE_MISSING ({failure_counts['FF_IMAGE_MISSING']} embryos):\n")
        f.write("   → Run Build01 to generate stitched full-frame images\n")
        f.write("   → Check that experiment 20250305 raw data exists\n")

    if 'MASK_FILE_MISSING' in failure_counts and failure_counts['MASK_FILE_MISSING'] > 0:
        f.write(f"\n2. MASK_FILE_MISSING ({failure_counts['MASK_FILE_MISSING']} embryos):\n")
        f.write("   → Run SAM2 pipeline to generate mask files\n")
        f.write("   → Check exported_masks directory permissions\n")

    if 'PROCESSING_INCOMPLETE' in failure_counts and failure_counts['PROCESSING_INCOMPLETE'] > 0:
        f.write(f"\n3. PROCESSING_INCOMPLETE ({failure_counts['PROCESSING_INCOMPLETE']} embryos):\n")
        f.write("   → Wait for Build03A processing to complete (currently ~4% done)\n")
        f.write("   → Check logs for processing errors\n")

    if 'NOT_IN_SAM2_CSV' in failure_counts and failure_counts['NOT_IN_SAM2_CSV'] > 0:
        f.write(f"\n4. NOT_IN_SAM2_CSV ({failure_counts['NOT_IN_SAM2_CSV']} embryos):\n")
        f.write("   → Regenerate SAM2 metadata bridge CSV\n")
        f.write("   → Check SAM2 tracking output files\n")

print(f"✓ Saved summary report: {summary_txt}")

# Save prioritized investigation list
priority_order = [
    'NOT_IN_SAM2_CSV',
    'NO_MASK_PATH_IN_CSV',
    'MASK_FILE_MISSING',
    'NO_IMAGE_ID',
    'FF_IMAGE_MISSING',
    'DUPLICATE_SNIP_IDS',
    'PROCESSING_INCOMPLETE',
    'SNIPS_EXIST_BUILD06_ISSUE'
]

problem_embryos_df['priority'] = problem_embryos_df['failure_mode'].apply(
    lambda x: priority_order.index(x) if x in priority_order else 999
)
prioritized_df = problem_embryos_df.sort_values('priority')

investigation_csv = OUTPUT_DIR / "embryos_to_investigate.csv"
prioritized_df.to_csv(investigation_csv, index=False)
print(f"✓ Saved prioritized investigation list: {investigation_csv}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nKey findings:")
print(f"  - Primary failure mode: {problem_embryos_df['failure_mode'].value_counts().index[0]}")
print(f"  - Total snips missing: {problem_embryos_df['snips_expected'].sum() - problem_embryos_df['snips_found'].sum()}")
print(f"\nReview {summary_txt} for detailed recommendations.")
