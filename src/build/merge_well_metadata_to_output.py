#!/usr/bin/env python3
"""
Quick utility to merge well-level metadata into existing pipeline output files.

Use case: You have a Build06 output file (or any df02/df03 output) and want to add
well-level metadata without re-running the entire pipeline. This script:
1. Loads the output file
2. Extracts well information from existing columns
3. Loads well metadata from Excel
4. Merges by well_id
5. Saves the updated file

Usage:
    python merge_well_metadata_to_output.py <output_file> <metadata_excel> [--backup]

Example:
    python merge_well_metadata_to_output.py \
      morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251106.csv \
      metadata/plate_metadata/20251106_well_metadata.xlsx \
      --backup
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import Tuple
import shutil


def extract_well_from_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Extract well information from existing columns.

    Tries multiple approaches:
    1. If 'well' column exists, use it
    2. If 'snip_id' exists (format: 20251106_A01_e01_...), extract from there
    3. If 'embryo_id' exists, extract from there
    4. If 'image_id' exists, extract from there

    Returns:
        DataFrame with 'well' column added
    """
    df = df.copy()

    if 'well' in df.columns and df['well'].notna().any():
        print(f"  ✓ Found existing 'well' column")
        return df

    # Try snip_id format: 20251106_A01_e01_t0001
    if 'snip_id' in df.columns:
        try:
            well_from_snip = df['snip_id'].str.split('_').str[1]
            if well_from_snip.notna().any():
                df['well'] = well_from_snip
                print(f"  ✓ Extracted 'well' from 'snip_id' column ({df['well'].nunique()} unique wells)")
                return df
        except Exception as e:
            print(f"  ⚠ Could not extract from snip_id: {e}")

    # Try embryo_id format
    if 'embryo_id' in df.columns:
        try:
            well_from_embryo = df['embryo_id'].str.split('_').str[1]
            if well_from_embryo.notna().any():
                df['well'] = well_from_embryo
                print(f"  ✓ Extracted 'well' from 'embryo_id' column ({df['well'].nunique()} unique wells)")
                return df
        except Exception as e:
            print(f"  ⚠ Could not extract from embryo_id: {e}")

    # Try image_id format
    if 'image_id' in df.columns:
        try:
            well_from_image = df['image_id'].str.split('_').str[1]
            if well_from_image.notna().any():
                df['well'] = well_from_image
                print(f"  ✓ Extracted 'well' from 'image_id' column ({df['well'].nunique()} unique wells)")
                return df
        except Exception as e:
            print(f"  ⚠ Could not extract from image_id: {e}")

    raise ValueError("Could not extract 'well' information from any column")


def load_well_metadata_from_excel(excel_path: str | Path) -> pd.DataFrame:
    """Load well metadata from Excel file using the same logic as export_utils.py.

    Returns:
        DataFrame with well-level metadata with columns: well, and all sheet data
    """
    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    print(f"\n📊 Loading metadata from Excel: {excel_path.name}")

    well_sheets = [
        "medium",
        "genotype",
        "chem_perturbation",
        "start_age_hpf",
        "embryos_per_well",
        "temperature",
        "pair",
    ]

    with pd.ExcelFile(excel_path) as xlf:
        print(f"   Available sheets: {xlf.sheet_names}")

        # Build base well names A01…H12
        wells = [f"{r}{c:02}" for r in "ABCDEFGH" for c in range(1, 13)]
        plate_df = pd.DataFrame({"well": wells})

        # Parse each sheet
        for sheet in well_sheets:
            if sheet in xlf.sheet_names:
                df = xlf.parse(sheet, header=0)
                block = df.iloc[:8, 1:13]  # 8 rows, 12 cols

                # Handle different data types
                if sheet in ["temperature", "embryos_per_well"]:
                    block = block.reindex(index=range(8), columns=range(1, 13), fill_value=np.nan)
                    arr = block.to_numpy(dtype=float).ravel()
                else:
                    block = block.reindex(index=range(8), columns=range(1, 13), fill_value='')
                    arr = block.to_numpy(dtype=str).ravel()

                plate_df[sheet] = arr
                print(f"   ✓ Loaded '{sheet}' (unique values: {len(plate_df[sheet].unique())})")
            else:
                plate_df[sheet] = np.nan
                print(f"   ⚠ Sheet '{sheet}' not found (will be NaN)")

    # Filter out empty rows (where start_age_hpf is empty)
    if "start_age_hpf" in plate_df.columns:
        plate_df = plate_df[plate_df["start_age_hpf"] != ''].reset_index(drop=True)
        print(f"\n✅ Loaded {len(plate_df)} wells with metadata")

    return plate_df


def merge_metadata(output_df: pd.DataFrame, metadata_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Merge well-level metadata into output dataframe.

    Args:
        output_df: DataFrame with output data (must have 'well' column)
        metadata_df: DataFrame with well-level metadata (has 'well' column)
        verbose: Print merge statistics

    Returns:
        Merged dataframe with metadata values prioritized, original values saved as _previous
    """
    if verbose:
        print(f"\n🔗 Merging metadata on 'well'...")
        print(f"   Output: {len(output_df)} rows, {output_df['well'].nunique()} unique wells")
        print(f"   Metadata: {len(metadata_df)} rows, {metadata_df['well'].nunique()} unique wells")

    result_df = output_df.copy()

    # Get all metadata columns except 'well'
    merge_cols = [col for col in metadata_df.columns if col != 'well']

    overwritten_cols = []

    # Create a mapping dict for each metadata column
    for col in merge_cols:
        # Create a dict: well -> metadata_value
        metadata_dict = dict(zip(metadata_df['well'], metadata_df[col]))

        if col in result_df.columns:
            # Column exists in output - check for differences and save as _previous
            original_values = result_df[col].copy()
            new_values = result_df['well'].map(metadata_dict)

            # Check if there are any actual differences
            diff_mask = (original_values.fillna('__NA__') != new_values.fillna('__NA__'))
            if diff_mask.any():
                result_df[f"{col}_previous"] = original_values
                overwritten_cols.append(col)

            # Update with metadata values, keeping original where metadata is missing
            result_df[col] = new_values.fillna(original_values)
        else:
            # New column - just add the metadata values
            result_df[col] = result_df['well'].map(metadata_dict)

    if verbose:
        matched = len(result_df)
        print(f"   ✅ Matched {matched}/{len(result_df)} rows ({100*matched/len(result_df):.1f}%)")
        print(f"   ✓ Merged {len(merge_cols)} metadata columns: {', '.join(merge_cols[:5])}{'...' if len(merge_cols) > 5 else ''}")
        if overwritten_cols:
            print(f"   ⚠️  Overwrote {len(overwritten_cols)} existing columns (saved as _previous): {', '.join(overwritten_cols[:5])}{'...' if len(overwritten_cols) > 5 else ''}")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Merge well-level metadata into existing pipeline output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge pair data into Build06 output
  python merge_well_metadata_to_output.py \\
    morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251112.csv \\
    metadata/plate_metadata/20251112_well_metadata.xlsx \\
    --backup

  # Update Build04 output with metadata
  python merge_well_metadata_to_output.py \\
    metadata/build04_output/qc_staged_20251106.csv \\
    metadata/plate_metadata/20251106_well_metadata.xlsx
        """
    )

    parser.add_argument("output_file", type=str, help="Path to output CSV file to update")
    parser.add_argument("metadata_excel", type=str, help="Path to well metadata Excel file")
    parser.add_argument("--backup", action="store_true", help="Create .backup of original file")
    parser.add_argument("--output", "-o", type=str, help="Output file path (default: overwrite)")

    args = parser.parse_args()

    output_path = Path(args.output_file)
    excel_path = Path(args.metadata_excel)

    # Validate inputs
    if not output_path.exists():
        print(f"❌ Output file not found: {output_path}")
        sys.exit(1)

    if not excel_path.exists():
        print(f"❌ Excel file not found: {excel_path}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"📋 MERGE WELL METADATA TO OUTPUT FILE")
    print(f"{'='*80}")

    try:
        # Load output file
        print(f"\n📖 Loading output file: {output_path.name}")
        output_df = pd.read_csv(output_path, low_memory=False)
        print(f"   ✓ Loaded {len(output_df)} rows, {len(output_df.columns)} columns")

        # Extract well information
        print(f"\n🔍 Extracting well information...")
        output_df = extract_well_from_columns(output_df)

        # Load metadata
        metadata_df = load_well_metadata_from_excel(excel_path)

        # Merge
        merged_df = merge_metadata(output_df, metadata_df, verbose=True)

        # Save
        output_file = args.output if args.output else str(output_path)

        if args.backup and Path(output_file).exists():
            backup_path = f"{output_file}.backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy(output_file, backup_path)
            print(f"\n💾 Created backup: {backup_path}")

        merged_df.to_csv(output_file, index=False)
        print(f"✅ Wrote merged data to: {output_file}")
        print(f"   {len(merged_df)} rows × {len(merged_df.columns)} columns")

        print(f"\n{'='*80}")
        print(f"✨ Done! Metadata merged successfully")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
