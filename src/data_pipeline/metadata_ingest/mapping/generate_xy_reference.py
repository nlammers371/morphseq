"""
Generate reference plate XY coordinates from 20251112 experiment.

This script extracts the verified XY stage positions from the 20251112 experiment
(which has correct well-to-position mapping) and saves them as a reference file
for use in XY-based well mapping.

Run once to create: morphseq_playground/metadata/ref_plate_xy_coordinates.csv

Usage:
    python generate_xy_reference.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import nd2
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Reference experiment with verified correct mapping
REFERENCE_EXP_ID = "20251112"

# Paths
BASE_PATH = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
PLAYGROUND_PATH = BASE_PATH / "morphseq_playground"
RAW_DATA_PATH = PLAYGROUND_PATH / "raw_image_data" / "YX1"
METADATA_PATH = BASE_PATH / "metadata" / "plate_metadata"
OUTPUT_PATH = PLAYGROUND_PATH / "metadata" / "YX1_nd2_ref_plate_xy_coordinates.csv"


def extract_nd2_stage_positions(nd2_path: Path) -> pd.DataFrame:
    """
    Extract stage positions from ND2 file.

    Parameters
    ----------
    nd2_path : Path
        Path to ND2 file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [P, x_um, y_um]
    """
    log.info(f"Extracting stage positions from: {nd2_path}")

    with nd2.ND2File(str(nd2_path)) as f:
        sizes = f.sizes
        P = sizes.get("P", sizes.get("W", 1))
        Z = sizes.get("Z", 1)
        C = sizes.get("C", 1)

        log.info(f"  ND2 dimensions: P={P}, Z={Z}, C={C}")

        records = []
        for w in range(P):
            idx = w * (Z * C)  # T=0, first frame per position
            try:
                md = f.frame_metadata(idx)
                ch = getattr(md, "channels", [None])[0]
                if ch and hasattr(ch, "position"):
                    stage = ch.position.stagePositionUm
                    records.append({
                        "P": w,
                        "x_um": getattr(stage, "x", np.nan),
                        "y_um": getattr(stage, "y", np.nan)
                    })
            except Exception as e:
                log.warning(f"  Skipping P={w}: {e}")
                continue

        df = pd.DataFrame(records)
        log.info(f"  Extracted {len(df)} positions")
        return df


def load_series_number_map(exp_id: str) -> dict:
    """
    Load explicit series-to-well mapping from Excel metadata.

    Mimics build01B_compile_yx1_images_torch.py lines 490-530 exactly.
    Iterates column-major: all rows in column 1, then all rows in column 2, etc.

    Parameters
    ----------
    exp_id : str
        Experiment ID

    Returns
    -------
    dict
        Dictionary mapping series number (1-based) to well name (e.g., "A01")
    """
    metadata_path = METADATA_PATH / f"{exp_id}_well_metadata.xlsx"

    if not metadata_path.exists():
        raise FileNotFoundError(f"No plate metadata found: {metadata_path}")

    log.info(f"Loading series_number_map from: {metadata_path}")

    # Read series_number_map sheet
    sm_raw = pd.read_excel(metadata_path, sheet_name='series_number_map', header=None)

    # Detect and handle header row
    data_rows = sm_raw
    try:
        header_like = list(sm_raw.iloc[0, 1:13].astype(object))
        if header_like == list(range(1, 13)):
            data_rows = sm_raw.iloc[1:9, :]  # rows A..H in 1..8
        else:
            data_rows = sm_raw.iloc[:8, :]
    except Exception:
        data_rows = sm_raw.iloc[:8, :]

    series_map = data_rows.iloc[:, 1:13]  # 8x12 numeric grid

    # Build mapping (column-major order, as in build01B)
    mapping = {}
    col_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    row_letter_list = ["A", "B", "C", "D", "E", "F", "G", "H"]

    for c in range(len(col_id_list)):
        for r in range(len(row_letter_list)):
            val = series_map.iloc[r, c]
            if pd.notna(val):
                try:
                    series_idx_1b = int(val)
                    well_name = row_letter_list[r] + f"{col_id_list[c]:02d}"
                    mapping[series_idx_1b] = well_name
                except Exception:
                    continue

    log.info(f"  Loaded {len(mapping)} series-to-well mappings")
    return mapping


def find_nd2_file(exp_id: str) -> Path:
    """Locate the ND2 file for a given experiment."""
    nd2_dir = RAW_DATA_PATH / exp_id
    nd2_files = list(nd2_dir.glob("*.nd2"))

    if len(nd2_files) == 0:
        raise FileNotFoundError(f"No ND2 file found in {nd2_dir}")
    elif len(nd2_files) > 1:
        log.warning(f"Multiple ND2 files found in {nd2_dir}, using first one")

    return nd2_files[0]


def generate_reference_coordinates(exp_id: str = REFERENCE_EXP_ID) -> pd.DataFrame:
    """
    Generate reference XY coordinates from a verified experiment.

    Parameters
    ----------
    exp_id : str
        Experiment ID to use as reference (default: 20251112)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [well, x_um, y_um]
    """
    log.info(f"Generating reference coordinates from experiment: {exp_id}")

    # Find ND2 file
    nd2_path = find_nd2_file(exp_id)

    # Extract stage positions
    positions_df = extract_nd2_stage_positions(nd2_path)

    # Load series-to-well mapping
    series_map = load_series_number_map(exp_id)

    # Map P indices to well names
    # P index (0-based) -> series number (1-based) -> well name
    positions_df['series_num'] = positions_df['P'] + 1
    positions_df['well'] = positions_df['series_num'].apply(
        lambda s: series_map.get(int(s), None)
    )

    # Filter out positions without valid well mapping
    valid_positions = positions_df[positions_df['well'].notna()].copy()

    if len(valid_positions) < len(positions_df):
        log.warning(
            f"  {len(positions_df) - len(valid_positions)} positions had no well mapping"
        )

    # Create reference DataFrame with just well, x_um, y_um
    reference_df = valid_positions[['well', 'x_um', 'y_um']].copy()
    reference_df = reference_df.sort_values('well').reset_index(drop=True)

    log.info(f"  Generated reference with {len(reference_df)} wells")

    # Print summary statistics
    log.info(f"  X range: {reference_df['x_um'].min():.1f} to {reference_df['x_um'].max():.1f} µm")
    log.info(f"  Y range: {reference_df['y_um'].min():.1f} to {reference_df['y_um'].max():.1f} µm")

    return reference_df


def main():
    """Generate and save reference coordinates."""
    log.info("=" * 80)
    log.info("GENERATING REFERENCE PLATE XY COORDINATES")
    log.info("=" * 80)

    # Generate reference
    reference_df = generate_reference_coordinates(REFERENCE_EXP_ID)

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    reference_df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"\nSaved reference coordinates to: {OUTPUT_PATH}")

    # Print sample
    log.info("\nSample of reference coordinates:")
    print(reference_df.head(10).to_string(index=False))

    log.info("\n" + "=" * 80)
    log.info("REFERENCE GENERATION COMPLETE")
    log.info("=" * 80)


if __name__ == '__main__':
    main()
