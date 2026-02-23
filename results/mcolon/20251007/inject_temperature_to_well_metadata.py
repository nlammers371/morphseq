#!/usr/bin/env python3
"""
Inject Temperature Data into Well Metadata Excel Files

This script extracts temperature information from embryo_stats_df.csv and injects it
into the well metadata Excel files that are missing temperature data.

The script carefully validates well patterns (A-H followed by 01-12) to distinguish
real wells from complex experiment IDs.

SAFETY FEATURES:
- Creates backups before modifying files
- Never overrides existing temperature data
- Test mode: only processes specified experiments

Author: Claude Code
Date: 2025-10-07
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from typing import Dict, Tuple, List, Optional
import warnings
import shutil
from datetime import datetime
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

# Paths
REPO_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
EMBRYO_STATS_CSV = REPO_ROOT / "results/mcolon/20250917/data/embryo_stats_df.csv"
WELL_METADATA_DIR = REPO_ROOT / "morphseq_playground/metadata/well_metadata"
BACKUP_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251007/backups")

# TEST MODE: Only process these experiments (set to None to process all)
TEST_EXPERIMENTS = None  # Change to None to process all experiments

# Well pattern regex: exactly A-H followed by 01-12
WELL_PATTERN = re.compile(r'^[A-H](0[1-9]|1[0-2])$')

def is_valid_well(well_str: str) -> bool:
    """
    Validate if a string matches the expected well pattern: A-H followed by 01-12.

    Examples:
        'A01' -> True
        'H12' -> True
        'A13' -> False (column out of range)
        'I01' -> False (row out of range)
        'atf6' -> False (not a well)
        'part2' -> False (not a well)

    Args:
        well_str: String to validate

    Returns:
        True if valid well, False otherwise
    """
    if not isinstance(well_str, str):
        return False
    return bool(WELL_PATTERN.match(well_str))


def extract_well_from_snip_id(snip_id: str) -> Optional[str]:
    """
    Extract well ID from snip_id, validating it's a real well.

    snip_id format: {date}_{well}_e{embryo}_t{time}
    Example: '20230525_A03_e00_t0001' -> 'A03'

    Args:
        snip_id: The snip ID string

    Returns:
        Well ID if valid, None otherwise
    """
    try:
        parts = snip_id.split('_')
        for token in parts:
            if is_valid_well(token):
                return token
    except Exception:
        pass
    return None


def extract_temperature_by_experiment_well(
    embryo_stats_path: Path
) -> Dict[str, Dict[str, float]]:
    """
    Extract temperature data from embryo_stats_df.csv grouped by experiment and well.

    Only includes entries with valid well IDs (A-H, 01-12 pattern).

    Args:
        embryo_stats_path: Path to embryo_stats_df.csv

    Returns:
        Nested dict: {experiment_date: {well: temperature}}
    """
    log.info(f"Reading embryo stats from: {embryo_stats_path}")
    df = pd.read_csv(embryo_stats_path)

    # Extract well from snip_id
    df['well'] = df['snip_id'].apply(extract_well_from_snip_id)

    # Filter to only valid wells
    valid_wells_mask = df['well'].notna()
    df_valid = df[valid_wells_mask].copy()

    invalid_count = (~valid_wells_mask).sum()
    if invalid_count > 0:
        log.info(f"Filtered out {invalid_count} entries with invalid/missing well IDs")

    # Convert experiment_date to string
    df_valid['experiment_date'] = df_valid['experiment_date'].astype(str)

    # Group by experiment and well, taking median temperature
    temp_data = {}

    for exp_date in df_valid['experiment_date'].unique():
        exp_df = df_valid[df_valid['experiment_date'] == exp_date]
        well_temps = {}

        for well in exp_df['well'].unique():
            well_df = exp_df[exp_df['well'] == well]
            temps = well_df['temperature'].dropna()

            if len(temps) > 0:
                # Use median to handle any inconsistencies
                median_temp = temps.median()
                well_temps[well] = median_temp

                # Warn if there's high variance
                if len(temps) > 1:
                    temp_std = temps.std()
                    if temp_std > 0.5:
                        log.warning(
                            f"  {exp_date} well {well}: temperature varies "
                            f"(std={temp_std:.2f}), using median={median_temp:.1f}"
                        )

        if well_temps:
            temp_data[exp_date] = well_temps
            log.info(
                f"  {exp_date}: extracted temperature for {len(well_temps)} wells"
            )

    return temp_data


def create_temperature_plate_array(well_temps: Dict[str, float], impute: bool = True) -> np.ndarray:
    """
    Create 8x12 plate array from well temperature mapping.

    Args:
        well_temps: Dict mapping well IDs to temperatures
        impute: If True, fill empty wells with the common temperature value

    Returns:
        8x12 numpy array with temperatures (NaN for missing wells if impute=False)
    """
    plate = np.full((8, 12), np.nan)

    # Fill in known wells
    for well, temp in well_temps.items():
        if not is_valid_well(well):
            continue

        row = ord(well[0]) - ord('A')  # A=0, B=1, ..., H=7
        col = int(well[1:]) - 1        # 01=0, 02=1, ..., 12=11

        if 0 <= row < 8 and 0 <= col < 12:
            plate[row, col] = temp

    # Impute missing wells if requested
    if impute and len(well_temps) > 0:
        # Get the most common temperature (mode)
        temps = list(well_temps.values())
        # Use the mode, or if all different, use median
        from collections import Counter
        temp_counts = Counter(temps)
        common_temp = temp_counts.most_common(1)[0][0]

        # Check if temperatures are consistent (within 0.5°C)
        temp_array = np.array(temps)
        temp_std = temp_array.std()

        if temp_std > 0.5:
            log.warning(
                f"    ⚠️  Temperature varies across wells (std={temp_std:.2f}°C). "
                f"Using most common: {common_temp}°C"
            )

        # Fill all NaN values with the common temperature
        plate[np.isnan(plate)] = common_temp
        log.info(f"    Imputed {96 - len(well_temps)} empty wells with {common_temp}°C")

    return plate


def update_well_metadata_excel(
    excel_path: Path,
    well_temps: Dict[str, float],
    create_if_missing: bool = False,
    impute_missing: bool = True
) -> bool:
    """
    Update or create temperature sheet in well metadata Excel file.

    Args:
        excel_path: Path to the Excel file
        well_temps: Dict mapping well IDs to temperatures
        create_if_missing: If True, create new Excel file if it doesn't exist
        impute_missing: If True, fill empty wells with common temperature

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create temperature plate array (with imputation)
        temp_plate = create_temperature_plate_array(well_temps, impute=impute_missing)

        # Create DataFrame with proper structure
        # Row labels: A-H, Column labels: 1-12
        temp_df = pd.DataFrame(
            temp_plate,
            index=list('ABCDEFGH'),
            columns=range(1, 13)
        )

        if excel_path.exists():
            # Read existing Excel file
            with pd.ExcelFile(excel_path) as xlf:
                existing_sheets = xlf.sheet_names

                # Read all existing sheets
                sheet_data = {}
                for sheet in existing_sheets:
                    if sheet != 'temperature':
                        sheet_data[sheet] = xlf.parse(sheet)

            # Write back all sheets including updated temperature
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Write existing sheets first
                for sheet_name, sheet_df in sheet_data.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Write temperature sheet
                temp_df.to_excel(writer, sheet_name='temperature', index=True)

            log.info(f"  ✓ Updated temperature sheet in: {excel_path.name}")
            return True

        elif create_if_missing:
            # Create new Excel file with just temperature sheet
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                temp_df.to_excel(writer, sheet_name='temperature', index=True)

            log.info(f"  ✓ Created new Excel file with temperature: {excel_path.name}")
            return True
        else:
            log.warning(f"  ✗ Excel file does not exist: {excel_path.name}")
            return False

    except Exception as e:
        log.error(f"  ✗ Error updating {excel_path.name}: {e}")
        return False


def check_existing_temperature(excel_path: Path) -> Tuple[bool, int]:
    """
    Check if Excel file already has valid temperature data.

    Args:
        excel_path: Path to Excel file

    Returns:
        Tuple of (has_temperature, num_wells_with_temp)
    """
    if not excel_path.exists():
        return False, 0

    try:
        with pd.ExcelFile(excel_path) as xlf:
            if 'temperature' not in xlf.sheet_names:
                return False, 0

            temp_df = xlf.parse('temperature', header=0)
            # Check if there are any non-NaN temperature values
            temp_values = temp_df.iloc[:8, 1:13]
            has_temps = temp_values.notna().any().any()
            num_wells = temp_values.notna().sum().sum()
            return has_temps, int(num_wells)
    except Exception:
        return False, 0


def create_backup(excel_path: Path, backup_dir: Path) -> Optional[Path]:
    """
    Create a backup of the Excel file before modification.

    Args:
        excel_path: Path to Excel file to backup
        backup_dir: Directory to store backups

    Returns:
        Path to backup file, or None if backup failed
    """
    if not excel_path.exists():
        return None

    try:
        # Create backup directory if needed
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{excel_path.stem}_backup_{timestamp}{excel_path.suffix}"
        backup_path = backup_dir / backup_name

        # Copy file
        shutil.copy2(excel_path, backup_path)
        log.info(f"  ✓ Created backup: {backup_name}")
        return backup_path

    except Exception as e:
        log.error(f"  ✗ Failed to create backup: {e}")
        return None


def main():
    """Main execution function."""
    log.info("=" * 70)
    log.info("Temperature Injection Script - Starting")
    log.info("=" * 70)

    # Show test mode status
    if TEST_EXPERIMENTS:
        log.info(f"\n🧪 TEST MODE: Only processing experiments: {TEST_EXPERIMENTS}")
    else:
        log.info("\n⚠️  FULL MODE: Will process all experiments")

    log.info(f"Backup directory: {BACKUP_DIR}")

    # Validate paths
    if not EMBRYO_STATS_CSV.exists():
        log.error(f"Embryo stats CSV not found: {EMBRYO_STATS_CSV}")
        return

    if not WELL_METADATA_DIR.exists():
        log.error(f"Well metadata directory not found: {WELL_METADATA_DIR}")
        return

    # Extract temperature data
    log.info("\nStep 1: Extracting temperature data from embryo_stats_df.csv")
    log.info("-" * 70)
    temp_data = extract_temperature_by_experiment_well(EMBRYO_STATS_CSV)

    if not temp_data:
        log.error("No temperature data extracted!")
        return

    log.info(f"\n✓ Extracted temperature data for {len(temp_data)} experiments")

    # Filter to test experiments if in test mode
    if TEST_EXPERIMENTS:
        temp_data = {k: v for k, v in temp_data.items() if k in TEST_EXPERIMENTS}
        log.info(f"✓ Filtered to {len(temp_data)} test experiments")

    if not temp_data:
        log.error("No experiments to process after filtering!")
        return

    # Process each experiment
    log.info("\nStep 2: Updating well metadata Excel files")
    log.info("-" * 70)

    updated_count = 0
    skipped_count = 0
    created_count = 0
    backup_count = 0

    for exp_date, well_temps in temp_data.items():
        excel_path = WELL_METADATA_DIR / f"{exp_date}_well_metadata.xlsx"

        log.info(f"\n{exp_date}:")
        log.info(f"  Wells with temperature in embryo_stats: {len(well_temps)}")

        # Check if already has temperature - NEVER OVERRIDE
        has_temp, num_existing = check_existing_temperature(excel_path)
        if has_temp:
            log.info(f"  ⊘ SKIPPING - already has temperature data ({num_existing} wells)")
            log.info(f"  ⊘ Will NOT override existing data")
            skipped_count += 1
            continue

        # Create backup before modifying (if file exists)
        if excel_path.exists():
            backup_path = create_backup(excel_path, BACKUP_DIR)
            if backup_path:
                backup_count += 1
            else:
                log.error(f"  ✗ Cannot proceed without backup - skipping")
                continue

        # Update or create (with imputation enabled)
        success = update_well_metadata_excel(
            excel_path,
            well_temps,
            create_if_missing=True,
            impute_missing=True
        )

        if success:
            if excel_path.exists():
                updated_count += 1
            else:
                created_count += 1

    # Summary
    log.info("\n" + "=" * 70)
    log.info("Summary:")
    log.info(f"  Backups created: {backup_count} files")
    log.info(f"  Updated: {updated_count} files")
    log.info(f"  Created: {created_count} files")
    log.info(f"  Skipped: {skipped_count} files (already had temperature)")
    log.info("=" * 70)

    # List experiments that were processed
    log.info("\nProcessed experiments:")
    for exp_date in sorted(temp_data.keys()):
        excel_path = WELL_METADATA_DIR / f"{exp_date}_well_metadata.xlsx"
        has_temp, num_wells = check_existing_temperature(excel_path)
        if excel_path.exists():
            if has_temp:
                log.info(f"  ✓ {exp_date} (has temperature: {num_wells} wells)")
            else:
                log.info(f"  ✓ {exp_date} (exists but no temperature)")
        else:
            log.info(f"  ✗ {exp_date} (file not found)")

    if backup_count > 0:
        log.info(f"\n💾 Backups saved to: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
