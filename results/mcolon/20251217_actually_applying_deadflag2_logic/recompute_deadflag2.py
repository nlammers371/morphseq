#!/usr/bin/env python3
"""
Recompute dead_flag2 and use_embryo_flag for Build04 outputs.

Uses corrected 80% persistence threshold (from QC_DEFAULTS)
instead of legacy 25% hardcoded value.

Usage:
    python recompute_deadflag2.py                    # Test on 20251121
    python recompute_deadflag2.py --all              # Process all experiments
    python recompute_deadflag2.py --archive          # Archive old files and replace
"""
import sys
import argparse
import shutil
from datetime import datetime
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.quality_control.death_detection import compute_dead_flag2_persistence
from src.build.qc.embryo_flags import determine_use_embryo_flag
from src.data_pipeline.quality_control.config import QC_DEFAULTS

# Paths
BUILD04_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build04_output"
OUTPUT_DIR = Path(__file__).parent / "output"


def recompute_flags(input_csv: Path, output_csv: Path, test_embryo_id: str = None):
    """Recompute dead_flag2 and use_embryo_flag for a build04 CSV."""

    print(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} rows, {df['embryo_id'].nunique()} embryos")

    # Save before state for test embryo
    if test_embryo_id:
        before = df[df['embryo_id'] == test_embryo_id].copy()
        print(f"\n=== BEFORE ({test_embryo_id}) ===")
        print(f"  dead_flag2 True: {before['dead_flag2'].sum()} / {len(before)} rows")
        print(f"  use_embryo_flag True: {before['use_embryo_flag'].sum()} / {len(before)} rows")
        if 'dead_inflection_time_int' in before.columns:
            inflection_val = before['dead_inflection_time_int'].iloc[0]
            print(f"  dead_inflection_time_int: {inflection_val}")

    # Reset dead_flag2 columns before recomputation
    df['dead_flag2'] = False
    df['dead_inflection_time_int'] = pd.NA

    # Recompute dead_flag2 using corrected persistence threshold (0.80)
    print(f"\nRecomputing dead_flag2 (persistence_threshold={QC_DEFAULTS['persistence_threshold']})...")
    df = compute_dead_flag2_persistence(df, dead_lead_time=QC_DEFAULTS['dead_lead_time_hours'])

    # Recompute use_embryo_flag (depends on updated dead_flag2)
    print("Recomputing use_embryo_flag...")
    df['use_embryo_flag'] = determine_use_embryo_flag(df)

    # Show after state for test embryo
    if test_embryo_id:
        after = df[df['embryo_id'] == test_embryo_id].copy()
        print(f"\n=== AFTER ({test_embryo_id}) ===")
        print(f"  dead_flag2 True: {after['dead_flag2'].sum()} / {len(after)} rows")
        print(f"  use_embryo_flag True: {after['use_embryo_flag'].sum()} / {len(after)} rows")
        if 'dead_inflection_time_int' in after.columns:
            inflection_val = after['dead_inflection_time_int'].iloc[0]
            print(f"  dead_inflection_time_int: {inflection_val}")

    # Save output
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    return df


def process_single_experiment(exp: str, test_embryo_id: str = None):
    """Process a single experiment."""
    input_csv = BUILD04_DIR / f"qc_staged_{exp}.csv"
    output_csv = OUTPUT_DIR / f"qc_staged_{exp}_recomputed.csv"

    if not input_csv.exists():
        print(f"ERROR: Input file not found: {input_csv}")
        return None

    return recompute_flags(input_csv, output_csv, test_embryo_id)


def process_all_experiments():
    """Process all experiments in build04_output."""
    experiments = [p.stem.replace("qc_staged_", "")
                   for p in BUILD04_DIR.glob("qc_staged_*.csv")]

    print(f"Found {len(experiments)} experiments to process")
    print(f"Experiments: {experiments}")
    print()

    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(experiments)}: {exp}")
        print(f"{'='*60}")
        process_single_experiment(exp)

    print(f"\nCompleted processing {len(experiments)} experiments")


def archive_and_replace():
    """Archive old build04 CSVs and replace with recomputed versions."""

    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = BUILD04_DIR / f"archive_{timestamp}_pre_deadflag2_fix"
    archive_dir.mkdir(exist_ok=True)

    # Get all recomputed files
    recomputed_files = list(OUTPUT_DIR.glob("qc_staged_*_recomputed.csv"))

    if not recomputed_files:
        print("ERROR: No recomputed files found in output directory")
        return

    print(f"Found {len(recomputed_files)} recomputed files to process")
    print(f"Archive directory: {archive_dir}")
    print()

    for recomputed_path in recomputed_files:
        # Extract experiment name (remove _recomputed suffix)
        exp = recomputed_path.stem.replace("qc_staged_", "").replace("_recomputed", "")
        original_path = BUILD04_DIR / f"qc_staged_{exp}.csv"
        archive_path = archive_dir / f"qc_staged_{exp}.csv"
        final_path = BUILD04_DIR / f"qc_staged_{exp}.csv"

        # Archive original
        if original_path.exists():
            print(f"Archiving: {original_path.name} -> archive/{original_path.name}")
            shutil.move(str(original_path), str(archive_path))
        else:
            print(f"WARNING: Original file not found: {original_path.name}")

        # Replace with recomputed (remove _recomputed suffix)
        print(f"Replacing: {recomputed_path.name} -> {final_path.name}")
        shutil.copy(str(recomputed_path), str(final_path))

    print(f"\nArchived {len(recomputed_files)} files to: {archive_dir}")
    print(f"Replaced build04 outputs with recomputed versions")


def main():
    parser = argparse.ArgumentParser(description="Recompute dead_flag2 for Build04 outputs")
    parser.add_argument("--all", action="store_true", help="Process all experiments")
    parser.add_argument("--archive", action="store_true", help="Archive old files and replace")
    parser.add_argument("--exp", type=str, default="20251121", help="Single experiment to process")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.archive:
        archive_and_replace()
    elif args.all:
        process_all_experiments()
    else:
        # Default: test on single experiment
        print(f"Testing on experiment: {args.exp}")
        print(f"Config: persistence_threshold={QC_DEFAULTS['persistence_threshold']}, "
              f"min_decline_rate={QC_DEFAULTS['min_decline_rate']}, "
              f"dead_lead_time={QC_DEFAULTS['dead_lead_time_hours']}hr")
        print()

        test_embryo = f"{args.exp}_D04_e01" if args.exp == "20251121" else None
        process_single_experiment(args.exp, test_embryo_id=test_embryo)


if __name__ == "__main__":
    main()
