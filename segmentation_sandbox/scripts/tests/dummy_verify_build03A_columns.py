#!/usr/bin/env python3
"""
Dummy script: verify Build03A-required columns exist in a CSV

Checks for presence of columns used by src/build/build03A_process_images.py
so downstream logic can rely on them.
"""

import argparse
import pandas as pd

REQUIRED_COLS = [
    'image_id', 'embryo_id', 'snip_id',
    'experiment_id', 'video_id', 'exported_mask_path',
    'Height (um)', 'Height (px)'
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to exported SAM2 CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
    else:
        print("✅ All required columns present for Build03A")
    print(f"Columns in file ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()

