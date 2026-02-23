#!/usr/bin/env python3
"""
Detect mixed staleness: where some files are old and some are new in the same directory.

This is the CORRECT way to detect staleness - not by checking the newest file,
but by checking if ANY files are significantly older than the stitched images.

This replicates the detection method we used for 20251125:
- Check individual file timestamps
- Find if there's a mix of old and new files
- Report experiments that have this mixing pattern
"""

import os
import glob
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")

def get_file_mtime_range(directory, extension):
    """
    Get min and max modification times for files in a directory.
    Returns (min_time, max_time, file_count) or (0, 0, 0) if no files.
    """
    times = []
    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith(extension):
                times.append(entry.stat().st_mtime)
    except FileNotFoundError:
        return 0, 0, 0

    if not times:
        return 0, 0, 0

    return min(times), max(times), len(times)

def get_stitched_time_range(exp_id):
    """Get the time range of stitched images."""
    stitched_dir = BASE_DIR / "built_image_data" / "stitched_FF_images" / exp_id
    min_t, max_t, count = get_file_mtime_range(stitched_dir, ".jpg")
    return min_t, max_t, count

def detect_staleness(exp_id):
    """
    Detect staleness by checking if:
    1. SAM2 masks have files OLDER than the oldest stitched image
    2. BF snips have files OLDER than the oldest stitched image
    """
    stitched_min, stitched_max, stitched_count = get_stitched_time_range(exp_id)

    if stitched_count == 0:
        return None  # No stitched images

    # Check SAM2 masks
    sam2_dir = BASE_DIR / "sam2_pipeline_files" / "exported_masks" / exp_id / "masks"
    sam2_min, sam2_max, sam2_count = get_file_mtime_range(sam2_dir, ".png")

    # Check BF snips
    snips_dir = BASE_DIR / "training_data" / "bf_embryo_snips" / exp_id
    snips_min, snips_max, snips_count = get_file_mtime_range(snips_dir, ".jpg")

    result = {
        'exp_id': exp_id,
        'stitched_min': stitched_min,
        'stitched_max': stitched_max,
        'stitched_count': stitched_count,
        'sam2_min': sam2_min,
        'sam2_max': sam2_max,
        'sam2_count': sam2_count,
        'snips_min': snips_min,
        'snips_max': snips_max,
        'snips_count': snips_count,
        'issues': []
    }

    # Key insight: Check if OLDEST files are older than stitched
    # (not newest files - that can be recent updates)
    if sam2_count > 0 and sam2_min < stitched_min:
        days_behind = (stitched_min - sam2_min) / 86400
        result['issues'].append(f"SAM2 has files {days_behind:.1f} days older than stitched")

    if snips_count > 0 and snips_min < stitched_min:
        days_behind = (stitched_min - snips_min) / 86400
        result['issues'].append(f"Snips have files {days_behind:.1f} days older than stitched")

    # Also check for mixed-age problem: old files + new files in same dir
    if sam2_count > 0:
        sam2_age_spread = (sam2_max - sam2_min) / 86400
        if sam2_age_spread > 5:  # More than 5 days spread in SAM2 files
            result['issues'].append(f"SAM2 files span {sam2_age_spread:.1f} days (mixed old/new)")

    if snips_count > 0:
        snips_age_spread = (snips_max - snips_min) / 86400
        if snips_age_spread > 5:
            result['issues'].append(f"Snips files span {snips_age_spread:.1f} days (mixed old/new)")

    return result if result['issues'] else None

def format_date(timestamp):
    if timestamp == 0:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

# Find all YX1 experiments
yx1_raw_dir = BASE_DIR / "raw_image_data" / "YX1"
experiments = sorted([d.name for d in yx1_raw_dir.iterdir() if d.is_dir()])

print("=" * 90)
print("DETECTING MIXED STALENESS IN YX1 EXPERIMENTS")
print("=" * 90)
print()
print("Method: Check if ANY files are older than stitched images")
print("This detects: SAM2 masks or snips that have a mix of old and new files")
print()

stale_experiments = []

for exp_id in experiments:
    result = detect_staleness(exp_id)
    if result:
        stale_experiments.append(result)

if not stale_experiments:
    print("✅ No stale data detected across all YX1 experiments")
else:
    print(f"⚠️  Found {len(stale_experiments)} experiments with stale files:\n")

    for result in stale_experiments:
        exp = result['exp_id']
        print(f"📊 {exp}")
        print(f"   Stitched images: {result['stitched_count']} files")
        print(f"     Range: {format_date(result['stitched_min'])} to {format_date(result['stitched_max'])}")

        if result['sam2_count'] > 0:
            print(f"   SAM2 masks: {result['sam2_count']} files")
            print(f"     Range: {format_date(result['sam2_min'])} to {format_date(result['sam2_max'])}")

        if result['snips_count'] > 0:
            print(f"   BF snips: {result['snips_count']} files")
            print(f"     Range: {format_date(result['snips_min'])} to {format_date(result['snips_max'])}")

        print(f"   Issues:")
        for issue in result['issues']:
            print(f"     ⚠️  {issue}")
        print()

# Summary
if stale_experiments:
    print("=" * 90)
    print("REGENERATION NEEDED:")
    print("=" * 90)
    print()

    sam2_needed = [r['exp_id'] for r in stale_experiments if any('SAM2' in i for i in r['issues'])]
    snips_needed = [r['exp_id'] for r in stale_experiments if any('Snips' in i or 'snips' in i for i in r['issues'])]

    if sam2_needed:
        print(f"SAM2 regeneration needed for {len(sam2_needed)} experiments:")
        print(f"  {','.join(sam2_needed)}")
        print()

    if snips_needed:
        print(f"Build03 regeneration needed for {len(snips_needed)} experiments:")
        print(f"  {','.join(snips_needed)}")
        print()
