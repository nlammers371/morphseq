#!/usr/bin/env python3
"""
Check for stale data across YX1 experiments.

Identifies experiments where:
1. SAM2 masks are older than stitched FF images
2. BF embryo snips are older than stitched FF images
3. Curvature metrics exist but are stale
4. Build04 output exists but is stale

Usage:
  python check_stale_data.py
  python check_stale_data.py --data-root /path/to/morphseq_playground
"""

from pathlib import Path
from datetime import datetime
import argparse
import sys
from typing import Dict, List, Tuple


def get_most_recent_file_time(directory: Path, pattern: str = "*") -> float:
    """Get the most recent file modification time in a directory."""
    if not directory.exists():
        return 0.0

    files = list(directory.glob(pattern))
    if not files:
        return 0.0

    return max(f.stat().st_mtime for f in files if f.is_file())


def check_experiment_staleness(data_root: Path, exp_id: str) -> Dict:
    """Check staleness of data files for a single experiment."""
    result = {
        'exp_id': exp_id,
        'stitched_ff_time': 0.0,
        'sam2_masks_time': 0.0,
        'bf_snips_time': 0.0,
        'curvature_time': 0.0,
        'build04_time': 0.0,
        'issues': [],
        'has_stitched': False,
        'has_sam2_masks': False,
        'has_snips': False,
        'has_curvature': False,
        'has_build04': False,
    }

    # Check stitched FF images (source data)
    stitched_dir = data_root / "built_image_data" / "stitched_FF_images" / exp_id
    if stitched_dir.exists():
        result['stitched_ff_time'] = get_most_recent_file_time(stitched_dir, "*.jpg")
        result['has_stitched'] = result['stitched_ff_time'] > 0

    # Check SAM2 masks
    sam2_masks_dir = data_root / "sam2_pipeline_files" / "exported_masks" / exp_id / "masks"
    if sam2_masks_dir.exists():
        result['sam2_masks_time'] = get_most_recent_file_time(sam2_masks_dir, "*.png")
        result['has_sam2_masks'] = result['sam2_masks_time'] > 0

    # Check BF embryo snips
    snips_dir = data_root / "training_data" / "bf_embryo_snips" / exp_id
    if snips_dir.exists():
        result['bf_snips_time'] = get_most_recent_file_time(snips_dir, "*.jpg")
        result['has_snips'] = result['bf_snips_time'] > 0

    # Check curvature metrics
    curvature_file = data_root / "metadata" / "body_axis" / "summary" / f"curvature_metrics_{exp_id}.csv"
    if curvature_file.exists():
        result['curvature_time'] = curvature_file.stat().st_mtime
        result['has_curvature'] = True

    # Check Build04 output
    build04_file = data_root / "metadata" / "build04_output" / f"qc_staged_{exp_id}.csv"
    if build04_file.exists():
        result['build04_time'] = build04_file.stat().st_mtime
        result['has_build04'] = True

    # Detect staleness issues
    if result['has_stitched']:
        # SAM2 masks older than stitched images
        if result['has_sam2_masks'] and result['sam2_masks_time'] < result['stitched_ff_time']:
            days_diff = (result['stitched_ff_time'] - result['sam2_masks_time']) / 86400
            result['issues'].append(f"SAM2 masks STALE ({days_diff:.1f} days behind stitched images)")

        # BF snips older than stitched images
        if result['has_snips'] and result['bf_snips_time'] < result['stitched_ff_time']:
            days_diff = (result['stitched_ff_time'] - result['bf_snips_time']) / 86400
            result['issues'].append(f"BF snips STALE ({days_diff:.1f} days behind stitched images)")

    # SAM2 masks older than curvature
    if result['has_sam2_masks'] and result['has_curvature'] and result['sam2_masks_time'] < result['curvature_time']:
        result['issues'].append(f"Curvature computed from stale SAM2 masks")

    # Build04 output older than SAM2 masks (suggests masks were regenerated after)
    if result['has_sam2_masks'] and result['has_build04'] and result['sam2_masks_time'] > result['build04_time']:
        days_diff = (result['sam2_masks_time'] - result['build04_time']) / 86400
        result['issues'].append(f"Build04 STALE (SAM2 masks newer by {days_diff:.1f} days)")

    return result


def format_timestamp(timestamp: float) -> str:
    """Format Unix timestamp to readable date."""
    if timestamp == 0:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def main():
    parser = argparse.ArgumentParser(description="Check for stale data across YX1 experiments")
    parser.add_argument("--data-root", default="/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground",
                       help="Path to morphseq_playground directory")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"❌ Data root not found: {data_root}")
        sys.exit(1)

    # Find all YX1 experiments (look at raw_image_data/YX1)
    yx1_raw_dir = data_root / "raw_image_data" / "YX1"
    if not yx1_raw_dir.exists():
        print(f"❌ YX1 raw data directory not found: {yx1_raw_dir}")
        sys.exit(1)

    experiments = sorted([d.name for d in yx1_raw_dir.iterdir() if d.is_dir()])
    print(f"Found {len(experiments)} YX1 experiments")
    print()

    # Check each experiment
    results = []
    for exp_id in experiments:
        result = check_experiment_staleness(data_root, exp_id)
        results.append(result)

    # Filter to experiments with issues
    stale_experiments = [r for r in results if r['issues']]

    print("=" * 100)
    print(f"STALE DATA SUMMARY: {len(stale_experiments)}/{len(experiments)} experiments need attention")
    print("=" * 100)
    print()

    if not stale_experiments:
        print("✅ No stale data detected!")
        return 0

    # Print experiments needing regeneration
    print("EXPERIMENTS REQUIRING DATA REGENERATION:")
    print()

    for result in stale_experiments:
        print(f"📊 {result['exp_id']}")
        print(f"   Stitched FF:  {format_timestamp(result['stitched_ff_time'])}")
        print(f"   SAM2 masks:   {format_timestamp(result['sam2_masks_time'])}")
        print(f"   BF snips:     {format_timestamp(result['bf_snips_time'])}")
        print(f"   Curvature:    {format_timestamp(result['curvature_time'])}")
        print(f"   Build04:      {format_timestamp(result['build04_time'])}")
        print(f"   Issues:")
        for issue in result['issues']:
            print(f"     ⚠️  {issue}")
        print()

    # Generate summary table
    print()
    print("=" * 100)
    print("REGENERATION COMMANDS:")
    print("=" * 100)
    print()

    # Group by which pipeline stage needs to run
    sam2_needed = [r['exp_id'] for r in stale_experiments if 'SAM2 masks STALE' in str(r['issues']) or 'SAM2' in str(r['issues'])]
    build03_needed = [r['exp_id'] for r in stale_experiments if 'BF snips STALE' in str(r['issues'])]
    build04_needed = [r['exp_id'] for r in stale_experiments if 'Build04 STALE' in str(r['issues']) or 'Curvature computed' in str(r['issues'])]

    if sam2_needed:
        print(f"✅ Re-run SAM2 for {len(sam2_needed)} experiments:")
        print(f"   python -m src.run_morphseq_pipeline.cli pipeline \\")
        print(f"     --data-root {data_root} \\")
        print(f"     --experiments {','.join(sam2_needed)} \\")
        print(f"     --action sam2 --force")
        print()

    if build03_needed:
        print(f"✅ Re-run Build03 for {len(build03_needed)} experiments (regenerate snips):")
        print(f"   python -m src.run_morphseq_pipeline.cli pipeline \\")
        print(f"     --data-root {data_root} \\")
        print(f"     --experiments {','.join(build03_needed)} \\")
        print(f"     --action build03 --force")
        print()

    if build04_needed:
        print(f"✅ Re-run Build04 for {len(build04_needed)} experiments (regenerate curvature):")
        print(f"   python -m src.run_morphseq_pipeline.cli pipeline \\")
        print(f"     --data-root {data_root} \\")
        print(f"     --experiments {','.join(build04_needed)} \\")
        print(f"     --action build04 --force")
        print()

    # Show comprehensive list for easy copy-paste
    all_affected = set(sam2_needed + build03_needed + build04_needed)
    print("=" * 100)
    print("AFFECTED EXPERIMENTS (for quick reference):")
    print("=" * 100)
    print(f"Comma-separated list: {','.join(sorted(all_affected))}")
    print()

    # Count summary
    print("=" * 100)
    print("SUMMARY BY ISSUE:")
    print("=" * 100)
    issue_counts = {}
    for result in stale_experiments:
        for issue in result['issues']:
            issue_type = issue.split('(')[0].strip()
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:3d} experiments: {issue}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
