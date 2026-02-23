"""
Combine experiment parts into a single unified experiment.

This module handles combining multi-part experiments (e.g., 20251017_part1 + 20251017_part2)
into a single combined experiment with continuous time progression and unique IDs.
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np


def combine_experiment_parts(
    part1_experiment: str,
    part2_experiment: str,
    combined_experiment_name: str,
    data_root: str | Path,
    build_stages: List[str] = None
) -> Dict[str, any]:
    """
    Combine two experiment parts into a single unified experiment.

    This function combines multi-part experiments by:
    1. Offsetting part2's time_int and Time Rel (s) to continue from part1
    2. Regenerating unique snip_ids for part2 based on new time_int
    3. Renaming all IDs to use the combined experiment name
    4. Recalculating predicted_stage_hpf with continuous time
    5. Concatenating and saving combined datasets

    Args:
        part1_experiment: Name of first experiment part (e.g., "20251017_part1")
        part2_experiment: Name of second experiment part (e.g., "20251017_part2")
        combined_experiment_name: Name for combined experiment (e.g., "20251017_combined")
        data_root: Root directory containing morphseq_playground
        build_stages: List of build stages to combine (default: ['build04', 'build06'])

    Returns:
        Dict with summary information:
        {
            'build04': {'part1_rows': int, 'part2_rows': int, 'combined_rows': int,
                       'time_int_offset': int, 'time_rel_offset_s': float, 'output_file': str},
            'build06': {...}  # if build06 included
        }

    Example:
        >>> result = combine_experiment_parts(
        ...     "20251017_part1",
        ...     "20251017_part2",
        ...     "20251017_combined",
        ...     "/path/to/data"
        ... )
        >>> print(f"Combined {result['build04']['combined_rows']} rows")
    """
    if build_stages is None:
        build_stages = ['build04', 'build06', 'curvature']

    data_root = Path(data_root)
    metadata_dir = data_root / "morphseq_playground" / "metadata"

    results = {}

    for build_stage in build_stages:
        print(f"\n{'='*80}")
        print(f"Processing {build_stage.upper()}")
        print(f"{'='*80}")

        if build_stage == 'build04':
            result = _combine_build04(
                part1_experiment, part2_experiment, combined_experiment_name,
                metadata_dir
            )
        elif build_stage == 'build06':
            result = _combine_build06(
                part1_experiment, part2_experiment, combined_experiment_name,
                metadata_dir
            )
        elif build_stage == 'curvature':
            result = _combine_curvature(
                part1_experiment, part2_experiment, combined_experiment_name,
                metadata_dir
            )
        else:
            print(f"⚠️  Unknown build stage: {build_stage}, skipping...")
            continue

        results[build_stage] = result

    print(f"\n{'='*80}")
    print(f"✅ COMBINATION COMPLETE")
    print(f"{'='*80}")
    for stage, result in results.items():
        print(f"{stage.upper()}: {result['combined_rows']} total rows → {result['output_file']}")

    return results


def _combine_build04(
    part1_exp: str,
    part2_exp: str,
    combined_exp: str,
    metadata_dir: Path
) -> Dict[str, any]:
    """
    Combine build04 (qc_staged) files.

    Args:
        part1_exp: Part 1 experiment name
        part2_exp: Part 2 experiment name
        combined_exp: Combined experiment name
        metadata_dir: Path to metadata directory

    Returns:
        Dict with combination summary
    """
    build04_dir = metadata_dir / "build04_output"

    # Load part1 and part2
    part1_file = build04_dir / f"qc_staged_{part1_exp}.csv"
    part2_file = build04_dir / f"qc_staged_{part2_exp}.csv"

    print(f"📂 Loading {part1_file.name}...")
    df_part1 = pd.read_csv(part1_file)
    print(f"   → {len(df_part1)} rows loaded")

    print(f"📂 Loading {part2_file.name}...")
    df_part2 = pd.read_csv(part2_file)
    print(f"   → {len(df_part2)} rows loaded")

    # Calculate offsets from part1
    max_time_int_part1 = df_part1['time_int'].max()
    max_time_rel_part1 = df_part1['Time Rel (s)'].max()

    time_int_offset = max_time_int_part1 + 1
    time_rel_offset = max_time_rel_part1

    print(f"\n⚙️  Calculated offsets:")
    print(f"   • time_int offset: {time_int_offset} (max from part1: {max_time_int_part1})")
    print(f"   • Time Rel (s) offset: {time_rel_offset:.2f} s (max from part1: {max_time_rel_part1:.2f} s)")

    # Process part2: Apply offsets
    print(f"\n🔧 Processing part2 with offsets...")
    df_part2 = df_part2.copy()

    # Apply time offsets
    df_part2['time_int'] = df_part2['time_int'] + time_int_offset
    df_part2['Time Rel (s)'] = df_part2['Time Rel (s)'] + time_rel_offset

    # Regenerate snip_ids with new time_int
    print(f"   • Regenerating snip_ids with new time_int...")
    df_part2 = _regenerate_snip_ids(df_part2, combined_exp)

    # Regenerate image_ids with new time_int
    print(f"   • Regenerating image_ids with new time_int...")
    df_part2 = _regenerate_image_ids(df_part2, combined_exp)

    # Rename IDs in part2
    print(f"   • Renaming IDs to use '{combined_exp}'...")
    df_part2 = _rename_experiment_ids(df_part2, part2_exp, combined_exp)

    # Also rename IDs in part1
    print(f"\n🔧 Renaming part1 IDs to use '{combined_exp}'...")
    df_part1 = _rename_experiment_ids(df_part1, part1_exp, combined_exp)

    # Fix start_age_hpf for continuous experiments to prevent double-counting time
    if 'start_age_hpf' in df_part2.columns:
        print(f"   • Adjusting start_age_hpf for continuous time...")
        df_part2 = _adjust_start_age_for_continuity(df_part1, df_part2)

    # Recalculate predicted_stage_hpf for part2 (with continuous time)
    has_start_age = 'start_age_hpf' in df_part2.columns
    has_temp = 'temperature' in df_part2.columns or 'temperature_c' in df_part2.columns
    if has_start_age and has_temp:
        print(f"   • Recalculating predicted_stage_hpf with continuous time...")
        df_part2 = _recalculate_predicted_stage_hpf(df_part2)

    # Concatenate
    print(f"\n🔗 Concatenating part1 + part2...")
    df_combined = pd.concat([df_part1, df_part2], ignore_index=True)
    print(f"   → Combined: {len(df_combined)} rows")

    # Save
    output_file = build04_dir / f"qc_staged_{combined_exp}.csv"
    print(f"\n💾 Saving to {output_file.name}...")
    df_combined.to_csv(output_file, index=False)
    print(f"   ✅ Saved successfully")

    return {
        'part1_rows': len(df_part1),
        'part2_rows': len(df_part2),
        'combined_rows': len(df_combined),
        'time_int_offset': time_int_offset,
        'time_rel_offset_s': time_rel_offset,
        'output_file': str(output_file)
    }


def _combine_build06(
    part1_exp: str,
    part2_exp: str,
    combined_exp: str,
    metadata_dir: Path
) -> Dict[str, any]:
    """
    Combine build06 (final output with latents) files.

    Args:
        part1_exp: Part 1 experiment name
        part2_exp: Part 2 experiment name
        combined_exp: Combined experiment name
        metadata_dir: Path to metadata directory

    Returns:
        Dict with combination summary
    """
    build06_dir = metadata_dir / "build06_output"

    # Load part1 and part2
    part1_file = build06_dir / f"df03_final_output_with_latents_{part1_exp}.csv"
    part2_file = build06_dir / f"df03_final_output_with_latents_{part2_exp}.csv"

    print(f"📂 Loading {part1_file.name}...")
    df_part1 = pd.read_csv(part1_file)
    print(f"   → {len(df_part1)} rows loaded")

    print(f"📂 Loading {part2_file.name}...")
    df_part2 = pd.read_csv(part2_file)
    print(f"   → {len(df_part2)} rows loaded")

    # Calculate offsets from part1
    max_time_int_part1 = df_part1['time_int'].max()
    max_time_rel_part1 = df_part1['Time Rel (s)'].max()

    time_int_offset = max_time_int_part1 + 1
    time_rel_offset = max_time_rel_part1

    print(f"\n⚙️  Calculated offsets:")
    print(f"   • time_int offset: {time_int_offset} (max from part1: {max_time_int_part1})")
    print(f"   • Time Rel (s) offset: {time_rel_offset:.2f} s (max from part1: {max_time_rel_part1:.2f} s)")

    # Process part2: Apply offsets
    print(f"\n🔧 Processing part2 with offsets...")
    df_part2 = df_part2.copy()

    # Apply time offsets
    df_part2['time_int'] = df_part2['time_int'] + time_int_offset
    df_part2['Time Rel (s)'] = df_part2['Time Rel (s)'] + time_rel_offset

    # Regenerate snip_ids with new time_int
    print(f"   • Regenerating snip_ids with new time_int...")
    df_part2 = _regenerate_snip_ids(df_part2, combined_exp)

    # Regenerate image_ids with new time_int
    print(f"   • Regenerating image_ids with new time_int...")
    df_part2 = _regenerate_image_ids(df_part2, combined_exp)

    # Rename IDs in part2
    print(f"   • Renaming IDs to use '{combined_exp}'...")
    df_part2 = _rename_experiment_ids(df_part2, part2_exp, combined_exp)

    # Also rename IDs in part1
    print(f"\n🔧 Renaming part1 IDs to use '{combined_exp}'...")
    df_part1 = _rename_experiment_ids(df_part1, part1_exp, combined_exp)

    # Fix start_age_hpf for continuous experiments to prevent double-counting time
    if 'start_age_hpf' in df_part2.columns:
        print(f"   • Adjusting start_age_hpf for continuous time...")
        df_part2 = _adjust_start_age_for_continuity(df_part1, df_part2)

    # Recalculate predicted_stage_hpf for part2 (with continuous time)
    has_start_age = 'start_age_hpf' in df_part2.columns
    has_temp = 'temperature' in df_part2.columns or 'temperature_c' in df_part2.columns
    if has_start_age and has_temp:
        print(f"   • Recalculating predicted_stage_hpf with continuous time...")
        df_part2 = _recalculate_predicted_stage_hpf(df_part2)

    # Concatenate
    print(f"\n🔗 Concatenating part1 + part2...")
    df_combined = pd.concat([df_part1, df_part2], ignore_index=True)
    print(f"   → Combined: {len(df_combined)} rows")

    # Save
    output_file = build06_dir / f"df03_final_output_with_latents_{combined_exp}.csv"
    print(f"\n💾 Saving to {output_file.name}...")
    df_combined.to_csv(output_file, index=False)
    print(f"   ✅ Saved successfully")

    return {
        'part1_rows': len(df_part1),
        'part2_rows': len(df_part2),
        'combined_rows': len(df_combined),
        'time_int_offset': time_int_offset,
        'time_rel_offset_s': time_rel_offset,
        'output_file': str(output_file)
    }


def _combine_curvature(
    part1_exp: str,
    part2_exp: str,
    combined_exp: str,
    metadata_dir: Path
) -> Dict[str, any]:
    """
    Combine curvature metrics files from body_axis/summary directory.

    This function needs the combined build06 metadata to exist first, as it uses
    the snip_id mapping from there to properly update the curvature snip_ids with
    the correct time offsets.

    Args:
        part1_exp: Part 1 experiment name
        part2_exp: Part 2 experiment name
        combined_exp: Combined experiment name
        metadata_dir: Path to metadata directory

    Returns:
        Dict with combination summary
    """
    curvature_dir = metadata_dir / "body_axis" / "summary"
    build06_dir = metadata_dir / "build06_output"

    # Load part1 and part2 curvature files
    part1_file = curvature_dir / f"curvature_metrics_{part1_exp}.csv"
    part2_file = curvature_dir / f"curvature_metrics_{part2_exp}.csv"

    if not part1_file.exists():
        print(f"⚠️  Part1 curvature file not found: {part1_file}")
        print(f"   Skipping curvature combination")
        return {
            'part1_rows': 0,
            'part2_rows': 0,
            'combined_rows': 0,
            'output_file': 'N/A (skipped)'
        }

    if not part2_file.exists():
        print(f"⚠️  Part2 curvature file not found: {part2_file}")
        print(f"   Skipping curvature combination")
        return {
            'part1_rows': 0,
            'part2_rows': 0,
            'combined_rows': 0,
            'output_file': 'N/A (skipped)'
        }

    print(f"📂 Loading {part1_file.name}...")
    df_curv_part1 = pd.read_csv(part1_file)
    print(f"   → {len(df_curv_part1)} rows loaded")

    print(f"📂 Loading {part2_file.name}...")
    df_curv_part2 = pd.read_csv(part2_file)
    print(f"   → {len(df_curv_part2)} rows loaded")

    # Load the original part1 and part2 metadata, and the combined metadata
    # to create a mapping from old snip_ids to new snip_ids
    part1_metadata_file = build06_dir / f"df03_final_output_with_latents_{part1_exp}.csv"
    part2_metadata_file = build06_dir / f"df03_final_output_with_latents_{part2_exp}.csv"
    combined_metadata_file = build06_dir / f"df03_final_output_with_latents_{combined_exp}.csv"

    if not combined_metadata_file.exists():
        print(f"⚠️  Combined metadata not found: {combined_metadata_file}")
        print(f"   Cannot map snip_ids without metadata - run build04/build06 combination first")
        return {
            'part1_rows': 0,
            'part2_rows': 0,
            'combined_rows': 0,
            'output_file': 'N/A (skipped - no metadata)'
        }

    print(f"\n🔧 Building snip_id mapping from metadata...")

    # Create mapping dictionary: old_snip_id -> new_snip_id
    snip_id_mapping = {}

    # Load part1 metadata and create mapping
    if part1_metadata_file.exists():
        print(f"   Loading part1 metadata...")
        df_part1_meta = pd.read_csv(part1_metadata_file, usecols=['snip_id'])
        # For part1, new snip_id is just experiment name replacement
        for old_snip_id in df_part1_meta['snip_id']:
            new_snip_id = old_snip_id.replace(part1_exp, combined_exp)
            snip_id_mapping[old_snip_id] = new_snip_id
        print(f"   → Added {len(df_part1_meta)} part1 mappings")

    # Load part2 metadata and create mapping
    if part2_metadata_file.exists():
        print(f"   Loading part2 metadata...")
        df_part2_meta = pd.read_csv(part2_metadata_file, usecols=['snip_id'])
        # For part2, we need to load the combined metadata to get the new snip_ids
        # because they have time offsets applied
        df_combined_meta = pd.read_csv(combined_metadata_file, usecols=['snip_id'])

        # The combined file has part2 snip_ids with time offsets already applied
        # Extract just the part2 entries (those that were renamed from part2_exp)
        part2_snip_ids_in_combined = [sid for sid in df_combined_meta['snip_id']
                                       if combined_exp in sid and sid not in snip_id_mapping.values()]

        # Build lookup: well_embryo -> list of snip_ids sorted by time
        from collections import defaultdict
        combined_by_embryo = defaultdict(list)
        for new_snip_id in part2_snip_ids_in_combined:
            parts = new_snip_id.split('_')
            if len(parts) >= 5:
                well = parts[-3]
                embryo = parts[-2]
                time_str = parts[-1]  # e.g., "t0095"
                well_embryo = f"{well}_{embryo}"
                combined_by_embryo[well_embryo].append((new_snip_id, time_str))

        # Sort each embryo's snip_ids by time
        for well_embryo in combined_by_embryo:
            combined_by_embryo[well_embryo].sort(key=lambda x: x[1])

        # Build lookup for part2_meta: well_embryo -> list of old snip_ids sorted by time
        part2_by_embryo = defaultdict(list)
        for old_snip_id in df_part2_meta['snip_id']:
            parts = old_snip_id.split('_')
            if len(parts) >= 5:
                well = parts[-3]
                embryo = parts[-2]
                time_str = parts[-1]  # e.g., "t0034"
                well_embryo = f"{well}_{embryo}"
                part2_by_embryo[well_embryo].append((old_snip_id, time_str))

        # Sort each embryo's snip_ids by time
        for well_embryo in part2_by_embryo:
            part2_by_embryo[well_embryo].sort(key=lambda x: x[1])

        # Map part2 old snip_ids to new ones
        # Match based on well_embryo AND position in sorted time sequence
        for well_embryo in part2_by_embryo:
            if well_embryo in combined_by_embryo:
                old_list = part2_by_embryo[well_embryo]
                new_list = combined_by_embryo[well_embryo]

                # Map by position in sorted sequence
                for i, (old_snip_id, _) in enumerate(old_list):
                    if i < len(new_list):
                        new_snip_id = new_list[i][0]
                        snip_id_mapping[old_snip_id] = new_snip_id

        print(f"   → Added {len([k for k in snip_id_mapping if part2_exp in k])} part2 mappings")

    print(f"   Total mappings created: {len(snip_id_mapping)}")

    # Apply mapping to curvature data
    print(f"\n🔧 Applying snip_id mapping to curvature data...")

    df_curv_part1['snip_id_new'] = df_curv_part1['snip_id'].map(snip_id_mapping)
    df_curv_part2['snip_id_new'] = df_curv_part2['snip_id'].map(snip_id_mapping)

    # Filter to rows that were successfully mapped
    df_curv_part1_matched = df_curv_part1[df_curv_part1['snip_id_new'].notna()].copy()
    df_curv_part2_matched = df_curv_part2[df_curv_part2['snip_id_new'].notna()].copy()

    n_part1_unmatched = len(df_curv_part1) - len(df_curv_part1_matched)
    n_part2_unmatched = len(df_curv_part2) - len(df_curv_part2_matched)

    if n_part1_unmatched > 0:
        print(f"   ⚠️  Part1: {n_part1_unmatched} curvature rows not mapped (likely QC filtered)")
    if n_part2_unmatched > 0:
        print(f"   ⚠️  Part2: {n_part2_unmatched} curvature rows not mapped (likely QC filtered)")

    # Replace snip_id with the new mapped values
    df_curv_part1_matched['snip_id'] = df_curv_part1_matched['snip_id_new']
    df_curv_part2_matched['snip_id'] = df_curv_part2_matched['snip_id_new']

    df_curv_part1_matched = df_curv_part1_matched.drop(columns=['snip_id_new'])
    df_curv_part2_matched = df_curv_part2_matched.drop(columns=['snip_id_new'])

    # Concatenate
    print(f"\n🔗 Concatenating part1 + part2...")
    df_combined = pd.concat([df_curv_part1_matched, df_curv_part2_matched], ignore_index=True)
    print(f"   → Combined: {len(df_combined)} rows")

    # Save with standard naming (no "summary" prefix)
    output_file = curvature_dir / f"curvature_metrics_{combined_exp}.csv"
    print(f"\n💾 Saving to {output_file.name}...")
    df_combined.to_csv(output_file, index=False)
    print(f"   ✅ Saved successfully")

    return {
        'part1_rows': len(df_curv_part1_matched),
        'part2_rows': len(df_curv_part2_matched),
        'combined_rows': len(df_combined),
        'output_file': str(output_file)
    }


def _regenerate_snip_ids(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Regenerate snip_ids based on current time_int values.

    Snip ID format: {experiment}_{well}_{embryo}_t{time_int:04d}

    Args:
        df: DataFrame with snip_id, time_int columns
        experiment_name: New experiment name to use

    Returns:
        DataFrame with regenerated snip_ids
    """
    df = df.copy()

    # Extract well and embryo from existing snip_id pattern
    # Example: "20251017_part2_A01_e01_t0002" → well="A01", embryo="e01"
    snip_id_parts = df['snip_id'].str.extract(r'_([A-H]\d{2})_(e\d+)_t\d+')
    df['_well'] = snip_id_parts[0]
    df['_embryo'] = snip_id_parts[1]

    # Regenerate snip_id with new experiment name and updated time_int
    df['snip_id'] = (
        experiment_name + '_' +
        df['_well'] + '_' +
        df['_embryo'] + '_t' +
        df['time_int'].astype(int).apply(lambda x: f'{x:04d}')
    )

    # Clean up temporary columns
    df = df.drop(columns=['_well', '_embryo'])

    return df


def _regenerate_image_ids(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Regenerate image_ids based on current time_int values.

    Image ID format: {experiment}_{well}_ch{channel}_t{time_int:04d}

    Args:
        df: DataFrame with image_id, time_int columns
        experiment_name: New experiment name to use

    Returns:
        DataFrame with regenerated image_ids
    """
    df = df.copy()

    # Extract well and channel from existing image_id pattern
    # Example: "20251017_part2_A01_ch00_t0002" → well="A01", channel="00"
    image_id_parts = df['image_id'].str.extract(r'_([A-H]\d{2})_ch(\d+)_t\d+')
    df['_well'] = image_id_parts[0]
    df['_channel'] = image_id_parts[1]

    # Regenerate image_id with new experiment name and updated time_int
    df['image_id'] = (
        experiment_name + '_' +
        df['_well'] + '_ch' +
        df['_channel'] + '_t' +
        df['time_int'].astype(int).apply(lambda x: f'{x:04d}')
    )

    # Clean up temporary columns
    df = df.drop(columns=['_well', '_channel'])

    return df


def _rename_experiment_ids(df: pd.DataFrame, old_exp: str, new_exp: str) -> pd.DataFrame:
    """
    Rename all experiment-related ID columns.

    Renames:
    - image_id
    - embryo_id
    - snip_id (only the experiment prefix)
    - experiment_id
    - video_id

    Args:
        df: DataFrame with ID columns
        old_exp: Old experiment name to replace
        new_exp: New experiment name

    Returns:
        DataFrame with renamed IDs
    """
    df = df.copy()

    # List of columns to rename
    id_columns = ['image_id', 'embryo_id', 'snip_id', 'video_id']

    for col in id_columns:
        if col in df.columns:
            df[col] = df[col].str.replace(old_exp, new_exp, regex=False)

    # Rename experiment_id directly
    if 'experiment_id' in df.columns:
        df['experiment_id'] = new_exp

    return df


def _adjust_start_age_for_continuity(df_part1: pd.DataFrame, df_part2: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust start_age_hpf in part2 to maintain temporal continuity with part1.

    For continuous experiments split into parts, part2's start_age_hpf should match
    part1's original start_age_hpf (not the biological age when part2 started).
    This prevents double-counting time when Time Rel (s) is offset.

    Algorithm:
    1. Find unique start_age pairs between part1 and part2 that represent the same embryos
    2. Map part2's start_age back to part1's original start_age
    3. For example: part1 (start=60) → part2 (start=97) should become part2 (start=60)

    Args:
        df_part1: Part1 DataFrame with original start_age_hpf
        df_part2: Part2 DataFrame with start_age_hpf to adjust

    Returns:
        DataFrame with adjusted start_age_hpf values
    """
    df_part2 = df_part2.copy()

    # Get unique start ages from each part
    part1_ages = sorted(df_part1['start_age_hpf'].unique())
    part2_ages = sorted(df_part2['start_age_hpf'].unique())

    print(f"      Part1 start_ages: {part1_ages}")
    print(f"      Part2 start_ages: {part2_ages}")

    # Create mapping: part2_age -> part1_age
    # Assumption: Same number of unique ages, sorted order matches
    if len(part1_ages) == len(part2_ages):
        age_mapping = dict(zip(part2_ages, part1_ages))
        print(f"      Mapping: {age_mapping}")

        # Apply mapping
        df_part2['start_age_hpf'] = df_part2['start_age_hpf'].map(age_mapping)
        print(f"      ✓ Adjusted start_age_hpf for continuity")
    else:
        print(f"      ⚠️  WARNING: Mismatched number of start ages ({len(part1_ages)} vs {len(part2_ages)})")
        print(f"      ⚠️  Skipping start_age adjustment - manual review needed")

    return df_part2


def _recalculate_predicted_stage_hpf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate predicted_stage_hpf using Kimmel formula with current Time Rel (s).

    Formula: predicted_stage_hpf = start_age_hpf + (Time Rel (s)/3600) * (0.055*temp - 0.57)

    Args:
        df: DataFrame with start_age_hpf, Time Rel (s), temperature/temperature_c columns

    Returns:
        DataFrame with recalculated predicted_stage_hpf
    """
    df = df.copy()

    try:
        # Determine which temperature column to use
        temp_col = 'temperature' if 'temperature' in df.columns else 'temperature_c'

        if temp_col not in df.columns:
            print(f"      ⚠️  No temperature column found, skipping recalculation")
            return df

        # Convert start_age_hpf to numeric (handles string values gracefully)
        start_age_numeric = pd.to_numeric(df['start_age_hpf'], errors='coerce')

        # Calculate for rows with numeric start_age
        mask = start_age_numeric.notna()

        if mask.any():
            df.loc[mask, 'predicted_stage_hpf'] = (
                start_age_numeric[mask].astype(float) +
                (df.loc[mask, 'Time Rel (s)'].astype(float) / 3600.0) *
                (0.055 * df.loc[mask, temp_col].astype(float) - 0.57)
            )

            n_updated = mask.sum()
            print(f"      → Updated predicted_stage_hpf for {n_updated} rows (using {temp_col})")
        else:
            print(f"      ⚠️  No numeric start_age_hpf values found, skipping recalculation")

    except Exception as e:
        print(f"      ⚠️  Error recalculating predicted_stage_hpf: {e}")

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python combine_experiments_parts.py <part1_exp> <part2_exp> <combined_exp> [data_root]")
        print("\nExample:")
        print("  python combine_experiments_parts.py 20251017_part1 20251017_part2 20251017_combined")
        sys.exit(1)

    part1 = sys.argv[1]
    part2 = sys.argv[2]
    combined = sys.argv[3]
    data_root = sys.argv[4] if len(sys.argv) > 4 else "/net/trapnell/vol1/home/mdcolon/proj/morphseq/"

    print(f"Combining experiments:")
    print(f"  Part 1: {part1}")
    print(f"  Part 2: {part2}")
    print(f"  Combined: {combined}")
    print(f"  Data root: {data_root}")
    print()

    results = combine_experiment_parts(part1, part2, combined, data_root)

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    for stage, result in results.items():
        print(f"\n{stage.upper()}:")
        print(f"  Part 1: {result['part1_rows']:,} rows")
        print(f"  Part 2: {result['part2_rows']:,} rows")
        print(f"  Combined: {result['combined_rows']:,} rows")
        if 'time_int_offset' in result:
            print(f"  Time offsets: time_int +{result['time_int_offset']}, Time Rel +{result['time_rel_offset_s']:.2f}s")
        print(f"  Output: {result['output_file']}")
