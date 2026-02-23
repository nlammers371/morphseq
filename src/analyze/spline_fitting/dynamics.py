"""Trajectory dynamics: analyze how systems change over time along splines.

This module provides tools for:
- Simulated random walks (journeys) through developmental segments
- Computing developmental shifts between phenotypes
- Analyzing progression rates along trajectories

Example:
    >>> from src.analyze.spline_fitting import run_bootstrap_journeys, compute_developmental_shifts
    >>>
    >>> # Simulate random walks through segments
    >>> journeys = run_bootstrap_journeys(df_augmented, num_journeys=1000)
    >>>
    >>> # Compute developmental timing shifts
    >>> shifts = compute_developmental_shifts(df_embryos, wt_summary)
"""

import random
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Helper Functions
# =============================================================================

def annotate_embryo_time_index(df_augmented, time_col="experiment_time", embryo_id_col="embryo_id"):
    """Add within-embryo time index to DataFrame.

    Parameters
    ----------
    df_augmented : pd.DataFrame
        Data with embryo IDs and time column.
    time_col : str, default='experiment_time'
        Column with time values.
    embryo_id_col : str, default='embryo_id'
        Column with embryo identifiers.

    Returns
    -------
    df : pd.DataFrame
        Input DataFrame with added 'within_embryo_t_idx' column.
    """
    df = df_augmented.copy()
    df = df.sort_values([embryo_id_col, time_col])
    df['within_embryo_t_idx'] = df.groupby(embryo_id_col).cumcount()
    return df


def preprocess_embryo_data(df_augmented, embryo_id_col="embryo_id", time_col="experiment_time"):
    """Create dictionary of per-embryo DataFrames sorted by time.

    Parameters
    ----------
    df_augmented : pd.DataFrame
        Data with embryo IDs and time.
    embryo_id_col : str, default='embryo_id'
        Column with embryo identifiers.
    time_col : str, default='experiment_time'
        Column with time values.

    Returns
    -------
    embryo_dict : dict
        Mapping from embryo_id to sorted DataFrame.
    """
    embryo_dict = {}
    for emb_id in df_augmented[embryo_id_col].unique():
        emb_df = df_augmented[df_augmented[embryo_id_col] == emb_id].copy()
        emb_df = emb_df.sort_values(time_col).reset_index(drop=True)
        embryo_dict[emb_id] = emb_df
    return embryo_dict


# =============================================================================
# Journey Simulation
# =============================================================================

def single_random_journey(
    embryo_dict,
    df_augmented,
    segments_sorted=None,
    start_segment=0,
    end_segment=None,
    max_hops=10_000,
    time_column="experiment_time",
    segment_id_col="segment_id",
):
    """Simulate a single random walk through developmental segments.

    Generates a trajectory from start_segment to end_segment by:
    1. Starting at a random point in start_segment
    2. Following temporal progression within an embryo
    3. Jumping to new embryo when trajectory doesn't progress
    4. Tracking cumulative time

    Parameters
    ----------
    embryo_dict : dict
        Mapping from embryo_id to time-sorted DataFrame.
    df_augmented : pd.DataFrame
        Data with segment assignments and metadata.
    segments_sorted : list, optional
        Sorted segment IDs. If None, auto-detected.
    start_segment : int, default=0
        Starting segment ID.
    end_segment : int, optional
        Ending segment ID. If None, uses max segment.
    max_hops : int, default=10000
        Maximum steps to prevent infinite loops.
    time_column : str, default='experiment_time'
        Column with time values.
    segment_id_col : str, default='segment_id'
        Column with segment assignments.

    Returns
    -------
    journey : list of dict
        Journey steps with keys: [segment_id, embryo_id, snip_id, cumulative_time].
    """
    if segments_sorted is None:
        segments_sorted = sorted(df_augmented[segment_id_col].unique())
    if end_segment is None:
        end_segment = max(segments_sorted)

    total_time = 0.0
    journey = []
    current_segment = start_segment

    # Random starting point
    start_candidates = df_augmented[df_augmented[segment_id_col] == current_segment]
    if start_candidates.empty:
        return journey

    row_start = start_candidates.sample(n=1).iloc[0]
    curr_emb_id = row_start["embryo_id"]
    curr_snip_id = row_start["snip_id"]

    journey.append({
        "segment_id": current_segment,
        "embryo_id": curr_emb_id,
        "snip_id": curr_snip_id,
        "cumulative_time": total_time
    })

    # Get embryo DataFrame
    emb_df = embryo_dict[curr_emb_id]
    try:
        row_index = emb_df.index[emb_df["snip_id"] == curr_snip_id][0]
    except IndexError:
        return journey

    hop_count = 0

    while current_segment < end_segment and hop_count < max_hops:
        hop_count += 1
        next_index = row_index + 1
        possible_move = False

        if next_index < len(emb_df):
            # Next time point in same embryo
            row_next = emb_df.iloc[next_index]
            next_seg = row_next[segment_id_col]
            delta_t = row_next[time_column] - emb_df.iloc[row_index][time_column]

            if next_seg == current_segment:
                # Stay in segment
                total_time += delta_t
                journey.append({
                    "segment_id": current_segment,
                    "embryo_id": curr_emb_id,
                    "snip_id": row_next["snip_id"],
                    "cumulative_time": total_time
                })
                row_index = next_index
                possible_move = True
            elif next_seg > current_segment:
                # Advance to higher segment
                total_time += delta_t
                current_segment = next_seg
                journey.append({
                    "segment_id": current_segment,
                    "embryo_id": curr_emb_id,
                    "snip_id": row_next["snip_id"],
                    "cumulative_time": total_time
                })
                row_index = next_index
                possible_move = True

        if not possible_move:
            # Jump to different embryo
            possible_segments = [seg for seg in segments_sorted if seg >= current_segment]
            if not possible_segments:
                break
            next_segment = min(possible_segments)

            next_seg_candidates = df_augmented[df_augmented[segment_id_col] == next_segment]
            if next_seg_candidates.empty:
                break

            row_new = next_seg_candidates.sample(n=1).iloc[0]
            new_emb_id = row_new["embryo_id"]
            new_snip_id = row_new["snip_id"]

            current_segment = next_segment
            curr_emb_id = new_emb_id
            curr_snip_id = new_snip_id
            emb_df = embryo_dict[curr_emb_id]

            try:
                row_index = emb_df.index[emb_df["snip_id"] == new_snip_id][0]
            except IndexError:
                break

            journey.append({
                "segment_id": current_segment,
                "embryo_id": curr_emb_id,
                "snip_id": curr_snip_id,
                "cumulative_time": total_time
            })

    return journey


def run_bootstrap_journeys(
    df_augmented,
    num_journeys=1000,
    start_segment=0,
    end_segment=None,
    random_seed=42,
    time_column="experiment_time",
    segment_id_col="segment_id"
):
    """Run multiple random journey simulations (bootstrap).

    Simulates many random walks through developmental segments to estimate
    typical progression times and variability.

    Parameters
    ----------
    df_augmented : pd.DataFrame
        Data with segment assignments and time info.
    num_journeys : int, default=1000
        Number of random walks to simulate.
    start_segment : int, default=0
        Starting segment ID.
    end_segment : int, optional
        Ending segment ID. If None, uses max segment.
    random_seed : int, default=42
        Random seed for reproducibility.
    time_column : str, default='experiment_time'
        Column with time values.
    segment_id_col : str, default='segment_id'
        Column with segment assignments.

    Returns
    -------
    journeys_df : pd.DataFrame
        All journey steps with columns:
        [journey_id, step_index, segment_id, embryo_id, snip_id, cumulative_time].
    """
    # Set random seeds
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Annotate time index
    df_aug = annotate_embryo_time_index(df_augmented, time_col=time_column)

    # Find valid start segment
    start_segment_init = start_segment
    while True:
        start_candidates = df_aug[df_aug[segment_id_col] == start_segment]

        if not start_candidates.empty:
            if start_segment != start_segment_init:
                print(f"No start candidates in segment {start_segment_init}. "
                      f"Using segment {start_segment} instead.")
            break

        start_segment += 1

    # Build embryo dictionary
    embryo_dict = preprocess_embryo_data(df_aug, time_col=time_column)

    # Determine segments
    segments_sorted = sorted(df_aug[segment_id_col].unique())
    if end_segment is None:
        end_segment = max(segments_sorted)

    all_records = []

    for j_id in tqdm(range(num_journeys), desc="Running bootstrap journeys"):
        journey_steps = single_random_journey(
            embryo_dict=embryo_dict,
            df_augmented=df_aug,
            segments_sorted=segments_sorted,
            start_segment=start_segment,
            end_segment=end_segment,
            time_column=time_column,
            segment_id_col=segment_id_col
        )

        for step_i, step_info in enumerate(journey_steps):
            record = {
                "journey_id": j_id,
                "step_index": step_i,
                "segment_id": step_info["segment_id"],
                "embryo_id": step_info["embryo_id"],
                "snip_id": step_info["snip_id"],
                "cumulative_time": step_info["cumulative_time"]
            }
            all_records.append(record)

    journeys_df = pd.DataFrame(all_records)
    return journeys_df


def summarize_journeys(journeys_df):
    """Compute statistics of journey times per segment.

    Parameters
    ----------
    journeys_df : pd.DataFrame
        Output from run_bootstrap_journeys().

    Returns
    -------
    summary_df : pd.DataFrame
        Summary with columns: [segment_id, mean_time, std_time, count].
    """
    grp = journeys_df.groupby("segment_id")["cumulative_time"]
    summary = grp.agg(['mean', 'std', 'count']).reset_index()
    summary.columns = ["segment_id", "mean_time", "std_time", "count"]
    return summary


# =============================================================================
# Developmental Shifts
# =============================================================================

def compute_developmental_shifts(
    df_embryos,
    summary_df_ref,
    color_by="phenotype",
    time_col="experiment_time",
    segment_col="ref_seg_id",
    embryo_id_col="embryo_id"
):
    """Compute developmental timing shifts relative to reference.

    For each embryo, compares its progression rate through segments to
    the reference (typically wild-type) progression rate.

    Parameters
    ----------
    df_embryos : pd.DataFrame
        Embryo data with columns: [embryo_id, time_col, segment_col, color_by].
    summary_df_ref : pd.DataFrame
        Reference journey summary with columns: [segment_id, mean_time_hours].
        Typically from wild-type journeys.
    color_by : str, default='phenotype'
        Column for grouping/coloring (e.g., genotype, condition).
    time_col : str, default='experiment_time'
        Column with time values (in seconds).
    segment_col : str, default='ref_seg_id'
        Column with segment assignments.
    embryo_id_col : str, default='embryo_id'
        Column with embryo identifiers.

    Returns
    -------
    shift_df : pd.DataFrame
        One row per embryo with columns:
        [embryo_id, earliest_segment, latest_segment, embryo_time_hrs,
         ref_time_hrs, time_ratio, time_shift_per_24hrs, {color_by}].

    Notes
    -----
    time_shift_per_24hrs interpretation:
    - Positive: embryo develops faster than reference (ahead)
    - Negative: embryo develops slower than reference (delayed)
    - Magnitude: hours gained/lost per 24-hour period
    """
    results = []

    grouped = df_embryos.groupby(embryo_id_col)
    for emb_id, group in grouped:
        group_sorted = group.sort_values(time_col)
        if len(group_sorted) < 2:
            continue

        first_row = group_sorted.iloc[0]
        last_row = group_sorted.iloc[-1]

        # Convert to hours
        time_early_hrs = first_row[time_col] / 3600.0
        time_late_hrs = last_row[time_col] / 3600.0
        delta_time_hrs = time_late_hrs - time_early_hrs

        seg_early = first_row[segment_col]
        seg_late = last_row[segment_col]

        if pd.isnull(seg_early) or pd.isnull(seg_late) or seg_early == seg_late:
            continue

        # Reference times
        ref_row_early = summary_df_ref.loc[summary_df_ref["segment_id"] == seg_early]
        ref_row_late = summary_df_ref.loc[summary_df_ref["segment_id"] == seg_late]

        if len(ref_row_early) == 0 or len(ref_row_late) == 0:
            continue

        ref_time_early = ref_row_early["mean_time_hours"].iloc[0]
        ref_time_late = ref_row_late["mean_time_hours"].iloc[0]
        ref_delta_time = ref_time_late - ref_time_early

        # Time ratio
        time_ratio = delta_time_hrs / ref_delta_time

        # Time shift: (1 - ratio) * 24
        # Positive = faster than ref, Negative = slower than ref
        time_shift_per_24hrs = (1 - time_ratio) * 24

        col_value = first_row.get(color_by, np.nan)

        results.append({
            embryo_id_col: emb_id,
            "earliest_segment": seg_early,
            "latest_segment": seg_late,
            "embryo_time_hrs": delta_time_hrs,
            "ref_time_hrs": ref_delta_time,
            "time_ratio": time_ratio,
            "time_shift_per_24hrs": time_shift_per_24hrs,
            color_by: col_value
        })

    return pd.DataFrame(results)
