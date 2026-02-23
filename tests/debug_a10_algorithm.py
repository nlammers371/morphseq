#!/usr/bin/env python3
"""
Debug the persistence algorithm step-by-step for embryo A10
to understand why it's not being detected despite low fraction_alive.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

# Load test data
data_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output/qc_staged_20250711.csv")
df = pd.read_csv(data_path)

# Find A10 embryo
a10_embryos = [eid for eid in df['embryo_id'].unique() if 'A10' in eid]
print(f"Found A10 embryos: {a10_embryos}")

if len(a10_embryos) == 0:
    print("No A10 embryo found!")
    exit()

target_embryo = a10_embryos[0]  # Use first A10 found
print(f"\nDebugging algorithm for: {target_embryo}")

# Get embryo data
embryo_data = df[df['embryo_id'] == target_embryo].sort_values('time_int').copy()

print(f"\nEMBRYO DATA OVERVIEW:")
print(f"Total timepoints: {len(embryo_data)}")
print(f"Time range: {embryo_data['time_int'].min():.1f} to {embryo_data['time_int'].max():.1f}")
print(f"Fraction alive range: {embryo_data['fraction_alive'].min():.3f} to {embryo_data['fraction_alive'].max():.3f}")
print(f"Dead_flag True count: {(embryo_data['dead_flag'] == True).sum()}")
print(f"Dead_flag False count: {(embryo_data['dead_flag'] == False).sum()}")

def debug_validate_death_persistence(embryo_data, inflection_time, threshold=0.5):
    """
    Debug version of persistence validation with detailed prints
    """
    print(f"\n  🔍 VALIDATING inflection at t={inflection_time:.1f}")

    time_col = 'time_int'

    # Get timepoints after inflection
    post_inflection = embryo_data[embryo_data[time_col] > inflection_time]

    print(f"    Post-inflection timepoints: {len(post_inflection)}")

    if len(post_inflection) == 0:
        print(f"    ❌ NO timepoints after inflection!")
        return False, {'post_count': 0, 'dead_count': 0, 'dead_fraction': 0.0}

    # Show the post-inflection data
    print(f"    Post-inflection data:")
    for _, row in post_inflection.head(10).iterrows():  # Show first 10
        print(f"      t={row['time_int']:6.1f}, fraction={row['fraction_alive']:5.3f}, dead_flag={row['dead_flag']}")

    if len(post_inflection) > 10:
        print(f"      ... and {len(post_inflection) - 10} more timepoints")

    # Check dead_flag status
    if 'dead_flag' not in post_inflection.columns:
        print(f"    ❌ NO dead_flag column!")
        return False, {'post_count': len(post_inflection), 'dead_count': 0, 'dead_fraction': 0.0, 'error': 'no_dead_flag_column'}

    dead_count = post_inflection['dead_flag'].sum()
    total_count = len(post_inflection)
    dead_fraction = dead_count / total_count

    is_persistent = dead_fraction >= threshold

    print(f"    Dead count: {dead_count}/{total_count} = {dead_fraction:.1%}")
    print(f"    Threshold: {threshold:.1%}")
    print(f"    Result: {'✅ PERSISTENT' if is_persistent else '❌ NOT PERSISTENT'}")

    stats = {
        'post_count': total_count,
        'dead_count': dead_count,
        'dead_fraction': dead_fraction,
        'threshold': threshold,
        'is_persistent': is_persistent
    }

    return is_persistent, stats

def debug_find_inflection_candidates(embryo_data, min_decline_rate=0.1):
    """
    Debug version of candidate finding with detailed prints
    """
    print(f"\n🔍 FINDING INFLECTION CANDIDATES (min_decline_rate={min_decline_rate})")

    if len(embryo_data) < 3:
        print(f"❌ Insufficient data: {len(embryo_data)} timepoints")
        return []

    time_col = 'time_int'
    fraction_col = 'fraction_alive'

    data = embryo_data.sort_values(time_col).copy()

    times = data[time_col].values
    fractions = data[fraction_col].values

    print(f"Raw data points: {len(times)}")

    if len(times) < 3 or np.all(np.isnan(fractions)):
        print(f"❌ Invalid data")
        return []

    # Light smoothing to reduce noise
    if len(fractions) >= 5:
        fractions_smooth = savgol_filter(fractions, window_length=min(5, len(fractions)), polyorder=2)
        print(f"Applied Savitzky-Golay smoothing (window={min(5, len(fractions))})")
    else:
        fractions_smooth = fractions
        print(f"No smoothing applied (too few points)")

    # Calculate rate of change
    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    print(f"Calculated {len(rates)} rate values")
    print(f"Rate range: {rates.min():.4f} to {rates.max():.4f}")

    # Find declining points
    decline_mask = rates < -min_decline_rate
    declining_count = decline_mask.sum()

    print(f"Found {declining_count} declining points (rate < -{min_decline_rate})")

    candidates = []
    for i, is_declining in enumerate(decline_mask):
        if is_declining:
            time_point = times[i]
            rate_value = rates[i]
            candidates.append((time_point, rate_value))
            print(f"  Candidate {len(candidates)}: t={time_point:.1f}, rate={rate_value:.4f}")

    print(f"Total candidates found: {len(candidates)}")
    return candidates

def debug_detect_persistent_death_inflection(embryo_data, persistence_threshold=0.5, min_decline_rate=0.1):
    """
    Debug version of full algorithm with detailed prints
    """
    print(f"\n🚀 STARTING PERSISTENT DEATH DETECTION")
    print(f"Parameters: persistence_threshold={persistence_threshold}, min_decline_rate={min_decline_rate}")

    candidates_tested = []
    current_data = embryo_data.copy()
    iteration = 0

    while len(current_data) >= 3:
        iteration += 1
        print(f"\n{'='*20} ITERATION {iteration} {'='*20}")
        print(f"Current data range: t={current_data['time_int'].min():.1f} to {current_data['time_int'].max():.1f}")
        print(f"Current data points: {len(current_data)}")

        # Find candidates in current data subset
        candidates = debug_find_inflection_candidates(current_data, min_decline_rate)

        if not candidates:
            print(f"❌ No more candidates found. Algorithm terminates.")
            break

        # Test earliest candidate first
        earliest_time, earliest_rate = candidates[0]
        print(f"\n🎯 TESTING earliest candidate: t={earliest_time:.1f}, rate={earliest_rate:.4f}")

        # Validate persistence using ORIGINAL full dataset (not subset)
        is_persistent, stats = debug_validate_death_persistence(embryo_data, earliest_time, persistence_threshold)

        candidate_info = {
            'iteration': iteration,
            'time': earliest_time,
            'rate': earliest_rate,
            'is_persistent': is_persistent,
            'stats': stats
        }
        candidates_tested.append(candidate_info)

        if is_persistent:
            print(f"\n🎉 SUCCESS! Found persistent inflection at t={earliest_time:.1f}")
            return {
                'inflection_time': earliest_time,
                'persistence_stats': stats,
                'candidates_tested': candidates_tested
            }

        print(f"\n❌ Candidate FAILED persistence test. Removing data up to t={earliest_time:.1f}")

        # This candidate failed, remove data up to this point and try again
        time_col = 'time_int'
        old_length = len(current_data)
        current_data = current_data[current_data[time_col] > earliest_time]
        new_length = len(current_data)

        print(f"Removed {old_length - new_length} timepoints. Remaining: {new_length}")

        if iteration > 10:  # Safety break
            print(f"⚠️ Breaking after {iteration} iterations to prevent infinite loop")
            break

    print(f"\n❌ ALGORITHM COMPLETE: No persistent inflection found")
    print(f"Tested {len(candidates_tested)} candidates across {iteration} iterations")

    return None

def run_a10_debug():
    """
    Run complete debug analysis on A10
    """
    print(f"="*80)
    print(f"DEBUGGING PERSISTENCE ALGORITHM FOR {target_embryo}")
    print(f"="*80)

    # Show some raw data first
    print(f"\nRAW DATA SAMPLE (first 10 and last 10 timepoints):")
    print(f"{'Time':<8} {'Fraction':<10} {'dead_flag':<10}")
    print(f"-" * 30)

    for _, row in embryo_data.head(5).iterrows():
        print(f"{row['time_int']:<8.1f} {row['fraction_alive']:<10.3f} {row['dead_flag']}")

    print("  ...")

    for _, row in embryo_data.tail(5).iterrows():
        print(f"{row['time_int']:<8.1f} {row['fraction_alive']:<10.3f} {row['dead_flag']}")

    # Run the debug algorithm with more aggressive parameters
    print(f"\n🔄 RUNNING WITH MORE AGGRESSIVE PARAMETERS:")
    print(f"   min_decline_rate: 0.1 → 0.05 (catch smaller declines)")
    print(f"   persistence_threshold: 0.5 → 0.25 (25% dead_flag=True required)")

    result = debug_detect_persistent_death_inflection(
        embryo_data,
        persistence_threshold=0.25,  # 25% instead of 50%
        min_decline_rate=0.05        # 0.05 instead of 0.1
    )

    # Show final result
    print(f"\n{'='*20} FINAL RESULT {'='*20}")
    if result is not None:
        print(f"✅ DETECTED persistent death at t={result['inflection_time']:.1f}")
        stats = result['persistence_stats']
        print(f"   Persistence: {stats['dead_count']}/{stats['post_count']} = {stats['dead_fraction']:.1%}")
    else:
        print(f"❌ NO persistent death detected")

    print(f"\n🔚 DEBUG COMPLETE!")
    return result

if __name__ == "__main__":
    result = run_a10_debug()