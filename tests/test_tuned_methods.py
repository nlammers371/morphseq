#!/usr/bin/env python3
"""
Tuned versions of inflection detection methods based on initial results.

FINDINGS FROM INITIAL TEST:
- Sustained Decline: Too conservative (required 2 consecutive, found nothing)
- Enhanced Smoothing: Too aggressive smoothing (found nothing)
- Second Derivative: Promising (found all embryos, earlier detection for F11)

TUNING STRATEGY:
- Sustained: Try requiring only 1 consecutive (less conservative)
- Enhanced: Reduce smoothing intensity
- Test combinations with Second Derivative
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# Load test data
data_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output/qc_staged_20250711.csv")
df = pd.read_csv(data_path)

# Find our target embryos
all_embryos = df['embryo_id'].unique()
target_wells = ['F11', 'H11', 'H05', 'F06']
TEST_EMBRYOS = [eid for eid in all_embryos if any(well in eid for well in target_wells)]

output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/tests/dead_flag2_inflection_analysis")

def detect_fraction_alive_inflection_original(embryo_data, dead_lead_time=2.0, min_decline_rate=0.1):
    """Original method for comparison"""
    if len(embryo_data) < 3:
        return None

    time_col = 'time_int'
    fraction_col = 'fraction_alive'
    data = embryo_data.sort_values(time_col).copy()

    if fraction_col not in data.columns or time_col not in data.columns:
        return None

    times = data[time_col].values
    fractions = data[fraction_col].values

    if len(times) < 3 or np.all(np.isnan(fractions)):
        return None

    if len(fractions) >= 5:
        fractions_smooth = savgol_filter(fractions, window_length=min(5, len(fractions)), polyorder=2)
    else:
        fractions_smooth = fractions

    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    decline_mask = rates < -min_decline_rate

    if not np.any(decline_mask):
        return None

    first_decline_idx = np.where(decline_mask)[0][0]
    inflection_time = times[first_decline_idx]

    return inflection_time

def detect_fraction_alive_inflection_sustained_tuned(embryo_data, dead_lead_time=2.0, min_decline_rate=0.1):
    """TUNED: Require sustained decline but be less strict"""
    if len(embryo_data) < 4:  # Need at least 4 for sustained check
        return None

    time_col = 'time_int'
    fraction_col = 'fraction_alive'
    data = embryo_data.sort_values(time_col).copy()

    if fraction_col not in data.columns or time_col not in data.columns:
        return None

    times = data[time_col].values
    fractions = data[fraction_col].values

    if len(times) < 4 or np.all(np.isnan(fractions)):
        return None

    if len(fractions) >= 5:
        fractions_smooth = savgol_filter(fractions, window_length=min(5, len(fractions)), polyorder=2)
    else:
        fractions_smooth = fractions

    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    # First find significant declines
    decline_mask = rates < -min_decline_rate

    if not np.any(decline_mask):
        return None

    decline_indices = np.where(decline_mask)[0]

    # TUNED: Look for decline followed by either:
    # 1. Another decline (sustained)
    # 2. OR a non-positive rate (not recovering immediately)
    for idx in decline_indices:
        if idx + 1 < len(rates):  # Make sure we have a next point
            next_rate = rates[idx + 1]
            # Accept if next rate is also declining OR at least not positive (not recovering)
            if next_rate <= 0.05:  # Allow small positive fluctuations
                inflection_time = times[idx]
                return inflection_time

    return None

def detect_fraction_alive_inflection_enhanced_tuned(embryo_data, dead_lead_time=2.0, min_decline_rate=0.1):
    """TUNED: Lighter smoothing to preserve signal"""
    if len(embryo_data) < 3:
        return None

    time_col = 'time_int'
    fraction_col = 'fraction_alive'
    data = embryo_data.sort_values(time_col).copy()

    if fraction_col not in data.columns or time_col not in data.columns:
        return None

    times = data[time_col].values
    fractions = data[fraction_col].values

    if len(times) < 3 or np.all(np.isnan(fractions)):
        return None

    # TUNED: Much lighter smoothing
    if len(fractions) >= 5:
        # Use smaller sigma for Gaussian (less aggressive)
        sigma = 0.8  # Much smaller than before
        fractions_smooth = gaussian_filter1d(fractions, sigma=sigma)
    else:
        fractions_smooth = fractions

    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    decline_mask = rates < -min_decline_rate

    if not np.any(decline_mask):
        return None

    first_decline_idx = np.where(decline_mask)[0][0]
    inflection_time = times[first_decline_idx]

    return inflection_time

def detect_fraction_alive_inflection_second_derivative(embryo_data, dead_lead_time=2.0, min_acceleration=0.05):
    """Second derivative method (unchanged - was working well)"""
    if len(embryo_data) < 4:
        return None

    time_col = 'time_int'
    fraction_col = 'fraction_alive'
    data = embryo_data.sort_values(time_col).copy()

    if fraction_col not in data.columns or time_col not in data.columns:
        return None

    times = data[time_col].values
    fractions = data[fraction_col].values

    if len(times) < 4 or np.all(np.isnan(fractions)):
        return None

    if len(fractions) >= 5:
        fractions_smooth = savgol_filter(fractions, window_length=min(5, len(fractions)), polyorder=2)
    else:
        fractions_smooth = fractions

    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    dt2 = dt[1:]
    d2f_dt2 = np.diff(rates)
    accelerations = d2f_dt2 / dt2

    accel_mask = accelerations < -min_acceleration

    if not np.any(accel_mask):
        return None

    first_accel_idx = np.where(accel_mask)[0][0]
    inflection_time = times[first_accel_idx + 1]

    return inflection_time

def detect_fraction_alive_inflection_hybrid(embryo_data, dead_lead_time=2.0, min_decline_rate=0.1, min_acceleration=0.05):
    """HYBRID: Combine Second Derivative with Sustained validation"""
    if len(embryo_data) < 4:
        return None

    time_col = 'time_int'
    fraction_col = 'fraction_alive'
    data = embryo_data.sort_values(time_col).copy()

    if fraction_col not in data.columns or time_col not in data.columns:
        return None

    times = data[time_col].values
    fractions = data[fraction_col].values

    if len(times) < 4 or np.all(np.isnan(fractions)):
        return None

    if len(fractions) >= 5:
        fractions_smooth = savgol_filter(fractions, window_length=min(5, len(fractions)), polyorder=2)
    else:
        fractions_smooth = fractions

    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    # Step 1: Find acceleration-based candidates (Second Derivative)
    dt2 = dt[1:]
    d2f_dt2 = np.diff(rates)
    accelerations = d2f_dt2 / dt2

    accel_mask = accelerations < -min_acceleration
    if not np.any(accel_mask):
        return None

    # Step 2: Validate with sustained decline check
    accel_indices = np.where(accel_mask)[0]

    for accel_idx in accel_indices:
        # Check if this acceleration is followed by sustained decline
        rate_idx = accel_idx + 1  # Convert to rates index
        if rate_idx < len(rates) and rate_idx + 1 < len(rates):
            current_rate = rates[rate_idx]
            next_rate = rates[rate_idx + 1]

            # Require current rate to be declining AND next rate to not recover strongly
            if current_rate < -min_decline_rate and next_rate <= 0.05:
                inflection_time = times[accel_idx + 1]
                return inflection_time

    return None

def plot_tuned_comparison(embryo_id, save_dir):
    """Plot comparison of tuned methods"""
    embryo_data = df[df['embryo_id'] == embryo_id].sort_values('time_int')

    if len(embryo_data) == 0:
        print(f"No data found for embryo {embryo_id}")
        return None

    # Run all methods
    methods = {
        'Original': detect_fraction_alive_inflection_original,
        'Sustained Tuned': detect_fraction_alive_inflection_sustained_tuned,
        'Enhanced Tuned': detect_fraction_alive_inflection_enhanced_tuned,
        'Second Derivative': detect_fraction_alive_inflection_second_derivative,
        'Hybrid': detect_fraction_alive_inflection_hybrid
    }

    inflection_times = {}
    for method_name, method_func in methods.items():
        inflection_times[method_name] = method_func(embryo_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))

    times = embryo_data['time_int']
    fractions = embryo_data['fraction_alive']

    # Plot fraction_alive data
    ax.plot(times, fractions, 'o-', alpha=0.7, color='blue', label='Fraction Alive', markersize=4)

    # Plot inflection points for each method
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (method_name, inflection_time) in enumerate(inflection_times.items()):
        if inflection_time is not None:
            ax.axvline(inflection_time, color=colors[i], linestyle='--',
                      label=f'{method_name}: t={inflection_time:.1f}', alpha=0.8, linewidth=2)

    # Color by current dead_flag2 if available
    if 'dead_flag2' in embryo_data.columns:
        current_dead = embryo_data['dead_flag2']
        dead_times = times[current_dead]
        dead_fractions = fractions[current_dead]
        ax.scatter(dead_times, dead_fractions, c='red', s=100, alpha=0.5,
                  marker='x', label='Current dead_flag2', linewidth=3)

    ax.set_xlabel('Time Int')
    ax.set_ylabel('Fraction Alive')
    ax.set_title(f'Embryo {embryo_id}: Tuned Methods Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Save plot
    save_path = save_dir / f"embryo_{embryo_id}_tuned_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return inflection_times

def run_tuned_testing():
    """Run tuned method testing"""
    print("Testing TUNED methods on specific embryos...")
    print(f"Target embryos: {TEST_EMBRYOS}")

    results = {}

    for embryo_id in TEST_EMBRYOS:
        print(f"\nTesting embryo {embryo_id}...")
        inflection_times = plot_tuned_comparison(embryo_id, output_dir)

        if inflection_times is not None:
            results[embryo_id] = inflection_times

            print(f"Results for {embryo_id}:")
            for method, time in inflection_times.items():
                if time is not None:
                    print(f"  {method}: t={time:.1f}")
                else:
                    print(f"  {method}: No inflection detected")
        else:
            results[embryo_id] = {}

    # Summary table
    print("\n" + "="*100)
    print("TUNED METHODS SUMMARY TABLE")
    print("="*100)
    print(f"{'Embryo':<18} {'Original':<10} {'Sustained':<12} {'Enhanced':<12} {'2nd Deriv':<12} {'Hybrid':<10}")
    print("-" * 100)

    for embryo_id in TEST_EMBRYOS:
        if embryo_id in results:
            row = f"{embryo_id:<18}"
            for method in ['Original', 'Sustained Tuned', 'Enhanced Tuned', 'Second Derivative', 'Hybrid']:
                time = results[embryo_id].get(method)
                if time is not None:
                    row += f" {time:<11.1f}"
                else:
                    row += f" {'None':<11}"
            print(row)

    print(f"\nTuned comparison plots saved to: {output_dir.absolute()}")
    return results

if __name__ == "__main__":
    results = run_tuned_testing()