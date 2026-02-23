#!/usr/bin/env python3
"""
Focused testing script for inflection detection improvements.

Testing on specific embryos:
- F11 & H11: Too sensitive to noise data
- H05 & F06: Working well with gradual decline

This script will test improvements one by one:
1. Sustained Decline Detection
2. Quality Control Filters
3. Enhanced Smoothing
4. Second Derivative Method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

# Load test data
data_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output/qc_staged_20250711.csv")
df = pd.read_csv(data_path)

# Find actual embryo IDs that contain our target wells
all_embryos = df['embryo_id'].unique()
target_wells = ['F11', 'H11', 'H05', 'F06']
TEST_EMBRYOS = [eid for eid in all_embryos if any(well in eid for well in target_wells)]

print(f"Loaded {len(df)} rows from {data_path}")
print(f"All embryo count: {len(all_embryos)}")
print(f"Found target embryos: {TEST_EMBRYOS}")
if len(TEST_EMBRYOS) == 0:
    print("No target embryos found! Showing first 10 embryo IDs:")
    print(all_embryos[:10])

# Create output directory
output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/tests/dead_flag2_inflection_analysis")
output_dir.mkdir(exist_ok=True)

def detect_fraction_alive_inflection_original(embryo_data, dead_lead_time=2.0, min_decline_rate=0.1):
    """Original inflection detection method (baseline)"""
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

    # Smooth the fraction_alive signal
    if len(fractions) >= 5:
        fractions_smooth = savgol_filter(fractions, window_length=min(5, len(fractions)), polyorder=2)
    else:
        fractions_smooth = fractions

    # Calculate derivative (rate of change)
    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    # Find points where decline rate exceeds threshold
    decline_mask = rates < -min_decline_rate

    if not np.any(decline_mask):
        return None

    # Find first significant decline
    first_decline_idx = np.where(decline_mask)[0][0]
    inflection_time = times[first_decline_idx]

    return inflection_time

def detect_fraction_alive_inflection_sustained(embryo_data, dead_lead_time=2.0, min_decline_rate=0.1,
                                              consecutive_declines=2):
    """IMPROVEMENT 1: Sustained decline detection - require consecutive declining points"""
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

    # Smooth the fraction_alive signal
    if len(fractions) >= 5:
        fractions_smooth = savgol_filter(fractions, window_length=min(5, len(fractions)), polyorder=2)
    else:
        fractions_smooth = fractions

    # Calculate derivative (rate of change)
    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    # Find points where decline rate exceeds threshold
    decline_mask = rates < -min_decline_rate

    if not np.any(decline_mask):
        return None

    # NEW LOGIC: Require consecutive declining points
    decline_indices = np.where(decline_mask)[0]

    # Look for consecutive declining points
    for i in range(len(decline_indices) - consecutive_declines + 1):
        # Check if we have consecutive_declines consecutive indices
        consecutive_check = decline_indices[i:i+consecutive_declines]
        if np.all(np.diff(consecutive_check) == 1):  # Consecutive indices
            inflection_time = times[consecutive_check[0]]
            return inflection_time

    return None

def detect_fraction_alive_inflection_qc_filtered(embryo_data, dead_lead_time=2.0, min_decline_rate=0.1,
                                                min_data_points=10, max_noise_std=0.3):
    """IMPROVEMENT 2: Quality control filters - skip noisy/insufficient data"""
    if len(embryo_data) < min_data_points:
        return None

    time_col = 'time_int'
    fraction_col = 'fraction_alive'

    data = embryo_data.sort_values(time_col).copy()

    if fraction_col not in data.columns or time_col not in data.columns:
        return None

    times = data[time_col].values
    fractions = data[fraction_col].values

    if len(times) < min_data_points or np.all(np.isnan(fractions)):
        return None

    # NEW LOGIC: Check for excessive noise
    fraction_std = np.nanstd(fractions)
    if fraction_std > max_noise_std:
        return None  # Skip noisy embryos

    # Rest is same as original method
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

def detect_fraction_alive_inflection_enhanced_smoothing(embryo_data, dead_lead_time=2.0, min_decline_rate=0.1,
                                                       smoothing_window=7, use_gaussian=True):
    """IMPROVEMENT 3: Enhanced smoothing - more aggressive noise reduction"""
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

    # NEW LOGIC: Enhanced smoothing
    if use_gaussian and len(fractions) >= 5:
        # Use Gaussian filter for smoother results
        sigma = min(2.0, len(fractions) / 6)  # Adaptive sigma
        fractions_smooth = gaussian_filter1d(fractions, sigma=sigma)
    elif len(fractions) >= smoothing_window:
        # Use larger Savitzky-Golay window
        window = min(smoothing_window, len(fractions))
        if window % 2 == 0:  # Ensure odd window
            window -= 1
        fractions_smooth = savgol_filter(fractions, window_length=window, polyorder=2)
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
    """IMPROVEMENT 4: Second derivative method - detect acceleration changes"""
    if len(embryo_data) < 4:  # Need at least 4 points for second derivative
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

    # Smooth the fraction_alive signal
    if len(fractions) >= 5:
        fractions_smooth = savgol_filter(fractions, window_length=min(5, len(fractions)), polyorder=2)
    else:
        fractions_smooth = fractions

    # Calculate first derivative (velocity)
    dt = np.diff(times)
    df_dt = np.diff(fractions_smooth)
    rates = df_dt / dt

    # Calculate second derivative (acceleration)
    dt2 = dt[1:]  # Time differences for second derivative
    d2f_dt2 = np.diff(rates)
    accelerations = d2f_dt2 / dt2

    # NEW LOGIC: Find negative acceleration peaks (inflection points)
    # Look for points where decline is accelerating (negative acceleration)
    accel_mask = accelerations < -min_acceleration

    if not np.any(accel_mask):
        return None

    # Find first significant acceleration in decline
    first_accel_idx = np.where(accel_mask)[0][0]
    # Add 1 because accelerations array is shifted by 1 from times
    inflection_time = times[first_accel_idx + 1]

    return inflection_time

def plot_embryo_method_comparison(embryo_id, save_dir):
    """Plot comparison of all methods for a single embryo"""
    embryo_data = df[df['embryo_id'] == embryo_id].sort_values('time_int')

    if len(embryo_data) == 0:
        print(f"No data found for embryo {embryo_id}")
        return

    # Run all methods (skip QC Filtered per user request)
    methods = {
        'Original': detect_fraction_alive_inflection_original,
        'Sustained': detect_fraction_alive_inflection_sustained,
        'Second Derivative': detect_fraction_alive_inflection_second_derivative,
        'Enhanced Smoothing': detect_fraction_alive_inflection_enhanced_smoothing
    }

    inflection_times = {}
    for method_name, method_func in methods.items():
        inflection_times[method_name] = method_func(embryo_data)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    times = embryo_data['time_int']
    fractions = embryo_data['fraction_alive']

    # Plot fraction_alive data
    ax.plot(times, fractions, 'o-', alpha=0.7, color='blue', label='Fraction Alive')

    # Plot inflection points for each method
    colors = ['red', 'orange', 'purple', 'green']
    for i, (method_name, inflection_time) in enumerate(inflection_times.items()):
        if inflection_time is not None:
            ax.axvline(inflection_time, color=colors[i], linestyle='--',
                      label=f'{method_name}: t={inflection_time:.1f}', alpha=0.8)

    # Color by current dead_flag2 if available
    if 'dead_flag2' in embryo_data.columns:
        current_dead = embryo_data['dead_flag2']
        dead_times = times[current_dead]
        dead_fractions = fractions[current_dead]
        ax.scatter(dead_times, dead_fractions, c='red', s=100, alpha=0.5,
                  marker='x', label='Current dead_flag2')

    ax.set_xlabel('Time Int')
    ax.set_ylabel('Fraction Alive')
    ax.set_title(f'Embryo {embryo_id}: Method Comparison')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    # Save plot
    save_path = save_dir / f"embryo_{embryo_id}_method_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return inflection_times

def run_focused_testing():
    """Run focused testing on the 4 specific embryos"""
    print("Starting focused testing on specific embryos...")

    results = {}

    for embryo_id in TEST_EMBRYOS:
        print(f"\nTesting embryo {embryo_id}...")
        inflection_times = plot_embryo_method_comparison(embryo_id, output_dir)
        if inflection_times is not None:
            results[embryo_id] = inflection_times

            # Print results for this embryo
            print(f"Results for {embryo_id}:")
            for method, time in inflection_times.items():
                if time is not None:
                    print(f"  {method}: t={time:.1f}")
                else:
                    print(f"  {method}: No inflection detected")
        else:
            results[embryo_id] = {}
            print(f"No results for {embryo_id} (no data found)")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Embryo':<8} {'Original':<10} {'Sustained':<10} {'2nd Deriv':<10} {'Enhanced':<10}")
    print("-" * 70)

    for embryo_id in TEST_EMBRYOS:
        row = f"{embryo_id:<8}"
        for method in ['Original', 'Sustained', 'Second Derivative', 'Enhanced Smoothing']:
            time = results[embryo_id].get(method)
            if time is not None:
                row += f" {time:<9.1f}"
            else:
                row += f" {'None':<9}"
        print(row)

    print(f"\nPlots saved to: {output_dir.absolute()}")
    return results

if __name__ == "__main__":
    results = run_focused_testing()