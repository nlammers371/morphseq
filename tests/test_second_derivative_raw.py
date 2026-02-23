#!/usr/bin/env python3
"""
Test Second Derivative method WITHOUT smoothing to see raw signal behavior.

H11 and F11 should NOT detect inflections (they're noisy cases).
H05 and F06 should detect inflections (they have gradual decline).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load test data
data_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output/qc_staged_20250711.csv")
df = pd.read_csv(data_path)

# Find our target embryos
all_embryos = df['embryo_id'].unique()
target_wells = ['F11', 'H11', 'H05', 'F06']
TEST_EMBRYOS = [eid for eid in all_embryos if any(well in eid for well in target_wells)]

output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/tests/dead_flag2_inflection_analysis")

def detect_fraction_alive_inflection_second_derivative_raw(embryo_data, dead_lead_time=2.0, min_acceleration=0.05):
    """Second derivative method WITHOUT any smoothing"""
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

    # NO SMOOTHING - use raw fraction_alive
    fractions_raw = fractions

    # Calculate first derivative (velocity)
    dt = np.diff(times)
    df_dt = np.diff(fractions_raw)
    rates = df_dt / dt

    # Calculate second derivative (acceleration)
    dt2 = dt[1:]  # Time differences for second derivative
    d2f_dt2 = np.diff(rates)
    accelerations = d2f_dt2 / dt2

    # Find negative acceleration peaks (inflection points)
    accel_mask = accelerations < -min_acceleration

    if not np.any(accel_mask):
        return None

    # Find first significant acceleration in decline
    first_accel_idx = np.where(accel_mask)[0][0]
    inflection_time = times[first_accel_idx + 1]

    return inflection_time

def plot_second_derivative_analysis(embryo_id, save_dir):
    """Plot detailed second derivative analysis"""
    embryo_data = df[df['embryo_id'] == embryo_id].sort_values('time_int')

    if len(embryo_data) == 0:
        print(f"No data found for embryo {embryo_id}")
        return None

    times = embryo_data['time_int'].values
    fractions = embryo_data['fraction_alive'].values

    if len(times) < 4:
        print(f"Insufficient data for {embryo_id}")
        return None

    # Calculate derivatives
    dt = np.diff(times)
    df_dt = np.diff(fractions)
    rates = df_dt / dt

    # Second derivative
    dt2 = dt[1:]
    d2f_dt2 = np.diff(rates)
    accelerations = d2f_dt2 / dt2

    # Times for derivatives (shifted)
    times_rates = times[1:]  # First derivative times
    times_accel = times[2:]  # Second derivative times

    # Detect inflection
    inflection_time = detect_fraction_alive_inflection_second_derivative_raw(embryo_data)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Fraction Alive
    ax1.plot(times, fractions, 'o-', color='blue', alpha=0.7, label='Fraction Alive (Raw)')
    if inflection_time is not None:
        ax1.axvline(inflection_time, color='red', linestyle='--', alpha=0.8,
                   label=f'Inflection: t={inflection_time:.1f}')
    ax1.set_ylabel('Fraction Alive')
    ax1.set_title(f'{embryo_id}: Raw Signal Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: First Derivative (Rates)
    ax2.plot(times_rates, rates, 'o-', color='green', alpha=0.7, label='Rate (1st Derivative)')
    ax2.axhline(-0.1, color='gray', linestyle=':', alpha=0.5, label='Min Decline Rate (-0.1)')
    if inflection_time is not None:
        ax2.axvline(inflection_time, color='red', linestyle='--', alpha=0.8)
    ax2.set_ylabel('Rate of Change')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Second Derivative (Accelerations)
    ax3.plot(times_accel, accelerations, 'o-', color='purple', alpha=0.7, label='Acceleration (2nd Derivative)')
    ax3.axhline(-0.05, color='gray', linestyle=':', alpha=0.5, label='Min Acceleration (-0.05)')
    if inflection_time is not None:
        ax3.axvline(inflection_time, color='red', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Time Int')
    ax3.set_ylabel('Acceleration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = save_dir / f"embryo_{embryo_id}_second_derivative_raw_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Return analysis data
    return {
        'inflection_time': inflection_time,
        'max_acceleration_magnitude': np.min(accelerations) if len(accelerations) > 0 else None,
        'mean_rate': np.mean(rates),
        'std_fraction': np.std(fractions)
    }

def run_raw_second_derivative_analysis():
    """Run second derivative analysis without smoothing"""
    print("Testing Second Derivative method WITHOUT smoothing...")
    print("H11 and F11 should NOT detect (noisy cases)")
    print("H05 and F06 should detect (gradual decline cases)")
    print(f"Target embryos: {TEST_EMBRYOS}")

    results = {}

    for embryo_id in TEST_EMBRYOS:
        print(f"\nAnalyzing embryo {embryo_id}...")
        analysis = plot_second_derivative_analysis(embryo_id, output_dir)

        if analysis is not None:
            results[embryo_id] = analysis

            print(f"Results for {embryo_id}:")
            if analysis['inflection_time'] is not None:
                print(f"  ❌ DETECTED inflection at t={analysis['inflection_time']:.1f}")
            else:
                print(f"  ✅ NO inflection detected")

            print(f"  Max acceleration magnitude: {analysis['max_acceleration_magnitude']:.4f}")
            print(f"  Std of fraction_alive: {analysis['std_fraction']:.4f}")

    # Summary
    print("\n" + "="*80)
    print("RAW SECOND DERIVATIVE ANALYSIS SUMMARY")
    print("="*80)
    print("Expected: H11, F11 should NOT detect (noisy)")
    print("Expected: H05, F06 should detect (gradual decline)")
    print("-"*80)

    for embryo_id in TEST_EMBRYOS:
        if embryo_id in results:
            result = results[embryo_id]
            detected = "YES" if result['inflection_time'] is not None else "NO"
            expected_behavior = "NO" if any(x in embryo_id for x in ['H11', 'F11']) else "YES"
            status = "✅ CORRECT" if detected == expected_behavior else "❌ WRONG"

            print(f"{embryo_id:<20} Detected: {detected:<3} Expected: {expected_behavior:<3} {status}")

    print(f"\nRaw analysis plots saved to: {output_dir.absolute()}")
    return results

if __name__ == "__main__":
    results = run_raw_second_derivative_analysis()