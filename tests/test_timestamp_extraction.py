#!/usr/bin/env python3
"""
Test script to validate ND2 timestamp extraction
Usage: python test_timestamp_extraction.py /path/to/experiment.nd2
"""
import sys
import nd2
import numpy as np
from pathlib import Path

def _fix_nd2_timestamp(nd, n_z):
    """Legacy timestamp extraction (working version)"""
    n_frames_total = nd.frame_metadata(0).contents.frameCount
    frame_time_vec = [nd.frame_metadata(i).channels[0].time.relativeTimeMs / 1000 for i in
                     range(0, n_frames_total, n_z)]

    # Check for timestamp jumps (ND2 artifact)
    dt_frame_approx = (nd.frame_metadata(n_z).channels[0].time.relativeTimeMs -
                      nd.frame_metadata(0).channels[0].time.relativeTimeMs) / 1000
    jump_ind = np.where(np.diff(frame_time_vec) > 2*dt_frame_approx)[0]

    if len(jump_ind) > 0:
        print(f"⚠️  Found timestamp jump at frame {jump_ind[0]}")
        # Fix jumps by extrapolation
        jump_ind = jump_ind[0]
        nf = jump_ind - 1 - int(jump_ind/2)
        dt_frame_est = (frame_time_vec[jump_ind-1] - frame_time_vec[int(jump_ind/2)]) / nf
        base_time = frame_time_vec[jump_ind-1]
        for f in range(jump_ind, len(frame_time_vec)):
            frame_time_vec[f] = base_time + dt_frame_est*(f - jump_ind)

    return np.asarray(frame_time_vec)

def test_timestamp_extraction(nd2_path):
    """Test both legacy and current timestamp extraction methods"""
    print(f"Testing: {nd2_path}")

    with nd2.ND2File(nd2_path) as nd:
        n_t, n_w, n_z = nd.shape[:3]
        print(f"Shape: T={n_t}, W={n_w}, Z={n_z}")

        # Legacy method
        legacy_times = _fix_nd2_timestamp(nd, n_z)
        print(f"\n📊 Legacy Method Results:")
        print(f"  First 5 times: {legacy_times[:5]}")
        print(f"  Last 5 times: {legacy_times[-5:]}")
        print(f"  Intervals (first 5): {np.diff(legacy_times[:6])}")
        print(f"  Total duration: {legacy_times[-1] - legacy_times[0]:.1f}s ({(legacy_times[-1] - legacy_times[0])/3600:.1f}h)")

        # Calculate relative times like the pipeline does
        relative_times = legacy_times - legacy_times.min()
        print(f"  Time Rel (s) first 5: {relative_times[:5]}")

        return legacy_times

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_timestamp_extraction.py /path/to/experiment.nd2")
        sys.exit(1)

    nd2_path = Path(sys.argv[1])
    if not nd2_path.exists():
        print(f"Error: {nd2_path} does not exist")
        sys.exit(1)

    test_timestamp_extraction(nd2_path)