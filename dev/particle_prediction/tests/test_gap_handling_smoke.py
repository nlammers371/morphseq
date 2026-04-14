from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dev.particle_prediction.data.loading import load_trajectories
from dev.particle_prediction.data.smoothing import smooth_trajectory


def test_gap_classification_and_segmentwise_smoothing_smoke(tmp_path: Path) -> None:
    build_dir = tmp_path / "build06_output"
    build_dir.mkdir()

    df = pd.DataFrame(
        {
            "embryo_id": [
                "emb_gap",
                "emb_gap",
                "emb_gap",
                "emb_gap",
                "emb_gap",
                "emb_regular",
                "emb_regular",
                "emb_regular",
                "emb_regular",
                "emb_regular",
            ],
            "frame_index": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            "relative_time_s": [0.0, 10.0, 40.0, 110.0, 120.0, 0.0, 10.0, 20.0, 30.0, 40.0],
            "temperature": [28.5] * 10,
            "genotype": ["wt"] * 10,
            "use_embryo_flag": [True] * 10,
            "z_mu_b0": [0.0, 1.0, 2.0, 100.0, 101.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_csv(build_dir / "df03_final_output_with_latents_exp01.csv", index=False)

    dataset = load_trajectories(
        experiment_ids=["exp01"],
        build_dir=build_dir,
        n_components=1,
        scale=False,
        min_trajectory_length=3,
        verbose=False,
    )

    gap_trajectory = next(traj for traj in dataset.trajectories if traj.embryo_id == "emb_gap")

    assert gap_trajectory.observed_dts is not None
    assert gap_trajectory.missing_frame_counts is not None
    assert gap_trajectory.interpolatable_gap_mask is not None
    assert gap_trajectory.hard_gap_mask is not None
    assert gap_trajectory.segment_ids is not None

    assert np.allclose(gap_trajectory.observed_dts, np.array([10.0, 30.0, 70.0, 10.0]))
    assert np.array_equal(gap_trajectory.missing_frame_counts, np.array([0, 2, 6, 0]))
    assert np.array_equal(gap_trajectory.interpolatable_gap_mask, np.array([False, True, False, False]))
    assert np.array_equal(gap_trajectory.hard_gap_mask, np.array([False, False, True, False]))
    assert np.array_equal(gap_trajectory.segment_ids, np.array([0, 0, 0, 1, 1]))

    smoothed = smooth_trajectory(gap_trajectory, window_seconds=50.0, poly_order=1)

    assert smoothed.diagnostics["n_segments"] == 2
    assert smoothed.diagnostics["n_interpolated_points"] == 2
    assert smoothed.diagnostics["n_hard_gaps"] == 1
    assert np.linalg.norm(smoothed.smoothed[2] - smoothed.smoothed[3]) > 20.0