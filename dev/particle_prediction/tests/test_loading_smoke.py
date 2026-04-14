from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dev.particle_prediction.data import load_trajectories


def test_load_trajectories_smoke(tmp_path: Path) -> None:
    build_dir = tmp_path / "build06_output"
    build_dir.mkdir()

    df = pd.DataFrame(
        {
            "embryo_id": ["emb_a", "emb_a", "emb_a", "emb_b", "emb_b", "emb_b"],
            "frame_index": [0, 1, 2, 0, 1, 2],
            "relative_time_s": [0.0, 60.0, 120.0, 0.0, 60.0, 120.0],
            "temperature": [28.5, 28.5, 28.5, 30.0, 30.0, 30.0],
            "genotype": ["wt", "wt", "wt", "mut", "mut", "mut"],
            "use_embryo_flag": [True, True, True, True, True, True],
            "background": ["bg1", "bg1", "bg1", "bg2", "bg2", "bg2"],
            "z_mu_b0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z_mu_b1": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        }
    )
    df.to_csv(build_dir / "df03_final_output_with_latents_exp01.csv", index=False)

    dataset = load_trajectories(
        experiment_ids=["exp01"],
        build_dir=build_dir,
        n_components=2,
        min_trajectory_length=3,
        verbose=False,
    )

    assert len(dataset) == 2
    assert dataset.n_components == 2
    assert dataset.class_names == ["mut", "wt"]
    assert dataset.z_mu_cols == ["z_mu_b0", "z_mu_b1"]

    first = dataset.trajectories[0]
    assert first.trajectory.shape == (3, 2)
    assert np.all(np.isfinite(first.trajectory))
    assert np.array_equal(first.time_seconds, np.array([0.0, 60.0, 120.0]))
    assert first.delta_t == 60.0