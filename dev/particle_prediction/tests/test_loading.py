from pathlib import Path

import numpy as np
import pandas as pd

from dev.particle_prediction.data.loading import load_trajectories


def _write_test_csv(path: Path) -> None:
    rows = []
    for embryo_id, n_frames in (("emb_a", 18), ("emb_short", 10)):
        for frame in range(n_frames):
            rows.append(
                {
                    "experiment_id": "exp_alpha",
                    "embryo_id": embryo_id,
                    "frame_index": frame,
                    "relative_time_s": frame * 60.0,
                    "temperature": 28.5,
                    "genotype": "wt",
                    "background": "demo_bg",
                    "phenotype": "normal",
                    "use_embryo_flag": True,
                    "z_mu_b0": frame,
                    "z_mu_b1": frame * 0.5,
                    "z_mu_b2": frame * 0.25,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_trajectories_filters_short_embryos_and_preserves_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "df03_final_output_with_latents_exp_alpha.csv"
    _write_test_csv(csv_path)

    dataset = load_trajectories(
        build_dir=tmp_path,
        experiment_ids=["exp_alpha"],
        n_components=2,
        min_trajectory_length=15,
        verbose=False,
        metadata_columns=("background", "phenotype"),
    )

    assert len(dataset.trajectories) == 1
    trajectory = dataset.trajectories[0]
    assert trajectory.embryo_id == "emb_a"
    assert trajectory.trajectory.shape == (18, 2)
    assert trajectory.frame_index is not None
    assert len(trajectory.time_seconds) == 18
    assert np.all(np.diff(trajectory.time_seconds) >= 0)
    assert trajectory.metadata == {"background": "demo_bg", "phenotype": "normal"}


def test_load_trajectories_autodiscovers_experiments(tmp_path: Path) -> None:
    csv_path = tmp_path / "df03_final_output_with_latents_exp_beta.csv"
    _write_test_csv(csv_path)

    dataset = load_trajectories(
        build_dir=tmp_path,
        experiment_ids=None,
        n_components=3,
        min_trajectory_length=15,
        verbose=False,
    )

    assert len(dataset.trajectories) == 1
    assert dataset.n_components == 3
    assert dataset.build_dir == tmp_path