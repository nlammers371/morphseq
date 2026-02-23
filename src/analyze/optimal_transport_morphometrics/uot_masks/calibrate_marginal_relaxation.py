"""Sweep marginal relaxation on identity pairs to calibrate reg_m."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from analyze.utils.optimal_transport import UOTConfig, UOTFrame, UOTFramePair, WorkingGridConfig
from .frame_mask_io import load_mask_from_csv
from .run_transport import run_uot_pair


def calibrate_on_identity(
    csv_path: Path,
    embryo_id: str,
    frame_index: int,
    reg_m_values: Iterable[float],
    base_config: UOTConfig,
    data_root: Path | None = None,
) -> pd.DataFrame:
    frame = load_mask_from_csv(csv_path, embryo_id, frame_index, data_root=data_root)
    results = []
    for reg_m in reg_m_values:
        cfg = UOTConfig(**{**base_config.__dict__, "marginal_relaxation": float(reg_m)})
        pair = UOTFramePair(src=frame, tgt=frame)
        res = run_uot_pair(pair, solver_cfg=cfg, working_cfg=WorkingGridConfig(), output_frame="work")
        metrics = res.diagnostics.get("metrics", {})
        transported = metrics.get("transported_mass", np.nan)
        total = float(frame.embryo_mask.sum())
        frac = transported / total if total > 0 else np.nan
        results.append({"reg_m": reg_m, "transported_mass": transported, "transported_frac": frac})
    return pd.DataFrame(results)


def plot_calibration(df: pd.DataFrame, output_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    ax.plot(df["reg_m"], df["transported_frac"], marker="o")
    ax.set_xscale("log")
    ax.set_title("Identity pair transport vs reg_m")
    ax.set_xlabel("marginal_relaxation (reg_m)")
    ax.set_ylabel("transported mass fraction")
    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig
