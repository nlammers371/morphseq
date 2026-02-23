"""Run UOT across consecutive frames for an embryo timeseries."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analyze.utils.coord.grids.canonical import CanonicalGridConfig
from analyze.utils.optimal_transport import UOTConfig, UOTFramePair, WorkingGridConfig
from analyze.utils.optimal_transport.backends.base import UOTBackend
from .frame_mask_io import load_mask_series_from_csv
from .run_transport import run_uot_pair


def run_timeseries_from_csv(
    csv_path: Path,
    embryo_id: str,
    frame_indices: Optional[List[int]] = None,
    solver_cfg: Optional[UOTConfig] = None,
    canonical_cfg: Optional[CanonicalGridConfig] = None,
    working_cfg: Optional[WorkingGridConfig] = None,
    backend: Optional[UOTBackend] = None,
    data_root: Optional[Path] = None,
) -> List[Tuple[int, int, dict]]:
    if solver_cfg is None:
        solver_cfg = UOTConfig()
    if canonical_cfg is None:
        canonical_cfg = CanonicalGridConfig()
    if working_cfg is None:
        working_cfg = WorkingGridConfig()

    frames = load_mask_series_from_csv(csv_path, embryo_id, frame_indices=frame_indices, data_root=data_root)
    results: List[Tuple[int, int, dict]] = []

    for i in range(len(frames) - 1):
        src = frames[i]
        tgt = frames[i + 1]
        pair = UOTFramePair(src=src, tgt=tgt)
        res = run_uot_pair(
            pair,
            canonical_cfg=canonical_cfg,
            working_cfg=working_cfg,
            solver_cfg=solver_cfg,
            backend=backend,
            output_frame="work",
        )
        frame_src = int(src.meta.get("frame_index", i))
        frame_tgt = int(tgt.meta.get("frame_index", i + 1))
        metrics = res.diagnostics.get("metrics", {})
        metrics.update({"frame_src": frame_src, "frame_tgt": frame_tgt, "cost": res.cost})
        results.append((frame_src, frame_tgt, metrics))

    return results


def results_to_dataframe(results: List[Tuple[int, int, dict]]) -> pd.DataFrame:
    rows = [metrics for _, _, metrics in results]
    return pd.DataFrame(rows)


def plot_timeseries_cost(results: List[Tuple[int, int, dict]], output_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No results to plot.")

    frames = [r[0] for r in results]
    costs = [r[2].get("cost", np.nan) for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    ax.plot(frames, costs, marker="o")
    ax.set_title("UOT cost over time")
    ax.set_xlabel("Frame index (source)")
    ax.set_ylabel("Cost")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig
