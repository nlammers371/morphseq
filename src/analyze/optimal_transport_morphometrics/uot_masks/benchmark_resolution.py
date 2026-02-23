"""Benchmark UOT runtime and cost across downsample factors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import time

import numpy as np
import pandas as pd

from analyze.utils.optimal_transport import UOTConfig, WorkingGridConfig
from .frame_mask_io import load_mask_pair_from_csv
from .run_transport import run_uot_pair


def benchmark_downsample(
    csv_path: Path,
    embryo_id: str,
    frame_index_src: int,
    frame_index_tgt: int,
    downsample_factors: Iterable[int],
    base_config: UOTConfig,
    data_root: Path | None = None,
) -> pd.DataFrame:
    pair = load_mask_pair_from_csv(csv_path, embryo_id, frame_index_src, frame_index_tgt, data_root=data_root)

    results = []
    for factor in downsample_factors:
        working_cfg = WorkingGridConfig(downsample_factor=int(factor))
        start = time.perf_counter()
        res = run_uot_pair(pair, solver_cfg=base_config, working_cfg=working_cfg, output_frame="work")
        elapsed = time.perf_counter() - start

        metrics = res.diagnostics.get("metrics", {})
        results.append(
            {
                "downsample_factor": int(factor),
                "elapsed_s": elapsed,
                "cost": res.cost,
                "transported_mass": metrics.get("transported_mass", np.nan),
                "support_src_n": int(res.support_src_yx.shape[0]),
                "support_tgt_n": int(res.support_tgt_yx.shape[0]),
            }
        )

    return pd.DataFrame(results)


def plot_benchmark(df: pd.DataFrame, output_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    axes[0].plot(df["downsample_factor"], df["elapsed_s"], marker="o")
    axes[0].set_title("Runtime vs downsample")
    axes[0].set_xlabel("Downsample factor")
    axes[0].set_ylabel("Elapsed (s)")
    axes[0].set_xscale("log")

    axes[1].plot(df["downsample_factor"], df["cost"], marker="o")
    axes[1].set_title("Cost vs downsample")
    axes[1].set_xlabel("Downsample factor")
    axes[1].set_ylabel("Cost")
    axes[1].set_xscale("log")

    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig
