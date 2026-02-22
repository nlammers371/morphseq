"""
UOT Solver Batch Benchmark — Cold vs Warm JIT timing (20260222)
===============================================================
Measures how much JIT amortization closes the CPU gap between OTTBackend and POTBackend.

For each (epsilon, max_pts, backend):
  1. Fresh backend instance
  2. Solve pair 0 → cold_time_s  (includes XLA compile for OTT)
  3. Solve pairs 1, 2 → warm_time_s  (mean of 2 warm calls)
  4. amortized_time_s = mean of all 3 calls

block_until_ready() is inside _solve_with_jit (leaf-based, robust), so timing
at benchmark level is automatically correct — no extra sync needed here.

Outputs:
  results/mcolon/20260222_uot_solver_benchmark/batch_results.csv
  results/mcolon/20260222_uot_solver_benchmark/plots/warmup_effect.png

Usage:
    PYTHONPATH=src python results/mcolon/20260222_uot_solver_benchmark/run_batch_benchmark.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure morphseq root and src/ are on sys.path
# ---------------------------------------------------------------------------
_morphseq_root = Path(__file__).resolve().parents[3]
for _p in [_morphseq_root / "src", _morphseq_root]:
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).parent
LOGS_DIR = BENCH_DIR / "logs"
PLOTS_DIR = BENCH_DIR / "plots"
BATCH_RESULTS_CSV = BENCH_DIR / "batch_results.csv"

DATA_CSV = Path(
    "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOGS_DIR / f"batch_benchmark_{ts}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep configuration (same 3 pairs as run_benchmark.py)
# ---------------------------------------------------------------------------
EPSILONS = [1e-4, 1e-3, 1e-2]
MAX_PTS = [1000, 3000]
MARGINAL_RELAXATION = 10.0
DOWNSAMPLE_FACTOR = 4
COORD_SCALE = 1.0 / 576
CANONICAL_GRID_HW = (256, 576)
CANONICAL_UM_PER_PX = 10.0
RANDOM_SEED = 42

# 3 pairs — pair 0 = cold call; pairs 1, 2 = warm calls
PAIRS = [
    ("20251113_E02_e01", "20251113_E04_e01", 88, "WT_vs_WT"),
    ("20251113_E04_e01", "20251113_A05_e01", 88, "WT_vs_HOM"),
    ("20251113_A05_e01", "20251113_B01_e01", 88, "HOM_vs_HOM"),
]


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
def _import_uot():
    from analyze.utils.optimal_transport import (
        UOTConfig,
        MassMode,
        WorkingGridConfig,
        prepare_working_grid_pair,
        run_uot_on_working_grid,
        lift_work_result_to_canonical,
        POTBackend,
        OTTBackend,
    )
    from analyze.utils.coord.grids.canonical import CanonicalGridConfig, to_canonical_grid_mask
    from analyze.optimal_transport_morphometrics.uot_masks.frame_mask_io import load_mask_from_csv

    return dict(
        UOTConfig=UOTConfig,
        MassMode=MassMode,
        WorkingGridConfig=WorkingGridConfig,
        prepare_working_grid_pair=prepare_working_grid_pair,
        run_uot_on_working_grid=run_uot_on_working_grid,
        lift_work_result_to_canonical=lift_work_result_to_canonical,
        POTBackend=POTBackend,
        OTTBackend=OTTBackend,
        CanonicalGridConfig=CanonicalGridConfig,
        to_canonical_grid_mask=to_canonical_grid_mask,
        load_mask_from_csv=load_mask_from_csv,
    )


# ---------------------------------------------------------------------------
# Mask helpers (copied from run_benchmark.py for self-containedness)
# ---------------------------------------------------------------------------
def load_frame(csv_path: Path, embryo_id: str, frame_index: int, api):
    frame = api["load_mask_from_csv"](csv_path, embryo_id, frame_index)
    um_per_px = frame.meta.get("um_per_pixel", np.nan) if frame.meta else np.nan
    if np.isnan(um_per_px):
        raise ValueError(f"No um_per_pixel for {embryo_id} frame {frame_index}")
    return frame, um_per_px


def canonicalize(mask: np.ndarray, um_per_px: float, api) -> object:
    canonical_cfg = api["CanonicalGridConfig"](
        reference_um_per_pixel=CANONICAL_UM_PER_PX,
        grid_shape_hw=CANONICAL_GRID_HW,
        align_mode="yolk",
    )
    bin_mask = (np.asarray(mask) > 0).astype(np.uint8)
    return api["to_canonical_grid_mask"](
        bin_mask,
        um_per_px=float(um_per_px),
        cfg=canonical_cfg,
    )


# ---------------------------------------------------------------------------
# Single timed solve
# ---------------------------------------------------------------------------
def timed_solve(src_can, tgt_can, epsilon: float, backend, max_pts: int, api) -> tuple[float, object]:
    """Run one solve and return (elapsed_s, result_work)."""
    working_cfg = api["WorkingGridConfig"](
        downsample_factor=DOWNSAMPLE_FACTOR,
        padding_px=16,
        mass_mode=api["MassMode"].UNIFORM,
    )
    uot_cfg = api["UOTConfig"](
        epsilon=epsilon,
        marginal_relaxation=MARGINAL_RELAXATION,
        max_support_points=max_pts,
        store_coupling=True,
        random_seed=RANDOM_SEED,
        metric="sqeuclidean",
        coord_scale=COORD_SCALE,
    )

    pair_work = api["prepare_working_grid_pair"](src_can, tgt_can, working_cfg)

    t0 = time.perf_counter()
    result_work = api["run_uot_on_working_grid"](pair_work, config=uot_cfg, backend=backend)
    elapsed = time.perf_counter() - t0
    return elapsed, result_work


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_batch_benchmark():
    log.info("=== UOT Batch Benchmark (Cold vs Warm JIT) 20260222 ===")
    log.info(f"Log: {log_path}")

    api = _import_uot()
    log.info("Imports OK")

    # Pre-canonicalize all unique embryos
    log.info("Pre-loading and canonicalizing embryo masks …")
    emb_frames: dict[tuple[str, int], object] = {}
    for src_id, tgt_id, frame_idx, _ in PAIRS:
        for emb_id in (src_id, tgt_id):
            key = (emb_id, frame_idx)
            if key not in emb_frames:
                log.info(f"  Loading {emb_id} frame {frame_idx}")
                frame, um = load_frame(DATA_CSV, emb_id, frame_idx, api)
                emb_frames[key] = canonicalize(frame.embryo_mask, um, api)
    log.info("All masks canonicalized.")

    records = []
    n_combos = len(EPSILONS) * len(MAX_PTS) * 2  # 2 backends
    combo_idx = 0

    for epsilon in EPSILONS:
        for max_pts in MAX_PTS:
            for backend_name in ("POT", "OTT"):
                combo_idx += 1
                log.info(
                    f"\n[{combo_idx}/{n_combos}] "
                    f"backend={backend_name} eps={epsilon:.0e} pts={max_pts}"
                )

                # Fresh backend per (eps, pts, backend) combo to ensure cold JIT
                backend = api["POTBackend"]() if backend_name == "POT" else api["OTTBackend"]()

                pair_times = []
                n_src_list = []
                n_tgt_list = []
                for pair_idx, (src_id, tgt_id, frame_idx, pair_label) in enumerate(PAIRS):
                    src_can = emb_frames[(src_id, frame_idx)]
                    tgt_can = emb_frames[(tgt_id, frame_idx)]

                    call_type = "COLD" if pair_idx == 0 else "WARM"
                    log.info(f"  pair {pair_idx} ({pair_label}) [{call_type}] …")
                    try:
                        elapsed, result_work = timed_solve(
                            src_can, tgt_can, epsilon, backend, max_pts, api
                        )
                        n_src = int(result_work.support_src_yx.shape[0]) if result_work.support_src_yx is not None else np.nan
                        n_tgt = int(result_work.support_tgt_yx.shape[0]) if result_work.support_tgt_yx is not None else np.nan
                        log.info(
                            f"    elapsed={elapsed:.2f}s  "
                            f"n_src={n_src}  n_tgt={n_tgt}  "
                            f"converged={result_work.diagnostics.get('converged', '?')}"
                        )
                    except Exception as exc:
                        log.error(f"    FAILED: {exc}", exc_info=True)
                        elapsed = np.nan
                        n_src = np.nan
                        n_tgt = np.nan

                    pair_times.append(elapsed)
                    n_src_list.append(n_src)
                    n_tgt_list.append(n_tgt)

                # Summarize cache info for OTT backend
                if backend_name == "OTT" and hasattr(backend, "cache_info"):
                    ci = backend.cache_info()
                    log.info(
                        f"  JIT cache: compiled {ci['n_entries']} bucket(s): {ci['shapes']}"
                    )

                cold_time = pair_times[0]
                warm_times = [t for t in pair_times[1:] if not np.isnan(t)]
                warm_time = float(np.mean(warm_times)) if warm_times else np.nan
                amortized_time = float(np.nanmean(pair_times)) if any(not np.isnan(t) for t in pair_times) else np.nan

                log.info(
                    f"  SUMMARY: cold={cold_time:.2f}s  "
                    f"warm={warm_time:.2f}s  "
                    f"amortized={amortized_time:.2f}s"
                )

                # Mean support size across pairs for reference
                mean_n_src = float(np.nanmean(n_src_list))
                mean_n_tgt = float(np.nanmean(n_tgt_list))

                records.append({
                    "backend": backend_name,
                    "epsilon": epsilon,
                    "max_support_pts": max_pts,
                    "cold_time_s": cold_time,
                    "warm_time_s": warm_time,
                    "amortized_time_s": amortized_time,
                    "pair0_time_s": pair_times[0],
                    "pair1_time_s": pair_times[1] if len(pair_times) > 1 else np.nan,
                    "pair2_time_s": pair_times[2] if len(pair_times) > 2 else np.nan,
                    "mean_n_src": mean_n_src,
                    "mean_n_tgt": mean_n_tgt,
                })

    df = pd.DataFrame(records)
    df.to_csv(BATCH_RESULTS_CSV, index=False)
    log.info(f"\nResults saved → {BATCH_RESULTS_CSV}  ({len(df)} rows)")

    # Print summary table
    cols = ["backend", "epsilon", "max_support_pts", "cold_time_s", "warm_time_s", "amortized_time_s"]
    log.info("\n=== Cold vs Warm Timing Summary ===\n" + df[cols].to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    eps_vals = sorted(df["epsilon"].unique())
    pts_vals = sorted(df["max_support_pts"].unique())
    backends = sorted(df["backend"].unique())

    colors = {"POT": "#3a7eca", "OTT": "#e86e1e"}

    fig, axes = plt.subplots(1, len(pts_vals), figsize=(7 * len(pts_vals), 5), sharey=False)
    if len(pts_vals) == 1:
        axes = [axes]

    for ax, max_pts in zip(axes, pts_vals):
        sub = df[df["max_support_pts"] == max_pts]
        n_eps = len(eps_vals)
        bar_width = 0.2
        x = np.arange(n_eps)

        for bi, be in enumerate(backends):
            be_sub = sub[sub["backend"] == be].set_index("epsilon")
            cold_times = [be_sub.loc[e, "cold_time_s"] if e in be_sub.index else np.nan for e in eps_vals]
            warm_times = [be_sub.loc[e, "warm_time_s"] if e in be_sub.index else np.nan for e in eps_vals]

            # Cold bars (hatched)
            offset_cold = x + (bi * 2 - (len(backends) - 1)) * bar_width - bar_width / 2
            offset_warm = offset_cold + bar_width

            bars_cold = ax.bar(
                offset_cold, cold_times, width=bar_width,
                label=f"{be} cold", color=colors.get(be, None), alpha=0.45, hatch="//",
            )
            bars_warm = ax.bar(
                offset_warm, warm_times, width=bar_width,
                label=f"{be} warm", color=colors.get(be, None), alpha=0.9,
            )

            for bars, times in [(bars_cold, cold_times), (bars_warm, warm_times)]:
                for bar, t in zip(bars, times):
                    if not np.isnan(t) and t > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.05,
                            f"{t:.1f}s",
                            ha="center", va="bottom", fontsize=7,
                        )

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:.0e}" for e in eps_vals])
        ax.set_xlabel("epsilon")
        ax.set_ylabel("time (s, log scale)")
        ax.set_title(f"max_pts = {max_pts}")
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        "OTT Cold (JIT compile) vs Warm (cached) vs POT\n"
        "Hatched = cold (first call); Solid = warm (subsequent calls)",
        fontsize=12,
    )
    fig.tight_layout()
    out = PLOTS_DIR / "warmup_effect.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = run_batch_benchmark()
    log.info("Generating plots …")
    make_plots(df)
    log.info("=== Batch benchmark complete ===")
    log.info(f"Results: {BATCH_RESULTS_CSV}")
    log.info(f"Plots:   {PLOTS_DIR}/warmup_effect.png")
