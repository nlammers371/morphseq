"""
UOT Solver Benchmark (20260222)
================================
3 real embryo pairs × 3 epsilons × 2 backends × 2 support sizes = 36 combos.

Records: solve time, cost, mass created/destroyed, velocity.

Usage:
    PYTHONPATH=src python results/mcolon/20260222_uot_solver_benchmark/run_benchmark.py
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
# (morphseq_root for segmentation_sandbox; morphseq_root/src for analyze.*)
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
RESULTS_CSV = BENCH_DIR / "results.csv"

DATA_CSV = Path(
    "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = LOGS_DIR / f"benchmark_{ts}.log"

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
# Sweep configuration
# ---------------------------------------------------------------------------
EPSILONS = [1e-4, 1e-3, 1e-2]
MAX_PTS = [1000, 3000]
MARGINAL_RELAXATION = 10.0
DOWNSAMPLE_FACTOR = 4
COORD_SCALE = 1.0 / 576
CANONICAL_GRID_HW = (256, 576)
CANONICAL_UM_PER_PX = 10.0
RANDOM_SEED = 42

# 3 pairs: (src_embryo_id, tgt_embryo_id, frame_index, pair_label)
PAIRS = [
    ("20251113_E02_e01", "20251113_E04_e01", 88, "WT_vs_WT"),
    ("20251113_E04_e01", "20251113_A05_e01", 88, "WT_vs_HOM"),
    ("20251113_A05_e01", "20251113_B01_e01", 88, "HOM_vs_HOM"),
]


# ---------------------------------------------------------------------------
# Imports (deferred so PYTHONPATH errors surface with a clear message)
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
# Load mask helpers
# ---------------------------------------------------------------------------
def load_frame(csv_path: Path, embryo_id: str, frame_index: int, api):
    """Load a UOTFrame from the CSV."""
    frame = api["load_mask_from_csv"](csv_path, embryo_id, frame_index)
    um_per_px = frame.meta.get("um_per_pixel", np.nan) if frame.meta else np.nan
    if np.isnan(um_per_px):
        raise ValueError(f"No um_per_pixel for {embryo_id} frame {frame_index}")
    return frame, um_per_px


def canonicalize(mask: np.ndarray, um_per_px: float, api) -> object:
    """Canonicalize a raw mask."""
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
# Core solver
# ---------------------------------------------------------------------------
def solve_one(
    src_can,
    tgt_can,
    epsilon: float,
    backend,
    max_pts: int,
    api,
) -> tuple[object, float]:
    """Solve one UOT combo. Returns (result_canonical, elapsed_s)."""
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

    result = api["lift_work_result_to_canonical"](result_work, pair_work)
    return result, result_work, elapsed


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------
def extract_metrics(result, result_work, elapsed: float, pair_id: int, pair_label: str,
                    backend_name: str, epsilon: float, max_pts: int) -> dict:
    """Extract all benchmark metrics from a solve result."""
    cost = float(result.cost) if result.cost is not None else np.nan
    cost_is_nan = bool(np.isnan(cost))

    # Support sizes
    n_support_src = int(result_work.support_src_yx.shape[0]) if result_work.support_src_yx is not None else np.nan
    n_support_tgt = int(result_work.support_tgt_yx.shape[0]) if result_work.support_tgt_yx is not None else np.nan

    # OTT-specific diagnostics
    diag = result_work.diagnostics or {}
    converged = diag.get("converged", np.nan)
    n_iters = diag.get("n_iter", np.nan)
    if converged is not None and not isinstance(converged, float):
        try:
            converged = bool(converged)
        except Exception:
            converged = np.nan

    # Mass metrics from canonical grid
    created = result.mass_created_canon
    destroyed = result.mass_destroyed_canon
    total_mass = float(np.sum(created) + np.sum(destroyed))

    if total_mass > 0:
        created_mass_pct = float(np.sum(created)) / total_mass * 100.0
        destroyed_mass_pct = float(np.sum(destroyed)) / total_mass * 100.0
    else:
        created_mass_pct = np.nan
        destroyed_mass_pct = np.nan

    proportion_transported = 1.0 - (created_mass_pct + destroyed_mass_pct) / 100.0 if not np.isnan(created_mass_pct) else np.nan

    # Velocity metrics (canonical px/step)
    vel = result.velocity_canon_px_per_step_yx  # (H, W, 2)
    vel_mag = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
    mean_velocity_px = float(np.nanmean(vel_mag))
    max_velocity_px = float(np.nanmax(vel_mag))

    return {
        "pair_id": pair_id,
        "pair_label": pair_label,
        "backend": backend_name,
        "epsilon": epsilon,
        "max_support_pts": max_pts,
        "solve_time_s": elapsed,
        "cost": cost,
        "cost_is_nan": cost_is_nan,
        "n_support_src": n_support_src,
        "n_support_tgt": n_support_tgt,
        "converged": converged,
        "n_iters": n_iters,
        "created_mass_pct": created_mass_pct,
        "destroyed_mass_pct": destroyed_mass_pct,
        "proportion_transported": proportion_transported,
        "mean_velocity_px": mean_velocity_px,
        "max_velocity_px": max_velocity_px,
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark():
    log.info("=== UOT Solver Benchmark 20260222 ===")
    log.info(f"Log: {log_path}")

    api = _import_uot()
    log.info("Imports OK")

    # Backends: instantiate once (OTTBackend might JIT-compile on first use)
    pot_backend = api["POTBackend"]()
    ott_backend = api["OTTBackend"]()
    backends = [
        ("POT", pot_backend),
        ("OTT", ott_backend),
    ]

    # Pre-canonicalize all unique embryos (avoid redundant canonicalization)
    log.info("Pre-loading and canonicalizing embryo masks …")
    emb_frames: dict[tuple[str, int], tuple] = {}
    for src_id, tgt_id, frame_idx, _ in PAIRS:
        for emb_id in (src_id, tgt_id):
            key = (emb_id, frame_idx)
            if key not in emb_frames:
                log.info(f"  Loading {emb_id} frame {frame_idx}")
                frame, um = load_frame(DATA_CSV, emb_id, frame_idx, api)
                canon = canonicalize(frame.embryo_mask, um, api)
                emb_frames[key] = canon
    log.info("All masks canonicalized.")

    # Run sweep
    records = []
    n_total = len(PAIRS) * len(EPSILONS) * len(backends) * len(MAX_PTS)
    combo_idx = 0

    for pair_id, (src_id, tgt_id, frame_idx, pair_label) in enumerate(PAIRS):
        src_can = emb_frames[(src_id, frame_idx)]
        tgt_can = emb_frames[(tgt_id, frame_idx)]

        for epsilon in EPSILONS:
            for max_pts in MAX_PTS:
                for backend_name, backend in backends:
                    combo_idx += 1
                    desc = (
                        f"[{combo_idx}/{n_total}] "
                        f"pair={pair_label} eps={epsilon:.0e} "
                        f"pts={max_pts} backend={backend_name}"
                    )
                    log.info(f"Starting {desc}")
                    try:
                        result, result_work, elapsed = solve_one(
                            src_can, tgt_can, epsilon, backend, max_pts, api
                        )
                        row = extract_metrics(
                            result, result_work, elapsed,
                            pair_id, pair_label, backend_name, epsilon, max_pts,
                        )
                        log.info(
                            f"  DONE  time={elapsed:.2f}s  cost={row['cost']:.4g}  "
                            f"n_src={row['n_support_src']}  n_tgt={row['n_support_tgt']}"
                        )
                    except Exception as exc:
                        log.error(f"  FAILED: {exc}", exc_info=True)
                        row = {
                            "pair_id": pair_id,
                            "pair_label": pair_label,
                            "backend": backend_name,
                            "epsilon": epsilon,
                            "max_support_pts": max_pts,
                            "solve_time_s": np.nan,
                            "cost": np.nan,
                            "cost_is_nan": True,
                            "n_support_src": np.nan,
                            "n_support_tgt": np.nan,
                            "converged": np.nan,
                            "n_iters": np.nan,
                            "created_mass_pct": np.nan,
                            "destroyed_mass_pct": np.nan,
                            "proportion_transported": np.nan,
                            "mean_velocity_px": np.nan,
                            "max_velocity_px": np.nan,
                        }
                    records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    log.info(f"Results saved → {RESULTS_CSV}  ({len(df)} rows)")

    # Quick summary
    summary = df.groupby(["backend", "epsilon", "max_support_pts"])["solve_time_s"].mean()
    log.info("\n=== Timing Summary (mean over pairs) ===\n" + summary.to_string())

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    eps_vals = sorted(df["epsilon"].unique())
    pts_vals = sorted(df["max_support_pts"].unique())
    backend_vals = sorted(df["backend"].unique())
    pair_ids = sorted(df["pair_id"].unique())

    colors = {"POT": "#3a7eca", "OTT": "#e86e1e"}

    # ------------------------------------------------------------------
    # 1. solve_time_grid.png
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(pts_vals), figsize=(8 * len(pts_vals), 5), sharey=False)
    if len(pts_vals) == 1:
        axes = [axes]

    for ax, max_pts in zip(axes, pts_vals):
        sub = df[df["max_support_pts"] == max_pts]
        # Mean over pairs
        grp = sub.groupby(["backend", "epsilon"])["solve_time_s"].mean().reset_index()

        n_eps = len(eps_vals)
        n_be = len(backend_vals)
        bar_width = 0.35
        x = np.arange(n_eps)

        for bi, be in enumerate(backend_vals):
            be_grp = grp[grp["backend"] == be].set_index("epsilon")
            times = [be_grp.loc[e, "solve_time_s"] if e in be_grp.index else np.nan for e in eps_vals]
            offsets = x + (bi - (n_be - 1) / 2) * bar_width
            bars = ax.bar(offsets, times, width=bar_width, label=be, color=colors.get(be, None))
            # Annotate bars
            for bar, t in zip(bars, times):
                if not np.isnan(t):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.05,
                        f"{t:.1f}s",
                        ha="center", va="bottom", fontsize=8,
                    )

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{e:.0e}" for e in eps_vals])
        ax.set_xlabel("epsilon")
        ax.set_ylabel("solve time (s, log scale)")
        ax.set_title(f"max_pts = {max_pts}")
        ax.legend()
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    fig.suptitle("UOT Solve Time: POT vs OTT (mean over 3 pairs)", fontsize=13)
    fig.tight_layout()
    out = PLOTS_DIR / "solve_time_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Saved {out}")

    # ------------------------------------------------------------------
    # 2. cost_vs_epsilon.png
    # ------------------------------------------------------------------
    pair_labels = sorted(df["pair_label"].unique())
    fig, axes = plt.subplots(1, len(pair_labels), figsize=(6 * len(pair_labels), 4), sharey=False)
    if len(pair_labels) == 1:
        axes = [axes]

    line_styles = {1000: "--", 3000: "-"}
    for ax, pair_label in zip(axes, pair_labels):
        sub = df[df["pair_label"] == pair_label]
        for be in backend_vals:
            for pts in pts_vals:
                grp = sub[(sub["backend"] == be) & (sub["max_support_pts"] == pts)]
                grp = grp.sort_values("epsilon")
                valid = grp[~grp["cost"].isna()]
                if valid.empty:
                    continue
                ax.plot(
                    valid["epsilon"], valid["cost"],
                    marker="o",
                    linestyle=line_styles.get(pts, "-"),
                    color=colors.get(be, None),
                    label=f"{be}-{pts}",
                )

        ax.set_xscale("log")
        ax.set_xlabel("epsilon (log)")
        ax.set_ylabel("UOT cost")
        ax.set_title(pair_label)
        ax.legend(fontsize=8)

    fig.suptitle("UOT Cost vs Epsilon (by pair)", fontsize=13)
    fig.tight_layout()
    out = PLOTS_DIR / "cost_vs_epsilon.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Saved {out}")

    # ------------------------------------------------------------------
    # 3. quality_grid.png
    # ------------------------------------------------------------------
    metrics = ["created_mass_pct", "mean_velocity_px"]
    fig, axes = plt.subplots(len(metrics), len(pair_labels),
                             figsize=(5 * len(pair_labels), 4 * len(metrics)),
                             squeeze=False)

    for row_i, metric in enumerate(metrics):
        for col_i, pair_label in enumerate(pair_labels):
            ax = axes[row_i][col_i]
            sub = df[df["pair_label"] == pair_label]
            for be in backend_vals:
                for pts in pts_vals:
                    grp = sub[(sub["backend"] == be) & (sub["max_support_pts"] == pts)]
                    grp = grp.sort_values("epsilon")
                    invalid = grp[grp[metric].isna() | grp["cost_is_nan"]]
                    valid = grp[~grp[metric].isna() & ~grp["cost_is_nan"]]
                    if not valid.empty:
                        ax.plot(
                            valid["epsilon"], valid[metric],
                            marker="o",
                            linestyle=line_styles.get(pts, "-"),
                            color=colors.get(be, None),
                            label=f"{be}-{pts}",
                        )
                    # Flag NaN/diverged in red
                    if not invalid.empty:
                        ax.scatter(
                            invalid["epsilon"],
                            [0] * len(invalid),
                            marker="X", color="red", s=80, zorder=5,
                        )

            ax.set_xscale("log")
            ax.set_xlabel("epsilon")
            ax.set_ylabel(metric)
            if row_i == 0:
                ax.set_title(pair_label)
            if col_i == 0:
                ax.set_ylabel(metric)
            ax.legend(fontsize=7)

    fig.suptitle("Solution Quality Metrics vs Epsilon", fontsize=13)
    fig.tight_layout()
    out = PLOTS_DIR / "quality_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = run_benchmark()
    log.info("Generating plots …")
    make_plots(df)
    log.info("=== Benchmark complete ===")
    log.info(f"Results: {RESULTS_CSV}")
    log.info(f"Plots:   {PLOTS_DIR}/")
