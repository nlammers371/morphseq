"""
06_full_nd2_scan.py
===================
Run the production zstack_motion_qc utility over an entire ND2 file,
in parallel using ProcessPoolExecutor.

For each (T, P) stack:
  - compute NCC grid (between-slice motion) and entropy grid
  - save as the primary artifact: 06_scan_output/grids/t<TTT>_p<PPP>.npz
  - derive scalar summaries and write stack_summaries.csv

On rerun, existing .npz files are reused; summaries are always recomputed
from the grids (CSV is a derived artifact, not authoritative).

Usage:
  # smoke test — first 3 timepoints, first 5 positions
  python 06_full_nd2_scan.py --t-limit 3 --p-limit 5

  # full scan (parallel, 16 workers)
  python 06_full_nd2_scan.py --workers 16
"""

from __future__ import annotations
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nd2

MORPHSEQ_ROOT = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(MORPHSEQ_ROOT))

from src.data_pipeline.quality_control.zstack_motion_qc import (
    compute_local_ncc_grid,
    compute_local_entropy_grid,
    ncc_stack_summary,
    entropy_stack_summary,
    save_grids,
    load_grids,
)

ND2_PATH   = MORPHSEQ_ROOT / "morphseq_playground/raw_image_data/YX1/20250912/20250912_WT_tricane_serial_dilution_experiment.nd2"
OUT_DIR    = Path(__file__).resolve().parent / "06_scan_output"
GRIDS_DIR  = OUT_DIR / "grids"
CSV_PATH   = OUT_DIR / "stack_summaries.csv"
LOOKUP_CSV = MORPHSEQ_ROOT / "docs/refactors/motion_blur_filtering_zstack/frame_nd2_lookup.csv"

TILE_SIZE        = 128
BAD_THRESH       = 0.90
CHECKPOINT_EVERY = 500   # write partial CSV every N completed stacks

LABEL_COLORS = {
    "Bad Images":  "#d62728",
    "Good Images": "#2ca02c",
    "Okay Images": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--t-start", type=int, default=0,
                   help="First timepoint to process (inclusive, default: 0)")
    p.add_argument("--t-end", type=int, default=None,
                   help="Last timepoint to process (exclusive, default: all)")
    p.add_argument("--t-limit", type=int, default=None,
                   help="Only process first N timepoints from t-start (smoke-test mode)")
    p.add_argument("--p-limit", type=int, default=None,
                   help="Only process first N positions (smoke-test mode)")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel worker processes (default: 8)")
    return p.parse_args()


# ── worker (runs in subprocess) ──────────────────────────────────────────────

def _process_stack(
    t: int,
    p: int,
    nd2_path: str,
    grids_dir: str,
    tile_size: int,
    bad_thresh: float,
) -> dict:
    """
    Load one (t, p) Z-stack, compute grids, save .npz, return summary dict.
    Each worker opens the ND2 file independently (nd2 handles are not
    picklable across processes).
    Returns timing breakdowns alongside summaries for bottleneck analysis.
    """
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")))

    from src.data_pipeline.quality_control.zstack_motion_qc import (
        compute_local_ncc_grid,
        compute_local_entropy_grid,
        ncc_stack_summary,
        entropy_stack_summary,
        save_grids,
        load_grids,
    )
    import nd2
    import numpy as np

    t0_total = time.perf_counter()
    npz_path = Path(grids_dir) / f"t{t:03d}_p{p:03d}.npz"

    t_load = t_compute = t_save = 0.0
    cache_hit = npz_path.exists()

    if cache_hit:
        t0 = time.perf_counter()
        data = load_grids(npz_path)
        t_load = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        with nd2.ND2File(nd2_path) as f:
            dask_arr = f.to_dask()
            stack = dask_arr[t, p].compute().astype(np.float32)  # (Z, Y, X)
        t_load = time.perf_counter() - t0

        t0 = time.perf_counter()
        ncc_grid     = compute_local_ncc_grid(stack, tile_size=tile_size)
        entropy_grid = compute_local_entropy_grid(stack, tile_size=tile_size)
        t_compute = time.perf_counter() - t0

        t0 = time.perf_counter()
        save_grids(
            npz_path,
            ncc_grid=ncc_grid,
            entropy_grid=entropy_grid,
            tile_size=tile_size,
            stride=tile_size,
            stack_shape_yx=(stack.shape[1], stack.shape[2]),
        )
        t_save = time.perf_counter() - t0
        data = {"ncc_grid": ncc_grid, "entropy_grid": entropy_grid}

    t_total = time.perf_counter() - t0_total
    timing = {
        "t_load_s":    round(t_load,    3),
        "t_compute_s": round(t_compute, 3),
        "t_save_s":    round(t_save,    3),
        "t_total_s":   round(t_total,   3),
        "cache_hit":   cache_hit,
    }
    return {"t": t, "p": p, **_summarize(data, bad_thresh), **timing}


def _summarize(data: dict, bad_thresh: float) -> dict:
    ncc_grid     = data["ncc_grid"]
    entropy_grid = data["entropy_grid"]

    ncc_s = ncc_stack_summary(ncc_grid, bad_thresh=bad_thresh)
    ent_s = entropy_stack_summary(entropy_grid)

    flat = ncc_grid.ravel()
    flat = flat[~np.isnan(flat)]
    extra = {
        "ncc_p05":    float(np.percentile(flat, 5)) if flat.size else float("nan"),
        "ncc_median": float(np.median(flat))         if flat.size else float("nan"),
    }
    return {**ncc_s, **ent_s, **extra}


# ── main scan ────────────────────────────────────────────────────────────────

def run_scan(
    t_start: int,
    t_end: int | None,
    t_limit: int | None,
    p_limit: int | None,
    workers: int,
    chunk_csv: Path | None = None,
) -> pd.DataFrame:
    GRIDS_DIR.mkdir(parents=True, exist_ok=True)

    with nd2.ND2File(str(ND2_PATH)) as f:
        dask_arr = f.to_dask()
        T_total, P_total = dask_arr.shape[0], dask_arr.shape[1]

    t_end_eff = T_total if t_end is None else min(t_end, T_total)
    t_end_eff = min(t_start + t_limit, t_end_eff) if t_limit else t_end_eff
    P = P_total if p_limit is None else min(p_limit, P_total)

    jobs = [(t, p) for t in range(t_start, t_end_eff) for p in range(P)]
    total = len(jobs)
    out_csv = chunk_csv if chunk_csv is not None else CSV_PATH
    import socket
    print(f"[{socket.gethostname()}] Scanning T={t_start}–{t_end_eff-1}/{T_total-1}, P=0–{P-1} → {total} stacks  (workers={workers})")

    rows: list[dict] = []
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_stack, t, p,
                str(ND2_PATH), str(GRIDS_DIR),
                TILE_SIZE, BAD_THRESH,
            ): (t, p)
            for t, p in jobs
        }

        for fut in as_completed(futures):
            t, p = futures[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                print(f"  ERROR t={t} p={p}: {exc}")
                rows.append({"t": t, "p": p,
                             "t_load_s": float("nan"), "t_compute_s": float("nan"),
                             "t_save_s": float("nan"), "t_total_s": float("nan"),
                             "cache_hit": False})

            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} stacks done")
            if done % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).sort_values(["t", "p"]).to_csv(out_csv, index=False)
                print(f"  checkpoint → {out_csv} ({done} rows)")

    df = pd.DataFrame(rows).sort_values(["t", "p"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV → {out_csv}  ({len(df)} rows)")
    _print_timing_report(df, workers)
    return df


def _dominant_phase(fresh: pd.DataFrame) -> str:
    phases = {
        "ND2 load (I/O-bound)":    fresh["t_load_s"].mean(),
        "grid compute (CPU-bound)": fresh["t_compute_s"].mean(),
        "npz save (write-bound)":  fresh["t_save_s"].mean(),
    }
    phases = {k: v for k, v in phases.items() if not np.isnan(v)}
    return max(phases, key=phases.get) if phases else "unknown"


def _print_timing_report(df: pd.DataFrame, workers: int) -> None:
    try:
        if "cache_hit" not in df.columns or "t_total_s" not in df.columns:
            print("\n[timing] No timing data available.")
            return
        cache_col = df["cache_hit"].fillna(False).astype(bool)
        fresh = df.loc[~cache_col].copy()
        for col in ["t_load_s", "t_compute_s", "t_save_s", "t_total_s"]:
            if col not in fresh.columns:
                fresh[col] = float("nan")
        fresh = fresh.dropna(subset=["t_total_s"])
        if fresh.empty:
            print("\n[timing] All stacks were cache hits — no fresh timing to report.")
            return

        print(f"\n=== Timing report ({len(fresh)} freshly computed stacks, {workers} workers) ===")
        print(f"  {'metric':<20}  {'mean':>7}  {'median':>7}  {'p95':>7}")
        print(f"  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*7}")
        for col, label in [
            ("t_load_s",    "ND2 load"),
            ("t_compute_s", "grid compute"),
            ("t_save_s",    "npz save"),
            ("t_total_s",   "total/stack"),
        ]:
            vals = fresh[col].dropna()
            if vals.empty:
                continue
            print(f"  {label:<20}  {vals.mean():>6.2f}s  {vals.median():>6.2f}s  {vals.quantile(0.95):>6.2f}s")

        total_wall = fresh["t_total_s"].dropna().sum()
        throughput = 60 * len(fresh) / (total_wall / workers) if total_wall > 0 else 0
        print(f"\n  Throughput (estimated): {throughput:.1f} stacks/min × {workers} workers")
        print(f"  Dominant phase: {_dominant_phase(fresh)}")
    except Exception as exc:
        print(f"\n[timing] Report failed: {exc}")


# ── viz ──────────────────────────────────────────────────────────────────────

def load_labeled_examples() -> pd.DataFrame | None:
    if not LOOKUP_CSV.exists():
        return None
    lk = pd.read_csv(LOOKUP_CSV)
    lk["p"] = lk["nd2_series_num"] - 1   # nd2_series_num is 1-based
    lk["t"] = lk["time_int"]
    return lk


def plot_scatter(df: pd.DataFrame, labeled: pd.DataFrame | None) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(df["ncc_min"], df["entropy_mean"], gridsize=50, cmap="Blues", mincnt=1)

    if labeled is not None:
        for category, grp in labeled.groupby("category"):
            color = LABEL_COLORS.get(category, "black")
            sub = df.merge(grp[["t", "p"]], on=["t", "p"], how="inner")
            if not sub.empty:
                ax.scatter(sub["ncc_min"], sub["entropy_mean"],
                           color=color, edgecolors="k", linewidths=0.5,
                           s=60, zorder=5, label=category)

    ax.axvline(0.85, color="red", linestyle="--", linewidth=1, label="NCC threshold (0.85)")
    ax.set_xlabel("ncc_min")
    ax.set_ylabel("entropy_mean")
    ax.set_title("Motion QC: NCC vs Entropy (all stacks)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "scatter_ncc_entropy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved scatter → {out}")


def plot_distributions(df: pd.DataFrame, labeled: pd.DataFrame | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col, thresh, title in [
        (axes[0], "ncc_min",      0.85, "Distribution of ncc_min"),
        (axes[1], "entropy_mean", None, "Distribution of entropy_mean"),
    ]:
        ax.hist(df[col].dropna(), bins=80, color="#4c72b0", edgecolor="none", alpha=0.8)
        if thresh is not None:
            ax.axvline(thresh, color="red", linestyle="--", linewidth=1.2,
                       label=f"threshold={thresh}")
            ax.legend(fontsize=8)

        if labeled is not None:
            for category, grp in labeled.groupby("category"):
                color = LABEL_COLORS.get(category, "black")
                sub = df.merge(grp[["t", "p"]], on=["t", "p"], how="inner")
                for val in sub[col].dropna():
                    ax.axvline(val, color=color, linewidth=0.8, alpha=0.6)

        ax.set_xlabel(col)
        ax.set_ylabel("count")
        ax.set_title(title)

    fig.tight_layout()
    out = OUT_DIR / "distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved distributions → {out}")


def print_exemplars(df: pd.DataFrame) -> None:
    ncc_q05 = df["ncc_min"].quantile(0.05)
    ncc_q95 = df["ncc_min"].quantile(0.95)
    ent_q05 = df["entropy_mean"].quantile(0.05)
    ent_q95 = df["entropy_mean"].quantile(0.95)

    cohorts = [
        ("Candidate GOOD (high NCC + high entropy)",
         df[(df["ncc_min"] >= ncc_q95) & (df["entropy_mean"] >= ent_q95)].head(5)),
        ("Candidate BAD — motion (low ncc_min)",
         df[df["ncc_min"] <= ncc_q05].head(5)),
        ("Candidate BAD — low entropy, ok NCC",
         df[(df["entropy_mean"] <= ent_q05) & (df["ncc_min"] > 0.85)].head(5)),
        ("Candidate EDGE (ncc_min in [0.80, 0.88])",
         df[(df["ncc_min"] > 0.80) & (df["ncc_min"] < 0.88)].head(5)),
    ]

    cols = ["t", "p", "ncc_min", "ncc_p05", "bad_pair_frac", "entropy_mean"]
    print("\n=== Candidate inspection cohorts (not final labels) ===")
    for label, sub in cohorts:
        print(f"\n--- {label} ---")
        if sub.empty:
            print("  (none)")
        else:
            print(sub[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    # per-chunk CSV if t-start is non-default (array job mode)
    chunk_csv = None
    if args.t_start > 0 or args.t_end is not None:
        label = f"t{args.t_start:03d}_t{(args.t_end or 999):03d}"
        chunk_csv = OUT_DIR / f"chunk_summaries_{label}.csv"
    df = run_scan(
        t_start=args.t_start,
        t_end=args.t_end,
        t_limit=args.t_limit,
        p_limit=args.p_limit,
        workers=args.workers,
        chunk_csv=chunk_csv,
    )
    # only generate figures on a full (non-chunked) run
    if chunk_csv is None:
        labeled = load_labeled_examples()
        plot_scatter(df, labeled)
        plot_distributions(df, labeled)
        print_exemplars(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
