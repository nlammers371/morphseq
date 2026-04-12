"""
Quick benchmark: test n_jobs scaling for run_classification.
Tests n_jobs = 1, 4, 8, 24 on a fixed 10 hpf window.

Each requested ``n_jobs`` value is benchmarked in a fresh subprocess so
worker-pool warmup and interpreter state do not contaminate later timings.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "src"))

from analyze.classification import run_classification


def normalize_genotype(g):
    g = str(g).strip().lower().replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    if g in ("ab_inj_ctrl", "wik-ab_inj_ctrl", "wik-ab_ctrl_inj", "wik_ab_inj_ctrl", "wik_ab_ctrl_inj"):
        return "inj_ctrl"
    return g.replace("wik-ab", "wik_ab")


def choose_hpf_window_bounds(stage_values: pd.Series, window_width: float, bin_width: float) -> tuple[float, float]:
    """Pick a centered, bin-aligned HPF window from the available stages."""
    clean = stage_values.dropna().astype(float)
    if clean.empty:
        raise ValueError("No valid predicted_stage_hpf values found.")

    stage_min = float(clean.min())
    stage_max = float(clean.max())
    if stage_max - stage_min <= window_width:
        return stage_min, stage_max

    center = float(clean.median())
    start = bin_width * math.floor((center - window_width / 2.0) / bin_width)
    end = start + window_width

    if start < stage_min:
        start = bin_width * math.floor(stage_min / bin_width)
        end = start + window_width

    if end > stage_max:
        start = bin_width * math.floor((stage_max - window_width) / bin_width)
        end = start + window_width

    return float(start), float(end)


BIN_WIDTH = 2.0
WINDOW_WIDTH_HPF = 10.0
N_PERMUTATIONS = 500
BENCHMARK_N_JOBS = [1, 4, 8, 24]
POSITIVE = "pbx1b_crispant"
NEGATIVE = "inj_ctrl"


def load_benchmark_dataframe() -> tuple[pd.DataFrame, dict]:
    frames = []
    for exp_id in ["20260304", "20260306"]:
        part = pd.read_csv(
            project_root / "morphseq_playground/metadata/build06_output" / f"df03_final_output_with_latents_{exp_id}.csv",
            low_memory=False,
        )
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    df["genotype"] = df["genotype"].fillna("unknown").map(normalize_genotype)

    window_start_hpf, window_end_hpf = choose_hpf_window_bounds(
        df["predicted_stage_hpf"],
        window_width=WINDOW_WIDTH_HPF,
        bin_width=BIN_WIDTH,
    )
    df = df[
        df["predicted_stage_hpf"].notna()
        & (df["predicted_stage_hpf"] >= window_start_hpf)
        & (df["predicted_stage_hpf"] < window_end_hpf)
    ].copy()
    window_bins = sorted((df["predicted_stage_hpf"] // BIN_WIDTH * BIN_WIDTH).astype(int).unique())

    meta = {
        "window_start_hpf": window_start_hpf,
        "window_end_hpf": window_end_hpf,
        "window_bins": window_bins,
        "positive": POSITIVE,
        "negative": NEGATIVE,
        "n_rows": len(df),
        "n_embryos": int(df["embryo_id"].nunique()),
        "genotypes": df["genotype"].unique().tolist(),
    }
    return df, meta


def run_single_benchmark(n_jobs: int) -> float:
    df, _ = load_benchmark_dataframe()
    t0 = time.time()
    run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        positive=POSITIVE,
        negative=NEGATIVE,
        features={"embedding": "z_mu_b"},
        n_jobs=n_jobs,
        n_permutations=N_PERMUTATIONS,
        bin_width=BIN_WIDTH,
        min_samples_per_group=3,
        verbose=True,
    )
    return time.time() - t0


def run_parent() -> None:
    import multiprocessing

    df, meta = load_benchmark_dataframe()
    print(f"CPUs available (os.cpu_count): {os.cpu_count()}")
    print(f"CPUs available (multiprocessing): {multiprocessing.cpu_count()}")
    print(f"Data: {meta['n_rows']} rows, {meta['n_embryos']} embryos")
    print(
        f"Window: {meta['window_start_hpf']:.1f}-{meta['window_end_hpf']:.1f} hpf "
        f"(width={WINDOW_WIDTH_HPF:.1f}, bin_width={BIN_WIDTH:.1f})"
    )
    print(f"Time bins: {meta['window_bins']} (n={len(meta['window_bins'])})")
    print(f"Genotypes: {df['genotype'].unique()}")
    print(f"Positive: {POSITIVE}")
    print(f"Negative: {NEGATIVE}")
    print(f"n_permutations={N_PERMUTATIONS}, embedding only")
    print("Benchmark mode: fresh subprocess per n_jobs\n")

    results = []
    script_path = Path(__file__).resolve()
    for n_jobs in BENCHMARK_N_JOBS:
        effective = os.cpu_count() if n_jobs == -1 else n_jobs
        print(f"Testing n_jobs={n_jobs} (effective={effective})...", flush=True)
        cmd = [
            sys.executable,
            str(script_path),
            "--child-run",
            "--n-jobs",
            str(n_jobs),
        ]
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")

        lines = [line for line in completed.stdout.splitlines() if line.startswith("BENCHMARK_RESULT_JSON=")]
        if not lines:
            raise RuntimeError(f"Child benchmark for n_jobs={n_jobs} did not emit BENCHMARK_RESULT_JSON")
        payload = json.loads(lines[-1].split("=", 1)[1])
        elapsed = float(payload["elapsed_seconds"])
        results.append((n_jobs, elapsed))
        print(f"  n_jobs={n_jobs}: {elapsed:.1f}s")

    print("\n=== Summary ===")
    baseline = results[0][1]
    for n_jobs, elapsed in results:
        speedup = baseline / elapsed
        print(f"  n_jobs={n_jobs:3d}: {elapsed:.1f}s  ({speedup:.1f}x speedup)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    if args.child_run:
        if args.n_jobs is None:
            raise ValueError("--child-run requires --n-jobs")
        elapsed = run_single_benchmark(args.n_jobs)
        print(
            "BENCHMARK_RESULT_JSON="
            + json.dumps({"n_jobs": int(args.n_jobs), "elapsed_seconds": float(elapsed)})
        )
        return

    run_parent()


if __name__ == "__main__":
    main()
