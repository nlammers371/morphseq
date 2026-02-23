#!/usr/bin/env python3
"""
Spike test: POT vs OTT concordance on CANONICAL GRID with real embryo data.

Uses the same embryo pair as debug_uot_params.py (A05 vs E04 near 48 hpf).
Sweeps epsilon to find where OTT agrees with POT on canonical grid.

USAGE:
    PYTHONPATH=src:$PYTHONPATH python src/analyze/optimal_transport_morphometrics/docs/phase2_implemnetation_tracking/stream_a_ott_backend/spike_test_results/canonical_grid_epsilon_spike.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(morphseq_root / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze.utils.optimal_transport import UOTConfig, UOTFramePair, POTBackend, MassMode
from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend
from analyze.optimal_transport_morphometrics.uot_masks.run_transport import run_uot_pair
from analyze.optimal_transport_morphometrics.uot_masks.frame_mask_io import load_mask_from_csv

# === Config ===
CSV_PATH = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv")
EMBRYO_A = "20251113_A05_e01"
EMBRYO_B = "20251113_E04_e01"
TARGET_HPF = 48.0
STAGE_TOL = 1.0

CANONICAL_GRID_SHAPE = (256, 576)
UM_PER_PX = 10.0
COORD_SCALE = 1.0 / max(CANONICAL_GRID_SHAPE)

EPSILON_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
REG_M = 10.0

OUTPUT_DIR = Path(__file__).resolve().parent


def find_frame_at_stage(csv_path, embryo_id, target_hpf, tolerance_hpf):
    df = pd.read_csv(csv_path, usecols=["embryo_id", "frame_index", "predicted_stage_hpf"])
    subset = df[
        (df["embryo_id"] == embryo_id) &
        (df["predicted_stage_hpf"] >= target_hpf - tolerance_hpf) &
        (df["predicted_stage_hpf"] <= target_hpf + tolerance_hpf)
    ]
    if subset.empty:
        return None, None
    subset = subset.copy()
    subset["dist"] = (subset["predicted_stage_hpf"] - target_hpf).abs()
    closest = subset.loc[subset["dist"].idxmin()]
    return int(closest["frame_index"]), float(closest["predicted_stage_hpf"])


def make_config(epsilon):
    return UOTConfig(
        epsilon=epsilon,
        marginal_relaxation=REG_M,
        downsample_factor=1,
        downsample_divisor=1,
        padding_px=16,
        mass_mode=MassMode.UNIFORM,
        align_mode="none",
        max_support_points=5000,
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=COORD_SCALE,
        use_pair_frame=True,
        use_canonical_grid=True,
        canonical_grid_um_per_pixel=UM_PER_PX,
        canonical_grid_shape_hw=CANONICAL_GRID_SHAPE,
        canonical_grid_align_mode="yolk",
        canonical_grid_center_mode="joint_centering",
    )


def main():
    print("=" * 70)
    print("CANONICAL GRID EPSILON SPIKE: POT vs OTT")
    print("=" * 70)

    # Find frames
    frame_a, stage_a = find_frame_at_stage(CSV_PATH, EMBRYO_A, TARGET_HPF, STAGE_TOL)
    frame_b, stage_b = find_frame_at_stage(CSV_PATH, EMBRYO_B, TARGET_HPF, STAGE_TOL)
    if frame_a is None or frame_b is None:
        print(f"Could not find frames near {TARGET_HPF} hpf")
        return

    print(f"Embryo A: {EMBRYO_A} frame={frame_a} stage={stage_a:.1f} hpf")
    print(f"Embryo B: {EMBRYO_B} frame={frame_b} stage={stage_b:.1f} hpf")
    print(f"Canonical grid: {CANONICAL_GRID_SHAPE}, coord_scale={COORD_SCALE:.6f}")
    print(f"reg_m={REG_M}, epsilon sweep: {EPSILON_GRID}")
    print()

    # Load frames
    src_frame = load_mask_from_csv(CSV_PATH, EMBRYO_A, frame_a)
    tgt_frame = load_mask_from_csv(CSV_PATH, EMBRYO_B, frame_b)
    pair = UOTFramePair(src=src_frame, tgt=tgt_frame)

    rows = []
    for eps in EPSILON_GRID:
        config = make_config(eps)
        tau = REG_M / (REG_M + eps)
        print(f"epsilon={eps:.0e}  (tau={tau:.8f})")

        # POT
        try:
            t0 = time.time()
            pot_result = run_uot_pair(pair, config=config, backend=POTBackend())
            pot_time = time.time() - t0
            pot_cost = pot_result.cost
            pot_ok = not np.isnan(pot_cost)
            print(f"  POT: cost={pot_cost:.6f}  time={pot_time:.1f}s  ok={pot_ok}")
        except Exception as e:
            print(f"  POT: ERROR {e}")
            pot_cost = np.nan
            pot_time = np.nan
            pot_ok = False

        # OTT
        try:
            t0 = time.time()
            ott_result = run_uot_pair(pair, config=config, backend=OTTBackend())
            ott_time = time.time() - t0
            ott_cost = ott_result.cost
            ott_ok = not np.isnan(ott_cost)
            ott_converged = ott_result.diagnostics.get("backend", {}).get("converged", None)
            print(f"  OTT: cost={ott_cost:.6f}  time={ott_time:.1f}s  ok={ott_ok}  converged={ott_converged}")
        except Exception as e:
            print(f"  OTT: ERROR {e}")
            ott_cost = np.nan
            ott_time = np.nan
            ott_ok = False
            ott_converged = None

        # Concordance
        if pot_ok and ott_ok:
            cost_ratio = ott_cost / pot_cost if pot_cost > 0 else np.nan
            cost_pct_diff = abs(ott_cost - pot_cost) / max(abs(pot_cost), 1e-12) * 100
            print(f"  CONCORDANCE: ratio={cost_ratio:.4f}  pct_diff={cost_pct_diff:.2f}%")
        else:
            cost_ratio = np.nan
            cost_pct_diff = np.nan

        rows.append({
            "epsilon": eps,
            "tau": tau,
            "reg_m": REG_M,
            "coord_scale": COORD_SCALE,
            "pot_cost": pot_cost,
            "pot_time_s": pot_time,
            "pot_ok": pot_ok,
            "ott_cost": ott_cost,
            "ott_time_s": ott_time,
            "ott_ok": ott_ok,
            "ott_converged": ott_converged,
            "cost_ratio": cost_ratio,
            "cost_pct_diff": cost_pct_diff,
        })
        print()

    # Save results
    df = pd.DataFrame(rows)
    csv_out = OUTPUT_DIR / "canonical_grid_epsilon_results.csv"
    df.to_csv(csv_out, index=False)
    print(f"Results saved to {csv_out}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    valid = df[df["pot_ok"] & df["ott_ok"]]

    # Panel 1: Cost comparison
    ax = axes[0]
    ax.plot(valid["epsilon"], valid["pot_cost"], "o-", label="POT (CPU)")
    ax.plot(valid["epsilon"], valid["ott_cost"], "s--", label="OTT (JAX)")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Transport cost")
    ax.set_title("Cost: POT vs OTT")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: % difference
    ax = axes[1]
    ax.plot(valid["epsilon"], valid["cost_pct_diff"], "o-", color="red")
    ax.axhline(5, ls="--", color="gray", label="5% threshold")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Cost % difference")
    ax.set_title("Concordance (lower = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Timing
    ax = axes[2]
    ax.plot(valid["epsilon"], valid["pot_time_s"], "o-", label="POT")
    ax.plot(valid["epsilon"], valid["ott_time_s"], "s--", label="OTT")
    ax.set_xscale("log")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Solve Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Canonical Grid Epsilon Spike: {EMBRYO_A} vs {EMBRYO_B}\n"
                 f"grid={CANONICAL_GRID_SHAPE}, coord_scale={COORD_SCALE:.4f}, reg_m={REG_M}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "canonical_grid_epsilon_spike.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {OUTPUT_DIR / 'canonical_grid_epsilon_spike.png'}")


if __name__ == "__main__":
    main()
