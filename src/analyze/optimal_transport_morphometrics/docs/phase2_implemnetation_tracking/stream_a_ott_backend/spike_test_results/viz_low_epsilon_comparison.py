#!/usr/bin/env python3
"""Viz comparison at eps=1e-5 and 1e-6: is OTT actually doing BETTER than POT?

POT gives near-zero cost at these epsilon values on canonical grid.
OTT gives ~18-34 cost. Let's visualize the actual fields to see which
result looks more physically reasonable.

USAGE:
    PYTHONPATH=src:$PYTHONPATH python src/analyze/optimal_transport_morphometrics/docs/phase2_implemnetation_tracking/stream_a_ott_backend/spike_test_results/viz_low_epsilon_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

morphseq_root = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(morphseq_root / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze.utils.optimal_transport import UOTConfig, UOTFramePair, POTBackend, MassMode
from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend
from analyze.optimal_transport_morphometrics.uot_masks.run_transport import run_uot_pair
from analyze.optimal_transport_morphometrics.uot_masks.frame_mask_io import load_mask_from_csv
from analyze.optimal_transport_morphometrics.uot_masks.viz import plot_uot_summary
import pandas as pd

CSV_PATH = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv")
EMBRYO_A = "20251113_A05_e01"
EMBRYO_B = "20251113_E04_e01"
TARGET_HPF = 48.0
STAGE_TOL = 1.0

CANONICAL_GRID_SHAPE = (256, 576)
UM_PER_PX = 10.0
COORD_SCALE = 1.0 / max(CANONICAL_GRID_SHAPE)
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
    print("Loading frames...")
    frame_a, stage_a = find_frame_at_stage(CSV_PATH, EMBRYO_A, TARGET_HPF, STAGE_TOL)
    frame_b, stage_b = find_frame_at_stage(CSV_PATH, EMBRYO_B, TARGET_HPF, STAGE_TOL)
    assert frame_a is not None and frame_b is not None

    src_frame = load_mask_from_csv(CSV_PATH, EMBRYO_A, frame_a)
    tgt_frame = load_mask_from_csv(CSV_PATH, EMBRYO_B, frame_b)
    pair = UOTFramePair(src=src_frame, tgt=tgt_frame)

    for eps in [1e-5, 1e-6]:
        print(f"\n{'='*60}")
        print(f"EPSILON = {eps:.0e}")
        print(f"{'='*60}")
        config = make_config(eps)

        print("  Running POT...")
        pot_result = run_uot_pair(pair, config=config, backend=POTBackend())
        print(f"  POT cost={pot_result.cost:.8f}")
        print(f"  POT vel max={np.linalg.norm(pot_result.velocity_px_per_frame_yx, axis=-1).max():.4f}")
        print(f"  POT destruction total={pot_result.mass_destroyed_px.sum():.6f}")
        print(f"  POT creation total={pot_result.mass_created_px.sum():.6f}")

        print("  Running OTT...")
        ott_result = run_uot_pair(pair, config=config, backend=OTTBackend())
        print(f"  OTT cost={ott_result.cost:.8f}")
        print(f"  OTT vel max={np.linalg.norm(ott_result.velocity_px_per_frame_yx, axis=-1).max():.4f}")
        print(f"  OTT destruction total={ott_result.mass_destroyed_px.sum():.6f}")
        print(f"  OTT creation total={ott_result.mass_created_px.sum():.6f}")

        eps_str = f"{eps:.0e}".replace("-", "m")

        plot_uot_summary(
            pot_result,
            output_path=str(OUTPUT_DIR / f"viz_pot_eps{eps_str}.png"),
            title=f"POT (CPU) — eps={eps:.0e}, cost={pot_result.cost:.6f}\n{EMBRYO_A} vs {EMBRYO_B}",
        )
        plt.close("all")

        plot_uot_summary(
            ott_result,
            output_path=str(OUTPUT_DIR / f"viz_ott_eps{eps_str}.png"),
            title=f"OTT (JAX) — eps={eps:.0e}, cost={ott_result.cost:.6f}\n{EMBRYO_A} vs {EMBRYO_B}",
        )
        plt.close("all")

        print(f"  Saved viz_pot_eps{eps_str}.png and viz_ott_eps{eps_str}.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
