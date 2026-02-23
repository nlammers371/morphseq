#!/usr/bin/env python3
"""Run compact POT-vs-OTT concordance checks on known control cases.

Control cases:
1) Cross-embryo known pair (A05 -> E04 near 48 hpf)
2) Identity pair (A05 -> A05 same frame) for near-zero sanity
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[7]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.utils.optimal_transport import MassMode, UOTConfig, UOTFramePair
from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend
from analyze.utils.optimal_transport.backends.pot_backend import POTBackend
from analyze.optimal_transport_morphometrics.uot_masks.frame_mask_io import load_mask_from_csv
from analyze.optimal_transport_morphometrics.uot_masks.run_transport import run_uot_pair


DEFAULT_CSV = (
    PROJECT_ROOT
    / "results"
    / "mcolon"
    / "20251229_cep290_phenotype_extraction"
    / "final_data"
    / "embryo_data_with_labels.csv"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent


def make_config(epsilon: float, reg_m: float, max_support_points: int) -> UOTConfig:
    shape = (256, 576)
    return UOTConfig(
        epsilon=epsilon,
        marginal_relaxation=reg_m,
        downsample_factor=1,
        downsample_divisor=1,
        padding_px=16,
        mass_mode=MassMode.UNIFORM,
        align_mode="none",
        max_support_points=int(max_support_points),
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=1.0 / max(shape),
        use_pair_frame=True,
        use_canonical_grid=True,
        canonical_grid_um_per_pixel=10.0,
        canonical_grid_shape_hw=shape,
        canonical_grid_align_mode="yolk",
        canonical_grid_center_mode="joint_centering",
    )


def _vel_stats(result) -> Dict[str, float]:
    speed = np.linalg.norm(np.asarray(result.velocity_px_per_frame_yx), axis=-1)
    return {
        "vel_p50": float(np.percentile(speed, 50)),
        "vel_p90": float(np.percentile(speed, 90)),
        "vel_p99": float(np.percentile(speed, 99)),
    }


def _mass_metrics(result) -> Dict[str, float]:
    metrics = result.diagnostics.get("metrics", {}) if result.diagnostics else {}
    return {
        "created_mass_pct": float(metrics.get("created_mass_pct", np.nan)),
        "destroyed_mass_pct": float(metrics.get("destroyed_mass_pct", np.nan)),
        "proportion_transported": float(metrics.get("proportion_transported", np.nan)),
    }


def run_case(
    *,
    case_name: str,
    pair: UOTFramePair,
    epsilon: float,
    reg_m: float,
    max_support_points: int,
) -> List[Dict]:
    rows: List[Dict] = []
    cfg = make_config(epsilon=epsilon, reg_m=reg_m, max_support_points=max_support_points)
    for backend_name, backend in (("POT", POTBackend()), ("OTT", OTTBackend())):
        t0 = time.time()
        result = run_uot_pair(pair, config=cfg, backend=backend)
        elapsed = time.time() - t0
        rows.append(
            {
                "case_name": case_name,
                "backend": backend_name,
                "epsilon": float(epsilon),
                "reg_m": float(reg_m),
                "runtime_sec": float(elapsed),
                "cost": float(result.cost),
                **_vel_stats(result),
                **_mass_metrics(result),
            }
        )
    return rows


def main(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    data_root = PROJECT_ROOT / "morphseq_playground"
    src = load_mask_from_csv(csv_path, args.control_src_embryo, args.control_src_frame, data_root=data_root)
    tgt = load_mask_from_csv(csv_path, args.control_tgt_embryo, args.control_tgt_frame, data_root=data_root)

    rows: List[Dict] = []
    eps_grid = [float(x) for x in args.epsilon_grid.split(",") if x.strip()]
    for eps in eps_grid:
        rows.extend(
            run_case(
                case_name="cross_embryo_control",
                pair=UOTFramePair(src=src, tgt=tgt),
                epsilon=eps,
                reg_m=args.reg_m,
                max_support_points=args.max_support_points,
            )
        )
        rows.extend(
            run_case(
                case_name="identity_control",
                pair=UOTFramePair(src=src, tgt=src),
                epsilon=eps,
                reg_m=args.reg_m,
                max_support_points=args.max_support_points,
            )
        )

    df = pd.DataFrame(rows).sort_values(["case_name", "epsilon", "backend"]).reset_index(drop=True)
    pivot = df.pivot_table(index=["case_name", "epsilon"], columns="backend", values="cost")
    if {"POT", "OTT"} <= set(pivot.columns):
        pivot = pivot.reset_index()
        pivot["cost_pct_diff"] = (
            np.abs(pivot["OTT"] - pivot["POT"]) / np.maximum(np.abs(pivot["POT"]), 1e-12) * 100.0
        )
    else:
        pivot = pd.DataFrame()

    csv_out = output_root / "control_concordance_results.csv"
    df.to_csv(csv_out, index=False)
    if not pivot.empty:
        pivot_out = output_root / "control_concordance_summary.csv"
        pivot.to_csv(pivot_out, index=False)
    else:
        pivot_out = None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for case_name, g in df.groupby("case_name"):
        for backend_name, gg in g.groupby("backend"):
            axes[0].plot(gg["epsilon"], gg["cost"], marker="o", label=f"{case_name}:{backend_name}")
            axes[1].plot(gg["epsilon"], gg["runtime_sec"], marker="o", label=f"{case_name}:{backend_name}")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_title("Control Cases: Cost vs Epsilon")
    axes[1].set_title("Control Cases: Runtime vs Epsilon")
    axes[0].set_xlabel("epsilon")
    axes[1].set_xlabel("epsilon")
    axes[0].set_ylabel("cost")
    axes[1].set_ylabel("runtime_sec")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    plot_out = output_root / "control_concordance_plot.png"
    fig.savefig(plot_out, dpi=180)
    plt.close(fig)

    print(f"Wrote: {csv_out}")
    if pivot_out is not None:
        print(f"Wrote: {pivot_out}")
    print(f"Wrote: {plot_out}")
    print(df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact POT-vs-OTT control concordance checks.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--control-src-embryo", type=str, default="20251113_A05_e01")
    parser.add_argument("--control-tgt-embryo", type=str, default="20251113_E04_e01")
    parser.add_argument("--control-src-frame", type=int, default=14)
    parser.add_argument("--control-tgt-frame", type=int, default=14)
    parser.add_argument("--epsilon-grid", type=str, default="1e-5,1e-4")
    parser.add_argument("--reg-m", type=float, default=10.0)
    parser.add_argument("--max-support-points", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
