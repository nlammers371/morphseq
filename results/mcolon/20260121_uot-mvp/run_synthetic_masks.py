#!/usr/bin/env python3
"""
UOT MVP: Synthetic Mask Sanity Checks

Creates simple synthetic masks (circle -> circle, circle -> oval)
and runs UOT to verify expected creation/destruction patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

from src.analyze.optimal_transport_morphometrics.uot_masks import run_uot_pair
from src.analyze.utils.optimal_transport import UOTConfig, UOTFramePair, UOTFrame, MassMode
from src.analyze.optimal_transport_morphometrics.uot_masks.viz import (
    plot_creation_destruction,
    plot_creation_destruction_overlay,
    plot_velocity_overlay,
    plot_transport_spectrum,
)


OUTPUT_DIR = Path("results/mcolon/20260121_uot-mvp/synthetic")
IMAGE_SHAPE = (256, 256)


def make_circle(shape: Tuple[int, int], center_yx: Tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask.astype(np.uint8)


def make_ellipse(shape: Tuple[int, int], center_yx: Tuple[int, int], radius_y: int, radius_x: int) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = center_yx
    mask = ((yy - cy) / float(radius_y)) ** 2 + ((xx - cx) / float(radius_x)) ** 2 <= 1.0
    return mask.astype(np.uint8)


def create_cases(shape: Tuple[int, int]) -> List[Dict]:
    cy, cx = shape[0] // 2, shape[1] // 2

    circle = make_circle(shape, (cy, cx), radius=40)
    circle_shift = make_circle(shape, (cy + 6, cx + 6), radius=40)
    circle_far = make_circle(shape, (cy + 90, cx + 90), radius=40)
    oval = make_ellipse(shape, (cy, cx), radius_y=30, radius_x=60)

    return [
        {
            "name": "circle_to_circle_shift",
            "src": circle,
            "tgt": circle_shift,
            "align_mode": "none",
        },
        {
            "name": "circle_to_circle_nonoverlap",
            "src": circle,
            "tgt": circle_far,
            "align_mode": "none",
        },
        {
            "name": "circle_to_oval",
            "src": circle,
            "tgt": oval,
            "align_mode": "none",
        },
    ]


def create_uot_config(
    mass_mode: MassMode,
    align_mode: str,
    epsilon: float,
    marginal_relaxation: float,
    metric: str,
    coord_scale: float,
) -> UOTConfig:
    return UOTConfig(
        epsilon=epsilon,
        marginal_relaxation=marginal_relaxation,
        downsample_factor=1,
        downsample_divisor=1,
        padding_px=0,
        mass_mode=mass_mode,
        align_mode=align_mode,
        max_support_points=5000,
        store_coupling=True,
        random_seed=42,
        metric=metric,
        coord_scale=coord_scale,
    )


def save_side_by_side(src_mask: np.ndarray, tgt_mask: np.ndarray, output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].imshow(src_mask, cmap="gray")
    axes[0].set_title("Source")
    axes[0].axis("off")

    axes[1].imshow(tgt_mask, cmap="gray")
    axes[1].set_title("Target")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_case(
    case: Dict,
    mass_mode: MassMode,
    epsilon: float,
    marginal_relaxation: float,
    metric: str,
    coord_scale: float,
    output_root: Path,
) -> Dict:
    output_dir = output_root / case["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    config = create_uot_config(
        mass_mode=mass_mode,
        align_mode=case["align_mode"],
        epsilon=epsilon,
        marginal_relaxation=marginal_relaxation,
        metric=metric,
        coord_scale=coord_scale,
    )

    pair = UOTFramePair(
        src=UOTFrame(embryo_mask=case["src"], meta={"case": case["name"], "role": "src"}),
        tgt=UOTFrame(embryo_mask=case["tgt"], meta={"case": case["name"], "role": "tgt"}),
        pair_meta={"case": case["name"]},
    )

    result = run_uot_pair(pair, config=config)
    pair.result = result

    save_side_by_side(case["src"], case["tgt"], output_dir / "side_by_side_masks.png", case["name"])

    fig = plot_creation_destruction(
        result.mass_created_hw,
        result.mass_destroyed_hw,
        output_path=str(output_dir / "creation_destruction.png"),
    )
    plt.close(fig)

    fig = plot_creation_destruction_overlay(
        case["src"],
        case["tgt"],
        result.mass_created_hw,
        result.mass_destroyed_hw,
        transform_meta=result.transform_meta,
        output_path=str(output_dir / "creation_destruction_overlay.png"),
    )
    plt.close(fig)

    fig = plot_velocity_overlay(
        result.mass_created_hw,
        result.velocity_field_yx_hw2,
        stride=6,
        output_path=str(output_dir / "velocity_field.png"),
    )
    plt.close(fig)

    if result.coupling is not None:
        fig = plot_transport_spectrum(
            result.coupling,
            result.support_src_yx,
            result.support_tgt_yx,
            output_path=str(output_dir / "transport_spectrum.png"),
        )
        plt.close(fig)

    metrics = result.diagnostics.get("metrics", {})
    backend = result.diagnostics.get("backend", {}) if result.diagnostics else {}
    m_src = float(backend.get("m_src", np.nan))
    m_tgt = float(backend.get("m_tgt", np.nan))
    created_mass = float(metrics.get("created_mass", np.nan))
    destroyed_mass = float(metrics.get("destroyed_mass", np.nan))
    created_frac = created_mass / m_tgt if m_tgt and m_tgt > 0 else np.nan
    destroyed_frac = destroyed_mass / m_src if m_src and m_src > 0 else np.nan
    metrics_dict = {
        "case": case["name"],
        "mass_mode": mass_mode.value,
        "metric": metric,
        "coord_scale": coord_scale,
        "epsilon": epsilon,
        "marginal_relaxation": marginal_relaxation,
        "cost": result.cost,
        "created_mass_fraction": created_frac,
        "destroyed_mass_fraction": destroyed_frac,
        **metrics,
    }

    print(
        f"  cost={metrics_dict.get('cost', float('nan')):.4f} "
        f"transported={metrics_dict.get('transported_mass', float('nan')):.4f} "
        f"created_frac={created_frac:.3f} destroyed_frac={destroyed_frac:.3f}"
    )

    return metrics_dict


def parse_mass_mode(value: str) -> MassMode:
    try:
        return MassMode(value)
    except ValueError as exc:
        raise ValueError(f"Invalid mass_mode: {value}. Options: {[m.value for m in MassMode]}") from exc


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run synthetic UOT mask sanity checks.")
    parser.add_argument(
        "--mass-mode",
        type=str,
        default=MassMode.UNIFORM.value,
        help=f"Mass mode: {[m.value for m in MassMode]}",
    )
    parser.add_argument(
        "--mass-modes",
        type=str,
        default="",
        help="Comma-separated mass modes to run (overrides --mass-mode).",
    )
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--marginal-relaxation", type=float, default=100.0)
    parser.add_argument(
        "--metric",
        type=str,
        default="sqeuclidean",
        choices=["sqeuclidean", "euclidean"],
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1.0 / max(IMAGE_SHAPE),
        help="Scale factor applied to coordinates before computing distances.",
    )
    args = parser.parse_args()

    if args.mass_modes:
        mass_modes = [parse_mass_mode(m.strip()) for m in args.mass_modes.split(",") if m.strip()]
    else:
        mass_modes = [parse_mass_mode(args.mass_mode)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = create_cases(IMAGE_SHAPE)

    all_metrics: List[Dict] = []
    for mass_mode in mass_modes:
        for case in cases:
            print(f"Running {case['name']} ({mass_mode.value})...")
            metrics_dict = run_case(
                case,
                mass_mode,
                args.epsilon,
                args.marginal_relaxation,
                args.metric,
                args.coord_scale,
                OUTPUT_DIR / mass_mode.value,
            )
            all_metrics.append(metrics_dict)

    if all_metrics:
        import pandas as pd

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    print("Done. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
