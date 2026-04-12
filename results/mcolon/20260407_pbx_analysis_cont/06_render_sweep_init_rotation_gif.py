from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260407_condensation_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("NUMBA_CACHE_LOCATOR_CLASSES", "UserProvidedCacheLocator")
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from analyze.trajectory_condensation import viz as tc_viz
from analyze.trajectory_condensation.viz.animation import _draw_stacked_3d, _save_animation

from common import GENOTYPE_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a rotating stacked-3D comparison GIF with the initialized state as the left panel."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Elastic sweep output directory containing elastic_strength_sweep/",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        required=True,
        help="Elastic strengths to render, in panel order.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="compare_stacked_3d_with_init_rotation.gif",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--n-frames", type=int, default=120)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim-start", type=float, default=-60.0)
    parser.add_argument("--azim-end", type=float, default=300.0)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument(
        "--figscale",
        type=float,
        default=0.75,
        help="Scale factor applied to the base figure width/height.",
    )
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument(
        "--static-name",
        type=str,
        default="compare_stacked_3d_with_init.png",
    )
    return parser.parse_args()


def _strength_tag(value: float) -> str:
    return str(value).replace(".", "p")


def _finite_xy_bounds(arrays: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    x_vals: list[np.ndarray] = []
    y_vals: list[np.ndarray] = []
    for arr in arrays:
        flat = arr.reshape(-1, 2)
        keep = np.isfinite(flat).all(axis=1)
        if keep.any():
            x_vals.append(flat[keep, 0])
            y_vals.append(flat[keep, 1])
    x_all = np.concatenate(x_vals)
    y_all = np.concatenate(y_vals)
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    x_pad = 0.05 * max(x_max - x_min, 1.0)
    y_pad = 0.05 * max(y_max - y_min, 1.0)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    sweep_dir = args.run_dir / "elastic_strength_sweep"
    run_paths = [
        sweep_dir / f"quadratic_plus_outlier_elastic_{_strength_tag(strength)}" / "condensed_positions.npz"
        for strength in args.strengths
    ]
    runs = [
        tc_viz.load_run(path, title=f"quadratic_plus_outlier | {strength:g}", color_map=GENOTYPE_COLORS)
        for path, strength in zip(run_paths, args.strengths)
    ]

    base_run = runs[0]
    if base_run.x0 is None:
        raise ValueError("Run does not contain x0; cannot render init comparison GIF.")

    xlim, ylim = _finite_xy_bounds([base_run.x0] + [run.positions for run in runs])
    zlim = (float(np.min(base_run.time_values)), float(np.max(base_run.time_values)))

    n_cols = 1 + len(runs)
    fig = plt.figure(figsize=(5.5 * n_cols * args.figscale, 6.5 * args.figscale))
    axes = [fig.add_subplot(1, n_cols, idx + 1, projection="3d") for idx in range(n_cols)]

    panel_specs = [("Initialized", base_run.x0)] + [(run.title, run.positions) for run in runs]
    subtitle = "z = time (hpf) — visualization metadata, not a learned coordinate"

    for ax, (title, positions) in zip(axes, panel_specs):
        _draw_stacked_3d(
            ax,
            positions,
            base_run.mask,
            base_run.time_values,
            base_run.labels,
            GENOTYPE_COLORS,
            point_size=8.0,
            alpha_point=0.6,
            alpha_line=0.2,
            linewidth=0.6,
            min_obs=2,
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=8)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

    fig.tight_layout()

    if args.static_only:
        for ax in axes:
            ax.view_init(elev=args.elev, azim=args.azim_start)
        output_path = args.run_dir / "elastic_strength_sweep" / args.static_name
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(output_path)
        return

    azimuths = np.linspace(args.azim_start, args.azim_end, args.n_frames)

    def update(frame: int):
        azim = float(azimuths[frame])
        for ax in axes:
            ax.view_init(elev=args.elev, azim=azim)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=args.n_frames,
        interval=1000 / args.fps,
        blit=False,
    )
    output_path = args.run_dir / "elastic_strength_sweep" / args.output_name
    _save_animation(anim, output_path, fps=args.fps, dpi=args.dpi)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
