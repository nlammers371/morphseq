from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260407_subset_gif_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.trajectory_condensation.viz.animation import _draw_stacked_3d, _resolve_color_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render synchronized 2x3 subset-condensation rotation GIF.")
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-frames", type=int, default=160)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim-start", type=float, default=-60.0)
    parser.add_argument("--azim-end", type=float, default=300.0)
    return parser.parse_args()


def _load_run(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path / "condensed_positions.npz", allow_pickle=True)
    return {
        "positions": np.asarray(payload["positions"], dtype=float),
        "mask": np.asarray(payload["mask"], dtype=bool),
        "time_values": np.asarray(payload["time_values"], dtype=float),
        "labels": np.asarray(payload["labels"], dtype=object),
    }


def _axis_limits(runs: list[dict[str, np.ndarray]]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    all_xy = []
    all_t = []
    for run in runs:
        pts = run["positions"][run["mask"]]
        all_xy.append(pts)
        all_t.append(run["time_values"])
    xy = np.vstack(all_xy)
    times = np.concatenate(all_t)
    xmin, xmax = float(np.nanmin(xy[:, 0])), float(np.nanmax(xy[:, 0]))
    ymin, ymax = float(np.nanmin(xy[:, 1])), float(np.nanmax(xy[:, 1]))
    zmin, zmax = float(np.nanmin(times)), float(np.nanmax(times))
    xpad = 0.05 * max(xmax - xmin, 1e-6)
    ypad = 0.05 * max(ymax - ymin, 1e-6)
    zpad = 0.02 * max(zmax - zmin, 1e-6)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad), (zmin - zpad, zmax + zpad)


def main() -> None:
    args = parse_args()
    variants = [("shrunk", "Weighted"), ("raw", "Unweighted")]
    subsets = [
        ("all_5class", "All 5"),
        ("all_except_wikab", "No wik_ab"),
        ("all_except_injctrl", "No inj_ctrl"),
    ]

    loaded = {}
    for variant, _ in variants:
        for subset, _ in subsets:
            loaded[(variant, subset)] = _load_run(args.matrix_root / variant / subset)

    xlim, ylim, zlim = _axis_limits(list(loaded.values()))
    fig = plt.figure(figsize=(18, 11))
    axes = []

    for r, (variant, row_title) in enumerate(variants, start=1):
        for c, (subset, col_title) in enumerate(subsets, start=1):
            ax = fig.add_subplot(2, 3, (r - 1) * 3 + c, projection="3d")
            run = loaded[(variant, subset)]
            color_map = _resolve_color_map(run["labels"], None)
            _draw_stacked_3d(
                ax,
                run["positions"],
                run["mask"],
                run["time_values"],
                run["labels"],
                color_map,
                point_size=8.0,
                alpha_point=0.6,
                alpha_line=0.2,
                linewidth=0.6,
                min_obs=2,
            )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.set_title(f"{row_title} | {col_title}\nz = time (hpf)", fontsize=8)
            axes.append(ax)

    fig.tight_layout()
    azimuths = np.linspace(float(args.azim_start), float(args.azim_end), int(args.n_frames))

    def update(frame: int):
        azim = azimuths[frame]
        for ax in axes:
            ax.view_init(elev=float(args.elev), azim=float(azim))
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=int(args.n_frames),
        interval=1000 / int(args.fps),
        blit=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.output), writer="pillow", fps=int(args.fps), dpi=int(args.dpi))
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
