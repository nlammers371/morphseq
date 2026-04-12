from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path('/tmp') / 'morphseq_20260407_subset_matrix_quick_cache'
    os.environ.setdefault('MPLCONFIGDIR', str(cache_root / 'matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(cache_root / 'xdg'))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(cache_root / 'numba'))
    os.environ.setdefault('NUMBA_CACHE_LOCATOR_CLASSES', 'UserProvidedCacheLocator')
    for name in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'NUMBA_CACHE_DIR'):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

import numpy as np

from analyze.trajectory_condensation import viz as tc_viz
from analyze.trajectory_condensation.viz.animation import _draw_stacked_3d, _save_animation

from common import GENOTYPE_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render a subset-condensation 2x3 comparison using the standard stacked-3D viz path.'
    )
    parser.add_argument('--matrix-root', type=Path, required=True)
    parser.add_argument('--output-name', type=str, default='subset_matrix_rotation_quick.gif')
    parser.add_argument('--static-only', action='store_true')
    parser.add_argument('--static-name', type=str, default='subset_matrix_static_quick.png')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--n-frames', type=int, default=72)
    parser.add_argument('--elev', type=float, default=25.0)
    parser.add_argument('--azim-start', type=float, default=-60.0)
    parser.add_argument('--azim-end', type=float, default=300.0)
    parser.add_argument('--dpi', type=int, default=100)
    parser.add_argument('--figscale', type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    variants = [('shrunk', 'Weighted'), ('raw', 'Unweighted')]
    subsets = [('all_5class', 'All 5'), ('all_except_wikab', 'No wik_ab'), ('all_except_injctrl', 'No inj_ctrl')]

    run_grid: list[list[tc_viz.RunDescriptor]] = []
    for variant, row_label in variants:
        row: list[tc_viz.RunDescriptor] = []
        for subset, col_label in subsets:
            path = args.matrix_root / variant / subset / 'condensed_positions.npz'
            row.append(tc_viz.load_run(path, title=f'{row_label} | {col_label}', color_map=GENOTYPE_COLORS))
        run_grid.append(row)

    fig = plt.figure(figsize=(5.5 * 3 * args.figscale, 6.5 * 2 * args.figscale))
    axes: list = []
    subtitle = 'z = time (hpf) — visualization metadata, not a learned coordinate'

    for r, row in enumerate(run_grid):
        for c, run in enumerate(row):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection='3d')
            _draw_stacked_3d(
                ax,
                run.positions,
                run.mask,
                run.time_values,
                run.labels,
                GENOTYPE_COLORS,
                point_size=8.0,
                alpha_point=0.6,
                alpha_line=0.2,
                linewidth=0.6,
                min_obs=2,
            )
            ax.set_title(f'{run.title}\n{subtitle}', fontsize=8)
            axes.append(ax)

    fig.tight_layout()

    if args.static_only:
        for ax in axes:
            ax.view_init(elev=args.elev, azim=args.azim_start)
        output_path = args.matrix_root / args.static_name
        fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(output_path)
        return

    azimuths = np.linspace(args.azim_start, args.azim_end, args.n_frames)

    def update(frame: int):
        azim = float(azimuths[frame])
        for ax in axes:
            ax.view_init(elev=args.elev, azim=azim)
        return []

    anim = animation.FuncAnimation(fig, update, frames=args.n_frames, interval=1000 / args.fps, blit=False)
    output_path = args.matrix_root / args.output_name
    _save_animation(anim, output_path, fps=args.fps, dpi=args.dpi)
    plt.close(fig)
    print(output_path)


if __name__ == '__main__':
    main()
