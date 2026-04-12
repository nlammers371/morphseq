from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(SCRIPT_DIR))

from common import GENOTYPE_COLORS


SUBSETS: dict[str, tuple[list[str], str]] = {
    'pbx1b_pbx4_double_trio': (
        ['pbx1b_crispant', 'pbx4_crispant', 'pbx1b_pbx4_crispant'],
        'pbx1b vs pbx4 vs double',
    ),
    'pbx4_vs_double': (
        ['pbx4_crispant', 'pbx1b_pbx4_crispant'],
        'pbx4 vs double',
    ),
    'pbx1b_vs_double': (
        ['pbx1b_crispant', 'pbx1b_pbx4_crispant'],
        'pbx1b vs double',
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Render lightweight Plotly time-slice views from a full 5-class run by label subsetting.'
    )
    parser.add_argument('--init-path', type=Path, required=True)
    parser.add_argument('--final-path', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--subsets', nargs='+', choices=sorted(SUBSETS), default=list(SUBSETS))
    parser.add_argument('--width', type=int, default=1100)
    parser.add_argument('--height', type=int, default=800)
    return parser.parse_args()


def _load_subset(path: Path, keep_labels: list[str], position_key: str) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    labels = np.asarray(d['labels'], dtype=object)
    keep_set = set(keep_labels)
    sel = np.array([str(x) in keep_set for x in labels], dtype=bool)
    return {
        'positions': np.asarray(d[position_key], dtype=float)[sel],
        'mask': np.asarray(d['mask'], dtype=bool)[sel],
        'time_values': np.asarray(d['time_values'], dtype=float),
        'labels': labels[sel],
        'embryo_ids': np.asarray(d['embryo_ids'], dtype=object)[sel],
    }


def _make_figure(
    *,
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray,
    embryo_ids: np.ndarray,
    title: str,
    width: int,
    height: int,
) -> go.Figure:
    finite = np.isfinite(positions).all(axis=2) & mask
    xy = np.where(finite[..., None], positions, np.nan)
    mins = np.nanmin(xy, axis=(0, 1))
    maxs = np.nanmax(xy, axis=(0, 1))
    pad = np.maximum((maxs - mins) * 0.08, 1e-6)
    mins = mins - pad
    maxs = maxs + pad
    zmin = float(np.nanmin(time_values))
    zmax = float(np.nanmax(time_values))

    unique_labels = [str(x) for x in np.unique(labels)]
    frames: list[go.Frame] = []
    for ti, time_value in enumerate(time_values):
        frame_traces: list[go.BaseTraceType] = []
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            color = GENOTYPE_COLORS.get(label, '#777777')
            for embryo_idx in idx:
                obs = mask[embryo_idx]
                traj = positions[embryo_idx, obs]
                if traj.shape[0] >= 2:
                    frame_traces.append(
                        go.Scatter3d(
                            x=traj[:, 0],
                            y=traj[:, 1],
                            z=time_values[obs],
                            mode='lines',
                            line=dict(color=color, width=2),
                            opacity=0.12,
                            showlegend=False,
                            hoverinfo='skip',
                        )
                    )
            obs_t = idx[mask[idx, ti]]
            slice_xy = positions[obs_t, ti]
            frame_traces.append(
                go.Scatter3d(
                    x=slice_xy[:, 0],
                    y=slice_xy[:, 1],
                    z=np.full(obs_t.shape[0], float(time_value)),
                    mode='markers',
                    marker=dict(size=4, color=color),
                    name=label,
                    legendgroup=label,
                    showlegend=(ti == 0),
                    text=[str(embryo_ids[i]) for i in obs_t],
                    hovertemplate='id=%{text}<extra>' + label + '</extra>',
                )
            )
        frames.append(go.Frame(name=str(time_value), data=frame_traces))

    fig = go.Figure()
    fig.frames = frames
    if frames:
        for trace in frames[0].data:
            fig.add_trace(trace)

    steps = [
        dict(
            method='animate',
            args=[
                [str(time_value)],
                {
                    'mode': 'immediate',
                    'frame': {'duration': 0, 'redraw': True},
                    'transition': {'duration': 0},
                },
            ],
            label=str(int(time_value)) if float(time_value).is_integer() else str(time_value),
        )
        for time_value in time_values
    ]
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis=dict(range=[float(mins[0]), float(maxs[0])], title='dim 1'),
            yaxis=dict(range=[float(mins[1]), float(maxs[1])], title='dim 2'),
            zaxis=dict(range=[zmin, zmax], title='time (hpf)'),
        ),
        updatemenus=[
            dict(
                type='buttons',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[
                            None,
                            {
                                'fromcurrent': True,
                                'frame': {'duration': 250, 'redraw': True},
                                'transition': {'duration': 0},
                            },
                        ],
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[
                            [None],
                            {
                                'mode': 'immediate',
                                'frame': {'duration': 0, 'redraw': False},
                                'transition': {'duration': 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[dict(active=0, steps=steps, currentvalue={'prefix': 'time='})],
        legend=dict(itemsizing='constant'),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _write_subset(
    *,
    out_dir: Path,
    init_path: Path,
    final_path: Path | None,
    keep_labels: list[str],
    title: str,
    width: int,
    height: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    init_data = _load_subset(init_path, keep_labels, 'x0')
    init_fig = _make_figure(title=f'UMAP init | {title}', width=width, height=height, **init_data)
    init_fig.write_html(out_dir / 'time_slice_init_plotly.html', include_plotlyjs='cdn')

    if final_path is not None and final_path.exists():
        final_data = _load_subset(final_path, keep_labels, 'positions')
        final_fig = _make_figure(title=f'Final | {title}', width=width, height=height, **final_data)
        final_fig.write_html(out_dir / 'time_slice_final_plotly.html', include_plotlyjs='cdn')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for slug in args.subsets:
        keep_labels, title = SUBSETS[slug]
        _write_subset(
            out_dir=args.output_dir / slug,
            init_path=args.init_path,
            final_path=args.final_path,
            keep_labels=keep_labels,
            title=title,
            width=args.width,
            height=args.height,
        )
        print(args.output_dir / slug)


if __name__ == '__main__':
    main()
