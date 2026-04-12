from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from common import GENOTYPE_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Plotly init-vs-final stacked 3D review HTML.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-dirs", nargs="+", type=Path, default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-name", type=str, default="init_vs_final_plotly.html")
    parser.add_argument("--title", type=str, default="Init vs Final")
    return parser.parse_args()


def _build_records(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray | None,
) -> dict[str, dict[str, np.ndarray]]:
    records: dict[str, dict[str, list[float] | list[str]]] = {}
    n_items, t_count, _ = positions.shape
    for i in range(n_items):
        label = str(labels[i]) if labels is not None else "unknown"
        ident = str(ids[i]) if ids is not None else str(i)
        if label not in records:
            records[label] = {"x": [], "y": [], "z": [], "id": []}
        for t in range(t_count):
            if not mask[i, t]:
                continue
            xy = positions[i, t]
            if not np.isfinite(xy).all():
                continue
            records[label]["x"].append(float(xy[0]))
            records[label]["y"].append(float(xy[1]))
            records[label]["z"].append(float(time_values[t]))
            records[label]["id"].append(ident)
        records[label]["x"].append(np.nan)
        records[label]["y"].append(np.nan)
        records[label]["z"].append(np.nan)
        records[label]["id"].append("")
    return {k: {kk: np.asarray(vv, dtype=object if kk == "id" else float) for kk, vv in v.items()} for k, v in records.items()}


def _load_run(run_dir: Path) -> dict[str, object]:
    payload = np.load(run_dir / "condensed_positions.npz", allow_pickle=True)
    x0_payload = np.load(run_dir / "x0_init.npz", allow_pickle=True)
    positions = np.asarray(payload["positions"], dtype=float)
    mask = np.asarray(payload["mask"], dtype=bool)
    time_values = np.asarray(payload["time_values"], dtype=float)
    labels = np.asarray(payload["labels"], dtype=object) if "labels" in payload.files else np.asarray(["unknown"] * positions.shape[0], dtype=object)
    ids = np.asarray(payload["embryo_ids"], dtype=object) if "embryo_ids" in payload.files else None
    x0 = np.asarray(x0_payload["x0"], dtype=float)
    return {
        "positions": positions,
        "mask": mask,
        "time_values": time_values,
        "labels": labels,
        "ids": ids,
        "x0": x0,
    }


def _axis_limits(runs: list[dict[str, object]]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    all_xy = []
    all_t = []
    for run in runs:
        mask = run["mask"]
        x0 = run["x0"]
        positions = run["positions"]
        all_xy.append(x0[mask])
        all_xy.append(positions[mask])
        all_t.append(np.asarray(run["time_values"], dtype=float))
    combined = np.vstack(all_xy)
    times = np.concatenate(all_t)
    xmin = float(np.nanmin(combined[:, 0]))
    xmax = float(np.nanmax(combined[:, 0]))
    ymin = float(np.nanmin(combined[:, 1]))
    ymax = float(np.nanmax(combined[:, 1]))
    zmin = float(np.nanmin(times))
    zmax = float(np.nanmax(times))
    xpad = 0.05 * max(xmax - xmin, 1e-6)
    ypad = 0.05 * max(ymax - ymin, 1e-6)
    zpad = 0.02 * max(zmax - zmin, 1e-6)
    return (xmin - xpad, xmax + xpad), (ymin - ypad, ymax + ypad), (zmin - zpad, zmax + zpad)


def _single_run_figure(run: dict[str, object], *, title: str) -> go.Figure:
    init_records = _build_records(run["x0"], run["mask"], run["time_values"], run["labels"], run["ids"])
    final_records = _build_records(run["positions"], run["mask"], run["time_values"], run["labels"], run["ids"])
    xlim, ylim, zlim = _axis_limits([run])
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("Initialized", "Final"), horizontal_spacing=0.03)
    for col, dataset in [(1, init_records), (2, final_records)]:
        for label, rec in dataset.items():
            color = GENOTYPE_COLORS.get(label, "#4C78A8")
            fig.add_trace(
                go.Scatter3d(
                    x=rec["x"], y=rec["y"], z=rec["z"], mode="lines+markers", name=label, legendgroup=label,
                    showlegend=(col == 1), line=dict(color=color, width=2), marker=dict(color=color, size=3, opacity=0.75),
                    hovertemplate="<b>%{text}</b><br>dim1=%{x:.3f}<br>dim2=%{y:.3f}<br>time=%{z:.1f}<extra></extra>", text=rec["id"],
                ), row=1, col=col,
            )
    scene_common = dict(xaxis=dict(title="dim 1", range=list(xlim)), yaxis=dict(title="dim 2", range=list(ylim)), zaxis=dict(title="time (hpf)", range=list(zlim)), camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)))
    fig.update_layout(title=title, template="plotly_white", width=1500, height=700, legend=dict(itemsizing="constant"), scene=scene_common, scene2=scene_common)
    return fig


def _multi_run_figure(runs: list[dict[str, object]], labels: list[str], *, title: str) -> go.Figure:
    xlim, ylim, zlim = _axis_limits(runs)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("Initialized", "Final"), horizontal_spacing=0.03)
    trace_groups: list[tuple[int, int]] = []
    default_run_idx = 0

    for run_idx, (run, run_label) in enumerate(zip(runs, labels)):
        init_records = _build_records(run["x0"], run["mask"], run["time_values"], run["labels"], run["ids"])
        final_records = _build_records(run["positions"], run["mask"], run["time_values"], run["labels"], run["ids"])
        start = len(fig.data)
        for col, dataset in [(1, init_records), (2, final_records)]:
            for label, rec in dataset.items():
                color = GENOTYPE_COLORS.get(label, "#4C78A8")
                fig.add_trace(
                    go.Scatter3d(
                        x=rec["x"], y=rec["y"], z=rec["z"], mode="lines+markers", name=label, legendgroup=f"{run_idx}:{label}",
                        showlegend=(col == 1 and run_idx == default_run_idx), visible=(run_idx == default_run_idx),
                        line=dict(color=color, width=2), marker=dict(color=color, size=3, opacity=0.75),
                        hovertemplate="<b>%{text}</b><br>dim1=%{x:.3f}<br>dim2=%{y:.3f}<br>time=%{z:.1f}<extra></extra>", text=rec["id"],
                    ), row=1, col=col,
                )
        trace_groups.append((start, len(fig.data)))

    buttons = []
    total_traces = len(fig.data)
    scene_common = dict(xaxis=dict(title="dim 1", range=list(xlim)), yaxis=dict(title="dim 2", range=list(ylim)), zaxis=dict(title="time (hpf)", range=list(zlim)), camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)))
    for idx, run_label in enumerate(labels):
        visible = [False] * total_traces
        start, end = trace_groups[idx]
        for j in range(start, end):
            visible[j] = True
        buttons.append(
            dict(
                label=run_label,
                method="update",
                args=[{"visible": visible}, {"title": f"{title} | {run_label}", "scene": scene_common, "scene2": scene_common}],
            )
        )

    fig.update_layout(
        title=f"{title} | {labels[default_run_idx]}",
        template="plotly_white",
        width=1500,
        height=700,
        legend=dict(itemsizing="constant"),
        scene=scene_common,
        scene2=scene_common,
        updatemenus=[dict(buttons=buttons, direction="down", x=0.02, y=1.12, xanchor="left", yanchor="top")],
    )
    return fig


def main() -> None:
    args = parse_args()
    if args.run_dirs:
        run_dirs = list(args.run_dirs)
        labels = list(args.labels) if args.labels else [p.name for p in run_dirs]
        if len(labels) != len(run_dirs):
            raise ValueError("--labels must match --run-dirs length")
        runs = [_load_run(p) for p in run_dirs]
        fig = _multi_run_figure(runs, labels, title=args.title)
        output_path = Path(run_dirs[0]) / args.output_name
    elif args.run_dir:
        run = _load_run(args.run_dir)
        fig = _single_run_figure(run, title=args.title)
        output_path = Path(args.run_dir) / args.output_name
    else:
        raise ValueError("Provide --run-dir or --run-dirs")

    fig.write_html(str(output_path))
    print(output_path)


if __name__ == "__main__":
    main()
