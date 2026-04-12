"""
06_pbx_trajectory_viz.py
------------------------
Interactive Plotly visualization of condensed PBX crispant trajectories.

Loads condensed_positions.npz and produces an HTML file with:
  - 3D scatter+lines: x=dim1, y=dim2, z=time (hpf), color=genotype
  - Hover: embryo_id, genotype, hpf
  - Toggle genotypes on/off via legend
  - Dropdown to switch between x0 (AlignedUMAP init) and condensed positions

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260329_pbx_crispant_analysis_cont/06_pbx_trajectory_viz.py \\
    --input results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
            force_calibration_v1/pbx_condensation_v1/condensed_positions.npz \\
    --output results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
             force_calibration_v1/pbx_condensation_v1/trajectories_3d.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Genotype color palette
# ---------------------------------------------------------------------------

GENOTYPE_COLORS: dict[str, str] = {
    "inj_ctrl":               "#2166AC",
    "wik_ab":                 "#808080",
    "pbx1b_crispant":         "#9467bd",
    "pbx4_crispant":          "#F7B267",
    "pbx1b_pbx4_crispant":    "#B2182B",
}

GENOTYPE_ORDER = [
    "inj_ctrl",
    "wik_ab",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]


def load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_traces(positions: np.ndarray,
                 mask: np.ndarray,
                 time_values: np.ndarray,
                 embryo_ids: np.ndarray,
                 labels: np.ndarray,
                 visible: bool = True,
                 line_opacity: float = 0.25,
                 point_opacity: float = 0.75,
                 point_size: int = 3,
                 show_lines: bool = True):
    """Return a list of plotly traces — one per genotype (lines + points).

    Lines and points for the same genotype share legendgroup so clicking
    the legend toggles both.
    """
    import plotly.graph_objects as go

    traces = []
    genotypes = [g for g in GENOTYPE_ORDER if g in np.unique(labels)]
    # Any unlisted genotypes appended at end
    for g in np.unique(labels):
        if g not in genotypes:
            genotypes.append(g)

    for geno in genotypes:
        color = GENOTYPE_COLORS.get(geno, "#555555")
        embryo_idx = np.where(labels == geno)[0]

        # --- Line traces (one per embryo, grouped) ---
        if show_lines:
            for i in embryo_idx:
                obs_t = np.where(mask[i, :])[0]
                if len(obs_t) < 2:
                    continue
                xs = positions[i, obs_t, 0]
                ys = positions[i, obs_t, 1]
                zs = time_values[obs_t]
                eid = str(embryo_ids[i])
                traces.append(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    opacity=line_opacity,
                    name=geno,
                    legendgroup=geno,
                    showlegend=False,
                    visible=visible,
                    hoverinfo="skip",
                ))

        # --- Point traces (all embryos of this genotype, one trace) ---
        rows = []
        for i in embryo_idx:
            obs_t = np.where(mask[i, :])[0]
            for t in obs_t:
                rows.append({
                    "x": positions[i, t, 0],
                    "y": positions[i, t, 1],
                    "z": float(time_values[t]),
                    "embryo_id": str(embryo_ids[i]),
                    "hpf": float(time_values[t]),
                    "genotype": geno,
                })
        if not rows:
            continue
        df = pd.DataFrame(rows)
        traces.append(go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"],
            mode="markers",
            marker=dict(color=color, size=point_size, opacity=point_opacity),
            name=geno,
            legendgroup=geno,
            showlegend=True,
            visible=visible,
            customdata=df[["embryo_id", "genotype", "hpf"]].values,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "embryo: %{customdata[0]}<br>"
                "hpf: %{customdata[2]:.0f}<br>"
                "dim1: %{x:.3f}<br>"
                "dim2: %{y:.3f}"
                "<extra></extra>"
            ),
        ))

    return traces


def make_figure(data: dict) -> "go.Figure":
    import plotly.graph_objects as go

    positions = data["positions"]   # (N_e, T, 2)
    x0        = data["x0"]          # (N_e, T, 2)
    mask      = data["mask"]        # (N_e, T) bool
    time_values = data["time_values"]
    embryo_ids  = data["embryo_ids"]
    labels      = data["labels"]

    N_e, T, _ = positions.shape

    # Build two sets of traces: condensed (visible) and x0 init (hidden)
    traces_condensed = build_traces(
        positions, mask, time_values, embryo_ids, labels,
        visible=True,
    )
    traces_init = build_traces(
        x0, mask, time_values, embryo_ids, labels,
        visible=False,
    )

    all_traces = traces_condensed + traces_init
    n_condensed = len(traces_condensed)
    n_init = len(traces_init)

    # Dropdown buttons to toggle between the two views
    btn_condensed = dict(
        label="Condensed (post-optimization)",
        method="update",
        args=[
            {"visible": [True] * n_condensed + [False] * n_init},
            {"title": "PBX trajectories — condensed (AlignedUMAP → 500 iters)"},
        ],
    )
    btn_init = dict(
        label="Init (AlignedUMAP only)",
        method="update",
        args=[
            {"visible": [False] * n_condensed + [True] * n_init},
            {"title": "PBX trajectories — AlignedUMAP initialization"},
        ],
    )

    fig = go.Figure(data=all_traces)
    fig.update_layout(
        title="PBX trajectories — condensed (AlignedUMAP → 500 iters)",
        width=1000,
        height=800,
        scene=dict(
            xaxis_title="dim 1",
            yaxis_title="dim 2",
            zaxis_title="time (hpf)",
            xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
            zaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
            bgcolor="white",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
        ),
        legend=dict(
            title="Genotype<br><sup>(click to toggle)</sup>",
            itemsizing="constant",
            font=dict(size=11),
        ),
        updatemenus=[dict(
            buttons=[btn_condensed, btn_init],
            direction="down",
            showactive=True,
            x=0.02,
            xanchor="left",
            y=0.98,
            yanchor="top",
            bgcolor="white",
            bordercolor="#ccc",
        )],
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12),
    )

    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/"
                "force_calibration_v1/pbx_condensation_v1/condensed_positions.npz",
    )
    p.add_argument(
        "--output",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/"
                "force_calibration_v1/pbx_condensation_v1/trajectories_3d.html",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading: {args.input}")
    data = load_npz(Path(args.input))
    print(f"  positions: {data['positions'].shape}  labels: {set(data['labels'].tolist())}")

    print("Building figure...")
    fig = make_figure(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved: {output_path}")
    print("Open in a browser to explore interactively.")


if __name__ == "__main__":
    main()
