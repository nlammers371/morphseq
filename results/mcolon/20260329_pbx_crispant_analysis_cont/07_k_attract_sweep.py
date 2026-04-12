"""
07_k_attract_sweep.py
---------------------
Sweep k_attract across [10, 20, 40, 80, None] to find a value where the
map tightens without collapsing. Uses PCA init (fast) and shared x0 so the
only variable is k_attract.

Outputs one HTML per k value (for browser comparison) + a static PNG grid.

Usage:
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260329_pbx_crispant_analysis_cont/07_k_attract_sweep.py \\
    --input-csv  results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
                 phenotypic_positioning_multiclass_bin4_perm500/multiclass_probability_vectors.csv \\
    --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
                 force_calibration_v1/k_attract_sweep_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from trajectory_cosmology import schema, init_embedding, plotting
from trajectory_cosmology.condensation.api import run_condensation
from trajectory_cosmology.condensation.state import CondensationConfig
from trajectory_cosmology.condensation.engine.stopping import StoppingConfig

GENOTYPE_COLORS: dict[str, str] = {
    "inj_ctrl":               "#2166AC",
    "wik_ab":                 "#808080",
    "pbx1b_crispant":         "#9467bd",
    "pbx4_crispant":          "#F7B267",
    "pbx1b_pbx4_crispant":    "#B2182B",
}

GENOTYPE_ORDER = ["inj_ctrl", "wik_ab", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant"]

K_VALUES: list[int | None] = [10, 20, 40, 80, None]  # None = all-pairs


def _gamma(h: float) -> float:
    return 2.0 ** (-1.0 / h)


def make_plotly_html(positions, mask, time_values, embryo_ids, labels, title, output_path):
    import plotly.graph_objects as go

    traces = []
    genotypes = [g for g in GENOTYPE_ORDER if g in np.unique(labels)]
    for g in np.unique(labels):
        if g not in genotypes:
            genotypes.append(g)

    for geno in genotypes:
        color = GENOTYPE_COLORS.get(geno, "#555")
        embryo_idx = np.where(labels == geno)[0]

        # lines
        for i in embryo_idx:
            obs_t = np.where(mask[i, :])[0]
            if len(obs_t) < 2:
                continue
            traces.append(go.Scatter3d(
                x=positions[i, obs_t, 0], y=positions[i, obs_t, 1], z=time_values[obs_t],
                mode="lines", line=dict(color=color, width=1.5), opacity=0.2,
                name=geno, legendgroup=geno, showlegend=False,
                hoverinfo="skip",
            ))

        # points
        rows = []
        for i in embryo_idx:
            obs_t = np.where(mask[i, :])[0]
            for t in obs_t:
                rows.append(dict(x=positions[i,t,0], y=positions[i,t,1],
                                 z=float(time_values[t]),
                                 eid=str(embryo_ids[i]), hpf=float(time_values[t])))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        traces.append(go.Scatter3d(
            x=df["x"], y=df["y"], z=df["z"],
            mode="markers", marker=dict(color=color, size=3, opacity=0.75),
            name=geno, legendgroup=geno, showlegend=True,
            customdata=df[["eid", "hpf"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>hpf: %{customdata[1]:.0f}<extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title, width=950, height=750,
        scene=dict(
            xaxis_title="dim 1", yaxis_title="dim 2", zaxis_title="time (hpf)",
            bgcolor="white",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
        ),
        legend=dict(title="Genotype", itemsizing="constant"),
        paper_bgcolor="white",
    )
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def run_one(k: int | None, x0: np.ndarray, mask: np.ndarray,
            n_iter: int) -> np.ndarray:
    config = CondensationConfig(
        sigma=0.5, delta=3, lr=1e-4, max_iter=n_iter,
        fidelity_init_strength=0.25, fidelity_half_life=_gamma(70.0),
        epsilon_r=0.005, lambda_stretch=0.04, lambda_bend=0.04,
        epsilon_void=0.014, k_attract=k,
    )
    stopping = StoppingConfig(
        disp_max_rel_threshold=None, disp_rms_rel_threshold=None,
        energy_change_rel_threshold=None, coherence_change_rel_threshold=None,
    )
    result = run_condensation(x0=x0, mask=mask, config=config, stopping=stopping,
                              log_every=n_iter // 10, save_every=None, verbose=False)
    return result.positions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/"
                "phenotypic_positioning_multiclass_bin4_perm500/multiclass_probability_vectors.csv")
    p.add_argument("--output-dir",
        default="results/mcolon/20260329_pbx_crispant_analysis_cont/results/"
                "force_calibration_v1/k_attract_sweep_v1")
    p.add_argument("--n-iter", type=int, default=300,
                   help="Iterations per condition (300 is enough to see the effect)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {args.input_csv}")
    data = schema.from_multiclass_csv(args.input_csv, label_col="genotype")
    schema.validate(data)
    N_e, T, K = data.features.shape
    print(f"  {N_e} embryos × {T} time bins × {K} features")

    print("PCA init (shared x0 for all k values)...")
    x0 = init_embedding.pca_init(data.features, data.mask, random_state=args.seed)
    np.savez(output_dir / "x0_pca.npz", x0=x0)

    summary_rows = []
    all_positions: dict[str, np.ndarray] = {"init": x0}

    for k in K_VALUES:
        label = f"k={k}" if k is not None else "k=all"
        print(f"\n--- {label} ({args.n_iter} iters) ---")
        positions = run_one(k, x0.copy(), data.mask, args.n_iter)
        all_positions[label] = positions

        # Save HTML
        html_path = output_dir / f"trajectories_{label.replace('=','')}.html"
        make_plotly_html(
            positions, data.mask, data.time_values,
            data.embryo_ids, data.labels,
            title=f"PBX trajectories — {label} ({args.n_iter} iters, PCA init)",
            output_path=html_path,
        )
        print(f"  Saved: {html_path}")

        # Simple spread metric: mean pairwise distance at last time bin
        last_t = T - 1
        obs = np.where(data.mask[:, last_t])[0]
        pts = positions[obs, last_t, :]
        if len(pts) > 1:
            diffs = pts[:, None, :] - pts[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            spread = float(np.median(dists[np.triu_indices(len(pts), k=1)]))
        else:
            spread = float("nan")
        summary_rows.append({"k_attract": str(k), "label": label,
                              "spread_last_t_median": spread})
        print(f"  spread (last t): {spread:.3f}")

    # Static comparison PNG — 2D panels at 3 time points for each k
    snap_indices = np.linspace(0, T - 1, 3, dtype=int)
    snap_times = [float(data.time_values[i]) for i in snap_indices]
    n_k = len(K_VALUES)
    n_t = len(snap_times)

    fig, axes = plt.subplots(n_k, n_t, figsize=(n_t * 3.5, n_k * 3.5))
    for row_i, k in enumerate(K_VALUES):
        label = f"k={k}" if k is not None else "k=all"
        pos = all_positions[label]
        for col_j, (t_val, t_idx) in enumerate(zip(snap_times, snap_indices)):
            ax = axes[row_i, col_j]
            obs = np.where(data.mask[:, t_idx])[0]
            for geno in GENOTYPE_ORDER:
                gi = obs[data.labels[obs] == geno]
                if len(gi) == 0:
                    continue
                ax.scatter(pos[gi, t_idx, 0], pos[gi, t_idx, 1],
                           c=GENOTYPE_COLORS.get(geno, "#555"),
                           s=6, alpha=0.7, label=geno if row_i == 0 else "_")
            if col_j == 0:
                ax.set_ylabel(label, fontsize=8, fontweight="bold")
            if row_i == 0:
                ax.set_title(f"{t_val:.0f} hpf", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    # Legend on first row
    handles = [plt.Line2D([0],[0], marker="o", color="w",
                markerfacecolor=GENOTYPE_COLORS.get(g,"#555"), markersize=6, label=g)
               for g in GENOTYPE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"k_attract sweep — {args.n_iter} iters, PCA init\n"
                 "columns: early / mid / late time", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    png_path = output_dir / "k_attract_sweep_panels.png"
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved grid: {png_path}")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs: {output_dir}")


if __name__ == "__main__":
    main()
