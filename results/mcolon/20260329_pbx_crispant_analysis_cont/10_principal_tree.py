"""
10_principal_tree.py
--------------------
Fit an elastic principal tree to embryo-level 3D space-time trajectories
using ElPiGraph, identify branch points, and test genotype enrichment at
each branch via embryo-level permutation chi-square.

Usage
-----
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260329_pbx_crispant_analysis_cont/10_principal_tree.py \\
    --positions-npz <path>/condensed_positions.npz \\
    --output-dir <out_dir>

  # Synthetic test:
    --positions-npz results/.../bifurcating_trunk_v5/E_all_on/condensed_positions.npz
    --output-dir /tmp/principal_tree_bifurcating_test
    --n-nodes 10

Parameters
----------
  --t-weight      3.0    temporal axis scale (higher → time dominates)
  --n-nodes       20     ElPiGraph tree nodes (complexity)
  --lambda-elpi   0.01   elasticity penalty
  --mu-elpi       0.1    bending/branching penalty
  --n-perm        1000
  --seed          42
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

from trajectory_cosmology.principal_tree import (
    build_embryo_spacetime_cloud,
    fit_principal_tree,
    project_observations_to_tree,
    identify_branch_nodes,
    run_all_branch_tests,
    branch_results_to_df,
)
from trajectory_cosmology.principal_tree_viz import (
    plot_tree_schematic,
    plot_branch_allocation_bars,
    plot_tree_3d,
    save_tree_3d_gif,
)


PBX_COLORS: dict[str, str] = {
    "inj_ctrl":               "#2166AC",
    "wik_ab":                 "#808080",
    "pbx1b_crispant":         "#9467bd",
    "pbx4_crispant":          "#F7B267",
    "pbx1b_pbx4_crispant":    "#B2182B",
}
SYNTHETIC_COLORS: dict[str, str] = {
    "0": "#2166AC",
    "1": "#B2182B",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ElPiGraph principal tree on embryo-level space-time cloud."
    )
    p.add_argument("--positions-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--t-weight", type=float, default=3.0)
    p.add_argument("--n-nodes", type=int, default=20,
                   help="Number of ElPiGraph tree nodes (default 20)")
    p.add_argument("--lambda-elpi", type=float, default=0.01,
                   help="ElPiGraph Lambda — elasticity penalty (default 0.01)")
    p.add_argument("--mu-elpi", type=float, default=0.1,
                   help="ElPiGraph Mu — bending/branching penalty (default 0.1)")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading: {args.positions_npz}")
    npz = np.load(args.positions_npz, allow_pickle=True)
    positions   = npz["positions"]
    mask        = npz["mask"]
    labels      = npz["labels"].astype(str)
    time_values = npz["time_values"]

    N_e, T, _ = positions.shape
    n_obs = int(mask.sum())
    conditions = sorted(np.unique(labels))
    print(f"  {N_e} embryos × {T} time bins, {n_obs} total obs")
    print(f"  Conditions: {conditions}")
    print(f"  Time: {time_values[0]:.0f}–{time_values[-1]:.0f} hpf")

    color_map = PBX_COLORS if "inj_ctrl" in conditions else {
        c: SYNTHETIC_COLORS.get(c, f"C{i}") for i, c in enumerate(conditions)
    }

    # ------------------------------------------------------------------
    # 2. Build embryo 3D cloud
    # ------------------------------------------------------------------
    print(f"\nBuilding embryo space-time cloud (t_weight={args.t_weight})...")
    pts_3d, obs_df = build_embryo_spacetime_cloud(
        positions, mask, time_values, t_weight=args.t_weight,
    )
    print(f"  {len(pts_3d)} observations in 3D cloud")

    # ------------------------------------------------------------------
    # 3. Fit elastic principal tree
    # ------------------------------------------------------------------
    print(f"\nFitting principal tree (n_nodes={args.n_nodes}, "
          f"Lambda={args.lambda_elpi}, Mu={args.mu_elpi})...")
    nodes_df, edges_df, raw = fit_principal_tree(
        pts_3d,
        n_nodes=args.n_nodes,
        Lambda=args.lambda_elpi,
        Mu=args.mu_elpi,
        verbose=False,
    )
    degree_dist = nodes_df["degree"].value_counts().sort_index().to_dict()
    print(f"  {len(nodes_df)} nodes, {len(edges_df)} edges")
    print(f"  Degree distribution: {degree_dist}")

    nodes_df.to_csv(out_dir / "tree_nodes.csv", index=False)
    edges_df.to_csv(out_dir / "tree_edges.csv", index=False)
    print(f"  Saved: tree_nodes.csv, tree_edges.csv")

    # ------------------------------------------------------------------
    # 4. Project observations to tree
    # ------------------------------------------------------------------
    print("\nProjecting observations to tree...")
    proj_df = project_observations_to_tree(pts_3d, obs_df, nodes_df, edges_df)
    proj_df.to_csv(out_dir / "observation_projections.csv", index=False)
    print(f"  Saved: observation_projections.csv")

    # ------------------------------------------------------------------
    # 5. Branch nodes
    # ------------------------------------------------------------------
    branch_node_ids = identify_branch_nodes(nodes_df)
    print(f"\nBranch nodes (degree ≥ 3): {branch_node_ids}")
    for bn in branch_node_ids:
        row = nodes_df.loc[bn]
        print(f"  Node {bn}: t_scaled={row['t_scaled']:.3f}, degree={row['degree']}")

    if not branch_node_ids:
        print("  No branch points found. Try reducing --n-nodes or --mu-elpi.")

    # ------------------------------------------------------------------
    # 6. Assign embryos + permutation tests
    # ------------------------------------------------------------------
    print(f"\nRunning branch tests (n_perm={args.n_perm}, unit=embryo)...")
    assignments_df, branch_results = run_all_branch_tests(
        proj_df, nodes_df, edges_df, branch_node_ids, labels,
        n_perm=args.n_perm, seed=args.seed,
    )

    assignments_df.to_csv(out_dir / "embryo_branch_assignments.csv", index=False)

    for res in branch_results:
        star = ("***" if res.pval < 0.001 else
                "**"  if res.pval < 0.01  else
                "*"   if res.pval < 0.05  else "ns")
        arm_info = "  ".join(
            f"arm{a}:{res.arm_conditions.get(a, [])}" for a in res.arm_ids
        )
        print(f"  Node {res.node_id} (t_sc={res.t_hpf_branch:.3f}): "
              f"p={res.pval:.4f} {star}  V={res.effect_size:.3f}  "
              f"n={res.n_embryos}  {arm_info}")

    branch_results_to_df(branch_results).to_csv(
        out_dir / "branch_test_summary.csv", index=False,
    )
    print(f"  Saved: branch_test_summary.csv, embryo_branch_assignments.csv")

    # ------------------------------------------------------------------
    # 7. Schematics
    # ------------------------------------------------------------------
    t_min_hpf = float(time_values.min())
    t_max_hpf = float(time_values.max())

    for spatial_axis in ("y", "x"):
        print(f"\nPlotting (t, {spatial_axis}) schematic...")
        fig, ax = plot_tree_schematic(
            nodes_df=nodes_df,
            edges_df=edges_df,
            branch_results=branch_results,
            obs_df=obs_df,
            labels=labels,
            color_map=color_map,
            spatial_axis=spatial_axis,
            t_min_hpf=t_min_hpf,
            t_max_hpf=t_max_hpf,
            t_weight=args.t_weight,
            annotate_ns=False,
            min_n_embryos=5,
            title=(
                f"Phenotypic Principal Tree  "
                f"(ElPiGraph, n_nodes={args.n_nodes}, "
                f"t_weight={args.t_weight}, n_perm={args.n_perm})"
            ),
        )
        out_path = out_dir / f"principal_tree_t{spatial_axis}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print("\nPlotting 3D diagnostic...")
    fig3d, _ = plot_tree_3d(
        nodes_df=nodes_df,
        edges_df=edges_df,
        obs_df=obs_df,
        labels=labels,
        color_map=color_map,
        title=f"Principal Tree 3D  (n_nodes={args.n_nodes}, t_weight={args.t_weight})",
    )
    fig3d.savefig(out_dir / "principal_tree_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig3d)
    print(f"  Saved: principal_tree_3d.png")

    print("  Rendering rotating 3D GIF...")
    save_tree_3d_gif(
        nodes_df=nodes_df,
        edges_df=edges_df,
        obs_df=obs_df,
        labels=labels,
        out_path=str(out_dir / "principal_tree_3d_rotate.gif"),
        color_map=color_map,
        figsize=(6, 5),
        n_frames=36,
        fps=12,
        title=f"Principal Tree 3D  (n_nodes={args.n_nodes}, t_weight={args.t_weight})",
    )
    print(f"  Saved: principal_tree_3d_rotate.gif")

    if branch_results:
        fig2, _ = plot_branch_allocation_bars(branch_results, color_map=color_map)
        fig2.savefig(out_dir / "branch_allocation_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved: branch_allocation_bars.png")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
