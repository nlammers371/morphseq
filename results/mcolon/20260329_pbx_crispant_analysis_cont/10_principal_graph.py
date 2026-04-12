"""
10_principal_graph.py  (v3)
---------------------------
Fit a principal tree to the 3D (x, y, t) space-time trajectory cloud from
cosmological condensation outputs, find branch points, and test whether
genotypes distribute non-uniformly across branches.

Pipeline
--------
1. Load condensed_positions.npz
2. Compute condition-mean trajectories in (x, y, t) — the compressed backbone
3. Fit MST on those trajectories in 3D (x, y, t_norm) space
4. Contract degree-2 chains → skeleton with leaves + branch points
5. Assign each embryo to an arm at each branch node (majority vote over time)
6. Permutation chi-square (embryo-level null) at each branch node
7. Save (t, y) and (t, x) 2D schematics + CSVs

Usage
-----
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260329_pbx_crispant_analysis_cont/10_principal_graph.py \\
    --positions-npz results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
pairwise_shrunk_condensation_aligned_umap_bin4_perm500/condensed_positions.npz \\
    --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_graph_v3

  # Synthetic test:
    --positions-npz results/.../bifurcating_trunk_v5/E_all_on/condensed_positions.npz
    --output-dir /tmp/principal_graph_bifurcating_test

  Parameters:
    --t-weight 3.0     (temporal vs spatial scale; default 3.0)
    --n-perm 1000
    --min-obs 2        (min embryos per condition per time bin for mean traj)
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

from trajectory_cosmology.principal_graph import (
    build_mean_trajectories,
    build_trajectory_mst,
    contract_to_skeleton,
    identify_branch_points,
    run_all_branch_tests,
    branch_results_to_df,
)
from trajectory_cosmology.principal_graph_viz import (
    plot_spacetime_schematic,
    plot_branch_allocation_bars,
)


# ---------------------------------------------------------------------------
# Genotype colors — works for PBX conditions and synthetic labels
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Principal graph on 3D (x,y,t) trajectory cloud."
    )
    p.add_argument("--positions-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--t-weight", type=float, default=3.0,
                   help="Scale of t axis relative to x/y (default 3.0)")
    p.add_argument("--min-obs", type=int, default=2,
                   help="Min embryos per condition per time bin (default 2)")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading: {args.positions_npz}")
    npz = np.load(args.positions_npz, allow_pickle=True)
    positions   = npz["positions"]    # (N_e, T, 2)
    mask        = npz["mask"]         # (N_e, T)
    labels      = npz["labels"].astype(str)  # (N_e,)
    time_values = npz["time_values"]  # (T,)

    N_e, T, _ = positions.shape
    n_obs = int(mask.sum())
    conditions = sorted(np.unique(labels))
    print(f"  {N_e} embryos × {T} time bins, {n_obs} total obs")
    print(f"  Conditions: {conditions}")
    print(f"  Time: {time_values[0]:.0f}–{time_values[-1]:.0f}")

    # Pick color map
    color_map = PBX_COLORS if "inj_ctrl" in conditions else {
        c: SYNTHETIC_COLORS.get(c, f"C{i}") for i, c in enumerate(conditions)
    }

    # ------------------------------------------------------------------
    # 2. Condition-mean trajectories in (x, y, t)
    # ------------------------------------------------------------------
    print(f"\nBuilding condition-mean trajectories (min_obs={args.min_obs})...")
    traj_df = build_mean_trajectories(
        positions, mask, labels, time_values, min_obs=args.min_obs,
    )
    print(f"  {len(traj_df)} trajectory nodes "
          f"({len(traj_df)//len(conditions)} bins/condition avg)")

    # ------------------------------------------------------------------
    # 3. MST on 3D trajectory cloud
    # ------------------------------------------------------------------
    print(f"\nFitting MST on 3D trajectories (t_weight={args.t_weight})...")
    traj_df, edges, adjacency = build_trajectory_mst(traj_df, t_weight=args.t_weight)
    print(f"  {len(traj_df)} nodes, {len(edges)} edges")

    # ------------------------------------------------------------------
    # 4. Contract to skeleton
    # ------------------------------------------------------------------
    print("Contracting to skeleton...")
    skel_df, skel_edges, skel_adj, owned = contract_to_skeleton(
        traj_df, edges, adjacency,
    )
    branch_node_ids = identify_branch_points(skel_adj)
    print(f"  Skeleton: {len(skel_df)} nodes, {len(skel_edges)} edges")
    print(f"  Degree distribution: {dict(zip(*np.unique(skel_adj.sum(axis=1), return_counts=True)))}")

    if branch_node_ids:
        for bn in branch_node_ids:
            row = skel_df.loc[bn]
            print(f"  Branch node {bn}: t≈{row['t_hpf_mean']:.1f}, "
                  f"degree={row['degree']}, "
                  f"conditions={row['conditions_through']}")
    else:
        print("  No branch points found. Try adjusting --t-weight.")

    skel_df.to_csv(out_dir / "skeleton_nodes.csv", index=False)
    print(f"Saved: {out_dir / 'skeleton_nodes.csv'}")

    # ------------------------------------------------------------------
    # 5–6. Assign embryos + permutation tests
    # ------------------------------------------------------------------
    print(f"\nRunning branch tests (n_perm={args.n_perm}, unit=embryo)...")
    assignments_df, branch_results = run_all_branch_tests(
        skel_df, skel_adj, branch_node_ids, owned,
        traj_df, positions, mask, labels, time_values,
        n_perm=args.n_perm, seed=args.seed,
    )

    for res in branch_results:
        star = ("***" if res.pval < 0.001 else
                "**"  if res.pval < 0.01  else
                "*"   if res.pval < 0.05  else "ns")
        arm_info = "  ".join(
            f"arm{a}:{res.arm_conditions.get(a, [])}" for a in res.arm_ids
        )
        print(f"  Node {res.node_id} (t≈{res.t_hpf_branch:.1f}): "
              f"p={res.pval:.4f} {star}  V={res.effect_size:.3f}  "
              f"n={res.n_embryos}  {arm_info}")

    branch_results_to_df(branch_results).to_csv(
        out_dir / "branch_test_summary.csv", index=False,
    )
    assignments_df.to_csv(out_dir / "node_assignments.csv", index=False)
    print(f"Saved: {out_dir / 'branch_test_summary.csv'}")

    # Also save mean trajectories for inspection
    traj_df.to_csv(out_dir / "mean_trajectories.csv", index=False)

    # ------------------------------------------------------------------
    # 7. Schematics
    # ------------------------------------------------------------------
    for spatial_axis in ("y", "x"):
        print(f"\nPlotting (t, {spatial_axis}) schematic...")
        fig, ax = plot_spacetime_schematic(
            skel_nodes_df=skel_df,
            skel_edges=skel_edges,
            branch_results=branch_results,
            positions=positions,
            mask=mask,
            labels=labels,
            time_values=time_values,
            color_map=color_map,
            spatial_axis=spatial_axis,
            annotate_ns=False,
            min_n_embryos=5,
            title=(
                f"Phenotypic Principal Graph  "
                f"(t_weight={args.t_weight}, n_perm={args.n_perm})"
            ),
        )
        out_path = out_dir / f"phenotypic_graph_t{spatial_axis}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    if branch_results:
        fig2, _ = plot_branch_allocation_bars(
            branch_results, color_map=color_map,
        )
        bars_path = out_dir / "branch_allocation_bars.png"
        fig2.savefig(bars_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {bars_path}")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
