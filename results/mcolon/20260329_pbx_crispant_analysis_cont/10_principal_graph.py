"""
10_principal_graph.py  (v2)
---------------------------
Space-time principal graph: coarsened grid MST + embryo-level permutation
branch tests on genotype arm-allocation distributions.

Pipeline
--------
1. Load condensed_positions.npz
2. Coarsen (x,y,t) cloud into spatial grid centroids per time bin
3. Fit k-NN MST on ~50-200 centroids
4. Contract degree-2 chains → skeleton
5. Identify branch points (degree >= 3)
6. Assign embryos to arms via BFS majority vote
7. Permutation chi-square test (embryo-level null)
8. Save (t_hpf, y) and (t_hpf, x) schematics + CSVs

Key design decisions
--------------------
* Permutation unit = embryo.  All timepoints from one embryo move together
  in the null.  Observation-level permutation would inflate significance.
* Graph built on coarsened grid, not raw observations, to avoid density artifacts.
* Both condensed axes saved to avoid hiding real branch geometry.

Usage
-----
  # Default run
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260329_pbx_crispant_analysis_cont/10_principal_graph.py \\
    --positions-npz results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
pairwise_shrunk_condensation_aligned_umap_bin4_perm500/condensed_positions.npz \\
    --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_graph_v2

  # Tune grid / graph
    --t-weight 2.0       # temporal vs spatial scale (default 2.0)
    --grid-cells 5       # spatial bins per time bin (default 5)
    --k-neighbors 5      # k for k-NN on centroids (default 5)
    --n-perm 1000
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
    build_spacetime_grid_centroids,
    build_spacetime_mst,
    contract_mst_skeleton,
    identify_branch_points,
    run_all_branch_tests,
    branch_results_to_df,
)
from trajectory_cosmology.principal_graph_viz import (
    plot_spacetime_schematic,
    plot_branch_allocation_bars,
)


# ---------------------------------------------------------------------------
# Genotype colors (consistent with rest of PBX analysis)
# ---------------------------------------------------------------------------

GENOTYPE_COLORS: dict[str, str] = {
    "inj_ctrl":               "#2166AC",
    "wik_ab":                 "#808080",
    "pbx1b_crispant":         "#9467bd",
    "pbx4_crispant":          "#F7B267",
    "pbx1b_pbx4_crispant":    "#B2182B",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Space-time principal graph with embryo-level branch tests."
    )
    p.add_argument(
        "--positions-npz",
        default=(
            "results/mcolon/20260329_pbx_crispant_analysis_cont/results/"
            "pairwise_shrunk_condensation_aligned_umap_bin4_perm500/"
            "condensed_positions.npz"
        ),
    )
    p.add_argument(
        "--output-dir",
        default=(
            "results/mcolon/20260329_pbx_crispant_analysis_cont/results/"
            "phenotypic_graph_v2"
        ),
    )
    p.add_argument("--t-weight", type=float, default=2.0,
                   help="Temporal scale weight (default 2.0)")
    p.add_argument("--grid-cells", type=int, default=5,
                   help="Spatial grid resolution per time bin (default 5)")
    p.add_argument("--k-neighbors", type=int, default=5,
                   help="k for k-NN on coarsened centroids (default 5)")
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
    labels      = npz["labels"]       # (N_e,)
    time_values = npz["time_values"]  # (T,)

    N_e, T, _ = positions.shape
    n_obs = int(mask.sum())
    print(f"  {N_e} embryos × {T} time bins, {n_obs} total observations")
    print(f"  Conditions: {sorted(np.unique(labels))}")

    # ------------------------------------------------------------------
    # 2. Coarsen into space-time grid centroids
    # ------------------------------------------------------------------
    print(f"\nCoarsening to grid (grid_cells={args.grid_cells}, "
          f"t_weight={args.t_weight})...")
    centroids_df, obs_ownership = build_spacetime_grid_centroids(
        positions, mask, time_values,
        grid_cells=args.grid_cells,
        t_weight=args.t_weight,
    )
    print(f"  {len(centroids_df)} centroid nodes "
          f"(mean {centroids_df['n_obs'].mean():.1f} obs/node, "
          f"max {centroids_df['n_obs'].max()})")

    # ------------------------------------------------------------------
    # 3. Fit MST on centroids
    # ------------------------------------------------------------------
    print(f"\nFitting k-NN MST (k={args.k_neighbors})...")
    nodes_df, edges, adjacency = build_spacetime_mst(
        centroids_df, k_neighbors=args.k_neighbors,
    )
    print(f"  MST: {len(nodes_df)} nodes, {len(edges)} edges")

    # ------------------------------------------------------------------
    # 4. Contract to skeleton
    # ------------------------------------------------------------------
    print("Contracting to skeleton...")
    skel_nodes_df, skel_edges, skel_adj, skel_obs_ownership = contract_mst_skeleton(
        nodes_df, edges, adjacency, obs_ownership,
    )
    branch_node_ids = identify_branch_points(skel_adj)
    degrees = skel_adj.sum(axis=1)
    print(f"  Skeleton: {len(skel_nodes_df)} nodes, {len(skel_edges)} edges")
    print(f"  Node degrees: {sorted(set(degrees.tolist()))}")
    if branch_node_ids:
        for bn in branch_node_ids:
            row = skel_nodes_df.loc[bn]
            print(f"  Branch node {bn}: t≈{row['t_hpf_mean']:.0f} hpf, "
                  f"n_obs={row['n_obs']}, degree={row['degree']}")
    else:
        print("  No branch points found (all nodes degree <= 2).")
        print("  Try reducing --t-weight or --grid-cells.")

    # Save skeleton node table
    skel_nodes_df.to_csv(out_dir / "skeleton_nodes.csv", index=False)
    print(f"Saved: {out_dir / 'skeleton_nodes.csv'}")

    # ------------------------------------------------------------------
    # 5–6. Assign embryos to arms + run permutation tests
    # ------------------------------------------------------------------
    print(f"\nRunning branch tests (n_perm={args.n_perm}, "
          f"permutation unit = embryo)...")
    assignments_df, branch_results = run_all_branch_tests(
        skel_nodes_df, skel_adj, branch_node_ids,
        labels, skel_obs_ownership,
        n_perm=args.n_perm, seed=args.seed,
    )

    for res in branch_results:
        star = ("***" if res.pval < 0.001 else
                "**"  if res.pval < 0.01  else
                "*"   if res.pval < 0.05  else "ns")
        print(f"  Node {res.node_id} (t≈{res.t_hpf_branch:.0f} hpf): "
              f"p={res.pval:.4f} {star},  V={res.effect_size:.3f},  "
              f"n_embryos={res.n_embryos}")

    # Save CSVs
    branch_results_to_df(branch_results).to_csv(
        out_dir / "branch_test_summary.csv", index=False
    )
    assignments_df.to_csv(out_dir / "node_assignments.csv", index=False)
    print(f"Saved: {out_dir / 'branch_test_summary.csv'}")
    print(f"Saved: {out_dir / 'node_assignments.csv'}")

    # ------------------------------------------------------------------
    # 7. Schematics
    # ------------------------------------------------------------------
    for spatial_axis in ("y", "x"):
        print(f"\nPlotting (t_hpf, {spatial_axis}) schematic...")
        fig, ax = plot_spacetime_schematic(
            skel_nodes_df=skel_nodes_df,
            skel_edges=skel_edges,
            branch_results=branch_results,
            positions=positions,
            mask=mask,
            labels=labels,
            time_values=time_values,
            color_map=GENOTYPE_COLORS,
            spatial_axis=spatial_axis,
            annotate_ns=False,
            min_n_embryos=20,
            title=(
                f"Phenotypic Principal Graph  "
                f"(t_weight={args.t_weight}, grid={args.grid_cells}×{args.grid_cells}, "
                f"k={args.k_neighbors}, n_perm={args.n_perm})"
            ),
        )
        out_path = out_dir / f"phenotypic_graph_t{spatial_axis}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Branch allocation bars
    if branch_results:
        print("\nPlotting branch allocation bars...")
        fig2, _ = plot_branch_allocation_bars(branch_results, color_map=GENOTYPE_COLORS)
        bars_path = out_dir / "branch_allocation_bars.png"
        fig2.savefig(bars_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {bars_path}")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
