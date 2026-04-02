"""
10_principal_graph.py
---------------------
Fit a principal graph (MST on condition centroids) to the condensed PBX
trajectories, identify branch points, run permutation tests on genotype
branch-allocation distributions, and produce a 2D schematic figure.

Statistical framework
---------------------
At each branch node (degree >= 3), we test:
  H₀: genotype is independent of which outgoing edge an embryo flows into.

We assign each embryo near the branch node to the outgoing edge whose far
endpoint centroid is closest to its 2D condensed position, then run a
permutation chi-square test (1000 resamples, shuffle genotype labels).

Outputs
-------
  phenotypic_graph.png         — 2D schematic with branch point annotations
  branch_allocation_bars.png   — per-branch stacked bar charts
  branch_test_summary.csv      — one row per branch node (p-val, effect size)
  node_assignments.csv         — per-embryo edge assignments

Usage
-----
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260329_pbx_crispant_analysis_cont/10_principal_graph.py \\
    --positions-npz results/mcolon/20260329_pbx_crispant_analysis_cont/results/\\
pairwise_shrunk_condensation_aligned_umap_bin4_perm500/condensed_positions.npz \\
    --output-dir results/mcolon/20260329_pbx_crispant_analysis_cont/results/phenotypic_graph_v1

  # With custom parameters:
    --n-perm 1000
    --radius-factor 2.5
    --t-idx 5          (time bin index; default = most-observed bin)
    --seed 42
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
    build_embryo_mst,
    build_centroid_mst,
    contract_mst_skeleton,
    identify_branch_points,
    assign_embryos_to_subtrees,
    run_all_branch_tests,
    branch_results_to_df,
)
from trajectory_cosmology.principal_graph_viz import (
    plot_principal_graph,
    plot_branch_allocation_bars,
)


# ---------------------------------------------------------------------------
# Genotype color palette (consistent with rest of PBX analysis)
# ---------------------------------------------------------------------------

GENOTYPE_COLORS: dict[str, str] = {
    "inj_ctrl":               "#2166AC",   # blue
    "wik_ab":                 "#808080",   # gray
    "pbx1b_crispant":         "#9467bd",   # purple
    "pbx4_crispant":          "#F7B267",   # amber
    "pbx1b_pbx4_crispant":    "#B2182B",   # crimson
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phenotypic principal graph: MST + branch permutation tests."
    )
    p.add_argument(
        "--positions-npz",
        default=(
            "results/mcolon/20260329_pbx_crispant_analysis_cont/results/"
            "pairwise_shrunk_condensation_aligned_umap_bin4_perm500/"
            "condensed_positions.npz"
        ),
        help="Path to condensed_positions.npz",
    )
    p.add_argument(
        "--output-dir",
        default=(
            "results/mcolon/20260329_pbx_crispant_analysis_cont/results/"
            "phenotypic_graph_v1"
        ),
        help="Directory for output figures and CSVs",
    )
    p.add_argument("--n-perm", type=int, default=1000,
                   help="Number of permutations for branch tests (default 1000)")
    p.add_argument("--radius-factor", type=float, default=2.5,
                   help="Radius factor × mean edge length for embryo collection (default 2.5)")
    p.add_argument("--t-idx", type=int, default=None,
                   help="Time bin index to use (default: most-observed bin)")
    p.add_argument("--k-neighbors", type=int, default=6,
                   help="k for k-NN graph in embryo-level MST (default 6)")
    p.add_argument("--use-centroid-mst", action="store_true",
                   help="Use 5-node centroid MST instead of embryo-level k-NN MST")
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
    # 1. Load condensed positions
    # ------------------------------------------------------------------
    print(f"Loading condensed positions from {args.positions_npz}")
    npz = np.load(args.positions_npz, allow_pickle=True)
    positions   = npz["positions"]    # (N_e, T, 2)
    mask        = npz["mask"]         # (N_e, T)
    labels      = npz["labels"]       # (N_e,)
    time_values = npz["time_values"]  # (T,)

    print(f"  positions shape: {positions.shape}")
    print(f"  conditions: {sorted(np.unique(labels))}")
    print(f"  time bins: {len(time_values)}  ({time_values[0]:.0f}–{time_values[-1]:.0f} hpf)")

    # Determine layout time bin
    t_idx = args.t_idx
    if t_idx is None:
        t_idx = int(mask.sum(axis=0).argmax())
    print(f"  Using t_idx={t_idx} ({time_values[t_idx]:.0f} hpf) for graph layout "
          f"[{mask.sum(axis=0)[t_idx]} embryos observed]")

    # ------------------------------------------------------------------
    # 2. Fit MST
    # ------------------------------------------------------------------
    # ---- Determine the time bin to use for the graph layout ----
    # Use a high-separation time bin (78 hpf ≈ index 14 for this dataset)
    # unless overridden by --t-idx.
    if t_idx is None:
        # Pick the time bin with the highest mean inter-centroid separation
        conditions_all = np.unique(labels)
        best_t, best_sep = 0, -1.0
        for ti in range(len(time_values)):
            centroids_ti = {}
            for cond in conditions_all:
                idx = np.where(labels == cond)[0]
                obs = mask[idx, ti]
                if obs.sum() >= 3:
                    centroids_ti[cond] = positions[idx[obs], ti, :].mean(axis=0)
            if len(centroids_ti) < 3:
                continue
            cnames = list(centroids_ti.keys())
            dists = [
                np.linalg.norm(centroids_ti[cnames[i]] - centroids_ti[cnames[j]])
                for i in range(len(cnames)) for j in range(i + 1, len(cnames))
            ]
            mean_sep = float(np.mean(dists))
            n_obs = int(mask[:, ti].sum())
            # Require at least 30 embryos total
            if n_obs >= 30 and mean_sep > best_sep:
                best_sep = mean_sep
                best_t = ti
        t_idx = best_t
        print(f"  Auto-selected t_idx={t_idx} ({time_values[t_idx]:.0f} hpf) "
              f"— max mean separation {best_sep:.3f}")

    raw_nodes_df = None
    if args.use_centroid_mst:
        print("\nBuilding 5-node condition-centroid MST...")
        nodes_df, edges, adjacency = build_centroid_mst(
            positions, mask, labels, time_values,
            t_idx_for_layout=t_idx,
        )
        print(f"  Nodes: {len(nodes_df)},  Edges: {len(edges)}")
        print(nodes_df[["condition", "time_bin_center", "x", "y", "n_embryos"]].to_string(index=False))
    else:
        print(f"\nBuilding embryo-level k-NN MST at t={time_values[t_idx]:.0f} hpf "
              f"(k={args.k_neighbors})...")
        raw_nodes_df, raw_edges, raw_adjacency = build_embryo_mst(
            positions, mask, labels, time_values,
            t_idx=t_idx, k_neighbors=args.k_neighbors, seed=args.seed,
        )
        cond_counts = raw_nodes_df["condition"].value_counts().to_dict()
        print(f"  Raw MST — Nodes: {len(raw_nodes_df)},  Edges: {len(raw_edges)}")
        print(f"  Embryos per condition: {cond_counts}")

        print("  Contracting degree-2 chains → skeleton...")
        nodes_df, edges, adjacency = contract_mst_skeleton(
            raw_nodes_df, raw_edges, raw_adjacency,
        )
        print(f"  Skeleton — Nodes: {len(nodes_df)},  Edges: {len(edges)}")

    # ------------------------------------------------------------------
    # 3. Identify branch points
    # ------------------------------------------------------------------
    branch_node_ids = identify_branch_points(adjacency)
    degrees = adjacency.sum(axis=1)
    print(f"\nNode degrees: {dict(zip(nodes_df['condition'].tolist(), degrees.tolist()))}")
    if branch_node_ids:
        print(f"Branch nodes (degree >= 3): {branch_node_ids} → "
              f"{[nodes_df.loc[b, 'condition'] for b in branch_node_ids]}")
    else:
        print("No branch points found (all nodes have degree <= 2).")
        print("  This is expected for a path-like topology with 5 conditions.")
        print("  Saving graph schematic and exiting.")

    # ------------------------------------------------------------------
    # 4. Run permutation branch tests (even if no branches, produces empty CSV)
    # ------------------------------------------------------------------
    use_subtrees = not args.use_centroid_mst
    orig_nodes = raw_nodes_df if (not args.use_centroid_mst) else None
    print(f"\nRunning branch tests (n_perm={args.n_perm}, "
          f"method={'subtree-BFS' if use_subtrees else 'radius'})...")
    assignments_df, branch_results = run_all_branch_tests(
        positions, mask, labels, time_values,
        nodes_df, edges, adjacency, branch_node_ids,
        radius_factor=args.radius_factor,
        n_perm=args.n_perm,
        seed=args.seed,
        use_subtrees=use_subtrees,
        orig_nodes_df=orig_nodes,
    )

    if branch_results:
        for res in branch_results:
            sig = "***" if res.pval < 0.001 else ("*" if res.pval < 0.05 else "ns")
            print(f"  Node {res.node_id} ({nodes_df.loc[res.node_id, 'condition']}): "
                  f"p={res.pval:.4f} {sig},  V={res.effect_size:.3f},  "
                  f"n={res.n_embryos_tested}")
    else:
        print("  No branch nodes to test.")

    # ------------------------------------------------------------------
    # 5. Save CSVs
    # ------------------------------------------------------------------
    summary_df = branch_results_to_df(branch_results)
    summary_path = out_dir / "branch_test_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    assign_path = out_dir / "node_assignments.csv"
    assignments_df.to_csv(assign_path, index=False)
    print(f"Saved: {assign_path}")

    # ------------------------------------------------------------------
    # 6. Principal graph schematic
    # ------------------------------------------------------------------
    print("\nPlotting principal graph schematic...")
    fig, ax = plot_principal_graph(
        nodes_df=nodes_df,
        edges=edges,
        branch_results=branch_results,
        positions=positions,
        mask=mask,
        labels=labels,
        time_values=time_values,
        assignments_df=assignments_df if len(assignments_df) > 0 else None,
        color_map=GENOTYPE_COLORS,
        show_scatter=True,
        t_idx=t_idx,
        title=(
            f"Phenotypic Principal Graph  "
            f"(t={time_values[t_idx]:.0f} hpf, n_perm={args.n_perm})"
        ),
    )
    schematic_path = out_dir / "phenotypic_graph.png"
    fig.savefig(schematic_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {schematic_path}")

    # ------------------------------------------------------------------
    # 7. Branch allocation bar charts
    # ------------------------------------------------------------------
    if branch_results:
        print("Plotting branch allocation bars...")
        fig2, axes2 = plot_branch_allocation_bars(
            branch_results,
            color_map=GENOTYPE_COLORS,
        )
        bars_path = out_dir / "branch_allocation_bars.png"
        fig2.savefig(bars_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {bars_path}")

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
