"""
candidate_tree_pass.py
----------------------
Second-pass principal-graph extraction for ranked condensation iterations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .principal_graph import (
    branch_results_to_df,
    build_mean_trajectories,
    build_trajectory_mst,
    contract_to_skeleton,
    identify_branch_points,
    run_all_branch_tests,
)
from .principal_graph_viz import plot_branch_allocation_bars, plot_spacetime_schematic


def run_tree_pass(
    selected_iteration_dirs: list[Path],
    *,
    output_root: Path,
    color_map: dict[str, str],
    t_weight: float,
    grid_cells: int,
    k_neighbors: int,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    del grid_cells, k_neighbors
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for selected_dir in selected_iteration_dirs:
        npz_path = selected_dir / "positions_iter.npz"
        meta_path = selected_dir / "metadata.json"
        if not npz_path.exists():
            continue

        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        npz = np.load(npz_path, allow_pickle=True)
        positions = npz["positions"]
        mask = npz["mask"]
        labels = npz["labels"]
        time_values = npz["time_values"]
        snapshot_iter = int(npz["snapshot_iter"])

        out_dir = output_root / selected_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        traj_df = build_mean_trajectories(positions, mask, labels, time_values)
        nodes_df, edges, adjacency = build_trajectory_mst(traj_df, t_weight=t_weight)
        skel_nodes_df, skel_edges, skel_adj, owned_nodes = contract_to_skeleton(nodes_df, edges, adjacency)
        if "n_obs" not in skel_nodes_df.columns and "n_orig" in skel_nodes_df.columns:
            skel_nodes_df = skel_nodes_df.copy()
            skel_nodes_df["n_obs"] = skel_nodes_df["n_orig"]
        branch_node_ids = identify_branch_points(skel_adj)
        assignments_df, branch_results = run_all_branch_tests(
            skel_nodes_df,
            skel_adj,
            branch_node_ids,
            owned_nodes,
            traj_df,
            positions,
            mask,
            labels,
            time_values,
            n_perm=n_perm,
            seed=seed,
        )

        skel_nodes_df.to_csv(out_dir / "skeleton_nodes.csv", index=False)
        assignments_df.to_csv(out_dir / "node_assignments.csv", index=False)
        branch_df = branch_results_to_df(branch_results)
        branch_df.to_csv(out_dir / "branch_test_summary.csv", index=False)

        for spatial_axis in ("y", "x"):
            fig, _ = plot_spacetime_schematic(
                skel_nodes_df=skel_nodes_df,
                skel_edges=skel_edges,
                branch_results=branch_results,
                positions=positions,
                mask=mask,
                labels=labels,
                time_values=time_values,
                color_map=color_map,
                spatial_axis=spatial_axis,
                annotate_ns=False,
                min_n_embryos=20,
                title=(
                    f"Candidate principal graph | iter {snapshot_iter} | "
                    f"t_weight={t_weight} n_perm={n_perm}"
                ),
            )
            fig.savefig(out_dir / f"phenotypic_graph_t{spatial_axis}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        fig, _ = plot_branch_allocation_bars(branch_results, color_map=color_map)
        fig.savefig(out_dir / "branch_allocation_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        min_pval = float(branch_df["pval"].min()) if not branch_df.empty else float("nan")
        effect_col = "effect_size_cramers_v" if "effect_size_cramers_v" in branch_df.columns else "effect_size"
        max_effect = float(branch_df[effect_col].max()) if (not branch_df.empty and effect_col in branch_df.columns) else float("nan")
        n_sig = int((branch_df["pval"] < 0.05).sum()) if not branch_df.empty else 0

        summary_rows.append({
            "candidate_dir": selected_dir.name,
            "iter": snapshot_iter,
            "rank": meta.get("rank"),
            "selection_score": meta.get("selection_score"),
            "n_branch_nodes": int(len(branch_node_ids)),
            "n_sig_branch_nodes": n_sig,
            "min_pval": min_pval,
            "max_effect_size": max_effect,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "tree_candidate_summary.csv", index=False)
    return summary_df
