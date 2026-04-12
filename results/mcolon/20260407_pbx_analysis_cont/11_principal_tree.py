"""
11_principal_tree.py
--------------------
Fit an elastic principal tree to the 4-class condensed trajectories using ElPiGraph.
Saves branches/segments as first-class objects (stable UIDs, projection CSV, NPZ
artifact pack) so any future interactive tool can subset embryos by segment without
re-fitting.

Usage
-----
  conda run -n segmentation_grounded_sam --no-capture-output python \\
    results/mcolon/20260407_pbx_analysis_cont/11_principal_tree.py \\
    --positions-npz <path>/condensed_positions.npz \\
    --output-dir <out_dir>

  # Dry run (print shapes only, no fitting):
    ... --dry-run

Parameters
----------
  --n-nodes       20     ElPiGraph tree nodes (default)
  --lambda-elpi   0.01   elasticity penalty
  --mu-elpi       0.1    bending/branching penalty
  --n-perm        1000   permutations for branch tests
  --seed          42
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_20260407_principal_tree_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


_configure_runtime_env()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.trajectory_condensation.principal_tree import (
    build_embryo_spacetime_cloud,
    fit_principal_tree,
    project_observations_to_tree,
    identify_branch_nodes,
    run_all_branch_tests,
    branch_results_to_df,
    extract_segments,
    prune_phantom_segments,
    segments_to_edges_df,
    plot_tree_schematic,
    plot_branch_allocation_bars,
    plot_tree_3d,
    save_tree_3d_gif,
)
from common import GENOTYPE_COLORS

# t_weight is a fixed constant (not a CLI arg) — changing it would invalidate all
# existing segment UIDs. It is recorded in metadata.json for provenance.
_T_WEIGHT = 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ElPiGraph principal tree on 4-class condensed trajectories."
    )
    p.add_argument("--positions-npz", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-nodes", type=int, default=20,
                   help="ElPiGraph tree nodes (default 20)")
    p.add_argument("--lambda-elpi", type=float, default=0.01,
                   help="ElPiGraph Lambda — elasticity penalty (default 0.01)")
    p.add_argument("--mu-elpi", type=float, default=0.1,
                   help="ElPiGraph Mu — bending/branching penalty (default 0.1)")
    p.add_argument("--n-perm", type=int, default=1000,
                   help="Permutations for branch enrichment test (default 1000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Load NPZ and print shapes/labels only, no fitting")
    return p.parse_args()


def _segment_uid(seg: list[tuple[int, int]]) -> str:
    """Stable 10-char hex UID for a segment based on its sorted edge set.

    Stable across reruns as long as tree topology is identical.
    Changes if tree refitting produces different node IDs.
    """
    sig = repr(sorted([(min(a, b), max(a, b)) for a, b in seg]))
    return hashlib.sha1(sig.encode()).hexdigest()[:10]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load condensed positions NPZ
    # Invariant: labels[i] corresponds to embryo_idx == i in positions cube.
    # ------------------------------------------------------------------
    print(f"Loading: {args.positions_npz}")
    npz = np.load(args.positions_npz, allow_pickle=True)
    positions   = npz["positions"]                # (N_e, T, 2)
    mask        = npz["mask"].astype(bool)        # (N_e, T)
    labels      = npz["labels"].astype(str)       # (N_e,) — index-aligned with positions
    time_values = npz["time_values"]              # (T,)
    embryo_ids  = npz["embryo_ids"].astype(str) if "embryo_ids" in npz else None

    N_e, T, _ = positions.shape
    n_obs = int(mask.sum())
    conditions = sorted(set(labels.tolist()))
    print(f"  {N_e} embryos × {T} time bins, {n_obs} observations")
    print(f"  Labels: {conditions}")
    print(f"  Time: {time_values[0]:.1f}–{time_values[-1]:.1f} hpf")

    if args.dry_run:
        print("--dry-run: done.")
        return

    # ------------------------------------------------------------------
    # 2. Build embryo-level 3D space-time cloud
    # ------------------------------------------------------------------
    print(f"\nBuilding space-time cloud (t_weight={_T_WEIGHT})...")
    pts_3d, obs_df = build_embryo_spacetime_cloud(
        positions, mask, time_values, t_weight=_T_WEIGHT,
    )
    if embryo_ids is not None:
        obs_df["embryo_id"] = embryo_ids[obs_df["embryo_idx"].values]
    print(f"  {len(pts_3d)} observations in 3D cloud")

    # ------------------------------------------------------------------
    # 3. Fit elastic principal tree
    # ------------------------------------------------------------------
    print(f"\nFitting principal tree "
          f"(n_nodes={args.n_nodes}, Lambda={args.lambda_elpi}, Mu={args.mu_elpi})...")
    nodes_df, edges_df, _raw = fit_principal_tree(
        pts_3d,
        n_nodes=args.n_nodes,
        Lambda=args.lambda_elpi,
        Mu=args.mu_elpi,
        verbose=False,
    )
    degree_dist = nodes_df["degree"].value_counts().sort_index().to_dict()
    print(f"  {len(nodes_df)} nodes, {len(edges_df)} edges")
    print(f"  Degree distribution: {degree_dist}")

    # ------------------------------------------------------------------
    # 4. Project observations to nearest tree edge
    # ------------------------------------------------------------------
    print("\nProjecting observations to tree...")
    proj_df = project_observations_to_tree(pts_3d, obs_df, nodes_df, edges_df)

    # ------------------------------------------------------------------
    # 5. Segments — first-class objects with stable UIDs
    # ------------------------------------------------------------------
    print("\nExtracting segments...")
    segments = extract_segments(nodes_df, edges_df)
    segments = prune_phantom_segments(segments, proj_df, min_embryos=1)
    print(f"  {len(segments)} segments after pruning")

    # Build edge → segment mapping and assign segment_id to each observation.
    # Vectorized via string keys to avoid iterrows.
    segments_rows = []
    edge_to_seg: dict[tuple[int, int], int] = {}
    for seg_id, seg in enumerate(segments):
        uid = _segment_uid(seg)
        edges_in_seg = [(min(a, b), max(a, b)) for a, b in seg]
        for ek in edges_in_seg:
            edge_to_seg[ek] = seg_id
        segments_rows.append({
            "segment_id":  seg_id,
            "segment_uid": uid,
            "n_edges":     len(seg),
            "edges_list":  str(edges_in_seg),
        })
    segments_df = pd.DataFrame(segments_rows)

    ea = proj_df["nearest_edge_a"].values.astype(int)
    eb = proj_df["nearest_edge_b"].values.astype(int)
    edge_key_series = pd.Series([f"{min(a,b)},{max(a,b)}" for a, b in zip(ea, eb)])
    seg_key_map = {f"{min(a,b)},{max(a,b)}": sid for (a, b), sid in edge_to_seg.items()}
    proj_df["segment_id"] = edge_key_series.map(seg_key_map).fillna(-1).astype(int).values

    uid_map: dict[int, str] = dict(zip(segments_df["segment_id"], segments_df["segment_uid"]))
    proj_df["segment_uid"] = proj_df["segment_id"].map(uid_map).fillna("")

    # ------------------------------------------------------------------
    # 6. Branch tests
    # ------------------------------------------------------------------
    branch_node_ids = identify_branch_nodes(nodes_df)
    print(f"\nBranch nodes (degree ≥ 3): {branch_node_ids}")

    print(f"Running branch tests (n_perm={args.n_perm}, unit=embryo)...")
    assignments_df, branch_results = run_all_branch_tests(
        proj_df, nodes_df, edges_df, branch_node_ids, labels,
        n_perm=args.n_perm, seed=args.seed,
    )
    for res in branch_results:
        star = ("***" if res.pval < 0.001 else "**" if res.pval < 0.01
                else "*" if res.pval < 0.05 else "ns")
        print(f"  Node {res.node_id}: p={res.pval:.4f} {star}  "
              f"V={res.effect_size:.3f}  n={res.n_embryos}")

    # ------------------------------------------------------------------
    # 7. Per-embryo segment membership (dominant segment, vectorized)
    # ------------------------------------------------------------------
    seg_mode = (
        proj_df[proj_df["segment_id"] >= 0]
        .groupby("embryo_idx")["segment_id"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .reset_index()
        .rename(columns={"segment_id": "dominant_segment_id"})
    )
    seg_mode["dominant_segment_uid"] = seg_mode["dominant_segment_id"].map(uid_map).fillna("")
    if embryo_ids is not None:
        seg_mode["embryo_id"] = embryo_ids[seg_mode["embryo_idx"].values]
    seg_mode["label"] = labels[seg_mode["embryo_idx"].values]

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_df.to_csv(out_dir / "tree_nodes.csv", index=False)
    edges_df.to_csv(out_dir / "tree_edges.csv", index=False)
    segments_df.to_csv(out_dir / "segments.csv", index=False)
    proj_df.to_csv(out_dir / "observation_projections.csv", index=False)
    assignments_df.to_csv(out_dir / "embryo_branch_assignments.csv", index=False)
    branch_results_to_df(branch_results).to_csv(out_dir / "branch_test_summary.csv", index=False)
    seg_mode.to_csv(out_dir / "embryo_segment_membership.csv", index=False)
    print(f"\nSaved CSVs to: {out_dir}")

    # NPZ artifact pack — compact fast-loader for interactive tools
    npz_kwargs: dict = dict(
        node_positions=nodes_df[["x", "y", "t_scaled"]].values,
        node_degrees=nodes_df["degree"].values,
        edge_pairs=edges_df[["source", "target"]].values,
        obs_embryo_idx=proj_df["embryo_idx"].values,
        obs_time_idx=proj_df["time_idx"].values,
        obs_nearest_edge_a=proj_df["nearest_edge_a"].values,
        obs_nearest_edge_b=proj_df["nearest_edge_b"].values,
        obs_proj_frac=proj_df["proj_frac"].values,
        obs_dist_to_edge=proj_df["dist_to_edge"].values,
        obs_segment_id=proj_df["segment_id"].values,
        labels=labels,
        time_values=time_values,
    )
    if embryo_ids is not None:
        npz_kwargs["embryo_ids"] = embryo_ids
    np.savez_compressed(out_dir / "tree_artifacts.npz", **npz_kwargs)
    print(f"Saved: tree_artifacts.npz")

    # Provenance
    metadata = {
        "input_npz":  str(args.positions_npz),
        "git_commit": _git_hash(),
        "timestamp":  datetime.datetime.now().isoformat(),
        "args": {
            "n_nodes":     args.n_nodes,
            "lambda_elpi": args.lambda_elpi,
            "mu_elpi":     args.mu_elpi,
            "n_perm":      args.n_perm,
            "seed":        args.seed,
        },
        "fixed_constants": {"t_weight": _T_WEIGHT},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved: metadata.json")

    # ------------------------------------------------------------------
    # 9. Visualizations
    # ------------------------------------------------------------------
    t_min_hpf = float(time_values.min())
    t_max_hpf = float(time_values.max())
    title_base = (
        f"Phenotypic Principal Tree  "
        f"(ElPiGraph n_nodes={args.n_nodes}, t_weight={_T_WEIGHT}, n_perm={args.n_perm})"
    )

    for spatial_axis in ("y", "x"):
        print(f"\nPlotting (t, {spatial_axis}) schematic...")
        fig, _ = plot_tree_schematic(
            nodes_df=nodes_df,
            edges_df=edges_df,
            branch_results=branch_results,
            obs_df=obs_df,
            labels=labels,
            color_map=GENOTYPE_COLORS,
            spatial_axis=spatial_axis,
            t_min_hpf=t_min_hpf,
            t_max_hpf=t_max_hpf,
            t_weight=_T_WEIGHT,
            annotate_ns=False,
            min_n_embryos=5,
            title=title_base,
        )
        fig.savefig(out_dir / f"principal_tree_t{spatial_axis}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("\nPlotting 3D diagnostic...")
    fig3d, _ = plot_tree_3d(
        nodes_df=nodes_df,
        edges_df=edges_df,
        obs_df=obs_df,
        labels=labels,
        color_map=GENOTYPE_COLORS,
        title=f"Principal Tree 3D  (n_nodes={args.n_nodes}, t_weight={_T_WEIGHT})",
    )
    fig3d.savefig(out_dir / "principal_tree_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig3d)

    print("Rendering rotating 3D GIF...")
    save_tree_3d_gif(
        nodes_df=nodes_df,
        edges_df=edges_df,
        obs_df=obs_df,
        labels=labels,
        out_path=str(out_dir / "principal_tree_3d_rotate.gif"),
        color_map=GENOTYPE_COLORS,
        figsize=(6, 5),
        n_frames=36,
        fps=12,
        title=f"Principal Tree 3D  (n_nodes={args.n_nodes}, t_weight={_T_WEIGHT})",
    )

    if branch_results:
        print("Plotting branch allocation bars...")
        fig2, _ = plot_branch_allocation_bars(branch_results, color_map=GENOTYPE_COLORS)
        fig2.savefig(out_dir / "branch_allocation_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)

    print(f"\nDone. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
