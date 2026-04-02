"""
principal_graph.py
------------------
Fit a principal tree to condition-level centroids and test genotype
branch-allocation distributions at each branch point.

Statistical framework
---------------------
Nodes are geometry; the statistical object is the *distribution of embryos
across outgoing edges* at each branch node.

At branch point v with outgoing edges e_1 ... e_k:
  - Assign each embryo near v to the edge whose far endpoint it is closest to.
  - Compute counts[genotype, edge].
  - Permutation chi-square: shuffle genotype labels among all embryos near v,
    recompute counts, build null distribution → p-value.

Public API
----------
  build_centroid_mst(positions, mask, labels, time_values, auroc_df)
      -> (nodes_df, edges, adjacency)
  identify_branch_points(adjacency) -> list[int]
  assign_embryos_to_edges(positions, mask, labels, time_values,
                          nodes_df, edges, branch_node_ids, radius_factor)
      -> assignments_df
  permutation_branch_test(assignments_df, branch_node_id,
                          n_perm, seed) -> BranchTestResult
  BranchTestResult  (dataclass)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.csgraph


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BranchTestResult:
    """Statistical test result at one branch node.

    Attributes
    ----------
    node_id : int
        Index into nodes_df.
    clade_edges : list[int]
        Edge indices incident on this branch node.
    stat_obs : float
        Observed chi-square-like statistic.
    pval : float
        Permutation p-value (fraction of null stats >= stat_obs).
    effect_size : float
        Cramér's V — scale-free measure of association strength.
    counts_gk : pd.DataFrame
        counts[genotype × edge_index] — raw counts used for the test.
    n_embryos_tested : int
        Total embryos assigned at this branch across all genotypes.
    bifurcation_time_hpf : float | None
        Time bin centre where the test was run (if time-resolved).
    """
    node_id: int
    clade_edges: list[int]
    stat_obs: float
    pval: float
    effect_size: float
    counts_gk: pd.DataFrame
    n_embryos_tested: int
    bifurcation_time_hpf: float | None = None


# ---------------------------------------------------------------------------
# MST construction
# ---------------------------------------------------------------------------

def build_embryo_mst(
    positions: np.ndarray,          # (N_e, T, 2)
    mask: np.ndarray,               # (N_e, T)
    labels: np.ndarray,             # (N_e,)
    time_values: np.ndarray,        # (T,)
    t_idx: int | None = None,
    k_neighbors: int = 6,
    subsample_n: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray]:
    """Fit a k-NN MST on the full embryo point cloud at a single time bin.

    This produces genuine branch points because individual embryo positions
    can fan out from a common hub — unlike the 5-node centroid MST which
    is always a chain.

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T)
    labels : (N_e,)
    time_values : (T,)
    t_idx : int | None — time bin index; defaults to most-observed bin
    k_neighbors : int — k for k-NN graph construction
    subsample_n : int | None — if set, subsample embryos per condition to
        at most this number (useful for balanced graphs)
    seed : int — for subsampling reproducibility

    Returns
    -------
    nodes_df : DataFrame
        Columns: node_id, condition, x, y, time_bin_center
        One row per embryo node used.
    edges : list of (node_id_a, node_id_b)
    adjacency : (n_nodes, n_nodes) dense int array
    """
    from sklearn.neighbors import kneighbors_graph

    if t_idx is None:
        t_idx = int(mask.sum(axis=0).argmax())

    rng = np.random.default_rng(seed)

    # Collect observed embryos at t_idx
    obs_indices = np.where(mask[:, t_idx])[0]

    if subsample_n is not None:
        conditions = np.unique(labels[obs_indices])
        keep = []
        for cond in conditions:
            cond_obs = obs_indices[labels[obs_indices] == cond]
            if len(cond_obs) > subsample_n:
                chosen = rng.choice(cond_obs, size=subsample_n, replace=False)
            else:
                chosen = cond_obs
            keep.append(chosen)
        obs_indices = np.concatenate(keep)

    if len(obs_indices) < k_neighbors + 1:
        raise ValueError(
            f"Only {len(obs_indices)} embryos at t_idx={t_idx}; "
            f"need at least k_neighbors+1={k_neighbors+1}."
        )

    pts = positions[obs_indices, t_idx, :]   # (n_obs, 2)
    rows = []
    for local_i, emb_i in enumerate(obs_indices):
        rows.append({
            "node_id":         local_i,
            "embryo_idx":      int(emb_i),
            "condition":       str(labels[emb_i]),
            "x":               float(pts[local_i, 0]),
            "y":               float(pts[local_i, 1]),
            "time_bin_center": float(time_values[t_idx]),
        })
    nodes_df = pd.DataFrame(rows)

    # Build k-NN graph → MST
    knn = kneighbors_graph(pts, n_neighbors=k_neighbors, mode="distance",
                           include_self=False)
    mst_sparse = scipy.sparse.csgraph.minimum_spanning_tree(knn)
    mst_sym = mst_sparse + mst_sparse.T   # symmetrise
    mst_coo = scipy.sparse.coo_matrix(mst_sym)

    edges: list[tuple[int, int]] = []
    seen = set()
    for i, j in zip(mst_coo.row, mst_coo.col):
        key = (min(i, j), max(i, j))
        if key not in seen:
            edges.append(key)
            seen.add(key)

    n_nodes = len(nodes_df)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
    for (i, j) in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    return nodes_df, edges, adjacency


def contract_mst_skeleton(
    nodes_df: pd.DataFrame,
    edges: list[tuple[int, int]],
    adjacency: np.ndarray,
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray]:
    """Contract degree-2 chains to produce a skeleton with only branch points and leaves.

    In a raw k-NN MST most nodes are degree-2 "pass-through" nodes on chains
    between branch points or leaves.  This function collapses those chains
    so the output graph contains only:
      - Leaf nodes (degree == 1)
      - Branch nodes (degree >= 3)

    Each contracted edge carries the majority-vote condition label of the
    nodes it absorbed.

    Parameters
    ----------
    nodes_df : from build_embryo_mst
    edges : from build_embryo_mst
    adjacency : from build_embryo_mst

    Returns
    -------
    skel_nodes_df : DataFrame
        Columns: node_id, condition, x, y, time_bin_center, n_embryos
        Centroid of all original nodes merged into each skeleton node.
    skel_edges : list of (node_id_a, node_id_b)
    skel_adjacency : (n_skel_nodes, n_skel_nodes)
    """
    from collections import defaultdict

    n_nodes = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)

    # Skeleton nodes: leaves (degree 1) or branch points (degree >= 3)
    skel_mask = (degrees == 1) | (degrees >= 3)
    skel_node_ids = sorted(np.where(skel_mask)[0])

    if not skel_node_ids:
        # All nodes are degree-2; nothing to contract — return original
        return nodes_df, edges, adjacency

    skel_set = set(skel_node_ids)

    # For each skeleton node, find the set of original nodes it "owns"
    # (the degree-2 chain between it and the next skeleton node)
    # We do this by BFS/DFS from each skeleton node along degree-2 chains.
    owned: dict[int, list[int]] = defaultdict(list)
    for skel_id in skel_node_ids:
        owned[skel_id].append(skel_id)

    # Map each original node to the nearest skeleton node
    orig_to_skel = {}
    for skel_id in skel_node_ids:
        orig_to_skel[skel_id] = skel_id

    # Walk degree-2 chains
    for skel_id in skel_node_ids:
        for start_nbr in np.where(adjacency[skel_id] > 0)[0]:
            if int(start_nbr) in skel_set:
                continue
            # Walk the chain until we hit another skeleton node
            prev = skel_id
            cur = int(start_nbr)
            chain = [cur]
            while cur not in skel_set:
                nbrs = [int(n) for n in np.where(adjacency[cur] > 0)[0]
                        if int(n) != prev]
                if not nbrs:
                    break
                prev = cur
                cur = nbrs[0]
                if cur not in skel_set:
                    chain.append(cur)
            # Assign first half of chain to skel_id
            # (second half will be assigned when we walk from the other skel node)
            mid = len(chain) // 2
            for c in chain[:mid]:
                if c not in orig_to_skel:
                    orig_to_skel[c] = skel_id
                    owned[skel_id].append(c)

    # Assign any unvisited nodes to nearest skeleton node by position
    coords_all = nodes_df[["x", "y"]].values
    skel_coords = np.array([coords_all[s] for s in skel_node_ids])
    for i in range(n_nodes):
        if i not in orig_to_skel:
            dists = np.linalg.norm(skel_coords - coords_all[i], axis=1)
            nearest = skel_node_ids[int(np.argmin(dists))]
            orig_to_skel[i] = nearest
            owned[nearest].append(i)

    # Build skeleton nodes: centroid of owned nodes
    new_node_rows = []
    skel_id_to_new = {s: i for i, s in enumerate(skel_node_ids)}
    for skel_id in skel_node_ids:
        members = owned[skel_id]
        member_rows = nodes_df.iloc[members]
        new_node_rows.append({
            "node_id":         skel_id_to_new[skel_id],
            "orig_node_id":    skel_id,
            "condition":       member_rows["condition"].mode().iloc[0],
            "x":               float(member_rows["x"].mean()),
            "y":               float(member_rows["y"].mean()),
            "time_bin_center": float(member_rows["time_bin_center"].iloc[0]),
            "n_embryos":       len(members),
        })
    skel_nodes_df = pd.DataFrame(new_node_rows).reset_index(drop=True)

    # Build skeleton edges: unique pairs of skeleton nodes connected by
    # an original edge (after mapping through orig_to_skel)
    skel_edge_set = set()
    for (i, j) in edges:
        si = orig_to_skel.get(i, i)
        sj = orig_to_skel.get(j, j)
        si_new = skel_id_to_new.get(si)
        sj_new = skel_id_to_new.get(sj)
        if si_new is not None and sj_new is not None and si_new != sj_new:
            key = (min(si_new, sj_new), max(si_new, sj_new))
            skel_edge_set.add(key)
    skel_edges = list(skel_edge_set)

    n_skel = len(skel_nodes_df)
    skel_adj = np.zeros((n_skel, n_skel), dtype=int)
    for (i, j) in skel_edges:
        skel_adj[i, j] = 1
        skel_adj[j, i] = 1

    # Also store owned node indices per skeleton node for subtree tests
    skel_nodes_df["owned_orig_nodes"] = [
        owned[skel_id] for skel_id in skel_node_ids
    ]

    return skel_nodes_df, skel_edges, skel_adj


def build_centroid_mst(
    positions: np.ndarray,          # (N_e, T, 2)
    mask: np.ndarray,               # (N_e, T)
    labels: np.ndarray,             # (N_e,)
    time_values: np.ndarray,        # (T,)
    auroc_df: pd.DataFrame | None = None,
    late_bins_n: int = 8,
    t_idx_for_layout: int | None = None,
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray]:
    """Fit a Minimum Spanning Tree on condition centroids.

    Each node in the MST represents one (condition, time_bin) pair where
    a centroid can be computed.  Edge weights are Euclidean distances between
    centroids in condensed 2D space.  When auroc_df is provided the final MST
    topology is also recorded but the edge-weight distance metric stays
    geometric (spatial) so the layout reflects the actual embedding.

    For the primary statistical uses (branch tests) we aggregate over all
    observed time bins.  For the schematic layout we use a representative
    time slice specified by t_idx_for_layout.

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T)
    labels : (N_e,)
    time_values : (T,)
    auroc_df : optional — contrast_specificity_by_timebin.csv, used only for
        recording cross-clade AUROC in the returned nodes_df.
    late_bins_n : int
        Number of late time bins used to build the aggregate distance matrix
        for UPGMA-style topology determination.
    t_idx_for_layout : int | None
        Index into time_values for the representative layout slice.
        Defaults to the time bin with the most observed embryos.

    Returns
    -------
    nodes_df : DataFrame
        Columns: node_id, condition, time_bin_center, x, y, n_embryos
        One row per (condition, time_bin) with a valid centroid.
    edges : list of (node_id_a, node_id_b)
        MST edges as pairs of node_df row indices.
    adjacency : (n_nodes, n_nodes) dense int array
        Adjacency matrix of the MST (symmetric, 0/1).

    Notes
    -----
    The MST is fit separately per time bin.  For the branch tests we use
    the time-aggregated MST (fitted on centroids stacked across all T bins).
    An additional "summary MST" is also fitted on condition-level mean
    positions averaged across the late bins — this is used for the schematic.
    """
    conditions = np.unique(labels)
    n_conditions = len(conditions)
    cond_idx_map = {c: np.where(labels == c)[0] for c in conditions}

    # ------------------------------------------------------------------
    # 1. Determine representative layout time bin
    # ------------------------------------------------------------------
    if t_idx_for_layout is None:
        t_idx_for_layout = int(mask.sum(axis=0).argmax())

    # ------------------------------------------------------------------
    # 2. Build per-condition centroids at the layout time bin
    # ------------------------------------------------------------------
    rows = []
    for cond in conditions:
        idx = cond_idx_map[cond]
        obs = mask[idx, t_idx_for_layout]
        if obs.sum() == 0:
            continue
        pts = positions[idx[obs], t_idx_for_layout, :]
        rows.append({
            "node_id":        len(rows),
            "condition":      cond,
            "time_bin_center": float(time_values[t_idx_for_layout]),
            "x":              pts[:, 0].mean(),
            "y":              pts[:, 1].mean(),
            "n_embryos":      int(obs.sum()),
        })
    nodes_df = pd.DataFrame(rows).reset_index(drop=True)

    if len(nodes_df) < 2:
        raise ValueError("Fewer than 2 conditions with observations at layout time bin.")

    # ------------------------------------------------------------------
    # 3. Build pairwise Euclidean distance matrix on centroid positions
    # ------------------------------------------------------------------
    coords = nodes_df[["x", "y"]].values          # (n_nodes, 2)
    n_nodes = len(nodes_df)
    dist_matrix = np.full((n_nodes, n_nodes), np.inf)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    # ------------------------------------------------------------------
    # 4. Fit MST via scipy
    # ------------------------------------------------------------------
    sparse_dist = scipy.sparse.csr_matrix(dist_matrix)
    mst_sparse = scipy.sparse.csgraph.minimum_spanning_tree(sparse_dist)
    mst_coo = mst_sparse.tocoo()

    edges: list[tuple[int, int]] = []
    for i, j in zip(mst_coo.row, mst_coo.col):
        edges.append((int(i), int(j)))

    # Build symmetric adjacency
    adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
    for (i, j) in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    return nodes_df, edges, adjacency


# ---------------------------------------------------------------------------
# Branch point detection
# ---------------------------------------------------------------------------

def identify_branch_points(adjacency: np.ndarray) -> list[int]:
    """Return node indices where degree >= 3 (branch points in the MST).

    Parameters
    ----------
    adjacency : (n_nodes, n_nodes) int array — symmetric adjacency matrix

    Returns
    -------
    List of node indices (row indices in nodes_df) with degree >= 3.
    """
    degrees = adjacency.sum(axis=1)
    return [int(i) for i in np.where(degrees >= 3)[0]]


# ---------------------------------------------------------------------------
# Embryo-to-edge assignment
# ---------------------------------------------------------------------------

def assign_embryos_to_edges(
    positions: np.ndarray,          # (N_e, T, 2)
    mask: np.ndarray,               # (N_e, T)
    labels: np.ndarray,             # (N_e,)
    time_values: np.ndarray,        # (T,)
    nodes_df: pd.DataFrame,
    edges: list[tuple[int, int]],
    branch_node_ids: list[int],
    radius_factor: float = 2.5,
    t_idx: int | None = None,
) -> pd.DataFrame:
    """Assign each embryo to the nearest outgoing edge at each branch node.

    For each branch node v and each embryo i observed at time t_idx:
      - Compute distance from embryo position to v's centroid.
      - If within radius (radius_factor × mean inter-centroid distance),
        assign the embryo to the outgoing edge whose *far endpoint* centroid
        is closest to the embryo.

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T)
    labels : (N_e,)
    time_values : (T,)
    nodes_df : from build_centroid_mst
    edges : from build_centroid_mst
    branch_node_ids : from identify_branch_points
    radius_factor : float
        Controls how far from a branch node to collect embryos.
        Radius = radius_factor × mean_edge_length.
    t_idx : int | None
        Time bin index to use.  Defaults to the time bin recorded in nodes_df
        (the layout time bin).

    Returns
    -------
    assignments_df : DataFrame
        Columns: embryo_idx, genotype, branch_node_id, edge_idx,
                 dist_to_node, time_bin_center
        One row per (embryo, branch_node) assignment.
    """
    if t_idx is None:
        # Use the time bin stored in nodes_df
        t_val = nodes_df["time_bin_center"].iloc[0]
        t_idx = int(np.searchsorted(time_values, t_val))

    # Compute mean edge length for radius
    coords = nodes_df[["x", "y"]].values
    edge_lengths = [
        np.linalg.norm(coords[i] - coords[j])
        for (i, j) in edges
    ]
    mean_edge_len = float(np.mean(edge_lengths)) if edge_lengths else 1.0
    radius = radius_factor * mean_edge_len

    rows = []
    for bn in branch_node_ids:
        branch_centroid = coords[bn]
        # Find outgoing edges from this branch node
        incident_edges = [(eidx, i, j) for eidx, (i, j) in enumerate(edges)
                          if i == bn or j == bn]
        if len(incident_edges) < 2:
            continue
        # Far endpoint of each incident edge
        far_endpoints = []
        for eidx, i, j in incident_edges:
            far = j if i == bn else i
            far_endpoints.append((eidx, far))

        # Collect embryos within radius
        for emb_idx in range(positions.shape[0]):
            if not mask[emb_idx, t_idx]:
                continue
            emb_pos = positions[emb_idx, t_idx, :]
            dist_to_node = float(np.linalg.norm(emb_pos - branch_centroid))
            if dist_to_node > radius:
                continue
            # Assign to nearest far-endpoint
            dists_to_far = [
                np.linalg.norm(emb_pos - coords[far])
                for (_, far) in far_endpoints
            ]
            best = int(np.argmin(dists_to_far))
            assigned_edge_idx = far_endpoints[best][0]
            rows.append({
                "embryo_idx":      emb_idx,
                "genotype":        str(labels[emb_idx]),
                "branch_node_id":  bn,
                "edge_idx":        assigned_edge_idx,
                "dist_to_node":    dist_to_node,
                "time_bin_center": float(time_values[t_idx]),
            })

    return pd.DataFrame(rows)


def assign_embryos_to_subtrees(
    nodes_df: pd.DataFrame,
    adjacency: np.ndarray,
    branch_node_ids: list[int],
    orig_nodes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assign graph nodes to subtrees for each branch node via BFS.

    Works with both raw embryo-level nodes_df and skeleton nodes_df
    (from contract_mst_skeleton).  If nodes_df has an 'owned_orig_nodes'
    column (skeleton case), each skeleton node's arm membership is expanded
    to cover all original embryo nodes it owns.

    Parameters
    ----------
    nodes_df : DataFrame — from build_embryo_mst or contract_mst_skeleton
    adjacency : (n_nodes, n_nodes) symmetric adjacency matrix
    branch_node_ids : list of branch node indices
    orig_nodes_df : optional original embryo-level nodes_df; used when
        nodes_df is a contracted skeleton so arm counts reflect individual
        embryos, not skeleton nodes.

    Returns
    -------
    assignments_df : DataFrame
        Columns: node_id, embryo_idx, genotype, branch_node_id, edge_idx,
                 time_bin_center
        One row per (embryo, branch_node) assignment.
    """
    from collections import deque

    rows = []
    t_hpf = float(nodes_df["time_bin_center"].iloc[0])
    has_owned = "owned_orig_nodes" in nodes_df.columns

    for bn in branch_node_ids:
        neighbours = list(np.where(adjacency[bn] > 0)[0])
        if len(neighbours) < 2:
            continue

        visited_global = set([bn])
        arms: list[list[int]] = []
        for start in neighbours:
            if start in visited_global:
                continue
            arm: list[int] = []
            queue = deque([start])
            visited_global.add(start)
            while queue:
                node = queue.popleft()
                arm.append(node)
                for nbr in np.where(adjacency[node] > 0)[0]:
                    if int(nbr) not in visited_global:
                        visited_global.add(int(nbr))
                        queue.append(int(nbr))
            arms.append(arm)

        for arm_idx, arm in enumerate(arms):
            for node_id in arm:
                row_data = nodes_df.iloc[node_id]
                if has_owned and orig_nodes_df is not None:
                    owned_orig = row_data["owned_orig_nodes"]
                    for orig_id in owned_orig:
                        orig_row = orig_nodes_df.iloc[orig_id]
                        emb_idx = int(orig_row.get("embryo_idx", orig_id))
                        rows.append({
                            "node_id":         orig_id,
                            "embryo_idx":      emb_idx,
                            "genotype":        str(orig_row["condition"]),
                            "branch_node_id":  bn,
                            "edge_idx":        arm_idx,
                            "time_bin_center": t_hpf,
                        })
                else:
                    emb_idx = int(row_data.get("embryo_idx", node_id))
                    rows.append({
                        "node_id":         node_id,
                        "embryo_idx":      emb_idx,
                        "genotype":        str(row_data["condition"]),
                        "branch_node_id":  bn,
                        "edge_idx":        arm_idx,
                        "time_bin_center": t_hpf,
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "node_id", "embryo_idx", "genotype",
            "branch_node_id", "edge_idx", "time_bin_center",
        ])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def _chi_square_stat(counts: np.ndarray) -> float:
    """Chi-square statistic for a (G × K) contingency table.

    Returns 0.0 if any marginal is zero or if the table has a single row/col.
    """
    row_sums = counts.sum(axis=1, keepdims=True)  # (G, 1)
    col_sums = counts.sum(axis=0, keepdims=True)  # (1, K)
    total = counts.sum()
    if total == 0 or counts.shape[0] < 2 or counts.shape[1] < 2:
        return 0.0
    expected = (row_sums * col_sums) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(expected > 0, (counts - expected) ** 2 / expected, 0.0)
    return float(stat.sum())


def _cramers_v(stat: float, n: int, min_dim: int) -> float:
    """Cramér's V from chi-square statistic."""
    if n == 0 or min_dim <= 1:
        return 0.0
    return float(np.sqrt(stat / (n * (min_dim - 1))))


def permutation_branch_test(
    assignments_df: pd.DataFrame,
    branch_node_id: int,
    n_perm: int = 1000,
    seed: int = 42,
) -> BranchTestResult:
    """Permutation chi-square test at one branch node.

    Tests H₀: genotype labels are independent of edge assignment.

    Permutation procedure:
      For each resample, shuffle genotype labels among all embryos assigned
      to this branch node, recompute the chi-square statistic, build null
      distribution.  p-value = fraction of null stats >= observed stat.

    Parameters
    ----------
    assignments_df : from assign_embryos_to_edges
    branch_node_id : int
    n_perm : int
    seed : int

    Returns
    -------
    BranchTestResult
    """
    sub = assignments_df[assignments_df["branch_node_id"] == branch_node_id].copy()

    rng = np.random.default_rng(seed)

    genotypes = sorted(sub["genotype"].unique())
    edge_idxs = sorted(sub["edge_idx"].unique())
    n_embryos = len(sub)

    # Observed contingency table
    counts_obs = np.zeros((len(genotypes), len(edge_idxs)), dtype=float)
    g_map = {g: i for i, g in enumerate(genotypes)}
    e_map = {e: i for i, e in enumerate(edge_idxs)}
    for _, row in sub.iterrows():
        counts_obs[g_map[row["genotype"]], e_map[row["edge_idx"]]] += 1

    stat_obs = _chi_square_stat(counts_obs)

    # Permutation null
    genotype_arr = sub["genotype"].values.copy()
    edge_arr = sub["edge_idx"].values.copy()

    null_stats = np.zeros(n_perm, dtype=float)
    for k in range(n_perm):
        shuffled = rng.permutation(genotype_arr)
        counts_null = np.zeros_like(counts_obs)
        for gi, ei in zip(shuffled, edge_arr):
            counts_null[g_map[gi], e_map[ei]] += 1
        null_stats[k] = _chi_square_stat(counts_null)

    pval = float((null_stats >= stat_obs).mean())
    # Avoid pval=0 floor: use 1/n_perm as minimum
    pval = max(pval, 1.0 / n_perm)

    min_dim = min(len(genotypes), len(edge_idxs))
    effect_size = _cramers_v(stat_obs, n_embryos, min_dim)

    t_hpf = float(sub["time_bin_center"].iloc[0]) if len(sub) > 0 else None

    counts_df = pd.DataFrame(
        counts_obs,
        index=genotypes,
        columns=[f"edge_{e}" for e in edge_idxs],
    )

    return BranchTestResult(
        node_id=branch_node_id,
        clade_edges=edge_idxs,
        stat_obs=stat_obs,
        pval=pval,
        effect_size=effect_size,
        counts_gk=counts_df,
        n_embryos_tested=n_embryos,
        bifurcation_time_hpf=t_hpf,
    )


def run_all_branch_tests(
    positions: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    nodes_df: pd.DataFrame,
    edges: list[tuple[int, int]],
    adjacency: np.ndarray,
    branch_node_ids: list[int],
    radius_factor: float = 2.5,
    n_perm: int = 1000,
    seed: int = 42,
    use_subtrees: bool = False,
    orig_nodes_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[BranchTestResult]]:
    """Convenience wrapper: assign embryos + run test at every branch node.

    Parameters
    ----------
    use_subtrees : bool
        If True, use subtree-BFS assignment (for embryo-level MST).
        If False, use radius-based centroid assignment.

    Returns
    -------
    assignments_df : full embryo-to-edge assignment table
    results : list of BranchTestResult, one per branch node
    """
    if use_subtrees:
        assignments_df = assign_embryos_to_subtrees(
            nodes_df, adjacency, branch_node_ids,
            orig_nodes_df=orig_nodes_df,
        )
    else:
        assignments_df = assign_embryos_to_edges(
            positions, mask, labels, time_values,
            nodes_df, edges, branch_node_ids,
            radius_factor=radius_factor,
        )
    results = []
    for bn in branch_node_ids:
        res = permutation_branch_test(assignments_df, bn, n_perm=n_perm, seed=seed)
        results.append(res)
    return assignments_df, results


def branch_results_to_df(results: list[BranchTestResult]) -> pd.DataFrame:
    """Flatten BranchTestResult list to a summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "node_id":              r.node_id,
            "n_edges":              len(r.clade_edges),
            "n_embryos_tested":     r.n_embryos_tested,
            "stat_obs":             round(r.stat_obs, 4),
            "pval":                 round(r.pval, 4),
            "effect_size_cramers_v": round(r.effect_size, 4),
            "bifurcation_time_hpf": r.bifurcation_time_hpf,
            "significant_p05":      r.pval < 0.05,
        })
    return pd.DataFrame(rows)
