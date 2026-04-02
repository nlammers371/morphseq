"""
principal_graph.py  (v2)
------------------------
Fit a principal tree to the (x, y, t) space-time trajectory cloud and test
whether genotypes distribute non-uniformly across branch arms.

Design decisions
----------------
* Graph is fitted on a COARSENED space-time grid (not raw observations) to
  avoid local density artifacts that plagued v1.
* Permutation unit = EMBRYO.  All timepoints from one embryo move together
  in the null.  Observation-level permutation would inflate significance
  because repeated observations per embryo are not independent.
* Labels enter only at the test stage.  Graph construction is label-agnostic.
* Two schematics are saved: (t_hpf, y) and (t_hpf, x) to avoid hiding real
  branch geometry by projection.

Public API
----------
  build_spacetime_grid_centroids(positions, mask, time_values,
                                  grid_cells, t_weight)
      -> (centroids_df, obs_ownership)

  build_spacetime_mst(centroids_df, k_neighbors)
      -> (nodes_df, edges, adjacency)

  contract_mst_skeleton(nodes_df, edges, adjacency, obs_ownership)
      -> (skel_nodes_df, skel_edges, skel_adj)

  identify_branch_points(adjacency) -> list[int]

  assign_embryos_to_arms(skel_nodes_df, skel_adj, branch_node_ids,
                          labels, obs_ownership)
      -> assignments_df

  permutation_branch_test(assignments_df, branch_node_id,
                           n_perm, seed) -> BranchTestResult

  BranchTestResult  (dataclass)
  branch_results_to_df(results) -> pd.DataFrame
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
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
    arm_ids : list[int]  — arm indices (= component indices after removing v)
    stat_obs : float     — chi-square statistic on embryo × arm table
    pval : float         — permutation p-value (embryo-level null)
    effect_size : float  — Cramér's V
    counts_ge : pd.DataFrame  — embryos × arm contingency table
    n_embryos : int      — total embryos tested at this branch
    t_hpf_branch : float — mean t_hpf of the branch node
    """
    node_id: int
    arm_ids: list[int]
    stat_obs: float
    pval: float
    effect_size: float
    counts_ge: pd.DataFrame
    n_embryos: int
    t_hpf_branch: float


# ---------------------------------------------------------------------------
# Step 1 — Coarsened space-time grid centroids
# ---------------------------------------------------------------------------

def build_spacetime_grid_centroids(
    positions: np.ndarray,    # (N_e, T, 2)
    mask: np.ndarray,         # (N_e, T)
    time_values: np.ndarray,  # (T,)
    grid_cells: int = 5,
    t_weight: float = 2.0,
) -> tuple[pd.DataFrame, dict[int, list[tuple[int, int]]]]:
    """Coarsen the (x, y, t) cloud into a spatial grid per time bin.

    For each time bin t, divide the observed (x, y) positions into a
    grid_cells × grid_cells grid.  Compute the centroid of each occupied cell.
    Assign a t_norm coordinate to each centroid.

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T)
    time_values : (T,)
    grid_cells : int — spatial grid resolution per time bin
    t_weight : float — scales temporal axis relative to spatial axes

    Returns
    -------
    centroids_df : DataFrame
        Columns: node_id, x, y, t_hpf, t_norm, n_obs
        One row per occupied grid cell.
    obs_ownership : dict[node_id -> list[(embryo_idx, t_idx)]]
        Which (embryo, time) observations fall in each centroid cell.
    """
    t_min, t_max = float(time_values.min()), float(time_values.max())
    t_range = t_max - t_min if t_max > t_min else 1.0

    # Global x/y range for grid bounds
    all_x, all_y = [], []
    for i in range(positions.shape[0]):
        for t in range(positions.shape[1]):
            if mask[i, t]:
                all_x.append(positions[i, t, 0])
                all_y.append(positions[i, t, 1])
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = (x_max - x_min) or 1.0
    y_range = (y_max - y_min) or 1.0

    # Bin each observation
    # Key: (t_idx, cell_x, cell_y)
    cell_obs: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
    cell_xy: dict[tuple, list[tuple[float, float]]] = defaultdict(list)

    for i in range(positions.shape[0]):
        for t in range(positions.shape[1]):
            if not mask[i, t]:
                continue
            x, y = positions[i, t, 0], positions[i, t, 1]
            cx = int(min(grid_cells - 1,
                         int((x - x_min) / x_range * grid_cells)))
            cy = int(min(grid_cells - 1,
                         int((y - y_min) / y_range * grid_cells)))
            key = (int(t), cx, cy)
            cell_obs[key].append((i, int(t)))
            cell_xy[key].append((x, y))

    # Build centroid nodes
    rows = []
    obs_ownership: dict[int, list[tuple[int, int]]] = {}
    for key, obs_list in sorted(cell_obs.items()):
        t_idx, cx, cy = key
        hpf = float(time_values[t_idx])
        t_norm = (hpf - t_min) / t_range * t_weight
        xys = cell_xy[key]
        x_c = float(np.mean([p[0] for p in xys]))
        y_c = float(np.mean([p[1] for p in xys]))
        node_id = len(rows)
        rows.append({
            "node_id": node_id,
            "x":       x_c,
            "y":       y_c,
            "t_hpf":   hpf,
            "t_norm":  t_norm,
            "n_obs":   len(obs_list),
        })
        obs_ownership[node_id] = obs_list

    centroids_df = pd.DataFrame(rows)
    return centroids_df, obs_ownership


# ---------------------------------------------------------------------------
# Step 2 — k-NN MST on coarsened centroids
# ---------------------------------------------------------------------------

def build_spacetime_mst(
    centroids_df: pd.DataFrame,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray]:
    """Fit a k-NN MST on the coarsened space-time centroids.

    Uses (x, y, t_norm) as the 3D coordinates.

    Parameters
    ----------
    centroids_df : from build_spacetime_grid_centroids
    k_neighbors : int

    Returns
    -------
    nodes_df : same as centroids_df (passed through)
    edges : list of (node_id_a, node_id_b)
    adjacency : (n_nodes, n_nodes) dense int array
    """
    from sklearn.neighbors import kneighbors_graph

    pts = centroids_df[["x", "y", "t_norm"]].values
    n = len(pts)

    if n < k_neighbors + 1:
        raise ValueError(
            f"Only {n} centroid nodes; need at least k_neighbors+1={k_neighbors+1}. "
            "Reduce grid_cells or k_neighbors."
        )

    knn = kneighbors_graph(pts, n_neighbors=k_neighbors,
                           mode="distance", include_self=False)
    mst_sparse = scipy.sparse.csgraph.minimum_spanning_tree(knn)
    mst_sym = mst_sparse + mst_sparse.T
    mst_coo = scipy.sparse.coo_matrix(mst_sym)

    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for i, j in zip(mst_coo.row, mst_coo.col):
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key not in seen:
            edges.append(key)
            seen.add(key)

    adjacency = np.zeros((n, n), dtype=int)
    for (i, j) in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    return centroids_df, edges, adjacency


# ---------------------------------------------------------------------------
# Step 3 — Contract degree-2 chains to skeleton
# ---------------------------------------------------------------------------

def contract_mst_skeleton(
    nodes_df: pd.DataFrame,
    edges: list[tuple[int, int]],
    adjacency: np.ndarray,
    obs_ownership: dict[int, list[tuple[int, int]]],
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray, dict[int, list[tuple[int, int]]]]:
    """Collapse degree-2 chains; return skeleton nodes and edges.

    Skeleton nodes are leaves (degree 1) or branch points (degree >= 3).
    Each skeleton node absorbs the centroid nodes on the chain it spans,
    and by extension all (embryo_idx, t_idx) observations from those nodes.

    Parameters
    ----------
    nodes_df : from build_spacetime_mst
    edges : from build_spacetime_mst
    adjacency : from build_spacetime_mst
    obs_ownership : from build_spacetime_grid_centroids

    Returns
    -------
    skel_nodes_df : DataFrame
        Columns: node_id, x_mean, y_mean, t_hpf_mean, n_obs, degree
    skel_edges : list of (node_id_a, node_id_b)
    skel_adj : (n_skel, n_skel) int array
    skel_obs_ownership : dict[skel_node_id -> list[(embryo_idx, t_idx)]]
    """
    n_nodes = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)

    # Skeleton = leaves (degree 1) + branches (degree >= 3)
    skel_mask = (degrees == 1) | (degrees >= 3)
    skel_orig_ids = sorted(np.where(skel_mask)[0])

    if len(skel_orig_ids) == 0:
        # Entire graph is a single chain — treat endpoints as skeleton
        endpoints = np.where(degrees <= 1)[0]
        skel_orig_ids = sorted(endpoints.tolist()) or [0, n_nodes - 1]

    skel_set = set(skel_orig_ids)

    # Map each original node to the nearest skeleton node via chain walking
    orig_to_skel: dict[int, int] = {}
    owned_by_skel: dict[int, list[int]] = defaultdict(list)

    for s in skel_orig_ids:
        orig_to_skel[s] = s
        owned_by_skel[s].append(s)

    # Walk chains from each skeleton node
    for s in skel_orig_ids:
        for start in np.where(adjacency[s] > 0)[0]:
            start = int(start)
            if start in skel_set or start in orig_to_skel:
                continue
            prev, cur = s, start
            chain = [cur]
            while cur not in skel_set:
                nbrs = [int(n) for n in np.where(adjacency[cur] > 0)[0]
                        if int(n) != prev]
                if not nbrs:
                    break
                prev, cur = cur, nbrs[0]
                if cur not in skel_set:
                    chain.append(cur)
            mid = max(1, len(chain) // 2)
            for c in chain[:mid]:
                if c not in orig_to_skel:
                    orig_to_skel[c] = s
                    owned_by_skel[s].append(c)

    # Assign any remaining nodes to nearest skeleton by Euclidean distance
    coords = nodes_df[["x", "y", "t_norm"]].values
    skel_coords = np.array([coords[s] for s in skel_orig_ids])
    for i in range(n_nodes):
        if i not in orig_to_skel:
            dists = np.linalg.norm(skel_coords - coords[i], axis=1)
            nearest = skel_orig_ids[int(np.argmin(dists))]
            orig_to_skel[i] = nearest
            owned_by_skel[nearest].append(i)

    # Build new skeleton node ids (0-indexed)
    skel_id_map = {s: new_id for new_id, s in enumerate(skel_orig_ids)}

    # Build skel_nodes_df
    skel_rows = []
    skel_obs_ownership: dict[int, list[tuple[int, int]]] = {}
    for s in skel_orig_ids:
        new_id = skel_id_map[s]
        members = owned_by_skel[s]
        member_rows = nodes_df.iloc[members]
        all_obs: list[tuple[int, int]] = []
        for m in members:
            all_obs.extend(obs_ownership.get(m, []))
        skel_obs_ownership[new_id] = all_obs
        skel_rows.append({
            "node_id":    new_id,
            "orig_id":    s,
            "x_mean":     float(member_rows["x"].mean()),
            "y_mean":     float(member_rows["y"].mean()),
            "t_hpf_mean": float(member_rows["t_hpf"].mean()),
            "t_norm_mean": float(member_rows["t_norm"].mean()),
            "n_obs":      len(all_obs),
        })
    skel_nodes_df = pd.DataFrame(skel_rows).reset_index(drop=True)

    # Build skeleton edges
    skel_edge_set: set[tuple[int, int]] = set()
    for (i, j) in edges:
        si = orig_to_skel.get(i, i)
        sj = orig_to_skel.get(j, j)
        si_new = skel_id_map.get(si)
        sj_new = skel_id_map.get(sj)
        if si_new is not None and sj_new is not None and si_new != sj_new:
            key = (min(si_new, sj_new), max(si_new, sj_new))
            skel_edge_set.add(key)
    skel_edges = list(skel_edge_set)

    n_skel = len(skel_nodes_df)
    skel_adj = np.zeros((n_skel, n_skel), dtype=int)
    for (i, j) in skel_edges:
        skel_adj[i, j] = 1
        skel_adj[j, i] = 1

    # Add degree column
    skel_nodes_df["degree"] = skel_adj.sum(axis=1)
    skel_nodes_df["is_branch"] = skel_nodes_df["degree"] >= 3

    return skel_nodes_df, skel_edges, skel_adj, skel_obs_ownership


# ---------------------------------------------------------------------------
# Step 4 — Identify branch points
# ---------------------------------------------------------------------------

def identify_branch_points(adjacency: np.ndarray) -> list[int]:
    """Return node indices with degree >= 3."""
    return [int(i) for i in np.where(adjacency.sum(axis=1) >= 3)[0]]


# ---------------------------------------------------------------------------
# Step 5 — Assign embryos to subtree arms (embryo-level)
# ---------------------------------------------------------------------------

def assign_embryos_to_arms(
    skel_nodes_df: pd.DataFrame,
    skel_adj: np.ndarray,
    branch_node_ids: list[int],
    labels: np.ndarray,               # (N_e,) genotype per embryo
    skel_obs_ownership: dict[int, list[tuple[int, int]]],
) -> pd.DataFrame:
    """Assign each embryo to an arm at each branch node.

    For each branch node v:
    1. Remove v from the skeleton graph.
    2. BFS from each neighbour → connected components = arms.
    3. Each arm owns a set of (embryo_idx, t_idx) observations
       (via skel_obs_ownership expanded through BFS).
    4. Each embryo is assigned to the arm where it has the MOST observations
       (majority vote). If tied, use arm with earlier mean t_hpf.

    This is the embryo-level assignment — the permutation test shuffles
    embryo genotype labels, not observation labels.

    Returns
    -------
    assignments_df : DataFrame
        Columns: embryo_idx, genotype, branch_node_id, arm_idx,
                 n_obs_in_arm, n_obs_total
        One row per (embryo, branch_node).
    """
    rows = []

    for bn in branch_node_ids:
        t_branch = float(skel_nodes_df.loc[bn, "t_hpf_mean"])
        neighbours = list(np.where(skel_adj[bn] > 0)[0])
        if len(neighbours) < 2:
            continue

        # BFS to find arms (connected components with bn removed)
        visited = {bn}
        arms: list[list[int]] = []
        for start in neighbours:
            start = int(start)
            if start in visited:
                continue
            arm_nodes: list[int] = []
            queue = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                arm_nodes.append(node)
                for nbr in np.where(skel_adj[node] > 0)[0]:
                    nbr = int(nbr)
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            arms.append(arm_nodes)

        if len(arms) < 2:
            continue

        # Compute mean t_hpf per arm (for tie-breaking)
        arm_mean_t = []
        for arm in arms:
            t_vals = [skel_nodes_df.loc[n, "t_hpf_mean"] for n in arm]
            arm_mean_t.append(float(np.mean(t_vals)))

        # Collect embryo → {arm_idx: n_obs} mapping
        emb_arm_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for arm_idx, arm in enumerate(arms):
            for node in arm:
                for (emb_i, _t_i) in skel_obs_ownership.get(node, []):
                    emb_arm_counts[emb_i][arm_idx] += 1

        if not emb_arm_counts:
            continue

        # Also include branch node's own observations — assign by majority too
        for (emb_i, _t_i) in skel_obs_ownership.get(bn, []):
            # Already in emb_arm_counts if any arm contains them — skip double-count
            if emb_i not in emb_arm_counts:
                emb_arm_counts[emb_i]  # touch to initialise defaultdict

        for emb_i, arm_counts in emb_arm_counts.items():
            n_total = sum(arm_counts.values())
            if n_total == 0:
                continue
            # Majority vote with tie-break on earlier arm t_hpf
            best_arm = max(
                arm_counts.keys(),
                key=lambda a: (arm_counts[a], -arm_mean_t[a]),
            )
            rows.append({
                "embryo_idx":     emb_i,
                "genotype":       str(labels[emb_i]),
                "branch_node_id": bn,
                "arm_idx":        best_arm,
                "n_obs_in_arm":   arm_counts[best_arm],
                "n_obs_total":    n_total,
                "t_hpf_branch":   t_branch,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "embryo_idx", "genotype", "branch_node_id", "arm_idx",
            "n_obs_in_arm", "n_obs_total", "t_hpf_branch",
        ])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6 — Permutation chi-square test (embryo-level)
# ---------------------------------------------------------------------------

def _chi_square_stat(counts: np.ndarray) -> float:
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0, keepdims=True)
    total = counts.sum()
    if total == 0 or counts.shape[0] < 2 or counts.shape[1] < 2:
        return 0.0
    expected = (row_sums * col_sums) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(expected > 0, (counts - expected) ** 2 / expected, 0.0)
    return float(stat.sum())


def _cramers_v(stat: float, n: int, min_dim: int) -> float:
    if n == 0 or min_dim <= 1:
        return 0.0
    return float(np.sqrt(max(0.0, stat) / (n * (min_dim - 1))))


def permutation_branch_test(
    assignments_df: pd.DataFrame,
    branch_node_id: int,
    n_perm: int = 1000,
    seed: int = 42,
) -> BranchTestResult:
    """Embryo-level permutation chi-square test at one branch node.

    H₀: genotype is independent of arm assignment.

    Permutation procedure: shuffle genotype labels across the embryos
    assigned to this branch node.  Each shuffle keeps one label per embryo
    (because assignments_df is already embryo-level).

    Parameters
    ----------
    assignments_df : from assign_embryos_to_arms
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
    arms = sorted(sub["arm_idx"].unique())
    n_embryos = len(sub)

    g_map = {g: i for i, g in enumerate(genotypes)}
    a_map = {a: i for i, a in enumerate(arms)}

    # Observed contingency (embryos × arms)
    counts_obs = np.zeros((len(genotypes), len(arms)), dtype=float)
    for _, row in sub.iterrows():
        counts_obs[g_map[row["genotype"]], a_map[row["arm_idx"]]] += 1

    stat_obs = _chi_square_stat(counts_obs)

    # Permutation null — shuffle embryo genotype labels
    genotype_arr = sub["genotype"].values.copy()
    arm_arr = sub["arm_idx"].values.copy()

    null_stats = np.zeros(n_perm, dtype=float)
    for k in range(n_perm):
        shuffled = rng.permutation(genotype_arr)
        counts_null = np.zeros_like(counts_obs)
        for gi, ai in zip(shuffled, arm_arr):
            counts_null[g_map[gi], a_map[ai]] += 1
        null_stats[k] = _chi_square_stat(counts_null)

    pval = float((null_stats >= stat_obs).mean())
    pval = max(pval, 1.0 / n_perm)

    min_dim = min(len(genotypes), len(arms))
    effect_size = _cramers_v(stat_obs, n_embryos, min_dim)

    t_branch = float(sub["t_hpf_branch"].iloc[0]) if len(sub) > 0 else 0.0

    counts_df = pd.DataFrame(
        counts_obs,
        index=genotypes,
        columns=[f"arm_{a}" for a in arms],
    )

    return BranchTestResult(
        node_id=branch_node_id,
        arm_ids=arms,
        stat_obs=stat_obs,
        pval=pval,
        effect_size=effect_size,
        counts_ge=counts_df,
        n_embryos=n_embryos,
        t_hpf_branch=t_branch,
    )


def run_all_branch_tests(
    skel_nodes_df: pd.DataFrame,
    skel_adj: np.ndarray,
    branch_node_ids: list[int],
    labels: np.ndarray,
    skel_obs_ownership: dict[int, list[tuple[int, int]]],
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[BranchTestResult]]:
    """Convenience wrapper: assign + test at every branch node."""
    assignments_df = assign_embryos_to_arms(
        skel_nodes_df, skel_adj, branch_node_ids,
        labels, skel_obs_ownership,
    )
    results = []
    for bn in branch_node_ids:
        if bn not in assignments_df["branch_node_id"].values:
            continue
        res = permutation_branch_test(assignments_df, bn,
                                      n_perm=n_perm, seed=seed)
        results.append(res)
    return assignments_df, results


def branch_results_to_df(results: list[BranchTestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "node_id":               r.node_id,
            "t_hpf_branch":          round(r.t_hpf_branch, 1),
            "n_arms":                len(r.arm_ids),
            "n_embryos":             r.n_embryos,
            "stat_obs":              round(r.stat_obs, 4),
            "pval":                  round(r.pval, 4),
            "effect_size_cramers_v": round(r.effect_size, 4),
            "significant_p05":       r.pval < 0.05,
        })
    return pd.DataFrame(rows)
