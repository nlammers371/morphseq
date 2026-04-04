"""
principal_graph.py  (v3)
------------------------
Fit a principal tree to the condition-mean trajectories in 3D (x, y, t) space,
find branch points where trajectories diverge, and test whether genotypes
distribute non-uniformly across branches.

Core idea
---------
The cosmological condensation already gives us smooth, compressed trajectories.
We build the tree from the CONDITION-MEAN trajectories (5 conditions × T time
bins = 135 3D points), not from raw embryo observations.  This is the right
representation because:
  - The condensation is the learned compressed manifold
  - Mean trajectories are noise-reduced and comparable across conditions
  - Branch points in (x, y, t) space correspond to genuine developmental forks

Pipeline
--------
1. build_mean_trajectories(positions, mask, labels, time_values)
     -> traj_df  [condition, t_hpf, x, y]

2. build_trajectory_mst(traj_df, t_weight)
     -> (nodes_df, edges, adjacency)
     MST on all 135 3D trajectory points

3. contract_to_skeleton(nodes_df, edges, adjacency)
     -> (skel_nodes_df, skel_edges, skel_adj, owned_nodes)
     Collapse degree-2 chains; each skeleton node owns original traj points

4. identify_branch_points(skel_adj) -> list[int]

5. assign_embryos_to_arms(skel_nodes_df, skel_adj, branch_node_ids,
                           owned_nodes, positions, mask, labels, time_values)
     -> assignments_df  [embryo_idx, genotype, branch_node_id, arm_idx, ...]
     Embryo majority-vote across all time bins

6. permutation_branch_test(assignments_df, branch_node_id, n_perm, seed)
     -> BranchTestResult
     Permutation unit = embryo (NOT observation)

7. 2D projection schematic:
     plot in (t_hpf, y) and (t_hpf, x) — tree derived from 3D, projected to 2D
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.sparse.csgraph


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class BranchTestResult:
    """Permutation test result at one branch node.

    Attributes
    ----------
    node_id : int
    arm_ids : list[int]
    stat_obs : float      — chi-square on embryo × arm table
    pval : float          — embryo-level permutation p-value
    effect_size : float   — Cramér's V
    counts_ge : pd.DataFrame  — embryos × arms contingency
    n_embryos : int
    t_hpf_branch : float
    arm_conditions : dict[int, list[str]]  — dominant conditions per arm
    """
    node_id: int
    arm_ids: list[int]
    stat_obs: float
    pval: float
    effect_size: float
    counts_ge: pd.DataFrame
    n_embryos: int
    t_hpf_branch: float
    arm_conditions: dict[int, list[str]]


# ---------------------------------------------------------------------------
# Step 1 — Build condition-mean trajectories
# ---------------------------------------------------------------------------

def build_mean_trajectories(
    positions: np.ndarray,    # (N_e, T, 2)
    mask: np.ndarray,         # (N_e, T)
    labels: np.ndarray,       # (N_e,)
    time_values: np.ndarray,  # (T,)
    min_obs: int = 2,
) -> pd.DataFrame:
    """Compute per-condition mean (x, y) at each time bin.

    Returns
    -------
    traj_df : DataFrame
        Columns: node_id, condition, t_hpf, t_idx, x, y, n_obs
        One row per (condition, time_bin) with at least min_obs embryos.
        node_id is a unique integer index (used as graph node id).
    """
    conditions = np.unique(labels)
    rows = []
    for cond in conditions:
        idx = np.where(labels == cond)[0]
        for ti, hpf in enumerate(time_values):
            obs = mask[idx, ti]
            if obs.sum() < min_obs:
                continue
            pts = positions[idx[obs], ti, :]
            rows.append({
                "node_id":   len(rows),
                "condition": cond,
                "t_hpf":     float(hpf),
                "t_idx":     int(ti),
                "x":         float(pts[:, 0].mean()),
                "y":         float(pts[:, 1].mean()),
                "n_obs":     int(obs.sum()),
            })
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 — Fit MST on 3D trajectory points
# ---------------------------------------------------------------------------

def build_trajectory_mst(
    traj_df: pd.DataFrame,
    t_weight: float = 3.0,
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray]:
    """Fit a k-NN-seeded MST on condition-mean trajectory points in 3D.

    Uses (x, y, t_norm) where t_norm = t_hpf normalised to [0, t_weight].
    Because the trajectories are already smooth and structured, a full
    pairwise MST (not k-NN restricted) works well at this scale (~135 nodes).

    Parameters
    ----------
    traj_df : from build_mean_trajectories
    t_weight : float — scale of temporal axis relative to spatial axes

    Returns
    -------
    nodes_df : same as traj_df (passed through)
    edges : list of (node_id_a, node_id_b)
    adjacency : (n, n) dense int array
    """
    t_min = traj_df["t_hpf"].min()
    t_max = traj_df["t_hpf"].max()
    t_range = (t_max - t_min) or 1.0

    traj_df = traj_df.copy()
    traj_df["t_norm"] = (traj_df["t_hpf"] - t_min) / t_range * t_weight

    pts = traj_df[["x", "y", "t_norm"]].values   # (n, 3)
    n = len(pts)

    # Full pairwise distance matrix → MST (fine at n~135)
    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = float(np.linalg.norm(pts[i] - pts[j]))

    sparse_dist = scipy.sparse.csr_matrix(dist)
    mst = scipy.sparse.csgraph.minimum_spanning_tree(sparse_dist)
    mst_sym = mst + mst.T
    coo = scipy.sparse.coo_matrix(mst_sym)

    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    for i, j in zip(coo.row, coo.col):
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key not in seen:
            edges.append(key)
            seen.add(key)

    adjacency = np.zeros((n, n), dtype=int)
    for (i, j) in edges:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    return traj_df, edges, adjacency


# ---------------------------------------------------------------------------
# Step 3 — Contract degree-2 chains to skeleton
# ---------------------------------------------------------------------------

def contract_to_skeleton(
    traj_df: pd.DataFrame,
    edges: list[tuple[int, int]],
    adjacency: np.ndarray,
) -> tuple[pd.DataFrame, list[tuple[int, int]], np.ndarray, dict[int, list[int]]]:
    """Collapse degree-2 chains; return skeleton with leaves + branch points.

    Each skeleton node absorbs the original trajectory nodes on the chain
    it spans, and records which conditions pass through it.

    Returns
    -------
    skel_df : DataFrame
        Columns: node_id, x_mean, y_mean, t_hpf_mean, t_norm_mean,
                 n_orig, conditions_through, is_branch, degree
    skel_edges : list of (a, b)
    skel_adj : (n_skel, n_skel) int array
    owned : dict[skel_node_id -> list[orig_node_id]]
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)

    skel_mask = (degrees == 1) | (degrees >= 3)
    skel_orig = sorted(np.where(skel_mask)[0])

    if not skel_orig:
        # Degenerate: all degree-2 chain — use endpoints
        endpoints = np.where(degrees <= 1)[0]
        skel_orig = sorted(endpoints.tolist()) or [0, n - 1]

    skel_set = set(skel_orig)
    orig_to_skel: dict[int, int] = {}
    owned: dict[int, list[int]] = defaultdict(list)

    for s in skel_orig:
        orig_to_skel[s] = s
        owned[s].append(s)

    # Walk chains from each skeleton node
    for s in skel_orig:
        for start in np.where(adjacency[s] > 0)[0]:
            start = int(start)
            if start in skel_set or start in orig_to_skel:
                continue
            prev, cur = s, start
            chain = [cur]
            while cur not in skel_set:
                nbrs = [int(nb) for nb in np.where(adjacency[cur] > 0)[0]
                        if int(nb) != prev]
                if not nbrs:
                    break
                prev, cur = cur, nbrs[0]
                if cur not in skel_set:
                    chain.append(cur)
            mid = max(1, len(chain) // 2)
            for c in chain[:mid]:
                if c not in orig_to_skel:
                    orig_to_skel[c] = s
                    owned[s].append(c)

    # Any unassigned → nearest skeleton by position
    coords = traj_df[["x", "y", "t_norm"]].values
    skel_coords = np.array([coords[s] for s in skel_orig])
    for i in range(n):
        if i not in orig_to_skel:
            dists = np.linalg.norm(skel_coords - coords[i], axis=1)
            nearest = skel_orig[int(np.argmin(dists))]
            orig_to_skel[i] = nearest
            owned[nearest].append(i)

    skel_id_map = {s: new_id for new_id, s in enumerate(skel_orig)}

    skel_rows = []
    new_owned: dict[int, list[int]] = {}
    for s in skel_orig:
        new_id = skel_id_map[s]
        members = owned[s]
        member_df = traj_df.iloc[members]
        conditions_through = sorted(member_df["condition"].unique().tolist())
        new_owned[new_id] = members
        skel_rows.append({
            "node_id":           new_id,
            "orig_id":           s,
            "x_mean":            float(member_df["x"].mean()),
            "y_mean":            float(member_df["y"].mean()),
            "t_hpf_mean":        float(member_df["t_hpf"].mean()),
            "t_norm_mean":       float(member_df["t_norm"].mean()),
            "n_orig":            len(members),
            "conditions_through": "|".join(conditions_through),
        })

    skel_df = pd.DataFrame(skel_rows).reset_index(drop=True)

    # Skeleton edges
    skel_edge_set: set[tuple[int, int]] = set()
    for (i, j) in edges:
        si = skel_id_map.get(orig_to_skel.get(i, i))
        sj = skel_id_map.get(orig_to_skel.get(j, j))
        if si is not None and sj is not None and si != sj:
            skel_edge_set.add((min(si, sj), max(si, sj)))
    skel_edges = list(skel_edge_set)

    n_skel = len(skel_df)
    skel_adj = np.zeros((n_skel, n_skel), dtype=int)
    for (i, j) in skel_edges:
        skel_adj[i, j] = 1
        skel_adj[j, i] = 1

    skel_df["degree"] = skel_adj.sum(axis=1)
    skel_df["is_branch"] = skel_df["degree"] >= 3

    return skel_df, skel_edges, skel_adj, new_owned


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------

def identify_branch_points(adjacency: np.ndarray) -> list[int]:
    return [int(i) for i in np.where(adjacency.sum(axis=1) >= 3)[0]]


# ---------------------------------------------------------------------------
# Step 5 — Assign embryos to arms (embryo majority-vote)
# ---------------------------------------------------------------------------

def assign_embryos_to_arms(
    skel_df: pd.DataFrame,
    skel_adj: np.ndarray,
    branch_node_ids: list[int],
    owned: dict[int, list[int]],    # skel_node_id -> orig traj node ids
    traj_df: pd.DataFrame,          # original trajectory nodes
    positions: np.ndarray,          # (N_e, T, 2) — for nearest-centroid assignment
    mask: np.ndarray,               # (N_e, T)
    labels: np.ndarray,             # (N_e,)
    time_values: np.ndarray,
) -> pd.DataFrame:
    """Assign each embryo to an arm at each branch node.

    Strategy
    --------
    For each branch node v:
    1. Remove v and find connected arms via BFS.
    2. Each arm owns a set of skeleton nodes → orig traj nodes → (condition, t_hpf) pairs.
    3. For each arm, build the set of (t_idx, centroid_xy) points it covers.
    4. For each embryo, at each observed time bin, find the nearest arm centroid.
    5. Count arm assignments per embryo; majority vote → arm assignment.
    6. Permutation null shuffles embryo labels (not obs labels).

    Returns
    -------
    assignments_df : DataFrame
        Columns: embryo_idx, genotype, branch_node_id, arm_idx,
                 n_obs_in_arm, n_obs_total, t_hpf_branch, arm_conditions
    """
    rows = []

    for bn in branch_node_ids:
        t_branch = float(skel_df.loc[bn, "t_hpf_mean"])
        neighbours = list(np.where(skel_adj[bn] > 0)[0])
        if len(neighbours) < 2:
            continue

        # BFS to find arms
        visited = {bn}
        arms: list[list[int]] = []
        for start in neighbours:
            start = int(start)
            if start in visited:
                continue
            arm_nodes: list[int] = []
            q = deque([start])
            visited.add(start)
            while q:
                node = q.popleft()
                arm_nodes.append(node)
                for nb in np.where(skel_adj[node] > 0)[0]:
                    nb = int(nb)
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            arms.append(arm_nodes)

        if len(arms) < 2:
            continue

        # Build (t_idx → arm_idx → centroid_xy) lookup
        # For each arm, collect all (t_idx, x, y) from owned traj points
        arm_centroids: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        arm_conds: dict[int, list[str]] = {}
        for arm_idx, arm in enumerate(arms):
            all_conds = []
            for skel_n in arm:
                for orig_n in owned.get(skel_n, []):
                    row = traj_df.iloc[orig_n]
                    ti = int(row["t_idx"])
                    xy = np.array([row["x"], row["y"]])
                    if ti not in arm_centroids[arm_idx]:
                        arm_centroids[arm_idx][ti] = xy
                    else:
                        # Average if multiple traj points for same t
                        arm_centroids[arm_idx][ti] = (arm_centroids[arm_idx][ti] + xy) / 2
                    all_conds.append(str(row["condition"]))
            arm_conds[arm_idx] = sorted(set(all_conds))

        # For each embryo, assign to nearest arm at each observed time
        emb_arm_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        n_embryos = positions.shape[0]

        for emb_i in range(n_embryos):
            for ti in range(positions.shape[1]):
                if not mask[emb_i, ti]:
                    continue
                emb_xy = positions[emb_i, ti, :]
                # Find arms that have a centroid for this time bin
                arm_dists = {}
                for arm_idx, t_cent in arm_centroids.items():
                    if ti in t_cent:
                        arm_dists[arm_idx] = np.linalg.norm(emb_xy - t_cent[ti])
                if not arm_dists:
                    continue
                best_arm = min(arm_dists, key=lambda a: arm_dists[a])
                emb_arm_counts[emb_i][best_arm] += 1

        # Majority vote per embryo
        for emb_i, arm_counts in emb_arm_counts.items():
            n_total = sum(arm_counts.values())
            if n_total == 0:
                continue
            # Tie-break: arm with earlier mean t_hpf in its centroid
            def arm_mean_t(a):
                ti_vals = list(arm_centroids[a].keys())
                return float(np.mean(ti_vals)) if ti_vals else 999.0

            best_arm = max(
                arm_counts.keys(),
                key=lambda a: (arm_counts[a], -arm_mean_t(a)),
            )
            rows.append({
                "embryo_idx":     emb_i,
                "genotype":       str(labels[emb_i]),
                "branch_node_id": bn,
                "arm_idx":        best_arm,
                "n_obs_in_arm":   arm_counts[best_arm],
                "n_obs_total":    n_total,
                "t_hpf_branch":   t_branch,
                "arm_conditions": "|".join(arm_conds.get(best_arm, [])),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "embryo_idx", "genotype", "branch_node_id", "arm_idx",
            "n_obs_in_arm", "n_obs_total", "t_hpf_branch", "arm_conditions",
        ])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 6 — Permutation chi-square (embryo-level)
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
    """Embryo-level permutation chi-square test.

    Shuffle genotype labels across the EMBRYOS assigned to this branch.
    Each embryo contributes one row → one label → permutation is clean.
    """
    sub = assignments_df[assignments_df["branch_node_id"] == branch_node_id].copy()
    rng = np.random.default_rng(seed)

    genotypes = sorted(sub["genotype"].unique())
    arms = sorted(sub["arm_idx"].unique())
    n_embryos = len(sub)

    g_map = {g: i for i, g in enumerate(genotypes)}
    a_map = {a: i for i, a in enumerate(arms)}

    counts_obs = np.zeros((len(genotypes), len(arms)), dtype=float)
    for _, row in sub.iterrows():
        counts_obs[g_map[row["genotype"]], a_map[row["arm_idx"]]] += 1

    stat_obs = _chi_square_stat(counts_obs)

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
        counts_obs, index=genotypes,
        columns=[f"arm_{a}" for a in arms],
    )

    # Collect dominant conditions per arm
    arm_conds: dict[int, list[str]] = {}
    for a in arms:
        arm_sub = sub[sub["arm_idx"] == a]["arm_conditions"]
        conds: list[str] = []
        for s in arm_sub:
            conds.extend(str(s).split("|"))
        arm_conds[a] = sorted(set(conds))

    return BranchTestResult(
        node_id=branch_node_id,
        arm_ids=arms,
        stat_obs=stat_obs,
        pval=pval,
        effect_size=effect_size,
        counts_ge=counts_df,
        n_embryos=n_embryos,
        t_hpf_branch=t_branch,
        arm_conditions=arm_conds,
    )


def run_all_branch_tests(
    skel_df: pd.DataFrame,
    skel_adj: np.ndarray,
    branch_node_ids: list[int],
    owned: dict[int, list[int]],
    traj_df: pd.DataFrame,
    positions: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[BranchTestResult]]:
    assignments_df = assign_embryos_to_arms(
        skel_df, skel_adj, branch_node_ids, owned,
        traj_df, positions, mask, labels, time_values,
    )
    results = []
    for bn in branch_node_ids:
        if bn not in assignments_df["branch_node_id"].values:
            continue
        res = permutation_branch_test(assignments_df, bn, n_perm=n_perm, seed=seed)
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
            "arm_conditions":        str({a: r.arm_conditions.get(a, [])
                                          for a in r.arm_ids}),
        })
    return pd.DataFrame(rows)
