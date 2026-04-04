"""
principal_tree.py
-----------------
Fit an elastic principal tree to the embryo-level 3D space-time trajectory
cloud using ElPiGraph, then test whether genotypes distribute non-uniformly
across outgoing branches.

Scientific rationale
--------------------
The condensed cosmological trajectories give per-embryo positions over time.
Rather than averaging per condition (which loses embryo-level heterogeneity),
we build a 3D cloud of all embryo-timepoint observations:

    z_{e,t} = (x_{e,t}, y_{e,t}, α·t_hpf)

and fit an elastic principal tree through this cloud with ElPiGraph.
Branch significance is then tested at the embryo level: each embryo is
assigned to an arm via majority-vote over its projected observations, and
genotype labels are permuted across embryos (not observations).

Pipeline
--------
1. build_embryo_spacetime_cloud      — (N_obs, 3) point cloud + metadata
2. fit_principal_tree                — ElPiGraph elastic tree
3. project_observations_to_tree      — nearest edge projection
4. identify_branch_nodes             — degree ≥ 3
5. assign_embryos_to_arms            — BFS arms + majority vote
6. permutation_branch_test           — embryo-level chi-square
7. run_all_branch_tests              — wrapper
8. branch_results_to_df             — tidy output table
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

import elpigraph


# ---------------------------------------------------------------------------
# Data types (reused from principal_graph.py for API compatibility)
# ---------------------------------------------------------------------------

@dataclass
class BranchTestResult:
    """Permutation test result at one branch node."""
    node_id: int
    arm_ids: list[int]
    stat_obs: float
    pval: float
    effect_size: float
    counts_ge: pd.DataFrame   # genotypes × arms
    n_embryos: int
    t_hpf_branch: float
    arm_conditions: dict[int, list[str]]


# ---------------------------------------------------------------------------
# Step 1 — Build embryo-level 3D cloud
# ---------------------------------------------------------------------------

def build_embryo_spacetime_cloud(
    positions: np.ndarray,    # (N_e, T, 2)
    mask: np.ndarray,         # (N_e, T)
    time_values: np.ndarray,  # (T,)
    t_weight: float = 3.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack all observed (embryo, time) pairs into a 3D point cloud.

    The temporal axis is scaled to [0, t_weight] so that time contributes
    proportionally to spatial axes during tree fitting.

    Returns
    -------
    pts_3d : (N_obs, 3)  — columns: x, y, t_scaled
    obs_df : DataFrame
        Columns: obs_idx, embryo_idx, time_idx, t_hpf, x, y, t_scaled
    """
    t_min = float(time_values.min())
    t_max = float(time_values.max())
    t_range = (t_max - t_min) or 1.0

    rows = []
    for emb_i in range(positions.shape[0]):
        for ti, hpf in enumerate(time_values):
            if not mask[emb_i, ti]:
                continue
            x, y = float(positions[emb_i, ti, 0]), float(positions[emb_i, ti, 1])
            t_sc = (float(hpf) - t_min) / t_range * t_weight
            rows.append({
                "obs_idx":    len(rows),
                "embryo_idx": emb_i,
                "time_idx":   ti,
                "t_hpf":      float(hpf),
                "x":          x,
                "y":          y,
                "t_scaled":   t_sc,
            })

    obs_df = pd.DataFrame(rows).reset_index(drop=True)
    pts_3d = obs_df[["x", "y", "t_scaled"]].values.astype(np.float64)
    return pts_3d, obs_df


# ---------------------------------------------------------------------------
# Step 2 — Fit elastic principal tree
# ---------------------------------------------------------------------------

def fit_principal_tree(
    pts_3d: np.ndarray,
    n_nodes: int = 20,
    Lambda: float = 0.01,
    Mu: float = 0.1,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit an ElPiGraph elastic principal tree to the 3D cloud.

    Parameters
    ----------
    pts_3d : (N_obs, 3)
    n_nodes : number of tree nodes (complexity)
    Lambda : elasticity — penalizes displacement of nodes from data
    Mu : bending/branching penalty

    Returns
    -------
    nodes_df : DataFrame
        Columns: node_id, x, y, t_scaled, degree
    edges_df : DataFrame
        Columns: source, target
    raw : dict — raw ElPiGraph output (first replicate)
    """
    results = elpigraph.computeElasticPrincipalTree(
        X=pts_3d,
        NumNodes=n_nodes,
        Lambda=Lambda,
        Mu=Mu,
        Do_PCA=True,
        CenterData=True,
        verbose=verbose,
    )
    raw = results[0]

    node_pos = raw["NodePositions"]   # (n_nodes, 3)
    edges = np.array(raw["Edges"][0])  # (n_edges, 2)

    # Degree
    n = len(node_pos)
    degree = np.zeros(n, dtype=int)
    for a, b in edges:
        degree[int(a)] += 1
        degree[int(b)] += 1

    nodes_df = pd.DataFrame({
        "node_id": np.arange(n),
        "x":       node_pos[:, 0],
        "y":       node_pos[:, 1],
        "t_scaled": node_pos[:, 2],
        "degree":  degree,
    })
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    edges_df["source"] = edges_df["source"].astype(int)
    edges_df["target"] = edges_df["target"].astype(int)

    return nodes_df, edges_df, raw


# ---------------------------------------------------------------------------
# Step 3 — Project observations to nearest tree edge
# ---------------------------------------------------------------------------

def _point_to_segment_projection(
    p: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> tuple[float, float]:
    """Project point p onto segment ab.

    Returns (t_frac, dist) where t_frac ∈ [0, 1] is the fraction along a→b
    and dist is the distance from p to the nearest point on the segment.
    """
    ab = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return 0.0, float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab)) / ab_sq
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return t, float(np.linalg.norm(p - proj))


def project_observations_to_tree(
    pts_3d: np.ndarray,
    obs_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """Project each observation onto the nearest tree segment.

    Returns
    -------
    proj_df : DataFrame
        All columns from obs_df plus:
        nearest_edge_a, nearest_edge_b  — endpoint node ids
        proj_frac                        — fraction along edge (0=a, 1=b)
        dist_to_edge                     — residual distance
    """
    node_pos = nodes_df[["x", "y", "t_scaled"]].values
    edge_list = edges_df[["source", "target"]].values.tolist()

    proj_rows = []
    for obs_i, row in obs_df.iterrows():
        p = pts_3d[int(row["obs_idx"])]
        best_dist = np.inf
        best_a, best_b, best_frac = 0, 0, 0.0
        for a_id, b_id in edge_list:
            frac, dist = _point_to_segment_projection(
                p, node_pos[a_id], node_pos[b_id],
            )
            if dist < best_dist:
                best_dist = dist
                best_a, best_b, best_frac = a_id, b_id, frac
        proj_rows.append({
            **row.to_dict(),
            "nearest_edge_a": best_a,
            "nearest_edge_b": best_b,
            "proj_frac":      best_frac,
            "dist_to_edge":   best_dist,
        })

    return pd.DataFrame(proj_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4 — Identify branch nodes
# ---------------------------------------------------------------------------

def identify_branch_nodes(nodes_df: pd.DataFrame) -> list[int]:
    """Return node_ids with degree ≥ 3."""
    return nodes_df.loc[nodes_df["degree"] >= 3, "node_id"].tolist()


# ---------------------------------------------------------------------------
# Step 5 — Assign embryos to arms (majority vote)
# ---------------------------------------------------------------------------

def _build_adjacency(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for _, e in edges_df.iterrows():
        a, b = int(e["source"]), int(e["target"])
        adj[a].append(b)
        adj[b].append(a)
    return adj


def assign_embryos_to_arms(
    proj_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    branch_node_ids: list[int],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Assign each embryo to an arm at each branch node.

    Strategy
    --------
    For each branch node v:
    1. Remove v from graph; BFS from each neighbor → connected components = arms.
    2. Each arm owns a set of tree nodes.
    3. For each observation: determine arm from projected edge endpoints.
       - If both endpoints in same arm → that arm.
       - If edge spans branch node v → assign to the non-v endpoint's arm.
    4. Count obs per arm per embryo; majority vote.
    5. Only count discriminating observations (obs on edges where ≥2 arms
       are represented for that edge — i.e., edges that are NOT the branch
       node's own edges, which are shared trunk).
    """
    adj = _build_adjacency(nodes_df, edges_df)
    all_node_ids = set(nodes_df["node_id"].tolist())
    rows = []

    for bn in branch_node_ids:
        t_branch = float(nodes_df.loc[nodes_df["node_id"] == bn, "t_scaled"].iloc[0])
        # Approximate hpf from t_scaled (we store it separately below)
        neighbours = adj[bn]
        if len(neighbours) < 2:
            continue

        # BFS from each neighbour (excluding bn) → arms
        visited = {bn}
        arms: list[list[int]] = []
        for start in neighbours:
            if start in visited:
                continue
            arm_nodes: list[int] = []
            q = deque([start])
            visited.add(start)
            while q:
                node = q.popleft()
                arm_nodes.append(node)
                for nb in adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        q.append(nb)
            arms.append(arm_nodes)

        if len(arms) < 2:
            continue

        # node_id → arm_idx mapping
        node_to_arm: dict[int, int] = {}
        for arm_idx, arm in enumerate(arms):
            for n in arm:
                node_to_arm[n] = arm_idx

        # For each observation: determine arm
        # An edge is discriminating if its two endpoints map to different arms
        # OR one endpoint is the branch node itself.
        emb_arm_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for _, obs in proj_df.iterrows():
            a_id = int(obs["nearest_edge_a"])
            b_id = int(obs["nearest_edge_b"])
            a_arm = node_to_arm.get(a_id)
            b_arm = node_to_arm.get(b_id)

            if a_id == bn:
                # edge from branch node to arm: assign to b's arm
                if b_arm is not None:
                    emb_arm_counts[int(obs["embryo_idx"])][b_arm] += 1
            elif b_id == bn:
                if a_arm is not None:
                    emb_arm_counts[int(obs["embryo_idx"])][a_arm] += 1
            elif a_arm is not None and b_arm is not None and a_arm == b_arm:
                # Both in same arm — only count if this arm is discriminating
                # (i.e., not the trunk pre-fork). We count all same-arm obs.
                emb_arm_counts[int(obs["embryo_idx"])][a_arm] += 1
            elif a_arm is not None and b_arm is not None and a_arm != b_arm:
                # Edge spanning two arms (shouldn't happen in a tree but be safe)
                frac = float(obs["proj_frac"])
                arm_choice = a_arm if frac < 0.5 else b_arm
                emb_arm_counts[int(obs["embryo_idx"])][arm_choice] += 1
            # else: edge to/from unlabelled node — skip

        # Majority vote per embryo
        # Get approximate hpf for tie-break (mean t_hpf of arm nodes)
        arm_t_hpf: dict[int, float] = {}
        for arm_idx, arm in enumerate(arms):
            arm_t = nodes_df.loc[nodes_df["node_id"].isin(arm), "t_scaled"]
            arm_t_hpf[arm_idx] = float(arm_t.mean()) if len(arm_t) > 0 else 999.0

        for emb_i, arm_counts in emb_arm_counts.items():
            if not arm_counts:
                continue
            n_total = sum(arm_counts.values())
            best_arm = max(
                arm_counts.keys(),
                key=lambda a: (arm_counts[a], -arm_t_hpf.get(a, 999.0)),
            )
            # collect dominant conditions in each arm from projections
            arm_cond_str = "|".join(sorted(set(
                labels[int(obs["embryo_idx"])]
                for _, obs in proj_df[
                    proj_df["embryo_idx"].isin(
                        [e for e, ac in emb_arm_counts.items() if best_arm in ac]
                    )
                ].iterrows()
            )))

            rows.append({
                "branch_node_id": bn,
                "embryo_idx":     emb_i,
                "genotype":       str(labels[emb_i]),
                "arm_idx":        best_arm,
                "n_obs_support":  arm_counts[best_arm],
                "n_obs_total":    n_total,
                "t_hpf_branch":   t_branch,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "branch_node_id", "embryo_idx", "genotype", "arm_idx",
            "n_obs_support", "n_obs_total", "t_hpf_branch",
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
    """Embryo-level permutation chi-square.

    Permutes genotype labels across embryos (NOT observations).
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

    # Dominant genotypes per arm
    arm_conds: dict[int, list[str]] = {}
    for a in arms:
        arm_sub = sub[sub["arm_idx"] == a]["genotype"].unique().tolist()
        arm_conds[a] = sorted(arm_sub)

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


# ---------------------------------------------------------------------------
# Step 7 — Run all branch tests
# ---------------------------------------------------------------------------

def run_all_branch_tests(
    proj_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    branch_node_ids: list[int],
    labels: np.ndarray,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[BranchTestResult]]:
    assignments_df = assign_embryos_to_arms(
        proj_df, nodes_df, edges_df, branch_node_ids, labels,
    )
    results = []
    for bn in branch_node_ids:
        sub = assignments_df[assignments_df["branch_node_id"] == bn]
        if len(sub) == 0:
            continue
        res = permutation_branch_test(assignments_df, bn, n_perm=n_perm, seed=seed)
        results.append(res)
    return assignments_df, results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def branch_results_to_df(results: list[BranchTestResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "node_id":               r.node_id,
            "t_scaled_branch":       round(r.t_hpf_branch, 3),
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
