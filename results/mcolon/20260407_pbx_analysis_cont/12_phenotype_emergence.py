"""
12_phenotype_emergence.py
--------------------------
Phenotype emergence graph and coherent summary for PBX 5-class pairwise data.

Onset rule (L-consecutive):
  tau_ij = first bin where pair is `separated` for L consecutive bins
  separated: pval < 0.01 AND auroc >= 0.70
  not_separated: pval > 0.10
  ambiguous: everything else

Outputs (results/mcolon/20260407_pbx_analysis_cont/results/emergence/):
  onset_matrix.csv
  onset_pairs.csv
  coherent_partitions.csv
  transitivity_timebin_summary.csv
  transitivity_violations.csv
  ultrametric_triples.csv
  ultrametric_summary.txt
  panel_A_onset_heatmap.png
  panel_B_relation_snapshots.png
  panel_C_partition_river.png
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from analyze.classification.transitivity import (
    TransitivityParams,
    build_transitivity_report,
    SEPARATED,
    NOT_SEPARATED,
    AMBIGUOUS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "results/positioning/pairwise/combined_pairwise_5class_bin4_perm500"
OUT_DIR  = Path(__file__).parent / "results/emergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARAMS = TransitivityParams(
    p_sep=0.01,
    auroc_sep=0.70,
    p_ns=0.10,
    L=3,
)

# Snapshot time bins: 10 hpf, then 24, 36, 48, ... 120 hpf
SNAPSHOT_BINS = [10.0] + list(range(24, 121, 12))

CLASS_LABELS = {
    "inj_ctrl":            "Inj. Ctrl",
    "wik_ab":              "WIK/AB",
    "pbx1b_crispant":      "pbx1b",
    "pbx4_crispant":       "pbx4",
    "pbx1b_pbx4_crispant": "pbx1b;pbx4",
}

CLASS_COLORS = {
    "inj_ctrl":            "#2166AC",
    "wik_ab":              "#6baed6",
    "pbx1b_crispant":      "#9467bd",
    "pbx4_crispant":       "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}

# Data-driven split topology derived from onset matrix.
#
# Never-separated pairs (NaN onset) → must stay in same block:
#   inj_ctrl  --  wik_ab
#   pbx1b_crispant  --  wik_ab          (weak signal, never durable)
#   pbx1b_pbx4_crispant  --  pbx4_crispant  (key finding: unresolved crispant pair)
#
# This gives three terminal blocks:
#   {pbx1b_pbx4_crispant, pbx4_crispant}  — jointly distinct from controls
#   {pbx1b_crispant}                       — intermediate
#   {inj_ctrl, wik_ab}                     — never split
#
# Split sequence (earliest → latest), g1 = group splitting off:
KNOWN_SPLITS = [
    # Split 1: the unresolved crispant pair peels from everyone else
    (
        {"inj_ctrl", "wik_ab", "pbx1b_crispant", "pbx4_crispant", "pbx1b_pbx4_crispant"},
        {"pbx4_crispant", "pbx1b_pbx4_crispant"},   # g1: composite crispant block
        {"inj_ctrl", "wik_ab", "pbx1b_crispant"},
    ),
    # Split 2: pbx1b peels from controls
    (
        {"inj_ctrl", "wik_ab", "pbx1b_crispant"},
        {"pbx1b_crispant"},
        {"inj_ctrl", "wik_ab"},
    ),
    # {inj_ctrl, wik_ab} never split — no further entry needed
]

# Fixed display order (top to bottom in river, top to bottom in graphs)
CLASS_ORDER = [
    "pbx1b_pbx4_crispant",
    "pbx4_crispant",
    "pbx1b_crispant",
    "inj_ctrl",
    "wik_ab",
]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(DATA_DIR / "scores.parquet")
    return scores[scores["feature_set"] == "vae"].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Coherent partition
# ---------------------------------------------------------------------------

def _cross_mean_onset(onset_matrix: pd.DataFrame, g1: set, g2: set) -> float:
    vals = []
    for a in g1:
        for b in g2:
            v = onset_matrix.loc[a, b]
            if not pd.isna(v):
                vals.append(float(v))
    return float(np.mean(vals)) if vals else float("inf")


def infer_split_times(onset_matrix: pd.DataFrame) -> list[dict]:
    results = []
    for block, g1, g2 in KNOWN_SPLITS:
        t = _cross_mean_onset(onset_matrix, g1, g2)
        results.append({"block": frozenset(block), "g1": frozenset(g1), "g2": frozenset(g2), "split_time": t})
    return results


def build_coherent_partitions(split_results: list[dict], all_time_bins: list[float]) -> pd.DataFrame:
    """
    Build monotone partition sequence by tracking a live set of frozenset blocks.

    For each split (applied in time order), locate the parent block containing g1,
    remove g1 from it, and add g1 as a new block. This enforces correct nesting
    regardless of split time ordering.
    """
    all_classes: set[str] = set()
    for s in split_results:
        all_classes |= s["block"]

    # Canonical block label: sorted comma-joined class names
    def _label(classes: frozenset) -> str:
        return "+".join(sorted(classes))

    # Start: one block containing all classes
    initial_partition: list[frozenset] = [frozenset(all_classes)]

    # Apply splits in time order to build the sequence of partitions
    # Each entry: (split_time, partition_after_split)
    splits_sorted = sorted(split_results, key=lambda x: x["split_time"])
    partition_history: list[tuple[float, list[frozenset]]] = [
        (float("-inf"), initial_partition)
    ]

    current_partition = list(initial_partition)
    for s in splits_sorted:
        g1 = s["g1"]  # frozenset of classes splitting off
        t  = s["split_time"]
        if np.isinf(t):
            continue  # never-split: skip
        # Find which current block contains all of g1
        parent = None
        for block in current_partition:
            if g1.issubset(block):
                parent = block
                break
        if parent is None:
            # g1 already split out in a prior step — skip (monotonicity)
            continue
        remainder = parent - g1
        new_partition = [b for b in current_partition if b != parent]
        new_partition.append(g1)
        if remainder:
            new_partition.append(remainder)
        current_partition = new_partition
        partition_history.append((t, list(current_partition)))

    # For each time bin, find the active partition (last one with split_time <= t)
    rows = []
    for t in all_time_bins:
        active_partition = partition_history[0][1]
        for split_t, partition in partition_history:
            if split_t <= t:
                active_partition = partition
            else:
                break

        # Assign block labels and IDs
        block_map: dict[str, str] = {}
        for block in active_partition:
            label = _label(block)
            for c in block:
                block_map[c] = label

        unique_blocks = sorted(set(block_map.values()))
        block_id = {b: i for i, b in enumerate(unique_blocks)}

        for c in sorted(all_classes):
            rows.append({
                "time_bin": t,
                "class": c,
                "partition_block_label": block_map.get(c, "unknown"),
                "partition_block_id": block_id.get(block_map.get(c, ""), -1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel A: Onset heatmap
# ---------------------------------------------------------------------------

def plot_onset_heatmap(onset_matrix: pd.DataFrame, ax: plt.Axes) -> None:
    # Use CLASS_ORDER for display
    classes = [c for c in CLASS_ORDER if c in onset_matrix.index]
    labels  = [CLASS_LABELS.get(c, c) for c in classes]
    n = len(classes)
    data = onset_matrix.reindex(index=classes, columns=classes).values.astype(float)

    finite = data[np.isfinite(data)]
    vmin = float(finite.min()) if len(finite) > 0 else 0.0
    vmax = float(finite.max()) if len(finite) > 0 else 1.0

    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#d0d0d0", edgecolor="white", hatch="///", linewidth=0,
                ))
            else:
                text_color = "white" if val > (vmin + (vmax - vmin) * 0.6) else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Pairwise Durable Onset (hpf)", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Onset (hpf)", shrink=0.8)
    gray_patch = mpatches.Patch(facecolor="#d0d0d0", hatch="///", label="Never separated (L=3)")
    ax.legend(handles=[gray_patch], loc="upper left", fontsize=7, framealpha=0.9)


# ---------------------------------------------------------------------------
# Panel B: Pairwise stripe plot (raw tri-state per pair × time)
# ---------------------------------------------------------------------------

# Canonical pair display order (top to bottom on y-axis)
PAIR_ORDER = [
    ("inj_ctrl",            "wik_ab"),
    ("inj_ctrl",            "pbx1b_crispant"),
    ("wik_ab",              "pbx1b_crispant"),
    ("inj_ctrl",            "pbx4_crispant"),
    ("wik_ab",              "pbx4_crispant"),
    ("pbx1b_crispant",      "pbx4_crispant"),
    ("inj_ctrl",            "pbx1b_pbx4_crispant"),
    ("wik_ab",              "pbx1b_pbx4_crispant"),
    ("pbx1b_crispant",      "pbx1b_pbx4_crispant"),
    ("pbx4_crispant",       "pbx1b_pbx4_crispant"),
]

STATE_COLORS = {
    NOT_SEPARATED: "#2166AC",   # blue  = still together
    AMBIGUOUS:     "#d9d9d9",   # light gray = uncertain
    SEPARATED:     "#B2182B",   # red   = distinguished
}


def plot_pairwise_stripe(
    classified_df: pd.DataFrame,
    split_results: list[dict],
    ax: plt.Axes,
) -> None:
    """
    x-axis: time
    y-axis: class pair (one row per pair)
    color:  blue=not_separated, gray=ambiguous, red=separated
    """
    time_bins = sorted(classified_df["time_bin_center"].unique())

    # Build pair → row index mapping
    pairs_present = []
    for ci, cj in PAIR_ORDER:
        pk = f"{min(ci,cj)}__{max(ci,cj)}"
        if pk in classified_df["pair_key"].values:
            pairs_present.append((ci, cj, pk))

    n_pairs = len(pairs_present)
    n_times = len(time_bins)
    t_idx   = {t: i for i, t in enumerate(time_bins)}

    # Build color matrix (n_pairs × n_times)
    color_matrix = np.full((n_pairs, n_times, 3), 0.85)  # default light gray

    for row_i, (ci, cj, pk) in enumerate(pairs_present):
        sub = classified_df[classified_df["pair_key"] == pk]
        for _, r in sub.iterrows():
            col_i = t_idx.get(r["time_bin_center"])
            if col_i is None:
                continue
            rgb = plt.matplotlib.colors.to_rgb(STATE_COLORS[r["edge_state"]])
            color_matrix[row_i, col_i] = rgb

    ax.imshow(
        color_matrix,
        aspect="auto",
        extent=[min(time_bins) - 2, max(time_bins) + 2, n_pairs - 0.5, -0.5],
        interpolation="nearest",
    )

    # y-axis labels
    pair_labels = [
        f"{CLASS_LABELS.get(ci,ci)} vs {CLASS_LABELS.get(cj,cj)}"
        for ci, cj, _ in pairs_present
    ]
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels, fontsize=7.5)
    ax.set_xlabel("Stage (hpf)", fontsize=9)
    ax.set_title("Panel B: Raw Pairwise Tri-State", fontsize=10, fontweight="bold")

    # Split time lines
    for s in split_results:
        if np.isinf(s["split_time"]):
            continue
        ax.axvline(s["split_time"], color="#444", linewidth=1.0, linestyle="--", alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=STATE_COLORS[NOT_SEPARATED], label="Not separated"),
        mpatches.Patch(facecolor=STATE_COLORS[AMBIGUOUS],     label="Ambiguous"),
        mpatches.Patch(facecolor=STATE_COLORS[SEPARATED],     label="Separated"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5,
              framealpha=0.9, edgecolor="#ccc")

    # Horizontal separator lines between control-vs-control / control-vs-crispant groups
    # After pair index 2 (end of ctrl-ctrl group) and 5 (end of ctrl-crispant group)
    group_breaks = [2.5, 5.5]
    for yb in group_breaks:
        if yb < n_pairs:
            ax.axhline(yb, color="#555", linewidth=0.8, linestyle=":", alpha=0.6)


# ---------------------------------------------------------------------------
# Panel C: Bundled class-line emergence plot
# ---------------------------------------------------------------------------

def _partition_y_positions(
    classes_in_order: list[str],
    block_map: dict[str, str],
    y_spread: float = 0.3,
    bundle_gap: float = 0.8,
) -> dict[str, float]:
    """
    Assign a y-position to each class at a given time bin.

    Classes in the same block are bundled close together (spaced by y_spread).
    Different blocks are separated by bundle_gap.

    The vertical ordering of blocks follows the first appearance of each
    block's classes in classes_in_order.
    """
    # Group classes by block, preserving order
    seen_blocks: list[str] = []
    block_members: dict[str, list[str]] = {}
    for c in classes_in_order:
        bl = block_map.get(c, c)
        if bl not in block_members:
            block_members[bl] = []
            seen_blocks.append(bl)
        block_members[bl].append(c)

    y = 0.0
    positions: dict[str, float] = {}
    for bi, bl in enumerate(seen_blocks):
        members = block_members[bl]
        n = len(members)
        # Center the bundle around y, spread members within it
        offsets = np.linspace(-(n - 1) * y_spread / 2, (n - 1) * y_spread / 2, n)
        for c, off in zip(members, offsets):
            positions[c] = y + off
        y += bundle_gap

    return positions


def plot_bundled_emergence(
    partitions_df: pd.DataFrame,
    split_results: list[dict],
    ax: plt.Axes,
) -> None:
    """
    One colored line per class. Lines travel together (same y) when in the
    same partition block, and spread apart after their block splits.

    The y-positions are derived from the coherent partition — not from raw
    pairwise relations — so the geometry is always consistent.
    """
    time_bins = sorted(partitions_df["time_bin"].unique())
    classes   = [c for c in CLASS_ORDER if c in partitions_df["class"].unique()]

    # For each time bin, compute y-position of each class
    all_positions: dict[float, dict[str, float]] = {}
    for t in time_bins:
        t_df = partitions_df[partitions_df["time_bin"] == t].set_index("class")
        block_map = {c: t_df.loc[c, "partition_block_label"] for c in classes if c in t_df.index}
        all_positions[t] = _partition_y_positions(classes, block_map)

    # Draw one line per class across all time bins
    for c in classes:
        xs = []
        ys = []
        for t in time_bins:
            y = all_positions[t].get(c)
            if y is not None:
                xs.append(t)
                ys.append(y)
        color = CLASS_COLORS.get(c, "#888")
        ax.plot(xs, ys, color=color, linewidth=2.5, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)

        # Label at the right end
        if xs:
            ax.text(xs[-1] + 0.8, ys[-1], CLASS_LABELS.get(c, c),
                    fontsize=7.5, va="center", ha="left", color=color, fontweight="bold")

    # Draw bundle region fills (shaded background per block per time segment)
    # For each consecutive pair of time bins, shade the bundle region
    for t_idx in range(len(time_bins) - 1):
        t0, t1 = time_bins[t_idx], time_bins[t_idx + 1]
        pos0 = all_positions[t0]
        pos1 = all_positions[t1]

        # Identify blocks at t0
        t_df0 = partitions_df[partitions_df["time_bin"] == t0].set_index("class")
        blocks_at_t0: dict[str, list[str]] = {}
        for c in classes:
            if c not in t_df0.index:
                continue
            bl = t_df0.loc[c, "partition_block_label"]
            blocks_at_t0.setdefault(bl, []).append(c)

        for bl, members in blocks_at_t0.items():
            if len(members) < 2:
                continue
            ys0 = [pos0[c] for c in members if c in pos0]
            ys1 = [pos1[c] for c in members if c in pos1]
            if not ys0 or not ys1:
                continue
            y_bot = min(min(ys0), min(ys1)) - 0.12
            y_top = max(max(ys0), max(ys1)) + 0.12
            ax.fill_betweenx(
                [y_bot, y_top], t0, t1,
                color="#eeeeee", alpha=0.55, zorder=1, linewidth=0,
            )

    # Split time markers
    splits_sorted = sorted(split_results, key=lambda x: x["split_time"])
    y_all = [y for pos in all_positions.values() for y in pos.values()]
    y_top = max(y_all) + 0.5 if y_all else 4.0
    y_bot = min(y_all) - 0.5 if y_all else 0.0

    for s in splits_sorted:
        if np.isinf(s["split_time"]):
            continue
        ax.axvline(s["split_time"], color="#555", linewidth=1.0, linestyle="--",
                   alpha=0.75, zorder=2)
        label = ", ".join(CLASS_LABELS.get(c, c) for c in sorted(s["g1"]))
        ax.text(s["split_time"] + 0.4, y_top * 0.97, label,
                fontsize=6.5, rotation=90, va="top", ha="left", color="#444")

    ax.set_xlim(min(time_bins) - 4, max(time_bins) + 12)
    ax.set_ylim(y_bot - 0.3, y_top + 0.5)
    ax.set_xlabel("Stage (hpf)", fontsize=9)
    ax.set_yticks([])
    ax.set_title("Panel C: Bundled Class-Line Emergence", fontsize=10, fontweight="bold")
    ax.spines[["left", "right", "top"]].set_visible(False)


# ---------------------------------------------------------------------------
# Panel E: Static composite block tree (dendrogram-style)
# ---------------------------------------------------------------------------

def _build_block_tree(split_results: list[dict], all_classes: list[str]) -> list[dict]:
    """
    Build a tree of partition blocks from split_results.

    Returns a list of node dicts:
        id          : unique int
        members     : frozenset of class names
        split_time  : time at which this node's children diverge (NaN = leaf)
        parent_id   : int or None
        children_ids: list of ints
        is_leaf     : bool
        depth       : int (root = 0)
    """
    nodes = []
    next_id = [0]

    def new_node(members, parent_id, depth):
        nid = next_id[0]
        next_id[0] += 1
        nodes.append({
            "id": nid,
            "members": frozenset(members),
            "split_time": float("nan"),
            "parent_id": parent_id,
            "children_ids": [],
            "is_leaf": True,
            "depth": depth,
        })
        return nid

    # Root = all classes
    root_id = new_node(all_classes, None, 0)

    # Apply splits in time order — same monotone logic as build_coherent_partitions
    splits_sorted = sorted(
        [s for s in split_results if not np.isinf(s["split_time"])],
        key=lambda x: x["split_time"],
    )

    def find_node(members_subset):
        """Find the current leaf node that contains members_subset."""
        for node in nodes:
            if node["is_leaf"] and frozenset(members_subset).issubset(node["members"]):
                return node
        return None

    for s in splits_sorted:
        g1 = frozenset(s["g1"])
        parent_node = find_node(g1)
        if parent_node is None:
            continue
        remainder = parent_node["members"] - g1
        # Convert parent to internal node
        parent_node["is_leaf"]    = False
        parent_node["split_time"] = s["split_time"]
        # Create children
        c1 = new_node(g1,        parent_node["id"], parent_node["depth"] + 1)
        c2 = new_node(remainder, parent_node["id"], parent_node["depth"] + 1)
        parent_node["children_ids"] = [c1, c2]

    return nodes


def _assign_leaf_x(nodes: list[dict], leaf_gap: float = 1.0) -> dict[int, float]:
    """
    Assign x positions to leaf nodes (evenly spaced), then propagate
    x to internal nodes as the midpoint of their children.
    Returns dict node_id -> x.
    """
    # Find leaves in left-to-right order (DFS)
    node_by_id = {n["id"]: n for n in nodes}
    root = next(n for n in nodes if n["parent_id"] is None)

    leaves_ordered = []
    stack = [root["id"]]
    while stack:
        nid = stack.pop()
        node = node_by_id[nid]
        if node["is_leaf"]:
            leaves_ordered.append(nid)
        else:
            # Push children right-first so left child is processed first
            for cid in reversed(node["children_ids"]):
                stack.append(cid)

    x_pos = {}
    for i, nid in enumerate(leaves_ordered):
        x_pos[nid] = i * leaf_gap

    # Propagate up: internal node x = mean of children x
    def _propagate(nid):
        node = node_by_id[nid]
        if node["is_leaf"]:
            return x_pos[nid]
        child_xs = [_propagate(cid) for cid in node["children_ids"]]
        x_pos[nid] = float(np.mean(child_xs))
        return x_pos[nid]

    _propagate(root["id"])
    return x_pos


def plot_block_tree(
    split_results: list[dict],
    all_classes: list[str],
    t_max: float,
    ax: plt.Axes,
    leaf_gap: float = 1.2,
    branch_lw: float = 3.5,
    leaf_box_h: float = 0.14,
    leaf_box_w: float = 0.55,
) -> None:
    """
    Panel E: static dendrogram-style composite block tree.

    x-axis = horizontal position (arbitrary, set by leaf layout)
    y-axis = time (hpf), root at top (t=0), leaves at bottom

    Each leaf is drawn as a stacked set of colored rectangles — one per
    class in that block. Internal nodes are connected by L-shaped lines
    whose junction height = split_time.
    """
    nodes   = _build_block_tree(split_results, all_classes)
    x_pos   = _assign_leaf_x(nodes, leaf_gap=leaf_gap)
    node_by_id = {n["id"]: n for n in nodes}

    # y-axis: time increases downward (root at top = early, leaves at bottom = t_max)
    # We invert so that splits appear at the right time.
    # y = split_time for internal nodes, y = t_max for leaves.

    def node_y(node):
        return t_max if node["is_leaf"] else node["split_time"]

    # Draw edges: for each internal node, draw L-shaped connectors to children
    for node in nodes:
        if node["is_leaf"]:
            continue
        nx  = x_pos[node["id"]]
        ny  = node_y(node)
        for cid in node["children_ids"]:
            child   = node_by_id[cid]
            cx      = x_pos[cid]
            cy      = node_y(child)
            # Vertical down from parent, then horizontal to child x, then vertical to child
            ax.plot([nx, nx], [ny, ny],      color="#888", lw=0.8, zorder=1)
            ax.plot([nx, cx], [ny, ny],      color="#555", lw=1.2, linestyle="-", zorder=2)
            ax.plot([cx, cx], [ny, cy],      color="#555", lw=1.2, zorder=2)

        # Dot at split point
        ax.scatter([nx], [ny], color="#333", s=18, zorder=4, linewidths=0)

    # Draw leaves: stacked colored boxes below the terminal lines.
    # Boxes are drawn in *axes fraction* coordinates via a transform so
    # their height is independent of the time y-axis scale.
    leaves      = [n for n in nodes if n["is_leaf"]]
    # Compute a fixed pixel-height per lane using a blended transform
    from matplotlib.transforms import blended_transform_factory
    xdata_yax = blended_transform_factory(ax.transData, ax.transData)

    # Use a simple approach: draw boxes at a fixed y below t_max in data coords.
    # One lane = (t_max_range / n_lanes_total) / 8 so they're visible but compact.
    n_total_classes = len(all_classes)
    lane_h_data = (t_max) / (n_total_classes * 8)   # small fraction of full y range

    box_y_start = t_max + lane_h_data * 1.5
    for node in leaves:
        members_ordered = [c for c in CLASS_ORDER if c in node["members"]]
        nx = x_pos[node["id"]]
        for ci, c in enumerate(members_ordered):
            color = CLASS_COLORS.get(c, "#888")
            y_box = box_y_start + ci * lane_h_data
            rect  = plt.Rectangle(
                (nx - leaf_box_w / 2, y_box),
                leaf_box_w, lane_h_data,
                facecolor=color, edgecolor="white", linewidth=0.4,
                alpha=0.92, zorder=3,
            )
            ax.add_patch(rect)
            label_color = "white" if c in ("pbx1b_pbx4_crispant", "pbx1b_crispant", "inj_ctrl") else "#333"
            ax.text(nx, y_box + lane_h_data * 0.5,
                    CLASS_LABELS.get(c, c),
                    ha="center", va="center", fontsize=6.0,
                    color=label_color, fontweight="bold", zorder=4)

        # Vertical line from t_max down to first box
        ax.plot([nx, nx], [node_y(node), box_y_start], color="#555", lw=1.2, zorder=2)

    # Root: vertical stem above first split
    root = next(n for n in nodes if n["parent_id"] is None)
    root_split_t = node_y(root)
    rx = x_pos[root["id"]]
    ax.plot([rx, rx], [0, root_split_t], color="#555", lw=1.2, zorder=2)
    ax.text(rx, 2, "All classes", ha="center", va="top", fontsize=7, color="#555")

    # Axes — y increases downward = time flows down
    n_leaves = len(leaves)

    # Split time horizontal reference lines and labels
    x_right = (n_leaves - 1) * leaf_gap + leaf_gap * 0.7
    for s in split_results:
        if np.isinf(s["split_time"]):
            continue
        t = s["split_time"]
        ax.axhline(t, color="#ccc", linewidth=0.7, linestyle=":", zorder=0)
        ax.text(x_right, t, f"{t:.0f} hpf", ha="right", va="bottom", fontsize=7, color="#666")
    max_block_size = max(len(n["members"]) for n in leaves) if leaves else 1
    y_bottom = box_y_start + max_block_size * lane_h_data + lane_h_data
    ax.set_xlim(-leaf_gap, (n_leaves - 1) * leaf_gap + leaf_gap)
    ax.set_ylim(y_bottom, -5)
    ax.set_ylabel("Stage (hpf)", fontsize=9)
    ax.set_xticks([])
    ax.set_title("Panel E: Composite Block Tree", fontsize=10, fontweight="bold")
    ax.spines[["bottom", "right", "top"]].set_visible(False)

    # Annotate split times on y-axis
    for s in split_results:
        if np.isinf(s["split_time"]):
            continue
        ax.axhline(s["split_time"], color="#ddd", lw=0.6, linestyle=":", zorder=0)


# ---------------------------------------------------------------------------
# Panel D: Block-branch partition plot
# ---------------------------------------------------------------------------

def _build_block_layout(
    partitions_df: pd.DataFrame,
    split_results: list[dict],
    classes_in_order: list[str],
    lane_height: float = 0.18,
    block_gap: float = 0.55,
) -> pd.DataFrame:
    """
    Build a layout table describing how to draw each block at each time bin.

    Each block is a stacked ribbon of colored lanes — one lane per member class.
    Block y-extents are computed so sibling blocks at the same level are
    separated by block_gap.

    Returns block_layout_df with columns:
        time_bin, block_label, member_classes (list), n_members,
        y_min, y_max, y_center
    Plus a separate splits_layout list: {split_time, parent_label, child_labels}
    """
    time_bins = sorted(partitions_df["time_bin"].unique())

    # Build split history: list of (split_time, partition_as_frozenset_list)
    # Re-derive from split_results in time order (same logic as build_coherent_partitions)
    all_classes = set(classes_in_order)
    splits_sorted = sorted(
        [s for s in split_results if not np.isinf(s["split_time"])],
        key=lambda x: x["split_time"],
    )

    # At each time bin, figure out the active partition and assign y positions
    # We want blocks that share a common ancestry to stay centered together.
    # Strategy: assign y-positions based on the order of classes in classes_in_order
    # within each block, then center blocks in the vertical space their members occupy.

    rows = []
    for t in time_bins:
        t_df = partitions_df[partitions_df["time_bin"] == t].set_index("class")

        # Group classes into blocks, preserving classes_in_order
        block_members: dict[str, list[str]] = {}
        for c in classes_in_order:
            if c not in t_df.index:
                continue
            bl = t_df.loc[c, "partition_block_label"]
            block_members.setdefault(bl, [])
            block_members[bl].append(c)

        # Assign y positions: stack all lanes top-to-bottom with gaps between blocks
        y = 0.0
        for bi, (bl, members) in enumerate(block_members.items()):
            if bi > 0:
                y += block_gap
            n = len(members)
            y_min = y
            y_max = y + n * lane_height
            y_center = (y_min + y_max) / 2
            rows.append({
                "time_bin": t,
                "block_label": bl,
                "member_classes": members,
                "n_members": n,
                "y_min": y_min,
                "y_max": y_max,
                "y_center": y_center,
            })
            y = y_max

    return pd.DataFrame(rows)


def plot_block_branch(
    partitions_df: pd.DataFrame,
    split_results: list[dict],
    ax: plt.Axes,
    lane_height: float = 0.18,
    block_gap: float = 0.55,
) -> None:
    """
    Panel D: block-branch partition plot.

    Branches = coherent partition blocks.
    Each branch is drawn as a stacked ribbon of colored lanes.
    When a block splits, the parent ribbon divides into child ribbons.
    Classes that remain unresolved within a block share a ribbon without
    implying a separate path for each.
    """
    classes = [c for c in CLASS_ORDER if c in partitions_df["class"].unique()]
    layout  = _build_block_layout(
        partitions_df, split_results, classes,
        lane_height=lane_height, block_gap=block_gap,
    )
    time_bins = sorted(layout["time_bin"].unique())
    bin_w = (time_bins[1] - time_bins[0]) * 0.88 if len(time_bins) > 1 else 4.0

    # Draw each block at each time bin as stacked colored rectangles
    for _, row in layout.iterrows():
        t        = row["time_bin"]
        members  = row["member_classes"]
        y_base   = row["y_min"]
        x_left   = t - bin_w / 2
        for ci, c in enumerate(members):
            color = CLASS_COLORS.get(c, "#888")
            rect  = plt.Rectangle(
                (x_left, y_base + ci * lane_height),
                bin_w, lane_height,
                facecolor=color, edgecolor="white", linewidth=0.25, alpha=0.90,
            )
            ax.add_patch(rect)

    # Draw connecting lines between consecutive time bins for each block.
    # A block persists if its label appears in both t and t+1.
    # At a split, the parent label disappears and child labels appear.
    for ti in range(len(time_bins) - 1):
        t0, t1 = time_bins[ti], time_bins[ti + 1]
        t0_df  = layout[layout["time_bin"] == t0].set_index("block_label")
        t1_df  = layout[layout["time_bin"] == t1].set_index("block_label")

        # Blocks present in both: draw horizontal continuation lines per lane
        for bl in t0_df.index:
            if bl not in t1_df.index:
                continue
            members = t0_df.loc[bl, "member_classes"]
            y0_base = t0_df.loc[bl, "y_min"]
            y1_base = t1_df.loc[bl, "y_min"]
            x0 = t0 + bin_w / 2
            x1 = t1 - bin_w / 2
            for ci, c in enumerate(members):
                color = CLASS_COLORS.get(c, "#888")
                y0 = y0_base + (ci + 0.5) * lane_height
                y1 = y1_base + (ci + 0.5) * lane_height
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.2,
                        alpha=0.7, zorder=2, solid_capstyle="round")

    # Class labels at the right end
    t_last = time_bins[-1]
    t_last_df = layout[layout["time_bin"] == t_last]
    for _, row in t_last_df.iterrows():
        members = row["member_classes"]
        y_base  = row["y_min"]
        x_label = t_last + bin_w / 2 + 1.0
        for ci, c in enumerate(members):
            y = y_base + (ci + 0.5) * lane_height
            color = CLASS_COLORS.get(c, "#888")
            ax.text(x_label, y, CLASS_LABELS.get(c, c),
                    fontsize=7.5, va="center", ha="left",
                    color=color, fontweight="bold")

    # Split time markers
    splits_sorted = sorted(split_results, key=lambda x: x["split_time"])
    y_all = layout["y_max"].values
    y_top = float(y_all.max()) + 0.2 if len(y_all) > 0 else 2.0
    for s in splits_sorted:
        if np.isinf(s["split_time"]):
            continue
        ax.axvline(s["split_time"], color="#555", linewidth=1.0,
                   linestyle="--", alpha=0.7, zorder=3)
        label = ", ".join(CLASS_LABELS.get(c, c) for c in sorted(s["g1"]))
        ax.text(s["split_time"] + 0.4, y_top * 0.98, label,
                fontsize=6.5, rotation=90, va="top", ha="left", color="#444")

    # Axes
    ax.set_xlim(min(time_bins) - 4, max(time_bins) + 14)
    ax.set_ylim(-0.15, y_top + 0.4)
    ax.set_xlabel("Stage (hpf)", fontsize=9)
    ax.set_yticks([])
    ax.set_title("Panel D: Block-Branch Partition", fontsize=10, fontweight="bold")
    ax.spines[["left", "right", "top"]].set_visible(False)

    # Legend: one patch per class
    legend_handles = [
        mpatches.Patch(facecolor=CLASS_COLORS.get(c, "#888"), label=CLASS_LABELS.get(c, c))
        for c in classes
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7.5,
              framealpha=0.9, edgecolor="#ccc", ncol=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading scores...")
    scores_df = load_scores()
    all_time_bins = sorted(scores_df["time_bin_center"].unique())
    all_classes   = sorted(
        set(scores_df["positive_label"].unique()) |
        set(scores_df["negative_label"].unique())
    )

    print(f"  {len(all_classes)} classes, {len(all_time_bins)} time bins")
    print(f"  Params: p_sep={PARAMS.p_sep}, auroc_sep={PARAMS.auroc_sep}, "
          f"p_ns={PARAMS.p_ns}, L={PARAMS.L}")

    # --- Full transitivity report ---
    print("Building transitivity report...")
    report = build_transitivity_report(
        scores_df, PARAMS,
        time_col="time_bin_center",
        class_i_col="positive_label",
        class_j_col="negative_label",
        pval_col="pval",
        auroc_col="auroc_obs",
    )

    # --- Print edge state overview ---
    state_counts = report.classified_df["edge_state"].value_counts()
    print(f"  Edge states across all (pair, time) rows:\n{state_counts.to_string()}")

    # --- Onset summary ---
    print("\nOnset summary (L=3 consecutive rule):")
    print(report.onset_df[["class_i","class_j","onset_hpf","n_separated_bins","n_total_bins"]].to_string(index=False))

    print("\nOnset matrix:")
    print(report.onset_matrix.to_string())

    # --- Save outputs ---
    report.onset_matrix.to_csv(OUT_DIR / "onset_matrix.csv")
    report.onset_df.to_csv(OUT_DIR / "onset_pairs.csv", index=False)
    report.timebin_summary.to_csv(OUT_DIR / "transitivity_timebin_summary.csv", index=False)
    report.triple_violations.to_csv(OUT_DIR / "transitivity_violations.csv", index=False)
    report.onset_triple_df.to_csv(OUT_DIR / "ultrametric_triples.csv", index=False)

    # Ultrametric summary
    s = report.onset_summary
    um_lines = [
        f"n_triples_total:        {s.n_triples_total}",
        f"n_triples_evaluable:    {s.n_triples_evaluable}",
        f"n_gap_zero:             {s.n_gap_zero}",
        f"frac_gap_zero:          {s.frac_gap_zero:.3f}" if not np.isnan(s.frac_gap_zero) else "frac_gap_zero: N/A",
        f"mean_gap_hpf:           {s.mean_gap:.2f}"   if not np.isnan(s.mean_gap)   else "mean_gap_hpf: N/A",
        f"median_gap_hpf:         {s.median_gap:.2f}" if not np.isnan(s.median_gap) else "median_gap_hpf: N/A",
        f"max_gap_hpf:            {s.max_gap:.2f}"    if not np.isnan(s.max_gap)    else "max_gap_hpf: N/A",
    ]
    print("\nUltrametric consistency:")
    for line in um_lines:
        print(" ", line)
    (OUT_DIR / "ultrametric_summary.txt").write_text("\n".join(um_lines) + "\n")

    # Violation summary
    tb = report.timebin_summary
    evaluable_t = tb[tb["n_triples_evaluable"] > 0]
    if len(evaluable_t) > 0:
        rate = (evaluable_t["n_violations"] > 0).mean()
        print(f"\nTransitivity: {rate:.1%} of evaluable time bins have at least one violation")
        print(f"  (evaluable = all 3 edges confidently classified)")
        print(f"  sep_sep_notsep violations: {report.triple_violations[report.triple_violations['violation_type']=='sep_sep_notsep'].shape[0]}")
        print(f"  ns_ns_sep violations:      {report.triple_violations[report.triple_violations['violation_type']=='ns_ns_sep'].shape[0]}")

    # --- Coherent partitions ---
    print("\nInferring coherent split times...")
    split_results = infer_split_times(report.onset_matrix)
    for s in split_results:
        label = " | ".join([sorted(s["g1"]).__str__(), sorted(s["g2"]).__str__()])
        inf_str = "inf (no durable onset)" if np.isinf(s["split_time"]) else f"{s['split_time']:.1f} hpf"
        print(f"  {inf_str}  →  {sorted(s['g1'])} splits from {sorted(s['g2'])}")

    partitions_df = build_coherent_partitions(split_results, all_time_bins)
    partitions_df.to_csv(OUT_DIR / "coherent_partitions.csv", index=False)

    # --- Panel A ---
    print("\nPanel A: onset heatmap...")
    fig_a, ax_a = plt.subplots(figsize=(6.5, 5.5))
    plot_onset_heatmap(report.onset_matrix, ax_a)
    fig_a.tight_layout()
    fig_a.savefig(OUT_DIR / "panel_A_onset_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig_a)

    # --- Panel B: pairwise stripe plot ---
    print("Panel B: pairwise stripe plot...")
    fig_b, ax_b = plt.subplots(figsize=(13, 4.5))
    plot_pairwise_stripe(report.classified_df, split_results, ax_b)
    fig_b.tight_layout()
    fig_b.savefig(OUT_DIR / "panel_B_pairwise_stripe.png", dpi=150, bbox_inches="tight")
    plt.close(fig_b)

    # --- Panel C: bundled class-line emergence ---
    print("Panel C: bundled class-line emergence...")
    fig_c, ax_c = plt.subplots(figsize=(13, 4))
    plot_bundled_emergence(partitions_df, split_results, ax_c)
    fig_c.tight_layout()
    fig_c.savefig(OUT_DIR / "panel_C_bundled_emergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig_c)

    # --- Panel E: composite block tree ---
    print("Panel E: composite block tree...")
    fig_e, ax_e = plt.subplots(figsize=(5, 7))
    plot_block_tree(
        split_results, all_classes,
        t_max=max(all_time_bins), ax=ax_e,
    )
    fig_e.tight_layout()
    fig_e.savefig(OUT_DIR / "panel_E_block_tree.png", dpi=150, bbox_inches="tight")
    plt.close(fig_e)

    # --- Panel D: block-branch ---
    print("Panel D: block-branch partition...")
    fig_d, ax_d = plt.subplots(figsize=(13, 3.5))
    plot_block_branch(partitions_df, split_results, ax_d)
    fig_d.tight_layout()
    fig_d.savefig(OUT_DIR / "panel_D_block_branch.png", dpi=150, bbox_inches="tight")
    plt.close(fig_d)

    print(f"\nDone. Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
