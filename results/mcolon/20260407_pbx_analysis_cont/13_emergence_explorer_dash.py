"""
13_emergence_explorer_dash.py
------------------------------
Interactive phenotype emergence explorer — Dash app.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \
        results/mcolon/20260407_pbx_analysis_cont/13_emergence_explorer_dash.py

Then open http://127.0.0.1:8050 in a browser.

Controls
--------
  Included genotypes  → recomputes block tree on selected subset
  Resolve relative to → anchor for tree orientation

Left panel:  adaptive composite block tree (Plotly scatter)
Right panel: full onset heatmap — excluded genotypes grayed,
             onset in hpf as cell text, '—' for never-separated
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, callback

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from analyze.classification.transitivity import (
    TransitivityParams,
    classify_pair_state_over_time,
    compute_pair_onsets,
    build_onset_matrix,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "results/positioning/pairwise/combined_pairwise_5class_bin4_perm500"

PARAMS = TransitivityParams(p_sep=0.01, auroc_sep=0.70, p_ns=0.10, L=3)

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

ALL_CLASSES = [
    "pbx1b_pbx4_crispant",
    "pbx4_crispant",
    "pbx1b_crispant",
    "inj_ctrl",
    "wik_ab",
]

SUBSET_OPTIONS = [
    {"label": "All genotypes",        "value": "all"},
    {"label": "Exclude WIK/AB",       "value": "no_wik_ab"},
    {"label": "Exclude pbx4",         "value": "no_pbx4"},
    {"label": "Exclude pbx1b",        "value": "no_pbx1b"},
    {"label": "Exclude pbx1b;pbx4",   "value": "no_double"},
    {"label": "Crispants only",       "value": "crispants"},
]

ANCHOR_OPTIONS = [
    {"label": "Global (no anchor)",              "value": "none"},
    {"label": "Resolve relative to: Controls",   "value": "controls"},
    {"label": "Resolve relative to: Inj. Ctrl",  "value": "inj_ctrl"},
    {"label": "Resolve relative to: WIK/AB",     "value": "wik_ab"},
    {"label": "Resolve relative to: pbx1b;pbx4", "value": "double"},
]

SUBSET_MAP = {
    "all":       list(ALL_CLASSES),
    "no_wik_ab": [c for c in ALL_CLASSES if c != "wik_ab"],
    "no_pbx4":   [c for c in ALL_CLASSES if c != "pbx4_crispant"],
    "no_pbx1b":  [c for c in ALL_CLASSES if c != "pbx1b_crispant"],
    "no_double":  [c for c in ALL_CLASSES if c != "pbx1b_pbx4_crispant"],
    "crispants": ["pbx1b_pbx4_crispant", "pbx4_crispant", "pbx1b_crispant"],
}

ANCHOR_MAP = {
    "none":     None,
    "controls": frozenset({"inj_ctrl", "wik_ab"}),
    "inj_ctrl": frozenset({"inj_ctrl"}),
    "wik_ab":   frozenset({"wik_ab"}),
    "double":   frozenset({"pbx1b_pbx4_crispant"}),
}

# ---------------------------------------------------------------------------
# Data — loaded once at startup
# ---------------------------------------------------------------------------

def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_parquet(DATA_DIR / "scores.parquet")
    scores = scores[scores["feature_set"] == "vae"].copy().reset_index(drop=True)
    classified = classify_pair_state_over_time(
        scores, PARAMS,
        time_col="time_bin_center",
        class_i_col="positive_label",
        class_j_col="negative_label",
        pval_col="pval",
        auroc_col="auroc_obs",
    )
    return scores, classified


print("Loading data...")
_, CLASSIFIED_DF = _load()

# Full onset matrix (all 5 classes)
_onset_df_full = compute_pair_onsets(
    CLASSIFIED_DF, PARAMS,
    time_col="time_bin_center",
    class_i_col="positive_label",
    class_j_col="negative_label",
)
FULL_ONSET_MAT = build_onset_matrix(_onset_df_full, ALL_CLASSES)
print("Data loaded.")

# Pre-compute color scale bounds
_finite_vals = [
    float(FULL_ONSET_MAT.loc[a, b])
    for a in ALL_CLASSES for b in ALL_CLASSES
    if a != b and not pd.isna(FULL_ONSET_MAT.loc[a, b])
]
VMIN = min(_finite_vals) if _finite_vals else 0.0
VMAX = max(_finite_vals) if _finite_vals else 1.0


# ---------------------------------------------------------------------------
# Block tree helpers
# ---------------------------------------------------------------------------

def get_subset_onset_matrix(selected: list[str]) -> pd.DataFrame:
    return FULL_ONSET_MAT.reindex(index=selected, columns=selected)


def infer_terminal_blocks(mat: pd.DataFrame, selected: list[str]) -> list[frozenset]:
    classes = [c for c in selected if c in mat.index]
    parent = {c: c for c in classes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b in combinations(classes, 2):
        if pd.isna(mat.loc[a, b]):
            union(a, b)

    groups: dict[str, list[str]] = {}
    for c in classes:
        groups.setdefault(find(c), []).append(c)
    return [frozenset(g) for g in groups.values()]


def cross_mean(mat: pd.DataFrame, g1: frozenset, g2: frozenset) -> float:
    vals = [float(mat.loc[a, b])
            for a in g1 for b in g2
            if a in mat.index and b in mat.columns and not pd.isna(mat.loc[a, b])]
    return float(np.mean(vals)) if vals else float("inf")


def build_split_sequence(
    mat: pd.DataFrame,
    terminal_blocks: list[frozenset],
    anchor: frozenset | None,
) -> list[dict]:
    if len(terminal_blocks) <= 1:
        return []

    meta = [frozenset([b]) for b in terminal_blocks]

    def meta_cls(mb):
        return frozenset().union(*mb)

    merges = []
    while len(meta) > 1:
        best_t, best_i, best_j = -1.0, 0, 1
        for i, j in combinations(range(len(meta)), 2):
            t = cross_mean(mat, meta_cls(meta[i]), meta_cls(meta[j]))
            if t == float("inf"):
                t = -1.0
            if t > best_t:
                best_t, best_i, best_j = t, i, j

        mb_i, mb_j = meta[best_i], meta[best_j]
        split_t = cross_mean(mat, meta_cls(mb_i), meta_cls(mb_j))
        if split_t == float("inf"):
            split_t = float("nan")

        if anchor is not None and anchor.issubset(meta_cls(mb_i)):
            mb_i, mb_j = mb_j, mb_i

        merges.append({
            "g1": meta_cls(mb_i),
            "g2": meta_cls(mb_j),
            "split_time": split_t,
        })
        meta = [m for k, m in enumerate(meta) if k not in (best_i, best_j)]
        meta.append(mb_i | mb_j)

    return list(reversed(merges))


def build_tree_nodes(split_seq: list[dict], selected: list[str]) -> list[dict]:
    nodes: list[dict] = []
    nid = [0]

    def new_node(members, parent_id):
        n = nid[0]; nid[0] += 1
        nodes.append({"id": n, "members": list(members),
                      "split_time": None, "parent_id": parent_id,
                      "children_ids": [], "is_leaf": True})
        return n

    new_node(selected, None)

    def find_leaf(subset):
        s = set(subset)
        for node in nodes:
            if node["is_leaf"] and s.issubset(set(node["members"])):
                return node
        return None

    for split in split_seq:
        g1 = set(split["g1"])
        parent = find_leaf(g1)
        if parent is None:
            continue
        remainder = set(parent["members"]) - g1
        parent["is_leaf"] = False
        t = split["split_time"]
        parent["split_time"] = None if (t != t) else t
        c1 = new_node(g1, parent["id"])
        c2 = new_node(remainder, parent["id"])
        parent["children_ids"] = [c1, c2]

    return nodes


def assign_x(nodes: list[dict]) -> dict[int, float]:
    by_id = {n["id"]: n for n in nodes}
    root  = next(n for n in nodes if n["parent_id"] is None)
    leaves = []
    stack = [root["id"]]
    while stack:
        nid = stack.pop()
        nd = by_id[nid]
        if nd["is_leaf"]:
            leaves.append(nid)
        else:
            for cid in reversed(nd["children_ids"]):
                stack.append(cid)
    x = {nid: float(i) for i, nid in enumerate(leaves)}

    def prop(nid):
        nd = by_id[nid]
        if nd["is_leaf"]:
            return x[nid]
        xs = [prop(c) for c in nd["children_ids"]]
        x[nid] = float(np.mean(xs))
        return x[nid]
    prop(root["id"])
    return x


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_tree_figure(selected: list[str], anchor: frozenset | None) -> go.Figure:
    mat    = get_subset_onset_matrix(selected)
    blocks = infer_terminal_blocks(mat, selected)
    splits = build_split_sequence(mat, blocks, anchor)
    nodes  = build_tree_nodes(splits, selected)
    x_pos  = assign_x(nodes)
    by_id  = {n["id"]: n for n in nodes}

    split_times = [n["split_time"] for n in nodes
                   if not n["is_leaf"] and n["split_time"] is not None]
    t_min = min(split_times) if split_times else 20.0
    t_max = max(split_times) if split_times else 120.0
    t_pad = max((t_max - t_min) * 0.15, 5.0)
    y_leaf = t_max + t_pad * 1.2

    traces = []

    # Connector lines
    line_x, line_y = [], []
    for nd in nodes:
        if nd["is_leaf"]:
            continue
        nx = x_pos[nd["id"]]
        ny = nd["split_time"]
        for cid in nd["children_ids"]:
            ch = by_id[cid]
            cx = x_pos[cid]
            cy = ch["split_time"] if not ch["is_leaf"] and ch["split_time"] is not None else y_leaf
            line_x += [nx, cx, cx, None]
            line_y += [ny, ny, cy, None]

    traces.append(go.Scatter(
        x=line_x, y=line_y, mode="lines",
        line=dict(color="#555", width=2),
        hoverinfo="skip", showlegend=False,
    ))

    # Root stem
    root = next(n for n in nodes if n["parent_id"] is None)
    rx   = x_pos[root["id"]]
    root_t = root["split_time"] if root["split_time"] is not None else t_max
    traces.append(go.Scatter(
        x=[rx, rx], y=[t_min - t_pad * 0.8, root_t],
        mode="lines", line=dict(color="#555", width=2),
        hoverinfo="skip", showlegend=False,
    ))
    traces.append(go.Scatter(
        x=[rx], y=[t_min - t_pad * 0.9],
        mode="text", text=["all"],
        textfont=dict(size=10, color="#999"),
        hoverinfo="skip", showlegend=False,
    ))

    # Split dots + time labels
    for nd in nodes:
        if nd["is_leaf"] or nd["split_time"] is None:
            continue
        nx = x_pos[nd["id"]]
        ny = nd["split_time"]
        traces.append(go.Scatter(
            x=[nx], y=[ny], mode="markers+text",
            marker=dict(color="#333", size=8),
            text=[f"{ny:.0f} hpf"],
            textposition="top right",
            textfont=dict(size=9, color="#777"),
            hovertext=f"Split: {ny:.1f} hpf",
            hoverinfo="text",
            showlegend=False,
        ))

    # Leaf boxes: stacked colored bars
    box_h = t_pad * 0.38
    leaves = [n for n in nodes if n["is_leaf"]]
    n_leaves = len(leaves)
    box_w = 0.55

    for nd in leaves:
        nx      = x_pos[nd["id"]]
        members = [c for c in ALL_CLASSES if c in nd["members"]]
        n_m     = len(members)

        # Heterogeneity check
        external = [c for c in selected if c not in nd["members"]]
        het_spread = 0.0
        member_mean_onsets: dict[str, float] = {}
        if n_m > 1 and external:
            per = []
            for m in members:
                if m not in mat.index:
                    continue
                vals = [float(mat.loc[m, e]) for e in external
                        if e in mat.columns and not pd.isna(mat.loc[m, e])]
                mo = float(np.mean(vals)) if vals else float("nan")
                member_mean_onsets[m] = mo
                if not np.isnan(mo):
                    per.append(mo)
            het_spread = (max(per) - min(per)) if len(per) > 1 else 0.0

        # Stem to leaf
        parent_t = by_id[nd["parent_id"]]["split_time"] if nd["parent_id"] is not None else t_min - t_pad * 0.8
        traces.append(go.Scatter(
            x=[nx, nx], y=[parent_t, y_leaf],
            mode="lines", line=dict(color="#555", width=1.5),
            hoverinfo="skip", showlegend=False,
        ))

        for ci, c in enumerate(members):
            color = CLASS_COLORS.get(c, "#888")
            y0 = y_leaf + ci * box_h
            y1 = y0 + box_h
            text_color = "white" if c in ("pbx1b_pbx4_crispant", "pbx1b_crispant", "inj_ctrl") else "#222"

            tip = f"<b>{CLASS_LABELS.get(c, c)}</b>"
            if n_m > 1:
                tip += f"<br>Composite block ({n_m} classes)"
                if c in member_mean_onsets and not np.isnan(member_mean_onsets[c]):
                    tip += f"<br>Mean onset to external: {member_mean_onsets[c]:.1f} hpf"
                if het_spread > 8:
                    tip += f"<br>⚠ Timing spread: {het_spread:.1f} hpf"

            border_color = "#FF8800" if (het_spread > 8) else "white"
            border_w     = 2.5 if (het_spread > 8) else 0.5

            # Filled box
            traces.append(go.Scatter(
                x=[nx - box_w/2, nx + box_w/2, nx + box_w/2, nx - box_w/2, nx - box_w/2],
                y=[y0, y0, y1, y1, y0],
                fill="toself", fillcolor=color,
                line=dict(color=border_color, width=border_w),
                mode="lines",
                hovertext=tip, hoverinfo="text",
                showlegend=False,
            ))
            # Label
            traces.append(go.Scatter(
                x=[nx], y=[(y0 + y1) / 2],
                mode="text",
                text=[CLASS_LABELS.get(c, c)],
                textfont=dict(size=10, color=text_color),
                hoverinfo="skip", showlegend=False,
            ))

    fig = go.Figure(traces)
    fig.update_layout(
        margin=dict(l=55, r=15, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[-0.8, n_leaves - 0.2]),
        yaxis=dict(
            autorange="reversed",
            title="Stage (hpf)",
            showgrid=True, gridcolor="#eeeeee",
            range=[t_min - t_pad, y_leaf + len(selected) * box_h + t_pad * 0.3],
        ),
        showlegend=False,
        height=520,
    )
    return fig


def build_heatmap_figure(selected: list[str]) -> go.Figure:
    n = len(ALL_CLASSES)
    labels = [CLASS_LABELS.get(c, c) for c in ALL_CLASSES]

    # Build z, text, customdata, colors
    z, text_mat, hover_mat, color_mat = [], [], [], []

    for ri, row_c in enumerate(ALL_CLASSES):
        z_row, text_row, hover_row, color_row = [], [], [], []
        for ci, col_c in enumerate(ALL_CLASSES):
            excluded = row_c not in selected or col_c not in selected
            is_diag  = row_c == col_c

            if is_diag:
                z_row.append(0)
                text_row.append("")
                hover_row.append(CLASS_LABELS.get(row_c, row_c))
                color_row.append("rgba(248,248,248,1)")
            elif excluded:
                v = FULL_ONSET_MAT.loc[row_c, col_c] \
                    if (row_c in FULL_ONSET_MAT.index and col_c in FULL_ONSET_MAT.columns) \
                    else None
                z_row.append(-1)
                text_row.append("" if v is None or pd.isna(v) else f"{v:.0f}")
                hover_row.append(
                    f"{CLASS_LABELS.get(row_c,row_c)} vs {CLASS_LABELS.get(col_c,col_c)}<br>(excluded)"
                )
                color_row.append("rgba(220,220,220,1)")
            else:
                v = FULL_ONSET_MAT.loc[row_c, col_c] \
                    if (row_c in FULL_ONSET_MAT.index and col_c in FULL_ONSET_MAT.columns) \
                    else None
                if v is None or pd.isna(v):
                    z_row.append(-1)
                    text_row.append("—")
                    hover_row.append(
                        f"{CLASS_LABELS.get(row_c,row_c)} vs {CLASS_LABELS.get(col_c,col_c)}<br>"
                        "Never durably separated"
                    )
                    color_row.append("rgba(208,208,208,1)")
                else:
                    fv = float(v)
                    z_row.append(fv)
                    text_row.append(f"{fv:.0f}")
                    hover_row.append(
                        f"{CLASS_LABELS.get(row_c,row_c)} vs {CLASS_LABELS.get(col_c,col_c)}<br>"
                        f"Onset: {fv:.0f} hpf"
                    )
                    color_row.append(None)   # use colorscale

        z.append(z_row)
        text_mat.append(text_row)
        hover_mat.append(hover_row)
        color_row_filtered = [c for c in color_row if c is not None]

    # Build cell color array for the heatmap via a custom annotation approach.
    # Use Heatmap for color scale cells + Annotations for text.
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=text_mat,
        texttemplate="%{text}",
        hovertext=hover_mat,
        hoverinfo="text",
        colorscale=[
            [0.0,  "rgb(208,208,208)"],   # -1 = never-sep / excluded (overridden below)
            [0.01, "#ffffb2"],
            [0.5,  "#fd8d3c"],
            [1.0,  "#800026"],
        ],
        zmin=-1,
        zmax=VMAX,
        showscale=True,
        colorbar=dict(
            title="Onset (hpf)",
            thickness=14,
            len=0.6,
            tickvals=[VMIN, (VMIN + VMAX) / 2, VMAX],
            ticktext=[f"{VMIN:.0f}", f"{(VMIN+VMAX)/2:.0f}", f"{VMAX:.0f}"],
        ),
        textfont=dict(size=11),
    ))

    # Gray overlay for excluded / diagonal cells using shapes
    shapes = []
    for ri, row_c in enumerate(ALL_CLASSES):
        for ci, col_c in enumerate(ALL_CLASSES):
            excluded = row_c not in selected or col_c not in selected
            is_diag  = row_c == col_c
            v = FULL_ONSET_MAT.loc[row_c, col_c] \
                if (row_c in FULL_ONSET_MAT.index and col_c in FULL_ONSET_MAT.columns) \
                else None
            never_sep = (not is_diag and not excluded and (v is None or pd.isna(v)))

            if is_diag or excluded or never_sep:
                fill = ("rgba(248,248,248,0.95)" if is_diag else
                        "rgba(220,220,220,0.85)" if excluded else
                        "rgba(200,200,200,0.75)")
                shapes.append(dict(
                    type="rect",
                    xref="x", yref="y",
                    x0=ci - 0.5, x1=ci + 0.5,
                    y0=ri - 0.5, y1=ri + 0.5,
                    fillcolor=fill,
                    line=dict(width=0),
                    layer="above",
                ))

    # Muted axis labels for excluded
    col_label_colors = ["#bbb" if c not in selected else "#333" for c in ALL_CLASSES]
    row_label_colors = ["#bbb" if c not in selected else "#333" for c in ALL_CLASSES]

    fig.update_layout(
        shapes=shapes,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        xaxis=dict(
            tickvals=list(range(n)),
            ticktext=[
                f'<span style="color:{col_label_colors[i]}">{labels[i]}</span>'
                for i in range(n)
            ],
            tickangle=40,
            side="bottom",
        ),
        yaxis=dict(
            tickvals=list(range(n)),
            ticktext=[
                f'<span style="color:{row_label_colors[i]}">{labels[i]}</span>'
                for i in range(n)
            ],
            autorange="reversed",
        ),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Dash app layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Phenotype Emergence Explorer")

app.layout = html.Div([
    html.Div([
        html.H2("Phenotype Emergence Explorer",
                style={"margin": "0 0 4px", "fontSize": 18}),
        html.P("Adaptive composite block tree · onset heatmap · class selection and anchor",
               style={"margin": 0, "fontSize": 12, "color": "#666"}),
    ], style={"padding": "16px 24px 8px", "borderBottom": "1px solid #ddd",
              "background": "white"}),

    html.Div([
        html.Div([
            html.Label("Included genotypes:", style={"fontWeight": "600", "fontSize": 12,
                                                      "marginRight": 6}),
            dcc.Dropdown(
                id="subset-dropdown",
                options=SUBSET_OPTIONS,
                value="all",
                clearable=False,
                style={"width": 220, "fontSize": 12},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": 8}),

        html.Div([
            html.Label("Resolve relative to:", style={"fontWeight": "600", "fontSize": 12,
                                                       "marginRight": 6}),
            dcc.Dropdown(
                id="anchor-dropdown",
                options=ANCHOR_OPTIONS,
                value="none",
                clearable=False,
                style={"width": 280, "fontSize": 12},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": 8}),
    ], style={"display": "flex", "gap": 32, "padding": "12px 24px",
              "background": "white", "borderBottom": "1px solid #eee",
              "flexWrap": "wrap", "alignItems": "center"}),

    html.Div([
        html.Div([
            html.H3("Composite Block Tree", style={"margin": "0 0 8px", "fontSize": 13,
                                                    "color": "#444"}),
            dcc.Graph(id="tree-graph", config={"displayModeBar": False}),
        ], style={"flex": "1.1", "padding": 16, "background": "white",
                  "borderRight": "1px solid #eee"}),

        html.Div([
            html.H3("Onset Heatmap (hpf)", style={"margin": "0 0 8px", "fontSize": 13,
                                                    "color": "#444"}),
            dcc.Graph(id="heatmap-graph", config={"displayModeBar": False}),
        ], style={"flex": "0.9", "padding": 16, "background": "white"}),

    ], style={"display": "flex", "height": "calc(100vh - 130px)"}),

], style={"fontFamily": "Arial, sans-serif", "background": "#fafafa"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("tree-graph",    "figure"),
    Output("heatmap-graph", "figure"),
    Input("subset-dropdown", "value"),
    Input("anchor-dropdown", "value"),
)
def update_figures(subset_key: str, anchor_key: str):
    selected = SUBSET_MAP.get(subset_key, list(ALL_CLASSES))
    anchor_raw = ANCHOR_MAP.get(anchor_key, None)
    anchor = None
    if anchor_raw is not None:
        anchor = anchor_raw & frozenset(selected)
        if not anchor:
            anchor = None

    tree_fig    = build_tree_figure(selected, anchor)
    heatmap_fig = build_heatmap_figure(selected)
    return tree_fig, heatmap_fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting Dash server at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop.")
    app.run(debug=False, host="127.0.0.1", port=8050)
