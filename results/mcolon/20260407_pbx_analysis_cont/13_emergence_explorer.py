"""
13_emergence_explorer.py
------------------------
Interactive phenotype emergence explorer — fully standalone HTML + inline D3.

Controls
--------
  Included genotypes    : checklist — which genotypes are in the analysis
  Emergence reference   : checklist — defines the baseline (reference set)
  AUROC threshold       : radio — none | 0.60 | 0.65 | 0.70

Tree rendering
--------------
  Layer 1 — Emergence from reference: each non-reference class is scored by
    median onset to any reference member.  Classes are grouped into emergence
    blocks by time bin (floor/bin_width * bin_width).  Each block is placed at
    the raw median emergence time (not the floored bin key).

  Layer 2 — Within-block resolution: for multi-member blocks, a recursive
    bipartition finds the best split by cross-median onset.  Accepted only when
    ≥50% of cross-partition pairs are finite.  Unresolved blocks shown with
    dashed border.

All tree computation happens client-side so reference switching is instant.
The heatmap and AUROC switch load pre-computed onset matrices.

Reference coherence badge shown in status bar (valid / ambiguous / invalid).

Output
------
  results/emergence/emergence_explorer.html   (fully standalone)

Run
---
  conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260407_pbx_analysis_cont/13_emergence_explorer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
OUT_DIR  = Path(__file__).parent / "results/emergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# AUROC thresholds to pre-compute. "none" = p-value only (auroc_sep=0).
AUROC_LEVELS = {
    "none": 0.0,
    "0.60": 0.60,
    "0.65": 0.65,
    "0.70": 0.70,
}
P_SEP            = 0.05
P_NS             = 0.10
SUBSEQUENT_FRAC  = 0.40   # fraction of bins from onset onward that must be significant

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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(DATA_DIR / "scores.parquet")
    return scores[scores["feature_set"] == "vae"].copy().reset_index(drop=True)


def compute_onset_matrix_for_auroc(scores: pd.DataFrame, auroc_sep: float) -> pd.DataFrame:
    params = TransitivityParams(
        p_sep=P_SEP, p_ns=P_NS, auroc_sep=auroc_sep, subsequent_frac=SUBSEQUENT_FRAC
    )
    classified = classify_pair_state_over_time(
        scores, params,
        time_col="time_bin_center",
        class_i_col="positive_label",
        class_j_col="negative_label",
        pval_col="pval",
        auroc_col="auroc_obs",
    )
    onset_df = compute_pair_onsets(
        classified, params,
        time_col="time_bin_center",
        class_i_col="positive_label",
        class_j_col="negative_label",
    )
    return build_onset_matrix(onset_df, ALL_CLASSES)


def mat_to_dict(mat: pd.DataFrame) -> dict:
    """Convert onset matrix to nested dict {classA: {classB: float|null}}."""
    result = {}
    for a in ALL_CLASSES:
        result[a] = {}
        for b in ALL_CLASSES:
            v = mat.loc[a, b] if a in mat.index and b in mat.columns else None
            if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
                result[a][b] = None
            else:
                result[a][b] = round(float(v), 1)
    return result


def build_data(scores: pd.DataFrame) -> dict:
    onset_matrices = {}
    for label, auroc_sep in AUROC_LEVELS.items():
        print(f"  Computing onset matrix for auroc_sep={label}...")
        mat = compute_onset_matrix_for_auroc(scores, auroc_sep)
        onset_matrices[label] = mat_to_dict(mat)

    # Global color scale: union of all finite values across all auroc levels
    all_finite = [
        v
        for mdict in onset_matrices.values()
        for adict in mdict.values()
        for v in adict.values()
        if v is not None
    ]
    global_vmin = float(min(all_finite)) if all_finite else 0.0
    global_vmax = float(max(all_finite)) if all_finite else 130.0

    return {
        "onset_matrices": onset_matrices,
        "auroc_levels":   list(AUROC_LEVELS.keys()),
        "all_classes":    ALL_CLASSES,
        "class_labels":   CLASS_LABELS,
        "class_colors":   CLASS_COLORS,
        "vmin":           global_vmin,
        "vmax":           global_vmax,
        "tree_tmin":      global_vmin,
        "tree_tmax":      global_vmax,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phenotype Emergence Explorer</title>
<script>__D3_PLACEHOLDER__</script>
<style>
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { font-family: Arial, sans-serif; background: #f5f5f5; color: #222; }

:root { --sb: 240px; }

#sidebar {
  position: fixed; top: 0; left: 0;
  width: var(--sb); height: 100vh;
  background: white; border-right: 1px solid #ddd;
  padding: 14px 16px; overflow-y: auto; z-index: 10;
}
#sidebar h3 {
  font-size: 11px; font-weight: 700; color: #555;
  margin: 6px 0 6px; text-transform: uppercase; letter-spacing: 0.05em;
}
#sidebar .section { margin-bottom: 16px; border-bottom: 1px solid #f0f0f0; padding-bottom: 14px; }
#sidebar .section:last-child { border-bottom: none; }
.helper-btns { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px; }
.helper-btns button {
  font-size: 10px; padding: 2px 7px; border: 1px solid #ccc;
  border-radius: 3px; background: #fafafa; cursor: pointer; color: #555;
}
.helper-btns button:hover { background: #eee; }

.check-item {
  display: flex; align-items: center; gap: 7px; padding: 3px 0; cursor: pointer;
}
.check-item input[type=checkbox] { cursor: pointer; accent-color: #2166AC; flex-shrink: 0; }
.check-item .swatch { width: 11px; height: 11px; border-radius: 2px; flex-shrink: 0; }
.check-item label { font-size: 12px; cursor: pointer; white-space: nowrap; }
.check-item.disabled { opacity: 0.35; cursor: not-allowed; }
.check-item.disabled input, .check-item.disabled label { cursor: not-allowed; }

.radio-group { display: flex; flex-direction: column; gap: 4px; }
.radio-item { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.radio-item input[type=radio] { cursor: pointer; accent-color: #2166AC; }
.radio-item label { font-size: 12px; cursor: pointer; }

#main {
  margin-left: var(--sb); display: flex; height: 100vh; overflow: auto;
}
#tree-panel {
  flex: 1 1 0; min-width: 320px; padding: 12px 14px 8px;
  background: white; border-right: 1px solid #eee;
  display: flex; flex-direction: column; overflow: auto;
}
#heatmap-panel {
  flex: 0 0 auto; width: 400px; padding: 12px 14px 8px;
  background: white; overflow: auto; display: flex; flex-direction: column;
}
.panel-title {
  font-size: 11px; font-weight: 700; color: #555; flex-shrink: 0;
  margin: 0 0 6px; text-transform: uppercase; letter-spacing: 0.04em;
}
svg#tree {
  flex: 1 1 auto; width: 100%; min-height: 340px; display: block; overflow: visible;
}
svg#heatmap { display: block; }

#tooltip {
  position: fixed; background: rgba(20,20,20,0.88); color: white;
  padding: 6px 10px; border-radius: 5px; font-size: 11px;
  pointer-events: none; display: none; max-width: 260px;
  line-height: 1.55; z-index: 999; white-space: pre-line;
}
#status { font-size: 10px; color: #aaa; margin-top: 4px; flex-shrink: 0; display: flex; align-items: center; gap: 6px; }
.badge {
  display: inline-block; font-size: 9px; font-weight: 700; padding: 1px 5px;
  border-radius: 3px; text-transform: uppercase; letter-spacing: 0.04em;
}
.badge-valid   { background: #d4edda; color: #155724; }
.badge-ambiguous { background: #fff3cd; color: #856404; }
.badge-invalid { background: #f8d7da; color: #721c24; }
#colorbar-wrap { display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-shrink: 0; }
#colorbar-wrap span { font-size: 10px; color: #888; }
#colorbar { display: block; border: 1px solid #ddd; border-radius: 2px; }
</style>
</head>
<body>

<div id="sidebar">
  <div class="section">
    <h3>Included genotypes</h3>
    <div class="helper-btns">
      <button onclick="setAll('inc',true)">All</button>
      <button onclick="setAll('inc',false)">None</button>
    </div>
    <div id="inc-list"></div>
  </div>
  <div class="section">
    <h3>Emergence reference <span style="font-weight:normal;color:#aaa;font-size:10px">(comparison baseline)</span></h3>
    <div class="helper-btns">
      <button onclick="setAll('ref',true)">All</button>
      <button onclick="setAll('ref',false)">None</button>
    </div>
    <div id="ref-list"></div>
  </div>
  <div class="section">
    <h3>AUROC threshold</h3>
    <div class="radio-group" id="auroc-radios"></div>
  </div>
  <div id="status"></div>
</div>

<div id="main">
  <div id="tree-panel">
    <div class="panel-title">Emergence Timeline</div>
    <svg id="tree"></svg>
  </div>
  <div id="heatmap-panel">
    <div class="panel-title">Onset Heatmap (hpf)</div>
    <svg id="heatmap"></svg>
    <div id="colorbar-wrap">
      <span class="cb-lo"></span>
      <canvas id="colorbar" width="120" height="12"></canvas>
      <span class="cb-hi"></span>
    </div>
  </div>
</div>
<div id="tooltip"></div>

<script>
const DATA      = __DATA_PLACEHOLDER__;
const allCls    = DATA.all_classes;
const labels    = DATA.class_labels;
const colors    = DATA.class_colors;
const VMIN      = DATA.vmin;
const VMAX      = DATA.vmax;
const TMIN      = DATA.tree_tmin;
const TMAX      = DATA.tree_tmax;
const BIN_WIDTH = 4.0;
const MIN_CROSS_SUPPORT = 0.5;
const tip       = document.getElementById("tooltip");

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  included:  new Set(allCls),
  reference: new Set(),
  auroc:     DATA.auroc_levels[0],   // first option = "none"
};

// ── Sidebar ───────────────────────────────────────────────────────────────────
function buildSidebar() {
  ["inc", "ref"].forEach(role => {
    const el = document.getElementById(`${role}-list`);
    el.innerHTML = "";
    allCls.forEach(c => {
      const row = document.createElement("div");
      row.className = "check-item";
      row.id = `${role}-row-${c}`;
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.id = `${role}-cb-${c}`;
      cb.checked = role === "inc" ? state.included.has(c) : state.reference.has(c);
      cb.addEventListener("change", () => onCheck(role, c, cb.checked));
      const sw = document.createElement("div");
      sw.className = "swatch"; sw.style.background = colors[c] || "#888";
      const lbl = document.createElement("label");
      lbl.htmlFor = cb.id; lbl.textContent = labels[c] || c;
      row.appendChild(cb); row.appendChild(sw); row.appendChild(lbl);
      el.appendChild(row);
    });
  });

  const rg = document.getElementById("auroc-radios");
  rg.innerHTML = "";
  const auroc_labels = {
    "none": "None (p-value only)",
    "0.60": "≥ 0.60",
    "0.65": "≥ 0.65",
    "0.70": "≥ 0.70",
  };
  DATA.auroc_levels.forEach(lv => {
    const row = document.createElement("div"); row.className = "radio-item";
    const rb  = document.createElement("input");
    rb.type = "radio"; rb.name = "auroc"; rb.id = `auroc-${lv}`;
    rb.value = lv; rb.checked = lv === state.auroc;
    rb.addEventListener("change", () => { state.auroc = lv; update(); });
    const lbl = document.createElement("label"); lbl.htmlFor = `auroc-${lv}`;
    lbl.textContent = auroc_labels[lv] || lv;
    row.appendChild(rb); row.appendChild(lbl); rg.appendChild(row);
  });
}

function onCheck(role, cls, checked) {
  if (role === "inc") {
    if (checked) state.included.add(cls);
    else { state.included.delete(cls); state.reference.delete(cls); }
  } else {
    if (checked && state.included.has(cls)) state.reference.add(cls);
    else state.reference.delete(cls);
  }
  syncSidebar(); update();
}

function syncSidebar() {
  allCls.forEach(c => {
    const incCb  = document.getElementById(`inc-cb-${c}`);
    const refRow = document.getElementById(`ref-row-${c}`);
    const refCb  = document.getElementById(`ref-cb-${c}`);
    if (incCb) incCb.checked = state.included.has(c);
    if (refCb) {
      const inInc = state.included.has(c);
      refCb.checked = state.reference.has(c);
      refCb.disabled = !inInc;
      refRow.classList.toggle("disabled", !inInc);
    }
  });
}

function setAll(role, val) {
  if (role === "inc") {
    allCls.forEach(c => val ? state.included.add(c) : state.included.delete(c));
    if (!val) state.reference.clear();
  } else {
    allCls.forEach(c => { if (state.included.has(c)) val ? state.reference.add(c) : state.reference.delete(c); });
  }
  syncSidebar(); update();
}

// ── Onset matrix helpers ──────────────────────────────────────────────────────
function getOnset(mat, a, b) {
  const v = mat[a] && mat[a][b] != null ? mat[a][b] : null;
  if (v !== null) return v;
  // symmetric fallback
  const v2 = mat[b] && mat[b][a] != null ? mat[b][a] : null;
  return v2;
}

// ── Emergence timeline: JS port of analyze.classification.emergence ───────────

function nanmedian(vals) {
  const f = vals.filter(v => v !== null && isFinite(v));
  if (!f.length) return null;
  f.sort((a, b) => a - b);
  const m = Math.floor(f.length / 2);
  return f.length % 2 ? f[m] : (f[m-1] + f[m]) / 2;
}

// Step 1: validate reference
function validateReference(mat, reference) {
  const pairs = [];
  for (let i = 0; i < reference.length; i++)
    for (let j = i+1; j < reference.length; j++)
      pairs.push([reference[i], reference[j]]);

  const n_total = pairs.length;
  if (n_total === 0) return { status: "valid", coherence_score: 1.0, offending_pairs: [], n_internal_pairs: 0 };

  const offending = [];
  for (const [a, b] of pairs) {
    const v = getOnset(mat, a, b);
    if (v !== null && isFinite(v)) offending.push([a, b, v]);
  }

  const n_nan = n_total - offending.length;
  const coherence = n_nan / n_total;
  let status = offending.length === 0 ? "valid" : (coherence >= 0.5 ? "ambiguous" : "invalid");
  return { status, coherence_score: coherence, offending_pairs: offending, n_internal_pairs: n_total };
}

// Step 2: compute emergence scores
function computeEmergenceScores(mat, nonRef, reference) {
  return nonRef.map(c => {
    const per_ref = {};
    reference.forEach(r => { per_ref[r] = getOnset(mat, c, r); });
    const finite = Object.values(per_ref).filter(v => v !== null && isFinite(v));
    return {
      class_name: c,
      emergence_time: nanmedian(finite),
      emergence_min: finite.length ? Math.min(...finite) : null,
      emergence_max: finite.length ? Math.max(...finite) : null,
      n_resolved_refs: finite.length,
      n_total_refs: reference.length,
      per_ref_onsets: per_ref,
    };
  }).sort((a, b) => {
    const na = a.emergence_time === null, nb = b.emergence_time === null;
    if (na && nb) return 0;
    if (na) return 1; if (nb) return -1;
    return a.emergence_time - b.emergence_time;
  });
}

// Step 3: form emergence blocks
function formEmergenceBlocks(scores, binWidth) {
  binWidth = binWidth || BIN_WIDTH;
  const finite = scores.filter(s => s.emergence_time !== null && isFinite(s.emergence_time));
  const nan_s  = scores.filter(s => s.emergence_time === null || !isFinite(s.emergence_time));

  const binGroups = {};
  finite.forEach(s => {
    const key = Math.floor(s.emergence_time / binWidth) * binWidth;
    if (!binGroups[key]) binGroups[key] = [];
    binGroups[key].push(s);
  });

  const blocks = [];
  let bid = 0;
  Object.keys(binGroups).map(Number).sort((a,b)=>a-b).forEach(bk => {
    const grp = binGroups[bk];
    const rawTimes = grp.map(s => s.emergence_time);
    blocks.push({
      block_id: bid++,
      members: grp.map(s => s.class_name),
      bin_key: bk,
      emergence_time: nanmedian(rawTimes),
      emergence_min: Math.min(...rawTimes),
      emergence_max: Math.max(...rawTimes),
    });
  });

  if (nan_s.length) {
    blocks.push({
      block_id: bid,
      members: nan_s.map(s => s.class_name),
      bin_key: null,
      emergence_time: null,
      emergence_min: null,
      emergence_max: null,
    });
  }
  return blocks;
}

// Step 4: recursive block resolution
function symmetricOnset(mat, a, b) {
  const v = getOnset(mat, a, b);
  return (v !== null && isFinite(v)) ? v : null;
}

function allBipartitions(members) {
  // All non-trivial bipartitions: pin members[0] in b1 to avoid duplicates
  const n = members.length;
  const rest = members.slice(1);
  const result = [];
  const M = 1 << rest.length;
  for (let mask = 0; mask < M; mask++) {
    const b1 = [members[0]], b2 = [];
    rest.forEach((m, i) => { if (mask & (1 << i)) b1.push(m); else b2.push(m); });
    if (b1.length > 0 && b2.length > 0) result.push([b1, b2]);
  }
  return result;
}

function scorePartition(mat, b1, b2) {
  const crossOnsets = [];
  b1.forEach(a => b2.forEach(b => {
    const v = symmetricOnset(mat, a, b);
    crossOnsets.push(v);
  }));
  const nTotal = crossOnsets.length;
  const finite = crossOnsets.filter(v => v !== null);
  if (nTotal === 0) return { accepted: false };
  const crossSupport = finite.length / nTotal;
  const crossMedian = nanmedian(finite);
  if (crossSupport < MIN_CROSS_SUPPORT || crossMedian === null || crossMedian <= 0)
    return { accepted: false, crossMedian, crossSupport };

  // internal finite count (lower = more coherent children)
  let internalFinite = 0;
  for (let i=0; i<b1.length; i++) for (let j=i+1; j<b1.length; j++)
    if (symmetricOnset(mat, b1[i], b1[j]) !== null) internalFinite++;
  for (let i=0; i<b2.length; i++) for (let j=i+1; j<b2.length; j++)
    if (symmetricOnset(mat, b2[i], b2[j]) !== null) internalFinite++;

  return { accepted: true, crossMedian, crossSupport, internalFinite };
}

function findBestSplit(mat, members) {
  const partitions = allBipartitions(members);
  let best = null, bestScore = null;

  partitions.forEach(([b1, b2]) => {
    const s = scorePartition(mat, b1, b2);
    if (!s.accepted) return;
    const score = [s.crossMedian, s.crossSupport, -s.internalFinite];
    if (
      best === null
      || score[0] > bestScore[0]
      || (score[0] === bestScore[0] && score[1] > bestScore[1])
      || (score[0] === bestScore[0] && score[1] === bestScore[1] && score[2] > bestScore[2])
    ) {
      best = { b1, b2, splitTime: s.crossMedian };
      bestScore = score;
    }
  });
  return best;
}

function resolveBlock(mat, members) {
  if (members.length <= 1)
    return { members, split_time: null, children: [], unresolved: false };

  const best = findBestSplit(mat, members);
  if (!best)
    return { members, split_time: null, children: [], unresolved: true };

  return {
    members,
    split_time: best.splitTime,
    children: [resolveBlock(mat, best.b1), resolveBlock(mat, best.b2)],
    unresolved: false,
  };
}

// Step 5: build emergence timeline
function buildEmergenceTimeline(mat, selected, reference) {
  const refSet = new Set(reference);
  const nonRef = selected.filter(c => !refSet.has(c));
  const refValidation = validateReference(mat, reference);
  const scores = computeEmergenceScores(mat, nonRef, reference);
  const blocks = formEmergenceBlocks(scores, BIN_WIDTH);
  const blockResolutions = {};
  blocks.forEach(b => { blockResolutions[b.block_id] = resolveBlock(mat, b.members); });
  return { refValidation, scores, blocks, blockResolutions, reference, nonRef };
}

// ── Color helper ──────────────────────────────────────────────────────────────
function heatColor(fv) {
  const t = Math.max(0, Math.min(1, (fv - VMIN) / Math.max(VMAX - VMIN, 1e-9)));
  let r, g, b;
  if (t < 0.5) {
    const tt = t*2; r=Math.round(255+(253-255)*tt); g=Math.round(255+(141-255)*tt); b=Math.round(178+(60-178)*tt);
  } else {
    const tt=(t-0.5)*2; r=Math.round(253+(128-253)*tt); g=Math.round(141+(0-141)*tt); b=Math.round(60+(38-60)*tt);
  }
  return `rgb(${r},${g},${b})`;
}

function renderColorbar() {
  const W=120, H=12;
  const cv = document.getElementById("colorbar");
  cv.width=W; cv.height=H;
  const ctx = cv.getContext("2d");
  for (let px=0; px<W; px++) { ctx.fillStyle=heatColor(VMIN+(px/W)*(VMAX-VMIN)); ctx.fillRect(px,0,1,H); }
  document.querySelector(".cb-lo").textContent = `${Math.round(VMIN)} hpf`;
  document.querySelector(".cb-hi").textContent = `${Math.round(VMAX)} hpf`;
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
function showTip(text, ev) { tip.textContent=text; tip.style.display="block"; moveTip(ev); }
function moveTip(ev) { tip.style.left=(ev.clientX+14)+"px"; tip.style.top=(ev.clientY+14)+"px"; }
function hideTip() { tip.style.display="none"; }

// ── Tree renderer ─────────────────────────────────────────────────────────────
//
// Top-down dendrogram.  Early hpf at top, late hpf at bottom.
//
// Structure:
//   Root node at top spans all leaves.
//   Main trunk runs downward.  At each emergence block's emergence_time the
//   block peels off as a left child; the trunk continues right.
//   Multi-member blocks attach their ResolutionNode subtree below the branch
//   point (resolution split_times are later than emergence_time).
//   Reference leaves and non-reference leaves all sit at the bottom.
//
// The timeline is converted to a flat node array identical in shape to the
// original buildTree output: { id, members, split_time, parent_id,
// children_ids, is_leaf, x }.  renderTree draws it with the same L-shaped
// edges and leaf boxes.

function timelineToNodes(timeline) {
  const { blocks, blockResolutions, reference } = timeline;
  const nodes = [];
  let nid = 0;
  function mk(members, splitTime, parentId) {
    const n = { id: nid++, members, split_time: splitTime, parent_id: parentId,
                children_ids: [], is_leaf: true, x: 0 };
    nodes.push(n);
    return n.id;
  }

  // Sort finite blocks by emergence_time ascending (earliest first = peeled first)
  const finite = blocks
    .filter(b => b.emergence_time !== null && isFinite(b.emergence_time))
    .sort((a,b) => a.emergence_time - b.emergence_time);
  const nanBlock = blocks.find(b => b.emergence_time === null || !isFinite(b.emergence_time));

  // Collect all leaf classes in tree order:
  //   for each block (earliest first): block members, then reference at the end
  const leafClasses = [];
  finite.forEach(b => b.members.forEach(c => leafClasses.push(c)));
  if (nanBlock) nanBlock.members.forEach(c => leafClasses.push(c));
  reference.forEach(c => { if (!leafClasses.includes(c)) leafClasses.push(c); });

  // Root = all classes
  const allMembers = [...leafClasses];
  const rootId = mk(allMembers, null, null);

  // Build trunk: peel off each emergence block in order.
  // "remainder" is the right child that continues the trunk.
  let trunkId = rootId;

  for (let bi = 0; bi < finite.length; bi++) {
    const block = finite[bi];
    const trunkNode = nodes[trunkId];
    trunkNode.split_time = block.emergence_time;
    trunkNode.is_leaf = false;

    // Left child = this block's subtree
    const leftId = buildBlockSubtree(block, trunkId);

    // Right child = remainder (everything not in this block)
    const remainMembers = trunkNode.members.filter(c => !block.members.includes(c));

    if (remainMembers.length === 0) {
      // This block IS the remainder — no right child needed.
      // Undo: make trunk a pass-through to the block subtree.
      // Actually just attach as single child.
      trunkNode.children_ids = [leftId];
      break;
    }

    const rightId = mk(remainMembers, null, trunkId);
    trunkNode.children_ids = [leftId, rightId];
    trunkId = rightId;
  }

  // Whatever is left on the trunk after all finite blocks:
  // it holds reference members (and possibly NaN-block members).
  // If it has >1 member it's a composite leaf (unresolved). Already is_leaf=true.
  // If NaN block exists and has members, they're already in the trunk remainder.

  // --- helper: build subtree for one emergence block ---
  function buildBlockSubtree(block, parentId) {
    const res = blockResolutions[block.block_id];
    if (!res || block.members.length === 1) {
      // Singleton leaf
      return mk(block.members, null, parentId);
    }
    if (res.unresolved) {
      // Multi-member unresolved composite leaf
      return mk(block.members, null, parentId);
    }
    // Resolved: recursively convert ResolutionNode → flat nodes
    return resNodeToFlat(res, parentId);
  }

  function resNodeToFlat(rn, parentId) {
    if (rn.children.length === 0) {
      return mk(rn.members, null, parentId);
    }
    const nId = mk(rn.members, rn.split_time, parentId);
    nodes[nId].is_leaf = false;
    rn.children.forEach(ch => {
      const cId = resNodeToFlat(ch, nId);
      nodes[nId].children_ids.push(cId);
    });
    return nId;
  }

  // --- Assign x from DFS leaf order ---
  const byId = {};
  nodes.forEach(n => byId[n.id] = n);
  const leaves = [];
  const stack = [rootId];
  while (stack.length) {
    const nd = byId[stack.pop()];
    if (nd.is_leaf) leaves.push(nd.id);
    else nd.children_ids.slice().reverse().forEach(c => stack.push(c));
  }
  leaves.forEach((id,i) => byId[id].x = i);
  function propX(id) {
    const nd = byId[id];
    if (nd.is_leaf) return nd.x;
    nd.x = nd.children_ids.reduce((s,c)=>s+propX(c),0)/nd.children_ids.length;
    return nd.x;
  }
  propX(rootId);

  return { nodes, rootId, leaves };
}

function renderTree(timeline, selected, mat) {
  const svg = d3.select("#tree");
  svg.selectAll("*").remove();
  if (!timeline) return;
  if (selected.length < 2) return;

  const { refValidation, reference } = timeline;
  const refSet = new Set(reference);

  const { nodes, rootId, leaves: leafIds } = timelineToNodes(timeline);
  if (!nodes.length) return;

  const byId = {};
  nodes.forEach(n => byId[n.id] = n);

  const W=500, H=420;
  const M={top:24, right:20, bottom:90, left:54};
  const iW=W-M.left-M.right, iH=H-M.top-M.bottom;

  svg.attr("viewBox",`0 0 ${W} ${H}`)
     .attr("preserveAspectRatio","xMidYMid meet")
     .style("width","100%").style("height","100%");

  const g = svg.append("g").attr("transform",`translate(${M.left},${M.top})`);

  const nLeaves = leafIds.length || 1;
  const xMin = Math.min(...leafIds.map(id=>byId[id].x));
  const xMax = Math.max(...leafIds.map(id=>byId[id].x), xMin+0.001);
  const xScale = d => (d-xMin)/(xMax-xMin)*iW;

  // Y scale: early hpf at TOP (y=0), late hpf at BOTTOM (y=iH).
  // Leaf boxes sit below iH.
  const tPad = Math.max((TMAX-TMIN)*0.15, 8);
  const yLo   = TMIN - tPad;
  const yHi   = TMAX + tPad*0.5;
  const yScale = t => (t-yLo)/(yHi-yLo)*iH;
  const yLeaf  = TMAX + tPad*0.20;

  // Grid lines + y-axis labels
  d3.ticks(TMIN, TMAX, 6).forEach(t => {
    g.append("line").attr("x1",0).attr("x2",iW)
      .attr("y1",yScale(t)).attr("y2",yScale(t))
      .attr("stroke","#eee").attr("stroke-width",1);
    g.append("text").attr("x",-6).attr("y",yScale(t)).attr("dy","0.35em")
      .attr("text-anchor","end").attr("font-size",9).attr("fill","#999")
      .text(Math.round(t));
  });
  g.append("text")
    .attr("transform",`translate(-40,${iH/2}) rotate(-90)`)
    .attr("text-anchor","middle").attr("font-size",10).attr("fill","#777")
    .text("Stage (hpf)");

  // Root stem
  const root  = byId[rootId];
  const rootX = xScale(root.x);
  const rootST = root.split_time;
  const rootStemTop = yScale(yLo + tPad*0.05);
  const rootStemBot = rootST != null ? yScale(rootST) : yScale(TMIN - tPad*0.3);
  g.append("line")
    .attr("x1",rootX).attr("x2",rootX)
    .attr("y1",rootStemTop).attr("y2",rootStemBot)
    .attr("stroke","#555").attr("stroke-width",1.8);
  g.append("text")
    .attr("x",rootX).attr("y",rootStemTop - 4)
    .attr("text-anchor","middle").attr("font-size",8).attr("fill","#bbb").text("all");

  // Internal edges (L-shaped: parent horizontal bar → vertical drop to child)
  nodes.filter(n => !n.is_leaf && n.split_time != null).forEach(nd => {
    const nx = xScale(nd.x);
    const ny = yScale(nd.split_time);
    nd.children_ids.forEach(cid => {
      const ch = byId[cid];
      const cx = xScale(ch.x);
      const cy = ch.is_leaf ? yScale(yLeaf)
                : (ch.split_time != null ? yScale(ch.split_time) : yScale(yLeaf));
      g.append("path")
        .attr("d",`M${nx},${ny} H${cx} V${cy}`)
        .attr("fill","none").attr("stroke","#555").attr("stroke-width",1.8);
    });

    // Split dot + time label
    g.append("circle").attr("cx",nx).attr("cy",ny).attr("r",3.5).attr("fill","#333");
    g.append("text").attr("x",nx+4).attr("y",ny-4)
      .attr("font-size",8).attr("fill","#888")
      .text(`${Math.round(nd.split_time)}`);
  });

  // Leaf boxes
  const boxH = 17;
  const boxW = Math.min(72, Math.max(36, iW/Math.max(nLeaves,1)*0.72));
  const leaves = nodes.filter(n => n.is_leaf);

  leaves.forEach(nd => {
    const nx = xScale(nd.x);
    const boxTopY = yScale(yLeaf);

    // Stem from parent
    const parentNd = nd.parent_id != null ? byId[nd.parent_id] : null;
    const parentY = parentNd && parentNd.split_time != null
                  ? yScale(parentNd.split_time) : rootStemTop;

    if (nd.members.length === 1) {
      // Single-class leaf
      const c = nd.members[0];
      const col = colors[c] || "#888";
      const isRef = refSet.has(c);
      const light = ["pbx4_crispant","wik_ab","inj_ctrl"].includes(c);
      g.append("rect")
        .attr("x",nx-boxW/2).attr("y",boxTopY)
        .attr("width",boxW).attr("height",boxH)
        .attr("fill",col)
        .attr("stroke", isRef ? "#333" : "white")
        .attr("stroke-width", isRef ? 1.5 : 0.6)
        .attr("rx",2)
        .on("mouseover", ev => showTip((isRef?"[Reference] ":"") + (labels[c]||c), ev))
        .on("mousemove", moveTip).on("mouseout", hideTip);
      g.append("text")
        .attr("x",nx).attr("y",boxTopY+boxH/2).attr("dy","0.35em")
        .attr("text-anchor","middle")
        .attr("font-size",Math.min(9, boxW*0.135))
        .attr("fill", light ? "#333" : "white").attr("font-weight","600")
        .text(labels[c]||c);
    } else {
      // Multi-member composite leaf (unresolved block or reference composite)
      const allRef = nd.members.every(c => refSet.has(c));
      const memberLabels = nd.members.map(c => labels[c]||c).join(", ");
      const bw = boxW * Math.min(nd.members.length, 2.5);
      g.append("rect")
        .attr("x",nx-bw/2).attr("y",boxTopY)
        .attr("width",bw).attr("height",boxH)
        .attr("fill", allRef ? "#e8eef6" : "#f0f0f0")
        .attr("stroke", allRef ? "#4477bb" : "#999")
        .attr("stroke-width",1.2)
        .attr("stroke-dasharray","4,2")
        .attr("rx",2)
        .on("mouseover", ev => showTip((allRef ? "[Reference] " : "[Unresolved] ") + memberLabels, ev))
        .on("mousemove", moveTip).on("mouseout", hideTip);
      g.append("text")
        .attr("x",nx).attr("y",boxTopY+boxH/2).attr("dy","0.35em")
        .attr("text-anchor","middle").attr("font-size",8)
        .attr("fill", allRef ? "#4477bb" : "#777")
        .text(allRef ? "ref" : memberLabels);
    }
  });

  // Reference coherence annotation
  if (reference.length > 0 && refValidation.status !== "valid") {
    const refLeaf = leaves.find(nd => nd.members.length > 1 && nd.members.every(c => refSet.has(c)));
    if (refLeaf) {
      const rx = xScale(refLeaf.x);
      const ry = yScale(yLeaf) + boxH + 10;
      g.append("text").attr("x",rx).attr("y",ry)
        .attr("text-anchor","middle").attr("font-size",7.5)
        .attr("fill", refValidation.status === "ambiguous" ? "#856404" : "#721c24")
        .text(`ref coherence: ${(refValidation.coherence_score*100).toFixed(0)}%`);
    }
  }
}

// ── Heatmap renderer ──────────────────────────────────────────────────────────
function renderHeatmap(mat, selected) {
  const n=allCls.length;
  const selSet=new Set(selected);
  const panelW=document.getElementById("heatmap-panel").clientWidth-28;
  const cellSz=Math.min(60, Math.max(36, Math.floor((panelW-104)/n)));
  const M={top:86, right:14, bottom:8, left:86};
  const sz=cellSz*n;
  const svg=d3.select("#heatmap").attr("width",sz+M.left+M.right).attr("height",sz+M.top+M.bottom);
  svg.selectAll("*").remove();
  const g=svg.append("g").attr("transform",`translate(${M.left},${M.top})`);

  allCls.forEach((c,i) => {
    const excl=!selSet.has(c), lbl=labels[c]||c;
    const fclr=excl?"#bbb":"#333", fw=excl?"normal":"600";
    g.append("text").attr("x",0).attr("y",0)
      .attr("transform",`translate(${i*cellSz+cellSz/2},-6) rotate(-45)`)
      .attr("text-anchor","start").attr("font-size",10).attr("fill",fclr).attr("font-weight",fw).text(lbl);
    g.append("text").attr("x",-8).attr("y",i*cellSz+cellSz/2).attr("dy","0.35em")
      .attr("text-anchor","end").attr("font-size",10).attr("fill",fclr).attr("font-weight",fw).text(lbl);
  });

  allCls.forEach((rc,ri) => allCls.forEach((cc,ci) => {
    const excl=!selSet.has(rc)||!selSet.has(cc), isDiag=rc===cc;
    const v=getOnset(mat,rc,cc);
    const x=ci*cellSz, y=ri*cellSz;
    let fill, text, tclr, tipText;
    if (isDiag) {
      fill="#f8f8f8"; text=""; tipText=labels[rc]||rc;
    } else if (excl) {
      fill="#e0e0e0"; text=v!=null?`${Math.round(v)}`:""; tclr="#aaa";
      tipText=`${labels[rc]||rc} vs ${labels[cc]||cc} (excluded)`;
    } else if (v===null) {
      fill="#d0d0d0"; text="—"; tclr="#888";
      tipText=`${labels[rc]||rc} vs ${labels[cc]||cc}\nNever durably separated`;
    } else {
      fill=heatColor(v); text=`${Math.round(v)}`; tclr="#222";
      tipText=`${labels[rc]||rc} vs ${labels[cc]||cc}\nOnset: ${Math.round(v)} hpf`;
    }
    g.append("rect").attr("x",x).attr("y",y).attr("width",cellSz-1).attr("height",cellSz-1)
      .attr("fill",fill).attr("rx",2)
      .on("mouseover",ev=>showTip(tipText,ev)).on("mousemove",moveTip).on("mouseout",hideTip);
    if (text) {
      g.append("text").attr("x",x+cellSz/2).attr("y",y+cellSz/2).attr("dy","0.35em")
        .attr("text-anchor","middle").attr("font-size",Math.min(10,cellSz*0.24))
        .attr("fill",tclr||"#222").attr("font-weight",excl?"normal":"500").text(text);
    }
  }));

  for (let i=0; i<=n; i++) {
    g.append("line").attr("x1",i*cellSz).attr("x2",i*cellSz).attr("y1",0).attr("y2",sz).attr("stroke","white").attr("stroke-width",1.2);
    g.append("line").attr("x1",0).attr("x2",sz).attr("y1",i*cellSz).attr("y2",i*cellSz).attr("stroke","white").attr("stroke-width",1.2);
  }
}

// ── Main update ───────────────────────────────────────────────────────────────
function update() {
  const selected  = allCls.filter(c => state.included.has(c));
  const reference = [...state.reference].filter(c => state.included.has(c));
  const mat       = DATA.onset_matrices[state.auroc];

  const statusEl = document.getElementById("status");
  if (selected.length < 2) {
    statusEl.innerHTML = "Need &ge; 2 included genotypes.";
    d3.select("#tree").selectAll("*").remove();
    d3.select("#heatmap").selectAll("*").remove();
    return;
  }

  let timeline = null;
  let badgeHTML = "";

  if (reference.length === 0) {
    statusEl.innerHTML = `${selected.length} included &middot; No reference set &middot; AUROC ${state.auroc}`;
    d3.select("#tree").selectAll("*").remove();
    renderHeatmap(mat, selected);
    return;
  }

  // Build emergence timeline
  const subMat = {};
  selected.forEach(a => { subMat[a]={}; selected.forEach(b => { subMat[a][b]=getOnset(mat,a,b); }); });
  timeline = buildEmergenceTimeline(subMat, selected, reference);

  const rv = timeline.refValidation;
  const badgeClass = rv.status === "valid" ? "badge-valid" : (rv.status === "ambiguous" ? "badge-ambiguous" : "badge-invalid");
  badgeHTML = `<span class="badge ${badgeClass}">${rv.status}</span>`;
  const refLabel = `Ref: ${reference.map(c=>labels[c]||c).join(", ")}`;
  statusEl.innerHTML = `${selected.length} included &middot; ${refLabel} &middot; AUROC ${state.auroc} ${badgeHTML}`;

  renderTree(timeline, selected, subMat);
  renderHeatmap(mat, selected);
}

// ── Init ──────────────────────────────────────────────────────────────────────
buildSidebar();
syncSidebar();
renderColorbar();
update();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading scores...")
    scores = load_scores()

    print("Computing onset matrices for all AUROC levels...")
    data = build_data(scores)

    # Inline D3
    d3_path = OUT_DIR / "d3.min.js"
    if not d3_path.exists():
        print("Downloading D3...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js",
            str(d3_path),
        )
    d3_text = d3_path.read_text(encoding="utf-8")

    data_json = json.dumps(data, default=str, allow_nan=False)
    html = HTML_TEMPLATE.replace("__D3_PLACEHOLDER__", d3_text)
    html = html.replace("__DATA_PLACEHOLDER__", data_json)

    out_path = OUT_DIR / "emergence_explorer.html"
    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"Saved: {out_path}  ({size_kb} KB)")
    print("Download and open locally in any browser.")


if __name__ == "__main__":
    main()
