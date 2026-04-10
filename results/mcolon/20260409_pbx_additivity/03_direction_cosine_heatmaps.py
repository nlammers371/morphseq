"""
Pairwise cosine similarity heatmaps of direction vectors across comparisons.

For each selected time bin, compute the full 5x5 cosine similarity matrix:
  - inj_ctrl vs wik_ab
  - pbx1b_crispant vs wik_ab
  - pbx4_crispant vs wik_ab
  - pbx1b_pbx4_crispant vs wik_ab
  - cep290_homozygous vs cep290_wildtype

Renders two figures:
  1. Grid of heatmaps at selected time bins (spot-check).
  2. Signed cosine over time as a line plot for every pair (full time series).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

SCRIPT_DIR       = Path(__file__).resolve().parent
REPO_ROOT        = SCRIPT_DIR.parents[2]
PBX_ANALYSIS_DIR = REPO_ROOT / "results" / "mcolon" / "20260407_pbx_analysis_cont"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PBX_ANALYSIS_DIR))

from phenotype_direction import load_classifier_directions

PBX_RESULTS_DIR = SCRIPT_DIR / "results" / "pbx_direction_smoke_5class_bin4_perm0_wt_ref"
CEP_RESULTS_DIR = SCRIPT_DIR / "results" / "cep290_direction_for_pbx_projection"
FIGURES_DIR     = SCRIPT_DIR / "figures" / "pbx_direction_smoke_5class_bin4_perm0_wt_ref"

FEATURE_SET = "vae"

# All comparisons to include, in display order
COMPARISONS = [
    ("pbx", "inj_ctrl__vs__wik_ab"),
    ("pbx", "pbx1b_crispant__vs__wik_ab"),
    ("pbx", "pbx4_crispant__vs__wik_ab"),
    ("pbx", "pbx1b_pbx4_crispant__vs__wik_ab"),
    ("cep", "cep290_homozygous__vs__cep290_wildtype"),
]
SHORT_LABELS = [
    "inj_ctrl",
    "pbx1b",
    "pbx4",
    "pbx1b+4",
    "cep290",
]

# Time bins to show as individual heatmap panels
SELECTED_BINS = [26.0, 38.0, 54.0, 70.0, 90.0, 110.0]


# ── load vectors ──────────────────────────────────────────────────────────────

def _load_all_vectors() -> dict[tuple[str, float], np.ndarray]:
    """
    Returns {(comparison_id, time_bin_center): unit_vector}.
    Only signed_unit_coef / raw_feature_space vectors.
    """
    pbx = load_classifier_directions(PBX_RESULTS_DIR)
    cep = load_classifier_directions(CEP_RESULTS_DIR)

    out: dict[tuple[str, float], np.ndarray] = {}
    for bundle, source in [(pbx, "pbx"), (cep, "cep")]:
        meta = bundle.metadata
        rows = meta[
            (meta["feature_set"] == FEATURE_SET)
            & (meta["direction_space"] == "raw_feature_space")
            & (meta["vector_kind"] == "signed_unit_coef")
        ]
        for row in rows.itertuples(index=False):
            vec = np.asarray(bundle.vectors[row.vector_id], dtype=float).ravel()
            out[(row.comparison_id, float(row.time_bin_center))] = vec
    return out


# ── build cosine matrix per bin ───────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _cosine_matrix_at_bin(
    vectors: dict[tuple[str, float], np.ndarray],
    tbc: float,
) -> np.ndarray:
    """5x5 signed cosine matrix. NaN where a comparison has no vector at that bin."""
    n = len(COMPARISONS)
    mat = np.full((n, n), float("nan"))
    vecs = [vectors.get((cid, tbc)) for _, cid in COMPARISONS]
    for i in range(n):
        for j in range(n):
            if vecs[i] is not None and vecs[j] is not None:
                mat[i, j] = _cosine(vecs[i], vecs[j])
    return mat


# ── figure 1: heatmap grid ────────────────────────────────────────────────────

def _plot_heatmap_grid(
    vectors: dict[tuple[str, float], np.ndarray],
    output_path: Path,
) -> None:
    n_panels = len(SELECTED_BINS)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.8))
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    n = len(COMPARISONS)

    for ax, tbc in zip(axes, SELECTED_BINS):
        mat = _cosine_matrix_at_bin(vectors, tbc)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(SHORT_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(SHORT_LABELS, fontsize=8)
        ax.set_title(f"{tbc:.0f} hpf", fontsize=9)
        # Annotate cells
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if np.isfinite(v):
                    txt_color = "white" if abs(v) > 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color=txt_color)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                 ax=axes.tolist(), shrink=0.7, label="Signed cosine similarity")
    fig.suptitle(
        "Pairwise direction cosine similarity at selected time bins\n"
        "(diagonal = 1 by construction; off-diagonal shows axis alignment)",
        fontsize=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── figure 2: cosine over time, all pairs ────────────────────────────────────

def _plot_cosine_over_time(
    vectors: dict[tuple[str, float], np.ndarray],
    output_path: Path,
) -> None:
    # Gather all time bins that exist for at least one comparison
    all_bins = sorted({tbc for (_, tbc) in vectors.keys()})

    # Choose pairs to highlight — skip diagonal and symmetric duplicates
    # Focus on the biologically meaningful pairs
    pairs = [
        (1, 2, "pbx1b vs pbx4"),
        (1, 3, "pbx1b vs pbx1b+4"),
        (2, 3, "pbx4 vs pbx1b+4"),
        (0, 2, "inj_ctrl vs pbx4"),
        (2, 4, "pbx4 vs cep290"),
        (3, 4, "pbx1b+4 vs cep290"),
    ]
    colors = ["#9467BD", "#B2182B", "#F7B267", "#2166AC", "#888888", "#444444"]
    linestyles = ["-", "-", "-", "--", ":", ":"]

    fig, ax = plt.subplots(figsize=(11, 5))
    for (i, j, label), color, ls in zip(pairs, colors, linestyles):
        cid_i = COMPARISONS[i][1]
        cid_j = COMPARISONS[j][1]
        xs, ys = [], []
        for tbc in all_bins:
            vi = vectors.get((cid_i, tbc))
            vj = vectors.get((cid_j, tbc))
            if vi is not None and vj is not None:
                xs.append(tbc)
                ys.append(_cosine(vi, vj))
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8,
                    color=color, linestyle=ls, label=label)

    ax.axhline(0.0, color="#aaaaaa", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="#dddddd", linestyle=":", linewidth=0.8)
    ax.axhline(-1.0, color="#dddddd", linestyle=":", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time bin center (hpf)")
    ax.set_ylabel("Signed cosine similarity")
    ax.set_title("Direction vector cosine similarity over time")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    vectors = _load_all_vectors()
    print(f"Loaded {len(vectors)} vectors across {len(set(cid for cid,_ in vectors))} comparisons.")

    _plot_heatmap_grid(
        vectors,
        FIGURES_DIR / "direction_cosine_heatmap_grid.png",
    )
    _plot_cosine_over_time(
        vectors,
        FIGURES_DIR / "direction_cosine_over_time.png",
    )


if __name__ == "__main__":
    main()
