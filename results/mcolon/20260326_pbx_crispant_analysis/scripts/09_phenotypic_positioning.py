"""
09_phenotypic_positioning.py
----------------------------
Embed embryos in pairwise classification probability space and visualize
genotype structure, within-genotype heterogeneity, and temporal trajectory
divergence using AlignedUMAP.

For each embryo at each timebin, collect predicted probabilities from all 10
pairwise binary classifiers (all pairs of 5 genotypes) into a 10D vector.
Use AlignedUMAP to embed these vectors in 2D/3D while enforcing temporal
coherence across timebins.

Outputs:
  - all_pairs_predictions.csv      raw per-embryo-per-timebin predictions
  - probability_vectors.csv        10D probability vectors per embryo-timebin
  - aligned_umap_2d_coordinates.csv
  - aligned_umap_3d_coordinates.csv
  - auroc_ribbon.png
  - auroc_heatmap_snapshots.png
  - snapshot_umap_2d_{time}hpf.png (for each snapshot time)
  - joint_3d_umap_genotype.html/.png
  - joint_3d_umap_time.html/.png
  - condition_graph_over_time.png
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import umap


EXPERIMENT_IDS = ["20260304", "20260306"]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification import run_classification
from analyze.viz.plotting.plotting_3d import plot_3d_scatter
from analyze.viz.styling.genotype_colors import (
    SPECIAL_GENOTYPE_COLORS,
    get_known_genotype_color,
)

RESULTS_BASE = REPO_ROOT / "results" / "mcolon" / "20260326_pbx_crispant_analysis" / "results" / "misclassification" / "embedding"
FIGURES_BASE = REPO_ROOT / "results" / "mcolon" / "20260326_pbx_crispant_analysis" / "figures" / "misclassification" / "embedding"
BUILD06_DIR = REPO_ROOT / "morphseq_playground" / "metadata" / "build06_output"

DEFAULT_GENOTYPES = [
    "inj_ctrl",
    "wik_ab",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]

# ── helpers ────────────────────────────────────────────────────────────────────

def _normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower().replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    if g in {
        "ab_inj_ctrl", "wik-ab_inj_ctrl", "wik-ab_ctrl_inj",
        "wik_ab_inj_ctrl", "wik_ab_ctrl_inj",
    }:
        return "inj_ctrl"
    return g.replace("wik-ab", "wik_ab")


def _short_name(label: str) -> str:
    label = str(label).strip().lower()
    mapping = {
        "inj_ctrl": "inj_ctrl",
        "wik_ab": "wik_ab",
        "pbx1b_crispant": "pbx1b",
        "pbx4_crispant": "pbx4",
        "pbx1b_pbx4_crispant": "pbx1b+4",
    }
    return mapping.get(label, label.replace("_", " "))


def _pair_id(group1: str, group2: str) -> str:
    return f"{group1}_vs_{group2}"


def _genotype_color(genotype: str, fallback_palette: dict[str, str]) -> str:
    key = str(genotype).strip().lower().replace(" ", "_")
    # strip _crispant suffix for special color lookup
    stripped = key.replace("_crispant", "")
    c = SPECIAL_GENOTYPE_COLORS.get(stripped) or SPECIAL_GENOTYPE_COLORS.get(key)
    if c:
        return c
    c = get_known_genotype_color(genotype)
    if c:
        return c
    return fallback_palette.get(genotype, "#888888")


def _build_color_palette(genotypes: list[str]) -> dict[str, str]:
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette: dict[str, str] = {}
    fb_idx = 0
    for g in genotypes:
        palette[g] = _genotype_color(g, {})
        if palette[g] == "#888888":
            palette[g] = fallback[fb_idx % len(fallback)]
            fb_idx += 1
    # ensure distinct (pbx4 and pbx1b_pbx4 both map to red by default — shift one)
    seen: dict[str, str] = {}
    extra = ["#e377c2", "#8c564b", "#17becf", "#bcbd22"]
    extra_idx = 0
    for g, c in palette.items():
        if c in seen.values():
            palette[g] = extra[extra_idx % len(extra)]
            extra_idx += 1
        seen[g] = palette[g]
    return palette

# ── data loading ───────────────────────────────────────────────────────────────

def load_dataframe(genotypes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for exp_id in EXPERIMENT_IDS:
        data_path = BUILD06_DIR / f"df03_final_output_with_latents_{exp_id}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file: {data_path}")
        part = pd.read_csv(data_path, low_memory=False)
        if "experiment_id" in part.columns:
            part = part[part["experiment_id"].astype(str) == exp_id].copy()
        else:
            part["experiment_id"] = exp_id
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()
    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df = df[df["genotype"].isin(genotypes)].copy()
    if df.empty:
        raise ValueError("No rows remain after genotype filtering.")

    embryo_meta = (
        df[["embryo_id", "genotype", "experiment_id"]]
        .drop_duplicates()
        .rename(columns={"genotype": "true_label"})
        .reset_index(drop=True)
    )
    return df, embryo_meta


def embedding_features(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = sorted([c for c in df.columns if c.startswith(prefix)])
    if not cols:
        raise ValueError(f"No embedding columns found with prefix {prefix!r}")
    return cols

# ── classification ─────────────────────────────────────────────────────────────

def run_pairwise_classification(
    df: pd.DataFrame,
    *,
    group1: str,
    group2: str,
    time_col: str,
    feature_cols: list[str],
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = run_classification(
        df=df.copy(),
        class_col="genotype",
        id_col="embryo_id",
        time_col=time_col,
        positive=group2,
        negative=group1,
        features={"embedding": feature_cols},
        bin_width=float(bin_width),
        n_permutations=int(n_permutations),
        n_splits=int(n_splits),
        min_samples_per_group=2,
        min_samples_per_member=2,
        n_jobs=1,
        random_state=int(random_state),
        verbose=True,
        save_predictions=True,
    )

    score_df = analysis.scores.copy().sort_values("time_bin_center").reset_index(drop=True)
    pred_df = analysis.layers["predictions"].copy().sort_values(["embryo_id", "time_bin_center"]).reset_index(drop=True)

    score_df["pair_id"] = _pair_id(group1, group2)
    score_df["group1"] = group1
    score_df["group2"] = group2
    score_df["n_samples"] = score_df["n_positive"] + score_df["n_negative"]
    score_df["n_group1"] = score_df["n_negative"]
    score_df["n_group2"] = score_df["n_positive"]

    pred_df["true_label"] = np.where(pred_df["y_true"].astype(int) == 1, group2, group1)
    pred_df["pred_prob_group2"] = pred_df["p_pos"].astype(float)
    pred_df["pair_id"] = _pair_id(group1, group2)
    pred_df["group1"] = group1
    pred_df["group2"] = group2
    return score_df, pred_df


def run_or_load_all_pairs(
    df: pd.DataFrame,
    genotypes: list[str],
    *,
    time_col: str,
    feature_cols: list[str],
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    random_state: int,
    results_dir: Path,
    skip_classification: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_cache = results_dir / "all_pairs_predictions.csv"
    auc_cache = results_dir / "auc_bins.csv"

    if skip_classification and pred_cache.exists() and auc_cache.exists():
        print(f"Loading cached predictions from {pred_cache}")
        return pd.read_csv(auc_cache), pd.read_csv(pred_cache)

    auc_rows: list[pd.DataFrame] = []
    pred_rows: list[pd.DataFrame] = []
    present = [g for g in genotypes if g in set(df["genotype"].unique())]

    for idx, (group1, group2) in enumerate(combinations(present, 2), start=1):
        print(f"[{idx}] {group1} vs {group2}")
        df_pair = df[df["genotype"].isin([group1, group2])].copy()
        score_df, pred_df = run_pairwise_classification(
            df_pair,
            group1=group1, group2=group2,
            time_col=time_col, feature_cols=feature_cols,
            bin_width=bin_width, n_splits=n_splits,
            n_permutations=n_permutations, random_state=random_state,
        )
        auc_rows.append(score_df)
        pred_rows.append(pred_df)

    auc_df = pd.concat(auc_rows, ignore_index=True)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    auc_df.to_csv(auc_cache, index=False)
    pred_df.to_csv(pred_cache, index=False)
    print(f"Saved predictions: {pred_cache}")
    return auc_df, pred_df

# ── probability vectors ────────────────────────────────────────────────────────

def build_probability_vectors(
    pred_df: pd.DataFrame,
    embryo_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Pivot per-embryo-per-timebin predictions into 10D probability vectors."""
    pivot = pred_df.pivot_table(
        index=["embryo_id", "time_bin_center"],
        columns="pair_id",
        values="pred_prob_group2",
        aggfunc="mean",
    )
    pair_cols = list(pivot.columns)
    pivot = pivot.fillna(0.5).reset_index()
    pivot = pivot.merge(
        embryo_meta.rename(columns={"true_label": "genotype"})[["embryo_id", "genotype"]],
        on="embryo_id", how="left",
    )
    return pivot, pair_cols

# ── AUROC heatmaps ─────────────────────────────────────────────────────────────

def _make_auroc_matrix(auc_df: pd.DataFrame, genotypes: list[str], time_bin: float) -> np.ndarray:
    n = len(genotypes)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)
    sub = auc_df[np.abs(auc_df["time_bin_center"] - time_bin) < 1.1]
    for i, g1 in enumerate(genotypes):
        for j, g2 in enumerate(genotypes):
            if i == j:
                continue
            row = sub[
                ((sub["group1"] == g1) & (sub["group2"] == g2)) |
                ((sub["group1"] == g2) & (sub["group2"] == g1))
            ]
            if not row.empty:
                matrix[i, j] = float(row["auroc_obs"].mean())
    return matrix


def plot_auroc_ribbon(auc_df: pd.DataFrame, pair_ids: list[str], figures_dir: Path) -> None:
    bins = sorted(auc_df["time_bin_center"].unique())
    ribbon = np.full((len(pair_ids), len(bins)), np.nan)
    for j, t in enumerate(bins):
        sub = auc_df[np.abs(auc_df["time_bin_center"] - t) < 1.1]
        for i, pid in enumerate(pair_ids):
            row = sub[sub["pair_id"] == pid]
            if not row.empty:
                ribbon[i, j] = float(row["auroc_obs"].mean())

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(ribbon, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(pair_ids)))
    short_labels = [f"{_short_name(p.split('_vs_')[0])} vs {_short_name(p.split('_vs_')[1])}" for p in pair_ids]
    ax.set_yticklabels(short_labels, fontsize=8)
    step = max(1, len(bins) // 12)
    ax.set_xticks(range(0, len(bins), step))
    ax.set_xticklabels([f"{bins[k]:.0f}" for k in range(0, len(bins), step)], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Time (hpf)")
    ax.set_title("Pairwise AUROC over developmental time", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="AUROC").ax.tick_params(labelsize=8)
    fig.tight_layout()
    out = figures_dir / "auroc_ribbon.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_auroc_heatmap_snapshots(
    auc_df: pd.DataFrame, genotypes: list[str],
    snapshot_bins: list[float], figures_dir: Path,
) -> None:
    n_snaps = len(snapshot_bins)
    n_cols = min(3, n_snaps)
    n_rows = int(np.ceil(n_snaps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    short = [_short_name(g) for g in genotypes]

    for ax_i, t in enumerate(snapshot_bins):
        ax = axes[ax_i]
        mat = _make_auroc_matrix(auc_df, genotypes, t)
        im = ax.imshow(mat, cmap="viridis", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(genotypes)))
        ax.set_xticklabels(short, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(genotypes)))
        ax.set_yticklabels(short, fontsize=8)
        ax.set_title(f"{t:.0f} hpf", fontsize=10)
        for i in range(len(genotypes)):
            for j in range(len(genotypes)):
                if not np.isnan(mat[i, j]):
                    color = "white" if mat[i, j] > 0.72 else "black"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax_i in range(n_snaps, len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Pairwise AUROC: N×N snapshots", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = figures_dir / "auroc_heatmap_snapshots.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── AlignedUMAP helpers ────────────────────────────────────────────────────────

def _build_aligned_umap_inputs(
    prob_vectors: pd.DataFrame,
    pair_cols: list[str],
) -> tuple[list[np.ndarray], list[dict[int, int]], list[pd.DataFrame]]:
    """
    Build slices and relations for AlignedUMAP.

    Returns
    -------
    slices : list of (n_i, 10) arrays, one per timebin
    relations : list of dicts mapping row_idx_in_slice_t -> row_idx_in_slice_t+1
    slice_meta : list of DataFrames with embryo_id, genotype, time_bin_center per slice
    """
    bins = sorted(prob_vectors["time_bin_center"].unique())
    slices: list[np.ndarray] = []
    slice_meta: list[pd.DataFrame] = []

    for t in bins:
        sub = prob_vectors[prob_vectors["time_bin_center"] == t].reset_index(drop=True)
        X = sub[pair_cols].values.astype(float)
        slices.append(X)
        slice_meta.append(sub[["embryo_id", "genotype", "time_bin_center"]].reset_index(drop=True))

    relations: list[dict[int, int]] = []
    for t_idx in range(len(bins) - 1):
        meta_curr = slice_meta[t_idx]
        meta_next = slice_meta[t_idx + 1]
        idx_map_next = {eid: i for i, eid in enumerate(meta_next["embryo_id"])}
        rel: dict[int, int] = {}
        for i, eid in enumerate(meta_curr["embryo_id"]):
            if eid in idx_map_next:
                rel[i] = idx_map_next[eid]
        relations.append(rel)

    return slices, relations, slice_meta


def run_aligned_umap(
    slices: list[np.ndarray],
    relations: list[dict[int, int]],
    slice_meta: list[pd.DataFrame],
    n_components: int,
    alignment_regularisation: float,
    alignment_window_size: int,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> pd.DataFrame:
    """Run AlignedUMAP and return tidy DataFrame with embedding coordinates."""
    print(f"Running AlignedUMAP (n_components={n_components}) on {len(slices)} timebins ...")
    aligned = umap.AlignedUMAP(
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    ).fit(slices, relations=relations)

    coord_cols = [f"UMAP_{i+1}" for i in range(n_components)]
    rows: list[pd.DataFrame] = []
    for t_idx, (emb, meta) in enumerate(zip(aligned.embeddings_, slice_meta)):
        chunk = meta.copy()
        for ci, col in enumerate(coord_cols):
            chunk[col] = emb[:, ci]
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True)

# ── 2D snapshot scatter ────────────────────────────────────────────────────────

def plot_2d_snapshot(
    coords_2d: pd.DataFrame,
    snapshot_time: float,
    color_palette: dict[str, str],
    figures_dir: Path,
) -> None:
    bins = coords_2d["time_bin_center"].unique()
    nearest = bins[np.argmin(np.abs(bins - snapshot_time))]
    sub = coords_2d[coords_2d["time_bin_center"] == nearest]

    fig, ax = plt.subplots(figsize=(6, 5))
    for genotype, grp in sub.groupby("genotype"):
        color = color_palette.get(genotype, "#888888")
        ax.scatter(grp["UMAP_1"], grp["UMAP_2"], c=color, label=_short_name(genotype),
                   s=40, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"AlignedUMAP 2D — {nearest:.0f} hpf", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()
    out = figures_dir / f"snapshot_umap_2d_{nearest:.0f}hpf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── condition graph ────────────────────────────────────────────────────────────

def plot_condition_graphs(
    auc_df: pd.DataFrame,
    genotypes: list[str],
    snapshot_bins: list[float],
    color_palette: dict[str, str],
    figures_dir: Path,
) -> None:
    n_snaps = len(snapshot_bins)
    n_cols = min(3, n_snaps)
    n_rows = int(np.ceil(n_snaps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    prev_pos: dict | None = None
    for ax_i, t in enumerate(snapshot_bins):
        ax = axes[ax_i]
        mat = _make_auroc_matrix(auc_df, genotypes, t)
        G = nx.Graph()
        G.add_nodes_from(genotypes)
        for i, g1 in enumerate(genotypes):
            for j, g2 in enumerate(genotypes):
                if j <= i:
                    continue
                if not np.isnan(mat[i, j]):
                    similarity = 1.0 - mat[i, j]  # lower AUROC = more similar
                    G.add_edge(g1, g2, weight=similarity)

        # warm-start layout from previous timepoint for temporal coherence
        pos = nx.spring_layout(G, weight="weight", seed=42, pos=prev_pos)
        prev_pos = pos

        node_colors = [color_palette.get(g, "#888888") for g in G.nodes()]
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        edge_widths = [w * 6 for w in edge_weights]
        edge_alphas = [0.3 + 0.5 * w for w in edge_weights]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=600, alpha=0.9)
        for (u, v), w, width, alpha in zip(G.edges(), edge_weights, edge_widths, edge_alphas):
            nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)],
                                   width=width, alpha=alpha, edge_color="#555555")
        nx.draw_networkx_labels(G, pos, ax=ax,
                                labels={g: _short_name(g) for g in G.nodes()},
                                font_size=8, font_weight="bold")
        ax.set_title(f"{t:.0f} hpf", fontsize=10)
        ax.axis("off")

    for ax_i in range(n_snaps, len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle("Phenotypic condition graph over time\n(edge thickness ∝ similarity = 1−AUROC)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = figures_dir / "condition_graph_over_time.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phenotypic positioning via pairwise classification probability space.")
    parser.add_argument("--results-dir", type=Path,
                        default=RESULTS_BASE / "phenotypic_positioning")
    parser.add_argument("--figures-dir", type=Path,
                        default=FIGURES_BASE / "phenotypic_positioning")
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--time-col", default="predicted_stage_hpf")
    parser.add_argument("--embedding-prefix", default="z_mu_b")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--genotypes", nargs="+", default=DEFAULT_GENOTYPES)
    parser.add_argument("--skip-classification", action="store_true",
                        help="Load cached predictions instead of re-running classification")
    parser.add_argument("--snapshot-times", nargs="+", type=float, default=[25.0, 55.0, 79.0],
                        help="Timepoints (hpf) for 2D UMAP snapshot plots")
    parser.add_argument("--auroc-snapshot-bins", nargs="+", type=float,
                        default=[25.0, 35.0, 45.0, 55.0, 65.0, 75.0],
                        help="Timepoints for AUROC N×N heatmap snapshots")
    parser.add_argument("--umap-alignment-regularisation", type=float, default=1e-2)
    parser.add_argument("--umap-alignment-window", type=int, default=3)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    genotypes = [_normalize_genotype(g) for g in args.genotypes]
    color_palette = _build_color_palette(genotypes)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data ...")
    df, embryo_meta = load_dataframe(genotypes)
    feature_cols = embedding_features(df, args.embedding_prefix)
    print(f"  {len(df)} rows, {len(embryo_meta)} embryos, {len(feature_cols)} features")

    # ── 2. Run or load all-pairs classifications ───────────────────────────────
    auc_df, pred_df = run_or_load_all_pairs(
        df, genotypes,
        time_col=args.time_col, feature_cols=feature_cols,
        bin_width=args.bin_width, n_splits=args.n_splits,
        n_permutations=args.n_permutations, random_state=args.random_state,
        results_dir=args.results_dir,
        skip_classification=args.skip_classification,
    )
    pair_ids = sorted(auc_df["pair_id"].unique())
    print(f"  {len(pair_ids)} pairs, {len(auc_df)} AUROC rows, {len(pred_df)} prediction rows")

    # ── 3. Build 10D probability vectors ─────────────────────────────────────
    print("Building probability vectors ...")
    prob_vectors, pair_cols = build_probability_vectors(pred_df, embryo_meta)
    pv_path = args.results_dir / "probability_vectors.csv"
    prob_vectors.to_csv(pv_path, index=False)
    print(f"  {len(prob_vectors)} embryo-timebin rows × {len(pair_cols)} pair dimensions")
    print(f"Saved: {pv_path}")

    # ── 4. AUROC ribbon ───────────────────────────────────────────────────────
    print("Plotting AUROC ribbon ...")
    plot_auroc_ribbon(auc_df, pair_ids, args.figures_dir)

    # ── 4b. AUROC snapshots ───────────────────────────────────────────────────
    print("Plotting AUROC heatmap snapshots ...")
    plot_auroc_heatmap_snapshots(auc_df, genotypes, args.auroc_snapshot_bins, args.figures_dir)

    # ── 5 & 6. AlignedUMAP 2D + 3D ───────────────────────────────────────────
    slices, relations, slice_meta = _build_aligned_umap_inputs(prob_vectors, pair_cols)
    umap_kwargs = dict(
        alignment_regularisation=args.umap_alignment_regularisation,
        alignment_window_size=args.umap_alignment_window,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.random_state,
    )

    # 2D
    coords_2d = run_aligned_umap(slices, relations, slice_meta, n_components=2, **umap_kwargs)
    coords_2d_path = args.results_dir / "aligned_umap_2d_coordinates.csv"
    coords_2d.to_csv(coords_2d_path, index=False)
    print(f"Saved: {coords_2d_path}")

    for snap_t in args.snapshot_times:
        plot_2d_snapshot(coords_2d, snap_t, color_palette, args.figures_dir)

    # 3D
    coords_3d = run_aligned_umap(slices, relations, slice_meta, n_components=3, **umap_kwargs)
    coords_3d_path = args.results_dir / "aligned_umap_3d_coordinates.csv"
    coords_3d.to_csv(coords_3d_path, index=False)
    print(f"Saved: {coords_3d_path}")

    # 3D — genotype-colored with trajectory lines
    fig_3d_genotype = plot_3d_scatter(
        df=coords_3d,
        coords=["UMAP_1", "UMAP_2", "UMAP_3"],
        color_by="genotype",
        color_palette=color_palette,
        line_by="embryo_id",
        min_points_per_line=6,
        show_lines=True,
        x_col="time_bin_center",
        line_opacity=0.35,
        show_mean=True,
        mean_line_width=5,
        point_opacity=0.6,
        title="Phenotypic Positioning: AlignedUMAP 3D (genotype)",
        output_path=args.figures_dir / "joint_3d_umap_genotype",
        hover_cols=["embryo_id", "time_bin_center"],
    )
    print(f"Saved: {args.figures_dir / 'joint_3d_umap_genotype.html'}")

    # 3D — time-colored (no trajectory lines to avoid clutter)
    fig_3d_time = plot_3d_scatter(
        df=coords_3d,
        coords=["UMAP_1", "UMAP_2", "UMAP_3"],
        color_by="time_bin_center",
        color_continuous=True,
        group_by="genotype",
        colorscale="Viridis",
        colorbar_title="Time (hpf)",
        line_by="embryo_id",
        min_points_per_line=6,
        show_lines=False,
        point_opacity=0.7,
        title="Phenotypic Positioning: AlignedUMAP 3D (time)",
        output_path=args.figures_dir / "joint_3d_umap_time",
        hover_cols=["embryo_id", "genotype", "time_bin_center"],
    )
    print(f"Saved: {args.figures_dir / 'joint_3d_umap_time.html'}")

    # ── 7. Condition graph ────────────────────────────────────────────────────
    print("Plotting condition graph ...")
    plot_condition_graphs(auc_df, genotypes, args.auroc_snapshot_bins, color_palette, args.figures_dir)

    print("\nDone. All outputs written to:")
    print(f"  Results: {args.results_dir}")
    print(f"  Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
