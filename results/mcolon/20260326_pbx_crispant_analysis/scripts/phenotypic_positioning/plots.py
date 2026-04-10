from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.viz.styling.genotype_colors import SPECIAL_GENOTYPE_COLORS, get_known_genotype_color

from .data import short_name


FIXED_GENOTYPE_COLORS = {
    "inj_ctrl": "#7f7f7f",
    "wik_ab": "#1f77b4",
    "pbx1b_crispant": "#6a3d9a",
    "pbx4_crispant": "#e66100",
    "pbx1b_pbx4_crispant": "#1b9e77",
}


def _genotype_color(genotype: str, fallback_palette: dict[str, str]) -> str:
    key = str(genotype).strip().lower().replace(" ", "_")
    if key in FIXED_GENOTYPE_COLORS:
        return FIXED_GENOTYPE_COLORS[key]
    stripped = key.replace("_crispant", "")
    c = SPECIAL_GENOTYPE_COLORS.get(stripped) or SPECIAL_GENOTYPE_COLORS.get(key)
    if c:
        return c
    c = get_known_genotype_color(genotype)
    if c:
        return c
    return fallback_palette.get(genotype, "#888888")


def build_color_palette(genotypes: list[str]) -> dict[str, str]:
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette: dict[str, str] = {}
    fb_idx = 0
    for genotype in genotypes:
        color = _genotype_color(genotype, {})
        if color == "#888888":
            color = fallback[fb_idx % len(fallback)]
            fb_idx += 1
        palette[genotype] = color
    return palette


def plot_control_control_qc(
    axis_df: pd.DataFrame,
    *,
    figures_dir: Path,
    pair_id: str,
    color_palette: dict[str, str],
) -> None:
    sub = axis_df[axis_df["pair_id"] == pair_id].copy()
    if sub.empty:
        return
    grouped = (
        sub.groupby(["time_bin_center", "genotype"], as_index=False)
        .agg(
            median_position=("position_logit_mean", "median"),
            median_support=("support_weight", "median"),
        )
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for genotype, grp in grouped.groupby("genotype"):
        axes[0].plot(
            grp["time_bin_center"],
            grp["median_position"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color_palette.get(genotype, "#888888"),
            label=short_name(genotype),
        )
        axes[1].plot(
            grp["time_bin_center"],
            grp["median_support"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=color_palette.get(genotype, "#888888"),
            label=short_name(genotype),
        )
    axes[0].axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Median axis position")
    axes[0].set_title("Control-control QC: raw axis position", fontweight="bold")
    axes[1].set_ylabel("Median support weight")
    axes[1].set_xlabel("Time (hpf)")
    axes[1].set_title("Control-control QC: support weight", fontweight="bold")
    axes[0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out = figures_dir / "control_control_qc.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_support_heatmap(summary_df: pd.DataFrame, *, figures_dir: Path) -> None:
    if summary_df.empty:
        return
    pair_ids = sorted(summary_df["pair_id"].unique())
    bins = sorted(summary_df["time_bin_center"].unique())
    grid = np.full((len(pair_ids), len(bins)), np.nan)
    for i, pair in enumerate(pair_ids):
        sub = summary_df[summary_df["pair_id"] == pair]
        for j, time_bin_center in enumerate(bins):
            row = sub[sub["time_bin_center"] == time_bin_center]
            if not row.empty:
                grid[i, j] = float(row["median_support_weight"].iloc[0])

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(pair_ids)))
    ax.set_yticklabels([pair.replace("_crispant", "") for pair in pair_ids], fontsize=8)
    step = max(1, len(bins) // 12)
    ax.set_xticks(range(0, len(bins), step))
    ax.set_xticklabels([f"{bins[idx]:.0f}" for idx in range(0, len(bins), step)], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Time (hpf)")
    ax.set_title("Median support weight by pair and time", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Median support weight")
    fig.tight_layout()
    out = figures_dir / "pairwise_support_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_novelty_residual_scatter(
    axis_df: pd.DataFrame,
    *,
    figures_dir: Path,
    pair_ids: list[str],
    color_palette: dict[str, str],
) -> None:
    if not pair_ids:
        return
    fig, axes = plt.subplots(1, len(pair_ids), figsize=(6 * len(pair_ids), 5), squeeze=False)
    for ax, pair in zip(axes.flatten(), pair_ids):
        sub = axis_df[(axis_df["pair_id"] == pair) & (axis_df["model_available"])].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        for genotype, grp in sub.groupby("genotype"):
            ax.scatter(
                grp["axis_residual_z"],
                grp["knn_novelty_z"],
                s=18,
                alpha=0.45,
                c=color_palette.get(genotype, "#888888"),
                label=short_name(genotype),
            )
        ax.set_xlabel("Axis residual z")
        ax.set_ylabel("kNN novelty z")
        ax.set_title(pair.replace("_crispant", ""))
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = figures_dir / "novelty_vs_residual_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_support_weight_distribution(
    axis_df: pd.DataFrame,
    *,
    figures_dir: Path,
    color_palette: dict[str, str],
) -> None:
    del color_palette
    if axis_df.empty:
        return
    summary = (
        axis_df[axis_df["model_available"]]
        .groupby(["pair_id", "genotype"], as_index=False)
        .agg(
            median_support_weight=("support_weight", "median"),
            mean_support_weight=("support_weight", "mean"),
        )
    )
    pair_ids = sorted(summary["pair_id"].unique())
    genotypes = sorted(summary["genotype"].unique())
    grid = np.full((len(pair_ids), len(genotypes)), np.nan)
    for row_idx, pair in enumerate(pair_ids):
        sub = summary[summary["pair_id"] == pair]
        for col_idx, genotype in enumerate(genotypes):
            hit = sub[sub["genotype"] == genotype]
            if not hit.empty:
                grid[row_idx, col_idx] = float(hit["median_support_weight"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, max(5, 0.45 * len(pair_ids))))
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(genotypes)))
    ax.set_xticklabels([short_name(genotype) for genotype in genotypes], rotation=30, ha="right")
    ax.set_yticks(range(len(pair_ids)))
    ax.set_yticklabels([pair.replace("_crispant", "") for pair in pair_ids], fontsize=8)
    ax.set_xlabel("Genotype")
    ax.set_title("Median support weight by comparison and genotype", fontweight="bold")
    for row_idx in range(len(pair_ids)):
        for col_idx in range(len(genotypes)):
            value = grid[row_idx, col_idx]
            if np.isfinite(value):
                text_color = "white" if value >= 0.55 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color)
    plt.colorbar(im, ax=ax, label="Median support weight")
    fig.tight_layout()
    out = figures_dir / "support_weight_by_genotype.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_aligned_umap_inputs(
    vectors_df: pd.DataFrame,
    vector_cols: list[str],
) -> tuple[list[np.ndarray], list[dict[int, int]], list[pd.DataFrame]]:
    bins = sorted(vectors_df["time_bin_center"].unique())
    slices: list[np.ndarray] = []
    meta_rows: list[pd.DataFrame] = []
    for time_bin_center in bins:
        sub = vectors_df[vectors_df["time_bin_center"] == time_bin_center].reset_index(drop=True)
        if len(sub) < 4:
            continue
        slices.append(sub[vector_cols].to_numpy(dtype=float))
        meta_rows.append(sub[["embryo_id", "genotype", "time_bin_center"]].reset_index(drop=True))
    relations: list[dict[int, int]] = []
    for idx in range(len(meta_rows) - 1):
        current = meta_rows[idx]
        nxt = meta_rows[idx + 1]
        next_index = {eid: i for i, eid in enumerate(nxt["embryo_id"])}
        relation = {}
        for row_idx, embryo_id in enumerate(current["embryo_id"]):
            if embryo_id in next_index:
                relation[row_idx] = next_index[embryo_id]
        relations.append(relation)
    return slices, relations, meta_rows


def _run_aligned_umap_coords(
    vectors_df: pd.DataFrame,
    *,
    vector_cols: list[str],
    random_state: int,
    alignment_regularisation: float = 0.1,
    alignment_window_size: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> pd.DataFrame | None:
    if vectors_df.empty or not vector_cols:
        return None
    import umap

    slices, relations, meta_rows = _build_aligned_umap_inputs(vectors_df, vector_cols)
    if len(slices) < 2:
        return None
    min_slice_size = min(len(slc) for slc in slices)
    effective_neighbors = max(2, min(int(n_neighbors), min_slice_size - 1))
    aligned = umap.AlignedUMAP(
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        init="random",
        random_state=random_state,
    ).fit(slices, relations=relations)

    rows = []
    for embedding, meta in zip(aligned.embeddings_, meta_rows):
        chunk = meta.copy()
        chunk["UMAP_1"] = embedding[:, 0]
        chunk["UMAP_2"] = embedding[:, 1]
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True)


def _run_plain_umap_coords(
    vectors_df: pd.DataFrame,
    *,
    vector_cols: list[str],
    random_state: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> pd.DataFrame | None:
    if vectors_df.empty or not vector_cols:
        return None
    import umap

    if len(vectors_df) < 3:
        return None
    effective_neighbors = max(2, min(int(n_neighbors), len(vectors_df) - 1))
    embedding = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        init="random",
        random_state=random_state,
    ).fit_transform(vectors_df[vector_cols].to_numpy(dtype=float))
    coords = vectors_df[["embryo_id", "genotype", "time_bin_center"]].copy()
    coords["UMAP_1"] = embedding[:, 0]
    coords["UMAP_2"] = embedding[:, 1]
    return coords.reset_index(drop=True)


def _plot_umap_snapshot(
    coords: pd.DataFrame,
    *,
    snapshot_time: float,
    color_palette: dict[str, str],
    title: str,
    output_path: Path,
    axis_limits: tuple[float, float, float, float] | None = None,
) -> None:
    available_bins = coords["time_bin_center"].unique()
    nearest = available_bins[np.argmin(np.abs(available_bins - snapshot_time))]
    sub = coords[coords["time_bin_center"] == nearest]
    fig, ax = plt.subplots(figsize=(6, 5))
    for genotype, grp in sub.groupby("genotype"):
        ax.scatter(
            grp["UMAP_1"],
            grp["UMAP_2"],
            s=26,
            alpha=0.75,
            c=color_palette.get(genotype, "#888888"),
            label=short_name(genotype),
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{title} - {nearest:.0f} hpf", fontweight="bold")
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_embedding_comparison_qc(
    raw_vectors_df: pd.DataFrame,
    supported_vectors_df: pd.DataFrame,
    *,
    pair_cols: list[str],
    results_dir: Path,
    figures_dir: Path,
    color_palette: dict[str, str],
    snapshot_times: list[float],
    random_state: int,
    alignment_regularisation: float = 0.1,
    alignment_window_size: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    raw_coords = _run_aligned_umap_coords(
        raw_vectors_df,
        vector_cols=pair_cols,
        random_state=random_state,
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    supported_coords = _run_aligned_umap_coords(
        supported_vectors_df,
        vector_cols=pair_cols,
        random_state=random_state,
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    if raw_coords is None or supported_coords is None:
        return

    raw_coords.to_csv(results_dir / "raw_aligned_umap_2d_coordinates.csv", index=False)
    supported_coords.to_csv(results_dir / "supported_aligned_umap_2d_coordinates.csv", index=False)

    available_raw = set(raw_coords["time_bin_center"].unique())
    available_supported = set(supported_coords["time_bin_center"].unique())
    common_bins = np.array(sorted(available_raw & available_supported), dtype=float)
    if len(common_bins) == 0:
        return

    all_x = np.concatenate([raw_coords["UMAP_1"].to_numpy(), supported_coords["UMAP_1"].to_numpy()])
    all_y = np.concatenate([raw_coords["UMAP_2"].to_numpy(), supported_coords["UMAP_2"].to_numpy()])
    pad_x = 0.05 * max(1e-6, float(all_x.max() - all_x.min()))
    pad_y = 0.05 * max(1e-6, float(all_y.max() - all_y.min()))
    axis_limits = (
        float(all_x.min() - pad_x),
        float(all_x.max() + pad_x),
        float(all_y.min() - pad_y),
        float(all_y.max() + pad_y),
    )

    for snapshot_time in snapshot_times:
        nearest = common_bins[np.argmin(np.abs(common_bins - snapshot_time))]
        _plot_umap_snapshot(
            raw_coords,
            snapshot_time=nearest,
            color_palette=color_palette,
            title="Raw AlignedUMAP 2D",
            output_path=figures_dir / f"raw_umap_2d_{nearest:.0f}hpf.png",
            axis_limits=axis_limits,
        )
        _plot_umap_snapshot(
            supported_coords,
            snapshot_time=nearest,
            color_palette=color_palette,
            title="Support-aware AlignedUMAP 2D",
            output_path=figures_dir / f"support_umap_2d_{nearest:.0f}hpf.png",
            axis_limits=axis_limits,
        )

        raw_sub = raw_coords[raw_coords["time_bin_center"] == nearest]
        supported_sub = supported_coords[supported_coords["time_bin_center"] == nearest]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for ax, sub, title in [
            (axes[0], raw_sub, "Raw"),
            (axes[1], supported_sub, "Support-aware"),
        ]:
            for genotype, grp in sub.groupby("genotype"):
                ax.scatter(
                    grp["UMAP_1"],
                    grp["UMAP_2"],
                    s=26,
                    alpha=0.75,
                    c=color_palette.get(genotype, "#888888"),
                    label=short_name(genotype),
                )
            ax.set_title(f"{title} - {nearest:.0f} hpf", fontweight="bold")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(figures_dir / f"compare_umap_2d_{nearest:.0f}hpf.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_multiclass_probability_trajectories(
    trajectory_df: pd.DataFrame,
    *,
    figures_dir: Path,
    color_palette: dict[str, str],
) -> None:
    if trajectory_df.empty:
        return
    genotypes = trajectory_df["genotype"].drop_duplicates().tolist()
    n_cols = min(3, len(genotypes))
    n_rows = int(np.ceil(len(genotypes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.8 * n_rows), squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for ax, genotype in zip(axes_flat, genotypes):
        sub = trajectory_df[trajectory_df["genotype"] == genotype]
        for predicted_class, grp in sub.groupby("predicted_class"):
            ax.plot(
                grp["time_bin_center"],
                grp["median_probability"],
                color=color_palette.get(predicted_class, "#888888"),
                linewidth=1.8,
                label=short_name(predicted_class),
            )
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel("Median class probability")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes_flat[len(genotypes):]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures_dir / "multiclass_probability_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_confusion_snapshots(
    confusion_df: pd.DataFrame,
    *,
    figures_dir: Path,
    class_labels: list[str],
    snapshot_times: list[float],
) -> None:
    if confusion_df.empty:
        return
    available_bins = np.array(sorted(confusion_df["time_bin_center"].unique()), dtype=float)
    if len(available_bins) == 0:
        return
    label_names = [short_name(label) for label in class_labels]
    for snapshot_time in snapshot_times:
        nearest = float(available_bins[np.argmin(np.abs(available_bins - snapshot_time))])
        sub = confusion_df[confusion_df["time_bin_center"] == nearest].copy()
        if sub.empty:
            continue
        grid = np.full((len(class_labels), len(class_labels)), np.nan)
        for i, true_class in enumerate(class_labels):
            for j, pred_class in enumerate(class_labels):
                hit = sub[(sub["true_class"] == true_class) & (sub["predicted_class"] == pred_class)]
                if not hit.empty:
                    grid[i, j] = float(hit["proportion"].iloc[0])
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(grid, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(class_labels)))
        ax.set_xticklabels(label_names, rotation=30, ha="right")
        ax.set_yticks(range(len(class_labels)))
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"Multiclass confusion - {nearest:.0f} hpf", fontweight="bold")
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                value = grid[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value >= 0.55 else "black")
        plt.colorbar(im, ax=ax, label="Proportion")
        fig.tight_layout()
        fig.savefig(figures_dir / f"multiclass_confusion_heatmap_{nearest:.0f}hpf.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_multiclass_centroid_distances(
    centroid_distance_df: pd.DataFrame,
    *,
    figures_dir: Path,
) -> None:
    if centroid_distance_df.empty:
        return
    highlight_pairs = {
        frozenset({"inj_ctrl", "wik_ab"}): "inj_ctrl vs wik_ab",
        frozenset({"pbx4_crispant", "pbx1b_pbx4_crispant"}): "pbx4 vs pbx1b+4",
        frozenset({"inj_ctrl", "pbx4_crispant"}): "inj_ctrl vs pbx4",
    }
    sub = centroid_distance_df[
        centroid_distance_df.apply(
            lambda row: frozenset({str(row["genotype_1"]), str(row["genotype_2"])}) in highlight_pairs,
            axis=1,
        )
    ].copy()
    if sub.empty:
        return
    sub["pair_label"] = sub.apply(
        lambda row: highlight_pairs[frozenset({str(row["genotype_1"]), str(row["genotype_2"])})],
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pair_label, grp in sub.groupby("pair_label"):
        ax.plot(grp["time_bin_center"], grp["distance_l2"], linewidth=2.0, label=pair_label)
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Centroid distance (L2)")
    ax.set_title("Genotype centroid distances in multiclass probability space", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "multiclass_centroid_distance_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_multiclass_embedding_qc(
    vector_df: pd.DataFrame,
    *,
    vector_cols: list[str],
    results_dir: Path,
    figures_dir: Path,
    color_palette: dict[str, str],
    snapshot_times: list[float],
    random_state: int,
    alignment_regularisation: float = 0.1,
    alignment_window_size: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> None:
    aligned_coords = _run_aligned_umap_coords(
        vector_df,
        vector_cols=vector_cols,
        random_state=random_state,
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    plain_coords = _run_plain_umap_coords(
        vector_df,
        vector_cols=vector_cols,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    if aligned_coords is None or plain_coords is None:
        return

    aligned_coords.to_csv(results_dir / "multiclass_aligned_umap_2d_coordinates.csv", index=False)
    plain_coords.to_csv(results_dir / "multiclass_plain_umap_2d_coordinates.csv", index=False)

    common_bins = np.array(
        sorted(set(aligned_coords["time_bin_center"].unique()) & set(plain_coords["time_bin_center"].unique())),
        dtype=float,
    )
    if len(common_bins) == 0:
        return

    all_x = np.concatenate([aligned_coords["UMAP_1"].to_numpy(), plain_coords["UMAP_1"].to_numpy()])
    all_y = np.concatenate([aligned_coords["UMAP_2"].to_numpy(), plain_coords["UMAP_2"].to_numpy()])
    pad_x = 0.05 * max(1e-6, float(all_x.max() - all_x.min()))
    pad_y = 0.05 * max(1e-6, float(all_y.max() - all_y.min()))
    axis_limits = (
        float(all_x.min() - pad_x),
        float(all_x.max() + pad_x),
        float(all_y.min() - pad_y),
        float(all_y.max() + pad_y),
    )

    for snapshot_time in snapshot_times:
        nearest = float(common_bins[np.argmin(np.abs(common_bins - snapshot_time))])
        _plot_umap_snapshot(
            aligned_coords,
            snapshot_time=nearest,
            color_palette=color_palette,
            title="Multiclass AlignedUMAP 2D",
            output_path=figures_dir / f"multiclass_aligned_umap_2d_{nearest:.0f}hpf.png",
            axis_limits=axis_limits,
        )
        _plot_umap_snapshot(
            plain_coords,
            snapshot_time=nearest,
            color_palette=color_palette,
            title="Multiclass Plain UMAP 2D",
            output_path=figures_dir / f"multiclass_plain_umap_2d_{nearest:.0f}hpf.png",
            axis_limits=axis_limits,
        )

        aligned_sub = aligned_coords[aligned_coords["time_bin_center"] == nearest]
        plain_sub = plain_coords[plain_coords["time_bin_center"] == nearest]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        for ax, sub, title in [
            (axes[0], aligned_sub, "Aligned"),
            (axes[1], plain_sub, "Plain"),
        ]:
            for genotype, grp in sub.groupby("genotype"):
                ax.scatter(
                    grp["UMAP_1"],
                    grp["UMAP_2"],
                    s=26,
                    alpha=0.75,
                    c=color_palette.get(genotype, "#888888"),
                    label=short_name(genotype),
                )
            ax.set_title(f"{title} - {nearest:.0f} hpf", fontweight="bold")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(
            figures_dir / f"multiclass_compare_aligned_vs_plain_{nearest:.0f}hpf.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
