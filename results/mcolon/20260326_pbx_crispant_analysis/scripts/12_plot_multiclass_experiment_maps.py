from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_multiclass_experiment_maps_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
import umap  # noqa: E402

from analyze.viz.plotting.plotting_3d import plot_3d_scatter  # noqa: E402
from phenotypic_positioning.config import DEFAULT_MULTICLASS_RESULTS_SUBDIR, DEFAULT_SNAPSHOT_TIMES, FIGURES_BASE, RESULTS_BASE  # noqa: E402
from phenotypic_positioning.data import aggregate_features_by_time, load_dataframe, resolve_feature_columns, short_name  # noqa: E402
from phenotypic_positioning.plots import build_color_palette  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment-split multiclass PCA and embedding maps.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_BASE / DEFAULT_MULTICLASS_RESULTS_SUBDIR,
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_BASE / DEFAULT_MULTICLASS_RESULTS_SUBDIR,
    )
    parser.add_argument("--output-subdir", default="experiment_maps")
    parser.add_argument("--snapshot-times", nargs="+", type=float, default=DEFAULT_SNAPSHOT_TIMES)
    return parser.parse_args()


def _compute_probability_pca(vector_df: pd.DataFrame, prob_cols: list[str]) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vector_df[prob_cols].to_numpy(dtype=float))
    out = vector_df[["embryo_id", "genotype", "experiment_id", "time_bin_center"]].copy()
    out["PC_1"] = coords[:, 0]
    out["PC_2"] = coords[:, 1]
    return out


def _compute_raw_latent_pca(latent_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(latent_df[feature_cols].to_numpy(dtype=float))
    out = latent_df[["embryo_id", "genotype", "experiment_id", "time_bin_center"]].copy()
    out["PC_1"] = coords[:, 0]
    out["PC_2"] = coords[:, 1]
    return out


def _compute_raw_latent_pca_3d(latent_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(latent_df[feature_cols].to_numpy(dtype=float))
    out = latent_df[["embryo_id", "genotype", "experiment_id", "time_bin_center"]].copy()
    out["PC_1"] = coords[:, 0]
    out["PC_2"] = coords[:, 1]
    out["PC_3"] = coords[:, 2]
    return out


def _merge_experiment_id(coords_df: pd.DataFrame, vector_df: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["embryo_id", "time_bin_center", "experiment_id"]
    meta = vector_df[meta_cols].drop_duplicates()
    merged = coords_df.merge(meta, on=["embryo_id", "time_bin_center"], how="left", validate="many_to_one")
    return merged


def _axis_limits(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, float, float]:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    pad_x = 0.05 * max(1e-6, float(x.max() - x.min()))
    pad_y = 0.05 * max(1e-6, float(y.max() - y.min()))
    return (
        float(x.min() - pad_x),
        float(x.max() + pad_x),
        float(y.min() - pad_y),
        float(y.max() + pad_y),
    )


def _plot_by_experiment(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    color_palette: dict[str, str],
    axis_limits: tuple[float, float, float, float],
) -> None:
    experiments = sorted(df["experiment_id"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(experiments), figsize=(6 * len(experiments), 5), sharex=True, sharey=True, squeeze=False)
    for ax, experiment_id in zip(axes.flatten(), experiments):
        sub = df[df["experiment_id"] == experiment_id]
        for genotype, grp in sub.groupby("genotype"):
            ax.scatter(
                grp[x_col],
                grp[y_col],
                s=18,
                alpha=0.75,
                c=color_palette.get(genotype, "#888888"),
                label=short_name(genotype),
            )
        ax.set_title(f"{experiment_id} (n={len(sub)})", fontweight="bold")
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " "))
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(5, len(labels)),
            fontsize=8,
        )
    fig.suptitle(title, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.86, bottom=0.18, wspace=0.12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    color_palette: dict[str, str],
    axis_limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for genotype, grp in df.groupby("genotype"):
        ax.scatter(
            grp[x_col],
            grp[y_col],
            s=20,
            alpha=0.75,
            c=color_palette.get(genotype, "#888888"),
            label=short_name(genotype),
        )
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_xlim(axis_limits[0], axis_limits[1])
    ax.set_ylim(axis_limits[2], axis_limits[3])
    ax.set_title(title, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(5, len(labels)),
            fontsize=8,
        )
    fig.subplots_adjust(top=0.90, bottom=0.18)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_time(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    axis_limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    times = df["time_bin_center"].to_numpy(dtype=float)
    scatter = ax.scatter(
        df[x_col],
        df[y_col],
        s=20,
        alpha=0.8,
        c=times,
        cmap="viridis",
        vmin=float(np.min(times)),
        vmax=float(np.max(times)),
    )
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(y_col.replace("_", " "))
    ax.set_xlim(axis_limits[0], axis_limits[1])
    ax.set_ylim(axis_limits[2], axis_limits[3])
    ax.set_title(title, fontweight="bold")
    fig.subplots_adjust(top=0.90, right=0.86)
    cax = fig.add_axes([0.89, 0.18, 0.02, 0.62])
    colorbar = fig.colorbar(scatter, cax=cax)
    colorbar.set_label("Time (hpf)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_by_experiment_time(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    axis_limits: tuple[float, float, float, float],
) -> None:
    experiments = sorted(df["experiment_id"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(experiments), figsize=(6 * len(experiments) + 1.2, 5), sharex=True, sharey=True, squeeze=False)
    all_times = df["time_bin_center"].to_numpy(dtype=float)
    vmin = float(np.min(all_times))
    vmax = float(np.max(all_times))
    last_scatter = None
    for ax, experiment_id in zip(axes.flatten(), experiments):
        sub = df[df["experiment_id"] == experiment_id]
        last_scatter = ax.scatter(
            sub[x_col],
            sub[y_col],
            s=20,
            alpha=0.8,
            c=sub["time_bin_center"].to_numpy(dtype=float),
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{experiment_id} (n={len(sub)})", fontweight="bold")
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " "))
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    fig.suptitle(title, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.86, bottom=0.12, right=0.86, wspace=0.12)
    if last_scatter is not None:
        cax = fig.add_axes([0.89, 0.18, 0.02, 0.62])
        colorbar = fig.colorbar(last_scatter, cax=cax)
        colorbar.set_label("Time (hpf)")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _nearest_snapshot(df: pd.DataFrame, target_time: float) -> float:
    bins = np.array(sorted(df["time_bin_center"].unique()), dtype=float)
    return float(bins[np.argmin(np.abs(bins - target_time))])


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
        meta_rows.append(sub[["embryo_id", "genotype", "experiment_id", "time_bin_center"]].reset_index(drop=True))
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


def _run_aligned_umap_3d(
    vectors_df: pd.DataFrame,
    *,
    vector_cols: list[str],
    random_state: int,
    alignment_regularisation: float = 0.1,
    alignment_window_size: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> pd.DataFrame:
    slices, relations, meta_rows = _build_aligned_umap_inputs(vectors_df, vector_cols)
    if len(slices) < 2:
        raise ValueError("Need at least two time slices for aligned UMAP 3D.")
    min_slice_size = min(len(slc) for slc in slices)
    effective_neighbors = max(2, min(int(n_neighbors), min_slice_size - 1))
    aligned = umap.AlignedUMAP(
        alignment_regularisation=alignment_regularisation,
        alignment_window_size=alignment_window_size,
        n_components=3,
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
        chunk["UMAP_3"] = embedding[:, 2]
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True)


def _plot_combined_3d(
    df: pd.DataFrame,
    *,
    coords: list[str],
    output_base: Path,
    title_prefix: str,
    color_palette: dict[str, str],
) -> None:
    plot_3d_scatter(
        df=df,
        coords=coords,
        color_by="genotype",
        color_palette=color_palette,
        line_by="embryo_id",
        min_points_per_line=6,
        show_lines=False,
        point_opacity=0.7,
        title=f"{title_prefix} 3D (genotype)",
        output_path=output_base.with_name(output_base.name + "_genotype"),
        hover_cols=["embryo_id", "experiment_id", "time_bin_center"],
    )
    plot_3d_scatter(
        df=df,
        coords=coords,
        color_by="time_bin_center",
        color_continuous=True,
        group_by="genotype",
        colorscale="Viridis",
        colorbar_title="Time (hpf)",
        line_by="embryo_id",
        min_points_per_line=6,
        show_lines=False,
        point_opacity=0.75,
        title=f"{title_prefix} 3D (time)",
        output_path=output_base.with_name(output_base.name + "_time"),
        hover_cols=["embryo_id", "experiment_id", "genotype", "time_bin_center"],
    )


def main() -> None:
    args = parse_args()
    output_dir = args.figures_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_json(args.results_dir / "multiclass_manifest.json", typ="series")
    vector_df = pd.read_csv(args.results_dir / "multiclass_probability_vectors.csv")
    aligned_df = pd.read_csv(args.results_dir / "multiclass_aligned_umap_2d_coordinates.csv")

    prob_cols = [c for c in vector_df.columns if c.startswith("pred_proba_")]
    if not prob_cols:
        raise ValueError("No probability columns found in multiclass_probability_vectors.csv")

    latent_df, _ = load_dataframe(list(manifest["genotypes"]))
    feature_cols = resolve_feature_columns(latent_df, str(manifest["embedding_prefix"]))
    latent_binned = aggregate_features_by_time(
        latent_df,
        feature_cols=feature_cols,
        time_col=str(manifest["time_col"]),
        bin_width=float(manifest["bin_width"]),
    )

    probability_pca_df = _compute_probability_pca(vector_df, prob_cols)
    raw_latent_pca_df = _compute_raw_latent_pca(latent_binned, feature_cols)
    raw_latent_pca_3d_df = _compute_raw_latent_pca_3d(latent_binned, feature_cols)
    aligned_3d_df = _run_aligned_umap_3d(
        vector_df,
        vector_cols=prob_cols,
        random_state=42,
    )
    aligned_df = _merge_experiment_id(aligned_df, vector_df)

    color_palette = build_color_palette(sorted(vector_df["genotype"].dropna().unique().tolist()))

    raw_latent_pca_3d_df.to_csv(args.results_dir / "combined_experiment_raw_z_mu_b_pca_3d_coordinates.csv", index=False)
    aligned_3d_df.to_csv(args.results_dir / "combined_experiment_multiclass_aligned_umap_3d_coordinates.csv", index=False)

    probability_pca_limits = _axis_limits(probability_pca_df, "PC_1", "PC_2")
    raw_latent_pca_limits = _axis_limits(raw_latent_pca_df, "PC_1", "PC_2")
    aligned_limits = _axis_limits(aligned_df, "UMAP_1", "UMAP_2")

    _plot_by_experiment(
        probability_pca_df,
        x_col="PC_1",
        y_col="PC_2",
        output_path=output_dir / "multiclass_probability_pca_by_experiment_all.png",
        title="Multiclass probability PCA by experiment",
        color_palette=color_palette,
        axis_limits=probability_pca_limits,
    )
    _plot_combined(
        probability_pca_df,
        x_col="PC_1",
        y_col="PC_2",
        output_path=output_dir / "combined_experiment_multiclass_probability_pca.png",
        title="Combined experiment multiclass probability PCA",
        color_palette=color_palette,
        axis_limits=probability_pca_limits,
    )
    _plot_by_experiment_time(
        probability_pca_df,
        x_col="PC_1",
        y_col="PC_2",
        output_path=output_dir / "multiclass_probability_pca_by_experiment_all_time.png",
        title="Multiclass probability PCA by experiment, colored by time",
        axis_limits=probability_pca_limits,
    )
    _plot_combined_time(
        probability_pca_df,
        x_col="PC_1",
        y_col="PC_2",
        output_path=output_dir / "combined_experiment_multiclass_probability_pca_time.png",
        title="Combined experiment multiclass probability PCA, colored by time",
        axis_limits=probability_pca_limits,
    )
    _plot_by_experiment(
        raw_latent_pca_df,
        x_col="PC_1",
        y_col="PC_2",
        output_path=output_dir / "raw_z_mu_b_pca_by_experiment_all.png",
        title="Raw z_mu_b PCA by experiment",
        color_palette=color_palette,
        axis_limits=raw_latent_pca_limits,
    )
    _plot_combined(
        raw_latent_pca_df,
        x_col="PC_1",
        y_col="PC_2",
        output_path=output_dir / "combined_experiment_raw_z_mu_b_pca.png",
        title="Combined experiment raw z_mu_b PCA",
        color_palette=color_palette,
        axis_limits=raw_latent_pca_limits,
    )
    _plot_by_experiment_time(
        raw_latent_pca_df,
        x_col="PC_1",
        y_col="PC_2",
        output_path=output_dir / "raw_z_mu_b_pca_by_experiment_all_time.png",
        title="Raw z_mu_b PCA by experiment, colored by time",
        axis_limits=raw_latent_pca_limits,
    )
    _plot_combined_time(
        raw_latent_pca_df,
        x_col="PC_1",
        y_col="PC_2",
        output_path=output_dir / "combined_experiment_raw_z_mu_b_pca_time.png",
        title="Combined experiment raw z_mu_b PCA, colored by time",
        axis_limits=raw_latent_pca_limits,
    )
    _plot_by_experiment(
        aligned_df,
        x_col="UMAP_1",
        y_col="UMAP_2",
        output_path=output_dir / "multiclass_aligned_umap_by_experiment_all.png",
        title="Multiclass aligned UMAP by experiment",
        color_palette=color_palette,
        axis_limits=aligned_limits,
    )
    _plot_combined(
        aligned_df,
        x_col="UMAP_1",
        y_col="UMAP_2",
        output_path=output_dir / "combined_experiment_multiclass_aligned_umap.png",
        title="Combined experiment multiclass aligned UMAP",
        color_palette=color_palette,
        axis_limits=aligned_limits,
    )
    _plot_by_experiment_time(
        aligned_df,
        x_col="UMAP_1",
        y_col="UMAP_2",
        output_path=output_dir / "multiclass_aligned_umap_by_experiment_all_time.png",
        title="Multiclass aligned UMAP by experiment, colored by time",
        axis_limits=aligned_limits,
    )
    _plot_combined_time(
        aligned_df,
        x_col="UMAP_1",
        y_col="UMAP_2",
        output_path=output_dir / "combined_experiment_multiclass_aligned_umap_time.png",
        title="Combined experiment multiclass aligned UMAP, colored by time",
        axis_limits=aligned_limits,
    )
    _plot_combined_3d(
        raw_latent_pca_3d_df,
        coords=["PC_1", "PC_2", "PC_3"],
        output_base=output_dir / "combined_experiment_raw_z_mu_b_pca_3d",
        title_prefix="Combined experiment raw z_mu_b PCA",
        color_palette=color_palette,
    )
    _plot_combined_3d(
        aligned_3d_df,
        coords=["UMAP_1", "UMAP_2", "UMAP_3"],
        output_base=output_dir / "combined_experiment_multiclass_aligned_umap_3d",
        title_prefix="Combined experiment multiclass aligned UMAP",
        color_palette=color_palette,
    )

    for snapshot_time in args.snapshot_times:
        nearest = _nearest_snapshot(vector_df, snapshot_time)
        probability_pca_sub = probability_pca_df[probability_pca_df["time_bin_center"] == nearest].copy()
        raw_latent_pca_sub = raw_latent_pca_df[raw_latent_pca_df["time_bin_center"] == nearest].copy()
        aligned_sub = aligned_df[aligned_df["time_bin_center"] == nearest].copy()
        if not probability_pca_sub.empty:
            _plot_by_experiment(
                probability_pca_sub,
                x_col="PC_1",
                y_col="PC_2",
                output_path=output_dir / f"multiclass_probability_pca_by_experiment_{nearest:.0f}hpf.png",
                title=f"Multiclass probability PCA by experiment - {nearest:.0f} hpf",
                color_palette=color_palette,
                axis_limits=probability_pca_limits,
            )
            _plot_by_experiment_time(
                probability_pca_sub,
                x_col="PC_1",
                y_col="PC_2",
                output_path=output_dir / f"multiclass_probability_pca_by_experiment_{nearest:.0f}hpf_time.png",
                title=f"Multiclass probability PCA by experiment - {nearest:.0f} hpf, colored by time",
                axis_limits=probability_pca_limits,
            )
            _plot_combined(
                probability_pca_sub,
                x_col="PC_1",
                y_col="PC_2",
                output_path=output_dir / f"combined_experiment_multiclass_probability_pca_{nearest:.0f}hpf.png",
                title=f"Combined experiment multiclass probability PCA - {nearest:.0f} hpf",
                color_palette=color_palette,
                axis_limits=probability_pca_limits,
            )
        if not raw_latent_pca_sub.empty:
            _plot_by_experiment(
                raw_latent_pca_sub,
                x_col="PC_1",
                y_col="PC_2",
                output_path=output_dir / f"raw_z_mu_b_pca_by_experiment_{nearest:.0f}hpf.png",
                title=f"Raw z_mu_b PCA by experiment - {nearest:.0f} hpf",
                color_palette=color_palette,
                axis_limits=raw_latent_pca_limits,
            )
            _plot_by_experiment_time(
                raw_latent_pca_sub,
                x_col="PC_1",
                y_col="PC_2",
                output_path=output_dir / f"raw_z_mu_b_pca_by_experiment_{nearest:.0f}hpf_time.png",
                title=f"Raw z_mu_b PCA by experiment - {nearest:.0f} hpf, colored by time",
                axis_limits=raw_latent_pca_limits,
            )
            _plot_combined(
                raw_latent_pca_sub,
                x_col="PC_1",
                y_col="PC_2",
                output_path=output_dir / f"combined_experiment_raw_z_mu_b_pca_{nearest:.0f}hpf.png",
                title=f"Combined experiment raw z_mu_b PCA - {nearest:.0f} hpf",
                color_palette=color_palette,
                axis_limits=raw_latent_pca_limits,
            )
        if not aligned_sub.empty:
            _plot_by_experiment(
                aligned_sub,
                x_col="UMAP_1",
                y_col="UMAP_2",
                output_path=output_dir / f"multiclass_aligned_umap_by_experiment_{nearest:.0f}hpf.png",
                title=f"Multiclass aligned UMAP by experiment - {nearest:.0f} hpf",
                color_palette=color_palette,
                axis_limits=aligned_limits,
            )
            _plot_by_experiment_time(
                aligned_sub,
                x_col="UMAP_1",
                y_col="UMAP_2",
                output_path=output_dir / f"multiclass_aligned_umap_by_experiment_{nearest:.0f}hpf_time.png",
                title=f"Multiclass aligned UMAP by experiment - {nearest:.0f} hpf, colored by time",
                axis_limits=aligned_limits,
            )
            _plot_combined(
                aligned_sub,
                x_col="UMAP_1",
                y_col="UMAP_2",
                output_path=output_dir / f"combined_experiment_multiclass_aligned_umap_{nearest:.0f}hpf.png",
                title=f"Combined experiment multiclass aligned UMAP - {nearest:.0f} hpf",
                color_palette=color_palette,
                axis_limits=aligned_limits,
            )

    print("Done.")
    print(f"  Figures: {output_dir}")


if __name__ == "__main__":
    main()
