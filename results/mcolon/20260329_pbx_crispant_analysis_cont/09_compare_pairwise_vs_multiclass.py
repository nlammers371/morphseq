from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_pairwise_compare_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "results/mcolon/20260326_pbx_crispant_analysis/scripts"))

from phenotypic_positioning.plots import build_color_palette
from phenotypic_positioning.data import short_name


KEY_COLS = ["embryo_id", "time_bin_center", "genotype"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare pairwise raw/shrunk coordinates against multiclass probability vectors.")
    parser.add_argument(
        "--pairwise-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_pairwise_bin4_perm500",
    )
    parser.add_argument(
        "--multiclass-path",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_multiclass_bin4_perm500" / "multiclass_probability_vectors.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "compare_pairwise_vs_multiclass_bin4_perm500",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "figures" / "compare_pairwise_vs_multiclass_bin4_perm500",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_representation(path: Path, *, prefixes: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c.startswith(prefixes)]
    if not feature_cols:
        raise ValueError(f"No feature columns with prefixes {prefixes} found in {path}")
    required = set(KEY_COLS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    return df, feature_cols


def align_representations(representations: dict[str, tuple[pd.DataFrame, list[str]]]) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    common = None
    for df, _ in representations.values():
        keys = df[KEY_COLS].drop_duplicates()
        common = keys if common is None else common.merge(keys, on=KEY_COLS, how="inner")
    assert common is not None
    aligned: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for name, (df, feature_cols) in representations.items():
        merged = common.merge(df, on=KEY_COLS, how="left", validate="one_to_one")
        aligned[name] = (merged.sort_values(["time_bin_center", "embryo_id"]).reset_index(drop=True), feature_cols)
    return aligned


def summarize_centroid_distances(df: pd.DataFrame, *, feature_cols: list[str], representation: str) -> pd.DataFrame:
    centroids = (
        df.groupby(["time_bin_center", "genotype"], as_index=False)[feature_cols]
        .median()
        .sort_values(["time_bin_center", "genotype"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    for time_bin_center, group in centroids.groupby("time_bin_center"):
        labels = group["genotype"].tolist()
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                left = group.iloc[i][feature_cols].to_numpy(dtype=float)
                right = group.iloc[j][feature_cols].to_numpy(dtype=float)
                rows.append({
                    "representation": representation,
                    "time_bin_center": float(time_bin_center),
                    "genotype_1": labels[i],
                    "genotype_2": labels[j],
                    "distance_l2": float(np.linalg.norm(left - right)),
                    "distance_l1": float(np.abs(left - right).sum()),
                })
    return pd.DataFrame(rows)


def summarize_silhouette(df: pd.DataFrame, *, feature_cols: list[str], representation: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_bin_center, group in df.groupby("time_bin_center"):
        labels = group["genotype"].astype(str)
        if labels.nunique() < 2 or len(group) <= labels.nunique():
            score = np.nan
        else:
            score = float(silhouette_score(group[feature_cols].to_numpy(dtype=float), labels))
        rows.append({
            "representation": representation,
            "time_bin_center": float(time_bin_center),
            "silhouette": score,
            "n_rows": int(len(group)),
            "n_genotypes": int(labels.nunique()),
        })
    return pd.DataFrame(rows)


def compute_umap(df: pd.DataFrame, *, feature_cols: list[str], random_state: int) -> pd.DataFrame:
    import umap

    if len(df) < 3:
        raise ValueError("Need at least 3 rows to compute UMAP.")
    n_neighbors = max(2, min(15, len(df) - 1))
    embedding = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        init="random",
        random_state=random_state,
    ).fit_transform(df[feature_cols].to_numpy(dtype=float))
    coords = df[KEY_COLS].copy()
    coords["UMAP_1"] = embedding[:, 0]
    coords["UMAP_2"] = embedding[:, 1]
    return coords


def plot_umap_comparison(
    coords_by_representation: dict[str, pd.DataFrame],
    *,
    output_path: Path,
) -> None:
    genotypes = sorted({g for df in coords_by_representation.values() for g in df["genotype"].unique()})
    palette = build_color_palette(genotypes)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=False, sharey=False)
    for ax, (name, coords) in zip(axes, coords_by_representation.items()):
        for genotype, group in coords.groupby("genotype"):
            ax.scatter(
                group["UMAP_1"],
                group["UMAP_2"],
                s=18,
                alpha=0.7,
                color=palette.get(genotype, "#888888"),
                label=short_name(genotype),
            )
        ax.set_title(name.replace("_", " "), fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    representations = {
        "multiclass": load_representation(args.multiclass_path, prefixes=("pred_proba_", "p_")),
        "pairwise_raw": load_representation(args.pairwise_dir / "pairwise_raw_vectors.csv", prefixes=("",)),
        "pairwise_shrunk": load_representation(args.pairwise_dir / "pairwise_shrunk_vectors.csv", prefixes=("",)),
    }
    representations["pairwise_raw"] = (
        representations["pairwise_raw"][0],
        [c for c in representations["pairwise_raw"][1] if "__vs__" in c],
    )
    representations["pairwise_shrunk"] = (
        representations["pairwise_shrunk"][0],
        [c for c in representations["pairwise_shrunk"][1] if "__vs__" in c],
    )
    if not representations["pairwise_raw"][1] or not representations["pairwise_shrunk"][1]:
        raise ValueError("Pairwise vector files do not contain comparison columns with '__vs__'.")

    aligned = align_representations(representations)

    centroid_tables = []
    silhouette_tables = []
    umap_tables: dict[str, pd.DataFrame] = {}
    alignment_counts: dict[str, int] = {}
    for name, (df, feature_cols) in aligned.items():
        centroid_tables.append(summarize_centroid_distances(df, feature_cols=feature_cols, representation=name))
        silhouette_tables.append(summarize_silhouette(df, feature_cols=feature_cols, representation=name))
        umap_df = compute_umap(df, feature_cols=feature_cols, random_state=int(args.random_state))
        umap_tables[name] = umap_df
        alignment_counts[name] = int(len(df))
        umap_df.to_csv(args.output_dir / f"{name}_umap_coordinates.csv", index=False)

    pd.concat(centroid_tables, ignore_index=True).to_csv(args.output_dir / "representation_centroid_distances.csv", index=False)
    pd.concat(silhouette_tables, ignore_index=True).to_csv(args.output_dir / "representation_silhouette_scores.csv", index=False)

    plot_umap_comparison(
        {
            "multiclass": umap_tables["multiclass"],
            "pairwise_raw": umap_tables["pairwise_raw"],
            "pairwise_shrunk": umap_tables["pairwise_shrunk"],
        },
        output_path=args.figures_dir / "multiclass_vs_pairwise_umap.png",
    )

    summary = {
        "aligned_rows": alignment_counts,
        "multiclass_path": str(args.multiclass_path),
        "pairwise_dir": str(args.pairwise_dir),
    }
    with open(args.output_dir / "comparison_manifest.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(args.output_dir)
    print(args.figures_dir)


if __name__ == "__main__":
    main()
