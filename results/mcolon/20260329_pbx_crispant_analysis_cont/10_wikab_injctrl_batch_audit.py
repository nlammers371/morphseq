from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_batch_audit_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

KEY_COLS = ["embryo_id", "genotype", "experiment_id", "time_bin_center"]
TARGET_GENOTYPES = ["inj_ctrl", "wik_ab"]
GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#808080",
}
EXPERIMENT_COLORS = {
    "20260304": "#1b9e77",
    "20260306": "#d95f02",
}
STAGE_ORDER = ["vector_space", "aligned_umap_init", "condensed_final"]
SCOPE_ORDER = ["pooled", "inj_ctrl_only", "wik_ab_only"]
BATCH_PRESENT_AUROC = 0.65
BATCH_STRONG_AUROC = 0.75
AMPLIFICATION_DELTA = 0.05
MIN_CLASS_COUNT = 3
DEFAULT_MULTICLASS_CONDENSATION = (
    REPO_ROOT
    / "results"
    / "mcolon"
    / "20260329_pbx_crispant_analysis_cont"
    / "results"
    / "force_calibration_v1"
    / "pbx_condensation_v1"
    / "condensed_positions.npz"
)


@dataclass
class RepresentationStage:
    representation: str
    stage: str
    df: pd.DataFrame
    feature_cols: list[str]


@dataclass
class StageSet:
    representation: str
    stages: dict[str, RepresentationStage]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit batch effects in wik_ab vs inj_ctrl negative-control geometry.")
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
        "--pairwise-raw-condensation",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "pairwise_raw_condensation_aligned_umap_bin4_perm500" / "condensed_positions.npz",
    )
    parser.add_argument(
        "--pairwise-shrunk-condensation",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "pairwise_shrunk_condensation_aligned_umap_bin4_perm500" / "condensed_positions.npz",
    )
    parser.add_argument(
        "--multiclass-condensation",
        type=Path,
        default=DEFAULT_MULTICLASS_CONDENSATION,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "wikab_injctrl_batch_audit_bin4_perm500",
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "shared" / "wikab_injctrl_batch_audit_bin4_perm500",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--min-class-count", type=int, default=MIN_CLASS_COUNT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_vector_table(path: Path, prefixes: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = set(KEY_COLS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {path}")
    feature_cols = [c for c in df.columns if c.startswith(prefixes)]
    if prefixes == ("",):
        feature_cols = [c for c in feature_cols if "__vs__" in c]
    if not feature_cols:
        raise ValueError(f"No feature columns found in {path}")
    df = df[df["genotype"].isin(TARGET_GENOTYPES)].copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    df["time_bin_center"] = pd.to_numeric(df["time_bin_center"], errors="coerce")
    df = df.dropna(subset=["time_bin_center"]).reset_index(drop=True)
    return df, feature_cols


def maybe_apply_smoke_subset(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    keep_bins = sorted(df["time_bin_center"].unique())[:3]
    return df[df["time_bin_center"].isin(keep_bins)].copy().reset_index(drop=True)


def load_condensation_stage_tables(
    npz_path: Path,
    *,
    representation: str,
    metadata: pd.DataFrame,
) -> dict[str, RepresentationStage]:
    if not npz_path.exists():
        return {}
    data = np.load(npz_path, allow_pickle=True)
    meta = metadata[["embryo_id", "genotype", "experiment_id"]].drop_duplicates()
    stage_map = {
        "aligned_umap_init": data["x0"],
        "condensed_final": data["positions"],
    }
    rows_by_stage: dict[str, RepresentationStage] = {}
    for stage_name, arr in stage_map.items():
        rows = []
        embryo_ids = data["embryo_ids"]
        labels = data["labels"]
        mask = data["mask"]
        time_values = data["time_values"]
        for i, embryo_id in enumerate(embryo_ids):
            for t, hpf in enumerate(time_values):
                if not bool(mask[i, t]):
                    continue
                rows.append(
                    {
                        "embryo_id": str(embryo_id),
                        "genotype": str(labels[i]),
                        "time_bin_center": float(hpf),
                        "dim_1": float(arr[i, t, 0]),
                        "dim_2": float(arr[i, t, 1]),
                    }
                )
        stage_df = pd.DataFrame(rows)
        if stage_df.empty:
            continue
        stage_df = stage_df[stage_df["genotype"].isin(TARGET_GENOTYPES)].copy()
        stage_df = stage_df.merge(meta, on=["embryo_id", "genotype"], how="left", validate="many_to_one")
        stage_df = stage_df.dropna(subset=["experiment_id"]).reset_index(drop=True)
        rows_by_stage[stage_name] = RepresentationStage(
            representation=representation,
            stage=stage_name,
            df=stage_df,
            feature_cols=["dim_1", "dim_2"],
        )
    return rows_by_stage


def load_stage_sets(args: argparse.Namespace) -> dict[str, StageSet]:
    pairwise_raw_df, pairwise_raw_cols = load_vector_table(args.pairwise_dir / "pairwise_raw_vectors.csv", prefixes=("",))
    pairwise_shrunk_df, pairwise_shrunk_cols = load_vector_table(args.pairwise_dir / "pairwise_shrunk_vectors.csv", prefixes=("",))
    multiclass_df, multiclass_cols = load_vector_table(args.multiclass_path, prefixes=("pred_proba_", "p_"))

    pairwise_raw_df = maybe_apply_smoke_subset(pairwise_raw_df, args.smoke)
    pairwise_shrunk_df = maybe_apply_smoke_subset(pairwise_shrunk_df, args.smoke)
    multiclass_df = maybe_apply_smoke_subset(multiclass_df, args.smoke)

    stage_sets = {
        "pairwise_raw": StageSet(
            representation="pairwise_raw",
            stages={
                "vector_space": RepresentationStage("pairwise_raw", "vector_space", pairwise_raw_df, pairwise_raw_cols),
                **load_condensation_stage_tables(
                    args.pairwise_raw_condensation,
                    representation="pairwise_raw",
                    metadata=pairwise_raw_df,
                ),
            },
        ),
        "pairwise_shrunk": StageSet(
            representation="pairwise_shrunk",
            stages={
                "vector_space": RepresentationStage("pairwise_shrunk", "vector_space", pairwise_shrunk_df, pairwise_shrunk_cols),
                **load_condensation_stage_tables(
                    args.pairwise_shrunk_condensation,
                    representation="pairwise_shrunk",
                    metadata=pairwise_shrunk_df,
                ),
            },
        ),
        "multiclass": StageSet(
            representation="multiclass",
            stages={
                "vector_space": RepresentationStage("multiclass", "vector_space", multiclass_df, multiclass_cols),
                **load_condensation_stage_tables(
                    args.multiclass_condensation,
                    representation="multiclass",
                    metadata=multiclass_df,
                ),
            },
        ),
    }
    return stage_sets


def build_support_table(stage_sets: dict[str, StageSet]) -> pd.DataFrame:
    vector_df = stage_sets["pairwise_raw"].stages["vector_space"].df.copy()
    rows = []
    for (genotype, experiment_id), group in vector_df.groupby(["genotype", "experiment_id"]):
        rows.append(
            {
                "genotype": genotype,
                "experiment_id": experiment_id,
                "n_rows": int(len(group)),
                "n_embryos": int(group["embryo_id"].nunique()),
                "n_time_bins": int(group["time_bin_center"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["genotype", "experiment_id"]).reset_index(drop=True)


def align_stage_rows(stage_set: StageSet) -> tuple[dict[str, RepresentationStage], pd.DataFrame]:
    common = None
    native_rows = []
    for stage_name, stage in stage_set.stages.items():
        keys = stage.df[KEY_COLS].drop_duplicates()
        native_rows.append(
            {
                "representation": stage_set.representation,
                "stage": stage_name,
                "native_rows": int(len(stage.df)),
                "native_embryos": int(stage.df["embryo_id"].nunique()),
            }
        )
        common = keys if common is None else common.merge(keys, on=KEY_COLS, how="inner")
    assert common is not None
    aligned = {}
    for stage_name, stage in stage_set.stages.items():
        df = common.merge(stage.df, on=KEY_COLS, how="left", validate="one_to_one")
        aligned[stage_name] = RepresentationStage(stage.representation, stage.stage, df, stage.feature_cols)
    counts = pd.DataFrame(native_rows)
    counts["aligned_rows"] = int(len(common))
    counts["aligned_embryos"] = int(common["embryo_id"].nunique())
    return aligned, counts


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "pooled":
        return pd.Series(True, index=df.index)
    if scope == "inj_ctrl_only":
        return df["genotype"].eq("inj_ctrl")
    if scope == "wik_ab_only":
        return df["genotype"].eq("wik_ab")
    raise ValueError(scope)


def can_score(y: np.ndarray, min_class_count: int, n_splits: int) -> bool:
    values, counts = np.unique(y, return_counts=True)
    return len(values) == 2 and counts.min() >= max(min_class_count, n_splits)


def cross_validated_auroc(X: np.ndarray, y: np.ndarray, *, n_splits: int, random_state: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probs = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state))
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return float(roc_auc_score(y, probs))


def permutation_auroc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[np.ndarray, float, float, float]:
    observed = cross_validated_auroc(X, y, n_splits=n_splits, random_state=random_state)
    rng = np.random.default_rng(random_state)
    null_scores = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(y)
        null_scores[i] = cross_validated_auroc(X, perm, n_splits=n_splits, random_state=random_state + i + 1)
    pval = float((1.0 + np.sum(null_scores >= observed)) / (1.0 + len(null_scores)))
    return null_scores, observed, float(null_scores.mean()), pval


def bh_qvalues(df: pd.DataFrame, p_col: str) -> pd.Series:
    pvals = df[p_col].to_numpy(dtype=float)
    order = np.argsort(np.nan_to_num(pvals, nan=1.0))
    ranked = pvals[order]
    n = len(ranked)
    qvals = np.full(n, np.nan, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        p = ranked[i]
        if math.isnan(p):
            val = np.nan
        else:
            val = min(prev, p * n / (i + 1))
            prev = val
        qvals[i] = val
    out = np.empty(n, dtype=float)
    out[order] = qvals
    return pd.Series(out, index=df.index)


def run_batch_predictability(
    aligned_stage_sets: dict[str, dict[str, RepresentationStage]],
    *,
    n_splits: int,
    n_permutations: int,
    min_class_count: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = []
    time_rows = []
    for representation, stages in aligned_stage_sets.items():
        for stage_name, stage in stages.items():
            base_df = stage.df.copy()
            for scope in SCOPE_ORDER:
                scope_df = base_df.loc[scope_mask(base_df, scope)].copy()
                X = scope_df[stage.feature_cols].to_numpy(dtype=float)
                y = pd.Categorical(scope_df["experiment_id"]).codes
                if can_score(y, min_class_count, n_splits):
                    null_scores, observed, null_mean, pval = permutation_auroc(
                        X,
                        y,
                        n_splits=n_splits,
                        n_permutations=n_permutations,
                        random_state=random_state,
                    )
                    null_std = float(np.std(null_scores, ddof=1)) if len(null_scores) > 1 else 0.0
                else:
                    observed = np.nan
                    null_mean = np.nan
                    null_std = np.nan
                    pval = np.nan
                global_rows.append(
                    {
                        "representation": representation,
                        "stage": stage_name,
                        "scope": scope,
                        "n_rows": int(len(scope_df)),
                        "n_embryos": int(scope_df["embryo_id"].nunique()),
                        "n_time_bins": int(scope_df["time_bin_center"].nunique()),
                        "auroc_obs": observed,
                        "auroc_null_mean": null_mean,
                        "auroc_null_std": null_std,
                        "pval": pval,
                        "n_permutations": int(n_permutations),
                    }
                )
                for time_bin_center, time_df in scope_df.groupby("time_bin_center"):
                    X_t = time_df[stage.feature_cols].to_numpy(dtype=float)
                    y_t = pd.Categorical(time_df["experiment_id"]).codes
                    if can_score(y_t, min_class_count, n_splits):
                        null_scores_t, observed_t, null_mean_t, pval_t = permutation_auroc(
                            X_t,
                            y_t,
                            n_splits=n_splits,
                            n_permutations=n_permutations,
                            random_state=random_state,
                        )
                        null_std_t = float(np.std(null_scores_t, ddof=1)) if len(null_scores_t) > 1 else 0.0
                    else:
                        observed_t = np.nan
                        null_mean_t = np.nan
                        null_std_t = np.nan
                        pval_t = np.nan
                    time_rows.append(
                        {
                            "representation": representation,
                            "stage": stage_name,
                            "scope": scope,
                            "time_bin_center": float(time_bin_center),
                            "n_rows": int(len(time_df)),
                            "n_embryos": int(time_df["embryo_id"].nunique()),
                            "auroc_obs": observed_t,
                            "auroc_null_mean": null_mean_t,
                            "auroc_null_std": null_std_t,
                            "pval": pval_t,
                            "n_permutations": int(n_permutations),
                        }
                    )
    global_df = pd.DataFrame(global_rows)
    time_df = pd.DataFrame(time_rows)
    if not global_df.empty:
        global_df["qval"] = bh_qvalues(global_df, "pval")
    if not time_df.empty:
        time_df["qval"] = bh_qvalues(time_df, "pval")
    return global_df, time_df


def run_knn_mixing(
    aligned_stage_sets: dict[str, dict[str, RepresentationStage]],
    *,
    knn_k: int,
) -> pd.DataFrame:
    rows = []
    for representation, stages in aligned_stage_sets.items():
        for stage_name, stage in stages.items():
            for scope in SCOPE_ORDER:
                df = stage.df.loc[scope_mask(stage.df, scope)].copy()
                if len(df) < 3 or df["experiment_id"].nunique() < 2:
                    fraction = np.nan
                    k_eff = np.nan
                else:
                    k_eff = min(knn_k, len(df) - 1)
                    nbrs = NearestNeighbors(n_neighbors=k_eff + 1)
                    nbrs.fit(df[stage.feature_cols].to_numpy(dtype=float))
                    idx = nbrs.kneighbors(return_distance=False)[:, 1:]
                    experiments = df["experiment_id"].to_numpy()
                    same = experiments[idx] == experiments[:, None]
                    fraction = float(same.mean())
                rows.append(
                    {
                        "representation": representation,
                        "stage": stage_name,
                        "scope": scope,
                        "n_rows": int(len(df)),
                        "same_experiment_fraction": fraction,
                        "k_eff": k_eff,
                    }
                )
    return pd.DataFrame(rows)


def run_centroid_separation(aligned_stage_sets: dict[str, dict[str, RepresentationStage]]) -> pd.DataFrame:
    rows = []
    for representation, stages in aligned_stage_sets.items():
        for stage_name, stage in stages.items():
            for genotype in TARGET_GENOTYPES:
                geno_df = stage.df[stage.df["genotype"] == genotype].copy()
                for time_bin_center, time_df in geno_df.groupby("time_bin_center"):
                    grouped = {exp: grp for exp, grp in time_df.groupby("experiment_id")}
                    if len(grouped) != 2:
                        continue
                    exps = sorted(grouped)
                    centroids = {}
                    dispersions = {}
                    for exp_id, grp in grouped.items():
                        X = grp[stage.feature_cols].to_numpy(dtype=float)
                        centroid = np.median(X, axis=0)
                        centroids[exp_id] = centroid
                        dispersions[exp_id] = float(np.median(np.linalg.norm(X - centroid[None, :], axis=1)))
                    distance = float(np.linalg.norm(centroids[exps[0]] - centroids[exps[1]]))
                    pooled_dispersion = float(np.mean([dispersions[e] for e in exps]))
                    ratio = float(distance / pooled_dispersion) if pooled_dispersion > 0 else np.nan
                    rows.append(
                        {
                            "representation": representation,
                            "stage": stage_name,
                            "genotype": genotype,
                            "time_bin_center": float(time_bin_center),
                            "experiment_1": exps[0],
                            "experiment_2": exps[1],
                            "centroid_distance": distance,
                            "within_dispersion_mean": pooled_dispersion,
                            "separation_ratio": ratio,
                            "n_rows": int(len(time_df)),
                        }
                    )
    return pd.DataFrame(rows)


def pca_coords(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if len(df) < 2:
        coords = np.zeros((len(df), 2), dtype=float)
    else:
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(df[feature_cols].to_numpy(dtype=float))
    out = df[KEY_COLS].copy()
    out["pc1"] = coords[:, 0] if len(coords) else []
    out["pc2"] = coords[:, 1] if len(coords) else []
    return out


def plot_support_heatmap(support_df: pd.DataFrame, output_path: Path) -> None:
    pivot = support_df.pivot(index="genotype", columns="experiment_id", values="n_embryos").reindex(index=TARGET_GENOTYPES)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=list(pivot.index))
    ax.set_title("Embryo support by genotype and experiment", fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _scatter_panel(ax, df: pd.DataFrame, x: str, y: str, *, color_col: str, title: str) -> None:
    palette = EXPERIMENT_COLORS if color_col == "experiment_id" else GENOTYPE_COLORS
    for key, grp in df.groupby(color_col):
        ax.scatter(grp[x], grp[y], s=18, alpha=0.75, color=palette.get(str(key), "#666666"), label=str(key))
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def plot_vector_pca(aligned_stage_sets: dict[str, dict[str, RepresentationStage]], output_path: Path, *, color_col: str) -> None:
    reps = ["multiclass", "pairwise_raw", "pairwise_shrunk"]
    fig, axes = plt.subplots(1, len(reps), figsize=(4.8 * len(reps), 4.2))
    for ax, rep in zip(np.atleast_1d(axes), reps):
        stage = aligned_stage_sets[rep]["vector_space"]
        coords = pca_coords(stage.df, stage.feature_cols)
        _scatter_panel(ax, coords, "pc1", "pc2", color_col=color_col, title=f"{rep} vector PCA")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_init_final(aligned_stage_sets: dict[str, dict[str, RepresentationStage]], output_path: Path, *, color_col: str) -> None:
    rows = []
    for rep in ["pairwise_raw", "pairwise_shrunk", "multiclass"]:
        for stage_name in ["aligned_umap_init", "condensed_final"]:
            if stage_name in aligned_stage_sets[rep]:
                rows.append((rep, stage_name))
    if not rows:
        return
    ncols = 2
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.8 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, (rep, stage_name) in zip(axes, rows):
        stage = aligned_stage_sets[rep][stage_name]
        plot_df = stage.df[[*KEY_COLS, *stage.feature_cols]].copy()
        plot_df = plot_df.rename(columns={stage.feature_cols[0]: "dim1", stage.feature_cols[1]: "dim2"})
        _scatter_panel(ax, plot_df, "dim1", "dim2", color_col=color_col, title=f"{rep} {stage_name}")
    for ax in axes[len(rows):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_auroc_over_time(time_df: pd.DataFrame, output_path: Path) -> None:
    pooled = time_df[time_df["scope"] == "pooled"].copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for (rep, stage), grp in pooled.groupby(["representation", "stage"]):
        grp = grp.sort_values("time_bin_center")
        ax.plot(grp["time_bin_center"], grp["auroc_obs"], marker="o", ms=3, lw=1.4, label=f"{rep}:{stage}")
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("time_bin_center")
    ax.set_ylabel("batch AUROC")
    ax.set_title("Batch predictability over time (pooled negative control)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stage_summary(global_df: pd.DataFrame, knn_df: pd.DataFrame, output_path: Path) -> None:
    pooled_auc = global_df[global_df["scope"] == "pooled"].copy()
    pooled_knn = knn_df[knn_df["scope"] == "pooled"].copy()
    pooled_auc["label"] = pooled_auc["representation"] + ":" + pooled_auc["stage"]
    pooled_knn["label"] = pooled_knn["representation"] + ":" + pooled_knn["stage"]
    order = pooled_auc["label"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(pooled_auc["label"], pooled_auc["auroc_obs"], color="#4C72B0")
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_title("Global batch AUROC", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].bar(pooled_knn["label"], pooled_knn["same_experiment_fraction"], color="#DD8452")
    axes[1].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_title("kNN same-experiment fraction", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def classify_audit(global_df: pd.DataFrame, knn_df: pd.DataFrame) -> dict[str, object]:
    pooled = global_df[global_df["scope"] == "pooled"].copy()
    pooled_knn = knn_df[knn_df["scope"] == "pooled"].copy()
    def fetch(rep: str, stage: str, table: pd.DataFrame, col: str) -> float:
        sub = table[(table["representation"] == rep) & (table["stage"] == stage)]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0][col])
    decisions = {
        "vector_space_batch_present": False,
        "aligned_umap_amplified": False,
        "condensation_amplified": False,
        "tree_work_blocked": True,
        "evidence": [],
    }
    vector_scores = []
    init_deltas = []
    final_deltas = []
    for rep in ["multiclass", "pairwise_raw", "pairwise_shrunk"]:
        vector = fetch(rep, "vector_space", pooled, "auroc_obs")
        init = fetch(rep, "aligned_umap_init", pooled, "auroc_obs")
        final = fetch(rep, "condensed_final", pooled, "auroc_obs")
        vector_q = fetch(rep, "vector_space", pooled, "qval")
        if not math.isnan(vector):
            vector_scores.append(vector)
        if not math.isnan(vector) and not math.isnan(init):
            init_deltas.append(init - vector)
        if not math.isnan(init) and not math.isnan(final):
            final_deltas.append(final - init)
        if not math.isnan(vector) and not math.isnan(vector_q) and vector >= BATCH_PRESENT_AUROC and vector_q <= 0.05:
            decisions["vector_space_batch_present"] = True
            decisions["evidence"].append(f"{rep} vector space AUROC={vector:.3f}, q={vector_q:.3g}")
        init_knn = fetch(rep, "aligned_umap_init", pooled_knn, "same_experiment_fraction")
        vector_knn = fetch(rep, "vector_space", pooled_knn, "same_experiment_fraction")
        if not math.isnan(vector) and not math.isnan(init) and not math.isnan(vector_knn) and not math.isnan(init_knn):
            if init - vector >= AMPLIFICATION_DELTA and init_knn > vector_knn:
                decisions["aligned_umap_amplified"] = True
                decisions["evidence"].append(f"{rep} init AUROC increase {init - vector:.3f}")
        final_knn = fetch(rep, "condensed_final", pooled_knn, "same_experiment_fraction")
        if not math.isnan(init) and not math.isnan(final) and not math.isnan(init_knn) and not math.isnan(final_knn):
            if final - init >= AMPLIFICATION_DELTA and final_knn > init_knn:
                decisions["condensation_amplified"] = True
                decisions["evidence"].append(f"{rep} final AUROC increase {final - init:.3f}")
    decisions["tree_work_blocked"] = bool(
        decisions["vector_space_batch_present"] or decisions["aligned_umap_amplified"] or decisions["condensation_amplified"]
    )
    decisions["max_vector_auroc"] = max(vector_scores) if vector_scores else np.nan
    decisions["max_init_delta"] = max(init_deltas) if init_deltas else np.nan
    decisions["max_final_delta"] = max(final_deltas) if final_deltas else np.nan
    return decisions


def write_summary(
    output_path: Path,
    *,
    support_df: pd.DataFrame,
    global_df: pd.DataFrame,
    knn_df: pd.DataFrame,
    decisions: dict[str, object],
    aligned_counts: pd.DataFrame,
) -> None:
    pooled = global_df[global_df["scope"] == "pooled"].copy().sort_values(["representation", "stage"])
    pooled_knn = knn_df[knn_df["scope"] == "pooled"].copy().sort_values(["representation", "stage"])
    lines = [
        "# Wik_ab vs Inj_ctrl Batch Audit",
        "",
        "## Support",
        support_df.to_markdown(index=False),
        "",
        "## Aligned row counts",
        aligned_counts.to_markdown(index=False),
        "",
        "## Pooled batch predictability",
        pooled[["representation", "stage", "n_rows", "auroc_obs", "auroc_null_mean", "pval", "qval"]].to_markdown(index=False),
        "",
        "## Pooled kNN mixing",
        pooled_knn[["representation", "stage", "same_experiment_fraction", "k_eff"]].to_markdown(index=False),
        "",
        "## Decision summary",
        f"1. Is batch structure already visible in the saved vector spaces? {'Yes' if decisions['vector_space_batch_present'] else 'No'}.",
        f"2. Does AlignedUMAP initialization increase experiment separation? {'Yes' if decisions['aligned_umap_amplified'] else 'No'}.",
        f"3. Does condensation further increase experiment separation? {'Yes' if decisions['condensation_amplified'] else 'No'}.",
        "",
        "Evidence:",
    ]
    evidence = decisions.get("evidence", [])
    if evidence:
        lines.extend([f"- {item}" for item in evidence])
    else:
        lines.append("- No threshold-triggering batch signal was detected.")
    lines.extend(
        [
            "",
            "Recommendation:",
            f"- {'tree work still blocked' if decisions['tree_work_blocked'] else 'tree work can proceed with caveats'}",
        ]
    )
    output_path.write_text("\n".join(lines))


def save_stage_exports(aligned_stage_sets: dict[str, dict[str, RepresentationStage]], output_dir: Path) -> None:
    rows = []
    for representation, stages in aligned_stage_sets.items():
        for stage_name, stage in stages.items():
            df = stage.df[KEY_COLS + stage.feature_cols].copy()
            df.insert(0, "stage", stage_name)
            df.insert(0, "representation", representation)
            rows.append(df)
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "stage_coordinates_long.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_permutations = 50
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage_sets = load_stage_sets(args)
    support_df = build_support_table(stage_sets)
    aligned_stage_sets = {}
    aligned_counts = []
    for representation, stage_set in stage_sets.items():
        aligned, counts = align_stage_rows(stage_set)
        aligned_stage_sets[representation] = aligned
        aligned_counts.append(counts)
    aligned_counts_df = pd.concat(aligned_counts, ignore_index=True)

    global_df, time_df = run_batch_predictability(
        aligned_stage_sets,
        n_splits=int(args.n_splits),
        n_permutations=int(args.n_permutations),
        min_class_count=int(args.min_class_count),
        random_state=int(args.random_state),
    )
    knn_df = run_knn_mixing(aligned_stage_sets, knn_k=int(args.knn_k))
    centroid_df = run_centroid_separation(aligned_stage_sets)
    decisions = classify_audit(global_df, knn_df)

    support_df.to_csv(args.output_dir / "support_by_genotype_experiment.csv", index=False)
    aligned_counts_df.to_csv(args.output_dir / "aligned_row_counts.csv", index=False)
    global_df.to_csv(args.output_dir / "batch_predictability_global.csv", index=False)
    time_df.to_csv(args.output_dir / "batch_predictability_by_timebin.csv", index=False)
    knn_df.to_csv(args.output_dir / "batch_knn_mixing.csv", index=False)
    centroid_df.to_csv(args.output_dir / "experiment_centroid_separation.csv", index=False)
    save_stage_exports(aligned_stage_sets, args.output_dir)

    plot_support_heatmap(support_df, args.output_dir / "support_heatmap.png")
    plot_vector_pca(aligned_stage_sets, args.output_dir / "vector_pca_by_experiment.png", color_col="experiment_id")
    plot_vector_pca(aligned_stage_sets, args.output_dir / "vector_pca_by_genotype.png", color_col="genotype")
    plot_init_final(aligned_stage_sets, args.output_dir / "init_final_scatter_by_experiment.png", color_col="experiment_id")
    plot_init_final(aligned_stage_sets, args.output_dir / "init_final_scatter_by_genotype.png", color_col="genotype")
    plot_auroc_over_time(time_df, args.output_dir / "batch_auroc_over_time.png")
    plot_stage_summary(global_df, knn_df, args.output_dir / "batch_stage_summary.png")
    write_summary(
        args.output_dir / "AUDIT_SUMMARY.md",
        support_df=support_df,
        global_df=global_df,
        knn_df=knn_df,
        decisions=decisions,
        aligned_counts=aligned_counts_df,
    )

    manifest = {
        "target_genotypes": TARGET_GENOTYPES,
        "n_permutations": int(args.n_permutations),
        "n_splits": int(args.n_splits),
        "knn_k": int(args.knn_k),
        "smoke": bool(args.smoke),
        "representations": {
            rep: sorted(stages.keys()) for rep, stages in aligned_stage_sets.items()
        },
        "vector_space_batch_present": bool(decisions["vector_space_batch_present"]),
        "aligned_umap_amplified": bool(decisions["aligned_umap_amplified"]),
        "condensation_amplified": bool(decisions["condensation_amplified"]),
        "tree_work_blocked": bool(decisions["tree_work_blocked"]),
    }
    with open(args.output_dir / "audit_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    if args.shared_dir and not args.smoke:
        args.shared_dir.parent.mkdir(parents=True, exist_ok=True)
        if args.shared_dir.exists():
            shutil.rmtree(args.shared_dir)
        shutil.copytree(args.output_dir, args.shared_dir)

    print(args.output_dir)
    if args.shared_dir and not args.smoke:
        print(args.shared_dir)


if __name__ == "__main__":
    main()
