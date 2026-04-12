from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import warnings
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_lineage_audit_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification.engine.comparison_resolution import resolve_comparisons
from analyze.classification.engine.contrast_coordinates import assemble_contrast_coordinates
from analyze.classification.engine.loop import (
    _bin_and_aggregate,
    _build_binary_labels,
    _collect_binary_margins,
    _collect_scores,
    _resolve_feature_columns,
    _run_binary_classification_loop,
)
from common import BUILD06_DIR, CURRENT_EXPERIMENT_IDS, normalize_genotype

KEY_COLS = ["embryo_id", "genotype", "experiment_id", "time_bin_center"]
TARGET_GENOTYPES = ["inj_ctrl", "wik_ab"]
SCOPE_ORDER = ["pooled", "inj_ctrl_only", "wik_ab_only"]
STAGE_ORDER = [
    "build06_embeddings",
    "binned_embeddings",
    "m_raw_injctrl_vs_wikab",
    "raw_coordinates",
    "shrunk_coordinates",
]
GENOTYPE_COLORS = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#808080",
}
EXPERIMENT_COLORS = {
    "20260304": "#1b9e77",
    "20260306": "#d95f02",
}
BATCH_PRESENT_AUROC = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace where experiment separation enters the wik_ab vs inj_ctrl lineage.")
    parser.add_argument(
        "--pairwise-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_pairwise_bin4_perm500",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "wikab_injctrl_lineage_audit_bin4_perm500",
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "shared" / "wikab_injctrl_lineage_audit_bin4_perm500",
    )
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_build06_current() -> pd.DataFrame:
    frames = []
    for exp_id in CURRENT_EXPERIMENT_IDS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{exp_id}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_csv(path, low_memory=False)
        if "experiment_id" in part.columns:
            part = part[part["experiment_id"].astype(str) == exp_id].copy()
        else:
            part["experiment_id"] = exp_id
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()
    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(normalize_genotype)
    df = df[df["genotype"].isin(TARGET_GENOTYPES)].copy()
    predicted = pd.to_numeric(df.get("predicted_stage_hpf"), errors="coerce")
    start_age = pd.to_numeric(df.get("start_age_hpf"), errors="coerce")
    relative_time_hpf = pd.to_numeric(df.get("relative_time_s"), errors="coerce") / 3600.0
    df["stage_hpf"] = predicted.where(predicted.notna(), start_age + relative_time_hpf)
    df = df[df["stage_hpf"].notna()].copy()
    df["experiment_id"] = df["experiment_id"].astype(str)
    return df.reset_index(drop=True)


def maybe_apply_smoke_subset_rows(df: pd.DataFrame, *, bin_width: float, enabled: bool) -> pd.DataFrame:
    if not enabled:
        return df
    time_bins = (np.floor(df["stage_hpf"] / bin_width) * bin_width).astype(int)
    keep_bins = sorted(time_bins.unique())[:3]
    return df.loc[time_bins.isin(keep_bins)].reset_index(drop=True)


def scope_mask(df: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "pooled":
        return pd.Series(True, index=df.index)
    if scope == "inj_ctrl_only":
        return df["genotype"].eq("inj_ctrl")
    if scope == "wik_ab_only":
        return df["genotype"].eq("wik_ab")
    raise ValueError(scope)


def can_score(y: np.ndarray, n_splits: int) -> bool:
    vals, counts = np.unique(y, return_counts=True)
    return len(vals) == 2 and counts.min() >= n_splits


def cross_validated_auroc(X: np.ndarray, y: np.ndarray, *, n_splits: int, random_state: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probs = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state))
        model.fit(X[train_idx], y[train_idx])
        probs[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return float(roc_auc_score(y, probs))


def permutation_auroc(X: np.ndarray, y: np.ndarray, *, n_splits: int, n_permutations: int, random_state: int) -> tuple[float, float, float, float]:
    observed = cross_validated_auroc(X, y, n_splits=n_splits, random_state=random_state)
    rng = np.random.default_rng(random_state)
    null_scores = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_scores[i] = cross_validated_auroc(X, y_perm, n_splits=n_splits, random_state=random_state + i + 1)
    pval = float((1.0 + np.sum(null_scores >= observed)) / (1.0 + len(null_scores)))
    return observed, float(null_scores.mean()), float(np.std(null_scores, ddof=1)), pval


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


def compute_batch_metrics(stage_name: str, df: pd.DataFrame, feature_cols: list[str], *, n_splits: int, n_permutations: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rows = []
    time_rows = []
    for scope in SCOPE_ORDER:
        scope_df = df.loc[scope_mask(df, scope)].copy()
        X = scope_df[feature_cols].fillna(0.0).to_numpy(dtype=float)
        y = pd.Categorical(scope_df["experiment_id"]).codes
        if len(scope_df) and can_score(y, n_splits):
            obs, null_mean, null_std, pval = permutation_auroc(X, y, n_splits=n_splits, n_permutations=n_permutations, random_state=random_state)
        else:
            obs = null_mean = null_std = pval = np.nan
        global_rows.append({
            "stage": stage_name,
            "scope": scope,
            "n_rows": int(len(scope_df)),
            "n_embryos": int(scope_df["embryo_id"].nunique()),
            "n_time_bins": int(scope_df["time_bin_center"].nunique()),
            "auroc_obs": obs,
            "auroc_null_mean": null_mean,
            "auroc_null_std": null_std,
            "pval": pval,
            "n_permutations": int(n_permutations),
        })
        for time_bin_center, time_df in scope_df.groupby("time_bin_center"):
            X_t = time_df[feature_cols].fillna(0.0).to_numpy(dtype=float)
            y_t = pd.Categorical(time_df["experiment_id"]).codes
            if len(time_df) and can_score(y_t, n_splits):
                obs_t, null_mean_t, null_std_t, pval_t = permutation_auroc(X_t, y_t, n_splits=n_splits, n_permutations=n_permutations, random_state=random_state)
            else:
                obs_t = null_mean_t = null_std_t = pval_t = np.nan
            time_rows.append({
                "stage": stage_name,
                "scope": scope,
                "time_bin_center": float(time_bin_center),
                "n_rows": int(len(time_df)),
                "n_embryos": int(time_df["embryo_id"].nunique()),
                "auroc_obs": obs_t,
                "auroc_null_mean": null_mean_t,
                "auroc_null_std": null_std_t,
                "pval": pval_t,
                "n_permutations": int(n_permutations),
            })
    global_df = pd.DataFrame(global_rows)
    time_df = pd.DataFrame(time_rows)
    if not global_df.empty:
        global_df["qval"] = bh_qvalues(global_df, "pval")
    if not time_df.empty:
        time_df["qval"] = bh_qvalues(time_df, "pval")
    return global_df, time_df


def pca_coords(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if len(df) < 2:
        coords = np.zeros((len(df), 2), dtype=float)
    else:
        coords = PCA(n_components=2, random_state=0).fit_transform(df[feature_cols].fillna(0.0).to_numpy(dtype=float))
    out = df[KEY_COLS].copy()
    out["pc1"] = coords[:, 0] if len(coords) else []
    out["pc2"] = coords[:, 1] if len(coords) else []
    return out


def plot_stage_pca(stage_frames: dict[str, tuple[pd.DataFrame, list[str]]], output_path: Path, *, color_col: str) -> None:
    fig, axes = plt.subplots(1, len(stage_frames), figsize=(5.0 * len(stage_frames), 4.5), squeeze=False)
    axes = axes.ravel()
    palette = EXPERIMENT_COLORS if color_col == "experiment_id" else GENOTYPE_COLORS
    for ax, (stage_name, (df, feature_cols)) in zip(axes, stage_frames.items()):
        coords = pca_coords(df, feature_cols)
        for key, grp in coords.groupby(color_col):
            ax.scatter(grp["pc1"], grp["pc2"], s=18, alpha=0.75, color=palette.get(str(key), "#666666"), label=str(key))
        ax.set_title(stage_name.replace("_", " "), fontweight="bold", fontsize=10)
        ax.set_xlabel("pc1")
        ax.set_ylabel("pc2")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_margin_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, genotype in zip(axes, TARGET_GENOTYPES):
        sub = df[df["genotype"] == genotype].copy()
        for exp_id, grp in sub.groupby("experiment_id"):
            ax.hist(grp["m_raw"], bins=25, alpha=0.55, color=EXPERIMENT_COLORS.get(exp_id, "#666666"), label=exp_id)
        ax.set_title(genotype, fontweight="bold")
        ax.set_xlabel("m_raw")
        ax.set_ylabel("count")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_probe_level_shift(raw_df: pd.DataFrame, shrunk_df: pd.DataFrame, residual_df: pd.DataFrame, *, n_splits: int, random_state: int) -> pd.DataFrame:
    probe_cols = [c for c in raw_df.columns if c not in {"feature_set", "embryo_id", "genotype", "time_bin", "time_bin_center", "experiment_id"}]
    rows = []
    for scope in SCOPE_ORDER:
        raw_scope = raw_df.loc[scope_mask(raw_df, scope)].copy()
        shrunk_scope = shrunk_df.loc[scope_mask(shrunk_df, scope)].copy()
        resid_scope = residual_df.loc[scope_mask(residual_df, scope)].copy()
        y = pd.Categorical(raw_scope["experiment_id"]).codes
        for probe in probe_cols:
            for table_name, table in [("raw", raw_scope), ("shrunk", shrunk_scope), ("residual", resid_scope)]:
                vals = table[[probe]].fillna(0.0).to_numpy(dtype=float)
                if len(table) and can_score(y, n_splits):
                    auroc = cross_validated_auroc(vals, y, n_splits=n_splits, random_state=random_state)
                else:
                    auroc = np.nan
                rows.append({
                    "scope": scope,
                    "probe": probe,
                    "table": table_name,
                    "auroc_obs": auroc,
                })
    probe_df = pd.DataFrame(rows)
    wide = probe_df.pivot_table(index=["scope", "probe"], columns="table", values="auroc_obs").reset_index()
    for col in ["raw", "shrunk", "residual"]:
        if col not in wide.columns:
            wide[col] = np.nan
    wide["delta_shrunk_minus_raw"] = wide["shrunk"] - wide["raw"]
    wide["delta_residual_minus_raw"] = wide["residual"] - wide["raw"]
    return wide.sort_values(["scope", "delta_shrunk_minus_raw"], ascending=[True, False]).reset_index(drop=True)


def plot_probe_shift(probe_df: pd.DataFrame, output_path: Path) -> None:
    pooled = probe_df[probe_df["scope"] == "pooled"].copy().sort_values("delta_shrunk_minus_raw", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(pooled["probe"], pooled["delta_shrunk_minus_raw"], color="#4C72B0")
    ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1)
    ax.set_title("Probe-level batch AUROC change after shrinkage", fontweight="bold")
    ax.set_ylabel("shrunk - raw AUROC")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_lineage_tables(args: argparse.Namespace) -> tuple[dict[str, tuple[pd.DataFrame, list[str]]], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_build06_current()
    df = maybe_apply_smoke_subset_rows(df, bin_width=float(args.bin_width), enabled=bool(args.smoke))

    resolved_features = _resolve_feature_columns(df, {"vae": "z_mu_b"})
    feature_cols = resolved_features["vae"]
    build_df = df[["embryo_id", "genotype", "experiment_id", "stage_hpf", *feature_cols]].copy()
    build_df["time_bin_center"] = build_df["stage_hpf"].astype(float)
    build_stage = build_df[["embryo_id", "genotype", "experiment_id", "time_bin_center", *feature_cols]].copy().reset_index(drop=True)

    resolved = resolve_comparisons(
        positive="inj_ctrl",
        negative="wik_ab",
        comparisons=None,
        available_labels=set(df["genotype"].astype(str).unique()),
        class_col="genotype",
    )
    comparison = resolved[0]
    df_labeled = _build_binary_labels(df, "genotype", comparison)
    df_binned = _bin_and_aggregate(df_labeled, "embryo_id", "stage_hpf", feature_cols, float(args.bin_width))
    embryo_meta = df[["embryo_id", "genotype", "experiment_id"]].drop_duplicates().copy()
    binned_df = df_binned.merge(embryo_meta, on="embryo_id", how="left", validate="many_to_one")
    binned_stage = binned_df[["embryo_id", "genotype", "experiment_id", "time_bin_center", *feature_cols]].copy().reset_index(drop=True)

    bin_results = _run_binary_classification_loop(
        df_binned=df_binned,
        feature_cols=feature_cols,
        id_col="embryo_id",
        bin_width=float(args.bin_width),
        n_splits=int(args.n_splits),
        n_permutations=int(args.n_permutations),
        n_jobs=int(args.n_jobs),
        random_state=int(args.random_state),
        verbose=False,
    )
    scores = pd.DataFrame(_collect_scores(bin_results, comparison, "vae"))
    margin_rows = _collect_binary_margins(bin_results, comparison, "vae", "embryo_id")
    contrast_layers = assemble_contrast_coordinates(
        margin_rows,
        scores,
        df[["embryo_id", "genotype"]].drop_duplicates(),
        "embryo_id",
        "genotype",
    )

    margin_long = contrast_layers["raw_contrast_scores_long"].copy()
    margin_long = margin_long.merge(embryo_meta, on=["embryo_id", "genotype"], how="left", validate="many_to_one")
    margin_stage = margin_long[["embryo_id", "genotype", "experiment_id", "time_bin_center", "m_raw"]].copy().reset_index(drop=True)

    pairwise_dir = args.pairwise_dir
    raw_coords = pd.read_csv(pairwise_dir / "raw_coordinates.csv")
    shrunk_coords = pd.read_csv(pairwise_dir / "shrunk_coordinates.csv")
    residual_coords = pd.read_csv(pairwise_dir / "residual_coordinates.csv")
    specificity = pd.read_csv(pairwise_dir / "contrast_specificity_by_timebin.csv")

    raw_coords = raw_coords[raw_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    shrunk_coords = shrunk_coords[shrunk_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    residual_coords = residual_coords[residual_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    experiment_lookup = embryo_meta.set_index("embryo_id")["experiment_id"]
    for table in (raw_coords, shrunk_coords, residual_coords):
        table["embryo_id"] = table["embryo_id"].astype(str)
        table["time_bin_center"] = pd.to_numeric(table["time_bin_center"], errors="coerce")
        table.dropna(subset=["time_bin_center"], inplace=True)
        table["experiment_id"] = table["embryo_id"].map(experiment_lookup)
        table.dropna(subset=["experiment_id"], inplace=True)
    if args.smoke:
        keep_bins = sorted(raw_coords["time_bin_center"].unique())[:3]
        raw_coords = raw_coords[raw_coords["time_bin_center"].isin(keep_bins)].copy()
        shrunk_coords = shrunk_coords[shrunk_coords["time_bin_center"].isin(keep_bins)].copy()
        residual_coords = residual_coords[residual_coords["time_bin_center"].isin(keep_bins)].copy()
        specificity = specificity[specificity["time_bin_center"].isin(keep_bins)].copy()

    raw_feature_cols = [c for c in raw_coords.columns if "__vs__" in c]
    shrunk_feature_cols = [c for c in shrunk_coords.columns if "__vs__" in c]

    stage_frames = {
        "build06_embeddings": (build_stage, feature_cols),
        "binned_embeddings": (binned_stage, feature_cols),
        "m_raw_injctrl_vs_wikab": (margin_stage, ["m_raw"]),
        "raw_coordinates": (raw_coords[[*KEY_COLS, *raw_feature_cols]].copy(), raw_feature_cols),
        "shrunk_coordinates": (shrunk_coords[[*KEY_COLS, *shrunk_feature_cols]].copy(), shrunk_feature_cols),
    }
    return stage_frames, specificity, raw_coords, shrunk_coords, residual_coords


def summarize_first_stage(global_df: pd.DataFrame) -> str | None:
    pooled = global_df[(global_df["scope"] == "pooled") & (global_df["qval"] <= 0.05) & (global_df["auroc_obs"] >= BATCH_PRESENT_AUROC)].copy()
    if pooled.empty:
        return None
    order = {name: i for i, name in enumerate(STAGE_ORDER)}
    pooled["stage_order"] = pooled["stage"].map(order)
    pooled = pooled.sort_values(["stage_order", "auroc_obs"], ascending=[True, False])
    return str(pooled.iloc[0]["stage"])


def write_summary(output_path: Path, global_df: pd.DataFrame, probe_df: pd.DataFrame, first_stage: str | None) -> None:
    stage_order = {name: i for i, name in enumerate(STAGE_ORDER)}
    pooled = global_df[global_df["scope"] == "pooled"].copy()
    pooled["stage_order"] = pooled["stage"].map(stage_order)
    pooled = pooled.sort_values(["stage_order", "stage"]).drop(columns="stage_order")
    lines = [
        "# Wik_ab vs Inj_ctrl Lineage Audit",
        "",
        "## Pooled batch AUROC by stage",
        pooled[["stage", "auroc_obs", "auroc_null_mean", "pval", "qval"]].to_markdown(index=False),
        "",
        f"First stage with strong batch signal: `{first_stage}`" if first_stage else "No stage crossed the reporting threshold.",
        "",
        "## Interpretation",
    ]
    if first_stage == "build06_embeddings":
        lines.append("- The experiment separation is already present in the original build06 VAE embeddings, before classifier binning or pairwise coordinate assembly.")
    elif first_stage:
        lines.append(f"- The earliest strong experiment separation appears at `{first_stage}` rather than at ingestion.")
    else:
        lines.append("- No stage crossed the current reporting threshold.")
    pooled_probe = probe_df[probe_df["scope"] == "pooled"].copy().head(10)
    lines.extend([
        "",
        "## Top probes by shrinkage-induced AUROC increase",
        pooled_probe[["probe", "raw", "shrunk", "delta_shrunk_minus_raw"]].to_markdown(index=False),
    ])
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_permutations = 50
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage_frames, specificity, raw_coords, shrunk_coords, residual_coords = build_lineage_tables(args)

    global_tables = []
    time_tables = []
    for stage_name in STAGE_ORDER:
        stage_df, feature_cols = stage_frames[stage_name]
        g, t = compute_batch_metrics(
            stage_name,
            stage_df,
            feature_cols,
            n_splits=int(args.n_splits),
            n_permutations=int(args.n_permutations),
            random_state=int(args.random_state),
        )
        global_tables.append(g)
        time_tables.append(t)
    global_df = pd.concat(global_tables, ignore_index=True)
    time_df = pd.concat(time_tables, ignore_index=True)
    probe_df = compute_probe_level_shift(raw_coords, shrunk_coords, residual_coords, n_splits=int(args.n_splits), random_state=int(args.random_state))
    first_stage = summarize_first_stage(global_df)

    global_df.to_csv(args.output_dir / "lineage_batch_predictability_global.csv", index=False)
    time_df.to_csv(args.output_dir / "lineage_batch_predictability_by_timebin.csv", index=False)
    probe_df.to_csv(args.output_dir / "probe_shrinkage_batch_shift.csv", index=False)
    specificity.to_csv(args.output_dir / "contrast_specificity_by_timebin_subset.csv", index=False)
    pd.concat(
        [df.assign(stage=stage_name) for stage_name, (df, _) in stage_frames.items()],
        ignore_index=True,
    ).to_csv(args.output_dir / "lineage_stage_tables_long.csv", index=False)

    pca_stages = {k: stage_frames[k] for k in ["build06_embeddings", "binned_embeddings", "raw_coordinates", "shrunk_coordinates"]}
    plot_stage_pca(pca_stages, args.output_dir / "lineage_pca_by_experiment.png", color_col="experiment_id")
    plot_stage_pca(pca_stages, args.output_dir / "lineage_pca_by_genotype.png", color_col="genotype")
    plot_margin_distribution(stage_frames["m_raw_injctrl_vs_wikab"][0], args.output_dir / "margin_distribution_by_experiment.png")
    plot_probe_shift(probe_df, args.output_dir / "probe_shrinkage_batch_shift.png")
    write_summary(args.output_dir / "LINEAGE_SUMMARY.md", global_df, probe_df, first_stage)

    manifest = {
        "smoke": bool(args.smoke),
        "n_permutations": int(args.n_permutations),
        "first_stage_with_batch_signal": first_stage,
        "stages": STAGE_ORDER,
    }
    with open(args.output_dir / "lineage_manifest.json", "w") as handle:
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
