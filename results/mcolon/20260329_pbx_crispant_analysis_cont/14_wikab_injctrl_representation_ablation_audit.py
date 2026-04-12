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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_rep_ablation_audit_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from trajectory_cosmology import init_embedding

TARGET_GENOTYPES = ["inj_ctrl", "wik_ab"]
DIRECT_PROBE = "inj_ctrl__vs__wik_ab"
FOCAL_GENOTYPE = "pbx1b_pbx4_crispant"
KEY_COLS = ["embryo_id", "genotype", "experiment_id", "time_bin", "time_bin_center"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Representation ablation audit for within-bin inj_ctrl vs wik_ab separation.")
    parser.add_argument(
        "--pairwise-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_pairwise_bin4_perm500",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "wikab_injctrl_representation_ablation_audit_bin4_perm10",
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "shared" / "wikab_injctrl_representation_ablation_audit_bin4_perm10",
    )
    parser.add_argument("--anchor-bins", type=str, default="26,54,78")
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_anchor_bins(spec: str) -> list[float]:
    return [float(part.strip()) for part in spec.split(",") if part.strip()] or [26.0, 54.0, 78.0]


def can_score(y: np.ndarray, n_splits: int) -> bool:
    vals, counts = np.unique(y, return_counts=True)
    return len(vals) == 2 and counts.min() >= n_splits


def cross_validated_auroc(X: np.ndarray, y: np.ndarray, *, n_splits: int, random_state: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probs = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state),
        )
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


def nearest_available_bins(available: list[float], anchors: list[float]) -> list[float]:
    picked: list[float] = []
    for anchor in anchors:
        nearest = min(available, key=lambda v: abs(v - anchor))
        if nearest not in picked:
            picked.append(nearest)
    return picked


def load_pairwise_tables(pairwise_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_coords = pd.read_csv(pairwise_dir / "raw_coordinates.csv")
    shrunk_coords = pd.read_csv(pairwise_dir / "shrunk_coordinates.csv")
    probe_index = pd.read_csv(pairwise_dir / "probe_index.csv")
    for table in (raw_coords, shrunk_coords):
        table["embryo_id"] = table["embryo_id"].astype(str)
        table["experiment_id"] = table["embryo_id"].str.split("_", n=1).str[0]
        table["time_bin_center"] = pd.to_numeric(table["time_bin_center"], errors="coerce")
        table["time_bin"] = pd.to_numeric(table["time_bin"], errors="coerce").astype(int)
        table.dropna(subset=["time_bin_center"], inplace=True)
    raw_coords = raw_coords[raw_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    shrunk_coords = shrunk_coords[shrunk_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    return raw_coords, shrunk_coords, probe_index


def feature_subsets(probe_index: pd.DataFrame) -> dict[str, list[str]]:
    all_pairs = probe_index["column_name"].astype(str).tolist()
    direct = [DIRECT_PROBE]
    focal = probe_index.loc[
        (probe_index["positive_label"].astype(str) == FOCAL_GENOTYPE)
        | (probe_index["negative_label"].astype(str) == FOCAL_GENOTYPE),
        "column_name",
    ].astype(str).tolist()
    return {
        "direct": direct,
        "focal_vs_pbx1b_pbx4": focal,
        "all_pairs": all_pairs,
    }


def build_variant_df(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    keep = [c for c in columns if c in df.columns]
    return df[[*KEY_COLS, *keep]].copy(), keep


def support_aware_feature_cols(sub: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    keep: list[str] = []
    inj = sub[sub["genotype"] == "inj_ctrl"]
    wik = sub[sub["genotype"] == "wik_ab"]
    for col in feature_cols:
        if inj[col].notna().any() and wik[col].notna().any():
            keep.append(col)
    return keep


def compute_vector_anchor_metrics(name: str, df: pd.DataFrame, feature_cols: list[str], anchor_bins: list[float], *, n_splits: int, n_permutations: int, random_state: int) -> pd.DataFrame:
    available = sorted(df["time_bin_center"].dropna().unique().tolist())
    chosen = nearest_available_bins(available, anchor_bins)
    rows: list[dict[str, object]] = []
    for anchor, bin_center in zip(anchor_bins, chosen):
        sub = df[df["time_bin_center"] == bin_center].copy()
        supported_cols = support_aware_feature_cols(sub, feature_cols)
        y = sub["genotype"].map({"inj_ctrl": 0, "wik_ab": 1}).to_numpy(dtype=int)
        if len(sub) and supported_cols and can_score(y, n_splits):
            X = sub[supported_cols].to_numpy(dtype=float)
            obs, null_mean, null_std, pval = permutation_auroc(X, y, n_splits=n_splits, n_permutations=n_permutations, random_state=random_state)
        else:
            obs = null_mean = null_std = pval = np.nan
        rows.append({
            "variant": name,
            "stage": "vector_space",
            "requested_anchor": float(anchor),
            "time_bin_center": float(bin_center),
            "n_rows": int(len(sub)),
            "n_supported_features": int(len(supported_cols)),
            "n_inj_ctrl": int((sub["genotype"] == "inj_ctrl").sum()),
            "n_wik_ab": int((sub["genotype"] == "wik_ab").sum()),
            "auroc_obs": obs,
            "auroc_null_mean": null_mean,
            "auroc_null_std": null_std,
            "pval": pval,
            "n_permutations": int(n_permutations),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["qval"] = bh_qvalues(out, "pval")
    return out


def build_tensor(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    embryos = sorted(df["embryo_id"].unique().tolist())
    time_values = sorted(df["time_bin_center"].unique().tolist())
    embryo_idx = {eid: i for i, eid in enumerate(embryos)}
    time_idx = {t: i for i, t in enumerate(time_values)}
    features = np.full((len(embryos), len(time_values), len(feature_cols)), np.nan, dtype=float)
    mask = np.zeros((len(embryos), len(time_values)), dtype=bool)
    meta = df[["embryo_id", "genotype", "experiment_id"]].drop_duplicates().set_index("embryo_id")
    for _, row in df.iterrows():
        i = embryo_idx[row["embryo_id"]]
        t = time_idx[float(row["time_bin_center"])]
        features[i, t, :] = row[feature_cols].to_numpy(dtype=float)
        mask[i, t] = True
    meta_df = pd.DataFrame({
        "embryo_id": embryos,
        "genotype": [str(meta.loc[eid, "genotype"]) for eid in embryos],
        "experiment_id": [str(meta.loc[eid, "experiment_id"]) for eid in embryos],
    })
    return features, mask, np.array(time_values, dtype=float), meta_df


def compute_init_anchor_metrics(name: str, df: pd.DataFrame, feature_cols: list[str], anchor_bins: list[float], *, n_splits: int, n_permutations: int, random_state: int) -> pd.DataFrame:
    features, mask, time_values, meta = build_tensor(df, feature_cols)
    x0 = init_embedding.aligned_umap_init(
        features,
        mask,
        n_neighbors=15,
        min_dist=0.1,
        alignment_regularisation=1e-2,
        alignment_window_size=3,
        random_state=random_state,
    )
    chosen = nearest_available_bins(time_values.tolist(), anchor_bins)
    rows: list[dict[str, object]] = []
    for anchor, bin_center in zip(anchor_bins, chosen):
        t_idx = int(np.where(time_values == bin_center)[0][0])
        observed = mask[:, t_idx]
        coords = x0[observed, t_idx, :]
        sub = meta.loc[observed].reset_index(drop=True)
        supported_cols = support_aware_feature_cols(df[df["time_bin_center"] == bin_center].copy(), feature_cols)
        y = sub["genotype"].map({"inj_ctrl": 0, "wik_ab": 1}).to_numpy(dtype=int)
        if len(sub) and supported_cols and can_score(y, n_splits):
            obs, null_mean, null_std, pval = permutation_auroc(coords, y, n_splits=n_splits, n_permutations=n_permutations, random_state=random_state)
        else:
            obs = null_mean = null_std = pval = np.nan
        rows.append({
            "variant": name,
            "stage": "aligned_umap_init",
            "requested_anchor": float(anchor),
            "time_bin_center": float(bin_center),
            "n_rows": int(len(sub)),
            "n_supported_features": int(len(supported_cols)),
            "n_inj_ctrl": int((sub["genotype"] == "inj_ctrl").sum()),
            "n_wik_ab": int((sub["genotype"] == "wik_ab").sum()),
            "auroc_obs": obs,
            "auroc_null_mean": null_mean,
            "auroc_null_std": null_std,
            "pval": pval,
            "n_permutations": int(n_permutations),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["qval"] = bh_qvalues(out, "pval")
    return out


def plot_ablation(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, stage in zip(axes, ["vector_space", "aligned_umap_init"]):
        sub = summary_df[summary_df["stage"] == stage].copy()
        for variant, grp in sub.groupby("variant"):
            ax.plot(grp["time_bin_center"], grp["auroc_obs"], marker="o", linewidth=1.8, label=variant)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(stage.replace("_", " "), fontweight="bold")
        ax.set_xlabel("time bin center (hpf)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("inj_ctrl vs wik_ab AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(output_path: Path, summary_df: pd.DataFrame, subsets: dict[str, list[str]]) -> None:
    lines = [
        "# Wik_ab vs Inj_ctrl Representation Ablation Audit",
        "",
        "Variants:",
        f"- direct: {subsets['direct']}",
        f"- focal_vs_pbx1b_pbx4: {subsets['focal_vs_pbx1b_pbx4']}",
        f"- all_pairs: {subsets['all_pairs']}",
        "",
        "Vector-space AUROC is now computed only on probes with support in both `inj_ctrl` and `wik_ab` within the anchor bin, with train-fold mean imputation rather than zero-imputation.",
        "",
    ]
    for stage in ["vector_space", "aligned_umap_init"]:
        sub = summary_df[summary_df["stage"] == stage].copy().sort_values(["requested_anchor", "variant"])
        lines.append(f"## {stage}")
        cols = ["variant", "requested_anchor", "time_bin_center", "auroc_obs", "auroc_null_mean", "pval", "qval", "n_inj_ctrl", "n_wik_ab"]
        if "n_supported_features" in sub.columns:
            cols.insert(5, "n_supported_features")
        lines.append(sub[cols].to_markdown(index=False))
        lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_permutations = min(int(args.n_permutations), 5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor_bins = parse_anchor_bins(args.anchor_bins)

    raw_coords, shrunk_coords, probe_index = load_pairwise_tables(args.pairwise_dir)
    subsets = feature_subsets(probe_index)

    runs: list[pd.DataFrame] = []
    for family_name, cols in subsets.items():
        for rep_name, table in [("raw", raw_coords), ("shrunk", shrunk_coords)]:
            variant_name = f"{family_name}_{rep_name}"
            variant_df, feature_cols = build_variant_df(table, cols)
            if not feature_cols:
                continue
            runs.append(compute_vector_anchor_metrics(variant_name, variant_df, feature_cols, anchor_bins, n_splits=int(args.n_splits), n_permutations=int(args.n_permutations), random_state=int(args.random_state)))
            runs.append(compute_init_anchor_metrics(variant_name, variant_df, feature_cols, anchor_bins, n_splits=int(args.n_splits), n_permutations=int(args.n_permutations), random_state=int(args.random_state)))
    summary_df = pd.concat(runs, ignore_index=True)
    summary_df.to_csv(args.output_dir / "representation_ablation_anchor_auroc.csv", index=False)
    plot_ablation(summary_df, args.output_dir / "representation_ablation_anchor_auroc.png")
    write_summary(args.output_dir / "REPRESENTATION_ABLATION_SUMMARY.md", summary_df, subsets)

    manifest = {
        "anchor_bins_requested": anchor_bins,
        "n_permutations": int(args.n_permutations),
        "focal_genotype": FOCAL_GENOTYPE,
        "variants": sorted(summary_df["variant"].drop_duplicates().tolist()),
        "vector_space_semantics": "support_aware_mean_imputed",
        "init_semantics": "nan_aware_pairwise_umap",
    }
    with open(args.output_dir / "representation_ablation_manifest.json", "w") as handle:
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
