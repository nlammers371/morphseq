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
    cache_root = Path("/tmp") / "morphseq_within_bin_null_audit_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification.engine.loop import _bin_and_aggregate, _resolve_feature_columns
from common import BUILD06_DIR, CURRENT_EXPERIMENT_IDS, normalize_genotype

TARGET_GENOTYPES = ["inj_ctrl", "wik_ab"]
TARGET_COMPARISON = "inj_ctrl__vs__wik_ab"
STAGE_ORDER = [
    "build06_rows",
    "binned_embeddings",
    "m_raw_injctrl_vs_wikab",
    "raw_coordinates",
    "shrunk_coordinates",
    "pairwise_raw_aligned_umap_init",
    "pairwise_raw_condensed_final",
    "pairwise_shrunk_aligned_umap_init",
    "pairwise_shrunk_condensed_final",
]
ANCHOR_DEFAULTS = [26.0, 54.0, 78.0]
COLOR_BY_STAGE = {
    "build06_rows": "#4C78A8",
    "binned_embeddings": "#72B7B2",
    "m_raw_injctrl_vs_wikab": "#F58518",
    "raw_coordinates": "#E45756",
    "shrunk_coordinates": "#B279A2",
    "pairwise_raw_aligned_umap_init": "#54A24B",
    "pairwise_raw_condensed_final": "#2C7FB8",
    "pairwise_shrunk_aligned_umap_init": "#EECA3B",
    "pairwise_shrunk_condensed_final": "#9D755D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Within-time-bin inj_ctrl vs wik_ab null-comparison audit across pipeline stages.")
    parser.add_argument(
        "--pairwise-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_pairwise_bin4_perm500",
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "wikab_injctrl_within_bin_null_audit_bin4_perm10",
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "shared" / "wikab_injctrl_within_bin_null_audit_bin4_perm10",
    )
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--anchor-bins", type=str, default="26,54,78")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_anchor_bins(spec: str) -> list[float]:
    if not spec.strip():
        return ANCHOR_DEFAULTS[:]
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def load_build06_current() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
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
    df["embryo_id"] = df["embryo_id"].astype(str)
    return df.reset_index(drop=True)


def assign_time_bins(df: pd.DataFrame, *, time_col: str, bin_width: float) -> pd.DataFrame:
    out = df.copy()
    out["time_bin"] = (np.floor(out[time_col] / bin_width) * bin_width).astype(int)
    out["time_bin_center"] = out["time_bin"].astype(float) + bin_width / 2.0
    return out


def nearest_available_bins(available: list[float], anchors: list[float]) -> list[float]:
    if not available:
        return []
    picked: list[float] = []
    for anchor in anchors:
        nearest = min(available, key=lambda val: abs(val - anchor))
        if nearest not in picked:
            picked.append(nearest)
    return picked


def can_score(y: np.ndarray, n_splits: int) -> bool:
    vals, counts = np.unique(y, return_counts=True)
    return len(vals) == 2 and counts.min() >= n_splits


def support_aware_feature_cols(sub: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    if not any("__vs__" in col for col in feature_cols):
        return feature_cols
    inj = sub[sub["genotype"] == "inj_ctrl"]
    wik = sub[sub["genotype"] == "wik_ab"]
    keep: list[str] = []
    for col in feature_cols:
        if inj[col].notna().any() and wik[col].notna().any():
            keep.append(col)
    return keep


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


def load_pairwise_coordinates(pairwise_dir: Path, embryo_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_coords = pd.read_csv(pairwise_dir / "raw_coordinates.csv")
    shrunk_coords = pd.read_csv(pairwise_dir / "shrunk_coordinates.csv")
    raw_long = pd.read_csv(pairwise_dir / "raw_contrast_scores_long.csv")
    specificity = pd.read_csv(pairwise_dir / "contrast_specificity_by_timebin.csv")
    experiment_lookup = embryo_meta.set_index("embryo_id")["experiment_id"]
    for table in (raw_coords, shrunk_coords, raw_long):
        table["embryo_id"] = table["embryo_id"].astype(str)
        table["time_bin_center"] = pd.to_numeric(table["time_bin_center"], errors="coerce")
        table.dropna(subset=["time_bin_center"], inplace=True)
        if "genotype" not in table.columns:
            table["genotype"] = table["embryo_id"].map(embryo_meta.set_index("embryo_id")["genotype"])
        table["experiment_id"] = table["embryo_id"].map(experiment_lookup)
        table.dropna(subset=["experiment_id"], inplace=True)
    raw_coords = raw_coords[raw_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    shrunk_coords = shrunk_coords[shrunk_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    raw_long = raw_long[(raw_long["genotype"].isin(TARGET_GENOTYPES)) & (raw_long["comparison_id"] == TARGET_COMPARISON)].copy()
    specificity = specificity[specificity["comparison_id"] == TARGET_COMPARISON].copy()
    return raw_coords, shrunk_coords, raw_long, specificity


def extract_condensation_stage(npz_path: Path, stage_key: str) -> tuple[pd.DataFrame, list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    arr = data[stage_key]
    rows: list[dict[str, object]] = []
    embryo_ids = data["embryo_ids"]
    labels = data["labels"]
    mask = data["mask"]
    time_values = data["time_values"]
    for i, embryo_id in enumerate(embryo_ids):
        genotype = str(labels[i])
        if genotype not in TARGET_GENOTYPES:
            continue
        for t, hpf in enumerate(time_values):
            if not bool(mask[i, t]):
                continue
            rows.append(
                {
                    "embryo_id": str(embryo_id),
                    "genotype": genotype,
                    "time_bin_center": float(hpf),
                    "dim_1": float(arr[i, t, 0]),
                    "dim_2": float(arr[i, t, 1]),
                }
            )
    return pd.DataFrame(rows), ["dim_1", "dim_2"]


def build_stage_frames(args: argparse.Namespace) -> tuple[dict[str, tuple[pd.DataFrame, list[str]]], pd.DataFrame, list[float]]:
    build06 = load_build06_current()
    resolved_features = _resolve_feature_columns(build06, {"vae": "z_mu_b"})
    feature_cols = resolved_features["vae"]
    build_rows = assign_time_bins(
        build06[["embryo_id", "genotype", "experiment_id", "stage_hpf", *feature_cols]].copy(),
        time_col="stage_hpf",
        bin_width=float(args.bin_width),
    )
    build_stage = build_rows[["embryo_id", "genotype", "experiment_id", "time_bin", "time_bin_center", *feature_cols]].copy()

    y_map = {"inj_ctrl": 0, "wik_ab": 1}
    binned_input = build06.copy()
    binned_input["_y"] = binned_input["genotype"].map(y_map)
    binned = _bin_and_aggregate(binned_input, "embryo_id", "stage_hpf", feature_cols, float(args.bin_width))
    embryo_meta = build06[["embryo_id", "genotype", "experiment_id"]].drop_duplicates().copy()
    binned = binned.merge(embryo_meta, on="embryo_id", how="left", validate="many_to_one")
    binned["time_bin"] = binned["_time_bin"].astype(int)
    binned_stage = binned[["embryo_id", "genotype", "experiment_id", "time_bin", "time_bin_center", *feature_cols]].copy()

    raw_coords, shrunk_coords, raw_long, specificity = load_pairwise_coordinates(args.pairwise_dir, embryo_meta)
    raw_probe_cols = [c for c in raw_coords.columns if "__vs__" in c]
    shrunk_probe_cols = [c for c in shrunk_coords.columns if "__vs__" in c]
    raw_coord_stage = raw_coords[["embryo_id", "genotype", "experiment_id", "time_bin", "time_bin_center", *raw_probe_cols]].copy()
    shrunk_coord_stage = shrunk_coords[["embryo_id", "genotype", "experiment_id", "time_bin", "time_bin_center", *shrunk_probe_cols]].copy()
    margin_stage = raw_long[["embryo_id", "genotype", "experiment_id", "time_bin", "time_bin_center", "m_raw"]].copy()

    raw_init_df, raw_init_cols = extract_condensation_stage(args.pairwise_raw_condensation, "x0")
    raw_final_df, raw_final_cols = extract_condensation_stage(args.pairwise_raw_condensation, "positions")
    shrunk_init_df, shrunk_init_cols = extract_condensation_stage(args.pairwise_shrunk_condensation, "x0")
    shrunk_final_df, shrunk_final_cols = extract_condensation_stage(args.pairwise_shrunk_condensation, "positions")
    for table in (raw_init_df, raw_final_df, shrunk_init_df, shrunk_final_df):
        table["experiment_id"] = table["embryo_id"].map(embryo_meta.set_index("embryo_id")["experiment_id"])
        table["time_bin"] = np.floor(table["time_bin_center"] - float(args.bin_width) / 2.0 + 1e-6).astype(int)
        table.dropna(subset=["experiment_id"], inplace=True)

    stage_frames = {
        "build06_rows": (build_stage, feature_cols),
        "binned_embeddings": (binned_stage, feature_cols),
        "m_raw_injctrl_vs_wikab": (margin_stage, ["m_raw"]),
        "raw_coordinates": (raw_coord_stage, raw_probe_cols),
        "shrunk_coordinates": (shrunk_coord_stage, shrunk_probe_cols),
        "pairwise_raw_aligned_umap_init": (raw_init_df, raw_init_cols),
        "pairwise_raw_condensed_final": (raw_final_df, raw_final_cols),
        "pairwise_shrunk_aligned_umap_init": (shrunk_init_df, shrunk_init_cols),
        "pairwise_shrunk_condensed_final": (shrunk_final_df, shrunk_final_cols),
    }
    available = sorted({float(v) for _, (df, _) in stage_frames.items() for v in df["time_bin_center"].dropna().unique().tolist()})
    return stage_frames, specificity, available


def compute_within_bin_metrics(stage_name: str, df: pd.DataFrame, feature_cols: list[str], *, n_splits: int, n_permutations: int, random_state: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for time_bin_center, sub in df.groupby("time_bin_center"):
        sub = sub[sub["genotype"].isin(TARGET_GENOTYPES)].copy()
        supported_cols = support_aware_feature_cols(sub, feature_cols)
        y = sub["genotype"].map({"inj_ctrl": 0, "wik_ab": 1}).to_numpy(dtype=int)
        if len(sub) and supported_cols and can_score(y, n_splits):
            X = sub[supported_cols].to_numpy(dtype=float)
            obs, null_mean, null_std, pval = permutation_auroc(
                X,
                y,
                n_splits=n_splits,
                n_permutations=n_permutations,
                random_state=random_state,
            )
        else:
            obs = null_mean = null_std = pval = np.nan
        rows.append(
            {
                "stage": stage_name,
                "time_bin_center": float(time_bin_center),
                "n_rows": int(len(sub)),
                "n_embryos": int(sub["embryo_id"].nunique()),
                "n_inj_ctrl": int((sub["genotype"] == "inj_ctrl").sum()),
                "n_wik_ab": int((sub["genotype"] == "wik_ab").sum()),
                "auroc_obs": obs,
                "auroc_null_mean": null_mean,
                "auroc_null_std": null_std,
                "pval": pval,
                "n_permutations": int(n_permutations),
            }
        )
    out = pd.DataFrame(rows).sort_values("time_bin_center").reset_index(drop=True)
    if not out.empty:
        out["qval"] = bh_qvalues(out, "pval")
    return out


def summarize_anchor_bins(within_bin_df: pd.DataFrame, anchor_bins: list[float]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for stage_name, sub in within_bin_df.groupby("stage"):
        available = sorted(sub["time_bin_center"].dropna().unique().tolist())
        if not available:
            continue
        chosen = nearest_available_bins(available, anchor_bins)
        stage_rows = sub[sub["time_bin_center"].isin(chosen)].copy()
        stage_rows["requested_anchor"] = stage_rows["time_bin_center"].map({nearest: anchor for anchor, nearest in zip(anchor_bins, chosen)})
        rows.append(stage_rows)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_within_bin_auroc(within_bin_df: pd.DataFrame, anchor_bins: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for stage_name in STAGE_ORDER:
        sub = within_bin_df[within_bin_df["stage"] == stage_name].copy()
        if sub.empty:
            continue
        ax.plot(
            sub["time_bin_center"],
            sub["auroc_obs"],
            marker="o",
            markersize=3.0,
            linewidth=1.6,
            label=stage_name,
            color=COLOR_BY_STAGE.get(stage_name, None),
            alpha=0.9,
        )
    for anchor in anchor_bins:
        ax.axvline(anchor, color="#BBBBBB", linestyle=":", linewidth=1)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Time bin center (hpf)")
    ax.set_ylabel("Within-bin inj_ctrl vs wik_ab AUROC")
    ax.set_title("Within-time-bin null comparison across pipeline stages", fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(output_path: Path, anchor_df: pd.DataFrame) -> None:
    lines = [
        "# Within-Bin Wik_ab vs Inj_ctrl Null Audit",
        "",
        "The target quantity is time-bin-internal `inj_ctrl` vs `wik_ab` AUROC. These controls should be near-null within matched biological stage.",
        "",
    ]
    for anchor in sorted(anchor_df["requested_anchor"].dropna().unique().tolist()):
        sub = anchor_df[anchor_df["requested_anchor"] == anchor].copy().sort_values("stage")
        lines.append(f"## Requested anchor {anchor:.1f} hpf")
        lines.append(sub[["stage", "time_bin_center", "auroc_obs", "auroc_null_mean", "pval", "qval", "n_inj_ctrl", "n_wik_ab"]].to_markdown(index=False))
        lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_permutations = min(int(args.n_permutations), 5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor_bins = parse_anchor_bins(args.anchor_bins)

    stage_frames, specificity, available = build_stage_frames(args)
    chosen_bins = nearest_available_bins(available, anchor_bins)

    within_bin_tables = []
    for stage_name in STAGE_ORDER:
        df, feature_cols = stage_frames[stage_name]
        table = compute_within_bin_metrics(
            stage_name,
            df,
            feature_cols,
            n_splits=int(args.n_splits),
            n_permutations=int(args.n_permutations),
            random_state=int(args.random_state),
        )
        within_bin_tables.append(table)
    within_bin_df = pd.concat(within_bin_tables, ignore_index=True)
    anchor_df = summarize_anchor_bins(within_bin_df, anchor_bins)

    within_bin_df.to_csv(args.output_dir / "within_bin_null_auroc_by_stage.csv", index=False)
    anchor_df.to_csv(args.output_dir / "anchor_bin_null_auroc_summary.csv", index=False)
    specificity.to_csv(args.output_dir / "injctrl_vs_wikab_specificity_by_timebin.csv", index=False)
    plot_within_bin_auroc(within_bin_df, chosen_bins, args.output_dir / "within_bin_null_auroc_by_stage.png")
    write_summary(args.output_dir / "WITHIN_BIN_NULL_SUMMARY.md", anchor_df)

    manifest = {
        "anchor_bins_requested": anchor_bins,
        "anchor_bins_used": chosen_bins,
        "n_permutations": int(args.n_permutations),
        "stages": STAGE_ORDER,
    }
    with open(args.output_dir / "within_bin_null_manifest.json", "w") as handle:
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
