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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_probe_init_audit_cache"
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
KEY_COLS = ["embryo_id", "genotype", "experiment_id", "time_bin", "time_bin_center"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit late-time probe drivers and controlled AlignedUMAP init for inj_ctrl vs wik_ab.")
    parser.add_argument(
        "--pairwise-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_pairwise_bin4_perm500",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "wikab_injctrl_probe_and_init_audit_bin4_perm10",
    )
    parser.add_argument(
        "--shared-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "shared" / "wikab_injctrl_probe_and_init_audit_bin4_perm10",
    )
    parser.add_argument("--late-anchor", type=float, default=78.0)
    parser.add_argument("--anchor-bins", type=str, default="26,54,78")
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_anchor_bins(spec: str) -> list[float]:
    if not spec.strip():
        return [26.0, 54.0, 78.0]
    return [float(part.strip()) for part in spec.split(",") if part.strip()]


def can_score(y: np.ndarray, n_splits: int) -> bool:
    vals, counts = np.unique(y, return_counts=True)
    return len(vals) == 2 and counts.min() >= n_splits


def cross_validated_auroc(X: np.ndarray, y: np.ndarray, *, n_splits: int, random_state: int) -> float:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    probs = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in cv.split(X, y):
        model = make_pipeline(
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


def nearest_available(available: list[float], target: float) -> float:
    return min(available, key=lambda val: abs(val - target))


def load_tables(pairwise_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_coords = pd.read_csv(pairwise_dir / "raw_coordinates.csv")
    shrunk_coords = pd.read_csv(pairwise_dir / "shrunk_coordinates.csv")
    probe_index = pd.read_csv(pairwise_dir / "probe_index.csv")
    for table in (raw_coords, shrunk_coords):
        table["embryo_id"] = table["embryo_id"].astype(str)
        table["experiment_id"] = table["embryo_id"].str.split("_", n=1).str[0]
        table["time_bin_center"] = pd.to_numeric(table["time_bin_center"], errors="coerce")
        table["time_bin"] = pd.to_numeric(table["time_bin"], errors="coerce").astype(int)
        table.dropna(subset=["time_bin_center"], inplace=True)
        table = table[table["genotype"].isin(TARGET_GENOTYPES)]
    raw_coords = raw_coords[raw_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    shrunk_coords = shrunk_coords[shrunk_coords["genotype"].isin(TARGET_GENOTYPES)].copy()
    return raw_coords, shrunk_coords, probe_index


def probe_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "__vs__" in c]


def compute_late_probe_attribution(
    df: pd.DataFrame,
    *,
    stage_name: str,
    late_time_bin_center: float,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    probe_cols = probe_columns(df)
    sub = df[df["time_bin_center"] == late_time_bin_center].copy()
    y = sub["genotype"].map({"inj_ctrl": 0, "wik_ab": 1}).to_numpy(dtype=int)
    rows: list[dict[str, object]] = []
    full_X = sub[probe_cols].fillna(0.0).to_numpy(dtype=float)
    if len(sub) and can_score(y, n_splits):
        full_obs, full_null_mean, full_null_std, full_pval = permutation_auroc(
            full_X,
            y,
            n_splits=n_splits,
            n_permutations=n_permutations,
            random_state=random_state,
        )
    else:
        full_obs = full_null_mean = full_null_std = full_pval = np.nan
    for probe in probe_cols:
        single_X = sub[[probe]].fillna(0.0).to_numpy(dtype=float)
        leave_cols = [c for c in probe_cols if c != probe]
        leave_X = sub[leave_cols].fillna(0.0).to_numpy(dtype=float) if leave_cols else np.zeros((len(sub), 1), dtype=float)
        if len(sub) and can_score(y, n_splits):
            single_obs = cross_validated_auroc(single_X, y, n_splits=n_splits, random_state=random_state)
            leave_obs = cross_validated_auroc(leave_X, y, n_splits=n_splits, random_state=random_state)
        else:
            single_obs = leave_obs = np.nan
        rows.append(
            {
                "stage": stage_name,
                "time_bin_center": float(late_time_bin_center),
                "probe": probe,
                "single_probe_auroc": single_obs,
                "leave_one_out_auroc": leave_obs,
                "delta_full_minus_leave_one_out": full_obs - leave_obs if pd.notna(full_obs) and pd.notna(leave_obs) else np.nan,
                "is_direct_probe": bool(probe == DIRECT_PROBE),
            }
        )
    attr_df = pd.DataFrame(rows).sort_values(
        ["delta_full_minus_leave_one_out", "single_probe_auroc"],
        ascending=[False, False],
    ).reset_index(drop=True)
    full_summary = {
        "stage": stage_name,
        "time_bin_center": float(late_time_bin_center),
        "n_rows": int(len(sub)),
        "n_inj_ctrl": int((sub["genotype"] == "inj_ctrl").sum()),
        "n_wik_ab": int((sub["genotype"] == "wik_ab").sum()),
        "full_vector_auroc": full_obs,
        "full_vector_null_mean": full_null_mean,
        "full_vector_null_std": full_null_std,
        "full_vector_pval": full_pval,
    }
    return attr_df, full_summary


def build_tensor(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    embryos = sorted(df["embryo_id"].unique().tolist())
    time_values = sorted(df["time_bin_center"].unique().tolist())
    embryo_index = {eid: i for i, eid in enumerate(embryos)}
    time_index = {t: i for i, t in enumerate(time_values)}
    features = np.zeros((len(embryos), len(time_values), len(feature_cols)), dtype=float)
    mask = np.zeros((len(embryos), len(time_values)), dtype=bool)
    meta = df[["embryo_id", "genotype", "experiment_id"]].drop_duplicates().set_index("embryo_id")
    for _, row in df.iterrows():
        i = embryo_index[row["embryo_id"]]
        t = time_index[float(row["time_bin_center"])]
        features[i, t, :] = row[feature_cols].fillna(0.0).to_numpy(dtype=float)
        mask[i, t] = True
    labels = np.array([str(meta.loc[eid, "genotype"]) for eid in embryos], dtype=object)
    experiments = np.array([str(meta.loc[eid, "experiment_id"]) for eid in embryos], dtype=object)
    return features, mask, np.array(embryos, dtype=object), np.array(time_values, dtype=float), pd.DataFrame({"embryo_id": embryos, "genotype": labels, "experiment_id": experiments})


def evaluate_init_variant(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    anchor_bins: list[float],
    *,
    n_splits: int,
    n_permutations: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    features, mask, embryo_ids, time_values, meta = build_tensor(df, feature_cols)
    x0 = init_embedding.aligned_umap_init(
        features,
        mask,
        n_neighbors=15,
        min_dist=0.1,
        alignment_regularisation=1e-2,
        alignment_window_size=3,
        random_state=random_state,
    )
    rows: list[dict[str, object]] = []
    for anchor in anchor_bins:
        chosen = nearest_available(time_values.tolist(), anchor)
        t_idx = int(np.where(time_values == chosen)[0][0])
        observed = mask[:, t_idx]
        coords = x0[observed, t_idx, :]
        sub_meta = meta.loc[observed].reset_index(drop=True)
        y = sub_meta["genotype"].map({"inj_ctrl": 0, "wik_ab": 1}).to_numpy(dtype=int)
        if len(sub_meta) and can_score(y, n_splits):
            obs, null_mean, null_std, pval = permutation_auroc(
                coords,
                y,
                n_splits=n_splits,
                n_permutations=n_permutations,
                random_state=random_state,
            )
        else:
            obs = null_mean = null_std = pval = np.nan
        rows.append(
            {
                "variant": name,
                "requested_anchor": float(anchor),
                "time_bin_center": float(chosen),
                "n_rows": int(len(sub_meta)),
                "n_inj_ctrl": int((sub_meta["genotype"] == "inj_ctrl").sum()),
                "n_wik_ab": int((sub_meta["genotype"] == "wik_ab").sum()),
                "auroc_obs": obs,
                "auroc_null_mean": null_mean,
                "auroc_null_std": null_std,
                "pval": pval,
                "n_permutations": int(n_permutations),
            }
        )
    return pd.DataFrame(rows), x0, time_values, meta


def plot_probe_attribution(attr_df: pd.DataFrame, output_path: Path) -> None:
    top = attr_df.head(8).copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=False)
    axes[0].barh(top["probe"], top["single_probe_auroc"], color=["#E45756" if v else "#4C78A8" for v in top["is_direct_probe"]])
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Top single-probe AUROC")
    axes[0].set_xlabel("AUROC")
    axes[1].barh(top["probe"], top["delta_full_minus_leave_one_out"], color=["#E45756" if v else "#54A24B" for v in top["is_direct_probe"]])
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Contribution to full-vector AUROC")
    axes[1].set_xlabel("full - leave-one-out")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_init_anchor_summary(init_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for variant, sub in init_df.groupby("variant"):
        ax.plot(sub["time_bin_center"], sub["auroc_obs"], marker="o", linewidth=1.8, label=variant)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Time bin center (hpf)")
    ax.set_ylabel("AlignedUMAP init AUROC")
    ax.set_title("Controlled AlignedUMAP init comparison", fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(output_path: Path, raw_attr: pd.DataFrame, shrunk_attr: pd.DataFrame, init_df: pd.DataFrame, raw_full: dict[str, float], shrunk_full: dict[str, float]) -> None:
    lines = [
        "# Wik_ab vs Inj_ctrl Probe And Init Audit",
        "",
        f"Late-time attribution target: {raw_full['time_bin_center']:.1f} hpf",
        "",
        "## Full-vector late-bin AUROC",
        pd.DataFrame([raw_full, shrunk_full]).to_markdown(index=False),
        "",
        "## Top raw late-bin probes",
        raw_attr.head(10)[["probe", "single_probe_auroc", "delta_full_minus_leave_one_out", "is_direct_probe"]].to_markdown(index=False),
        "",
        "## Top shrunk late-bin probes",
        shrunk_attr.head(10)[["probe", "single_probe_auroc", "delta_full_minus_leave_one_out", "is_direct_probe"]].to_markdown(index=False),
        "",
        "## Controlled AlignedUMAP init comparison",
        init_df[["variant", "requested_anchor", "time_bin_center", "auroc_obs", "auroc_null_mean", "pval"]].to_markdown(index=False),
    ]
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.n_permutations = min(int(args.n_permutations), 5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor_bins = parse_anchor_bins(args.anchor_bins)

    raw_coords, shrunk_coords, probe_index = load_tables(args.pairwise_dir)
    available = sorted(raw_coords["time_bin_center"].dropna().unique().tolist())
    late_bin = nearest_available(available, float(args.late_anchor))

    raw_attr, raw_full = compute_late_probe_attribution(
        raw_coords,
        stage_name="raw_coordinates",
        late_time_bin_center=float(late_bin),
        n_splits=int(args.n_splits),
        n_permutations=int(args.n_permutations),
        random_state=int(args.random_state),
    )
    shrunk_attr, shrunk_full = compute_late_probe_attribution(
        shrunk_coords,
        stage_name="shrunk_coordinates",
        late_time_bin_center=float(late_bin),
        n_splits=int(args.n_splits),
        n_permutations=int(args.n_permutations),
        random_state=int(args.random_state),
    )
    raw_attr = raw_attr.merge(probe_index[["comparison_id", "positive_label", "negative_label"]], left_on="probe", right_on="comparison_id", how="left")
    shrunk_attr = shrunk_attr.merge(probe_index[["comparison_id", "positive_label", "negative_label"]], left_on="probe", right_on="comparison_id", how="left")

    direct_raw_df = raw_coords[[*KEY_COLS, DIRECT_PROBE]].copy()
    full_raw_df = raw_coords[[*KEY_COLS, *probe_columns(raw_coords)]].copy()
    direct_shrunk_df = shrunk_coords[[*KEY_COLS, DIRECT_PROBE]].copy()
    full_shrunk_df = shrunk_coords[[*KEY_COLS, *probe_columns(shrunk_coords)]].copy()

    init_tables = []
    for name, table, cols in [
        ("direct_raw_only", direct_raw_df, [DIRECT_PROBE]),
        ("full_pairwise_raw", full_raw_df, probe_columns(raw_coords)),
        ("direct_shrunk_only", direct_shrunk_df, [DIRECT_PROBE]),
        ("full_pairwise_shrunk", full_shrunk_df, probe_columns(shrunk_coords)),
    ]:
        init_df, _, _, _ = evaluate_init_variant(
            name,
            table,
            cols,
            anchor_bins,
            n_splits=int(args.n_splits),
            n_permutations=int(args.n_permutations),
            random_state=int(args.random_state),
        )
        init_tables.append(init_df)
    init_summary = pd.concat(init_tables, ignore_index=True)

    raw_attr.to_csv(args.output_dir / "late_bin_raw_probe_attribution.csv", index=False)
    shrunk_attr.to_csv(args.output_dir / "late_bin_shrunk_probe_attribution.csv", index=False)
    pd.DataFrame([raw_full, shrunk_full]).to_csv(args.output_dir / "late_bin_full_vector_summary.csv", index=False)
    init_summary.to_csv(args.output_dir / "controlled_init_anchor_auroc.csv", index=False)
    plot_probe_attribution(raw_attr, args.output_dir / "late_bin_raw_probe_attribution.png")
    plot_probe_attribution(shrunk_attr, args.output_dir / "late_bin_shrunk_probe_attribution.png")
    plot_init_anchor_summary(init_summary, args.output_dir / "controlled_init_anchor_auroc.png")
    write_summary(args.output_dir / "PROBE_AND_INIT_SUMMARY.md", raw_attr, shrunk_attr, init_summary, raw_full, shrunk_full)

    manifest = {
        "late_anchor_requested": float(args.late_anchor),
        "late_bin_used": float(late_bin),
        "anchor_bins_requested": anchor_bins,
        "n_permutations": int(args.n_permutations),
        "variants": ["direct_raw_only", "full_pairwise_raw", "direct_shrunk_only", "full_pairwise_shrunk"],
    }
    with open(args.output_dir / "probe_and_init_manifest.json", "w") as handle:
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
