"""Validate CEP290 comparisons using PROPER GENOTYPE groupings (not cluster_categories).

This script fixes the fundamental issue: the refined analysis compared "Penetrant" (phenotype
clusters) vs "Not Penetrant" (mixed WT/Het/non-penetrant homo), which is scientifically unclear.

Correct comparisons (matching working code in 20260102_labmeeting_plots):
1. cep290_homozygous vs cep290_wildtype (primary genetic comparison)
2. cep290_homozygous vs cep290_heterozygous (dosage sensitivity)
3. cep290_heterozygous vs cep290_wildtype (het baseline)

This validation script runs the same stratified-permutation checks as validate_12hpf_signal.py,
but uses genotype-based grouping instead of cluster_categories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv",
        help="CSV data source.",
    )

    parser.add_argument(
        "--comparison",
        type=str,
        choices=["homo_vs_wt", "homo_vs_het", "het_vs_wt"],
        default="homo_vs_wt",
        help="Which genotype comparison to validate.",
    )

    parser.add_argument(
        "--time-col",
        type=str,
        default="predicted_stage_hpf",
        help="Time column name.",
    )

    parser.add_argument(
        "--embryo-id-col",
        type=str,
        default="embryo_id",
        help="Embryo ID column name.",
    )

    parser.add_argument(
        "--bin-start",
        type=float,
        default=12.0,
        help="Bin start (hpf).",
    )

    parser.add_argument(
        "--bin-width",
        type=float,
        default=4.0,
        help="Bin width (hours).",
    )

    parser.add_argument(
        "--curvature-col",
        type=str,
        default="baseline_deviation_normalized",
        help="Curvature metric column.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Stratified CV folds.",
    )

    parser.add_argument(
        "--n-permutations",
        type=int,
        default=500,
        help="Number of permutations.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--time-strata-width",
        type=float,
        default=0.5,
        help="Width (hours) for within-bin time strata.",
    )

    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding checks.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (auto-generated if not specified).",
    )

    return parser.parse_args()


def _compute_time_bin(series: pd.Series, bin_width: float) -> pd.Series:
    return (np.floor(series.astype(float) / float(bin_width)) * float(bin_width)).astype(int)


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    denom = len(x) * len(y)
    return float((gt - lt) / denom)


def _perm_pval(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_permutations: int,
    random_state: int,
    scale: bool,
    strata: Optional[np.ndarray] = None,
) -> dict:
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    min_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits_actual = int(min(n_splits, min_count))
    cv = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=random_state)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )
    model = make_pipeline(StandardScaler(), clf) if scale else clf

    probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auroc_observed = float(roc_auc_score(y, probs))

    rng = np.random.default_rng(int(random_state))

    def _permute_labels(labels: np.ndarray) -> np.ndarray:
        if strata is None:
            return rng.permutation(labels)

        strata_ids = np.asarray(strata)
        out = labels.copy()
        for s in np.unique(strata_ids):
            idx = np.where(strata_ids == s)[0]
            if len(idx) <= 1:
                continue
            out[idx] = rng.permutation(out[idx])
        return out

    null_aurocs: list[float] = []
    for _ in range(int(n_permutations)):
        y_perm = _permute_labels(y)
        try:
            probs_perm = cross_val_predict(model, X, y_perm, cv=cv, method="predict_proba")[:, 1]
            null_aurocs.append(float(roc_auc_score(y_perm, probs_perm)))
        except Exception:
            continue

    null = np.asarray(null_aurocs, dtype=float)
    null = null[np.isfinite(null)]

    if len(null) == 0:
        return {
            "auroc_observed": auroc_observed,
            "auroc_null_mean": float("nan"),
            "auroc_null_std": float("nan"),
            "pval": float("nan"),
            "n_null": 0,
        }

    k = int(np.sum(null >= auroc_observed))
    pval = float((k + 1) / (len(null) + 1))

    return {
        "auroc_observed": auroc_observed,
        "auroc_null_mean": float(np.mean(null)),
        "auroc_null_std": float(np.std(null)),
        "pval": pval,
        "n_null": int(len(null)),
    }


def main() -> int:
    args = _parse_args()

    # Define genotype labels based on comparison
    COMPARISON_MAP = {
        "homo_vs_wt": {
            "group1": "cep290_homozygous",
            "group2": "cep290_wildtype",
            "label": "Homo_vs_WT",
        },
        "homo_vs_het": {
            "group1": "cep290_homozygous",
            "group2": "cep290_heterozygous",
            "label": "Homo_vs_Het",
        },
        "het_vs_wt": {
            "group1": "cep290_heterozygous",
            "group2": "cep290_wildtype",
            "label": "Het_vs_WT",
        },
    }

    comp_cfg = COMPARISON_MAP[args.comparison]
    group1 = comp_cfg["group1"]
    group2 = comp_cfg["group2"]
    label = comp_cfg["label"]

    if args.output_dir is None:
        bin_str = f"{int(args.bin_start)}hpf"
        out_dir = (
            Path(__file__).parent
            / "output"
            / "cep290"
            / f"validation_{label.lower()}_{bin_str}"
        )
    else:
        out_dir = args.output_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv, low_memory=False)

    if "genotype" not in df.columns:
        raise ValueError("genotype column not found in data")

    # Filter to just the two genotypes
    df = df[df["genotype"].isin([group1, group2])].copy()

    if len(df) == 0:
        raise ValueError(f"No data found for {group1} or {group2}")

    df["time_bin"] = _compute_time_bin(df[args.time_col], args.bin_width)

    sub = df[df["time_bin"] == int(args.bin_start)].copy()

    if len(sub) == 0:
        raise ValueError(f"No rows for time_bin={int(args.bin_start)}")

    # Per-embryo summaries
    per_embryo = (
        sub.groupby([args.embryo_id_col, "genotype"], as_index=False)
        .agg(
            mean_time=(args.time_col, "mean"),
            median_time=(args.time_col, "median"),
            n_obs=(args.time_col, "size"),
            curvature_mean=(args.curvature_col, "mean"),
            curvature_median=(args.curvature_col, "median"),
            curvature_std=(args.curvature_col, "std"),
        )
    )

    per_embryo.to_csv(out_dir / "per_embryo_bin_summary.csv", index=False)

    n_g1 = int((per_embryo["genotype"] == group1).sum())
    n_g2 = int((per_embryo["genotype"] == group2).sum())

    g1_time = per_embryo.loc[per_embryo["genotype"] == group1, "mean_time"].to_numpy()
    g2_time = per_embryo.loc[per_embryo["genotype"] == group2, "mean_time"].to_numpy()

    time_d = _cohens_d(g1_time, g2_time)
    time_cd = _cliffs_delta(g1_time, g2_time)

    g1_curv = per_embryo.loc[per_embryo["genotype"] == group1, "curvature_mean"].to_numpy()
    g2_curv = per_embryo.loc[per_embryo["genotype"] == group2, "curvature_mean"].to_numpy()

    curv_d = _cohens_d(g1_curv, g2_curv)
    curv_cd = _cliffs_delta(g1_curv, g2_curv)

    # Build y (group1 = 1)
    y = (per_embryo["genotype"].values == group1).astype(int)

    # Time strata
    strata = np.floor(
        (per_embryo["mean_time"].values - args.bin_start) / float(args.time_strata_width)
    ).astype(int)

    # 1) Time-only
    time_perm = _perm_pval(
        X=per_embryo["mean_time"].values,
        y=y,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        scale=True,
        strata=strata,
    )

    # 2) Curvature-only
    curv_perm = _perm_pval(
        X=per_embryo["curvature_mean"].values,
        y=y,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        scale=True,
        strata=strata,
    )

    # 3) Curvature + time
    X_ct = np.column_stack([per_embryo["curvature_mean"].values, per_embryo["mean_time"].values])
    curv_time_perm = _perm_pval(
        X=X_ct,
        y=y,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        scale=True,
        strata=strata,
    )

    # 4) n_obs
    nobs_perm = _perm_pval(
        X=per_embryo["n_obs"].values,
        y=y,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        scale=True,
        strata=strata,
    )

    # 5) Embedding (optional)
    embedding_perm = None
    if not args.skip_embedding:
        emb_cols = [c for c in df.columns if "z_mu_b" in c]
        if len(emb_cols) > 0:
            emb_binned = (
                sub.groupby([args.embryo_id_col, "genotype"], as_index=False)[emb_cols]
                .mean()
                .merge(
                    per_embryo[[args.embryo_id_col, "genotype", "mean_time"]],
                    on=[args.embryo_id_col, "genotype"],
                    how="left",
                )
            )
            emb_y = (emb_binned["genotype"].values == group1).astype(int)
            emb_X = emb_binned[emb_cols].values
            emb_strata = np.floor(
                (emb_binned["mean_time"].values - args.bin_start) / float(args.time_strata_width)
            ).astype(int)

            embedding_perm = _perm_pval(
                X=emb_X,
                y=emb_y,
                n_splits=args.n_splits,
                n_permutations=args.n_permutations,
                random_state=args.random_state,
                scale=True,
                strata=emb_strata,
            )

    # Save summary
    rows = []

    def _add_row(name: str, d: dict) -> None:
        rows.append({"analysis": name, **d})

    _add_row("time_only (stratified perm)", time_perm)
    _add_row("curvature_only (stratified perm)", curv_perm)
    _add_row("curvature_plus_time (stratified perm)", curv_time_perm)
    _add_row("n_obs_only (stratified perm)", nobs_perm)
    if embedding_perm is not None:
        _add_row("embedding_only (stratified perm)", embedding_perm)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "auroc_permutation_summary.csv", index=False)

    # Human-readable report
    report_lines = []
    report_lines.append(f"Comparison: {label} ({group1} vs {group2})")
    report_lines.append(f"Data: {args.data_csv}")
    report_lines.append(
        f"Bin: {args.bin_start:.1f}–{args.bin_start + args.bin_width:.1f} hpf (time_bin={int(args.bin_start)})"
    )
    report_lines.append(f"Per-embryo samples: {group1}={n_g1}, {group2}={n_g2}")
    report_lines.append("")
    report_lines.append("Within-bin time skew:")
    report_lines.append(
        f"  mean_time {group1}={np.mean(g1_time):.3f}, {group2}={np.mean(g2_time):.3f}"
    )
    report_lines.append(f"  Cohen's d = {time_d:.3f}, Cliff's delta = {time_cd:.3f}")
    report_lines.append("")
    report_lines.append("Curvature distribution:")
    report_lines.append(
        f"  curvature_mean {group1}={np.mean(g1_curv):.4g}, {group2}={np.mean(g2_curv):.4g}"
    )
    report_lines.append(f"  Cohen's d = {curv_d:.3f}, Cliff's delta = {curv_cd:.3f}")
    report_lines.append("")
    report_lines.append("Stratified permutation AUROC (labels shuffled within time strata):")
    for r in rows:
        report_lines.append(
            f"  {r['analysis']}: AUROC={r['auroc_observed']:.3f}, p={r['pval']:.4g} (null n={r['n_null']})"
        )

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")

    print("\n".join(report_lines))
    print(f"\nWrote outputs to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
