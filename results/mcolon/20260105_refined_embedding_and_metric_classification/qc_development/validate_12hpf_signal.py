"""Validate the apparent 12 hpf AUROC signal in CEP290 penetrant vs control.

This is a focused diagnostic to reconcile:
- Visual overlap of early trajectories ("by eye")
- Significant AUROC at time_bin=12 with p ~ 0.01

Key checks implemented
----------------------
1) ID / labeling sanity:
   - embryo IDs are disjoint between groups
   - no missing group labels
2) Within-bin time skew:
   - compares per-embryo mean/median predicted_stage_hpf within the 12–16 hpf bin
   - computes "time-only" AUROC (can time alone classify groups?)
3) Distribution shift vs mean shift:
   - compares per-embryo curvature distributions (not just mean trajectories)
   - effect sizes (Cohen's d, Cliff's delta)
4) Leakage-safe AUROC with permutations:
   - cross-validated AUROC via Pipeline(StandardScaler + LogisticRegression)
   - permutation p-value (overall label shuffle)
   - stratified permutation p-value (shuffle labels within fine time strata)

Outputs
-------
Writes outputs under:
`results/mcolon/20260105_refined_embedding_and_metric_classification/output/cep290/validation_12hpf/`

Notes
-----
- With n_permutations=100, the minimum achievable p-value is 1/(100+1)=0.00990099.
- The "12 hpf" time_bin in the current pipeline (bin_width=4) corresponds to 12–16 hpf.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BinSpec:
    start: float
    width: float

    @property
    def end(self) -> float:
        return float(self.start) + float(self.width)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv",
        help="CSV data source (must include embryo_id, cluster_categories, predicted_stage_hpf, baseline_deviation_normalized, and z_mu_b_* columns).",
    )

    parser.add_argument(
        "--time-col",
        type=str,
        default="predicted_stage_hpf",
        help="Time column name (hpf).",
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
        help="Bin start (hpf). For the standard pipeline with bin_width=4, this corresponds to 12–16 hpf.",
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
        help="Curvature metric column used for the metric-only checks.",
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
        help="Number of permutations for p-value estimation.",
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
        help="Width (hours) for within-bin time strata when doing stratified label permutations.",
    )

    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip embedding-based checks (faster).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output" / "cep290" / "validation_12hpf",
        help="Output directory for CSV/plots.",
    )

    return parser.parse_args()


def _add_group_labels(
    df: pd.DataFrame,
    embryo_id_col: str,
    group_col: str = "group",
    penetrant_label: str = "Penetrant",
    control_label: str = "Control",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    penetrant_categories = ["Low_to_High", "High_to_Low", "Intermediate"]
    penetrant_ids = (
        df[df["cluster_categories"].isin(penetrant_categories)][embryo_id_col]
        .dropna()
        .unique()
        .tolist()
    )
    control_ids = (
        df[df["cluster_categories"] == "Not Penetrant"][embryo_id_col]
        .dropna()
        .unique()
        .tolist()
    )

    embryo_to_group: dict[str, str] = {
        **{eid: penetrant_label for eid in penetrant_ids},
        **{eid: control_label for eid in control_ids},
    }

    out = df.copy()
    out[group_col] = out[embryo_id_col].map(embryo_to_group)

    return out, penetrant_ids, control_ids


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
    """Cliff's delta: P(x>y) - P(x<y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    # O(n*m) but n here is small (embryos per bin)
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    denom = len(x) * len(y)
    return float((gt - lt) / denom)


def _cv_auroc(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    scale: bool,
) -> float:
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    min_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    if min_count < 2:
        return float("nan")

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
    return float(roc_auc_score(y, probs))


def _perm_pval(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    n_permutations: int,
    random_state: int,
    scale: bool,
    strata: Optional[np.ndarray] = None,
) -> dict:
    """Permutation test for AUROC.

    If `strata` is provided, labels are permuted *within* each stratum.
    """
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

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv, low_memory=False)

    required = {args.embryo_id_col, args.time_col, "cluster_categories", args.curvature_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df, penetrant_ids, control_ids = _add_group_labels(
        df,
        embryo_id_col=args.embryo_id_col,
        group_col="group",
        penetrant_label="Penetrant",
        control_label="Control",
    )

    # Sanity: disjoint IDs
    overlap = set(penetrant_ids).intersection(control_ids)
    if overlap:
        raise ValueError(f"Found {len(overlap)} embryo IDs in BOTH penetrant and control lists")

    # Filter to labeled rows
    df = df[df["group"].isin(["Penetrant", "Control"])].copy()

    # Standard binning definition (matches compare_groups)
    df["time_bin"] = _compute_time_bin(df[args.time_col], args.bin_width)

    bin_spec = BinSpec(start=float(args.bin_start), width=float(args.bin_width))
    sub = df[df["time_bin"] == int(bin_spec.start)].copy()

    if len(sub) == 0:
        raise ValueError(
            f"No rows found for time_bin={int(bin_spec.start)}. "
            f"Check --bin-start/--bin-width and your {args.time_col} values."
        )

    # Per-embryo summaries inside the bin (this is close to what compare_groups uses)
    per_embryo = (
        sub.groupby([args.embryo_id_col, "group"], as_index=False)
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

    # Basic counts
    n_pen = int((per_embryo["group"] == "Penetrant").sum())
    n_ctl = int((per_embryo["group"] == "Control").sum())

    # Within-bin time skew
    pen_time = per_embryo.loc[per_embryo["group"] == "Penetrant", "mean_time"].to_numpy()
    ctl_time = per_embryo.loc[per_embryo["group"] == "Control", "mean_time"].to_numpy()

    time_d = _cohens_d(pen_time, ctl_time)
    time_cd = _cliffs_delta(pen_time, ctl_time)

    # Curvature distribution shift
    pen_curv = per_embryo.loc[per_embryo["group"] == "Penetrant", "curvature_mean"].to_numpy()
    ctl_curv = per_embryo.loc[per_embryo["group"] == "Control", "curvature_mean"].to_numpy()

    curv_d = _cohens_d(pen_curv, ctl_curv)
    curv_cd = _cliffs_delta(pen_curv, ctl_curv)

    # Build y (Penetrant = 1)
    y = (per_embryo["group"].values == "Penetrant").astype(int)

    # Time strata for stratified permutations
    # (12–16) with width=0.5 => up to 8 strata; if some are empty, that's OK.
    strata = np.floor((per_embryo["mean_time"].values - bin_spec.start) / float(args.time_strata_width)).astype(int)

    # 1) Time-only classification
    time_perm = _perm_pval(
        X=per_embryo["mean_time"].values,
        y=y,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        scale=True,
        strata=strata,
    )

    # 2) Curvature-only classification
    curv_perm = _perm_pval(
        X=per_embryo["curvature_mean"].values,
        y=y,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        scale=True,
        strata=strata,
    )

    # 3) Curvature + time classification
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

    # 4) n_obs-only classification (sometimes reveals sampling/procedure confound)
    nobs_perm = _perm_pval(
        X=per_embryo["n_obs"].values,
        y=y,
        n_splits=args.n_splits,
        n_permutations=args.n_permutations,
        random_state=args.random_state,
        scale=True,
        strata=strata,
    )

    # 5) Embedding-based (optional; may be slower)
    embedding_perm = None
    if not args.skip_embedding:
        emb_cols = [c for c in df.columns if "z_mu_b" in c]
        if len(emb_cols) == 0:
            print("WARNING: No z_mu_b columns found; skipping embedding check.")
        else:
            emb_binned = (
                sub.groupby([args.embryo_id_col, "group"], as_index=False)[emb_cols]
                .mean()
                .merge(per_embryo[[args.embryo_id_col, "group", "mean_time"]], on=[args.embryo_id_col, "group"], how="left")
            )
            emb_y = (emb_binned["group"].values == "Penetrant").astype(int)
            emb_X = emb_binned[emb_cols].values
            emb_strata = np.floor((emb_binned["mean_time"].values - bin_spec.start) / float(args.time_strata_width)).astype(int)

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

    _add_row("time_only (stratified perm within time strata)", time_perm)
    _add_row("curvature_only (stratified perm within time strata)", curv_perm)
    _add_row("curvature_plus_time (stratified perm within time strata)", curv_time_perm)
    _add_row("n_obs_only (stratified perm within time strata)", nobs_perm)
    if embedding_perm is not None:
        _add_row("embedding_only (stratified perm within time strata)", embedding_perm)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "auroc_permutation_summary.csv", index=False)

    # Also write a concise human-readable summary
    report_lines = []
    report_lines.append(f"Data: {args.data_csv}")
    report_lines.append(f"Bin: {bin_spec.start:.1f}–{bin_spec.end:.1f} hpf (time_bin={int(bin_spec.start)}, bin_width={bin_spec.width})")
    report_lines.append(f"Per-embryo samples in bin: Penetrant={n_pen}, Control={n_ctl}")
    report_lines.append("")
    report_lines.append("Within-bin time skew (per-embryo mean_time):")
    report_lines.append(f"  mean_time Penetrant mean={np.mean(pen_time):.3f}, Control mean={np.mean(ctl_time):.3f}")
    report_lines.append(f"  Cohen's d (Pen-Control) = {time_d:.3f}")
    report_lines.append(f"  Cliff's delta           = {time_cd:.3f}")
    report_lines.append("")
    report_lines.append("Curvature distribution (per-embryo curvature_mean):")
    report_lines.append(f"  curvature_mean Penetrant mean={np.mean(pen_curv):.4g}, Control mean={np.mean(ctl_curv):.4g}")
    report_lines.append(f"  Cohen's d (Pen-Control) = {curv_d:.3f}")
    report_lines.append(f"  Cliff's delta           = {curv_cd:.3f}")
    report_lines.append("")
    report_lines.append("Permutation AUROC checks (labels shuffled within time strata):")
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
