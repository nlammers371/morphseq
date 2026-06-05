"""
Diagnose why KNN-q loses to multiclass-q in the conformal benchmark.

Inputs:
    q_conformal_benchmark_image_predictions.csv
    results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv

Outputs:
    q_diagnostic_true_label_rank.csv
    q_diagnostic_true_label_rank_summary.csv
    q_diagnostic_neighbor_geometry.csv
    q_diagnostic_neighbor_geometry_summary.csv
    q_diagnostic_rescue_groups.csv
    q_diagnostic_rescue_group_summary.csv
    q_diagnostic_set_composition.csv
    q_diagnostic_report.md

Usage:
    /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
        results/mcolon/20260601_label_transfer_method/run_q_failure_diagnostics.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

from run_q_conformal_benchmark import (  # noqa: E402
    DATA_PATH,
    EMBRYO_COL,
    EXPERIMENT_COL,
    LABEL_COL,
    MAIN_LABELS,
    MAX_HPF,
    MIN_HPF,
    SNIP_COL,
    TIME_COL,
    add_hpf_bin,
    get_feature_cols,
    split_reference_calibration,
)


PREDICTION_PATH = HERE / "q_conformal_benchmark_image_predictions.csv"
K_VALUES = [5, 15, 30, 50, 100, 200]
COARSE_HPF_BINS = [(30.0, 36.0), (36.0, 42.0), (42.0, 48.0)]


def coarse_hpf_bin(hpf: float) -> str:
    for lo, hi in COARSE_HPF_BINS:
        if lo <= hpf < hi or (hi == MAX_HPF and hpf <= hi):
            return f"{int(lo)}-{int(hi)}"
    return "outside"


def true_label_rank_rows(pred: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for _, r in pred.iterrows():
        true_label = r["true_label"]
        true_score = float(r[f"q_{true_label}"])
        rank = 1 + sum(float(r[f"q_{label}"]) > true_score for label in labels)
        rows.append({
            "method": r["method"],
            "q_source": r["q_source"],
            "heldout_experiment_id": r["heldout_experiment_id"],
            "snip_id": r[SNIP_COL],
            "embryo_id": r[EMBRYO_COL],
            "hpf": float(r[TIME_COL]),
            "hpf_bin": r["_hpf_bin"],
            "coarse_hpf_bin": coarse_hpf_bin(float(r[TIME_COL])),
            "true_label": true_label,
            "argmax_label": r["argmax_label"],
            "argmax_correct": bool(r["argmax_label"] == true_label),
            "prediction_set": r["prediction_set"],
            "set_size": int(r["set_size"]),
            "covered": bool(r[f"in_set_{true_label}"]),
            "q_true": true_score,
            "true_label_rank": int(rank),
            **{f"q_{label}": float(r[f"q_{label}"]) for label in labels},
        })
    return pd.DataFrame(rows)


def summarize_rank(rank_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["method", "true_label", "coarse_hpf_bin"]
    return (
        rank_df
        .groupby(group_cols, as_index=False)
        .agg(
            n=("snip_id", "size"),
            argmax_acc=("argmax_correct", "mean"),
            coverage=("covered", "mean"),
            mean_set_size=("set_size", "mean"),
            mean_q_true=("q_true", "mean"),
            median_q_true=("q_true", "median"),
            q10_true=("q_true", lambda x: x.quantile(0.10)),
            q25_true=("q_true", lambda x: x.quantile(0.25)),
            q75_true=("q_true", lambda x: x.quantile(0.75)),
            q90_true=("q_true", lambda x: x.quantile(0.90)),
            mean_true_label_rank=("true_label_rank", "mean"),
            rank1_rate=("true_label_rank", lambda x: (x == 1).mean()),
            rank2_or_better_rate=("true_label_rank", lambda x: (x <= 2).mean()),
            rank4_rate=("true_label_rank", lambda x: (x == 4).mean()),
        )
    )


def neighbor_geometry_rows(
    data: pd.DataFrame,
    feature_cols: list[str],
    calibration_frac: float,
    seed: int,
    max_rank: int,
) -> pd.DataFrame:
    rows = []
    experiments = sorted(data[EXPERIMENT_COL].dropna().astype(str).unique())
    for fold in experiments:
        train_all = data[data[EXPERIMENT_COL] != fold].copy()
        query = data[data[EXPERIMENT_COL] == fold].copy().reset_index(drop=True)
        ref, _ = split_reference_calibration(train_all, calibration_frac, seed)
        ref = ref.reset_index(drop=True)
        if len(ref) <= 1 or len(query) == 0:
            continue

        k_query = min(max_rank, len(ref))
        nn = NearestNeighbors(n_neighbors=k_query, metric="euclidean")
        nn.fit(ref[feature_cols].to_numpy(dtype=float))
        dists, idxs = nn.kneighbors(query[feature_cols].to_numpy(dtype=float))
        ref_labels = ref[LABEL_COL].astype(str).to_numpy()

        for qi, qrow in query.iterrows():
            true_label = str(qrow[LABEL_COL])
            labels_ranked = ref_labels[idxs[qi]]
            dists_ranked = dists[qi]
            same = labels_ranked == true_label
            np_mask = labels_ranked == "Not Penetrant"

            first_same = int(np.where(same)[0][0] + 1) if same.any() else np.nan
            first_np = int(np.where(np_mask)[0][0] + 1) if np_mask.any() else np.nan
            same_d = dists_ranked[same]
            np_d = dists_ranked[np_mask]

            rec = {
                "heldout_experiment_id": fold,
                "snip_id": qrow[SNIP_COL],
                "embryo_id": qrow[EMBRYO_COL],
                "hpf": float(qrow[TIME_COL]),
                "hpf_bin": qrow["_hpf_bin"],
                "coarse_hpf_bin": coarse_hpf_bin(float(qrow[TIME_COL])),
                "true_label": true_label,
                "rank_first_true_label_neighbor": first_same,
                "rank_first_np_neighbor": first_np,
                "distance_first_true_label_neighbor": float(same_d[0]) if len(same_d) else np.nan,
                "distance_first_np_neighbor": float(np_d[0]) if len(np_d) else np.nan,
                "mean_distance_to_true_label_neighbors_top200": float(same_d.mean()) if len(same_d) else np.nan,
                "mean_distance_to_np_neighbors_top200": float(np_d.mean()) if len(np_d) else np.nan,
            }
            for k in K_VALUES:
                kk = min(k, len(labels_ranked))
                top = labels_ranked[:kk]
                rec[f"n_true_label_neighbors_top{k}"] = int((top == true_label).sum())
                rec[f"frac_true_label_neighbors_top{k}"] = float((top == true_label).mean())
                rec[f"n_np_neighbors_top{k}"] = int((top == "Not Penetrant").sum())
                rec[f"frac_np_neighbors_top{k}"] = float((top == "Not Penetrant").mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize_neighbor_geometry(geom: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "snip_id": "size",
        "rank_first_true_label_neighbor": "mean",
        "rank_first_np_neighbor": "mean",
        "distance_first_true_label_neighbor": "mean",
        "distance_first_np_neighbor": "mean",
    }
    for k in K_VALUES:
        agg[f"frac_true_label_neighbors_top{k}"] = "mean"
        agg[f"frac_np_neighbors_top{k}"] = "mean"
    out = (
        geom
        .groupby(["true_label", "coarse_hpf_bin"], as_index=False)
        .agg(agg)
        .rename(columns={"snip_id": "n"})
    )
    return out


def rescue_rows(rank_df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    wide = {}
    for method in ["knn_q", "multiclass_q"]:
        cols = [
            "snip_id", "embryo_id", "heldout_experiment_id", "hpf", "hpf_bin",
            "coarse_hpf_bin", "true_label", "argmax_label", "argmax_correct",
            "prediction_set", "set_size", "covered", "q_true", "true_label_rank",
            *[f"q_{label}" for label in labels],
        ]
        sub = rank_df[rank_df["method"] == method][cols].copy()
        rename = {
            c: f"{method}_{c}"
            for c in sub.columns
            if c not in ["snip_id", "embryo_id", "heldout_experiment_id", "hpf", "hpf_bin", "coarse_hpf_bin", "true_label"]
        }
        wide[method] = sub.rename(columns=rename)
    out = wide["knn_q"].merge(
        wide["multiclass_q"],
        on=["snip_id", "embryo_id", "heldout_experiment_id", "hpf", "hpf_bin", "coarse_hpf_bin", "true_label"],
        how="inner",
    )

    def group(row):
        k = bool(row["knn_q_argmax_correct"])
        m = bool(row["multiclass_q_argmax_correct"])
        if k and m:
            return "both_right"
        if k and not m:
            return "knn_right_multiclass_wrong"
        if not k and m:
            return "knn_wrong_multiclass_right"
        return "both_wrong"

    out["rescue_group"] = out.apply(group, axis=1)
    out["rank_delta_multiclass_minus_knn"] = (
        out["multiclass_q_true_label_rank"] - out["knn_q_true_label_rank"]
    )
    out["q_true_delta_multiclass_minus_knn"] = out["multiclass_q_q_true"] - out["knn_q_q_true"]
    return out


def summarize_rescue(rescue: pd.DataFrame) -> pd.DataFrame:
    return (
        rescue
        .groupby(["true_label", "coarse_hpf_bin", "rescue_group"], as_index=False)
        .agg(
            n=("snip_id", "size"),
            mean_hpf=("hpf", "mean"),
            knn_mean_q_true=("knn_q_q_true", "mean"),
            multiclass_mean_q_true=("multiclass_q_q_true", "mean"),
            mean_q_true_delta=("q_true_delta_multiclass_minus_knn", "mean"),
            knn_mean_rank=("knn_q_true_label_rank", "mean"),
            multiclass_mean_rank=("multiclass_q_true_label_rank", "mean"),
            mean_rank_delta=("rank_delta_multiclass_minus_knn", "mean"),
            knn_coverage=("knn_q_covered", "mean"),
            multiclass_coverage=("multiclass_q_covered", "mean"),
            knn_mean_set_size=("knn_q_set_size", "mean"),
            multiclass_mean_set_size=("multiclass_q_set_size", "mean"),
        )
    )


def set_composition(pred: pd.DataFrame) -> pd.DataFrame:
    return (
        pred
        .groupby(["method", "prediction_set"], as_index=False)
        .agg(n=("snip_id", "size"))
        .assign(frac=lambda x: x["n"] / x.groupby("method")["n"].transform("sum"))
        .sort_values(["method", "frac"], ascending=[True, False])
    )


def write_report(
    path: Path,
    rank_summary: pd.DataFrame,
    geom_summary: pd.DataFrame,
    rescue_summary: pd.DataFrame,
    set_comp: pd.DataFrame,
) -> None:
    lines = [
        "# Q Failure Diagnostic Report",
        "",
        "## True-Label Rank",
        "",
        "Pooled across HPF bins, true-label rank/coverage by method:",
        "",
    ]
    pooled_rank = (
        rank_summary
        .groupby(["method", "true_label"], as_index=False)
        .apply(_weighted_rank_summary)
        .reset_index(drop=True)
    )
    lines.append(pooled_rank.round(3).to_markdown(index=False))

    lines.extend([
        "",
        "## Neighbor Geometry",
        "",
        "Mean first same-label neighbor rank and top-K same-label fraction:",
        "",
    ])
    geom_pooled = (
        geom_summary
        .groupby("true_label", as_index=False)
        .apply(_weighted_geom_summary)
        .reset_index(drop=True)
    )
    keep_geom = [
        "true_label", "n", "rank_first_true_label_neighbor",
        "frac_true_label_neighbors_top15", "frac_true_label_neighbors_top50",
        "frac_true_label_neighbors_top200", "frac_np_neighbors_top15",
    ]
    lines.append(geom_pooled[keep_geom].round(3).to_markdown(index=False))

    lines.extend([
        "",
        "## Rescue Groups",
        "",
        "Counts for rare/mixed labels:",
        "",
    ])
    rescue_focus = rescue_summary[
        rescue_summary["true_label"].isin(["Low_to_High", "Intermediate"])
    ]
    rescue_counts = (
        rescue_focus
        .groupby(["true_label", "rescue_group"], as_index=False)["n"]
        .sum()
        .sort_values(["true_label", "n"], ascending=[True, False])
    )
    lines.append(rescue_counts.to_markdown(index=False))

    lines.extend([
        "",
        "## Set Composition",
        "",
        "Top conformal sets by method:",
        "",
    ])
    top_sets = set_comp.groupby("method", as_index=False).head(8)
    lines.append(top_sets.round(3).to_markdown(index=False))
    lines.append("")
    path.write_text("\n".join(lines))


def _weighted_rank_summary(g: pd.DataFrame) -> pd.Series:
    w = g["n"].to_numpy(dtype=float)
    return pd.Series({
        "method": g["method"].iloc[0] if "method" in g else None,
        "true_label": g["true_label"].iloc[0] if "true_label" in g else None,
        "n": int(w.sum()),
        "argmax_acc": np.average(g["argmax_acc"], weights=w),
        "coverage": np.average(g["coverage"], weights=w),
        "mean_set_size": np.average(g["mean_set_size"], weights=w),
        "mean_q_true": np.average(g["mean_q_true"], weights=w),
        "mean_true_label_rank": np.average(g["mean_true_label_rank"], weights=w),
        "rank1_rate": np.average(g["rank1_rate"], weights=w),
        "rank2_or_better_rate": np.average(g["rank2_or_better_rate"], weights=w),
        "rank4_rate": np.average(g["rank4_rate"], weights=w),
    })


def _weighted_geom_summary(g: pd.DataFrame) -> pd.Series:
    w = g["n"].to_numpy(dtype=float)
    out = {"true_label": g["true_label"].iloc[0] if "true_label" in g else None, "n": int(w.sum())}
    for col in g.columns:
        if col in {"true_label", "coarse_hpf_bin", "n"}:
            continue
        out[col] = np.average(g[col], weights=w)
    return pd.Series(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=PREDICTION_PATH)
    p.add_argument("--calibration-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bin-width", type=float, default=4.0)
    p.add_argument("--max-rank", type=int, default=200)
    p.add_argument("--output-prefix", default="q_diagnostic")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    rank = true_label_rank_rows(pred, MAIN_LABELS)
    rank_summary = summarize_rank(rank)
    rescue = rescue_rows(rank, MAIN_LABELS)
    rescue_summary = summarize_rescue(rescue)
    sets = set_composition(pred)

    data = pd.read_csv(DATA_PATH, low_memory=False)
    feature_cols = get_feature_cols(data)
    keep = [SNIP_COL, EMBRYO_COL, EXPERIMENT_COL, TIME_COL, LABEL_COL, *feature_cols]
    data = data[data[LABEL_COL].isin(MAIN_LABELS)].copy()
    data = data[(data[TIME_COL] >= MIN_HPF) & (data[TIME_COL] <= MAX_HPF)].copy()
    data = data[keep].dropna(subset=[TIME_COL, LABEL_COL, *feature_cols]).reset_index(drop=True)
    data = add_hpf_bin(data, args.bin_width)
    geom = neighbor_geometry_rows(
        data,
        feature_cols,
        calibration_frac=args.calibration_frac,
        seed=args.seed,
        max_rank=args.max_rank,
    )
    geom_summary = summarize_neighbor_geometry(geom)

    out = HERE
    rank.to_csv(out / f"{args.output_prefix}_true_label_rank.csv", index=False)
    rank_summary.to_csv(out / f"{args.output_prefix}_true_label_rank_summary.csv", index=False)
    geom.to_csv(out / f"{args.output_prefix}_neighbor_geometry.csv", index=False)
    geom_summary.to_csv(out / f"{args.output_prefix}_neighbor_geometry_summary.csv", index=False)
    rescue.to_csv(out / f"{args.output_prefix}_rescue_groups.csv", index=False)
    rescue_summary.to_csv(out / f"{args.output_prefix}_rescue_group_summary.csv", index=False)
    sets.to_csv(out / f"{args.output_prefix}_set_composition.csv", index=False)
    write_report(
        out / f"{args.output_prefix}_report.md",
        rank_summary,
        geom_summary,
        rescue_summary,
        sets,
    )

    print("Saved diagnostic outputs with prefix:", out / args.output_prefix)
    print("\nRank summary (pooled over HPF bins):")
    pooled = (
        rank_summary
        .groupby(["method", "true_label"], as_index=False)
        .apply(_weighted_rank_summary)
        .reset_index(drop=True)
    )
    with pd.option_context("display.width", 180, "display.max_columns", None):
        print(pooled.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
