from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze.classification.misclassification.io import (
    infer_class_labels_from_predictions,
    load_stage1_metadata,
)
from analyze.classification.misclassification.trajectory import (
    STAGE_DELTA,
    STAGE_HARD,
    STAGE_RESIDUAL,
    STAGE_RESIDUAL_DTW,
    STAGE_SOFT,
    VALID_STAGES,
    compute_rolling_window_destination_confusion_significance,
    compute_rolling_window_wrong_rate_significance,
    run_stage_geometry,
)
from analyze.classification.viz.trajectory import (
    plot_cluster_feature_trends,
    save_rolling_destination_significance_counts,
    save_rolling_window_significance_counts,
    save_pca_scatter,
    save_wrong_rate_null_diagnostics,
)


def _as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes", "y"})



def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return ""



def _load_plot_dataframe(
    csv_path: Path,
    *,
    time_min: float,
    time_max: float,
    embryo_id_col: str,
    time_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    use_mask = _as_bool(df["use_embryo_flag"]) if "use_embryo_flag" in df.columns else pd.Series(True, index=df.index)
    dead_mask = _as_bool(df["dead_flag2"]) if "dead_flag2" in df.columns else pd.Series(False, index=df.index)

    required = [embryo_id_col, time_col, "baseline_deviation_um", "total_length_um"]
    if "phenotype" in df.columns:
        required.append("phenotype")

    work = df[use_mask & (~dead_mask)].copy()
    work = work.dropna(subset=required)
    work = work[(work[time_col] >= float(time_min)) & (work[time_col] <= float(time_max))].copy()

    work[embryo_id_col] = work[embryo_id_col].astype(str)
    if "phenotype" in work.columns and "true_class" not in work.columns:
        work["true_class"] = work["phenotype"].astype(str)
    return work



def _parse_stages(raw: str) -> list[str]:
    stages = [s.strip() for s in raw.split(",") if s.strip()]
    if not stages:
        raise ValueError("No stages provided")
    bad = [s for s in stages if s not in VALID_STAGES]
    if bad:
        raise ValueError(f"Unknown stages: {bad}. Valid: {sorted(VALID_STAGES)}")
    return stages



def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")



def _parse_destination_pairs(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    txt = raw.strip()
    if not txt:
        return pairs
    for token in [x.strip() for x in txt.split(",") if x.strip()]:
        if "->" not in token:
            raise ValueError(
                f"Invalid destination pair '{token}'. Expected format: Source->Target"
            )
        src, dst = [p.strip() for p in token.split("->", 1)]
        if not src or not dst:
            raise ValueError(f"Invalid destination pair '{token}'. Empty source/target.")
        pairs.append((src, dst))
    return pairs



def _make_stage_report(
    *,
    stage_result,
    stage_mode: str,
) -> dict:
    report = {
        "stage": stage_mode,
        "n_embryos": int(len(stage_result.stage_table)),
        "n_features": int(len(stage_result.feature_columns)),
        "n_classes": int(len(stage_result.class_labels)),
        "n_time_bins": int(len(stage_result.time_bins)),
        "time_bins": [int(x) for x in stage_result.time_bins],
        "explained_variance_ratio": [float(x) if pd.notna(x) else None for x in stage_result.explained_variance_ratio],
        "corr_pc1_wrong_frac": float(stage_result.stage_table["corr_pc1_wrong_frac"].iloc[0])
        if "corr_pc1_wrong_frac" in stage_result.stage_table.columns
        else None,
        "n_wrong_rate_window_significant": int(
            stage_result.stage_table["is_wrong_significant_in_window_perm"].sum()
        )
        if "is_wrong_significant_in_window_perm" in stage_result.stage_table.columns
        else 0,
        "trend_interpretability_score": None,
        "trend_interpretability_note": "",
        "metrics_by_k": stage_result.metrics_by_k,
    }
    return report



def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged CEP290 misclassification geometry analysis")
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path(
            "results/mcolon/20260222_dev_consistently_misclassified_embryos/real_runs/"
            "cep290_phenotype_all_vs_rest_stage1_focus36_60"
        ),
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=Path(
            "results/mcolon/20260213_subtle_phenotype_methods/input_data/experiments/"
            "cep290_20251229/input_core.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/mcolon/20260222_dev_consistently_misclassified_embryos/real_runs/trajectory_stages"
        ),
    )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--stages", type=str, default=f"{STAGE_HARD},{STAGE_SOFT},{STAGE_DELTA},{STAGE_RESIDUAL}")
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--wrong-rate-n-permutations", type=int, default=400)
    parser.add_argument("--wrong-rate-q-threshold", type=float, default=0.10)
    parser.add_argument("--wrong-rate-window-min", type=float, default=None)
    parser.add_argument("--wrong-rate-window-max", type=float, default=None)
    parser.add_argument("--rolling-window-hpf", type=float, default=5.0)
    parser.add_argument(
        "--destination-pairs",
        type=str,
        default="Low_to_High->High_to_Low,Intermediate->Low_to_High",
        help="Comma-separated Source->Target pairs for destination confusion rolling null",
    )
    parser.add_argument("--destination-rolling-window-hpf", type=float, default=5.0)
    parser.add_argument("--destination-n-permutations", type=int, default=300)
    parser.add_argument("--destination-q-threshold", type=float, default=0.10)
    parser.add_argument("--dtw-window", type=int, default=1)
    parser.add_argument("--time-min", type=float, default=36.0)
    parser.add_argument("--time-max", type=float, default=60.0)
    parser.add_argument("--time-col", type=str, default="predicted_stage_hpf")
    parser.add_argument("--embryo-id-col", type=str, default="embryo_id")
    parser.add_argument("--group-color-by", type=str, default="true_class")
    parser.add_argument("--facet-col-override", type=str, default="")
    args = parser.parse_args()

    stages = _parse_stages(args.stages)
    destination_pairs = _parse_destination_pairs(args.destination_pairs)
    k_values = tuple(range(int(args.k_min), int(args.k_max) + 1))

    stage1_dir = args.stage1_dir
    pred_path = stage1_dir / "embryo_predictions_augmented.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    pred_df = pd.read_parquet(pred_path)
    pred_df["embryo_id"] = pred_df["embryo_id"].astype(str)

    try:
        stage1_meta = load_stage1_metadata(stage1_dir)
        class_labels = list(stage1_meta.class_labels)
        class_labels_source = "null_metadata"
    except FileNotFoundError:
        class_labels = infer_class_labels_from_predictions(pred_df)
        class_labels_source = "inferred_union"

    run_id = args.run_id.strip() or _default_run_id()
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_df = _load_plot_dataframe(
        args.raw_csv,
        time_min=float(args.time_min),
        time_max=float(args.time_max),
        embryo_id_col=args.embryo_id_col,
        time_col=args.time_col,
    )
    raw_df = raw_df[raw_df[args.embryo_id_col].astype(str).isin(set(pred_df["embryo_id"].unique()))].copy()

    summary_rows: list[dict] = []

    for idx, stage_mode in enumerate(stages):
        stage_name = f"stage_{idx:02d}_{stage_mode}"
        stage_dir = run_dir / stage_name
        tables_dir = stage_dir / "tables"
        plots_dir = stage_dir / "plots"
        tables_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        result = run_stage_geometry(
            pred_df,
            class_labels=class_labels,
            stage_mode=stage_mode,
            k_values=k_values,
            pca_components=int(args.pca_components),
            random_state=int(args.random_state),
            kmeans_n_init=int(args.kmeans_n_init),
            dtw_window=int(args.dtw_window),
            wrong_rate_n_permutations=int(args.wrong_rate_n_permutations),
            wrong_rate_q_threshold=float(args.wrong_rate_q_threshold),
            wrong_rate_window_min=args.wrong_rate_window_min,
            wrong_rate_window_max=args.wrong_rate_window_max,
        )

        # Persist stage core tables.
        stage_table = result.stage_table.copy()
        stage_table.to_parquet(tables_dir / "embryo_stage_table.parquet", index=False)
        stage_table.to_csv(tables_dir / "embryo_stage_table.csv", index=False)

        pd.DataFrame({"feature_column": result.feature_columns}).to_csv(
            tables_dir / "feature_columns.csv", index=False
        )

        metrics_df = pd.DataFrame(result.metrics_by_k)
        metrics_df.to_csv(tables_dir / "kmeans_metrics.csv", index=False)

        if result.baseline_mu is not None:
            result.baseline_mu.to_parquet(tables_dir / "baseline_mu.parquet", index=False)

        if result.distance_matrix is not None:
            np.save(tables_dir / "distance_matrix.npy", result.distance_matrix)

        report = _make_stage_report(stage_result=result, stage_mode=stage_mode)
        (stage_dir / "stage_acceptance_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

        # PCA scatter plots.
        cluster_color_col = "cluster_k3" if "cluster_k3" in stage_table.columns else None
        if cluster_color_col is None:
            cluster_cols = [c for c in stage_table.columns if c.startswith("cluster_k")]
            cluster_color_col = cluster_cols[0] if cluster_cols else None

        if cluster_color_col is not None:
            save_pca_scatter(
                stage_table,
                color_col=cluster_color_col,
                output_path=plots_dir / "pca_cluster_k3.png",
                title=f"{stage_mode}: PCA colored by {cluster_color_col}",
            )

        for col, fname in [
            ("is_wrong_significant_in_window_perm", "pca_wrong_rate_window_significant_perm.png"),
            ("wrong_rate_window_sig_tier", "pca_wrong_rate_window_sig_tier.png"),
            ("is_wrong_more_often", "pca_is_wrong_more_often_heuristic.png"),
            ("true_class", "pca_true_class.png"),
        ]:
            if col in stage_table.columns:
                save_pca_scatter(
                    stage_table,
                    color_col=col,
                    output_path=plots_dir / fname,
                    title=f"{stage_mode}: PCA colored by {col}",
                )

        if {"wrong_frac", "wrong_rate_window_null_mean", "qval_wrong_rate_window_perm"} <= set(stage_table.columns):
            save_wrong_rate_null_diagnostics(
                stage_table,
                output_path=plots_dir / "wrong_rate_window_null_diagnostics.png",
                title=f"{stage_mode}: wrong-rate null diagnostics (window permutation)",
            )

        # Trend plots for each cluster_k2..k5 (or available k-range).
        facet_override = args.facet_col_override.strip() or None
        for k in k_values:
            cluster_col = f"cluster_k{k}"
            if cluster_col not in stage_table.columns:
                continue
            plot_cluster_feature_trends(
                raw_df=raw_df,
                stage_table=stage_table,
                cluster_col=cluster_col,
                output_path=plots_dir / f"features_over_time_cluster_k{k}.png",
                features=["baseline_deviation_um", "total_length_um"],
                time_col=args.time_col,
                embryo_id_col=args.embryo_id_col,
                group_color_by=args.group_color_by,
                facet_col_override=facet_override,
            )

        row: dict[str, object] = {
            "stage": stage_mode,
            "n_embryos": int(len(stage_table)),
            "n_features": int(len(result.feature_columns)),
            "n_classes": int(len(result.class_labels)),
            "n_time_bins": int(len(result.time_bins)),
            "corr_pc1_wrong_frac": float(report["corr_pc1_wrong_frac"]) if report["corr_pc1_wrong_frac"] is not None else np.nan,
            "n_wrong_rate_window_significant": int(report["n_wrong_rate_window_significant"]),
            "trend_interpretability_score": np.nan,
            "trend_interpretability_note": "",
        }

        for i, v in enumerate(result.explained_variance_ratio, start=1):
            row[f"explained_var_pc{i}"] = float(v) if pd.notna(v) else np.nan

        for m in result.metrics_by_k:
            k = int(m["k"])
            row[f"inertia_k{k}"] = float(m["inertia"])
            row[f"silhouette_k{k}"] = float(m["silhouette"]) if pd.notna(m["silhouette"]) else np.nan
            row[f"dbi_k{k}"] = float(m["davies_bouldin"]) if pd.notna(m["davies_bouldin"]) else np.nan

        summary_rows.append(row)

    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_dir / "stage_comparison.csv", index=False)

    if float(args.rolling_window_hpf) > 0:
        rolling = compute_rolling_window_wrong_rate_significance(
            pred_df,
            class_labels=class_labels,
            window_hpf=float(args.rolling_window_hpf),
            n_permutations=int(args.wrong_rate_n_permutations),
            random_state=int(args.random_state),
            q_threshold=float(args.wrong_rate_q_threshold),
        )
        rolling_csv = summary_dir / f"rolling_wrong_rate_significance_{args.rolling_window_hpf:g}hpf.csv"
        rolling.to_csv(rolling_csv, index=False)
        save_rolling_window_significance_counts(
            rolling,
            output_path=summary_dir / f"rolling_wrong_rate_significance_counts_{args.rolling_window_hpf:g}hpf.png",
            title=f"Rolling-window wrong-rate significance ({args.rolling_window_hpf:g} hpf)",
        )

    if destination_pairs and float(args.destination_rolling_window_hpf) > 0:
        for pair_idx, (src_class, dst_class) in enumerate(destination_pairs):
            rolling_dest = compute_rolling_window_destination_confusion_significance(
                pred_df,
                class_labels=class_labels,
                source_class=src_class,
                target_class=dst_class,
                window_hpf=float(args.destination_rolling_window_hpf),
                n_permutations=int(args.destination_n_permutations),
                random_state=int(args.random_state) + 1000 + pair_idx * 100,
                q_threshold=float(args.destination_q_threshold),
            )
            safe_pair = f"{src_class}__to__{dst_class}".replace(" ", "_")
            csv_path = summary_dir / (
                f"rolling_destination_confusion_{safe_pair}_{args.destination_rolling_window_hpf:g}hpf.csv"
            )
            rolling_dest.to_csv(csv_path, index=False)
            save_rolling_destination_significance_counts(
                rolling_dest,
                output_path=summary_dir
                / f"rolling_destination_confusion_counts_{safe_pair}_{args.destination_rolling_window_hpf:g}hpf.png",
                title=(
                    f"Rolling destination confusion significance ({src_class} -> {dst_class}, "
                    f"{args.destination_rolling_window_hpf:g} hpf)"
                ),
            )

    metadata = {
        "run_id": run_id,
        "stage1_dir": str(stage1_dir),
        "raw_csv": str(args.raw_csv),
        "stages": stages,
        "k_values": list(k_values),
        "pca_components": int(args.pca_components),
        "kmeans_n_init": int(args.kmeans_n_init),
        "random_state": int(args.random_state),
        "wrong_rate_n_permutations": int(args.wrong_rate_n_permutations),
        "wrong_rate_q_threshold": float(args.wrong_rate_q_threshold),
        "wrong_rate_window": [args.wrong_rate_window_min, args.wrong_rate_window_max],
        "rolling_window_hpf": float(args.rolling_window_hpf),
        "destination_pairs": [f"{s}->{d}" for s, d in destination_pairs],
        "destination_rolling_window_hpf": float(args.destination_rolling_window_hpf),
        "destination_n_permutations": int(args.destination_n_permutations),
        "destination_q_threshold": float(args.destination_q_threshold),
        "dtw_window": int(args.dtw_window),
        "time_window": [float(args.time_min), float(args.time_max)],
        "time_col": args.time_col,
        "embryo_id_col": args.embryo_id_col,
        "group_color_by": args.group_color_by,
        "facet_col_override": args.facet_col_override,
        "class_labels": class_labels,
        "class_labels_source": class_labels_source,
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    print(f"Saved trajectory stage run to: {run_dir}")
    print(f"Summary: {summary_dir / 'stage_comparison.csv'}")


if __name__ == "__main__":
    main()
