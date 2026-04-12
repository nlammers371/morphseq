from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


EXPERIMENT_IDS = ["20260304", "20260306"]
EXPERIMENT_LABEL = "20260304_20260306"
FEATURE_SETS: dict[str, str | list[str]] = {
    "curvature": ["baseline_deviation_normalized"],
    "length": ["total_length_um"],
    "embedding": "z_mu_b",
}


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return ""


def _normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower().replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")

    if g in {
        "ab_inj_ctrl",
        "wik-ab_inj_ctrl",
        "wik-ab_ctrl_inj",
        "wik_ab_inj_ctrl",
        "wik_ab_ctrl_inj",
    }:
        return "inj_ctrl"

    return g.replace("wik-ab", "wik_ab")


def _time_bin_definition_from_df(df: pd.DataFrame, bin_width: float) -> list[int]:
    bins = sorted(df["time_bin"].dropna().astype(int).unique().tolist())
    if not bins:
        return []
    return bins + [bins[-1] + int(round(bin_width))]


def _time_edges_hash(edges: list[int]) -> str:
    raw = json.dumps(edges, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_dataframe(project_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for exp_id in EXPERIMENT_IDS:
        data_path = (
            project_root
            / "morphseq_playground"
            / "metadata"
            / "build06_output"
            / f"df03_final_output_with_latents_{exp_id}.csv"
        )
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        part = pd.read_csv(data_path, low_memory=False)
        if "experiment_id" in part.columns:
            part = part[part["experiment_id"].astype(str) == exp_id].copy()
        else:
            part["experiment_id"] = exp_id
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)

    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()

    required = {
        "embryo_id",
        "genotype",
        "predicted_stage_hpf",
        "baseline_deviation_normalized",
        "total_length_um",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    return df


def _write_stage1_artifacts(
    *,
    analysis,
    stage1_dir: Path,
    feature_set: str,
    bin_width: float,
    classification_permutations: int,
    random_state: int,
) -> pd.DataFrame:
    stage1_dir.mkdir(parents=True, exist_ok=True)
    analysis.save(stage1_dir, overwrite=True)

    pred_df = analysis.layers["multiclass_predictions"].copy()
    pred_df.to_parquet(stage1_dir / "embryo_predictions_augmented.parquet", index=False)

    confusion_df = analysis.layers.get("confusion")
    if confusion_df is not None:
        confusion_df.to_parquet(stage1_dir / "confusion_profile.parquet", index=False)

    class_labels = sorted(pred_df["true_class"].astype(str).unique().tolist())
    time_edges = _time_bin_definition_from_df(pred_df, bin_width)
    null_dir = stage1_dir / "null"
    null_dir.mkdir(parents=True, exist_ok=True)
    null_metadata = {
        "class_labels": class_labels,
        "time_bin_definition": time_edges,
        "time_bin_center_formula": "midpoint",
        "time_bin_edges_sha256": _time_edges_hash(time_edges),
        "seed": int(random_state),
        "n_permutations": int(classification_permutations),
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(),
        "schema_version": "classification_v1",
        "feature_set": feature_set,
        "experiment_ids": EXPERIMENT_IDS,
        "experiment_label": EXPERIMENT_LABEL,
    }
    (null_dir / "null_metadata.json").write_text(
        json.dumps(null_metadata, indent=2, sort_keys=True) + "\n"
    )
    return pred_df


def _build_summary_tables(
    *,
    stage2_dir: Path,
    feature_set: str,
) -> dict[str, Path]:
    tables_dir = stage2_dir / "tables"
    flagged_path = tables_dir / "flagged_embryos.csv"
    per_embryo_path = tables_dir / "per_embryo_metrics.csv"
    confusion_path = tables_dir / "confusion_enrichment.csv"

    flagged = pd.read_csv(flagged_path) if flagged_path.exists() else pd.DataFrame()
    per_embryo = pd.read_csv(per_embryo_path) if per_embryo_path.exists() else pd.DataFrame()
    confusion = pd.read_csv(confusion_path) if confusion_path.exists() else pd.DataFrame()

    outputs: dict[str, Path] = {}

    if not flagged.empty:
        flagged_ranked = flagged.sort_values(
            ["true_class", "wrong_rate", "top_confused_frac", "n_wrong"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        ranked_path = tables_dir / "flagged_embryos_ranked.csv"
        flagged_ranked.to_csv(ranked_path, index=False)
        outputs["flagged_ranked"] = ranked_path

        summary_by_pair = (
            flagged_ranked.groupby(["true_class", "top_confused_as"], as_index=False)
            .agg(
                n_flagged=("embryo_id", "nunique"),
                mean_wrong_rate=("wrong_rate", "mean"),
                mean_top_confused_frac=("top_confused_frac", "mean"),
            )
            .sort_values(["n_flagged", "mean_wrong_rate"], ascending=[False, False])
        )
        pair_path = tables_dir / "flagged_summary_by_true_and_confused_as.csv"
        summary_by_pair.to_csv(pair_path, index=False)
        outputs["flagged_pair_summary"] = pair_path

    if not per_embryo.empty:
        genotype_summary = (
            per_embryo.groupby("true_class", as_index=False)
            .agg(
                n_embryos=("embryo_id", "nunique"),
                mean_wrong_rate=("wrong_rate", "mean"),
                median_wrong_rate=("wrong_rate", "median"),
                n_flagged=("is_flagged", "sum"),
            )
            .sort_values(["n_flagged", "mean_wrong_rate"], ascending=[False, False])
        )
        genotype_path = tables_dir / "per_genotype_misclassification_summary.csv"
        genotype_summary.to_csv(genotype_path, index=False)
        outputs["genotype_summary"] = genotype_path

    summary_lines = [
        f"feature_set: {feature_set}",
        f"stage2_dir: {stage2_dir}",
        "",
    ]

    if per_embryo.empty:
        summary_lines.append("No per-embryo metrics were produced.")
    else:
        summary_lines.append("per_genotype_summary:")
        for _, row in (
            per_embryo.groupby("true_class", as_index=False)
            .agg(
                n_embryos=("embryo_id", "nunique"),
                mean_wrong_rate=("wrong_rate", "mean"),
                n_flagged=("is_flagged", "sum"),
            )
            .sort_values(["n_flagged", "mean_wrong_rate"], ascending=[False, False])
            .iterrows()
        ):
            summary_lines.append(
                f"- {row['true_class']}: n_embryos={int(row['n_embryos'])}, "
                f"mean_wrong_rate={float(row['mean_wrong_rate']):.3f}, "
                f"n_flagged={int(row['n_flagged'])}"
            )

    summary_lines += ["", "top_flagged_embryos:"]
    if flagged.empty:
        summary_lines.append("- none")
    else:
        for _, row in (
            flagged.sort_values(
                ["wrong_rate", "top_confused_frac", "n_wrong"],
                ascending=[False, False, False],
            )
            .head(12)
            .iterrows()
        ):
            summary_lines.append(
                f"- {row['embryo_id']} [{row['true_class']} -> {row['top_confused_as'] or 'mixed'}]: "
                f"wrong_rate={float(row['wrong_rate']):.3f}, "
                f"n_wrong={int(row['n_wrong'])}, "
                f"top_confused_frac={float(row['top_confused_frac']):.3f}"
            )

    summary_lines += ["", "confusion_enrichment_top_rows:"]
    if confusion.empty:
        summary_lines.append("- none")
    else:
        sort_cols = ["qval", "observed_frac", "n_flagged"]
        available_sort_cols = [c for c in sort_cols if c in confusion.columns]
        confusion_sorted = confusion.sort_values(
            available_sort_cols,
            ascending=[True, False, False][: len(available_sort_cols)],
            na_position="last",
        )
        for _, row in confusion_sorted.head(12).iterrows():
            qval = row["qval"] if "qval" in row and pd.notna(row["qval"]) else float("nan")
            summary_lines.append(
                f"- {row['true_class']} -> {row['confused_as']}: "
                f"observed_frac={float(row['observed_frac']):.3f}, "
                f"expected_frac={float(row['expected_frac']):.3f}, "
                f"n_flagged={int(row['n_flagged'])}, "
                f"qval={qval:.4f}"
            )

    summary_path = stage2_dir / "summary_consistent_misclassification.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    outputs["summary_text"] = summary_path
    return outputs


def _write_cross_feature_summaries(
    *,
    output_root: Path,
    flagged_tables: list[pd.DataFrame],
) -> dict[str, Path]:
    if not flagged_tables:
        return {}

    combined = pd.concat(flagged_tables, ignore_index=True)
    combined = combined.sort_values(
        ["feature_set", "true_class", "wrong_rate", "top_confused_frac", "n_wrong"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)

    combined_path = output_root / "flagged_embryos_all_feature_sets.csv"
    combined.to_csv(combined_path, index=False)

    overlap = (
        combined.groupby(["embryo_id", "true_class"], as_index=False)
        .agg(
            n_feature_sets=("feature_set", "nunique"),
            feature_sets=("feature_set", lambda s: ",".join(sorted(set(s.astype(str))))),
            mean_wrong_rate=("wrong_rate", "mean"),
            max_wrong_rate=("wrong_rate", "max"),
            top_confused_labels=("top_confused_as", lambda s: ",".join(sorted(set(x for x in s.astype(str) if x)))),
        )
        .sort_values(["n_feature_sets", "max_wrong_rate", "mean_wrong_rate"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    overlap_path = output_root / "flagged_embryo_overlap_across_feature_sets.csv"
    overlap.to_csv(overlap_path, index=False)

    pair_summary = (
        combined.groupby(["feature_set", "true_class", "top_confused_as"], as_index=False)
        .agg(
            n_flagged=("embryo_id", "nunique"),
            mean_wrong_rate=("wrong_rate", "mean"),
            mean_top_confused_frac=("top_confused_frac", "mean"),
        )
        .sort_values(["n_flagged", "mean_wrong_rate"], ascending=[False, False])
    )
    pair_path = output_root / "flagged_summary_by_feature_true_and_confused_as.csv"
    pair_summary.to_csv(pair_path, index=False)

    summary_lines = ["cross_feature_overlap:"]
    for _, row in overlap.head(20).iterrows():
        summary_lines.append(
            f"- {row['embryo_id']} [{row['true_class']}]: "
            f"n_feature_sets={int(row['n_feature_sets'])}, "
            f"feature_sets={row['feature_sets']}, "
            f"max_wrong_rate={float(row['max_wrong_rate']):.3f}, "
            f"top_confused={row['top_confused_labels'] or 'mixed'}"
        )

    summary_lines += ["", "top_cross_feature_pairs:"]
    for _, row in pair_summary.head(20).iterrows():
        summary_lines.append(
            f"- [{row['feature_set']}] {row['true_class']} -> {row['top_confused_as'] or 'mixed'}: "
            f"n_flagged={int(row['n_flagged'])}, "
            f"mean_wrong_rate={float(row['mean_wrong_rate']):.3f}"
        )

    summary_path = output_root / "cross_feature_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    return {
        "combined_flagged": combined_path,
        "overlap": overlap_path,
        "pair_summary": pair_path,
        "summary_text": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PBX multiclass one-vs-all classification and embryo-level persistent misclassification analysis."
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", *FEATURE_SETS.keys()],
        default="embedding",
        help="Feature set to run. 'all' runs curvature, length, and embedding separately.",
    )
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--classification-permutations", type=int, default=100)
    parser.add_argument("--misclassification-permutations", type=int, default=500)
    parser.add_argument("--misclassification-simulations", type=int, default=2000)
    parser.add_argument("--min-samples-per-group", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(project_root / "src"))

    from analyze.classification import run_classification
    from analyze.classification.misclassification.pipeline import run_misclassification_pipeline

    df = _load_dataframe(project_root)

    feature_names = list(FEATURE_SETS) if args.feature_set == "all" else [args.feature_set]
    output_root = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "consistent_misclassification"
        / f"bin_width_{args.bin_width:.1f}hpf"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    run_index: list[dict[str, object]] = []
    flagged_tables: list[pd.DataFrame] = []

    for feature_set in feature_names:
        feature_dir = output_root / feature_set
        stage1_dir = feature_dir / "stage1_multiclass"
        stage2_dir = feature_dir / "misclassification"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        stage2_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {feature_set}: Stage 1 multiclass one-vs-all ===")
        analysis = run_classification(
            df,
            class_col="genotype",
            id_col="embryo_id",
            time_col="predicted_stage_hpf",
            comparisons="all_vs_rest",
            features={feature_set: FEATURE_SETS[feature_set]},
            n_jobs=-1,
            n_permutations=int(args.classification_permutations),
            bin_width=float(args.bin_width),
            min_samples_per_group=int(args.min_samples_per_group),
            random_state=int(args.random_state),
            verbose=True,
            save_multiclass_predictions=True,
        )

        pred_df = _write_stage1_artifacts(
            analysis=analysis,
            stage1_dir=stage1_dir,
            feature_set=feature_set,
            bin_width=float(args.bin_width),
            classification_permutations=int(args.classification_permutations),
            random_state=int(args.random_state),
        )

        print(f"=== {feature_set}: Stage 2 persistent misclassification ===")
        stage2_result = run_misclassification_pipeline(
            input_dir=stage1_dir,
            output_dir=stage2_dir,
            config={
                "n_permutations": int(args.misclassification_permutations),
                "n_sim": int(args.misclassification_simulations),
                "random_state": int(args.random_state),
                "require_n_windows_min": 3,
                "require_n_wrong_min": 3,
                "q_val_threshold": 0.05,
                "wrong_rate_z_threshold": 2.0,
                "wrong_rate_delta_threshold": 0.20,
                "top_confused_frac_threshold": 0.70,
                "rolling_window_bins": 3,
                "rolling_threshold": 0.60,
            },
        )

        summary_outputs = _build_summary_tables(stage2_dir=stage2_dir, feature_set=feature_set)
        flagged = stage2_result["flagged_embryos"]
        if not flagged.empty:
            flagged_tables.append(flagged.assign(feature_set=feature_set).copy())

        run_index.append(
            {
                "feature_set": feature_set,
                "n_prediction_rows": int(len(pred_df)),
                "n_scores_rows": int(len(analysis.scores)),
                "n_flagged_embryos": int(len(flagged)),
                "stage1_dir": str(stage1_dir),
                "stage2_dir": str(stage2_dir),
                "summary_text": str(summary_outputs["summary_text"]),
            }
        )

        print(
            f"{feature_set}: "
            f"{len(flagged)} flagged embryos across {pred_df['true_class'].nunique()} genotypes. "
            f"Summary: {summary_outputs['summary_text']}"
        )

    run_index_df = pd.DataFrame(run_index)
    run_index_path = output_root / "run_index.csv"
    run_index_df.to_csv(run_index_path, index=False)
    cross_feature_outputs = _write_cross_feature_summaries(
        output_root=output_root,
        flagged_tables=flagged_tables,
    )
    print(f"\nSaved run index: {run_index_path}")
    if cross_feature_outputs:
        print(f"Cross-feature summary: {cross_feature_outputs['summary_text']}")


if __name__ == "__main__":
    main()
