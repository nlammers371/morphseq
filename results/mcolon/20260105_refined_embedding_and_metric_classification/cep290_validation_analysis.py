"""
CEP290 pre-20 hpf validation / discrepancy audit.

Purpose
-------
Investigate why the refined implementation reports early significant AUROC
(e.g. 12 hpf bin) when the raw trajectories may look indistinguishable.

This script focuses on the most common root causes:
1) Bin-width effects (e.g. 4h bins mix 12–16 hpf, which can change results)
2) Within-bin time-distribution confounds (Penetrant embryos skew older/younger
   within the same coarse bin, inducing apparent signal)
3) Data-source differences (working .pkl vs refined .csv embryo inclusion)
4) Permutation resolution (with 100 permutations, min p-value is 1/(100+1)=0.0099)

Outputs
-------
Writes all outputs under:
`results/mcolon/20260105_refined_embedding_and_metric_classification/output/cep290/validation/`

Typical outputs:
- auroc_comparison.csv
- within_bin_time_confounds.csv
- embryo_id_set_differences.txt

Notes
-----
This script assumes your usual analysis environment (pandas, numpy, sklearn)
is available (e.g. in conda). It is not intended to run in a minimal system Python.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--refined-csv",
        type=Path,
        default=PROJECT_ROOT
        / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv",
        help="Refined data source (CSV with cluster_categories and z_mu_b_* columns).",
    )
    parser.add_argument(
        "--working-pkl",
        type=Path,
        default=PROJECT_ROOT
        / "results/mcolon/20251229_cep290_phenotype_extraction/data/clustering_data__early_homo.pkl",
        help="Working data source (pkl used in earlier CEP290 analyses).",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="predicted_stage_hpf",
        help="Time column name.",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=20.0,
        help="Upper time bound (hpf) for the 'pre-20' comparison window.",
    )
    parser.add_argument(
        "--bin-widths",
        type=float,
        nargs="+",
        default=[2.0, 4.0],
        help="Bin widths (hours) to compare.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="Number of label permutations for compare_groups().",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for compare_groups().",
    )
    parser.add_argument(
        "--min-samples-per-bin",
        type=int,
        default=5,
        help="Minimum samples per class per time bin for compare_groups().",
    )
    parser.add_argument(
        "--time-bin-to-audit",
        type=float,
        default=12.0,
        help="Time-bin label to audit for within-bin confounds (e.g. 12 for the 12–16 hpf bin in 4h binning).",
    )

    return parser.parse_args()


@dataclass(frozen=True)
class Cep290Groups:
    penetrant_ids: list[str]
    control_ids: list[str]


def _load_refined_groups(df) -> Cep290Groups:
    penetrant_categories = ["Low_to_High", "High_to_Low", "Intermediate"]
    penetrant_ids = (
        df[df["cluster_categories"].isin(penetrant_categories)]["embryo_id"].dropna().unique().tolist()
    )
    control_ids = (
        df[df["cluster_categories"] == "Not Penetrant"]["embryo_id"].dropna().unique().tolist()
    )
    return Cep290Groups(penetrant_ids=penetrant_ids, control_ids=control_ids)


def _extract_working_df(working_obj):
    # Heuristic: grab the first DataFrame-like object with embryo_id + predicted_stage_hpf
    for _, v in working_obj.items():
        if hasattr(v, "columns") and "embryo_id" in v.columns and "predicted_stage_hpf" in v.columns:
            return v
    raise ValueError("Could not find a dataframe with embryo_id + predicted_stage_hpf in working .pkl")


def _write_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _audit_within_bin_time_confounds(
    df,
    *,
    time_col: str,
    group_col: str,
    positive_label: str,
    negative_label: str,
    bin_width: float,
    time_bin: float,
):
    """
    For a given (bin_width, time_bin), quantify whether groups differ in
    within-bin time distribution (mean time per embryo, number of rows per embryo).
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    df = df.copy()
    df["time_bin"] = (np.floor(df[time_col] / bin_width) * bin_width).astype(float)
    df = df[df["time_bin"] == float(time_bin)].copy()

    df = df[df[group_col].isin([positive_label, negative_label])].copy()

    per_embryo = (
        df.groupby(["embryo_id", group_col], as_index=False)
        .agg(
            mean_hpf=(time_col, "mean"),
            median_hpf=(time_col, "median"),
            n_rows=(time_col, "size"),
        )
    )
    per_embryo["y"] = (per_embryo[group_col] == positive_label).astype(int)

    # If mean_hpf predicts label, morphology correlated with time can become a confound.
    auc_mean_hpf = float(roc_auc_score(per_embryo["y"].values, per_embryo["mean_hpf"].values))
    auc_n_rows = float(roc_auc_score(per_embryo["y"].values, per_embryo["n_rows"].values))

    summary = (
        per_embryo.groupby(group_col)[["mean_hpf", "median_hpf", "n_rows"]]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )

    return per_embryo, summary, auc_mean_hpf, auc_n_rows


def main() -> None:
    args = _parse_args()

    out_dir = Path(__file__).parent / "output" / "cep290" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Imports local to main to keep module import lightweight in environments without pandas/sklearn
    import pickle
    import pandas as pd

    from analyze.difference_detection.comparison import compare_groups
    from utils.data_prep import prepare_comparison_data

    print(f"Loading refined CSV: {args.refined_csv}")
    df_refined = pd.read_csv(args.refined_csv, low_memory=False)
    print(f"  rows={len(df_refined)} embryos={df_refined['embryo_id'].nunique()}")

    groups = _load_refined_groups(df_refined)
    print(f"  Penetrant embryos={len(groups.penetrant_ids)} Control embryos={len(groups.control_ids)}")

    df_prep = prepare_comparison_data(
        df_refined,
        group1_ids=groups.penetrant_ids,
        group2_ids=groups.control_ids,
        group1_label="Penetrant",
        group2_label="Control",
    )

    # ---------------------------------------------------------------------
    # Compare embryo ID sets between "working" and "refined" sources pre-20
    # ---------------------------------------------------------------------
    print(f"Loading working PKL: {args.working_pkl}")
    with args.working_pkl.open("rb") as f:
        working_obj = pickle.load(f)
    df_working = _extract_working_df(working_obj)

    refined_ids_pre = set(df_refined[df_refined[args.time_col] < args.time_max]["embryo_id"].dropna().unique())
    working_ids_pre = set(df_working[df_working[args.time_col] < args.time_max]["embryo_id"].dropna().unique())

    only_refined = sorted(refined_ids_pre - working_ids_pre)
    only_working = sorted(working_ids_pre - refined_ids_pre)

    _write_text(
        out_dir / "embryo_id_set_differences.txt",
        [
            f"pre<{args.time_max}hpf embryo_id set comparison",
            f"refined_n={len(refined_ids_pre)} working_n={len(working_ids_pre)}",
            "",
            f"only_in_refined (n={len(only_refined)}):",
            *only_refined[:200],
            *(["... (truncated)"] if len(only_refined) > 200 else []),
            "",
            f"only_in_working (n={len(only_working)}):",
            *only_working[:200],
            *(["... (truncated)"] if len(only_working) > 200 else []),
        ],
    )

    # ---------------------------------------------------------------------
    # Bin-width sensitivity: run compare_groups with multiple bin widths
    # ---------------------------------------------------------------------
    records = []
    for bin_width in args.bin_widths:
        for feature_set, features in [
            ("metric_curvature", ["baseline_deviation_normalized"]),
            ("embedding", "z_mu_b"),
        ]:
            print(f"Running compare_groups: bin_width={bin_width} features={feature_set}")
            res = compare_groups(
                df_prep,
                group_col="group",
                group1="Penetrant",  # positive / phenotype
                group2="Control",  # negative / reference
                features=features,
                morphology_metric=None,
                bin_width=bin_width,
                n_permutations=args.n_permutations,
                n_jobs=args.n_jobs,
                min_samples_per_bin=args.min_samples_per_bin,
                random_state=42,
                verbose=True,
            )
            df_class = res["classification"].copy()
            df_class["bin_width"] = float(bin_width)
            df_class["feature_set"] = feature_set
            records.append(df_class)

    df_auc_all = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    df_auc_all = df_auc_all[df_auc_all["time_bin"] < args.time_max].copy()
    df_auc_all.to_csv(out_dir / "auroc_comparison.csv", index=False)

    # ---------------------------------------------------------------------
    # Within-bin time distribution confounds
    # ---------------------------------------------------------------------
    confound_rows = []
    for bin_width in args.bin_widths:
        time_bin = float(args.time_bin_to_audit)
        print(f"Auditing within-bin confounds: bin_width={bin_width} time_bin={time_bin}")
        per_embryo, summary, auc_mean_hpf, auc_n_rows = _audit_within_bin_time_confounds(
            df_prep,
            time_col=args.time_col,
            group_col="group",
            positive_label="Penetrant",
            negative_label="Control",
            bin_width=bin_width,
            time_bin=time_bin,
        )

        per_embryo.to_csv(out_dir / f"within_bin_per_embryo__bin{bin_width:g}_t{time_bin:g}.csv", index=False)
        summary.to_csv(out_dir / f"within_bin_summary__bin{bin_width:g}_t{time_bin:g}.csv", index=False)

        confound_rows.append(
            {
                "bin_width": float(bin_width),
                "time_bin": float(time_bin),
                "auc_mean_hpf_predicts_label": float(auc_mean_hpf),
                "auc_n_rows_predicts_label": float(auc_n_rows),
                "n_embryos_positive": int((per_embryo["y"] == 1).sum()),
                "n_embryos_negative": int((per_embryo["y"] == 0).sum()),
            }
        )

    pd.DataFrame(confound_rows).to_csv(out_dir / "within_bin_time_confounds.csv", index=False)

    # ---------------------------------------------------------------------
    # Reminder about permutation resolution
    # ---------------------------------------------------------------------
    min_p = 1.0 / (args.n_permutations + 1) if args.n_permutations > 0 else float("nan")
    _write_text(
        out_dir / "notes.txt",
        [
            "Interpretation notes",
            "--------------------",
            f"n_permutations={args.n_permutations} => min possible p-value is 1/(n_perm+1) = {min_p:.6f}",
            "",
            "If you see p=0.009900..., that just means 'more extreme than all 100 permutations'.",
            "Increase n_permutations (e.g. 1000+) to better resolve borderline effects.",
            "",
            "If `auc_mean_hpf_predicts_label` is far from 0.5 within a bin,",
            "you likely have within-bin time-distribution imbalance, which can induce apparent signal.",
        ],
    )

    print(f"\nWrote validation outputs to: {out_dir}")


if __name__ == "__main__":
    main()

