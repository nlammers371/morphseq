#!/usr/bin/env python
"""Phase-1 smoke test for subtle phenotype methods using shared resampling utility.

Computes embryo-level AUC on baseline deviation and runs:
- bootstrap CI on group mean difference
- permutation p-value on the same statistic

This demonstrates the intended resampling backbone for downstream Parts 3-6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze.utils import resampling as resample  # noqa: E402
from load_validation_data import load_validation_data  # noqa: E402


DEFAULT_METRIC = "baseline_deviation_normalized"


def _difference_of_means(values: np.ndarray, labels01: np.ndarray) -> float:
    pos = values[labels01 == 1]
    neg = values[labels01 == 0]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Both classes must be present to compute mean difference.")
    return float(np.mean(pos) - np.mean(neg))


def _compute_embryo_auc(
    df: pd.DataFrame,
    *,
    group_col: str,
    positive_group: str,
    negative_group: str,
    metric_col: str,
) -> pd.DataFrame:
    rows = []

    for embryo_id, gdf in df.groupby("embryo_id", sort=False):
        group_value = gdf[group_col].iloc[0]
        if group_value not in {positive_group, negative_group}:
            continue

        work = gdf[["predicted_stage_hpf", metric_col]].dropna().sort_values("predicted_stage_hpf")
        if len(work) < 2:
            continue

        auc = float(np.trapz(work[metric_col].to_numpy(), work["predicted_stage_hpf"].to_numpy()))
        rows.append(
            {
                "embryo_id": embryo_id,
                "group": group_value,
                "auc": auc,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No embryos passed AUC preprocessing.")

    return out


def _run_resampling_test(
    embryo_auc: pd.DataFrame,
    *,
    positive_group: str,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> Dict[str, float]:
    labels01 = (embryo_auc["group"].to_numpy() == positive_group).astype(int)
    values = embryo_auc["auc"].to_numpy(dtype=float)

    observed = _difference_of_means(values, labels01)

    boot_spec = resample.bootstrap(size=len(values))

    def _bootstrap_stat(data: dict, _rng) -> float:
        idx = data.get("indices")
        if idx is None:
            return _difference_of_means(data["values"], data["labels"])
        idx = np.asarray(idx, dtype=int)
        return _difference_of_means(data["values"][idx], data["labels"][idx])

    boot_out = resample.run(
        data={"n": len(values), "values": values, "labels": labels01},
        spec=boot_spec,
        statistic=resample.statistic("mean_auc_diff", _bootstrap_stat),
        n_iters=n_bootstrap,
        seed=seed,
        n_jobs=1,
        store="all",
        max_retries_per_iter=2,
    )
    boot_summary = resample.aggregate(boot_out, alpha=0.05)

    perm_spec = resample.permute_labels()

    def _perm_stat(data: dict, _rng) -> float:
        return _difference_of_means(data["values"], np.asarray(data["labels"], dtype=int))

    perm_out = resample.run(
        data={"values": values, "labels": labels01.copy()},
        spec=perm_spec,
        statistic=resample.statistic("mean_auc_diff", _perm_stat),
        n_iters=n_permutations,
        seed=seed + 1,
        n_jobs=1,
        store="all",
        alternative="two-sided",
    )
    perm_summary = resample.aggregate(perm_out)

    return {
        "n_embryos": int(len(values)),
        "n_positive": int((labels01 == 1).sum()),
        "n_negative": int((labels01 == 0).sum()),
        "observed_diff": float(observed),
        "bootstrap_mean": float(boot_summary.mean),
        "bootstrap_ci_low": float(boot_summary.ci_low),
        "bootstrap_ci_high": float(boot_summary.ci_high),
        "permutation_pvalue": float(perm_summary.pvalue),
        "null_mean": float(perm_summary.null_mean),
        "null_std": float(perm_summary.null_std),
        "n_bootstrap": int(n_bootstrap),
        "n_permutations": int(n_permutations),
    }


def run_smoke(
    *,
    metric_col: str,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    frames, _paths = load_validation_data()

    configs = [
        {
            "dataset": "cep290",
            "group_col": "genotype",
            "positive_group": "cep290_homozygous",
            "negative_group": "cep290_wildtype",
            "comparison": "homozygous_vs_wildtype",
        },
        {
            "dataset": "b9d2",
            "group_col": "phenotype_label",
            "positive_group": "CE",
            "negative_group": "wildtype",
            "comparison": "CE_vs_wildtype",
        },
    ]

    rows = []
    for cfg in configs:
        df = frames[cfg["dataset"]]
        embryo_auc = _compute_embryo_auc(
            df,
            group_col=cfg["group_col"],
            positive_group=cfg["positive_group"],
            negative_group=cfg["negative_group"],
            metric_col=metric_col,
        )

        stats = _run_resampling_test(
            embryo_auc,
            positive_group=cfg["positive_group"],
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            seed=seed,
        )

        rows.append(
            {
                "dataset": cfg["dataset"],
                "comparison": cfg["comparison"],
                "group_col": cfg["group_col"],
                "positive_group": cfg["positive_group"],
                "negative_group": cfg["negative_group"],
                **stats,
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metric column for embryo-level AUC.")
    parser.add_argument("--n-bootstrap", type=int, default=400, help="Bootstrap iterations.")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Permutation iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for resampling.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for result TSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results_df = run_smoke(
        metric_col=args.metric,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "phase1_resampling_smoke.tsv"
    results_df.to_csv(out_path, sep="\t", index=False)

    print("\nPhase-1 resampling smoke results")
    print(results_df.to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
