#!/usr/bin/env python
"""Cluster enrichment analysis for embryo-first persistence outputs.

For a persistence run directory, this script:
1. Builds embryo-level metadata with cluster assignments
2. Tests enrichment of cluster x genotype and cluster x phenotype labels
3. Applies FDR correction
4. Generates proportion plots (including inverse cluster-colored views)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parents[3]

import sys

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analyze.viz.plotting import plot_proportions  # noqa: E402


DEFAULT_PERSISTENCE_ROOT = Path(__file__).resolve().parent / "output" / "embryo_first_persistence"


def _resolve_latest_run(persistence_root: Path) -> Path:
    if not persistence_root.exists():
        raise FileNotFoundError(f"Persistence root does not exist: {persistence_root}")
    run_dirs = sorted([p for p in persistence_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {persistence_root}")
    return run_dirs[-1]


def _fdr_bh(pvals: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan)
    finite = np.isfinite(pvals)
    if not finite.any():
        return out

    p = pvals[finite]
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]

    adjusted = np.empty_like(ranked)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adjusted[i] = prev

    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[finite] = restored
    return out


def _load_run_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments_path = run_dir / "cohort_assignments.tsv"
    binned_path = run_dir / "binned_data.tsv"

    if not assignments_path.exists() or not binned_path.exists():
        raise FileNotFoundError(
            f"Missing required files in {run_dir}. Expected cohort_assignments.tsv and binned_data.tsv"
        )

    assignments = pd.read_csv(assignments_path, sep="\t")
    binned = pd.read_csv(binned_path, sep="\t")
    return assignments, binned


def _prepare_embryo_table(assignments: pd.DataFrame, binned: pd.DataFrame) -> pd.DataFrame:
    meta_cols = [
        c
        for c in [
            "embryo_id",
            "dataset_id",
            "experiment_id",
            "genotype",
            "genotype_group",
            "phenotype",
            "phenotype_group",
            "cluster",
            "cluster_topq",
        ]
        if c in assignments.columns
    ]

    embryo_df = assignments[meta_cols].drop_duplicates(subset=["embryo_id"]).copy()

    if "cluster" not in embryo_df.columns and "cluster" in binned.columns:
        cluster_map = binned[["embryo_id", "cluster"]].drop_duplicates(subset=["embryo_id"])
        embryo_df = embryo_df.merge(cluster_map, on="embryo_id", how="left")

    # Fall back to raw columns when *_group columns are absent.
    if "genotype_group" not in embryo_df.columns and "genotype" in embryo_df.columns:
        embryo_df["genotype_group"] = embryo_df["genotype"]
    if "phenotype_group" not in embryo_df.columns and "phenotype" in embryo_df.columns:
        embryo_df["phenotype_group"] = embryo_df["phenotype"]

    return embryo_df


def _run_enrichment(embryo_df: pd.DataFrame, group_col: str, cluster_col: str = "cluster") -> tuple[pd.DataFrame, pd.DataFrame]:
    work = embryo_df[["embryo_id", cluster_col, group_col]].dropna().copy()

    if work[cluster_col].nunique() < 2 or work[group_col].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    contingency = pd.crosstab(work[cluster_col], work[group_col], dropna=False)
    chi2, p_global, dof, _ = chi2_contingency(contingency)
    global_df = pd.DataFrame(
        [
            {
                "group_col": group_col,
                "chi2": float(chi2),
                "p_value": float(p_global),
                "dof": int(dof),
                "n_clusters": int(contingency.shape[0]),
                "n_group_levels": int(contingency.shape[1]),
                "n_embryos": int(len(work)),
            }
        ]
    )

    rows: List[Dict[str, object]] = []
    total_n = len(work)

    for cluster in sorted(work[cluster_col].unique()):
        cluster_mask = work[cluster_col] == cluster
        n_cluster = int(cluster_mask.sum())

        for level in sorted(work[group_col].unique()):
            level_mask = work[group_col] == level

            a = int((cluster_mask & level_mask).sum())
            b = int((cluster_mask & ~level_mask).sum())
            c = int((~cluster_mask & level_mask).sum())
            d = int((~cluster_mask & ~level_mask).sum())

            table = np.array([[a, b], [c, d]], dtype=int)
            odds_ratio, p_val = fisher_exact(table, alternative="two-sided")

            in_cluster_rate = a / n_cluster if n_cluster > 0 else math.nan
            out_cluster_n = total_n - n_cluster
            out_cluster_rate = c / out_cluster_n if out_cluster_n > 0 else math.nan

            rows.append(
                {
                    "group_col": group_col,
                    "cluster": cluster,
                    "level": level,
                    "a_in_cluster_and_level": a,
                    "b_in_cluster_not_level": b,
                    "c_out_cluster_and_level": c,
                    "d_out_cluster_not_level": d,
                    "n_cluster": n_cluster,
                    "n_total": total_n,
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_val),
                    "in_cluster_rate": float(in_cluster_rate),
                    "out_cluster_rate": float(out_cluster_rate),
                    "rate_diff": float(in_cluster_rate - out_cluster_rate),
                }
            )

    enrichment_df = pd.DataFrame(rows)
    if enrichment_df.empty:
        return enrichment_df, global_df

    enrichment_df["p_fdr_bh"] = _fdr_bh(enrichment_df["p_value"].to_numpy(dtype=float))
    enrichment_df = enrichment_df.sort_values(["p_fdr_bh", "p_value", "odds_ratio"], ascending=[True, True, False]).reset_index(drop=True)
    return enrichment_df, global_df


def _make_proportion_plots(embryo_df: pd.DataFrame, out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    required = {"embryo_id", "cluster"}
    if not required.issubset(set(embryo_df.columns)):
        return paths

    def _has(cols: Iterable[str]) -> bool:
        return all(c in embryo_df.columns for c in cols)

    # Required plan outputs
    if _has(["genotype_group", "cluster", "phenotype_group"]):
        path = out_dir / "cluster_proportions_by_genotype.png"
        plot_proportions(
            embryo_df,
            color_by_grouping="phenotype_group",
            row_by="genotype_group",
            col_by="cluster",
            count_by="embryo_id",
            normalize=True,
            bar_mode="grouped",
            output_path=path,
            title="Cluster proportions by genotype",
            show_counts=True,
        )
        paths.append(str(path))

    if _has(["phenotype_group", "cluster", "genotype_group"]):
        path = out_dir / "cluster_proportions_by_phenotype.png"
        plot_proportions(
            embryo_df,
            color_by_grouping="genotype_group",
            row_by="phenotype_group",
            col_by="cluster",
            count_by="embryo_id",
            normalize=True,
            bar_mode="grouped",
            output_path=path,
            title="Cluster proportions by phenotype",
            show_counts=True,
        )
        paths.append(str(path))

    # Inverse cluster-color composition views
    if _has(["genotype_group", "phenotype_group", "cluster"]):
        path = out_dir / "cluster_mix_by_genotype_phenotype.png"
        plot_proportions(
            embryo_df,
            color_by_grouping="cluster",
            row_by="genotype_group",
            col_by="phenotype_group",
            count_by="embryo_id",
            normalize=True,
            bar_mode="grouped",
            output_path=path,
            title="Cluster mix by genotype x phenotype",
            show_counts=True,
        )
        paths.append(str(path))

    if _has(["experiment_id", "cluster", "genotype_group"]):
        path = out_dir / "cluster_mix_by_experiment.png"
        plot_proportions(
            embryo_df,
            color_by_grouping="genotype_group",
            row_by="experiment_id",
            col_by="cluster",
            count_by="embryo_id",
            normalize=True,
            bar_mode="grouped",
            output_path=path,
            title="Genotype mix by experiment x cluster",
            show_counts=True,
        )
        paths.append(str(path))

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="One persistence run directory")
    parser.add_argument("--persistence-root", type=Path, default=DEFAULT_PERSISTENCE_ROOT)
    parser.add_argument("--cluster-col", type=str, default="cluster")
    parser.add_argument("--genotype-col", type=str, default="genotype_group")
    parser.add_argument("--phenotype-col", type=str, default="phenotype_group")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir else _resolve_latest_run(args.persistence_root)
    run_dir = run_dir.resolve()

    assignments, binned = _load_run_tables(run_dir)
    embryo_df = _prepare_embryo_table(assignments, binned)

    if args.cluster_col != "cluster" and args.cluster_col in embryo_df.columns:
        embryo_df = embryo_df.rename(columns={args.cluster_col: "cluster"})

    enrich_dir = run_dir / "enrichment"
    enrich_dir.mkdir(parents=True, exist_ok=True)

    all_global: List[pd.DataFrame] = []

    geno_col = args.genotype_col if args.genotype_col in embryo_df.columns else "genotype"
    pheno_col = args.phenotype_col if args.phenotype_col in embryo_df.columns else "phenotype"

    geno_df, geno_global = _run_enrichment(embryo_df, group_col=geno_col, cluster_col="cluster")
    pheno_df, pheno_global = _run_enrichment(embryo_df, group_col=pheno_col, cluster_col="cluster")

    if not geno_df.empty:
        geno_df.to_csv(enrich_dir / "enrichment_genotype.tsv", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(enrich_dir / "enrichment_genotype.tsv", sep="\t", index=False)

    if not pheno_df.empty:
        pheno_df.to_csv(enrich_dir / "enrichment_phenotype.tsv", sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(enrich_dir / "enrichment_phenotype.tsv", sep="\t", index=False)

    if not geno_global.empty:
        all_global.append(geno_global)
    if not pheno_global.empty:
        all_global.append(pheno_global)

    global_df = pd.concat(all_global, ignore_index=True) if all_global else pd.DataFrame()
    global_df.to_csv(enrich_dir / "enrichment_global.tsv", sep="\t", index=False)

    plot_paths: List[str] = []
    if not args.skip_plots:
        plot_paths = _make_proportion_plots(embryo_df, out_dir=enrich_dir)

    summary = pd.DataFrame(
        [
            {
                "run_dir": str(run_dir),
                "n_embryos": int(embryo_df["embryo_id"].nunique()),
                "n_clusters": int(embryo_df["cluster"].nunique()),
                "genotype_tests": int(len(geno_df)),
                "genotype_fdr_lt_0_05": int((geno_df.get("p_fdr_bh", pd.Series(dtype=float)) < 0.05).sum()) if len(geno_df) else 0,
                "phenotype_tests": int(len(pheno_df)),
                "phenotype_fdr_lt_0_05": int((pheno_df.get("p_fdr_bh", pd.Series(dtype=float)) < 0.05).sum()) if len(pheno_df) else 0,
                "global_rows": int(len(global_df)),
                "plots_written": int(len(plot_paths)),
            }
        ]
    )
    summary_path = enrich_dir / "enrichment_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    print(f"Run directory: {run_dir}")
    print(f"Wrote: {enrich_dir / 'enrichment_genotype.tsv'}")
    print(f"Wrote: {enrich_dir / 'enrichment_phenotype.tsv'}")
    print(f"Wrote: {enrich_dir / 'enrichment_global.tsv'}")
    print(f"Wrote: {summary_path}")
    print(f"Plots: {len(plot_paths)}")


if __name__ == "__main__":
    main()
