#!/usr/bin/env python
"""Load CEP290 and B9D2 validation datasets used for subtle phenotype methods.

This script resolves labeled datasets from either the current repository or
`morphseq_CORRUPT_OLD` fallback locations, loads them, standardizes
`phenotype_label`, and emits quick summary tables.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


DATA_CANDIDATES = {
    "cep290": [
        PROJECT_ROOT
        / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv",
        Path(
            "/net/trapnell/vol1/home/mdcolon/proj/morphseq_CORRUPT_OLD/"
            "results/mcolon/20251229_cep290_phenotype_extraction/final_data/"
            "embryo_data_with_labels.csv"
        ),
    ],
    "b9d2": [
        PROJECT_ROOT
        / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv",
        Path(
            "/net/trapnell/vol1/home/mdcolon/proj/morphseq_CORRUPT_OLD/"
            "results/mcolon/20251219_b9d2_phenotype_extraction/data/"
            "b9d2_labeled_data.csv"
        ),
    ],
}

_MISSING_TOKENS = {"", "na", "nan", "none", "null"}
_UNLABELED_TOKENS = _MISSING_TOKENS | {"unlabeled", "unknown"}

_WILDTYPE_PATTERNS = (
    re.compile(r"wild[\s_-]*type"),
    re.compile(r"(^|[_\s/])wt($|[_\s/])"),
    re.compile(r"^ab$"),
    re.compile(r"^a/b$"),
)


def _resolve_first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n  - ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the candidate paths exist:\n  - {joined}")


def resolve_validation_paths() -> Dict[str, Path]:
    """Return resolved CSV paths for CEP290 and B9D2."""
    return {name: _resolve_first_existing(candidates) for name, candidates in DATA_CANDIDATES.items()}


def _normalize_token(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_")


def _is_unknown_like(token: str) -> bool:
    if token in _UNLABELED_TOKENS:
        return True
    if token.endswith("_unknown"):
        return True
    return False


def _is_wildtype_control(genotype: str, phenotype_label: str) -> bool:
    g = _normalize_token(genotype)
    p = _normalize_token(phenotype_label)

    for candidate in (g, p):
        if candidate in {"ab", "a/b"}:
            return True
        if any(pat.search(candidate) for pat in _WILDTYPE_PATTERNS):
            return True
    return False


def _normalize_dataframe(
    df: pd.DataFrame,
    dataset: str,
    *,
    drop_unknown_genotype: bool,
    drop_unlabeled_phenotype: bool,
) -> pd.DataFrame:
    required = {"embryo_id", "predicted_stage_hpf"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{dataset}: missing required columns: {missing}")

    out = df.copy()

    if "experiment_id" not in out.columns:
        out["experiment_id"] = out["embryo_id"].astype(str).str.split("_", n=1).str[0]

    if "phenotype_label" not in out.columns:
        if "cluster_categories" in out.columns:
            out["phenotype_label"] = out["cluster_categories"]
        elif "clusters" in out.columns:
            out["phenotype_label"] = out["clusters"].astype(str)
        else:
            out["phenotype_label"] = "unlabeled"

    # Canonical control detection:
    # treat wild-type / A-B style labels as wild-type controls.
    out["is_wildtype_control"] = [
        _is_wildtype_control(g, p)
        for g, p in zip(
            out.get("genotype", pd.Series([""] * len(out))),
            out.get("phenotype_label", pd.Series([""] * len(out))),
        )
    ]

    unlabeled_mask = out["phenotype_label"].map(_normalize_token).isin(_UNLABELED_TOKENS)
    out.loc[out["is_wildtype_control"] & unlabeled_mask, "phenotype_label"] = "wildtype"

    if drop_unknown_genotype and "genotype" in out.columns:
        unknown_genotype_mask = out["genotype"].map(_normalize_token).map(_is_unknown_like)
        out = out.loc[~unknown_genotype_mask].copy()

    if drop_unlabeled_phenotype:
        unlabeled_pheno_mask = out["phenotype_label"].map(_normalize_token).isin(_UNLABELED_TOKENS)
        out = out.loc[~unlabeled_pheno_mask].copy()

    out["dataset"] = dataset
    return out


def load_validation_data(
    *,
    drop_unknown_genotype: bool = True,
    drop_unlabeled_phenotype: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Path]]:
    """Load and normalize CEP290 and B9D2 dataframes."""
    paths = resolve_validation_paths()
    frames: Dict[str, pd.DataFrame] = {}

    for dataset, csv_path in paths.items():
        df = pd.read_csv(csv_path, low_memory=False)
        frames[dataset] = _normalize_dataframe(
            df,
            dataset,
            drop_unknown_genotype=drop_unknown_genotype,
            drop_unlabeled_phenotype=drop_unlabeled_phenotype,
        )

    return frames, paths


def build_summary_tables(
    frames: Dict[str, pd.DataFrame],
    paths: Dict[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build dataset overview + embryo-level genotype/phenotype counts."""
    overview_rows = []
    genotype_rows = []
    phenotype_rows = []

    for dataset, df in frames.items():
        overview_rows.append(
            {
                "dataset": dataset,
                "source_path": str(paths[dataset]),
                "n_rows": int(len(df)),
                "n_embryos": int(df["embryo_id"].nunique()),
                "n_experiments": int(df["experiment_id"].nunique()),
                "hpf_min": float(df["predicted_stage_hpf"].min()),
                "hpf_max": float(df["predicted_stage_hpf"].max()),
            }
        )

        embryo_meta = (
            df[["embryo_id", "genotype", "phenotype_label"]]
            .drop_duplicates(subset=["embryo_id"])  # each embryo counted once
            .copy()
        )

        geno_counts = (
            embryo_meta.groupby("genotype", dropna=False)["embryo_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        for genotype, count in geno_counts.items():
            genotype_rows.append(
                {
                    "dataset": dataset,
                    "genotype": "NA" if pd.isna(genotype) else str(genotype),
                    "n_embryos": int(count),
                }
            )

        pheno_counts = (
            embryo_meta.groupby("phenotype_label", dropna=False)["embryo_id"]
            .nunique()
            .sort_values(ascending=False)
        )
        for phenotype_label, count in pheno_counts.items():
            phenotype_rows.append(
                {
                    "dataset": dataset,
                    "phenotype_label": "NA" if pd.isna(phenotype_label) else str(phenotype_label),
                    "n_embryos": int(count),
                }
            )

    overview_df = pd.DataFrame(overview_rows).sort_values("dataset").reset_index(drop=True)
    genotype_df = pd.DataFrame(genotype_rows).sort_values(["dataset", "n_embryos"], ascending=[True, False]).reset_index(drop=True)
    phenotype_df = pd.DataFrame(phenotype_rows).sort_values(["dataset", "n_embryos"], ascending=[True, False]).reset_index(drop=True)

    return overview_df, genotype_df, phenotype_df


def write_summary_artifacts(
    output_dir: Path,
    overview_df: pd.DataFrame,
    genotype_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    paths: Dict[str, Path],
) -> None:
    """Write small summary artifacts for downstream method setup."""
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_df.to_csv(output_dir / "dataset_overview.tsv", sep="\t", index=False)
    genotype_df.to_csv(output_dir / "genotype_embryo_counts.tsv", sep="\t", index=False)
    phenotype_df.to_csv(output_dir / "phenotype_embryo_counts.tsv", sep="\t", index=False)

    payload = {name: str(path) for name, path in paths.items()}
    (output_dir / "resolved_paths.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _print_summary(overview_df: pd.DataFrame, genotype_df: pd.DataFrame, phenotype_df: pd.DataFrame) -> None:
    print("\nDataset overview")
    print(overview_df.to_string(index=False))

    for dataset in overview_df["dataset"]:
        print(f"\n{dataset} genotype embryo counts")
        view = genotype_df[genotype_df["dataset"] == dataset][["genotype", "n_embryos"]]
        print(view.to_string(index=False))

        print(f"\n{dataset} phenotype embryo counts")
        view = phenotype_df[phenotype_df["dataset"] == dataset][["phenotype_label", "n_embryos"]]
        print(view.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for TSV/JSON summary artifacts.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print summaries only; do not write output files.",
    )
    parser.add_argument(
        "--keep-unknown-genotype",
        action="store_true",
        help="Do not filter unknown/unset genotype rows.",
    )
    parser.add_argument(
        "--keep-unlabeled-phenotype",
        action="store_true",
        help="Do not filter unlabeled/NA phenotype_label rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames, paths = load_validation_data(
        drop_unknown_genotype=not args.keep_unknown_genotype,
        drop_unlabeled_phenotype=not args.keep_unlabeled_phenotype,
    )
    overview_df, genotype_df, phenotype_df = build_summary_tables(frames, paths)

    _print_summary(overview_df, genotype_df, phenotype_df)

    if not args.no_save:
        write_summary_artifacts(args.output_dir, overview_df, genotype_df, phenotype_df, paths)
        print(f"\nWrote summary artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
