"""
Data loading utilities for morphseq experiments.

This module handles loading and combining experimental data from multiple
build06 output files.
"""

from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
import numpy as np


def _hydrate_curvature_from_body_axis(
    df: pd.DataFrame,
    experiment_id: str,
    build_dir: Union[str, Path],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    If curvature columns are missing (or entirely NaN) in Build06 outputs, rehydrate
    them from `metadata/body_axis/summary/curvature_metrics_<exp>.csv`.
    """
    build_dir = Path(build_dir)
    summary_path = build_dir.parent / "body_axis" / "summary" / f"curvature_metrics_{experiment_id}.csv"
    if not summary_path.exists():
        return df
    if "snip_id" not in df.columns:
        return df

    needs = (
        ("baseline_deviation_normalized" not in df.columns)
        or pd.to_numeric(df["baseline_deviation_normalized"], errors="coerce").notna().sum() == 0
    )
    if not needs:
        return df

    try:
        curv = pd.read_csv(
            summary_path,
            usecols=lambda c: c in {"snip_id", "total_length_um", "baseline_deviation_um"},
        )
    except Exception:
        return df
    if curv.empty or "snip_id" not in curv.columns:
        return df

    curv = curv.dropna(subset=["snip_id"]).drop_duplicates(subset=["snip_id"]).copy()
    if "total_length_um" in curv.columns:
        curv["total_length_um"] = pd.to_numeric(curv["total_length_um"], errors="coerce")
    if "baseline_deviation_um" in curv.columns:
        curv["baseline_deviation_um"] = pd.to_numeric(curv["baseline_deviation_um"], errors="coerce")

    merged = df.merge(curv, on="snip_id", how="left", suffixes=("", "_curv"))
    if "total_length_um_curv" in merged.columns:
        merged["total_length_um"] = merged.get("total_length_um").where(
            pd.to_numeric(merged.get("total_length_um"), errors="coerce").notna(), merged["total_length_um_curv"]
        )
        merged = merged.drop(columns=["total_length_um_curv"])
    if "baseline_deviation_um_curv" in merged.columns:
        merged["baseline_deviation_um"] = merged.get("baseline_deviation_um").where(
            pd.to_numeric(merged.get("baseline_deviation_um"), errors="coerce").notna(), merged["baseline_deviation_um_curv"]
        )
        merged = merged.drop(columns=["baseline_deviation_um_curv"])

    denom = pd.to_numeric(merged.get("total_length_um"), errors="coerce")
    numer = pd.to_numeric(merged.get("baseline_deviation_um"), errors="coerce")
    norm = np.where((denom > 0) & np.isfinite(denom) & np.isfinite(numer), numer / denom, np.nan)
    norm = pd.Series(norm, index=merged.index, dtype="float64")
    if "baseline_deviation_normalized" in merged.columns:
        existing = pd.to_numeric(merged["baseline_deviation_normalized"], errors="coerce")
        merged["baseline_deviation_normalized"] = existing.where(existing.notna(), norm)
    else:
        merged["baseline_deviation_normalized"] = norm

    if verbose:
        n_ok = int(pd.to_numeric(merged["baseline_deviation_normalized"], errors="coerce").notna().sum())
        print(f"Hydrated curvature for {experiment_id} from {summary_path.name}: {n_ok}/{len(merged)} rows non-NaN")
    return merged


def load_experiment(
    experiment_id: str,
    build_dir: Union[str, Path],
    hydrate_curvature: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load a single experiment's data from build06 output.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., "20230615")
    build_dir : str or Path
        Path to build06 output directory

    Returns
    -------
    pd.DataFrame
        Experiment data with added 'source_experiment' column
    """
    build_dir = Path(build_dir)
    file_path = build_dir / f"df03_final_output_with_latents_{experiment_id}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {file_path}")

    df = pd.read_csv(file_path)
    if hydrate_curvature:
        df = _hydrate_curvature_from_body_axis(df, experiment_id=experiment_id, build_dir=build_dir, verbose=verbose)
    df['source_experiment'] = experiment_id

    return df


def load_experiments(
    experiment_ids: List[str],
    build_dir: Union[str, Path],
    verbose: bool = True,
    hydrate_curvature: bool = True,
) -> pd.DataFrame:
    """
    Load and combine multiple experiments.

    Parameters
    ----------
    experiment_ids : list of str
        List of experiment identifiers
    build_dir : str or Path
        Path to build06 output directory
    verbose : bool, default=True
        Print loading progress

    Returns
    -------
    pd.DataFrame
        Combined data from all experiments
    """
    dfs = []
    build_dir = Path(build_dir)

    for exp_id in experiment_ids:
        try:
            df = load_experiment(exp_id, build_dir, hydrate_curvature=hydrate_curvature, verbose=verbose)
            dfs.append(df)

            if verbose:
                print(f"Loaded {exp_id}: {len(df)} rows")
                print(f"  Genotypes: {df['genotype'].value_counts().to_dict()}")

        except FileNotFoundError:
            if verbose:
                print(f"Missing: {exp_id}")
            continue
        except Exception as e:
            if verbose:
                print(f"Error loading {exp_id}: {e}")
            continue

    if not dfs:
        raise ValueError("No experiments could be loaded")

    combined_df = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"\nTotal: {len(combined_df)} rows from {len(dfs)} experiments")
        print(f"Overall genotype distribution:")
        print(combined_df['genotype'].value_counts())

    return combined_df


def filter_by_genotypes(
    df: pd.DataFrame,
    genotypes: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter dataframe to specific genotypes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    genotypes : list of str
        Genotype labels to keep
    verbose : bool, default=True
        Print filtering information

    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    df_filtered = df[df['genotype'].isin(genotypes)].copy()

    if verbose:
        print(f"Filtered to {genotypes}: {len(df_filtered)} rows")
        print(f"Genotype distribution:\n{df_filtered['genotype'].value_counts()}")

    return df_filtered


def get_genotype_summary(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get summary statistics for genotypes in dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'genotype' column

    Returns
    -------
    dict
        Mapping of genotype to count
    """
    return df['genotype'].value_counts().to_dict()


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str]
) -> None:
    """
    Check that dataframe has all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    required_cols : list of str
        Required column names

    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )
