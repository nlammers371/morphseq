"""
Data loading utilities for morphseq experiments.

This module handles loading and combining experimental data from multiple
build06 output files.
"""

from pathlib import Path
from typing import List, Dict, Union
import pandas as pd


def load_experiment(
    experiment_id: str,
    build_dir: Union[str, Path]
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
    df['source_experiment'] = experiment_id

    return df


def load_experiments(
    experiment_ids: List[str],
    build_dir: Union[str, Path],
    verbose: bool = True
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
            df = load_experiment(exp_id, build_dir)
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
