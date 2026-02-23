"""
File I/O and path utilities for analysis outputs.

This module provides helper functions for generating consistent file paths
and managing analysis outputs.
"""

import os
from typing import Tuple


def make_safe_comparison_name(group1: str, group2: str) -> str:
    """
    Generate a safe filename for a genotype comparison.

    Parameters
    ----------
    group1, group2 : str
        Genotype labels (e.g., "cep290_wildtype", "cep290_homozygous")

    Returns
    -------
    str
        Safe filename (e.g., "wildtype_vs_homozygous")

    Examples
    --------
    >>> make_safe_comparison_name("cep290_wildtype", "cep290_homozygous")
    'wildtype_vs_homozygous'
    """
    name1 = group1.split('_')[-1]
    name2 = group2.split('_')[-1]
    return f"{name1}_vs_{name2}"


def get_plot_path(
    plot_dir: str,
    gene: str,
    plot_type: str,
    comparison_name: str,
    extension: str = "png"
) -> str:
    """
    Generate standardized plot file path.

    Parameters
    ----------
    plot_dir : str
        Base directory for plots
    gene : str
        Gene name (e.g., "cep290")
    plot_type : str
        Type of plot (e.g., "auroc", "heatmap", "trajectories")
    comparison_name : str
        Comparison identifier (e.g., "wildtype_vs_homozygous")
    extension : str, default="png"
        File extension

    Returns
    -------
    str
        Full path to plot file

    Examples
    --------
    >>> get_plot_path("/plots", "cep290", "auroc", "wildtype_vs_homozygous")
    '/plots/cep290/auroc_wildtype_vs_homozygous.png'
    """
    gene_plot_dir = os.path.join(plot_dir, gene)
    os.makedirs(gene_plot_dir, exist_ok=True)

    filename = f"{plot_type}_{comparison_name}.{extension}"
    return os.path.join(gene_plot_dir, filename)


def get_data_path(
    data_dir: str,
    gene: str,
    data_type: str,
    comparison_name: str = None,
    extension: str = "csv"
) -> str:
    """
    Generate standardized data file path.

    Parameters
    ----------
    data_dir : str
        Base directory for data
    gene : str
        Gene name (e.g., "cep290")
    data_type : str
        Type of data (e.g., "embryo_predictions", "penetrance", "auroc_results")
    comparison_name : str or None
        Comparison identifier (optional)
    extension : str, default="csv"
        File extension

    Returns
    -------
    str
        Full path to data file

    Examples
    --------
    >>> get_data_path("/data", "cep290", "embryo_predictions", "wildtype_vs_homozygous")
    '/data/cep290/embryo_predictions_wildtype_vs_homozygous.csv'
    """
    gene_data_dir = os.path.join(data_dir, gene)
    os.makedirs(gene_data_dir, exist_ok=True)

    if comparison_name:
        filename = f"{data_type}_{comparison_name}.{extension}"
    else:
        filename = f"{data_type}.{extension}"

    return os.path.join(gene_data_dir, filename)


def save_dataframe(
    df,
    filepath: str,
    verbose: bool = True
) -> None:
    """
    Save dataframe to CSV with optional logging.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    filepath : str
        Output file path
    verbose : bool, default=True
        Print save confirmation
    """
    df.to_csv(filepath, index=False)

    if verbose:
        print(f"  Saved: {filepath}")


def extract_genotype_short_names(group1: str, group2: str) -> Tuple[str, str]:
    """
    Extract short genotype names for display.

    Parameters
    ----------
    group1, group2 : str
        Full genotype labels

    Returns
    -------
    tuple of str
        Short names (e.g., ("WT", "HOM"))

    Examples
    --------
    >>> extract_genotype_short_names("cep290_wildtype", "cep290_homozygous")
    ('wildtype', 'homozygous')
    """
    name1 = group1.split('_')[-1]
    name2 = group2.split('_')[-1]
    return name1, name2
