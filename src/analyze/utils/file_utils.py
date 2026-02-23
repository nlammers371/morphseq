"""
File I/O and path utilities for analysis outputs.

This module provides helper functions for generating consistent file paths
and managing analysis outputs.
"""

from pathlib import Path
from typing import Tuple, Union


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
    plot_dir: Union[str, Path],
    gene: str,
    plot_type: str,
    comparison_name: str,
    extension: str = "png"
) -> Path:
    """
    Generate standardized plot file path.

    Parameters
    ----------
    plot_dir : str or Path
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
    Path
        Full path to plot file

    Examples
    --------
    >>> path = get_plot_path("/plots", "cep290", "auroc", "wildtype_vs_homozygous")
    >>> str(path)
    '/plots/cep290/auroc_wildtype_vs_homozygous.png'
    """
    plot_dir = Path(plot_dir)
    gene_plot_dir = plot_dir / gene
    gene_plot_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{plot_type}_{comparison_name}.{extension}"
    return gene_plot_dir / filename


def get_data_path(
    data_dir: Union[str, Path],
    gene: str,
    data_type: str,
    comparison_name: str = None,
    extension: str = "csv"
) -> Path:
    """
    Generate standardized data file path.

    Parameters
    ----------
    data_dir : str or Path
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
    Path
        Full path to data file

    Examples
    --------
    >>> path = get_data_path("/data", "cep290", "embryo_predictions", "wildtype_vs_homozygous")
    >>> str(path)
    '/data/cep290/embryo_predictions_wildtype_vs_homozygous.csv'
    """
    data_dir = Path(data_dir)
    gene_data_dir = data_dir / gene
    gene_data_dir.mkdir(parents=True, exist_ok=True)

    if comparison_name:
        filename = f"{data_type}_{comparison_name}.{extension}"
    else:
        filename = f"{data_type}.{extension}"

    return gene_data_dir / filename


def save_dataframe(
    df,
    filepath: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Save dataframe to CSV with optional logging.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to save
    filepath : str or Path
        Output file path
    verbose : bool, default=True
        Print save confirmation
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
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
