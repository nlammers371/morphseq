"""
Phenotype file I/O utilities.

Simple functions to load and save phenotype lists (embryo IDs) from text files.
Handles common formats including simple lists, pair headers, and comments.

Example
-------
>>> from morphseq.trajectory_analysis import load_phenotype_file, save_phenotype_file
>>>
>>> # Load phenotype list
>>> ce_ids = load_phenotype_file('phenotype_lists/ce_embryos.txt')
>>>
>>> # Save phenotype list
>>> save_phenotype_file(ce_ids, 'phenotype_lists/my_phenotype.txt')
"""

from pathlib import Path
from typing import List, Optional, Union


def load_phenotype_file(
    filepath: Union[str, Path],
    skip_headers: bool = True,
    skip_comments: bool = True,
) -> List[str]:
    """
    Load embryo IDs from phenotype text file.

    Handles files with:
    - Simple list of embryo_ids (one per line)
    - Files with "b9d2_pair_X" or similar headers (skipped if skip_headers=True)
    - Comment lines starting with # (skipped if skip_comments=True)
    - Inline comments after embryo IDs

    Parameters
    ----------
    filepath : str or Path
        Path to phenotype file
    skip_headers : bool, default=True
        Skip lines that look like headers (e.g., "b9d2_pair_2")
    skip_comments : bool, default=True
        Skip lines starting with #

    Returns
    -------
    embryo_ids : List[str]
        List of embryo IDs

    Examples
    --------
    >>> ce_ids = load_phenotype_file('phenotype_lists/ce_embryos.txt')
    >>> len(ce_ids)
    15

    >>> # File format can be:
    >>> # 20251121_A01_e01
    >>> # 20251121_A02_e01  # this is a comment
    >>> # b9d2_pair_2  <- this header is skipped
    >>> # 20251121_B01_e01
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Phenotype file not found: {filepath}")

    embryo_ids = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip comment lines
            if skip_comments and line.startswith('#'):
                continue

            # Skip header lines (e.g., "b9d2_pair_2", "pair_8")
            if skip_headers:
                # Common header patterns
                if line.startswith('b9d2_pair') or line.startswith('pair_'):
                    continue
                # Could add more header patterns here

            # Handle inline comments
            if skip_comments and '#' in line:
                line = line.split('#')[0].strip()
                if not line:
                    continue

            embryo_ids.append(line)

    return embryo_ids


def save_phenotype_file(
    embryo_ids: List[str],
    filepath: Union[str, Path],
    header: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Save embryo IDs to phenotype text file.

    Creates a simple text file with one embryo ID per line.

    Parameters
    ----------
    embryo_ids : List[str]
        List of embryo IDs to save
    filepath : str or Path
        Path to save file
    header : str, optional
        Optional header comment to add at top of file
    overwrite : bool, default=False
        If True, overwrite existing file. If False, raise error if file exists.

    Examples
    --------
    >>> save_phenotype_file(ce_ids, 'phenotype_lists/my_phenotype.txt')

    >>> # With header comment
    >>> save_phenotype_file(
    ...     ce_ids,
    ...     'phenotype_lists/my_phenotype.txt',
    ...     header='CE phenotype embryos from k=4 cluster 2'
    ... )
    """
    filepath = Path(filepath)

    # Check if file exists
    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. Use overwrite=True to replace."
        )

    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        # Write header if provided
        if header:
            f.write(f"# {header}\n")

        # Write embryo IDs
        for eid in embryo_ids:
            f.write(f"{eid}\n")

    print(f"Saved {len(embryo_ids)} embryo IDs to: {filepath}")
