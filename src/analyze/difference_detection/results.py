"""
Result containers for difference detection analyses.

This module provides structured result objects that wrap analysis outputs,
offering ergonomic access patterns for both interactive exploration and
programmatic consumption (e.g., plotting functions).

Key Classes
-----------
- MulticlassOVRResults : Container for One-vs-Rest multiclass classification results
- ComparisonSpec : Dataclass describing a single comparison (positive vs negative)

Design Philosophy
-----------------
- Store data in canonical long-format DataFrames (one row per observation)
- Provide dict-like access for human interaction: results['CE', 'WT']
- Provide iteration methods for programmatic use: results.iter_comparisons()
- Bundle metadata with data for reproducibility
- Support easy serialization (parquet + JSON)

Example
-------
>>> results = run_multiclass_classification_test(df, groupby='cluster', ...)
>>> 
>>> # Dict-like access
>>> ce_vs_wt = results['CE', 'WT']
>>> 
>>> # Iteration
>>> for (pos, neg), df_pair in results.iter_comparisons():
...     print(f"{pos} vs {neg}: max AUROC = {df_pair['auroc_obs'].max():.2f}")
>>> 
>>> # Save/load
>>> results.save('results/my_analysis/')
>>> loaded = MulticlassOVRResults.from_dir('results/my_analysis/')
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

__all__ = [
    'MulticlassOVRResults',
    'ComparisonSpec',
]


# =============================================================================
# ComparisonSpec: Internal dataclass for comparison definitions
# =============================================================================

@dataclass
class ComparisonSpec:
    """
    Specification for a single comparison between groups.
    
    Used internally by _resolve_comparison_groups() to define what comparisons
    to run before executing the classification loop.
    
    Attributes
    ----------
    positive : str
        Label for the positive/target class (e.g., "CE")
    negative : str
        Label for the negative/reference class (e.g., "WT" or "WT+Het")
    positive_members : List[str]
        List of embryo IDs in the positive class
    negative_members : List[str]
        List of embryo IDs in the negative class
    negative_mode : str
        How the negative class was constructed:
        - "rest": All other classes combined
        - "single": A single specified class
        - "pooled": Multiple classes pooled together
    
    Example
    -------
    >>> spec = ComparisonSpec(
    ...     positive="CE",
    ...     negative="WT+Het", 
    ...     positive_members=["emb1", "emb2"],
    ...     negative_members=["emb3", "emb4", "emb5"],
    ...     negative_mode="pooled"
    ... )
    """
    positive: str
    negative: str
    positive_members: List[str]
    negative_members: List[str]
    negative_mode: str  # "rest", "single", "pooled"
    
    def __post_init__(self):
        if self.negative_mode not in ("rest", "single", "pooled"):
            raise ValueError(
                f"negative_mode must be 'rest', 'single', or 'pooled', "
                f"got: {self.negative_mode}"
            )
    
    @property
    def comparison_id(self) -> str:
        """Generate a filesystem-safe comparison ID: 'CE__vs__WT+Het'"""
        # Escape any existing '__' in labels (rare but possible)
        pos_safe = self.positive.replace("__", "_")
        neg_safe = self.negative.replace("__", "_")
        return f"{pos_safe}__vs__{neg_safe}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "positive": self.positive,
            "negative": self.negative,
            "positive_members": self.positive_members,
            "negative_members": self.negative_members,
            "negative_mode": self.negative_mode,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComparisonSpec":
        """Reconstruct from dictionary."""
        return cls(
            positive=d["positive"],
            negative=d["negative"],
            positive_members=d["positive_members"],
            negative_members=d["negative_members"],
            negative_mode=d["negative_mode"],
        )


# =============================================================================
# MulticlassOVRResults: Main result container
# =============================================================================

@dataclass
class MulticlassOVRResults:
    """
    Container for One-vs-Rest multiclass difference detection results.
    
    Stores results in a canonical long-format DataFrame with dict-like access
    for human interaction and iterator methods for programmatic consumption.
    
    Attributes
    ----------
    comparisons : pd.DataFrame
        The canonical long-format table containing all test results.
        Required columns: ['positive', 'negative', 'comparison_id']
        Typical columns: ['time_bin', 'time_bin_center', 'auroc_obs', 'pval', 
                          'n_pos', 'n_neg', 'negative_members', 'negative_mode']
    metadata : dict
        Parameters used for the run. Common keys:
        - 'groupby': Column name used for grouping
        - 'groups': Which groups were tested
        - 'reference': Reference specification used
        - 'bin_width': Time binning width
        - 'n_permutations': Number of permutations
        - 'timestamp': When the analysis was run
        - 'version': Package version
    
    Examples
    --------
    Dict-like access:
    
    >>> ce_vs_wt = results['CE', 'WT']
    >>> ce_vs_wt = results[('CE', 'WT')]  # Also works
    
    Iteration:
    
    >>> for (pos, neg), df in results.iter_comparisons():
    ...     print(f"{pos} vs {neg}")
    
    >>> for key, df in results.items():  # Dict-like
    ...     pos, neg = key
    
    Filtering:
    
    >>> sig_only = results.filter(pval_lt=0.01)
    >>> ce_results = results.filter(positive='CE')
    
    Faceted access:
    
    >>> for neg, sub in results.by_negative().items():
    ...     for pos, df in sub.items():
    ...         # Plot pos vs neg
    
    Save/load:
    
    >>> results.save('output/multiclass_ovr/')
    >>> loaded = MulticlassOVRResults.from_dir('output/multiclass_ovr/')
    """
    comparisons: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    
    def __post_init__(self):
        """Validate that required columns exist."""
        required = {'positive', 'negative', 'comparison_id'}
        missing = required - set(self.comparisons.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}. "
                f"Required: {required}"
            )
    
    # -------------------------------------------------------------------------
    # Dict-like Access (Human Interaction)
    # -------------------------------------------------------------------------
    
    def __getitem__(self, key: Union[Tuple[str, str], str]) -> pd.DataFrame:
        """
        Get results for a specific comparison.
        
        Parameters
        ----------
        key : tuple of (positive, negative)
            The comparison to retrieve. Supports both:
            - results['CE', 'WT']
            - results[('CE', 'WT')]
        
        Returns
        -------
        pd.DataFrame
            Filtered DataFrame for that comparison.
        
        Raises
        ------
        KeyError
            If the comparison is not found.
        """
        # Handle both results['CE', 'WT'] and results[('CE', 'WT')]
        if isinstance(key, tuple) and len(key) == 2:
            pos, neg = key
        else:
            raise KeyError(
                f"Key must be (positive, negative) tuple, got: {key!r}. "
                f"Example: results['CE', 'WT'] or results[('CE', 'WT')]"
            )
        
        mask = (
            (self.comparisons['positive'] == pos) & 
            (self.comparisons['negative'] == neg)
        )
        df = self.comparisons.loc[mask].copy()
        
        if df.empty:
            available = self.keys()
            raise KeyError(
                f"No results found for ('{pos}', '{neg}'). "
                f"Available comparisons: {available}"
            )
        
        return df
    
    def get(
        self, 
        positive: str, 
        negative: str, 
        default: Optional[pd.DataFrame] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get results for a specific comparison, with optional default.
        
        Parameters
        ----------
        positive : str
            Positive class label
        negative : str
            Negative class label
        default : pd.DataFrame, optional
            Value to return if comparison not found (default: None)
        
        Returns
        -------
        pd.DataFrame or default
            Results for the comparison, or default if not found.
        """
        try:
            return self[(positive, negative)]
        except KeyError:
            return default
    
    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Check if a comparison exists: ('CE', 'WT') in results"""
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        pos, neg = key
        mask = (
            (self.comparisons['positive'] == pos) & 
            (self.comparisons['negative'] == neg)
        )
        return mask.any()
    
    def keys(self) -> List[Tuple[str, str]]:
        """
        List all (positive, negative) comparison pairs.
        
        Returns
        -------
        List[Tuple[str, str]]
            List of (positive, negative) tuples in stable order.
        
        Example
        -------
        >>> results.keys()
        [('CE', 'WT'), ('CE', 'WT+Het'), ('HTA', 'WT'), ('HTA', 'WT+Het')]
        """
        pairs = self.comparisons[['positive', 'negative']].drop_duplicates()
        return list(pairs.itertuples(index=False, name=None))
    
    def list_comparisons(self) -> List[Tuple[str, str]]:
        """Alias for keys() - list all comparison pairs."""
        return self.keys()
    
    def items(self) -> Iterator[Tuple[Tuple[str, str], pd.DataFrame]]:
        """
        Iterate over comparisons like a dict.
        
        Yields
        ------
        ((positive, negative), DataFrame)
            Tuple of comparison key and filtered DataFrame.
        
        Example
        -------
        >>> for key, df in results.items():
        ...     pos, neg = key
        ...     print(f"{pos} vs {neg}: {len(df)} time bins")
        """
        for (pos, neg), df in self.comparisons.groupby(
            ['positive', 'negative'], sort=False
        ):
            yield (pos, neg), df.copy()
    
    def __len__(self) -> int:
        """Number of unique comparisons."""
        return len(self.keys())
    
    # -------------------------------------------------------------------------
    # Iteration Methods (Programmatic Use)
    # -------------------------------------------------------------------------
    
    def iter_comparisons(
        self,
        positive: Optional[str] = None,
        negative: Optional[str] = None,
        sort: bool = False,
    ) -> Iterator[Tuple[Tuple[str, str], pd.DataFrame]]:
        """
        Iterate over comparisons with optional filtering.
        
        Parameters
        ----------
        positive : str, optional
            Filter to only this positive class
        negative : str, optional
            Filter to only this negative class
        sort : bool
            If True, sort by (positive, negative) alphabetically
        
        Yields
        ------
        ((positive, negative), DataFrame)
            Comparison key and filtered DataFrame.
        
        Examples
        --------
        >>> # All comparisons
        >>> for (pos, neg), df in results.iter_comparisons():
        ...     print(f"{pos} vs {neg}")
        
        >>> # Only comparisons against WT
        >>> for (pos, neg), df in results.iter_comparisons(negative='WT'):
        ...     print(f"{pos} vs WT")
        
        >>> # Only CE comparisons
        >>> for (pos, neg), df in results.iter_comparisons(positive='CE'):
        ...     print(f"CE vs {neg}")
        """
        df = self.comparisons
        
        # Apply filters
        if positive is not None:
            df = df[df['positive'] == positive]
        if negative is not None:
            df = df[df['negative'] == negative]
        
        # Iterate
        for (pos, neg), group in df.groupby(
            ['positive', 'negative'], sort=sort
        ):
            yield (pos, neg), group.copy()
    
    # -------------------------------------------------------------------------
    # Structured Views (Plotting Helpers)
    # -------------------------------------------------------------------------
    
    def by_negative(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Reorganize into nested dict keyed by negative (reference) class.
        
        Useful for faceted plotting where columns = reference groups.
        
        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Structure: { 'WT': {'CE': df, 'HTA': df}, 'WT+Het': {...} }
        
        Example
        -------
        >>> for neg, sub in results.by_negative().items():
        ...     print(f"=== Reference: {neg} ===")
        ...     for pos, df in sub.items():
        ...         print(f"  {pos}: {len(df)} time bins")
        """
        tree: Dict[str, Dict[str, pd.DataFrame]] = {}
        for (pos, neg), df in self.items():
            if neg not in tree:
                tree[neg] = {}
            tree[neg][pos] = df
        return tree
    
    def by_positive(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Reorganize into nested dict keyed by positive (target) class.
        
        Useful for faceted plotting where columns = target groups.
        
        Returns
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Structure: { 'CE': {'WT': df, 'WT+Het': df}, 'HTA': {...} }
        """
        tree: Dict[str, Dict[str, pd.DataFrame]] = {}
        for (pos, neg), df in self.items():
            if pos not in tree:
                tree[pos] = {}
            tree[pos][neg] = df
        return tree
    
    def to_dict_of_dfs(self) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Convert to legacy dict-of-DataFrames format.
        
        For backwards compatibility with older plotting code that expects
        {('CE', 'WT'): df, ('HTA', 'WT'): df, ...}
        
        Returns
        -------
        Dict[Tuple[str, str], pd.DataFrame]
            Dictionary keyed by (positive, negative) tuples.
        
        Note
        ----
        Prefer using iter_comparisons() or items() for new code.
        """
        return dict(self.items())
    
    # Alias for compatibility
    to_legacy_dict = to_dict_of_dfs
    
    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------
    
    def filter(
        self,
        positive: Optional[str] = None,
        negative: Optional[str] = None,
        time_bin: Optional[int] = None,
        pval_lt: Optional[float] = None,
        auroc_gt: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Filter the comparisons DataFrame with readable named parameters.
        
        Parameters
        ----------
        positive : str, optional
            Filter to this positive class
        negative : str, optional
            Filter to this negative class
        time_bin : int, optional
            Filter to this time bin
        pval_lt : float, optional
            Filter to p-values less than this threshold
        auroc_gt : float, optional
            Filter to AUROC greater than this threshold
        
        Returns
        -------
        pd.DataFrame
            Filtered copy of the comparisons DataFrame.
        
        Examples
        --------
        >>> results.filter(positive='CE', pval_lt=0.01)
        >>> results.filter(time_bin=16, negative='WT')
        >>> results.filter(pval_lt=0.05, auroc_gt=0.7)
        """
        df = self.comparisons.copy()
        
        if positive is not None:
            df = df[df['positive'] == positive]
        if negative is not None:
            df = df[df['negative'] == negative]
        if time_bin is not None:
            df = df[df['time_bin'] == time_bin]
        if pval_lt is not None:
            if 'pval' in df.columns:
                df = df[df['pval'] < pval_lt]
            else:
                warnings.warn("No 'pval' column found, pval_lt filter ignored")
        if auroc_gt is not None:
            auroc_col = 'auroc_obs' if 'auroc_obs' in df.columns else 'auroc_observed'
            if auroc_col in df.columns:
                df = df[df[auroc_col] > auroc_gt]
            else:
                warnings.warn("No AUROC column found, auroc_gt filter ignored")
        
        return df
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def save(
        self, 
        path: Union[str, Path], 
        format: str = 'parquet',
        overwrite: bool = False,
    ) -> Path:
        """
        Save results to a directory.
        
        Creates:
        - comparisons.parquet (or .csv)
        - metadata.json
        
        Parameters
        ----------
        path : str or Path
            Directory to save to. Will be created if it doesn't exist.
        format : str
            'parquet' (default, recommended) or 'csv'
        overwrite : bool
            If True, overwrite existing files
        
        Returns
        -------
        Path
            Path to the output directory.
        
        Example
        -------
        >>> results.save('results/my_analysis/multiclass_ovr/')
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save comparisons DataFrame
        if format == 'parquet':
            comp_path = path / 'comparisons.parquet'
            if comp_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{comp_path} exists. Use overwrite=True to replace."
                )
            self.comparisons.to_parquet(comp_path, index=False)
        elif format == 'csv':
            comp_path = path / 'comparisons.csv'
            if comp_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{comp_path} exists. Use overwrite=True to replace."
                )
            self.comparisons.to_csv(comp_path, index=False)
        else:
            raise ValueError(f"format must be 'parquet' or 'csv', got: {format}")
        
        # Save metadata
        meta_path = path / 'metadata.json'
        if meta_path.exists() and not overwrite:
            raise FileExistsError(
                f"{meta_path} exists. Use overwrite=True to replace."
            )
        
        # Prepare metadata (convert non-serializable types)
        meta_to_save = self._prepare_metadata_for_json(self.metadata)
        with open(meta_path, 'w') as f:
            json.dump(meta_to_save, f, indent=2, default=str)
        
        return path
    
    @classmethod
    def from_dir(cls, path: Union[str, Path]) -> "MulticlassOVRResults":
        """
        Load results from a directory.
        
        Parameters
        ----------
        path : str or Path
            Directory containing comparisons.parquet (or .csv) and metadata.json
        
        Returns
        -------
        MulticlassOVRResults
            Reconstructed results object.
        
        Example
        -------
        >>> results = MulticlassOVRResults.from_dir('results/my_analysis/')
        """
        path = Path(path)
        
        # Load comparisons (prefer parquet over csv)
        parquet_path = path / 'comparisons.parquet'
        csv_path = path / 'comparisons.csv'
        
        if parquet_path.exists():
            comparisons = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            comparisons = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"No comparisons file found in {path}. "
                f"Expected 'comparisons.parquet' or 'comparisons.csv'"
            )
        
        # Load metadata
        meta_path = path / 'metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
            warnings.warn(f"No metadata.json found in {path}")
        
        return cls(comparisons=comparisons, metadata=metadata)
    
    @staticmethod
    def _prepare_metadata_for_json(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata values to JSON-serializable types."""
        result = {}
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                # Handle nested tuples (e.g., reference=[('WT', 'Het')])
                result[key] = [
                    list(v) if isinstance(v, tuple) else v 
                    for v in value
                ]
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, Path):
                result[key] = str(value)
            elif hasattr(value, 'tolist'):  # numpy arrays
                result[key] = value.tolist()
            else:
                result[key] = value
        return result
    
    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        n_pairs = len(self.keys())
        n_rows = len(self.comparisons)
        groupby = self.metadata.get('groupby', 'unknown')
        return (
            f"<MulticlassOVRResults: {n_pairs} comparisons, "
            f"{n_rows} rows (groupby='{groupby}')>"
        )
    
    def summary(self) -> pd.DataFrame:
        """
        Generate a summary table of all comparisons.
        
        Returns
        -------
        pd.DataFrame
            Summary with columns: positive, negative, n_time_bins, 
            max_auroc, min_pval, n_significant
        """
        records = []
        for (pos, neg), df in self.items():
            auroc_col = 'auroc_obs' if 'auroc_obs' in df.columns else 'auroc_observed'
            
            record = {
                'positive': pos,
                'negative': neg,
                'n_time_bins': len(df),
            }
            
            if auroc_col in df.columns:
                record['max_auroc'] = df[auroc_col].max()
            
            if 'pval' in df.columns:
                record['min_pval'] = df['pval'].min()
                record['n_significant_01'] = (df['pval'] < 0.01).sum()
                record['n_significant_05'] = (df['pval'] < 0.05).sum()
            
            records.append(record)
        
        return pd.DataFrame(records)
