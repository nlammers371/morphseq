#!/usr/bin/env python3
"""
Load and prepare curvature + embedding data for temporal analysis.

This module provides a single source of truth for loading and merging the
curvature summary data with full metadata (including embeddings). All downstream
analyses import from here.

Functions
---------
load_curvature_data() : Load curvature metrics
load_embedding_metadata() : Load main metadata with latent embeddings
merge_data() : Merge curvature and embeddings on snip_id
normalize_metrics() : Add normalized curvature columns
get_analysis_dataframe() : Full pipeline in one call
"""

from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
METADATA_ROOT = PROJECT_ROOT / 'morphseq_playground' / 'metadata'

CURVATURE_SUMMARY_FILE = (
    METADATA_ROOT / 'body_axis' / 'summary' / 'curvature_metrics_summary_20251017_combined.csv'
)

EMBEDDING_METADATA_FILE = (
    METADATA_ROOT / 'build06_output' / 'df03_final_output_with_latents_20251017_combined.csv'
)

# Analysis parameters
GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
GENOTYPE_SHORT = {
    'cep290_wildtype': 'WT',
    'cep290_heterozygous': 'Het',
    'cep290_homozygous': 'Homo'
}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_curvature_data(filter_genotypes=True):
    """
    Load curvature summary metrics.

    Parameters
    ----------
    filter_genotypes : bool
        If True, filter to only the three target genotypes

    Returns
    -------
    pd.DataFrame
        Curvature metrics with columns: snip_id, embryo_id, frame_index,
        predicted_stage_hpf, genotype, arc_length_ratio, baseline_deviation_um,
        total_length_um, mean_curvature_per_um, etc.

    Raises
    ------
    FileNotFoundError
        If curvature file not found
    """
    if not CURVATURE_SUMMARY_FILE.exists():
        raise FileNotFoundError(f"Curvature summary file not found: {CURVATURE_SUMMARY_FILE}")

    print(f"\n  Loading curvature data from {CURVATURE_SUMMARY_FILE.name}...")
    df = pd.read_csv(CURVATURE_SUMMARY_FILE)

    print(f"    Loaded {len(df)} curvature measurements")
    print(f"    Columns: {', '.join(df.columns[:10])}...")

    if filter_genotypes:
        initial_count = len(df)
        df = df[df['genotype'].isin(GENOTYPES)].copy()
        filtered_count = len(df)
        print(f"    Filtered to {filtered_count} timepoints ({filtered_count}/{initial_count})")

    return df


def load_embedding_metadata(filter_genotypes=True):
    """
    Load main metadata with latent embeddings.

    Parameters
    ----------
    filter_genotypes : bool
        If True, filter to only the three target genotypes

    Returns
    -------
    pd.DataFrame
        Main metadata with embedding columns (look for 'latent*' columns)

    Raises
    ------
    FileNotFoundError
        If embedding metadata file not found
    """
    if not EMBEDDING_METADATA_FILE.exists():
        raise FileNotFoundError(f"Embedding metadata file not found: {EMBEDDING_METADATA_FILE}")

    print(f"\n  Loading metadata with embeddings from {EMBEDDING_METADATA_FILE.name}...")
    df = pd.read_csv(EMBEDDING_METADATA_FILE)

    print(f"    Loaded {len(df)} samples")

    # Find embedding columns
    embedding_cols = [col for col in df.columns if 'latent' in col.lower()]
    print(f"    Found {len(embedding_cols)} embedding dimensions")

    if filter_genotypes:
        initial_count = len(df)
        df = df[df['genotype'].isin(GENOTYPES)].copy()
        filtered_count = len(df)
        print(f"    Filtered to {filtered_count} samples ({filtered_count}/{initial_count})")

    return df


def get_embedding_columns(df):
    """
    Auto-detect embedding/latent columns in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list of str
        Column names containing embeddings (sorted)
    """
    # Look for z_mu columns (VAE embeddings)
    embedding_cols = [col for col in df.columns if col.startswith('z_mu_')]

    # Fallback to latent if z_mu not found
    if not embedding_cols:
        embedding_cols = [col for col in df.columns if 'latent' in col.lower()]

    return sorted(embedding_cols)


# ============================================================================
# Data Merging
# ============================================================================

def merge_data(curvature_df, metadata_df, on='snip_id', how='inner'):
    """
    Merge curvature data with embedding metadata.

    Parameters
    ----------
    curvature_df : pd.DataFrame
        Curvature metrics (from load_curvature_data)
    metadata_df : pd.DataFrame
        Metadata with embeddings (from load_embedding_metadata)
    on : str
        Column name to merge on (default: 'snip_id')
    how : str
        Merge method ('inner', 'left', 'right', 'outer')

    Returns
    -------
    pd.DataFrame
        Merged dataframe with all curvature metrics and embedding columns

    Notes
    -----
    Inner merge by default to ensure both curvature and embeddings are present
    for every row. Keeps genotype from metadata (source of truth).
    """
    print(f"\n  Merging curvature and metadata on '{on}'...")

    # Check that merge column exists
    if on not in curvature_df.columns:
        raise ValueError(f"Column '{on}' not found in curvature data")
    if on not in metadata_df.columns:
        raise ValueError(f"Column '{on}' not found in metadata")

    # Keep only curvature-specific columns from curvature_df
    # Drop columns that will come from metadata (embryo_id, predicted_stage_hpf, frame_index, genotype)
    curvature_only_cols = [col for col in curvature_df.columns
                          if col == 'snip_id'  # Keep merge key
                          or 'curvature' in col.lower() or 'baseline' in col.lower()
                          or 'arc_length' in col.lower() or 'keypoint' in col.lower()
                          or 'total_length' in col.lower() or 'um_per_pixel' in col.lower()]

    curvature_df_clean = curvature_df[curvature_only_cols].copy()

    # Merge - metadata provides embryo_id, predicted_stage_hpf, frame_index, genotype, embeddings
    merged = curvature_df_clean.merge(metadata_df, on=on, how=how, suffixes=('_curv', '_meta'))

    print(f"    Merged {len(merged)} samples (curvature: {len(curvature_df)}, metadata: {len(metadata_df)})")

    return merged


# ============================================================================
# Metric Normalization
# ============================================================================

def normalize_metrics(df):
    """
    Add normalized curvature metrics for cross-embryo comparison.

    Creates new columns:
    - normalized_baseline_deviation: baseline_deviation_um / total_length_um
    - arc_length_ratio_std: standardized arc_length_ratio within genotype

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Copy with new normalized columns added
    """
    df = df.copy()

    print(f"\n  Normalizing metrics...")

    # Normalize baseline deviation by embryo length
    if 'baseline_deviation_um' in df.columns and 'total_length_um' in df.columns:
        df['normalized_baseline_deviation'] = (
            df['baseline_deviation_um'] / df['total_length_um']
        )
        print(f"    Added normalized_baseline_deviation (baseline_deviation / total_length)")

    # Standardize arc_length_ratio within each genotype
    if 'arc_length_ratio' in df.columns:
        df['arc_length_ratio_std'] = df.groupby('genotype')['arc_length_ratio'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        print(f"    Added arc_length_ratio_std (within-genotype standardized)")

    return df


# ============================================================================
# Main Pipeline
# ============================================================================

def get_analysis_dataframe(normalize=True):
    """
    Load and merge all data in one call.

    This is the main entry point for all downstream analyses.

    Parameters
    ----------
    normalize : bool
        If True, add normalized metric columns

    Returns
    -------
    pd.DataFrame
        Complete merged dataframe ready for analysis
    dict
        Metadata including: embedding_cols, genotypes, genotype_labels
    """
    print("\n" + "="*80)
    print("LOADING AND MERGING CURVATURE + EMBEDDING DATA")
    print("="*80)

    # Load individual components
    curv_df = load_curvature_data(filter_genotypes=True)
    meta_df = load_embedding_metadata(filter_genotypes=True)

    # Merge
    df = merge_data(curv_df, meta_df, on='snip_id', how='inner')

    # Normalize
    if normalize:
        df = normalize_metrics(df)

    # Get embedding columns
    embedding_cols = get_embedding_columns(df)

    # Summary statistics
    print(f"\n  Final dataset summary:")
    print(f"    Total samples: {len(df)}")
    print(f"    Unique embryos: {df['embryo_id'].nunique()}")
    print(f"    Genotype distribution:")

    for genotype in GENOTYPES:
        count = (df['genotype'] == genotype).sum()
        pct = 100 * count / len(df)
        short = GENOTYPE_SHORT[genotype]
        print(f"      {short:5s} ({genotype:25s}): {count:4d} ({pct:5.1f}%)")

    print(f"    Embedding dimensions: {len(embedding_cols)}")

    metadata = {
        'embedding_cols': embedding_cols,
        'genotypes': GENOTYPES,
        'genotype_labels': GENOTYPE_SHORT,
        'n_samples': len(df),
        'n_embryos': df['embryo_id'].nunique()
    }

    print("\n" + "="*80)
    print("DATA LOADING COMPLETE")
    print("="*80 + "\n")

    return df, metadata


# ============================================================================
# Helper Functions for Downstream Use
# ============================================================================

def get_genotype_short_name(genotype):
    """Get short label for a genotype."""
    return GENOTYPE_SHORT.get(genotype, genotype)


def get_genotype_color(genotype):
    """Get a standard color for a genotype for consistent plotting."""
    colors = {
        'cep290_wildtype': '#1f77b4',    # Blue
        'cep290_heterozygous': '#ff7f0e',  # Orange
        'cep290_homozygous': '#d62728'    # Red
    }
    return colors.get(genotype, '#7f7f7f')  # Gray fallback


if __name__ == '__main__':
    # Test the data loading pipeline
    df, metadata = get_analysis_dataframe()

    print("\nDataframe shape:", df.shape)
    print("Columns:", list(df.columns[:20]))
    print("\nMetadata:", metadata)

    # Show sample of data
    print("\nSample of data:")
    print(df[['embryo_id', 'genotype', 'arc_length_ratio', 'normalized_baseline_deviation',
             'predicted_stage_hpf']].head(10))
