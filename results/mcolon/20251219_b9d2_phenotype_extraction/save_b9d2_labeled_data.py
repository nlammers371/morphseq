"""
Save B9D2 Labeled Data

Loads b9d2 experiment data, applies phenotype labels from manual curation files,
and saves complete dataframe with cluster_categories column to CSV for streamlined
downstream analysis.

This eliminates the need to reload and re-label data for each analysis script.

Usage:
    python save_b9d2_labeled_data.py

Output:
    - data/b9d2_labeled_data.csv (complete labeled dataframe)

Author: Generated via Claude Code
Date: 2026-01-05
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_IDS = ['20251121', '20251125']

# Phenotype file paths
PHENOTYPE_DIR = Path(__file__).parent / 'phenotype_lists'
CE_FILE = PHENOTYPE_DIR / 'b9d2-CE-phenotype.txt'
HTA_FILE = PHENOTYPE_DIR / 'b9d2-HTA-embryos.txt'
BA_RESCUE_FILE = PHENOTYPE_DIR / 'b9d2-curved-rescue.txt'

# Output path
OUTPUT_DIR = Path(__file__).parent / 'data'
OUTPUT_FILE = OUTPUT_DIR / 'b9d2_labeled_data.csv'

# Column names
EMBRYO_ID_COL = 'embryo_id'
GENOTYPE_COL = 'genotype'


# =============================================================================
# Phenotype File Parsing
# =============================================================================

def parse_phenotype_file(filepath: Path) -> List[str]:
    """
    Parse phenotype file to extract embryo IDs.

    Handles files with:
    - Simple list of embryo_ids (one per line)
    - Files with "b9d2_pair_X" headers (skip those lines)
    - Comments with # (stripped)

    Parameters
    ----------
    filepath : Path
        Path to phenotype file

    Returns
    -------
    embryo_ids : List[str]
        List of embryo IDs
    """
    embryo_ids = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip pair headers
            if line.startswith('b9d2_pair'):
                continue

            # Skip comments
            if '#' in line:
                # Take only the part before the comment
                line = line.split('#')[0].strip()
                if not line:
                    continue

            embryo_ids.append(line)

    return embryo_ids


def load_phenotype_lists() -> Dict[str, List[str]]:
    """
    Load all phenotype lists from files.

    Returns
    -------
    phenotype_dict : Dict[str, List[str]]
        Dictionary mapping phenotype name to list of embryo IDs
    """
    phenotype_dict = {
        'CE': parse_phenotype_file(CE_FILE),
        'HTA': parse_phenotype_file(HTA_FILE),
        'BA_rescue': parse_phenotype_file(BA_RESCUE_FILE),
    }

    print("Loaded phenotype lists:")
    for phenotype, ids in phenotype_dict.items():
        print(f"  {phenotype}: {len(ids)} embryos")

    return phenotype_dict


# =============================================================================
# Data Loading
# =============================================================================

def load_experiment_data() -> pd.DataFrame:
    """
    Load experiment data from standard locations.

    Returns
    -------
    df_combined : pd.DataFrame
        Combined dataframe from all experiments
    """
    print(f"\nLoading experiment data from {len(EXPERIMENT_IDS)} experiments...")

    dfs = []
    for exp_id in EXPERIMENT_IDS:
        print(f"  Loading experiment {exp_id}...")
        df = load_experiment_dataframe(exp_id, format_version='df03')
        df['experiment_id'] = exp_id
        dfs.append(df)

    df_combined = pd.concat(dfs, ignore_index=True)

    # Handle column name variations for baseline deviation
    if 'baseline_deviation_normalized' not in df_combined.columns:
        if 'normalized_baseline_deviation' in df_combined.columns:
            df_combined['baseline_deviation_normalized'] = df_combined['normalized_baseline_deviation']
        elif 'baseline_deviation_um' in df_combined.columns and 'total_length_um' in df_combined.columns:
            # Normalize by total length if raw values exist
            df_combined['baseline_deviation_normalized'] = (
                df_combined['baseline_deviation_um'] / df_combined['total_length_um']
            )

    print(f"  Loaded {len(df_combined)} rows, {df_combined[EMBRYO_ID_COL].nunique()} unique embryos")

    return df_combined


def extract_wildtype_embryos(df: pd.DataFrame, phenotype_dict: Dict[str, List[str]]) -> List[str]:
    """
    Extract wildtype embryo IDs from genotype column, excluding phenotype embryos.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe
    phenotype_dict : Dict[str, List[str]]
        Dictionary of phenotype embryo IDs

    Returns
    -------
    wildtype_ids : List[str]
        List of wildtype embryo IDs
    """
    # Get all embryos labeled as b9d2_wildtype
    wildtype_mask = df[GENOTYPE_COL] == 'b9d2_wildtype'
    wildtype_embryos = df[wildtype_mask][EMBRYO_ID_COL].unique().tolist()

    # Exclude any embryos that appear in phenotype lists
    all_phenotype_embryos = set()
    for phenotype_ids in phenotype_dict.values():
        all_phenotype_embryos.update(phenotype_ids)

    wildtype_ids = [eid for eid in wildtype_embryos if eid not in all_phenotype_embryos]

    print(f"\nWildtype embryos: {len(wildtype_ids)} (excluded {len(wildtype_embryos) - len(wildtype_ids)} phenotype embryos)")

    return wildtype_ids


# =============================================================================
# Label Application
# =============================================================================

def apply_cluster_categories(df: pd.DataFrame, phenotype_dict: Dict[str, List[str]], 
                             wildtype_ids: List[str]) -> pd.DataFrame:
    """
    Add cluster_categories column to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    phenotype_dict : Dict[str, List[str]]
        Dictionary mapping phenotype name to list of embryo IDs
    wildtype_ids : List[str]
        List of wildtype embryo IDs

    Returns
    -------
    df : pd.DataFrame
        Dataframe with cluster_categories column added
    """
    print("\nApplying cluster_categories labels...")

    # Create embryo_id to phenotype mapping
    embryo_to_phenotype = {}
    
    # Add phenotype embryos
    for phenotype, ids in phenotype_dict.items():
        for embryo_id in ids:
            embryo_to_phenotype[embryo_id] = phenotype
    
    # Add wildtype embryos
    for embryo_id in wildtype_ids:
        embryo_to_phenotype[embryo_id] = 'wildtype'
    
    # Apply labels
    df['cluster_categories'] = df[EMBRYO_ID_COL].map(embryo_to_phenotype)
    
    # Mark unlabeled embryos
    df['cluster_categories'] = df['cluster_categories'].fillna('unlabeled')
    
    # Print summary
    print("\nCluster categories distribution:")
    category_counts = df.groupby('cluster_categories')[EMBRYO_ID_COL].nunique().sort_values(ascending=False)
    for category, count in category_counts.items():
        print(f"  {category}: {count} embryos")
    
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("B9D2 Data Labeling and Export")
    print("="*80)
    
    # Load phenotype lists
    phenotype_dict = load_phenotype_lists()
    
    # Load experiment data
    df = load_experiment_data()
    
    # Extract wildtype embryos
    wildtype_ids = extract_wildtype_embryos(df, phenotype_dict)
    
    # Apply cluster_categories
    df = apply_cluster_categories(df, phenotype_dict, wildtype_ids)
    
    # Save to CSV
    print(f"\nSaving labeled data to: {OUTPUT_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"  Saved {len(df)} rows")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
