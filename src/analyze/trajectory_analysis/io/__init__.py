"""I/O Functions for Trajectory Analysis

This subpackage contains data loading and file I/O functions.

Modules
=======
data_loading : Load experiment data and compute DTW distances
phenotype_io : Load and save phenotype files
"""

# Data loading functions
from .data_loading import (
    load_experiment_dataframe,
    extract_trajectory_dataframe,
    dataframe_to_trajectories,
    interpolate_trajectories,
    compute_dtw_distance_from_df,
)

# Phenotype I/O
from .phenotype_io import (
    load_phenotype_file,
    save_phenotype_file,
)

__all__ = [
    # Data loading
    'load_experiment_dataframe',
    'extract_trajectory_dataframe',
    'dataframe_to_trajectories',
    'interpolate_trajectories',
    'compute_dtw_distance_from_df',
    # Phenotype I/O
    'load_phenotype_file',
    'save_phenotype_file',
]
