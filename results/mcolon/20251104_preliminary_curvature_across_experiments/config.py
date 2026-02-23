"""
Configuration file for multi-experiment DTW clustering pipeline.

This script runs hierarchical consensus clustering across multiple experiments
and genotypes, organizing outputs by experiment/genotype/plot_type.
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

# Project root (parent of morphseq directory)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Results directory (where this script lives)
RESULTS_DIR = Path(__file__).resolve().parent

# Output directory for all results
OUTPUT_DIR = RESULTS_DIR / "output"

# Reference data directory (from previous analysis)
REFERENCE_ANALYSIS_DIR = RESULTS_DIR.parent / "20251029_curvature_temporal_analysis"

# Build06 output directory for experiment metadata
BUILD_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# Experiments to process
EXPERIMENTS = ['20251017_combined','20251017_part2', '20250305', '20250416', '20250711', '20251020', "20251017_part1", '20250501']

# Genotypes are determined dynamically from data for each experiment
# (e.g., 'cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous', 'cep290_unknown')

# ============================================================================
# DATA LOADING PARAMETERS
# ============================================================================

# Which metric to cluster on
# Options: 'arc_length_ratio', 'baseline_deviation_um', 'normalized_baseline_deviation'
METRIC_NAME = 'normalized_baseline_deviation'

# Minimum number of timepoints per trajectory for inclusion
MIN_TIMEPOINTS = 3

# ============================================================================
# DTW PARAMETERS
# ============================================================================

# Sakoe-Chiba band width constraint for DTW
DTW_WINDOW = 3

# Grid step for interpolating trajectories to common time points
GRID_STEP = 0.5

# ============================================================================
# CLUSTERING PARAMETERS
# ============================================================================

# K values to test
K_VALUES = [2, 3, 4, 5, 6, 7, 8]

# Number of bootstrap iterations
N_BOOTSTRAP = 100

# Fraction of data to use in each bootstrap sample
BOOTSTRAP_FRAC = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# MEMBERSHIP PARAMETERS
# ============================================================================

# Threshold for core membership (must be >= this to be "core")
CORE_THRESHOLD = 0.70

# Threshold for outlier classification (must be <= this to be "outlier")
OUTLIER_THRESHOLD = 0.4

# ============================================================================
# PLOTTING PARAMETERS
# ============================================================================

# Figure DPI
FIGURE_DPI = 100

# Alpha transparency for trajectories with membership coloring
TRAJECTORY_ALPHA = 0.8

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Whether to print verbose output
VERBOSE_OUTPUT = True
