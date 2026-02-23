"""
Configuration file for DTW clustering pipeline.

Centralizes all paths, data parameters, clustering parameters, and other
settings used across the entire analysis pipeline.
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

# ============================================================================
# DATA LOADING PARAMETERS
# ============================================================================

# Which genotype to analyze
GENOTYPE_FILTER = 'cep290_homozygous'

# Which metric to cluster on
# Options: 'arc_length_ratio', 'baseline_deviation_um', 'normalized_baseline_deviation'
METRIC_NAME = 'normalized_baseline_deviation'

# Minimum number of timepoints per trajectory for inclusion
MIN_TIMEPOINTS = 3

# Whether to filter to test embryos (useful for debugging)
TEST_MODE = False
TEST_EMBRYOS = []  # Set to list of embryo_ids if TEST_MODE=True

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

# Prior/expected k value for methods that need it
PRIOR_K = 3

# Number of bootstrap iterations
N_BOOTSTRAP = 100

# Fraction of data to use in each bootstrap sample
BOOTSTRAP_FRAC = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# K-SELECTION PARAMETERS
# ============================================================================

# Methods to use for k-selection
K_SELECTION_METHODS = ['elbow', 'gap_statistic', 'eigengap', 'consensus']

# ============================================================================
# MEMBERSHIP PARAMETERS
# ============================================================================

# Threshold for core membership (must be >= this to be "core")
CORE_THRESHOLD = 0.70

# Threshold for outlier classification (must be <= this to be "outlier")
OUTLIER_THRESHOLD = 0.4

# ============================================================================
# MODEL FITTING PARAMETERS
# ============================================================================

# Degree of spline for trajectory fitting
SPLINE_DEGREE = 3

# Smoothing factor for spline
SPLINE_SMOOTH = None  # None = automatic via cross-validation

# Whether to use DBA for cluster centroids (True) or simple mean (False)
USE_DBA = True

# Maximum iterations for DBA algorithm
DBA_MAX_ITER = 10

# Gaussian smoothing sigma for DBA (0 = no smoothing)
DBA_SMOOTH_SIGMA = 0.0

# ============================================================================
# PLOTTING PARAMETERS
# ============================================================================

# Figure DPI
FIGURE_DPI = 100

# Figure size defaults (width, height) in inches
FIGURE_SIZE_DEFAULT = (10, 6)
FIGURE_SIZE_TALL = (10, 12)
FIGURE_SIZE_WIDE = (16, 8)

# Color scheme for clusters
CLUSTER_COLORS = None  # None = use matplotlib default, else provide list of hex colors

# Alpha transparency for overlaid trajectories
TRAJECTORY_ALPHA = 0.3

# Line width for cluster fits
FIT_LINE_WIDTH = 2.5

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Whether to save all intermediate results (verbose output)
VERBOSE_OUTPUT = True

# Whether to generate comparison report
GENERATE_COMPARISON_REPORT = False

# ============================================================================
# VALIDATION PARAMETERS
# ============================================================================

# Whether to run validation on holdout set (Step 7)
RUN_VALIDATION = False

# Fraction of data to hold out for validation
VALIDATION_FRAC = 0.2

# ============================================================================
# FUNCTIONAL DATA ANALYSIS PARAMETERS
# ============================================================================

# Whether to run functional PCA (Step 8)
RUN_FUNCTIONAL_PCA = False

# Number of FPC components to compute
N_FPC_COMPONENTS = 5
