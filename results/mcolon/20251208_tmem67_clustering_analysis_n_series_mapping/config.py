"""
Configuration for tmem67 clustering analysis.
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "output"

# Data directories
CURV_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "body_axis" / "summary"
META_DIR = PROJECT_ROOT / "morphseq_playground" / "metadata" / "build06_output"

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================
EXPERIMENT_ID = '20250711'

# ============================================================================
# CLUSTERING PARAMETERS
# ============================================================================
K_VALUES = [3, 4, 5, 6]
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42

# ============================================================================
# TRAJECTORY PARAMETERS
# ============================================================================
METRIC_NAME = 'normalized_baseline_deviation'
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
MIN_TIMEPOINTS = 3
GRID_STEP = 0.5  # HPF

# ============================================================================
# DTW PARAMETERS
# ============================================================================
DTW_WINDOW = 3  # Sakoe-Chiba window

# ============================================================================
# CLASSIFICATION THRESHOLDS
# ============================================================================
THRESHOLD_MAX_P = 0.8           # Core membership confidence
THRESHOLD_LOG_ODDS_GAP = 0.7    # Core membership disambiguation
THRESHOLD_OUTLIER_MAX_P = 0.5   # Outlier threshold

# ============================================================================
# MUTANT IDENTIFICATION
# ============================================================================
MUTANT_THRESHOLD = 0.05  # Mean trajectory value threshold for mutant clusters

# ============================================================================
# PLOTTING PARAMETERS
# ============================================================================
GENERATE_PNG = True
GENERATE_PLOTLY = True
PLOTLY_ALPHA_INDIVIDUAL = 0.2
PLOTLY_WIDTH_INDIVIDUAL = 0.5
PLOTLY_WIDTH_MEAN = 3

# Color scheme for clusters (will cycle through for k>6)
CLUSTER_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
]

# ============================================================================
# VERBOSE OUTPUT
# ============================================================================
VERBOSE = True
