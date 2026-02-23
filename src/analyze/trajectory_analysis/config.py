"""
Configuration Module

Centralized default parameters for trajectory analysis pipeline.

All default values are defined here for consistency across modules.
Users can override these in function calls.
"""

from analyze.viz.styling import ColorLookup

# ==============================================================================
# Bootstrap Parameters
# ==============================================================================

N_BOOTSTRAP = 100
"""Number of bootstrap iterations for consensus clustering"""

BOOTSTRAP_FRAC = 0.8
"""Fraction of samples to include in each bootstrap iteration (80%)"""

RANDOM_SEED = 42
"""Random seed for reproducibility"""


# ==============================================================================
# DTW Parameters
# ==============================================================================

DTW_WINDOW = 5
"""Sakoe-Chiba band constraint for DTW alignment (max warping distance)"""

GRID_STEP = 0.5
"""Time step for common grid interpolation (in HPF units)"""


# ==============================================================================
# Data Processing
# ==============================================================================

MIN_TIMEPOINTS = 3
"""Minimum number of timepoints required for a valid trajectory"""

DEFAULT_EMBRYO_ID_COL = 'embryo_id'
"""Default column name for embryo identifiers"""

DEFAULT_METRIC_COL = 'normalized_baseline_deviation'
"""Default column name for metric values"""

DEFAULT_TIME_COL = 'predicted_stage_hpf'
"""Default column name for time values"""

DEFAULT_GENOTYPE_COL = 'genotype'
"""Default column name for genotype labels"""


# ==============================================================================
# Classification Thresholds
# ==============================================================================

THRESHOLD_MAX_P = 0.8
"""Minimum max probability for core membership (80% confidence)"""

THRESHOLD_LOG_ODDS_GAP = 0.7
"""Minimum log-odds gap between top clusters for core membership"""

THRESHOLD_OUTLIER_MAX_P = 0.5
"""Maximum max probability before being classified as outlier (50%)"""

ADAPTIVE_PERCENTILE = 0.75
"""Percentile for adaptive per-cluster thresholds (75th percentile)"""


# ==============================================================================
# Two-Stage Outlier Filtering
# ==============================================================================

ENABLE_IQR_FILTERING = True
"""Enable two-stage IQR filtering in consensus pipeline"""

IQR_MULTIPLIER = 2
"""IQR multiplier for outlier detection (2x IQR; less conservative than prior 4x default)"""

KNN_K = 5
"""Number of nearest neighbors for k-NN distance calculation in Stage 1 filtering"""

POSTERIOR_OUTLIER_THRESHOLD = 0.5
"""Minimum max_p for Stage 2 filtering (embryos below this are removed)"""


# ==============================================================================
# Plotting Parameters
# ==============================================================================

DEFAULT_DPI = 120
"""Default resolution for saved figures"""

DEFAULT_FIGSIZE = (12, 6)
"""Default figure size (width, height) in inches"""

MEMBERSHIP_COLORS = {
    'core': '#2ecc71',       # Green
    'uncertain': '#f1c40f',  # Yellow
    'outlier': '#e74c3c'     # Red
}
"""Standard colors for membership categories"""


# ==============================================================================
# File Paths (Optional - can be overridden)
# ==============================================================================

# These are examples - actual paths should be provided by users
DEFAULT_CURV_DIR = '/net/trapnell/vol1/home/nlammers/projects/data/morphseq/built_image_data/Keyence'
DEFAULT_META_DIR = '/net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/built_metadata_files'


# ==============================================================================
# Genotype Styling (from plot_config.py)
# ==============================================================================

# Color mapping based on genotype suffix (independent of gene prefix)
GENOTYPE_SUFFIX_COLORS = {
    'wildtype': '#2E7D32',      # Green
    'heterozygous': '#FFA500',  # Orange
    'homozygous': '#D32F2F',    # Red
    'crispant': '#9467bd',      # Purple
    'unknown': '#808080',       # Gray
}
"""Standard colors for genotype suffixes. Works with any gene prefix."""

# Standard ordering for genotype suffixes
GENOTYPE_SUFFIX_ORDER = ['crispant', 'homozygous', 'heterozygous', 'wildtype', 'unknown']
"""Default order for displaying genotype suffixes."""

# Pre-configured color lookup object for genotype suffixes
GENOTYPE_COLORS = ColorLookup(
    suffix_colors=GENOTYPE_SUFFIX_COLORS,
    suffix_order=GENOTYPE_SUFFIX_ORDER
)
"""ColorLookup object that can be passed directly to color_lookup parameters."""


# ==============================================================================
# Phenotype Styling (from plot_config.py)
# ==============================================================================

# Color mapping for phenotype categories (muted/earthy tones - distinct from genotype)
PHENOTYPE_COLORS = {
    'CE': '#5B7C99',           # Slate blue
    'HTA': '#7FA87F',          # Sage green
    'BA_rescue': '#C4956A',    # Terracotta/tan
    'non_penetrant': '#9E9E9E', # Warm gray
}
"""Colors for phenotype categories. Muted/earthy tones, distinct from genotype suffix colors."""

# Standard ordering for phenotypes
PHENOTYPE_ORDER = ['CE', 'HTA', 'BA_rescue', 'non_penetrant']
"""Default order for displaying phenotypes."""


# ==============================================================================
# Matplotlib Styling (from plot_config.py)
# ==============================================================================

# Line and trace styling
INDIVIDUAL_TRACE_ALPHA = 0.2        # Faded individual trajectories
INDIVIDUAL_TRACE_LINEWIDTH = 0.8    # Thin individual lines
MEAN_TRACE_LINEWIDTH = 2.2          # Bold mean trajectory line
OVERLAY_ALPHA = 0.25                # Faded when overlaying multiple groups
FACETED_ALPHA = 0.25                # Faded when faceting

# Font sizes
TITLE_FONTSIZE = 14
SUBPLOT_TITLE_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 9

# Grid and styling
GRID_ALPHA = 0.3
GRID_LINESTYLE = '--'
GRID_LINEWIDTH = 0.5


# ==============================================================================
# Plotly Styling (from plot_config.py)
# ==============================================================================

# Figure defaults
DEFAULT_PLOTLY_HEIGHT = 500
DEFAULT_PLOTLY_WIDTH = 1400
HEIGHT_PER_ROW = 350              # Height multiplier for row count
WIDTH_PER_COL = 400               # Width multiplier for column count

# Hover template defaults
HOVER_TEMPLATE_BASE = (
    '<b>Embryo:</b> %{customdata[0]}<br>'
    '<b>Time:</b> %{x:.2f} hpf<br>'
    '<b>Value:</b> %{y:.4f}<br>'
    '<extra></extra>'
)


# ==============================================================================
# Faceted Plot Sizing (from plot_config.py)
# ==============================================================================

# Default sizing for dynamic faceted plots
MIN_FIGSIZE_WIDTH = 6
MIN_FIGSIZE_HEIGHT = 4
DEFAULT_FIGSIZE_WIDTH_PER_COL = 5
DEFAULT_FIGSIZE_HEIGHT_PER_ROW = 4.5
