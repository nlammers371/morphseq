"""
Configuration for WT quantile envelope penetrance pipeline.

Design decisions:
- Primary presentation outputs are embryo-level: one embryo summary per time bin.
- Embryo-level classification defaults to a hybrid rule:
  raw WT quantiles where support is clean, robust-smoothed thresholds otherwise.
- Frame-level calibration and scatter remain available as diagnostics only.
- Het embryos serve as an additional WT-like reference for calibration comparison.
"""

from pathlib import Path

# ============================================================================
# Data Paths
# ============================================================================

DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction/final_data")
EMBRYO_DATA_PATH = DATA_DIR / "embryo_data_with_labels.csv"

OUTPUT_DIR = Path(__file__).parent / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# ============================================================================
# Column Names
# ============================================================================

METRIC_NAME = "baseline_deviation_normalized"
TIME_COL = "predicted_stage_hpf"
EMBRYO_COL = "embryo_id"
GENOTYPE_COL = "genotype"
CATEGORY_COL = "cluster_categories"
SUBCATEGORY_COL = "cluster_subcategories"
PAIR_COL = "pair"

# ============================================================================
# Genotypes
# ============================================================================

WT_GENOTYPE = "cep290_wildtype"
HET_GENOTYPE = "cep290_heterozygous"
HOMO_GENOTYPE = "cep290_homozygous"

# ============================================================================
# Envelope Parameters
# ============================================================================

TIME_BIN_WIDTH = 2.0  # hpf

QUANTILE_LOW = 0.025
QUANTILE_HIGH = 0.975

# Candidate LOESS fracs swept from smallest to largest; smallest passing
# validity checks is selected independently for lower and upper curves.
LOESS_CANDIDATE_FRACS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Set to a float to force a specific frac for both curves (skips selection).
LOESS_FRAC_OVERRIDE = None

# Used if no candidate frac passes validity checks.
LOESS_FALLBACK_FRAC = 0.10

# Minimum WT frames in a bin for frame-level diagnostic envelopes.
MIN_WT_FRAMES_PER_BIN = 10

# Minimum WT embryo-bin summaries in a bin for embryo-level envelopes.
MIN_WT_EMBRYOS_PER_BIN = 10

# Metric is non-negative (baseline deviation); enforce lower >= 0 in envelope.
METRIC_NONNEG = True

# Embryo-bin summary statistic used for the main presentation path.
EMBRYO_BIN_AGG = "median"

# Robust smoothing excludes bins whose raw quantile deviates strongly from
# a local median before LOESS is fit.
ROBUST_SMOOTHING_WINDOW = 5
ROBUST_SMOOTHING_MIN_POINTS = 3
ROBUST_SMOOTHING_SIGMA_THRESHOLD = 4.0
ROBUST_SMOOTHING_MIN_RESID_FRACTION = 0.15

# Threshold source for calling penetrance.
# "raw"      -> empirical per-bin quantiles only
# "smoothed" -> smoothed envelope only
# "hybrid"   -> raw when support is clean, smoothed for excluded/unsupported bins
EMBRYO_CALL_MODE = "hybrid"
FRAME_DIAGNOSTIC_CALL_MODE = "smoothed"

# Presentation curve display controls. These affect plotted penetrance curves,
# not the saved penetrance table or threshold calls.
PRESENTATION_CURVE_MODE = "smoothed"   # "raw" or "smoothed"
PRESENTATION_CURVE_FRAC = 0.20
PRESENTATION_CURVE_SHOW_POINTS = True
PRESENTATION_CURVE_SMOOTH_SE = True

# One-directional penetrance: only flag frames that EXCEED the upper bound.
# Use True for deviation metrics (baseline_deviation_normalized) where "too low"
# is not a meaningful phenotype.  When True, the lower bound is ignored for
# marking penetrant frames and for plotting.
UPPER_BOUND_ONLY = True

# ============================================================================
# Category Definitions
# ============================================================================

BROAD_CATEGORIES = [
    "Not Penetrant",
    "Intermediate",
    "High_to_Low",
    "Low_to_High",
]

SUBCATEGORIES = [
    "Not Penetrant",
    "Intermediate",
    "High_to_Low_A",
    "High_to_Low_B",
    "Low_to_High_A",
    "Low_to_High_B",
]

# ============================================================================
# Color Schemes
# ============================================================================

CATEGORY_COLORS = {
    "Low_to_High": "#E74C3C",
    "High_to_Low": "#3498DB",
    "Intermediate": "#9B59B6",
    "Not Penetrant": "#2ECC71",
}

SUBCATEGORY_COLORS = {
    "Low_to_High_A": "#E74C3C",
    "Low_to_High_B": "#C0392B",
    "High_to_Low_A": "#3498DB",
    "High_to_Low_B": "#2980B9",
    "Intermediate": "#9B59B6",
    "Not Penetrant": "#2ECC71",
}

GENOTYPE_COLORS = {
    "cep290_wildtype": "#1f77b4",
    "cep290_heterozygous": "#ff7f0e",
    "cep290_homozygous": "#d62728",
}

# Spawn-only 24-120 hpf phenotype-story figures.
# The combined-transition green is an assumption until a user-specified hex
# is provided explicitly in chat.
STORY_TIME_MIN_HPF = 24.0
STORY_TIME_MAX_HPF = 120.0
SPAWN_PAIR_VALUE = "cep290_spawn"
SPAWN_PAIR_FALLBACK_VALUES = {"none", ""}
STORY_OUTPUT_SUBDIR = "spawn_24_120_transition_story"
STORY_CATEGORIES = ["Low_to_High", "High_to_Low"]
STORY_COMBINED_GROUP = "Transition_Combined"
STORY_COLORS = {
    "Low_to_High": CATEGORY_COLORS["Low_to_High"],
    "High_to_Low": CATEGORY_COLORS["High_to_Low"],
    "Transition_Combined": "#7FC97F",
}

# ============================================================================
# Plotting Parameters
# ============================================================================

KEY_STAGES_HPF = [24, 48, 72, 96]

FIGSIZE_CURVES = (12, 8)
FIGSIZE_HEATMAP = (14, 6)
FIGSIZE_DIAGNOSTIC = (14, 8)
FIGSIZE_BARS = (10, 6)
FIGSIZE_STORY_CURVES = (12, 8)

DPI = 150

STORY_LABEL_FONTSIZE = 20
STORY_TICK_FONTSIZE = 16
STORY_TITLE_FONTSIZE = 22
STORY_LEGEND_FONTSIZE = 16
