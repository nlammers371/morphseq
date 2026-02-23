"""
Configuration constants for trajectory-specific penetrance analysis.
"""

from pathlib import Path

# ============================================================================
# Data Paths
# ============================================================================

DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction/final_data")
EMBRYO_DATA_PATH = DATA_DIR / "embryo_data_with_labels.csv"
CLUSTER_LABELS_PATH = DATA_DIR / "embryo_cluster_labels.csv"

# Output directories
OUTPUT_DIR = Path(__file__).parent / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# ============================================================================
# Analysis Parameters
# ============================================================================

# Column names
METRIC_NAME = "baseline_deviation_normalized"
TIME_COL = "predicted_stage_hpf"
EMBRYO_COL = "embryo_id"
GENOTYPE_COL = "genotype"
CATEGORY_COL = "cluster_categories"        # Phase A: 4 broad groups
SUBCATEGORY_COL = "cluster_subcategories"  # Phase B: 6 specific groups

# Threshold parameters
IQR_K = 2.0  # IQR multiplier (~95% coverage for normal distribution)
EARLY_CUTOFF_HPF = 30.0  # Switch from time-binned to global IQR
TIME_BIN_WIDTH = 2.0  # hpf

# WT genotype for baseline
WT_GENOTYPE = "cep290_wildtype"

# Time range
MIN_TIME_HPF = 12.0
MAX_TIME_HPF = 140.0

# ============================================================================
# Cluster Definitions
# ============================================================================

# Broad categories (4 groups) - Phase A
BROAD_CATEGORIES = [
    "Not Penetrant",
    "Intermediate",
    "High_to_Low",
    "Low_to_High",
]

# Subcategories (6 groups) - Phase B
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

# Broad categories (4 colors)
CATEGORY_COLORS = {
    "Low_to_High": "#E74C3C",   # Red (progressive worsening)
    "High_to_Low": "#3498DB",   # Blue (potential recovery)
    "Intermediate": "#9B59B6",  # Purple (unstable, large spread)
    "Not Penetrant": "#2ECC71", # Green (WT-like)
}

# Subcategories (6 colors)
SUBCATEGORY_COLORS = {
    "Low_to_High_A": "#E74C3C",   # Red
    "Low_to_High_B": "#C0392B",   # Dark red
    "High_to_Low_A": "#3498DB",   # Blue
    "High_to_Low_B": "#2980B9",   # Dark blue
    "Intermediate": "#9B59B6",    # Purple
    "Not Penetrant": "#2ECC71",   # Green
}

# Genotype colors (for composition plots)
GENOTYPE_COLORS = {
    "cep290_wildtype": "#1f77b4",      # Blue
    "cep290_heterozygous": "#ff7f0e",  # Orange
    "cep290_homozygous": "#d62728",    # Red
}

# ============================================================================
# Plotting Parameters
# ============================================================================

# Key developmental stages for bar chart comparisons
KEY_STAGES_HPF = [24, 48, 72, 96]

# Figure sizes
FIGSIZE_CURVES = (12, 8)
FIGSIZE_HEATMAP = (14, 6)
FIGSIZE_DIAGNOSTIC = (14, 8)
FIGSIZE_BARS = (10, 6)

# DPI for saved figures
DPI = 300
