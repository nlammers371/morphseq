"""
Configuration constants for phenotype emergence classification analysis.

This module centralizes all configuration parameters to make the analysis
reproducible and easy to modify.
"""

import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Root directory for this analysis
RESULTS_DIR = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251016"

# Data directories
DATA_DIR = os.path.join(RESULTS_DIR, "data")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")

# Morphseq repository root
MORPHSEQ_ROOT = "/net/trapnell/vol1/home/mdcolon/proj/morphseq"

# Build06 metadata directory
BUILD06_DIR = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output"

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Experiment IDs organized by genotype family
WT_EXPERIMENTS = ["20230615", "20230531", "20230525", "20250912"]
B9D2_EXPERIMENTS = ["20250519", "20250520"]
CEP290_EXPERIMENTS = ["20250305", "20250416", "20250512", "20250515_part2", "20250519"]
TMEM67_EXPERIMENTS = ["20250711"]

# All experiments combined
ALL_EXPERIMENTS = WT_EXPERIMENTS + B9D2_EXPERIMENTS + CEP290_EXPERIMENTS + TMEM67_EXPERIMENTS

# Genotype groupings for pairwise comparisons
GENOTYPE_GROUPS = {
    "cep290": ["cep290_wildtype", "cep290_heterozygous", "cep290_homozygous"],
    "b9d2": ["b9d2_wildtype", "b9d2_heterozygous", "b9d2_homozygous"],
    "tmem67": ["tmem67_wildtype", "tmem67_heterozygote", "tmem67_homozygous"],
}

# ============================================================================
# CLASSIFICATION ANALYSIS PARAMETERS
# ============================================================================

# Number of permutations for null distribution
# Higher = more accurate p-values but slower
# Quick test: 100, Standard: 500, Publication: 1000-5000
N_PERMUTATIONS = int(os.environ.get("MORPHSEQ_N_PERMUTATIONS", 100))

# Number of cross-validation folds for AUROC estimation
# - n_splits=3: Fast, less stable (66% train / 33% test each fold)
# - n_splits=5: Standard, good balance (80% train / 20% test)
# - n_splits=10: More stable, slower (90% train / 10% test)
N_CV_SPLITS = 5

# Random seed for reproducibility
RANDOM_SEED = 42

# Significance threshold for detecting onset
ALPHA = 0.05

# Use class weights to handle class imbalance
# This is the default and recommended setting
USE_CLASS_WEIGHTS = True

# ============================================================================
# BINNING PARAMETERS
# ============================================================================

# Time bin width in hours
TIME_BIN_WIDTH = 2.0

# Time column name in data
TIME_COLUMN = "predicted_stage_hpf"

# Latent column pattern for auto-detection
LATENT_COLUMN_PATTERN = "z_mu_b"

# ============================================================================
# PENETRANCE ANALYSIS PARAMETERS
# ============================================================================

# Confidence threshold for "confident" predictions
# Confidence is defined as |p - 0.5|
CONFIDENCE_THRESHOLD = 0.1

# Penetrance category bins (low, medium, high)
PENETRANCE_BINS = [0, 0.1, 0.2, 0.5]
PENETRANCE_LABELS = ["low", "medium", "high"]

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# Maximum number of embryos to show in trajectory plots
MAX_TRAJECTORY_EMBRYOS = 30

# DPI for saved figures
FIGURE_DPI = 300

# Figure format
FIGURE_FORMAT = "png"

# Color palettes
GENOTYPE_PALETTE = ["dodgerblue", "orangered", "mediumseagreen", "mediumpurple"]

# Heatmap colormap
HEATMAP_CMAP = "RdBu_r"

# Signed margin plot range
SIGNED_MARGIN_VMIN = -0.5
SIGNED_MARGIN_VMAX = 0.5

# ============================================================================
# COMPUTATIONAL PARAMETERS
# ============================================================================

# Number of parallel jobs for analysis (-1 = all cores)
N_JOBS = -1

# Minimum samples per class for analysis
MIN_SAMPLES_PER_CLASS = 5

# Minimum total samples for pairwise comparison
MIN_SAMPLES_TOTAL = 10

# ============================================================================
# LOGGING
# ============================================================================

# Verbosity level (0 = quiet, 1 = normal, 2 = verbose)
VERBOSITY = 1

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_data_dir(gene: str = None) -> str:
    """Get data directory path, optionally for a specific gene."""
    if gene:
        return os.path.join(DATA_DIR, gene)
    return DATA_DIR

def get_plot_dir(gene: str = None) -> str:
    """Get plot directory path, optionally for a specific gene."""
    if gene:
        return os.path.join(PLOT_DIR, gene)
    return PLOT_DIR

def make_dirs():
    """Create all necessary directories."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    for gene in GENOTYPE_GROUPS.keys():
        os.makedirs(get_data_dir(gene), exist_ok=True)
        os.makedirs(get_plot_dir(gene), exist_ok=True)

def print_config():
    """Print current configuration."""
    print("="*80)
    print("ANALYSIS CONFIGURATION")
    print("="*80)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Morphseq root: {MORPHSEQ_ROOT}")
    print(f"\nClassification Parameters:")
    print(f"  Permutations: {N_PERMUTATIONS}")
    print(f"  CV splits: {N_CV_SPLITS}")
    print(f"  Random seed: {RANDOM_SEED}")
    print(f"  Alpha: {ALPHA}")
    print(f"  Use class weights: {USE_CLASS_WEIGHTS}")
    print(f"\nBinning Parameters:")
    print(f"  Time bin width: {TIME_BIN_WIDTH} hours")
    print(f"  Time column: {TIME_COLUMN}")
    print(f"\nPenetrance Parameters:")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"\nVisualization Parameters:")
    print(f"  Max trajectory embryos: {MAX_TRAJECTORY_EMBRYOS}")
    print(f"  Figure DPI: {FIGURE_DPI}")
    print("="*80)
