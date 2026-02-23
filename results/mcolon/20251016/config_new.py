"""
Simplified configuration - just paths and experiment lists.
Everything else uses function defaults.
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

# That's it! Everything else is function defaults.
