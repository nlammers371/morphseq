"""
Configuration for CEP290 Genotype-Blind Pair Clustering Analysis

This analysis runs bootstrap consensus clustering on each cep290 pair
across experiments to identify WT-like vs mutant-like clusters without
relying on genotype labels.
"""
from pathlib import Path

# Experiments to analyze
EXPERIMENT_IDS = ['20251106', '20251112', '20251113']

# Clustering parameters
K_VALUES = [3, 4, 5, 6]
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42

# Threshold for identifying WT-like clusters
WT_THRESHOLD = 0.05  # clusters with mean < 0.05 are WT-like

# Data columns
METRIC_COL = 'baseline_deviation_normalized'
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'

# Trajectory processing
MIN_TIMEPOINTS = 3
GRID_STEP = 0.5

# DTW parameters
DTW_WINDOW = 3

# Output settings
OUTPUT_DIR = Path(__file__).parent / 'output'
GENERATE_PNG = True
GENERATE_PLOTLY = True
VERBOSE = True
