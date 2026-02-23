"""
Configuration for curvature-embedding prediction analysis.

Modify these settings to adjust:
- Which data file to load
- Which curvature metrics to predict
- Which models to test
- Analysis hyperparameters
"""

from pathlib import Path

# ============================================================================
# Data Paths
# ============================================================================

# Primary data source: build06 embeddings + metadata
DATA_FILE = Path(
    '/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/'
    'df03_final_output_with_latents_20251106.csv'
)

# Output directories (relative to this script)
RESULTS_DIR = Path(__file__).parent
FIGURE_DIR = RESULTS_DIR / 'outputs' / 'figures'
TABLE_DIR = RESULTS_DIR / 'outputs' / 'tables'


# ============================================================================
# Regression Targets: Curvature Metrics
# ============================================================================

# Primary metrics: can be extended with others like 'total_length_um', 'mean_curvature_per_um'
CURVATURE_METRICS = [
    'arc_length_ratio',              # Ratio of arc length to baseline
    'normalized_baseline_deviation'  # Curvature normalized by baseline
]

# Additional metrics to consider (uncomment to add):
# CURVATURE_METRICS += ['total_length_um', 'mean_curvature_per_um', 'baseline_deviation_um']


# ============================================================================
# Features: Embeddings
# ============================================================================

# Pattern to identify embedding columns in the data
EMBEDDING_PATTERN = 'z_mu_b_'  # Matches z_mu_b_0, z_mu_b_1, etc.

# Alternative patterns:
# EMBEDDING_PATTERN = 'latent_'  # For other naming schemes


# ============================================================================
# Sample Filtering
# ============================================================================

# Genotypes to analyze
GENOTYPES = [
    'cep290_wildtype',
    'cep290_heterozygous',
    'cep290_homozygous'
]

# Minimum samples per embryo (for LOEO validation)
MIN_SAMPLES_PER_EMBRYO = 1


# ============================================================================
# Model Configuration
# ============================================================================

# Models to test: {model_type: hyperparameters}
MODELS_TO_TEST = {
    'ridge': {
        'alpha': 1.0,  # Regularization strength (higher = more regularization)
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
    },
}

# Optional: Add more models
# MODELS_TO_TEST['random_forest'] = {
#     'n_estimators': 100,
#     'max_depth': None,
# }


# ============================================================================
# Cross-Validation Strategy
# ============================================================================

# Leave-one-embryo-out (LOEO): each embryo is held out once for testing
# All samples from one embryo are together in train OR test set (no leakage)
CV_STRATEGY = 'loeo'  # Currently only option

# Whether to scale features before fitting
SCALE_FEATURES = True


# ============================================================================
# Analysis Parameters
# ============================================================================

# Number of top features to display in plots
TOP_N_FEATURES = 15

# Random seeds for reproducibility
RANDOM_SEED = 42

# Verbosity
VERBOSE = True
