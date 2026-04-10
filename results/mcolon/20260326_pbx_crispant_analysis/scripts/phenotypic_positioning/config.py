from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_IDS = ["20260304", "20260306"]
BUILD06_DIR = REPO_ROOT / "morphseq_playground" / "metadata" / "build06_output"

RESULTS_BASE = (
    REPO_ROOT
    / "results"
    / "mcolon"
    / "20260326_pbx_crispant_analysis"
    / "results"
    / "misclassification"
    / "embedding"
)
FIGURES_BASE = (
    REPO_ROOT
    / "results"
    / "mcolon"
    / "20260326_pbx_crispant_analysis"
    / "figures"
    / "misclassification"
    / "embedding"
)

DEFAULT_RESULTS_SUBDIR = "phenotypic_positioning_phase2"
DEFAULT_MULTICLASS_RESULTS_SUBDIR = "phenotypic_positioning_multiclass"

DEFAULT_GENOTYPES = [
    "inj_ctrl",
    "wik_ab",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]

DEFAULT_BIN_WIDTH = 2.0
DEFAULT_TIME_COL = "predicted_stage_hpf"
DEFAULT_EMBEDDING_PREFIX = "z_mu_b"
DEFAULT_N_SPLITS = 5
DEFAULT_N_BOOTSTRAPS = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_K_NEIGHBORS = 10
DEFAULT_MULTICLASS_N_PERMUTATIONS = 100
DEFAULT_MULTICLASS_MIN_SAMPLES_PER_GROUP = 3
DEFAULT_MULTICLASS_N_JOBS = -1
DEFAULT_SNAPSHOT_TIMES = [25.0, 55.0, 79.0]
