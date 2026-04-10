from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
EXPERIMENT_IDS = ["20260304", "20260306"]
BUILD04_DIR = REPO_ROOT / "morphseq_playground" / "metadata" / "build04_output"
SNIP_ROOT = REPO_ROOT / "morphseq_playground" / "training_data" / "bf_embryo_snips"

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

DEFAULT_RESULTS_SUBDIR = "embryo_attrition_qc_audit"
DEFAULT_EXAMPLE_SUBDIR = "qc_flag_examples"
DEFAULT_BIN_WIDTH = 4.0
DEFAULT_TIME_COL = "predicted_stage_hpf"
DEFAULT_TARGET_FLAGS = ["no_yolk_flag", "frame_flag"]
DEFAULT_EXAMPLES_PER_GENOTYPE = 2
DEFAULT_VIDEO_WINDOW_HPF = 8.0
DEFAULT_VIDEO_FPS = 12
DEFAULT_VIDEO_FRAMES = 96
DEFAULT_RAW_TO_ANALYSIS_GENOTYPE = {
    "wik-ab_inj_ctrl": "inj_ctrl",
    "wik_ab_inj_ctrl": "inj_ctrl",
    "ab_inj_ctrl": "inj_ctrl",
    "wik-ab": "wik_ab",
    "wik_ab": "wik_ab",
    "pbx1b_crispant": "pbx1b_crispant",
    "pbx4_crispant": "pbx4_crispant",
    "pbx1b_pbx4_crispant": "pbx1b_pbx4_crispant",
}
DEFAULT_GENOTYPES = [
    "inj_ctrl",
    "wik_ab",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]
CANONICAL_QC_FLAGS = [
    "sa_outlier_flag",
    "sam2_qc_flag",
    "frame_flag",
    "no_yolk_flag",
]
INFO_QC_FLAGS = [
    "focus_flag",
    "bubble_flag",
]
