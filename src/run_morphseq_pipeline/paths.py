"""
Centralized path templates and helpers (single source of truth).

This module provides organized, typed helpers for constructing per‑experiment
and per‑model file/directory paths used by the pipeline orchestrators and
wrappers. Keep this minimal and deterministic — no filesystem I/O or
fallback logic here. Callers are responsible for existence checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union


# ============================================================================
# Path templates (format with {root}, {exp}, {model})
# ============================================================================

# Non-generated inputs
STAGE_REF_CSV = "{root}/metadata/stage_ref_df.csv"
PERTURBATION_NAME_KEY_CSV = "{root}/metadata/perturbation_name_key.csv"
WELL_METADATA_XLSX = "{root}/metadata/well_metadata/{exp}_well_metadata.xlsx" #or /metadata/plate_metadata/...

# Build01: Raw → Stitched FF images + metadata
KEYENCE_FF_IMAGES_DIR = "{root}/built_image_data/Keyence/FF_images/{exp}"
STITCHED_FF_IMAGES_DIR = "{root}/built_image_data/stitched_FF_images/{exp}"
BUILT_METADATA_CSV = "{root}/metadata/built_metadata_files/{exp}_metadata.csv"

# SAM2: Segmentation pipeline (authoritative per‑experiment locations)
EXPERIMENT_METADATA_JSON = "{root}/sam2_pipeline_files/raw_data_organized/experiment_metadata_{exp}.json"
GDINO_DETECTIONS_JSON = "{root}/sam2_pipeline_files/detections/gdino_detections_{exp}.json"
SAM2_SEGMENTATIONS_JSON = "{root}/sam2_pipeline_files/segmentation/grounded_sam_segmentations_{exp}.json"
SAM2_MASKS_DIR = "{root}/sam2_pipeline_files/exported_masks/{exp}/masks"
SAM2_MASK_EXPORT_MANIFEST = "{root}/sam2_pipeline_files/exported_masks/{exp}/mask_export_manifest_{exp}.json"
SAM2_METADATA_CSV = "{root}/sam2_pipeline_files/sam2_expr_files/sam2_metadata_{exp}.csv"


# Build03: Snip extraction & metadata
BUILD03_OUTPUT_CSV = "{root}/metadata/build03_output/expr_embryo_metadata_{exp}.csv"
BUILD03_SNIPS_DIR = "{root}/training_data/bf_embryo_snips/{exp}"

# Build04: QC & staging
BUILD04_OUTPUT_CSV = "{root}/metadata/build04_output/qc_staged_{exp}.csv"

# Build06: Embeddings & per‑experiment df03
LATENTS_CSV = "{root}/analysis/latent_embeddings/legacy/{model}/morph_latents_{exp}.csv"
BUILD06_OUTPUT_CSV = (
    "{root}/metadata/build06_output/df03_final_output_with_latents_{exp}.csv"
)


# ============================================================================
# Helper utilities
# ============================================================================

_PathLike = Union[str, Path]


def _fmt(template: str, **kwargs) -> Path:
    """Format a template with kwargs and return as a Path object."""
    return Path(template.format(**kwargs))


# Non-generated inputs
def get_stage_ref_csv(root: _PathLike) -> Path:
    return _fmt(STAGE_REF_CSV, root=root)


def get_perturbation_key_csv(root: _PathLike) -> Path:
    """Path to the global perturbation name key CSV."""
    return _fmt(PERTURBATION_NAME_KEY_CSV, root=root)


def get_well_metadata_xlsx(root: _PathLike, exp: str) -> Path:
    """Path to the per‑experiment well metadata Excel (non‑generated input)."""
    return _fmt(WELL_METADATA_XLSX, root=root, exp=exp)


# Build01
def get_keyence_ff_dir(root: _PathLike, exp: str) -> Path:
    """Path to Keyence raw FF images directory."""
    return _fmt(KEYENCE_FF_IMAGES_DIR, root=root, exp=exp)


def get_stitched_ff_dir(root: _PathLike, exp: str) -> Path:
    """Path to stitched FF images directory (SAM2 input for both microscope types)."""
    return _fmt(STITCHED_FF_IMAGES_DIR, root=root, exp=exp)


def get_built_metadata_csv(root: _PathLike, exp: str) -> Path:
    return _fmt(BUILT_METADATA_CSV, root=root, exp=exp)


# SAM2
def get_sam2_masks_dir(root: _PathLike, exp: str) -> Path:
    return _fmt(SAM2_MASKS_DIR, root=root, exp=exp)


def get_sam2_csv(root: _PathLike, exp: str) -> Path:
    """Single, authoritative path for the per‑experiment SAM2 metadata CSV."""
    return _fmt(SAM2_METADATA_CSV, root=root, exp=exp)


def get_gdino_detections_json(root: _PathLike, exp: str) -> Path:
    return _fmt(GDINO_DETECTIONS_JSON, root=root, exp=exp)


def get_sam2_segmentations_json(root: _PathLike, exp: str) -> Path:
    return _fmt(SAM2_SEGMENTATIONS_JSON, root=root, exp=exp)


def get_sam2_mask_export_manifest(root: _PathLike, exp: str) -> Path:
    return _fmt(SAM2_MASK_EXPORT_MANIFEST, root=root, exp=exp)


def get_experiment_metadata_json(root: _PathLike, exp: str) -> Path:
    """Per‑experiment metadata JSON emitted by SAM2 prep (if configured)."""
    return _fmt(EXPERIMENT_METADATA_JSON, root=root, exp=exp)


# Build03
def get_build03_output(root: _PathLike, exp: str) -> Path:
    return _fmt(BUILD03_OUTPUT_CSV, root=root, exp=exp)


def get_snips_dir(root: _PathLike, exp: str) -> Path:
    """Path to Build03 training snips directory for an experiment."""
    return _fmt(BUILD03_SNIPS_DIR, root=root, exp=exp)


# Build04
def get_build04_output(root: _PathLike, exp: str) -> Path:
    return _fmt(BUILD04_OUTPUT_CSV, root=root, exp=exp)


# Build06
def get_latents_csv(root: _PathLike, model: str, exp: str) -> Path:
    return _fmt(LATENTS_CSV, root=root, model=model, exp=exp)


def get_build06_output(root: _PathLike, exp: str) -> Path:
    return _fmt(BUILD06_OUTPUT_CSV, root=root, exp=exp)
