# Parallel Execution Plan for Streamline-Snakemake Refactor

**Date:** 2025-11-06
**Purpose:** Maximize parallelism using multiple agents working simultaneously

---

## Execution Strategy

Work is organized into **waves** where all tasks in a wave can run **completely in parallel** with no dependencies on each other. Each subsequent wave depends on the previous wave completing.

---

## Wave 1: Foundation & Infrastructure (ALL PARALLEL)

**Duration:** ~2-4 hours
**Dependencies:** None - all can start immediately

### Agent 1: Schema Definitions
**Task:** Create all schema files with REQUIRED_COLUMNS_* lists

**Files to create:**
```
src/data_pipeline/schemas/
├── __init__.py
├── plate_metadata.py
├── scope_metadata.py
├── scope_and_plate_metadata.py
├── segmentation.py
├── snip_processing.py
├── features.py
├── quality_control.py
├── analysis_ready.py
├── channel_normalization.py
└── image_manifest.py
```

**Deliverable:** All schema files with complete REQUIRED_COLUMNS_* definitions

---

### Agent 2: Directory Structure & Validation Helpers
**Task:** Create directory structure and shared validation utilities

**Directories to create:**
```
src/data_pipeline/
├── schemas/                   # (Agent 1 populates this)
├── metadata_ingest/
│   ├── plate/
│   ├── scope/
│   ├── mapping/
│   └── manifests/
├── image_building/
│   ├── keyence/
│   ├── yx1/
│   └── shared/
├── segmentation/
│   ├── grounded_sam2/
│   ├── unet/
│   └── mask_utilities.py
├── snip_processing/
├── feature_extraction/
├── quality_control/
│   ├── segmentation_qc/
│   ├── auxiliary_mask_qc/
│   ├── morphology_qc/
│   └── consolidation/
├── embeddings/
├── analysis_ready/
├── identifiers/
├── io/
└── config/
```

**Files to create:**
```python
# src/data_pipeline/io/validation.py
def validate_dataframe_schema(df, required_columns, stage_name):
    """
    Validate DataFrame against schema.

    Checks:
    1. All required columns exist
    2. No required columns contain null values

    Raises ValueError with clear message if validation fails.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {stage_name}: {sorted(missing)}")

    for col in required_columns:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null values in {stage_name}")

    return True
```

**Deliverable:** Full directory structure + validation helper

---

### Agent 3: Development & Testing Scripts
**Task:** Create helper scripts for rapid iteration and validation

**Scripts to create:**

**1. `scripts/rapid_test.sh`**
```bash
#!/bin/bash
# Quick pipeline test on subset data

EXPERIMENT=${1:-test_yx1_001}
TARGET_RULE=${2:-all}

echo "Testing: $EXPERIMENT → $TARGET_RULE"

snakemake \
    --config experiments=$EXPERIMENT \
    --cores 2 \
    --printshellcmds \
    $TARGET_RULE

# Auto-validate if hitting checkpoints
case $TARGET_RULE in
    "rule_generate_image_manifest")
        python scripts/validate_phase2_outputs.py $EXPERIMENT
        ;;
    "all")
        python scripts/validate_full_pipeline.py $EXPERIMENT
        ;;
esac
```

**2. `scripts/validate_phase2_outputs.py`**
```python
#!/usr/bin/env python3
"""Validate Phase 1-2 outputs (metadata + image manifest)."""

import sys
import pandas as pd
from pathlib import Path

def validate_phase2(exp_id, microscope=None):
    """Validate metadata and manifest for experiment."""

    # Import schemas
    from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
    from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA
    from data_pipeline.schemas.scope_and_plate_metadata import REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA
    from data_pipeline.io.validation import validate_dataframe_schema

    metadata_dir = Path(f"experiment_metadata/{exp_id}")

    # Validate each metadata file
    plate = pd.read_csv(metadata_dir / "plate_metadata.csv")
    validate_dataframe_schema(plate, REQUIRED_COLUMNS_PLATE_METADATA, "plate_metadata")
    print(f"✓ plate_metadata.csv: {len(plate)} rows")

    scope = pd.read_csv(metadata_dir / "scope_metadata.csv")
    validate_dataframe_schema(scope, REQUIRED_COLUMNS_SCOPE_METADATA, "scope_metadata")
    print(f"✓ scope_metadata.csv: {len(scope)} rows")

    aligned = pd.read_csv(metadata_dir / "scope_and_plate_metadata.csv")
    validate_dataframe_schema(aligned, REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA, "scope_and_plate")
    print(f"✓ scope_and_plate_metadata.csv: {len(aligned)} rows")

    # Validate manifest exists
    manifest_path = metadata_dir / "experiment_image_manifest.json"
    assert manifest_path.exists(), f"Missing: {manifest_path}"
    print(f"✓ experiment_image_manifest.json exists")

    # Check channel normalization
    assert "BF" in aligned['channel_name'].values, "BF channel missing (normalization failed)"
    print(f"✓ Channel normalization validated")

    print(f"\n✓✓✓ Phase 2 validation PASSED for {exp_id} ✓✓✓")

if __name__ == "__main__":
    validate_phase2(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
```

**3. `scripts/validate_full_pipeline.py`**
```python
#!/usr/bin/env python3
"""Validate end-to-end pipeline outputs."""

import sys
import pandas as pd
from pathlib import Path

def validate_full_pipeline(exp_id):
    """Validate all phases completed successfully."""

    phases = [
        ("Metadata", "experiment_metadata/{exp}/scope_and_plate_metadata.csv"),
        ("Segmentation", "segmentation/{exp}/segmentation_tracking.csv"),
        ("Snips", "processed_snips/{exp}/snip_manifest.csv"),
        ("Features", "computed_features/{exp}/consolidated_snip_features.csv"),
        ("QC", "quality_control/{exp}/consolidated_qc_flags.csv"),
        ("Analysis", "analysis_ready/{exp}/features_qc_embeddings.csv"),
    ]

    for phase_name, path_template in phases:
        path = Path(path_template.format(exp=exp_id))
        assert path.exists(), f"{phase_name} output missing: {path}"

        df = pd.read_csv(path)
        assert len(df) > 0, f"{phase_name} output is empty"
        print(f"✓ {phase_name}: {len(df)} rows")

    # Validate ID formats
    tracking = pd.read_csv(f"segmentation/{exp_id}/segmentation_tracking.csv")
    assert tracking['embryo_id'].str.match(r'.*_e\d+').all(), "Invalid embryo_id format"
    assert tracking['snip_id'].str.match(r'.*_e\d+_t\d+').all(), "Invalid snip_id format"
    print(f"✓ ID formats validated")

    # Validate use_embryo flag
    qc = pd.read_csv(f"quality_control/{exp_id}/consolidated_qc_flags.csv")
    assert 'use_embryo' in qc.columns, "use_embryo flag missing"
    print(f"✓ QC flags validated")

    print(f"\n✓✓✓ Full pipeline validation PASSED for {exp_id} ✓✓✓")

if __name__ == "__main__":
    validate_full_pipeline(sys.argv[1])
```

**4. `scripts/extract_test_subset.sh`**
```bash
#!/bin/bash
# Extract subset of real data for testing

EXPERIMENT=$1
MICROSCOPE=$2
WELLS=$3
FRAMES=$4
OUTPUT_DIR=$5

# Example usage:
# ./extract_test_subset.sh 20250529_30hpf_ctrl YX1 "A01,B03" "0-9" test_data/real_subset_yx1/

echo "Extracting test subset:"
echo "  Experiment: $EXPERIMENT"
echo "  Microscope: $MICROSCOPE"
echo "  Wells: $WELLS"
echo "  Frames: $FRAMES"
echo "  Output: $OUTPUT_DIR"

# Create output structure
mkdir -p "$OUTPUT_DIR/raw_image_data/$MICROSCOPE/$EXPERIMENT"
mkdir -p "$OUTPUT_DIR/plate_metadata"
mkdir -p "$OUTPUT_DIR/expected_outputs"

# Copy raw data (implementation depends on microscope file structure)
# TODO: Implement microscope-specific copy logic

echo "✓ Test subset extracted to $OUTPUT_DIR"
```

**Deliverable:** All development scripts ready to use

---

### Agent 4: Extract Test Data - YX1
**Task:** Extract YX1 test subset from real data

**Requirements:**
- 2 wells (different conditions)
- 10 timepoints each
- BF channel + 1 fluorescence (if available)
- Known good segmentation quality

**Output structure:**
```
test_data/real_subset_yx1/
├── raw_image_data/
│   └── YX1/
│       └── test_yx1_001/
│           └── [raw ND2 or image files]
├── plate_metadata/
│   └── test_yx1_001_plate_layout.csv
└── README.md  (documents which wells, why chosen, expected results)
```

**Deliverable:** YX1 test subset ready for pipeline testing

---

### Agent 5: Extract Test Data - Keyence
**Task:** Extract Keyence test subset from real data

**Requirements:**
- 1 well
- 10 timepoints
- BF channel
- Known good segmentation quality

**Output structure:**
```
test_data/real_subset_keyence/
├── raw_image_data/
│   └── Keyence/
│       └── test_keyence_001/
│           └── [raw Keyence files]
├── plate_metadata/
│   └── test_keyence_001_plate_layout.csv
└── README.md
```

**Deliverable:** Keyence test subset ready for pipeline testing

---

## Wave 2: Phase 1 Implementation - Metadata Alignment (3 PARALLEL)

**Duration:** ~1-2 days
**Dependencies:** Wave 1 complete (schemas + test data available)

### Agent 1: Shared Metadata Modules
**Task:** Implement microscope-agnostic metadata processing

**Files to implement:**
```
metadata_ingest/plate/plate_processing.py
  - normalize_plate_layout()
  - validate and write plate_metadata.csv

metadata_ingest/mapping/align_scope_plate.py
  - join_scope_and_plate_metadata()
  - validate and write scope_and_plate_metadata.csv
```

**Test:** Run on both YX1 and Keyence test data (after other agents complete)

**Deliverable:** Shared metadata modules with validation

---

### Agent 2: YX1 Pipeline
**Task:** Implement YX1-specific metadata extraction and series mapping

**Files to implement:**
```
metadata_ingest/scope/yx1_scope_metadata.py
  - extract_yx1_scope_metadata()
  - normalize YX1 channel names
  - validate and write scope_metadata.csv

metadata_ingest/mapping/series_well_mapper_yx1.py
  - map_series_to_wells_yx1()
  - handle YX1-specific series conventions
  - write series_well_mapping.csv + provenance.json
```

**Test:** Run on YX1 test data through Phase 2

**Deliverable:** YX1 pipeline working end-to-end through metadata

---

### Agent 3: Keyence Pipeline
**Task:** Implement Keyence-specific metadata extraction and series mapping

**Files to implement:**
```
metadata_ingest/scope/keyence_scope_metadata.py
  - extract_keyence_scope_metadata()
  - normalize Keyence channel names
  - validate and write scope_metadata.csv

metadata_ingest/mapping/series_well_mapper_keyence.py
  - map_series_to_wells_keyence()
  - handle Keyence-specific series conventions
  - write series_well_mapping.csv + provenance.json
```

**Test:** Run on Keyence test data through Phase 2

**Deliverable:** Keyence pipeline working end-to-end through metadata

---

## Wave 3: Phase 2 Implementation - Image Building (2 PARALLEL)

**Duration:** ~1 day
**Dependencies:** Wave 2 complete (metadata extraction working)

### Agent 1: YX1 Image Building
**Task:** Implement YX1 stitching and manifest generation

**Files to implement:**
```
image_building/yx1/stitched_ff_builder.py
  - stitch_yx1_images()
  - write to built_image_data/{exp}/stitched_ff_images/

metadata_ingest/manifests/generate_image_manifest.py
  - generate_experiment_image_manifest()
  - validate channel normalization
  - write experiment_image_manifest.json
```

**Test:** Run YX1 test data through Phase 2, validate manifest

**Deliverable:** YX1 images stitched + manifest validated

---

### Agent 2: Keyence Image Building
**Task:** Implement Keyence stitching (reuse manifest generator)

**Files to implement:**
```
image_building/keyence/stitched_ff_builder.py
  - stitch_keyence_images()
  - handle z-stacking
  - write to built_image_data/{exp}/stitched_ff_images/
```

**Test:** Run Keyence test data through Phase 2, validate manifest

**Deliverable:** Keyence images stitched + manifest validated

---

## Wave 4: Phase 3 Implementation - SAM2 Segmentation (3 PARALLEL)

**Duration:** ~2-3 days
**Dependencies:** Wave 3 complete (image manifests available)

### Agent 1: GDINO Detection + Frame Organization
**Task:** Implement GroundingDINO detection and frame prep

**Files to implement:**
```
segmentation/grounded_sam2/frame_organization_for_sam2.py
  - organize frames for SAM2 (temp dir, symlinks)

segmentation/grounded_sam2/gdino_detection.py
  - run_gdino_detection()
  - select seed frame
  - write gdino_detections.json
```

**Deliverable:** GDINO detection + frame organization working

---

### Agent 2: SAM2 Propagation
**Task:** Implement SAM2 tracking

**Files to implement:**
```
segmentation/grounded_sam2/propagation.py
  - propagate_forward()
  - propagate_bidirectional()
  - write sam2_raw_output.json

segmentation/grounded_sam2/mask_export.py
  - export_masks_to_png()
  - write to segmentation/{exp}/mask_images/
```

**Deliverable:** SAM2 tracking + mask export working

---

### Agent 3: CSV Formatter + Validation
**Task:** Flatten JSON to validated CSV

**Files to implement:**
```
segmentation/grounded_sam2/csv_formatter.py
  - flatten_sam2_json_to_csv()
  - add mask_rle, well_id, is_seed_frame, source paths
  - validate against REQUIRED_COLUMNS_SEGMENTATION_TRACKING
  - write segmentation_tracking.csv
```

**Test:** Run full Phase 3, validate tracking CSV

**Deliverable:** Validated segmentation_tracking.csv

---

## Wave 5: Phases 4-5 Implementation - Snips + Features (2 PARALLEL)

**Duration:** ~2-3 days
**Dependencies:** Wave 4 complete (segmentation tracking available)

### Agent 1: Snip Processing
**Task:** Implement snip extraction and processing

**Files to implement:**
```
snip_processing/extraction.py
snip_processing/rotation.py
snip_processing/augmentation.py
snip_processing/process_snips.py
  - combined pipeline with --save-raw-crops config

snip_processing/manifest_generation.py
  - validate snip completeness
  - write snip_manifest.csv
```

**Deliverable:** Processed snips + validated manifest

---

### Agent 2: Feature Extraction
**Task:** Implement all feature computations

**Files to implement:**
```
feature_extraction/mask_geometry_metrics.py
feature_extraction/pose_kinematics_metrics.py
feature_extraction/fraction_alive.py  (stub for MVP)
feature_extraction/stage_inference.py
feature_extraction/consolidate_features.py
  - merge all features + metadata
  - validate against REQUIRED_COLUMNS_FEATURES
  - write consolidated_snip_features.csv
```

**Deliverable:** Validated consolidated features

---

## Wave 6: Phases 6-8 Implementation - QC + Analysis (2 PARALLEL)

**Duration:** ~2 days
**Dependencies:** Wave 5 complete (features available)

### Agent 1: QC Modules
**Task:** Implement QC checks and consolidation

**Files to implement:**
```
quality_control/segmentation_qc/segmentation_quality_qc.py
quality_control/auxiliary_mask_qc/death_detection.py  (stub)
quality_control/morphology_qc/size_validation_qc.py
quality_control/consolidation/consolidate_qc.py
quality_control/consolidation/compute_use_embryo.py
```

**Deliverable:** Validated consolidated_qc_flags.csv

---

### Agent 2: Analysis-Ready Assembly
**Task:** Create final analysis table

**Files to implement:**
```
analysis_ready/assemble_features_qc_embeddings.py
  - join features + QC + metadata
  - add embedding_calculated (stub False for MVP)
  - validate against REQUIRED_COLUMNS_ANALYSIS_READY
  - write features_qc_embeddings.csv
```

**Deliverable:** Validated analysis-ready table

---

## Execution Summary

**Total Waves:** 6
**Total Parallelism:** Up to 5 agents simultaneously in Wave 1

**Critical Path:**
Wave 1 (4 hrs) → Wave 2 (1-2 days) → Wave 3 (1 day) → Wave 4 (2-3 days) → Wave 5 (2-3 days) → Wave 6 (2 days)

**With Parallelism:** ~7-11 days
**Without Parallelism:** ~4-6 weeks

**Speedup:** 3-4x faster!

---

## Launch Commands

```bash
# Wave 1: Launch all 5 agents in parallel
snakemake_agent_1 create_schemas
snakemake_agent_2 setup_directories
snakemake_agent_3 create_scripts
snakemake_agent_4 extract_yx1_data
snakemake_agent_5 extract_keyence_data

# Wave 2: Launch 3 agents after Wave 1 completes
snakemake_agent_1 shared_metadata
snakemake_agent_2 yx1_pipeline
snakemake_agent_3 keyence_pipeline

# ... etc for subsequent waves
```

---

**Ready to launch Wave 1? 🚀**
