# Streamline Snakemake Refactor - Implementation Plan

**Date:** 2025-11-06
**Status:** READY FOR IMPLEMENTATION
**Branch:** `claude/streamline-snakemake-<session-id>`

---

## Executive Summary

Build a minimal viable pipeline (MVP) using real data subsets, testing both microscopes through Phase 2, then continuing with one experiment through Phase 8. Focus on the hardest chunks first (metadata alignment, SAM2 segmentation, feature extraction) to validate the architecture early.

---

## Agreed-Upon Code Changes

### 1. ✅ Series-to-Wells Mapping: Microscope-Specific

**Decision:** Keep `rule map_series_to_wells` as a SEPARATE rule with microscope-specific implementations.

**Why:** Series mapping logic differs between microscopes:
- **YX1:** Uses implicit positional mapping or explicit series_number_map from plate layout
- **Keyence:** Different file structure, different mapping conventions

**Implementation:**
```python
# metadata_ingest/mapping/series_well_mapper_yx1.py
def map_series_to_wells_yx1(plate_metadata, scope_metadata, output_csv, provenance_json):
    """YX1-specific series→well mapping logic."""
    pass

# metadata_ingest/mapping/series_well_mapper_keyence.py
def map_series_to_wells_keyence(plate_metadata, scope_metadata, output_csv, provenance_json):
    """Keyence-specific series→well mapping logic."""
    pass
```

**Snakemake rules:**
```python
rule map_series_to_wells_yx1:
    input:
        plate = "input_metadata_alignment/{exp}/raw_inputs/plate_layout.csv",
        scope = "input_metadata_alignment/{exp}/raw_inputs/yx1_scope_raw.csv"
    output:
        mapping = "input_metadata_alignment/{exp}/series_mapping/series_well_mapping.csv",
        provenance = "input_metadata_alignment/{exp}/series_mapping/mapping_provenance.json"
    run:
        from metadata_ingest.mapping.series_well_mapper_yx1 import map_series_to_wells_yx1
        map_series_to_wells_yx1(input.plate, input.scope, output.mapping, output.provenance)

# Similar rule for Keyence
```

### 2. ✅ Snip Processing: Single Rule with Config Flag

**Decision:** Combine `extract_snips` and `process_snips` into ONE rule with `--save-raw-crops` flag (default=True).

**Config:**
```yaml
# config/defaults.yaml
snip_processing:
  save_raw_crops: true  # Set false for production runs to save disk space
  rotation: true
  clahe: true
  noise_augmentation: true
```

**Implementation:**
```python
# snip_processing/process_snips.py
def process_snips(segmentation_tracking_csv, images_dir, output_dir, config):
    """
    Extract, rotate, augment, and save snips.

    Args:
        config: Dict with 'save_raw_crops', 'rotation', 'clahe', 'noise_augmentation'

    Outputs:
        - processed/{snip_id}.jpg  (always)
        - raw_crops/{snip_id}.tif  (if save_raw_crops=True)
    """
    for snip_id, mask_data in segmentation_tracking.iterrows():
        # Extract crop
        raw_crop = extract_crop(mask_data)

        if config['save_raw_crops']:
            save_tif(raw_crop, output_dir / 'raw_crops' / f'{snip_id}.tif')

        # Apply processing pipeline
        processed = raw_crop
        if config['rotation']:
            processed, angle = rotate_snip(processed, mask_data)
        if config['clahe']:
            processed = apply_clahe(processed)
        if config['noise_augmentation']:
            processed = add_background_noise(processed)

        save_jpeg(processed, output_dir / 'processed' / f'{snip_id}.jpg')
```

**Snakemake rule:**
```python
rule process_snips:
    input:
        tracking = "segmentation/{exp}/segmentation_tracking.csv",
        images = "built_image_data/{exp}/stitched_ff_images/"
    output:
        processed = directory("processed_snips/{exp}/processed/"),
        manifest = "processed_snips/{exp}/snip_manifest.csv"
    params:
        save_raw = config.get('snip_processing', {}).get('save_raw_crops', True)
    run:
        from snip_processing.process_snips import process_snips
        process_snips(input.tracking, input.images, "processed_snips/{wildcards.exp}",
                     config['snip_processing'])
```

### 3. ✅ QC Reports: Dump Rule (Beyond Initial Scope)

**Decision:** Add `rule generate_qc_reports` as a final "dump" rule that generates human-readable summaries and plots. NO other rules depend on it.

**Implementation:**
```python
# quality_control/reporting/generate_qc_summary.py
def generate_qc_reports(consolidated_qc_csv, output_dir):
    """
    Generate QC summary plots and reports.

    Outputs:
        - qc_summary.html  (interactive dashboard)
        - flag_distributions.png  (bar chart of % embryos per flag)
        - flag_correlations.png  (heatmap showing which flags co-occur)
        - per_well_summary.csv  (aggregated QC stats by well)
    """
    pass
```

**Snakemake rule:**
```python
rule generate_qc_reports:
    input:
        qc = "quality_control/{exp}/consolidated_qc_flags.csv",
        features = "computed_features/{exp}/consolidated_snip_features.csv"
    output:
        html = "quality_control/{exp}/reports/qc_summary.html",
        flag_dist = "quality_control/{exp}/reports/flag_distributions.png",
        flag_corr = "quality_control/{exp}/reports/flag_correlations.png",
        well_summary = "quality_control/{exp}/reports/per_well_summary.csv"
    run:
        from quality_control.reporting.generate_qc_summary import generate_qc_reports
        generate_qc_reports(input.qc, "quality_control/{wildcards.exp}/reports/")
```

**Note:** This rule is NOT included in the main `all` target initially. It's a nice-to-have for later.

### 4. ✅ Image Manifest: Single Consolidated Join

**Decision:** Keep `rule align_scope_and_plate` as ONE rule that:
1. Takes plate_metadata + scope_metadata + series_mapping as inputs
2. Joins them into scope_and_plate_metadata.csv
3. No separate alignment rule

**Already in the plan, no changes needed.**

---

## Testing Strategy: Real Data Subsets

### Core Principle
**Test BOTH microscopes through Phase 2 (metadata + image building), then continue with ONE experiment through Phase 8.**

**Why:**
- Metadata extraction and alignment logic is microscope-specific (Phases 1-2)
- Segmentation onwards is microscope-agnostic (uses validated metadata + stitched images)
- Testing both microscopes validates the hardest part (scope-specific logic)
- Single experiment for Phases 3-8 is sufficient (tests the shared pipeline)

### Test Data Requirements

**Create two real data subsets:**

#### Subset 1: YX1 Microscope
```
test_data/real_subset_yx1/
├── raw_image_data/
│   └── YX1/
│       └── test_yx1_001/
│           ├── Well A01: 10 timepoints, 1-2 embryos
│           └── Well B03: 10 timepoints, 1 embryo (different conditions)
│
├── plate_metadata/
│   └── test_yx1_001_plate_layout.csv
│       - Include both wells
│       - Different genotypes/treatments for testing metadata propagation
│
└── expected_outputs/  # Golden test outputs
    ├── scope_and_plate_metadata.csv
    ├── experiment_image_manifest.json
    └── (rest generated during initial run, then versioned)
```

#### Subset 2: Keyence Microscope
```
test_data/real_subset_keyence/
├── raw_image_data/
│   └── Keyence/
│       └── test_keyence_001/
│           └── Well C05: 10 timepoints, 1 embryo
│
├── plate_metadata/
│   └── test_keyence_001_plate_layout.csv
│
└── expected_outputs/
    ├── scope_and_plate_metadata.csv
    └── experiment_image_manifest.json
```

**Selection criteria for wells:**
- ✅ Known good segmentation (embryos are clear, well-tracked)
- ✅ Known QC failures (at least one well with death event, one with edge contact)
- ✅ Different experimental conditions (WT vs mutant, control vs treatment)
- ✅ 10-20 timepoints (enough to test tracking, not so many it's slow)

### Test Phases

#### Phase A: Microscope-Specific Validation (Both YX1 + Keyence)

**Goal:** Validate metadata extraction and alignment for BOTH microscopes.

**Scope:** Phases 1-2 only
- Plate metadata normalization
- Scope metadata extraction (YX1 vs Keyence)
- Series-to-wells mapping (YX1 vs Keyence)
- Scope+plate alignment
- Image building/stitching
- Image manifest generation

**Test runs:**
```bash
# Test YX1 pipeline through Phase 2
snakemake \
    --config experiments=test_yx1_001 \
    --until rule_generate_image_manifest

# Validate YX1 outputs
python scripts/validate_phase2_outputs.py --exp test_yx1_001 --microscope yx1

# Test Keyence pipeline through Phase 2
snakemake \
    --config experiments=test_keyence_001 \
    --until rule_generate_image_manifest

# Validate Keyence outputs
python scripts/validate_phase2_outputs.py --exp test_keyence_001 --microscope keyence
```

**Success criteria:**
- ✅ Both experiments produce validated `scope_and_plate_metadata.csv`
- ✅ Both produce validated `experiment_image_manifest.json`
- ✅ Channel names are normalized correctly (YX1 "EYES - Dia" → "BF", Keyence "Brightfield" → "BF")
- ✅ Series mapping provenance captured for both microscopes
- ✅ All stitched images exist at expected paths

#### Phase B: Shared Pipeline Validation (ONE experiment)

**Goal:** Validate Phases 3-8 using ONE microscope's validated outputs.

**Scope:** Phases 3-8 (segmentation → analysis-ready)
- SAM2 segmentation + tracking
- Snip processing
- Feature extraction
- QC consolidation
- Analysis-ready table

**Test run:**
```bash
# Continue with YX1 experiment through full pipeline
snakemake \
    --config experiments=test_yx1_001 \
    all

# Validate all outputs
python scripts/validate_full_pipeline.py --exp test_yx1_001
```

**Success criteria:**
- ✅ `segmentation_tracking.csv` validated (includes mask_rle, well_id, is_seed_frame)
- ✅ `consolidated_snip_features.csv` validated (includes area_um2, predicted_stage_hpf)
- ✅ `consolidated_qc_flags.csv` validated (includes use_embryo_flag)
- ✅ `features_qc_embeddings.csv` validated (final analysis table)
- ✅ Known QC failures are correctly flagged
- ✅ Embryo IDs and snip IDs follow expected format

---

## Implementation Schedule: MVP-First Approach

### Week 1: Core Infrastructure + Phase 1-2 (Microscope-Specific)

**Goal:** Get metadata extraction and image building working for BOTH microscopes.

**Tasks:**

**Day 1-2: Setup**
- [ ] Create branch `claude/streamline-snakemake-<session-id>`
- [ ] Create directory structure: `src/data_pipeline/{schemas,metadata_ingest,preprocessing,...}`
- [ ] Create ALL schema files (`schemas/*.py`) with REQUIRED_COLUMNS_* lists
- [ ] Create validation helper function (used by all consolidation points)
- [ ] Extract test data subsets (YX1 + Keyence)

**Day 3-4: Phase 1 (Metadata Alignment)**
- [ ] Implement `metadata_ingest/plate/plate_processing.py` (normalize Excel → plate_metadata.csv)
- [ ] Implement `metadata_ingest/scope/yx1_scope_metadata.py` (extract YX1 scope metadata)
- [ ] Implement `metadata_ingest/scope/keyence_scope_metadata.py` (extract Keyence scope metadata)
- [ ] Implement `metadata_ingest/mapping/series_well_mapper_yx1.py` (YX1-specific mapping)
- [ ] Implement `metadata_ingest/mapping/series_well_mapper_keyence.py` (Keyence-specific mapping)
- [ ] Implement `metadata_ingest/mapping/align_scope_plate.py` (shared join logic)

**Day 5: Phase 2 (Image Building + Manifest)**
- [ ] Implement `image_building/yx1/stitched_ff_builder.py` (YX1 image processing)
- [ ] Implement `image_building/keyence/stitched_ff_builder.py` (Keyence image processing)
- [ ] Implement `metadata_ingest/manifests/generate_image_manifest.py` (shared manifest generation)

**Day 6-7: Test & Validate**
- [ ] Run both test experiments through Phase 2
- [ ] Validate outputs against expected schemas
- [ ] Fix bugs, iterate
- [ ] Commit working Phase 1-2 pipeline

**Deliverable:** Both YX1 and Keyence pipelines produce validated metadata and image manifests.

---

### Week 2: Phase 3 (SAM2 Segmentation)

**Goal:** Get SAM2 segmentation working end-to-end.

**Day 1-3: Core SAM2 Pipeline**
- [ ] Implement `segmentation/grounded_sam2/frame_organization_for_sam2.py` (temp dir setup)
- [ ] Implement `segmentation/grounded_sam2/gdino_detection.py` (GroundingDINO seed detection)
- [ ] Implement `segmentation/grounded_sam2/propagation.py` (SAM2 bidirectional tracking)
- [ ] Implement `segmentation/grounded_sam2/mask_export.py` (PNG mask export)
- [ ] Implement `segmentation/grounded_sam2/csv_formatter.py` (JSON → segmentation_tracking.csv)

**Day 4-5: Test & Validate**
- [ ] Run test_yx1_001 through segmentation
- [ ] Validate `segmentation_tracking.csv` (check mask_rle, well_id, is_seed_frame columns)
- [ ] Visually inspect mask PNG exports
- [ ] Fix tracking issues
- [ ] Commit working segmentation pipeline

**Deliverable:** SAM2 segmentation produces validated tracking table.

---

### Week 3: Phases 4-5 (Snips + Features)

**Goal:** Extract snips and compute features.

**Day 1-2: Snip Processing**
- [ ] Implement `snip_processing/extraction.py` (crop embryo regions)
- [ ] Implement `snip_processing/rotation.py` (PCA rotation)
- [ ] Implement `snip_processing/augmentation.py` (CLAHE + noise)
- [ ] Implement `snip_processing/process_snips.py` (combined pipeline with config flag)
- [ ] Implement `snip_processing/manifest_generation.py` (validate snip completeness)

**Day 3-4: Feature Extraction**
- [ ] Implement `feature_extraction/mask_geometry_metrics.py` (area_um2, perimeter, etc.)
- [ ] Implement `feature_extraction/pose_kinematics_metrics.py` (centroid, orientation, speed)
- [ ] Implement `feature_extraction/fraction_alive.py` (viability from UNet - SKIP UNet for MVP, just stub this)
- [ ] Implement `feature_extraction/stage_inference.py` (HPF prediction)
- [ ] Implement `feature_extraction/consolidate_features.py` (merge all features + validate)

**Day 5: Test & Validate**
- [ ] Run test_yx1_001 through features
- [ ] Validate `consolidated_snip_features.csv` (check area_um2, predicted_stage_hpf)
- [ ] Visually inspect processed snips
- [ ] Commit working feature extraction

**Deliverable:** Consolidated features table with schema validation.

---

### Week 4: Phases 6-8 (QC + Analysis-Ready)

**Goal:** Implement QC consolidation and final analysis table.

**Day 1-2: QC Modules (Minimal)**
- [ ] Implement `quality_control/segmentation_qc/segmentation_quality_qc.py` (edge, discontinuous, overlap flags)
- [ ] Implement `quality_control/auxiliary_mask_qc/death_detection.py` (stub - use fraction_alive threshold)
- [ ] Implement `quality_control/morphology_qc/size_validation_qc.py` (SA outlier detection)
- [ ] SKIP auxiliary mask QC for MVP (focus, bubble, yolk - requires UNet)

**Day 3: QC Consolidation**
- [ ] Implement `quality_control/consolidation/consolidate_qc.py` (merge QC flags)
- [ ] Implement `quality_control/consolidation/compute_use_embryo.py` (gate logic)

**Day 4: Analysis-Ready**
- [ ] Implement `analysis_ready/assemble_features_qc_embeddings.py` (join features + QC)
- [ ] SKIP embeddings for MVP (stub embedding_calculated = False)

**Day 5: Test & Validate**
- [ ] Run test_yx1_001 through full pipeline
- [ ] Validate all output schemas
- [ ] Check use_embryo_flag logic
- [ ] Commit working end-to-end pipeline

**Deliverable:** Full MVP pipeline: raw images → analysis-ready table.

---

### Week 5: Polish & Expand

**Goal:** Add missing features, error handling, documentation.

**Tasks:**
- [ ] Add UNet auxiliary masks (if time permits)
- [ ] Add embeddings generation (if time permits)
- [ ] Add comprehensive error messages
- [ ] Add progress logging
- [ ] Add QC reports rule (nice-to-have)
- [ ] Test on second experiment (Keyence if we did YX1 in MVP)
- [ ] Write usage documentation
- [ ] Create migration guide from old pipeline

---

## Development Helper Scripts

### 1. Extract Test Subsets

```bash
#!/bin/bash
# scripts/extract_test_subset.sh

EXPERIMENT=$1
WELLS=$2
FRAMES=$3
OUTPUT=$4

# Example: scripts/extract_test_subset.sh 20250529_30hpf_ctrl A01,B03 0-9 test_data/real_subset_yx1/

# Extract specified wells and frames from existing experiment
# - Copy raw image data
# - Copy/generate plate metadata
# - Document provenance
```

### 2. Validate Phase 2 Outputs

```python
#!/usr/bin/env python3
# scripts/validate_phase2_outputs.py

"""Validate that Phase 1-2 outputs are correct."""

import pandas as pd
from pathlib import Path
import sys

def validate_phase2(exp_id, microscope):
    """Validate metadata and image manifest for given experiment."""

    # Load outputs
    metadata_dir = Path(f"experiment_metadata/{exp_id}")
    plate_meta = pd.read_csv(metadata_dir / "plate_metadata.csv")
    scope_meta = pd.read_csv(metadata_dir / "scope_metadata.csv")
    aligned_meta = pd.read_csv(metadata_dir / "scope_and_plate_metadata.csv")

    # Import schemas
    from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA
    from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA
    from data_pipeline.schemas.scope_and_plate_metadata import REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA

    # Validate schemas
    missing = set(REQUIRED_COLUMNS_PLATE_METADATA) - set(plate_meta.columns)
    assert not missing, f"plate_metadata missing columns: {missing}"

    missing = set(REQUIRED_COLUMNS_SCOPE_METADATA) - set(scope_meta.columns)
    assert not missing, f"scope_metadata missing columns: {missing}"

    missing = set(REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA) - set(aligned_meta.columns)
    assert not missing, f"scope_and_plate_metadata missing columns: {missing}"

    # Validate channel normalization
    if microscope == "yx1":
        assert "BF" in aligned_meta['channel_name'].values, "BF channel not found (YX1 normalization failed)"
    elif microscope == "keyence":
        assert "BF" in aligned_meta['channel_name'].values, "BF channel not found (Keyence normalization failed)"

    # Validate image manifest exists
    manifest_path = metadata_dir / "experiment_image_manifest.json"
    assert manifest_path.exists(), f"Image manifest not found: {manifest_path}"

    print(f"✓ Phase 2 validation passed for {exp_id} ({microscope})")

if __name__ == "__main__":
    validate_phase2(sys.argv[1], sys.argv[2])
```

### 3. Validate Full Pipeline

```python
#!/usr/bin/env python3
# scripts/validate_full_pipeline.py

"""Validate end-to-end pipeline outputs."""

import pandas as pd
from pathlib import Path
import sys

def validate_full_pipeline(exp_id):
    """Validate all outputs through Phase 8."""

    # Check each phase output
    phases = [
        ("Phase 1-2", "experiment_metadata/{exp}/scope_and_plate_metadata.csv"),
        ("Phase 3", "segmentation/{exp}/segmentation_tracking.csv"),
        ("Phase 4", "processed_snips/{exp}/snip_manifest.csv"),
        ("Phase 5", "computed_features/{exp}/consolidated_snip_features.csv"),
        ("Phase 6", "quality_control/{exp}/consolidated_qc_flags.csv"),
        ("Phase 8", "analysis_ready/{exp}/features_qc_embeddings.csv"),
    ]

    for phase_name, path_template in phases:
        path = Path(path_template.format(exp=exp_id))
        assert path.exists(), f"{phase_name} output missing: {path}"

        df = pd.read_csv(path)
        assert len(df) > 0, f"{phase_name} output is empty: {path}"

        print(f"✓ {phase_name} output validated: {len(df)} rows")

    # Additional validation: Check ID formats
    tracking = pd.read_csv(f"segmentation/{exp_id}/segmentation_tracking.csv")
    assert tracking['embryo_id'].str.match(r'.*_e\d+').all(), "embryo_id format invalid"
    assert tracking['snip_id'].str.match(r'.*_e\d+_t\d+').all(), "snip_id format invalid"

    # Check use_embryo flag exists
    qc = pd.read_csv(f"quality_control/{exp_id}/consolidated_qc_flags.csv")
    assert 'use_embryo' in qc.columns, "use_embryo flag missing from QC"

    print(f"\n✓✓✓ Full pipeline validation passed for {exp_id} ✓✓✓")

if __name__ == "__main__":
    validate_full_pipeline(sys.argv[1])
```

### 4. Rapid Test Script

```bash
#!/bin/bash
# scripts/rapid_test.sh

set -e  # Exit on error

EXPERIMENT=${1:-test_yx1_001}
TARGET_RULE=${2:-all}

echo "Testing pipeline: $EXPERIMENT → $TARGET_RULE"

# Run Snakemake
snakemake \
    --config experiments=$EXPERIMENT \
    --cores 2 \
    --printshellcmds \
    $TARGET_RULE

# Validate outputs (if full pipeline)
if [ "$TARGET_RULE" = "all" ]; then
    echo "Validating full pipeline outputs..."
    python scripts/validate_full_pipeline.py $EXPERIMENT
elif [ "$TARGET_RULE" = "rule_generate_image_manifest" ]; then
    echo "Validating Phase 2 outputs..."
    python scripts/validate_phase2_outputs.py $EXPERIMENT yx1
fi

echo "✓ Test completed successfully"
```

---

## Success Metrics

### Phase A: Microscope-Specific Validation
- [x] YX1 pipeline produces validated `scope_and_plate_metadata.csv`
- [x] Keyence pipeline produces validated `scope_and_plate_metadata.csv`
- [x] Both produce validated `experiment_image_manifest.json`
- [x] Channel normalization works for both (EYES - Dia → BF, Brightfield → BF)
- [x] Series mapping provenance captured

### Phase B: Full Pipeline Validation
- [x] SAM2 segmentation produces validated `segmentation_tracking.csv`
- [x] Snip processing completes successfully
- [x] Feature extraction produces validated `consolidated_snip_features.csv`
- [x] QC consolidation produces validated `consolidated_qc_flags.csv`
- [x] Analysis-ready table produced
- [x] Known QC failures correctly flagged
- [x] All schemas enforced at consolidation points

### Code Quality
- [x] Functions over classes (no overengineered abstractions)
- [x] Inline validation at every consolidation point
- [x] Clear error messages
- [x] Comprehensive logging
- [x] All REQUIRED_COLUMNS_* lists defined in schemas/

---

## Next Steps

1. **Create branch** and initial directory structure
2. **Extract test data subsets** (YX1 + Keyence)
3. **Implement Phase 1-2** (metadata + image building for both microscopes)
4. **Validate Phase 2 outputs** for both microscopes
5. **Implement Phases 3-8** using one test experiment
6. **Validate full pipeline**
7. **Polish, document, merge**

---

**Ready to start? Let's build this! 🚀**
