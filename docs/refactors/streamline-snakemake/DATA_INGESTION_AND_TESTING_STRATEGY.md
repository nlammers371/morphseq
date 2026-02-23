# Data Ingestion & Testing Strategy for Streamline-Snakemake Refactor

**Date:** 2025-11-06
**Status:** Answering critical questions from implementation log
**Approach:** Symlink-based data management (replacing poorly-named `morphseq_playground`)

## 2026-02-10 - Addendum, highlighting what we need to change in the original doc
This addendum updates ingest interpretation while preserving the existing testing strategy.

What stays the same:
- Symlink-based data setup strategy.
- Incremental validation by pipeline phase.
- Downstream phase testing expectations (segmentation onward).

What to update:
1. Run ingest as scope-first (YX1 and Keyence remain separate through extraction/mapping).
2. Keep materialization scope-specific and emit `stitched_image_index.csv` during building (reporter pattern).
3. Validate and use `frame_manifest.csv` as the canonical pre-segmentation frame metadata table.
4. Use canonical frame-level naming in checks:
   - `channel_id`
   - `channel_name_raw`
   - `temperature`
   - required `micrometers_per_pixel`

---

## Executive Summary

This document addresses all 5 critical questions from `2025-11-06_initial-implementation.md`:

1. ✅ **Test Data Access** - Network mount accessible; using `20250912` (YX1) and `20240509_24hpf` (Keyence)
2. ✅ **Snakefile Priority** - Create test data structure via symlinks first, then write Snakefile
3. ✅ **Testing Strategy** - Recommend **Option 1** (extract test subset → Snakefile → validate incrementally)
4. ✅ **MVP Stubs** - UNet and embeddings OK to leave stubbed for initial validation
5. ✅ **Config Management** - Centralized `data_pipeline_output/` with configurable `data_root` path

---

## Question 1: Test Data Access

### Status: ✅ ACCESSIBLE

The network mount is available at:
```
/net/trapnell/vol1/home/nlammers/projects/data/morphseq/
```

**Verification:**
```bash
ls -la /net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/
```

Expected output:
```
Keyence/  YX1/  reference_data/  ...
```

### Available Test Experiments

#### YX1 Confirmed Good:
- **20240418** ✅ RECOMMENDED
  - Date: April 18, 2024
  - Wells: A01, C01 (and others)
  - Status: Confirmed working experiment
  - Location: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/YX1/20240418/`

#### Keyence Available:
- **20240509_24hpf** ✅ RECOMMENDED
  - Date: May 9, 2024
  - Stage: 24 hours post-fertilization
  - Status: Confirmed available
  - Location: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/Keyence/20240509_24hpf/`

- **20240507** (Alternative)
  - Date: May 7, 2024
  - Status: Available
  - Location: `/net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data/Keyence/20240507/`

**Note:** The experiments in test_data READMEs (20250612, 20250911) do NOT exist. We'll update extraction scripts to use confirmed available experiments above.

---

## Question 2: Snakefile Priority

### Status: ✅ CREATE TEST DATA STRUCTURE FIRST

**Recommended Sequence:**

```
Step 1: Create data_pipeline_output/ with symlinks
   └─ 1-2 hours (one-time setup)

Step 2: Create Snakefile with all rules defined
   └─ 2-3 hours (straightforward, rules already documented)

Step 3: Extract test data subset
   └─ 30 minutes (pull from network, create test dirs)

Step 4: Test Phase 1-2 (metadata + image building)
   └─ 15 minutes per microscope type

Step 5: Test Phase 3-8 (segmentation through analysis-ready)
   └─ 1-2 hours (depends on image count)
```

**Why this order:**
- Test data structure ready before Snakefile writes outputs
- Symlinks avoid copying large files (they're on network)
- Snakefile can be created while someone else extracts data
- Parallelizable effort

---

## Question 3: Testing Strategy

### Status: ✅ RECOMMEND OPTION 1 (Modified)

**Original Options:** 1 (full), 2 (mock), 3 (manual)

**Our Recommendation: Option 1 with Symlinks**

```
1. Create data_pipeline_output/ directory structure with symlinks
   ├── inputs/raw_image_data -> /net/trapnell/.../raw_image_data/
   ├── inputs/plate_metadata -> /net/trapnell/.../plate_metadata/
   └── inputs/models -> /net/trapnell/.../models/

2. Create Snakefile with all Phase 1-8 rules

3. Extract minimal test subset:
   ├── YX1: wells A01, C01 from 20240418 (first 10 frames each)
   ├── Keyence: well A12 from 20240509_24hpf (first 10 frames)
   └── Create plate_metadata CSVs for test wells

4. Run full pipeline on test data:
   snakemake --config experiments=test_yx1_001,test_keyence_001 -j 4

5. Validate each phase outputs:
   ├── Phase 1-2: Metadata alignment + image building
   ├── Phase 3: Segmentation tracking CSV
   ├── Phase 4-5: Feature consolidation
   ├── Phase 6-8: QC flags + analysis-ready table
```

**Why this approach:**
- ✅ Tests both YX1 and Keyence paths
- ✅ Real data (not synthetic/mock)
- ✅ Validates full pipeline end-to-end
- ✅ Symlinks avoid large file copies
- ✅ Incremental validation at each phase

---

## Question 4: MVP Stubs

### Status: ✅ STUBS ARE FINE

**UNet Auxiliary Masks:** OK to stub
- Returns empty DataFrames with correct schema
- QC consolidation handles missing flags gracefully
- Not critical for SAM2 validation

**Embeddings Generation:** OK to stub
- `analysis_ready/assemble_features_qc_embeddings.py` sets `embedding_calculated=False`
- Analysis notebooks can filter on this flag
- VAE inference can be added later without breaking pipeline

**What we CANNOT stub:**
- SAM2 segmentation (primary path - required)
- Feature extraction (required for QC)
- QC consolidation (required for gating)

---

## Question 5: Config Management

### Status: ✅ CENTRALIZED data_pipeline_output/ WITH CONFIGURABLE data_root

---

## The Data Pipeline Output Strategy

### Directory Structure (NEW)

Replace poorly-named `morphseq_playground` with:

```
data_pipeline_output/                           # NEW PIPELINE OUTPUT ROOT
│
├── .gitignore                                  # Ignore all outputs (except structure)
│
├── inputs/                                     # SYMLINKS ONLY (no copies)
│   ├── raw_image_data -> /net/trapnell/.../raw_image_data/
│   │                     [READ-ONLY SYMLINK TO NETWORK MOUNT]
│   ├── plate_metadata -> /net/trapnell/.../metadata/plate_metadata/
│   │                     [READ-ONLY SYMLINK TO NETWORK MOUNT]
│   ├── models -> /net/trapnell/.../models/
│   │              [READ-ONLY SYMLINK TO NETWORK MOUNT]
│   └── reference_data -> /net/trapnell/.../reference_data/
│                         [READ-ONLY SYMLINK TO NETWORK MOUNT]
│
├── experiment_metadata/                        # PHASE 1 OUTPUTS (pipeline writes)
│   └── {experiment_id}/
│       ├── plate_metadata.csv [VALIDATED]
│       ├── scope_metadata_raw.csv [VALIDATED]
│       ├── series_well_mapping.csv
│       ├── scope_metadata_mapped.csv [VALIDATED]
│       ├── stitched_image_index.csv [VALIDATED]
│       └── frame_manifest.csv [VALIDATED]
│
├── built_image_data/                           # PHASE 2 OUTPUTS (pipeline writes)
│   └── {experiment_id}/
│       └── stitched_ff_images/
│           └── {well_index}/{channel_id}/{image_id}.tif
│
├── segmentation/                               # PHASE 3 OUTPUTS (pipeline writes)
│   └── {experiment_id}/
│       ├── segmentation_tracking.csv [VALIDATED]
│       ├── mask_images/
│       │   └── {image_id}_masks.png
│       └── unet_masks/ (stubbed for MVP)
│
├── processed_snips/                            # PHASE 4 OUTPUTS (pipeline writes)
│   └── {experiment_id}/
│       ├── raw_crops/
│       ├── processed/
│       └── snip_manifest.csv [VALIDATED]
│
├── computed_features/                          # PHASE 5 OUTPUTS (pipeline writes)
│   └── {experiment_id}/
│       ├── mask_geometry_metrics.csv
│       ├── pose_kinematics_metrics.csv
│       ├── stage_predictions.csv
│       └── consolidated_snip_features.csv [VALIDATED]
│
├── quality_control/                            # PHASE 6-7 OUTPUTS (pipeline writes)
│   └── {experiment_id}/
│       ├── segmentation_quality_qc.csv
│       ├── auxiliary_mask_qc.csv (stubbed for MVP)
│       ├── embryo_death_qc.csv
│       ├── surface_area_outliers_qc.csv
│       └── consolidated_qc_flags.csv [VALIDATED]
│
├── latent_embeddings/                          # PHASE 7 OUTPUTS (stubbed for MVP)
│   └── {model_name}/
│       ├── {experiment_id}_embedding_manifest.csv
│       └── {experiment_id}_latents.csv
│
└── analysis_ready/                             # PHASE 8 OUTPUTS (pipeline writes)
    └── {experiment_id}/
        └── features_qc_embeddings.csv [VALIDATED]
```

### Why Symlinks?

1. **Avoid Duplication** - Raw data lives on network mount (read-only)
2. **Fast Setup** - Creating symlinks is instant vs copying GBs of data
3. **Single Source of Truth** - All raw data in one place
4. **Easy Cleanup** - Remove symlinks, keep network data intact
5. **Matches morphseq_playground** - We're replacing it with the same approach but better organized

### Snakemake Configuration

**Snakemake config.yaml:**
```yaml
# src/data_pipeline/pipeline_orchestrator/config.yaml
#
# You can also keep a local copy under data_pipeline_output/ for convenience, but
# the repo-tracked canonical config lives under src/.

# Data root - can be overridden at runtime
data_root: "data_pipeline_output"  # LOCAL: symlinks

# OR for production (read-only network mount)
# data_root: "/net/trapnell/vol1/home/nlammers/projects/data/morphseq"

# Experiments to process (can be overridden: snakemake --config experiments=exp1,exp2)
experiments:
  - test_yx1_001
  - test_keyence_001

# Device configuration (GPU auto-detect or override)
device: "auto"  # "cuda", "cpu", or "auto"

# Processing parameters
processing:
  save_raw_crops: true
  clahe_clip_limit: 2.0
  background_noise_std_multiplier: 0.1

# QC thresholds
qc:
  death_viability_threshold: 0.9
  surface_area_outlier_zscore: 3.0
  gdino_confidence_threshold: 0.3
  sam2_confidence_threshold: 0.5
```

**Using alternative data_root:**
```bash
# Run on local symlinks (fast for testing)
snakemake -j 4 --config data_root=data_pipeline_output

# OR run on network mount directly (production)
snakemake -j 4 --config data_root=/net/trapnell/.../morphseq

# OR run with specific experiments only
snakemake -j 4 \
  --config data_root=data_pipeline_output \
  --config experiments=test_yx1_001
```

---

## Setup Instructions

### 1. Create data_pipeline_output/ with Symlinks

```bash
# Create directory structure
mkdir -p data_pipeline_output/{inputs,experiment_metadata,built_image_data,segmentation,processed_snips,computed_features,quality_control,latent_embeddings,analysis_ready}

# Create symlinks to network data (read-only)
ln -s /net/trapnell/vol1/home/nlammers/projects/data/morphseq/raw_image_data \
      data_pipeline_output/inputs/raw_image_data

ln -s /net/trapnell/vol1/home/nlammers/projects/data/morphseq/metadata/plate_metadata \
      data_pipeline_output/inputs/plate_metadata

ln -s /net/trapnell/vol1/home/nlammers/projects/data/morphseq/models \
      data_pipeline_output/inputs/models

ln -s /net/trapnell/vol1/home/nlammers/projects/data/morphseq/reference_data \
      data_pipeline_output/inputs/reference_data

# Verify symlinks
ls -la data_pipeline_output/inputs/
```

### 2. Extract Test Data Subset

#### Option A: Use existing network experiments (FAST)

Create minimal test directories that just symlink to subset of network data:

```bash
# YX1 test data
mkdir -p data_pipeline_output/inputs/raw_image_data/YX1/test_yx1_001
# Copy/link first 10 frames from 20240418 wells A01, C01

# Keyence test data
mkdir -p data_pipeline_output/inputs/raw_image_data/Keyence/test_keyence_001
# Copy/link first 10 frames from 20240509_24hpf well A12

# Create plate metadata CSVs
cat > data_pipeline_output/inputs/plate_metadata/test_yx1_001_plate_layout.csv <<EOF
experiment_id,well_id,well_index,genotype,treatment,temperature,embryos_per_well
test_yx1_001,test_yx1_001_A01,A01,wildtype,control,28.5,1
test_yx1_001,test_yx1_001_C01,C01,wildtype,control,28.5,1
EOF

cat > data_pipeline_output/inputs/plate_metadata/test_keyence_001_plate_layout.csv <<EOF
experiment_id,well_id,well_index,genotype,treatment,temperature,embryos_per_well
test_keyence_001,test_keyence_001_A12,A12,atf6_ctrl,control,28.5,1
EOF
```

#### Option B: Create proper test subset (THOROUGH)

Use updated extraction scripts (see below) to copy select frames:

```bash
# Update and run extraction script
cd test_data/real_subset_yx1/
bash extract_yx1_subset.sh  # (updated script)

# Update and run Keyence extraction
cd test_data/real_subset_keyence/
bash extract_keyence_subset.sh  # (updated script)

# Copy to data_pipeline_output
cp -r test_data/real_subset_yx1/raw_image_data/YX1/test_yx1_001 \
      data_pipeline_output/inputs/raw_image_data/YX1/

cp -r test_data/real_subset_keyence/raw_image_data/Keyence/test_keyence_001 \
      data_pipeline_output/inputs/raw_image_data/Keyence/
```

### 3. Add to .gitignore

```bash
# data_pipeline_output/.gitignore (or append to repo .gitignore)

# Ignore all pipeline outputs EXCEPT this file and structure
data_pipeline_output/**/*
!data_pipeline_output/.gitignore
!data_pipeline_output/inputs/
!data_pipeline_output/inputs/.gitkeep

# Exception: Keep config files and symlinks (they're small)
!data_pipeline_output/config.yaml
!data_pipeline_output/config/

# Exception: Keep validation scripts
!data_pipeline_output/validate_outputs.sh
```

---

## Updated Test Data Extraction

### Problem with Existing READMEs

Test data extraction scripts reference **non-existent experiments:**
- YX1: `20250911` (doesn't exist)
- Keyence: `20250612_24hpf_ctrl_atf6` (doesn't exist)

### Solution: Update Extraction Scripts

**New YX1 extraction** (`test_data/real_subset_yx1/extract_yx1_subset.sh`):
```bash
EXPERIMENT_ID="20240418"  # Changed from 20250911 (doesn't exist)
WELLS=("A01" "C01")       # Changed from A6, B4 (update accordingly)
NUM_FRAMES=10
NETWORK_ROOT="/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
```

**New Keyence extraction** (`test_data/real_subset_keyence/extract_keyence_subset.sh`):
```bash
EXPERIMENT_ID="20240509_24hpf"  # Changed from 20250612_24hpf_ctrl_atf6 (doesn't exist)
WELL="A12"
NUM_FRAMES=10
NETWORK_ROOT="/net/trapnell/vol1/home/nlammers/projects/data/morphseq"
```

**Then update README.md files** with actual experiment info and successful extraction notes.

---

## Snakefile Creation Checklist

### Rules to Define (Phase 1-8)

Note: the rule names below are illustrative; prefer the canonical rule names in `snakemake_rules_data_flow.md`.

- [ ] **Phase 1: Metadata Processing**
  - `rule_extract_plate_metadata`
  - `rule_extract_yx1_scope_metadata`
  - `rule_extract_keyence_scope_metadata`
  - `rule_map_series_to_wells_yx1`
  - `rule_map_series_to_wells_keyence`
  - `rule_apply_series_mapping_yx1`
  - `rule_apply_series_mapping_keyence`

- [ ] **Phase 2: Image Building + Frame Contract**
  - `rule_build_yx1_stitched_images`
  - `rule_build_keyence_stitched_images`
  - `rule_validate_stitched_image_index`
  - `rule_build_frame_manifest`

- [ ] **Phase 3: Segmentation**
  - `rule_prepare_frames_for_sam2`
  - `rule_detect_embryos_gdino`
  - `rule_propagate_sam2_masks`
  - `rule_export_masks_to_png`
  - `rule_format_segmentation_tracking_csv`

- [ ] **Phase 4-5: Snip Processing & Features**
  - `rule_process_snips`
  - `rule_extract_mask_geometry`
  - `rule_extract_pose_kinematics`
  - `rule_compute_fraction_alive`
  - `rule_predict_developmental_stage`
  - `rule_consolidate_features`

- [ ] **Phase 6-7: QC**
  - `rule_segmentation_quality_qc`
  - `rule_auxiliary_mask_qc` (stub: returns empty with correct schema)
  - `rule_death_detection`
  - `rule_size_validation_qc`
  - `rule_consolidate_qc_flags`

- [ ] **Phase 8: Analysis Ready**
  - `rule_assemble_analysis_ready_table`

### Key Snakefile Patterns

```python
# Access data_root from config
DATA_ROOT = config.get("data_root", "data_pipeline_output")
RAW_DATA = f"{DATA_ROOT}/inputs/raw_image_data"
PLATE_META = f"{DATA_ROOT}/inputs/plate_metadata"

# Get experiments from config
EXPERIMENTS = config.get("experiments", ["test_yx1_001", "test_keyence_001"])

# Input for Phase 1 (raw images)
rule rule_extract_yx1_scope_metadata:
    input:
        expand(f"{RAW_DATA}/YX1/{{exp}}/{{well}}_Seq0001.nd2",
               exp=EXPERIMENTS, well=["A01"])
    output:
        f"{DATA_ROOT}/experiment_metadata/{{exp}}/scope_metadata_raw.csv"
    shell:
        "\"$PYTHON\" -m data_pipeline.metadata_ingest.scope.yx1.extract_scope_metadata {input} {output}"

# Rules build on each other
rule rule_apply_series_mapping_yx1:
    input:
        scope=rules.rule_extract_yx1_scope_metadata.output,
        plate=rules.rule_extract_plate_metadata.output,
        mapping=rules.rule_map_series_to_wells_yx1.output
    output:
        f"{DATA_ROOT}/experiment_metadata/{{exp}}/scope_metadata_mapped.csv"
    shell:
        "\"$PYTHON\" -m data_pipeline.metadata_ingest.scope.yx1.apply_series_mapping {input.scope} {input.plate} {input.mapping} {output}"
```

---

## Validation & Testing Workflow

### Phase 1-2 Validation (Metadata + Image Building)

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python

# Run first 2 phases
snakemake \
    --config data_root=data_pipeline_output \
    --until build_frame_manifest \
    -j 4

# Validate Phase 1 outputs (metadata alignment)
#
# Note: the repo currently has a Phase 1 validation harness under test_data/.
"$PYTHON" test_data/test_phase1_metadata_alignment.py

# Expected outputs
ls data_pipeline_output/experiment_metadata/test_yx1_001/
# Should show: plate_metadata.csv, scope_metadata_raw.csv, series_well_mapping.csv, scope_metadata_mapped.csv, stitched_image_index.csv, frame_manifest.csv
```

### Phase 3-8 Validation (Full Pipeline)

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python

# Run full pipeline
snakemake \
    --config data_root=data_pipeline_output \
    --config experiments=test_yx1_001 \
    -j 4

# Validate final outputs
# TODO: add a dedicated validation script once the refactor implementation is wired end-to-end.
# For now, validate by checking that expected contract files exist and are non-empty.
ls -la data_pipeline_output/analysis_ready/test_yx1_001/features_qc_embeddings.csv

# Expected: features_qc_embeddings.csv with correct columns
wc -l data_pipeline_output/analysis_ready/test_yx1_001/features_qc_embeddings.csv
# Should be: 1 header + N snips rows
```

---

## Migration from morphseq_playground

### Current State
`morphseq_playground` contains legacy pipeline outputs with poorly organized naming.

### Transition Plan

1. **Create data_pipeline_output/** with symlinks (doesn't disturb morphseq_playground)
2. **Run new pipeline** targeting `data_pipeline_output/` instead
3. **Compare outputs** - Verify new pipeline matches old results
4. **Archive morphseq_playground** - Rename to `morphseq_playground.old` when validated
5. **Deprecate references** - Update docs/scripts to point to `data_pipeline_output/`

### Backwards Compatibility

If needed to reference old `morphseq_playground` outputs:
```bash
# Symlink for compatibility
ln -s /path/to/morphseq_playground \
      data_pipeline_output/legacy_outputs

# Config override for comparison
snakemake --config data_root=data_pipeline_output/legacy_outputs
```

---

## Summary: Answering All 5 Questions

| Question | Answer | Status |
|----------|--------|--------|
| **1. Test Data Access?** | Network mount at `/net/trapnell/.../morphseq/`; use 20240418 (YX1) and 20240509_24hpf (Keyence) | ✅ |
| **2. Snakefile Priority?** | Create test data structure via symlinks FIRST, then Snakefile | ✅ |
| **3. Testing Strategy?** | Option 1 with symlinks: structure → Snakefile → extract → validate incremental | ✅ |
| **4. MVP Stubs OK?** | Yes, UNet and embeddings can be stubbed; SAM2 and QC are required | ✅ |
| **5. Config Management?** | Centralized `data_pipeline_output/` with symlinks; configurable `data_root` path in Snakemake | ✅ |

---

## Next Steps

1. **Create data_pipeline_output/** directory structure with symlinks
2. **Update test data extraction scripts** to use available experiments
3. **Create Snakefile** with Phase 1-8 rules (use above patterns)
4. **Extract test subset** (10 frames per well)
5. **Run Phase 1-2** validation
6. **Run full pipeline** on test data
7. **Archive morphseq_playground** once validated

---

**Document created:** 2025-11-06
**Ready for:** Snakefile creation and test data extraction
