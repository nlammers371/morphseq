# Snip ID Creation & Tracking Table Schema

**Date:** 2025-10-09
**Key Question:** Is `snip_id` created by SAM2 pipeline and included in tracking_table.csv?

---

## Answer: YES ✅

**`snip_id` IS created by SAM2 pipeline and IS in tracking_table.csv**

---

## How snip_id is Created

### Creation Logic:
```python
# From: segmentation_sandbox/scripts/detection_segmentation/sam2_utils.py:953

def create_snip_id(embryo_id: str, image_id: str) -> str:
    """Create snip_id via parsing_utils using canonical format"""
    frame = extract_frame_number(image_id)  # Extract from image_id
    snip_id = build_snip_id(embryo_id, frame)
    return snip_id

# From: segmentation_sandbox/scripts/utils/parsing_utils.py:368
def build_snip_id(embryo_id: str, frame_number: int) -> str:
    """Create snip ID: embryo_id + _s + 4-digit frame"""
    return f"{embryo_id}_s{frame_number:04d}"
```

### Format:
```
snip_id = {embryo_id}_s{frame_number:04d}

Examples:
    20240915_A01_BF_e01_s0023
    20240915_A01_GFP_e02_s0100

Components:
    - embryo_id: 20240915_A01_e0001
    - separator: _s
    - frame_number: 0023 (4 digits, zero-padded)
```

### Key Points:
- **Uses `_s` separator** (not just underscore with frame number)
- **4-digit zero-padded frame** (s0023, not s23)
- **Created during SAM2 processing** (not later in Build03)
- **Stored in tracking_table.csv** (primary key for downstream analysis)

---

## Tracking Table CSV Schema

**File:** `segmentation/embryo_tracking/{experiment_id}/tracking_table.csv`

**Source:** `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py:112-134`

### Complete Schema (43 columns):

```python
# Core SAM2 columns (14):
'image_id'              # e.g., "20240915_A01_t0023"
'embryo_id'             # e.g., "20240915_A01_e0001"
'snip_id'               # e.g., "20240915_A01_e0001_s0023" ✅ CREATED HERE
'frame_index'           # Integer frame number (same as _s suffix)
'area_px'               # Mask area in pixels
'bbox_x_min'            # Bounding box coordinates
'bbox_y_min'
'bbox_x_max'
'bbox_y_max'
'mask_confidence'       # SAM2 confidence score
'exported_mask_path'    # Path to PNG mask file
'experiment_id'         # e.g., "20240915"
'video_id'              # Well identifier, e.g., "20240915_A01"
'is_seed_frame'         # Boolean: was this a GroundingDINO seed frame?

# Raw image metadata (16):
'Height (um)', 'Height (px)', 'Width (um)', 'Width (px)',
'BF Channel', 'Objective', 'Time (s)', 'Time Rel (s)',
'height_um', 'height_px', 'width_um', 'width_px',
'bf_channel', 'objective', 'raw_time_s', 'relative_time_s',
'microscope', 'nd2_series_num'

# Well-level metadata (7):
'medium', 'genotype', 'chem_perturbation', 'start_age_hpf',
'embryos_per_well', 'temperature', 'well_qc_flag'

# Build03 compatibility columns (3):
'well'                  # Well name (e.g., "A01")
'time_int'              # Integer timepoint
'time_string'           # Formatted time string

# SAM2 QC flags (1):
'sam2_qc_flags'         # QC flags from SAM2 processing
```

---

## Critical Implications for Pipeline

### ✅ snip_id is ALREADY created by sam2_format_csv

**This means:**

1. **No need to create snip_id later** - it's in tracking_table.csv from the start

2. **All downstream modules use existing snip_id:**
   - extract_snips: reads snip_id from tracking_table.csv
   - compute_pose_kinematics_metrics: reads snip_id from tracking_table.csv
   - compute_mask_geometry_metrics: reads snip_id from tracking_table.csv
   - ALL QC modules: use snip_id from tracking_table.csv

3. **tracking_table.csv is the SOURCE OF TRUTH** for:
   - snip_id (unique identifier per embryo×frame)
   - image_id → embryo_id → snip_id mapping
   - bbox coordinates (already computed!)
   - area_px (already computed!)
   - All metadata needed for downstream processing

---

## Revised Understanding: What Gets Computed Where

### Created by SAM2 pipeline (tracking_table.csv):
```
✅ snip_id                   # Generated from embryo_id + frame_number
✅ image_id, embryo_id       # Parsed from SAM2 output
✅ bbox_x/y_min/max          # From SAM2 masks
✅ area_px                   # From SAM2 masks
✅ frame_index               # Frame number
✅ time_int                  # Integer timepoint
✅ All metadata              # From raw data + well metadata
```

### Computed by feature_extraction (NEW features only):
```
❓ centroid_x, centroid_y   # From SAM2 masks (not in tracking_table.csv)
❓ orientation_angle        # From PCA on mask (not in tracking_table.csv)
❓ perimeter                # From mask contour (not in tracking_table.csv)
❓ circularity              # From area + perimeter (not in tracking_table.csv)
❓ curvature                # From mask contour (FUTURE)
```

### Key Insight:
**tracking_table.csv already has bbox + area!**
- Don't duplicate computation
- feature_extraction adds NEW metrics only
- Centroids, orientation, perimeter, etc.

---

## Recommended Data Flow

### Step 1: SAM2 creates tracking_table.csv
```
Columns include:
    - snip_id (PRIMARY KEY) ✅
    - image_id, embryo_id, frame_index
    - bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max ✅
    - area_px ✅
    - All metadata
```

### Step 2: feature_extraction ADDS new columns
```
Approach: Augment tracking_table.csv with NEW features

spatial.csv:
    - snip_id
    - centroid_x, centroid_y
    - centroid_x_um, centroid_y_um
    - orientation_angle

shape.csv:
    - snip_id
    - area_px (ALREADY in tracking_table.csv - just use it!)
    - area_um2 (scaled from area_px)
    - perimeter_px, perimeter_um
    - circularity
    - aspect_ratio
    - curvature_stats (FUTURE)
```

### Step 3: Merge for downstream use
```
consolidated_snip_features.csv = tracking_table.csv + spatial.csv + shape.csv
    - Merged on snip_id
    - Contains ALL features + metadata
    - Used by QC modules
```

---

## Alternative: Should We Merge Immediately?

### Option A: Separate feature CSVs (current plan)
```
tracking_table.csv       # SAM2 output
spatial.csv              # New spatial features
shape.csv                # New shape features
```
**Pros:** Clear separation of what SAM2 provides vs what we compute
**Cons:** Multiple files to manage, need joins in QC modules

### Option B: Augment tracking_table.csv directly
```
tracking_table_augmented.csv = tracking_table.csv + new features
```
**Pros:** Single source of truth, simpler for QC modules
**Cons:** Modifies SAM2 output, harder to debug

### Option C: Create consolidated_snip_features.csv (RECOMMENDED)
```
computed_features/{experiment_id}/consolidated_snip_features.csv
    = tracking_table.csv + spatial.csv + shape.csv + stage.csv
```
**Pros:**
- Keeps SAM2 output pristine
- Single file for downstream (QC modules)
- Clear provenance (original + computed)

**Cons:**
- One extra file
- Slight duplication (area_px in both tracking_table and consolidated)

---

## Recommendation: Option C

### Proposed Structure:
```
segmentation/embryo_tracking/{experiment_id}/
    ├── tracking_table.csv              # SAM2 output (pristine)
    └── ...

computed_features/{experiment_id}/
    ├── spatial.csv                     # New spatial features only
    ├── shape.csv                       # New shape features only
    ├── developmental_stage.csv         # Stage predictions
    ├── consolidated_snip_features.csv       # ALL snip-level features merged ⭐
    └── use_embryo_flags.csv             # QC-derived keep/drop decisions

analysis_ready/{experiment_id}/
    ├── snip_features_qc.csv            # consolidated_snip_features + QC context
    └── features_qc_embeddings.csv      # snip_features_qc + embeddings + embedding_available flag ⭐
```

### consolidated_snip_features.csv contains:
```
From tracking_table.csv:
    - snip_id, image_id, embryo_id, frame_index
    - bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max
    - area_px
    - time_int
    - All metadata

From spatial.csv:
    - centroid_x, centroid_y, centroid_x_um, centroid_y_um
    - orientation_angle

From shape.csv:
    - area_um2, perimeter_px, perimeter_um
    - circularity, aspect_ratio
    - (curvature_stats in future)

From developmental_stage.csv:
    - predicted_stage_hpf
    - stage_confidence
```

### Snip-level QC & embeddings layering
- QC modules read consolidated_snip_features.csv and emit `use_embryo_flags.csv`
- `snip_features_qc.csv` = consolidated_snip_features + QC flags + keep/drop rationale
- Embedding jobs append columns to snip_features_qc.csv and set `embedding_available`
- Final `features_qc_embeddings.csv` feeds analysis notebooks, stays analysis-ready

---

## Updated Snakemake Rules

### Add consolidation rule:
```python
rule consolidate_snip_features:
    input:
        tracking=segmentation/embryo_tracking/{experiment_id}/tracking_table.csv,
        spatial=computed_features/{experiment_id}/spatial.csv,
        shape=computed_features/{experiment_id}/shape.csv,
        stage=computed_features/{experiment_id}/developmental_stage.csv
    output:
        computed_features/{experiment_id}/consolidated_snip_features.csv
    run:
        from data_pipeline.feature_extraction.consolidation import consolidate_snip_features

rule consolidate_snip_qc:
    input:
        features=computed_features/{experiment_id}/consolidated_snip_features.csv,
        qc_flags=quality_control/{experiment_id}/consolidated_qc.csv,
        use_flags=quality_control/{experiment_id}/use_embryo_flags.csv
    output:
        analysis_ready/{experiment_id}/snip_features_qc.csv
    run:
        from data_pipeline.quality_control.consolidation import merge_snip_features_and_qc

rule attach_snip_embeddings:
    input:
        features=analysis_ready/{experiment_id}/snip_features_qc.csv,
        embeddings=embeddings/{experiment_id}/snip_embeddings.parquet
    output:
        analysis_ready/{experiment_id}/features_qc_embeddings.csv
    run:
        from data_pipeline.embeddings.consolidation import append_embeddings_with_flag
```

### QC modules use consolidated_snip_features.csv:
```python
rule qc_tracking:
    input:
        features=computed_features/{experiment_id}/consolidated_snip_features.csv  # Single file!
    output:
        quality_control_flags/{experiment_id}/tracking_quality.csv
    run:
        from data_pipeline.quality_control.segmentation_qc.tracking_quality import compute_tracking_qc
```

---

## Summary

### ✅ Key Findings:
1. **snip_id IS created by SAM2** - in tracking_table.csv
2. **bbox + area_px already computed** - don't duplicate
3. **tracking_table.csv is comprehensive** - 43 columns of metadata

### ✅ Recommended Changes:
1. Create **consolidated_snip_features.csv** (merge tracking + computed features)
2. QC modules read **single file** (simpler, faster)
3. Keep **tracking_table.csv pristine** (SAM2 output unchanged)
4. feature_extraction computes **NEW metrics only** (centroid, orientation, perimeter, etc.)

### ✅ Updated Pipeline Flow:
```
sam2_format_csv → tracking_table.csv (includes snip_id, bbox, area_px)
    ↓
compute_pose_kinematics_metrics → pose_kinematics_metrics.csv (NEW: centroid, orientation, motion deltas)
compute_mask_geometry_metrics → mask_geometry_metrics.csv (NEW: perimeter, circularity)
infer_embryo_stage → stage.csv
    ↓
consolidate_snip_features → consolidated_snip_features.csv ⭐
    ↓
QC modules → consolidated_qc.csv + use_embryo_flags.csv
    ↓
consolidate_snip_qc → snip_features_qc.csv
    ↓
attach_snip_embeddings → features_qc_embeddings.csv (embedding_available column)
```
