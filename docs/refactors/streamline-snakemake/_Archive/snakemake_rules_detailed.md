# MorphSeq Pipeline: Snakemake Rules Specification

**Status:** APPROVED - Option 1 Viability Architecture
**Date:** 2025-10-09

---

## Design Principles

1. **Clear Step Boundaries:**
   - Step 1-3: Generate **data** (images, masks, features)
   - Step 4: Generate **metadata** (QC flags)
   - Step 5: QC-gated analysis (embeddings)

2. **Minimal Rules:**
   - One rule per logical operation
   - Clear input/output dependencies
   - No redundant intermediate files

3. **Unified Viability Handling:**
   - Single `compute_embryo_death_qc` rule computes `fraction_alive` + `dead_flag`
   - No separate `dead_flag` vs `dead_flag2` (unified robust flag)
   - All death detection logic in one module

---

## Pipeline Rule DAG

```
STEP 1: RAW DATA → STANDARDIZED IMAGES
├─ preprocess_keyence
└─ preprocess_yx1

STEP 2: STANDARDIZED IMAGES → SEGMENTATION MASKS
├─ SAM2 Pipeline (primary embryo tracking)
│  ├─ gdino_detect
│  ├─ sam2_segment_and_track
│  ├─ sam2_format_csv          ← NEW (was missing!)
│  └─ sam2_export_masks
└─ UNet Pipeline (auxiliary masks for QC)
   └─ unet_segment

STEP 3: MASKS + IMAGES → SNIPS + FEATURES
├─ extract_snips
├─ compute_mask_geometry_metrics
├─ compute_pose_kinematics_metrics
└─ infer_embryo_stage

STEP 4: FEATURES + MASKS → QC FLAGS
├─ qc_imaging         (UNet yolk/focus/bubble + SAM2 tracking)
├─ qc_death_detection (UNet viability + SAM2 tracking) ← UNIFIED
├─ qc_tracking        (SAM2 tracking + pose/kinematics metrics)
├─ qc_segmentation    (SAM2 masks)
└─ qc_size            (mask geometry + stage)

STEP 5: QC-GATED EMBEDDINGS
└─ generate_embeddings (snips + all QC flags)
```

---

## Step 1: Raw Data → Standardized Images

### `rule preprocess_keyence`

**Purpose:** Convert raw Keyence BZ-X800 microscope data to standardized FF images

**Input:**
```
inputs/raw_image_data/Keyence/{experiment_id}/
```

**Output:**
```
processed_images/stitched_FF/{experiment_id}/
    ├── {well_id}_t{timepoint:04d}.jpg
    └── ...
processed_images/stitched_FF/preprocessing_logs/{experiment_id}_preprocessing.csv
```

**Run:**
```python
from data_pipeline.preprocessing.keyence import stitch_images, extract_metadata
```

**Notes:**
- Handles tile stitching, z-stacking
- Extracts microscope metadata
- Outputs standardized 2D FF images

---

### `rule preprocess_yx1`

**Purpose:** Convert raw YX1 microscope data to standardized FF images

**Input:**
```
inputs/raw_image_data/YX1/{experiment_id}/
```

**Output:**
```
processed_images/stitched_FF/{experiment_id}/
    ├── {well_id}_t{timepoint:04d}.jpg
    └── ...
processed_images/stitched_FF/preprocessing_logs/{experiment_id}_preprocessing.csv
```

**Run:**
```python
from data_pipeline.preprocessing.yx1 import process_images
```

---

## Step 2: Standardized Images → Segmentation Masks

### SAM2 Pipeline (Primary Embryo Segmentation + Tracking)

#### `rule gdino_detect`

**Purpose:** Detect embryos in initial frame using Grounding DINO

**Input:**
```
processed_images/stitched_FF/{experiment_id}/
```

**Output:**
```
segmentation/embryo_tracking/{experiment_id}/initial_detections.json
```

**Run:**
```python
from data_pipeline.segmentation.grounded_sam2.gdino_detection import detect_embryos
```

**Notes:**
- Runs on first frame of each well
- Generates seed bounding boxes for SAM2

---

#### `rule sam2_segment_and_track`

**Purpose:** Propagate embryo masks across all frames using SAM2

**Input:**
```
images: processed_images/stitched_FF/{experiment_id}/
detections: segmentation/embryo_tracking/{experiment_id}/initial_detections.json
```

**Output:**
```
segmentation/embryo_tracking/{experiment_id}/propagated_masks.json
```

**Run:**
```python
from data_pipeline.segmentation.grounded_sam2.propagation import propagate_masks
```

**Notes:**
- Internally creates temporary video structure
- Runs SAM2 forward/bidirectional propagation
- Cleans up temp files
- Outputs nested JSON with RLE masks

---

#### `rule sam2_format_csv` ⚠️ NEW RULE

**Purpose:** Flatten SAM2 JSON output to tabular CSV format

**Input:**
```
segmentation/embryo_tracking/{experiment_id}/propagated_masks.json
```

**Output:**
```
segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
```

**Columns:**
```
- image_id           # e.g., "20240915_A01_ch00_t0023"
- embryo_id          # e.g., "20240915_A01_e0001"
- time_int           # Integer timepoint
- bbox_x, bbox_y, bbox_w, bbox_h
- area               # Mask area in pixels
- centroid_x, centroid_y
- rle_mask           # RLE encoded mask (for validation)
```

**Run:**
```python
from data_pipeline.segmentation.grounded_sam2.csv_formatter import flatten_to_csv
```

**Notes:**
- **CRITICAL:** This rule was missing from original mapping!
- Required by downstream rules (snip extraction, viability QC, tracking QC)
- Flattens nested JSON to flat table

---

#### `rule sam2_export_masks`

**Purpose:** Export integer-labeled PNG masks for visualization/validation

**Input:**
```
json: segmentation/embryo_tracking/{experiment_id}/propagated_masks.json
csv: segmentation/embryo_tracking/{experiment_id}/tracking_table.csv  # NEW dependency
```

**Output:**
```
segmentation/embryo_tracking/{experiment_id}/masks/{video_id}/
    ├── {image_id}_e01.png
    ├── {image_id}_e0N.png
    └── ...
```

**Run:**
```python
from data_pipeline.segmentation.grounded_sam2.mask_export import export_masks
```

**Notes:**
- Uses tracking_table.csv to map embryo IDs to masks
- Integer-labeled PNGs (1, 2, 3, ...)
- Used by snip extraction

---

### UNet Pipeline (Auxiliary Masks for QC)

#### `rule unet_segment`

**Purpose:** Generate auxiliary masks for QC using 5 UNet models

**Input:**
```
processed_images/stitched_FF/{experiment_id}/
```

**Output:**
```
segmentation/auxiliary_masks/{experiment_id}/
    ├── embryo/        # Validation mask (NOT primary segmentation)
    ├── viability/     # Dead/necrotic tissue regions
    ├── yolk/          # Yolk sac
    ├── focus/         # Out-of-focus regions
    └── bubbles/       # Air bubbles
```

**Run:**
```python
from data_pipeline.segmentation.unet.inference import run_all_models
```

**Notes:**
- All 5 models use same inference pipeline (different checkpoints)
- Embryo mask used for validation/comparison ONLY (SAM2 is primary)
- Viability mask detects necrotic tissue (1 = dead, 0 = alive)

---

## Step 3: Masks + Images → Snips + Features

### `rule extract_snips`

**Purpose:** Crop and align embryo regions from images

**Input:**
```
images: processed_images/stitched_FF/{experiment_id}/
sam2_masks: segmentation/embryo_tracking/{experiment_id}/masks/
tracking_csv: segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
```

**Output:**
```
extracted_snips/{experiment_id}/
    ├── snip_manifest.csv
    └── {snip_id}.jpg
```

**Run:**
```python
from data_pipeline.snip_processing.extraction import crop_embryos
```

**Notes:**
- Uses SAM2 masks (primary segmentation)
- PCA-based rotation alignment
- Generates unique snip_id per embryo+timepoint

---

### `rule compute_mask_geometry_metrics`

**Purpose:** Derive mask-intrinsic geometry metrics from SAM2 masks

**Input:**
```
tracking_csv: segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
sam2_masks: segmentation/embryo_tracking/{experiment_id}/masks/
```

**Output:**
```
computed_features/{experiment_id}/mask_geometry_metrics.csv
```

**Columns:**
```
- snip_id
- area_um2, area_px (both retained; μm² uses metadata pixel size)
- convex_hull_area_um2, convex_hull_area_px
- solidity, circularity, elongation
- contour_statistics (json or encoded)
```

**Run:**
```python
from data_pipeline.feature_extraction.mask_geometry_metrics import compute_mask_geometry_metrics
```

**Notes:**
- Consumes SAM2 masks directly; no dependence on JPEG snips
- Applies microscope pixel-size metadata to convert per-mask area to μm² (critical for biology-facing metrics)
- Focuses exclusively on geometry intrinsic to the mask footprint
- Provides the surface-area series (area_um2) used for stage inference and SA QC

---

### `rule compute_pose_kinematics_metrics`

**Purpose:** Capture embryo pose and frame-to-frame motion using tracking outputs

**Input:**
```
tracking_csv: segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
mask_geometry: computed_features/{experiment_id}/mask_geometry_metrics.csv
```

**Output:**
```
computed_features/{experiment_id}/pose_kinematics_metrics.csv
```

**Columns:**
```
- snip_id
- centroid_x, centroid_y (frame coordinates)
- bbox_center_x, bbox_center_y, bbox_width, bbox_height
- orientation_angle_rad
- delta_centroid_x, delta_centroid_y, speed_px_per_frame
```

**Run:**
```python
from data_pipeline.feature_extraction.pose_kinematics_metrics import compute_pose_kinematics_metrics
```

**Notes:**
- Operates on tracking metadata; no image reads required
- Feeds tracking QC (speed/outlier detection) and imaging QC (proximity flags)

---

### `rule infer_embryo_stage`

**Purpose:** Predict developmental stage (HPF) from mask geometry

**Input:**
```
mask_geometry: computed_features/{experiment_id}/mask_geometry_metrics.csv
```

**Output:**
```
computed_features/{experiment_id}/developmental_stage.csv
```

**Columns:**
```
- snip_id
- predicted_stage_hpf
- stage_confidence
```

**Run:**
```python
from data_pipeline.feature_extraction.stage_inference import infer_hpf_stage
```

**Notes:**
- Uses surface-area reference curves (area_um2 → HPF mapping; area_um2 derived from pixel metadata)
- Stage predictions support viability persistence buffer and SA QC (2 hr buffer logic)

---

## Step 4: Features + Masks → QC Flags

### `rule qc_imaging`

**Purpose:** Detect imaging quality issues using UNet auxiliary masks

**Input:**
```
tracking_csv: segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
unet_yolk: segmentation/auxiliary_masks/{experiment_id}/yolk/
unet_focus: segmentation/auxiliary_masks/{experiment_id}/focus/
unet_bubble: segmentation/auxiliary_masks/{experiment_id}/bubbles/
```

**Output:**
```
quality_control_flags/{experiment_id}/auxiliary_unet_imaging_quality.csv
```

**Columns:**
```
- snip_id
- frame_flag          # Embryo near image boundary
- no_yolk_flag        # Yolk sac missing (abnormal development)
- focus_flag          # Out-of-focus regions nearby
- bubble_flag         # Air bubbles nearby
```

**Run:**
```python
from data_pipeline.quality_control.auxiliary_unet_imaging_quality_qc import compute_qc_flags
```

**Notes:**
- Uses SAM2 tracking_table.csv for embryo locations
- Uses UNet auxiliary masks for spatial proximity analysis
- Does NOT use UNet embryo mask (SAM2 is ground truth)

---

### `rule compute_embryo_death_qc` ✨ UNIFIED MODULE

**Purpose:** Compute fraction_alive + persistent embryo death detection

**Input:**
```
tracking_csv: segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
sam2_masks: segmentation/embryo_tracking/{experiment_id}/masks/
viability_masks: segmentation/auxiliary_masks/{experiment_id}/viability/
stage_data: computed_features/{experiment_id}/developmental_stage.csv
```

**Output:**
```
quality_control_flags/{experiment_id}/embryo_death_qc.csv
```

**Columns:**
```
- snip_id
- embryo_id
- time_int
- fraction_alive             # Raw viability: 1 - (necrotic/total)
- dead_flag                  # Unified death flag (persistence + 2hr buffer)
- dead_inflection_time_int   # Frame index where persistent death is detected
- death_predicted_stage_hpf  # Predicted HPF at the inflection point
```

**Run:**
```python
from data_pipeline.quality_control.embryo_death_qc import compute_embryo_death_qc
```

**Algorithm:**
1. For each snip, load SAM2 embryo mask + UNet viability mask
2. Compute `fraction_alive = 1 - (dead_pixels / embryo_pixels)`
3. For each embryo, detect `fraction_alive` decline points
4. Validate persistence: ≥25% of post-inflection points must have simple threshold
5. If persistent, set `dead_inflection_time_int` for all snips of embryo
6. Capture `death_predicted_stage_hpf` from `predicted_stage_hpf` at inflection
7. Apply 2hr buffer using `predicted_stage_hpf`, set `dead_flag=True`

**Notes:**
- **CONSOLIDATES** legacy `dead_flag` (simple threshold) + `dead_flag2` (persistence)
- Single robust `dead_flag` output
- All viability logic in one module
- UNet viability mask shows necrotic tissue (1 = dead, 0 = alive)

---

### `rule qc_tracking`

**Purpose:** Validate embryo tracking quality

**Input:**
```
tracking_csv: segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
pose_metrics: computed_features/{experiment_id}/pose_kinematics_metrics.csv
```

**Output:**
```
quality_control_flags/{experiment_id}/tracking_metrics.csv
```

**Columns:**
- embryo_id
- time_int
- speed_px_per_frame          # Movement speed between frames
- trajectory_smoothness       # Savitzky-Golay smoothed trajectory
- tracking_error_flag         # Jumps, discontinuities detected
```

**Run:**
```python
from data_pipeline.quality_control.segmentation_qc.tracking_metrics_qc import compute_tracking_qc
```

**Notes:**
- QC for tracking results (NOT the tracking itself)
- Detects tracking errors: jumps, ID switches, discontinuities

---

### `rule qc_segmentation`

**Purpose:** Validate SAM2 mask quality

**Input:**
```
propagated_masks: segmentation/embryo_tracking/{experiment_id}/propagated_masks.json
tracking_csv: segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
```

**Output:**
```
quality_control_flags/{experiment_id}/segmentation_quality.csv
```

**Columns:**
```
- snip_id
- HIGH_SEGMENTATION_VAR_SNIP    # High area variance vs nearby frames (>20%)
- MASK_ON_EDGE                  # Mask touches image edges
- DETECTION_FAILURE             # Missing expected embryos
- OVERLAPPING_MASKS             # Embryo masks overlap (IoU > 0.1)
- LARGE_MASK                    # Unusually large (>15% frame area)
- SMALL_MASK                    # Unusually small (<0.1% frame area)
- DISCONTINUOUS_MASK            # Multiple disconnected components
```

**Run:**
```python
from data_pipeline.quality_control.segmentation_quality_qc import validate_masks
```

**Notes:**
- Validates SAM2 output quality
- Multiple granular flags for different issues

---

### `rule qc_size`

**Purpose:** Validate embryo sizes are biologically plausible

**Input:**
```
mask_geometry: computed_features/{experiment_id}/mask_geometry_metrics.csv
stage: computed_features/{experiment_id}/developmental_stage.csv
```

**Output:**
```
quality_control_flags/{experiment_id}/size_validation.csv
```

**Columns:**
```
- snip_id
- sa_outlier_flag    # Surface area outlier (abnormally large/small)
```

**Run:**
```python
from data_pipeline.quality_control.morphology_qc.size_validation_qc import validate_sizes
```

**Notes:**
- Operates on `area_um2` to evaluate biologically meaningful growth (pixel areas alone are insufficient)
- Compares to internal controls or stage reference
- One-sided detection (flags abnormally large only, by default)

---

## Step 5: QC-Gated Embeddings

### `rule generate_embeddings`

**Purpose:** Generate VAE latent embeddings for QC-passed snips

**Input:**
```
snips: extracted_snips/{experiment_id}/
manifest: extracted_snips/{experiment_id}/snip_manifest.csv
use_embryo_flags: quality_control_flags/{experiment_id}/use_embryo_flags.csv
```

**Output:**
```
latent_embeddings/{model_name}/{experiment_id}_latents.csv
```

**Columns:**
```
- snip_id
- z0, z1, z2, ..., z{dim-1}    # Latent dimensions
```

**Run:**
```python
from data_pipeline.embeddings.inference import ensure_embeddings
```

**Notes:**
- **ONLY** uses `use_embryo_flags.csv` (does NOT touch individual QC files)
- `use_embryo_flags.csv` contains: snip_id, use_embryo (bool)
- Simple filter: just list of valid snip_ids per experiment
- Uses Python 3.9 subprocess for legacy model compatibility
- Clean separation: embeddings doesn't need to know QC logic

---

## Summary of Key Changes from Original Mapping

### ✅ Added Rules:
1. **`sam2_format_csv`** - Critical missing rule to create tracking_table.csv

### ✅ Unified Rules:
2. **`compute_embryo_death_qc`** - Consolidates fraction_alive computation + death detection
   - Old: Split across `qc_utils.py` (Build03) + `death_detection.py` (Build04)
   - New: Single module, single output file

### ✅ Clarified Dependencies:
3. **All rules now show explicit input/output dependencies**
   - Clear which rules need tracking_table.csv
   - Clear which rules need UNet masks vs SAM2 masks
   - Clear which rules need features vs QC flags

### ✅ Output Consolidation:
4. **Unified death flag** - No more `dead_flag` vs `dead_flag2`
5. **Single death QC output** - `embryo_death_qc.csv` contains both raw metric + flag

---

## Data Flow Validation

### Critical Path: Raw → Embeddings
```
preprocess_keyence/yx1
    ↓
gdino_detect → sam2_segment_and_track → sam2_format_csv → sam2_export_masks
    ↓                                          ↓
unet_segment                         compute_mask_geometry_metrics
    ↓                                          ↓
compute_embryo_death_qc (uses UNet + SAM2)     compute_pose_kinematics_metrics
                                               ↓
                                       infer_embryo_stage
                                               ↓
extract_snips ─────────────────────────────────┴──→ ALL QC RULES
                                                   ↓
                                           generate_embeddings
```

### UNet Auxiliary Mask Usage:
```
unet_segment → {embryo, viability, yolk, focus, bubbles}
    │
    ├─ viability → compute_embryo_death_qc (fraction_alive + dead_flag)
    ├─ yolk      → qc_imaging (no_yolk_flag)
    ├─ focus     → qc_imaging (focus_flag)
    ├─ bubble    → qc_imaging (bubble_flag)
    └─ embryo    → (validation only, NOT used in pipeline)
```

---

## Next Steps

1. ✅ Review this rule specification
2. ⚠️ Implement `sam2_format_csv` rule (critical missing piece)
3. ⚠️ Create unified `embryo_death_qc.py` module
4. ⚠️ Update Snakefile with these rules
5. ⚠️ Test DAG dependency resolution
