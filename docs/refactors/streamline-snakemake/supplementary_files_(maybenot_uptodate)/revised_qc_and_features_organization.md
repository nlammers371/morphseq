# Revised QC & Features Organization with Consolidation

**Date:** 2025-10-09
**Key Insights:**
1. Features should be computed **from masks** (not snips)
2. Snips are independent (can generate as soon as masks exist)
3. Need QC consolidation + `use_embryo` flag determination
4. File structure should reflect: masks → features → QC

---

## Conceptual Data Flow

```
MASKS (SAM2 + UNet)
    ↓
    ├──→ SNIPS (extraction, rotation, augmentation)  ← Independent branch
    │
    └──→ FEATURE EXTRACTION (computed from masks)
            ↓
            ├─→ Spatial features (centroid, orientation)
            ├─→ Shape features (area, perimeter, contours, curvature)
            └─→ Stage inference (HPF prediction)
                    ↓
                    QC FLAGS (imaging, viability, segmentation, tracking, size)
                        ↓
                        QC CONSOLIDATION (merge all flags)
                            ↓
                            USE_EMBRYO FLAG (filter logic)
```

---

## Revised Directory Structure

```
src/data_pipeline/

├── snip_processing/                    # INDEPENDENT: masks → snips
│   ├── extraction.py                   # Crop embryo regions
│   ├── rotation.py                     # PCA-based alignment
│   ├── augmentation.py                 # Synthetic noise
│   └── io.py                           # Save snip images
│
├── feature_extraction/                 # NEW: Mask-based features
│   ├── spatial.py                      # Centroid, orientation
│   ├── shape.py                        # Area, perimeter, contours, curvature
│   └── stage_inference.py              # HPF prediction from surface area
│
├── quality_control/
│   ├── auxiliary_mask_qc/              # UNet-dependent QC
│   │   ├── imaging_quality.py          # frame, yolk, focus, bubble
│   │   └── embryo_death_qc.py          # fraction_alive, dead_flag, inflection metadata
│   │
│   ├── segmentation_qc/                # SAM2-only QC
│   │   ├── mask_quality.py             # SAM2 validation (7 checks)
│   │   └── tracking_quality.py         # speed, trajectory, errors
│   │
│   ├── morphology_qc/                  # Feature-based QC
│   │   └── size_validation.py          # SA outlier (two-sided)
│   │
│   ├── consolidation.py                # NEW: Merge all QC flags
│   └── use_embryo_filter.py            # NEW: Determine use_embryo flag
│
└── references/
    └── build_sa_reference.py           # SA reference curve generation
```

---

## Data Output Structure

```
processed_images/stitched_FF/{experiment_id}/
    └── {image_id}.jpg

segmentation/
    ├── embryo_tracking/{experiment_id}/
    │   ├── initial_detections.json
    │   ├── propagated_masks.json
    │   ├── tracking_table.csv          # Flattened SAM2 output
    │   └── masks/{video_id}/{image_id}_embryo_{N}.png
    │
    └── auxiliary_masks/{experiment_id}/
        ├── viability/
        ├── yolk/
        ├── focus/
        └── bubbles/

extracted_snips/{experiment_id}/        # INDEPENDENT BRANCH
    ├── snip_manifest.csv
    └── {snip_id}.jpg

computed_features/{experiment_id}/      # NEW: Mask-based features
    ├── spatial.csv                     # Centroid, bbox, orientation
    ├── shape.csv                       # Area, perimeter, contours, curvature
    └── developmental_stage.csv         # Predicted HPF

quality_control_flags/{experiment_id}/
    ├── imaging_quality.csv             # frame, yolk, focus, bubble
    ├── embryo_death_qc.csv             # fraction_alive, dead_flag, dead_inflection_time_int, death_predicted_stage_hpf
    ├── segmentation_quality.csv        # 7 SAM2 validation flags
    ├── tracking_quality.csv            # speed, trajectory, tracking_error_flag
    ├── size_validation.csv             # sa_outlier_flag
    ├── consolidated_qc.csv             # NEW: All QC flags merged
    └── use_embryo_flags.csv            # NEW: Final filter decisions

latent_embeddings/{model_name}/
    └── {experiment_id}_latents.csv     # Uses use_embryo_flags.csv
```

---

## Detailed Module Specifications

### 1. `feature_extraction/spatial.py`

**Purpose:** Compute spatial features directly from SAM2 masks

**Input:**
- SAM2 masks (PNG or RLE from tracking_table.csv)
- Image dimensions

**Output:** `computed_features/{experiment_id}/spatial.csv`
```
Columns:
    - snip_id
    - centroid_x, centroid_y         # Mask centroid (pixels)
    - centroid_x_um, centroid_y_um   # Mask centroid (microns)
    - orientation_angle              # PCA-based orientation (degrees)
```

**Functions:**
```python
def compute_spatial_features(mask: np.ndarray, px_dim_um: float) -> dict:
    """Compute centroid and orientation from binary mask"""
```

**Note:** This is pure geometric computation from masks, no image needed. Bounding box is in tracking_table.csv already.

---

### 2. `feature_extraction/shape.py`

**Purpose:** Compute shape features directly from SAM2 masks

**Input:**
- SAM2 masks (PNG or RLE)
- Pixel dimensions

**Output:** `computed_features/{experiment_id}/shape.csv`
```
Columns:
    - snip_id
    - area_px, area_um2              # Mask area
    - perimeter_px, perimeter_um     # Contour perimeter
    - aspect_ratio                   # Length/width ratio
    - circularity                    # 4π×area / perimeter²
    - contour_complexity             # Perimeter² / area
    - curvature_stats                # FUTURE: Mean/max curvature from contour
```

**Functions:**
```python
def compute_shape_features(mask: np.ndarray, px_dim_um: float) -> dict:
    """
    Compute area, perimeter, circularity, curvature from binary mask.

    Future extensions:
        - compute_curvature(): Analyze contour curvature at each point
        - compute_texture(): Analyze mask boundary texture
        - compute_convexity(): Measure deviation from convex hull
    """
```

**Note:** All computed from mask geometry, extensible for future metrics

---

### 3. `feature_extraction/stage_inference.py`

**Purpose:** Predict developmental stage (HPF) from surface area

**Input:**
- `computed_features/{experiment_id}/shape.csv` (needs area_um2)
- Reference curves (metadata/stage_ref_df.csv)

**Output:** `computed_features/{experiment_id}/developmental_stage.csv`
```
Columns:
    - snip_id
    - predicted_stage_hpf
    - stage_confidence
```

**Functions:**
```python
def infer_hpf_stage(surface_area_um: float, reference_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Predict HPF stage from surface area using reference curves.

    Args:
        surface_area_um: Embryo surface area in µm²
        reference_df: Stage reference with (stage_hpf, mean_area, p5, p95)

    Returns:
        (predicted_hpf, confidence)
    """
```

**Note:** Uses shape features (not raw masks)

---

### 4. `quality_control/consolidation.py` ⭐ NEW

**Purpose:** Merge all QC flags into single table for easy access

**Input:**
```
quality_control_flags/{experiment_id}/
    ├── imaging_quality.csv
    ├── embryo_death_qc.csv
    ├── segmentation_quality.csv
    ├── tracking_quality.csv
    └── size_validation.csv
```

**Output:** `quality_control_flags/{experiment_id}/consolidated_qc.csv`
```
Columns:
    - snip_id
    - embryo_id
    - time_int

    # Imaging quality (4 flags)
    - frame_flag
    - no_yolk_flag
    - focus_flag
    - bubble_flag

    # Embryo death QC (4 columns)
    - fraction_alive
    - dead_flag
    - dead_inflection_time_int
    - death_predicted_stage_hpf

    # Segmentation quality (7 flags)
    - HIGH_SEGMENTATION_VAR_SNIP
    - MASK_ON_EDGE
    - DETECTION_FAILURE
    - OVERLAPPING_MASKS
    - LARGE_MASK
    - SMALL_MASK
    - DISCONTINUOUS_MASK

    # Tracking quality (3 columns)
    - speed_um_per_s
    - trajectory_smoothness_score
    - tracking_error_flag

    # Size validation (1 flag)
    - sa_outlier_flag
```

**Functions:**
```python
def consolidate_qc_flags(
    qc_dir: Path,
    experiment_id: str,
) -> pd.DataFrame:
    """
    Load all QC CSV files and merge on snip_id.

    For now: ALL QC files are REQUIRED (no graceful skipping).
    If any QC file is missing, raise error and fail pipeline.

    This ensures complete QC coverage - no partial results.

    Future: Could add optional QC with config flags if needed.
    """
```

**Note:** Lightweight merge operation, no computation

---

### 5. `quality_control/use_embryo_filter.py` ⭐ NEW

**Purpose:** Determine `use_embryo` flag from consolidated QC flags

**Input:**
- `quality_control_flags/{experiment_id}/consolidated_qc.csv`
- Filter configuration (default flags to check)

**Output:** `quality_control_flags/{experiment_id}/use_embryo_flags.csv`
```
Columns:
    - snip_id
    - use_embryo                    # Boolean: pass all filters?
    - exclusion_reasons             # Comma-separated failed flags (for debugging)

Note: This is the ONLY file embeddings needs - just valid snip_ids per experiment
```

**Functions:**
```python
def compute_use_embryo_flag(
    consolidated_qc_df: pd.DataFrame,
    default_filter_flags: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Lightweight filter: check if any critical flags are True.

    Default filter flags (configurable):
        - dead_flag
        - sa_outlier_flag
        - DETECTION_FAILURE
        - OVERLAPPING_MASKS
        - tracking_error_flag
        - frame_flag
        - no_yolk_flag
        - bubble_flag

    use_embryo = NOT (any critical flag is True)

    Returns DataFrame with:
        - snip_id
        - use_embryo (bool)
        - exclusion_reasons (comma-separated string of failed flags)
    """
```

**Configuration:**
```python
# Default in config/defaults.yaml
use_embryo_filter:
  critical_flags:
    - dead_flag
    - sa_outlier_flag
    - DETECTION_FAILURE
    - OVERLAPPING_MASKS
    - tracking_error_flag

  optional_flags:  # Can enable/disable per experiment
    - frame_flag
    - no_yolk_flag
    - bubble_flag
    - MASK_ON_EDGE
    - focus_flag
```

**Note:** Very lightweight, just boolean logic on existing flags

---

## Revised Pipeline Steps

### STEP 1: RAW DATA → STANDARDIZED IMAGES
```
preprocess_keyence / preprocess_yx1
```

### STEP 2: STANDARDIZED IMAGES → SEGMENTATION MASKS
```
SAM2 Pipeline:
    gdino_detect → sam2_segment_and_track → sam2_format_csv → sam2_export_masks

UNet Pipeline:
    unet_segment (viability, yolk, focus, bubble)
```

### STEP 3A: MASKS → SNIPS (Independent)
```
extract_snips (uses SAM2 masks only)
```

### STEP 3B: MASKS → FEATURE EXTRACTION (Independent)
```
compute_spatial_features  (SAM2 masks → spatial.csv)
    ↓
compute_shape_features    (SAM2 masks → shape.csv)
    ↓
infer_embryo_stage        (shape.csv → developmental_stage.csv)
```

### STEP 4: FEATURES + MASKS → QC FLAGS (All parallel)
```
qc_imaging          (UNet masks + SAM2 tracking)
qc_viability        (UNet viability + SAM2 masks + stage)
qc_segmentation     (SAM2 JSON + tracking)
qc_tracking         (SAM2 tracking + spatial features)
qc_size             (shape features + stage)
```

### STEP 5: QC CONSOLIDATION
```
consolidate_qc      (merge all QC CSVs)
    ↓
compute_use_embryo  (apply filter logic)
```

### STEP 6: QC-GATED EMBEDDINGS
```
generate_embeddings (snips + use_embryo_flags.csv)
```

---

## Snakemake Rule Changes

### Feature Extraction Rules (replace old snip-based features):

```python
rule compute_spatial_features:
    input:
        masks=segmentation/embryo_tracking/{experiment_id}/masks/,
        tracking_csv=segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
    output:
        computed_features/{experiment_id}/spatial.csv
    run:
        from data_pipeline.feature_extraction.spatial import compute_spatial_features

rule compute_shape_features:
    input:
        masks=segmentation/embryo_tracking/{experiment_id}/masks/,
        tracking_csv=segmentation/embryo_tracking/{experiment_id}/tracking_table.csv
    output:
        computed_features/{experiment_id}/shape.csv
    run:
        from data_pipeline.feature_extraction.shape import compute_shape_features

rule infer_embryo_stage:
    input:
        shape=computed_features/{experiment_id}/shape.csv,
        reference=inputs/reference_data/surface_area_ref.csv
    output:
        computed_features/{experiment_id}/developmental_stage.csv
    run:
        from data_pipeline.feature_extraction.stage_inference import infer_hpf_stage
```

### QC Consolidation Rules (NEW):

```python
rule consolidate_qc:
    input:
        imaging=quality_control_flags/{experiment_id}/imaging_quality.csv,
        embryo_death=quality_control_flags/{experiment_id}/embryo_death_qc.csv,
        segmentation=quality_control_flags/{experiment_id}/segmentation_quality.csv,
        tracking=quality_control_flags/{experiment_id}/tracking_quality.csv,
        size=quality_control_flags/{experiment_id}/size_validation.csv
    output:
        quality_control_flags/{experiment_id}/consolidated_qc.csv
    run:
        from data_pipeline.quality_control.consolidation import consolidate_qc_flags

rule compute_use_embryo:
    input:
        consolidated_qc=quality_control_flags/{experiment_id}/consolidated_qc.csv
    output:
        quality_control_flags/{experiment_id}/use_embryo_flags.csv
    run:
        from data_pipeline.quality_control.use_embryo_filter import compute_use_embryo_flag
```

### Embeddings Rule (updated dependency):

```python
rule generate_embeddings:
    input:
        snips=extracted_snips/{experiment_id}/,
        manifest=extracted_snips/{experiment_id}/snip_manifest.csv,
        use_embryo=quality_control_flags/{experiment_id}/use_embryo_flags.csv
    output:
        latent_embeddings/{model_name}/{experiment_id}_latents.csv
    run:
        from data_pipeline.embeddings.inference import ensure_embeddings
```

---

## Key Advantages of This Structure

### ✅ Conceptual Clarity:
- **Masks → Morphology Features** is explicit
- **Snips are independent** (can run as soon as masks exist)
- **QC consolidation** is a clear distinct step

### ✅ Extensibility:
```python
# Adding new feature (e.g., curvature):
# Just add to feature_extraction/shape.py::compute_shape_features()
# No need to touch snip processing or QC

# Adding new QC check:
# Add new QC module
# Add to consolidation.py input list
# Add to use_embryo_filter.py config (if critical)
```

### ✅ Parallel Execution:
```python
# Can run simultaneously (no dependencies):
- extract_snips
- compute_spatial_features
- compute_shape_features

# Then these can run in parallel:
- All 5 QC modules (imaging, viability, segmentation, tracking, size)

# Then sequential:
- consolidate_qc
- compute_use_embryo
```

### ✅ Flexible Filtering:
```yaml
# Can easily adjust filter criteria per experiment/analysis
# via config file without changing code

# Example: Strict filtering
critical_flags:
  - dead_flag
  - sa_outlier_flag
  - DETECTION_FAILURE
  - frame_flag
  - bubble_flag
  - no_yolk_flag

# Example: Permissive filtering (exploratory analysis)
critical_flags:
  - dead_flag
  - sa_outlier_flag
```

### ✅ Debugging Friendly:
- `exclusion_reasons` column shows exactly why each snip was filtered
- All QC flags preserved in `consolidated_qc.csv` for analysis
- Can re-run `compute_use_embryo` with different criteria without recomputing QC

---

## Summary of Changes from Previous Plan

### Added:
1. **`feature_extraction/` package** - Mask-based feature computation
   - `spatial.py`, `shape.py`, `stage_inference.py`
2. **`quality_control/consolidation.py`** - Merge all QC flags
3. **`quality_control/use_embryo_filter.py`** - Determine filtering

### Moved:
- Features OUT of `snip_processing/embryo_features/`
- Now computed from masks, not snips
- Snip processing is purely image cropping/rotation

### Clarified:
- Snips are **independent** (masks → snips, done)
- Features are **mask-based** (masks → features → QC)
- QC consolidation is **explicit step** (not buried in embeddings)
- `use_embryo` determination is **configurable** (default list of critical flags)
- Bbox already in tracking_table.csv, not duplicated in spatial.csv

---

## Next Steps

1. ✅ Review this revised structure
2. Create `feature_extraction/` package
3. Create `consolidation.py` and `use_embryo_filter.py`
4. Update Snakemake rules document with new DAG
5. Begin implementation
