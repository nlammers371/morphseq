# Data Validation Integration Plan (MVP - FINAL)

**Author:** Claude Code
**Date:** 2025-10-11
**Status:** PENDING APPROVAL

---

## Goal

Add schema-based validation at **8 consolidation points** to catch missing/malformed data early in the pipeline. Validation logic lives inline in consolidation functions—no separate validator module.

**Key principle:** ALL required columns are checked for both **existence** AND **non-empty** (no separate lists).

---

## What We're Creating

### 1. Schema Definitions (`src/data_pipeline/schemas/`)

Eight files with organized column lists:

#### `plate_metadata.py`
```python
REQUIRED_COLUMNS_PLATE_METADATA = [
    # Core identifiers
    'experiment_id',
    'well_id',
    'well_index',

    # Biological metadata
    'genotype',
    'treatment',              # or 'chem_perturbation'
    'embryos_per_well',
    'start_age_hpf',

    # Experimental conditions
    'temperature_c',          # Critical for developmental timing normalization (note plates currently have temperature so it'd need to be mapped)
    'medium',


]
```

#### `scope_metadata.py`
```python
REQUIRED_COLUMNS_SCOPE_METADATA = [
    # Core identifiers
    'experiment_id',
    'well_id',
    'well_index',
    'image_id',
    'time_int',

    # Spatial calibration (extracted from microscope)
    'micrometers_per_pixel',
    'image_width_px',
    'image_height_px',
    'objective_magnification',

    # Temporal calibration
    'frame_interval_s',
    'absolute_start_time',
    'experiment_time_s',

    # Acquisition metadata
    'microscope_id',
    'channel',
    'z_position',
    'frame_index',
]
```

#### `scope_and_plate_metadata.py`
```python
REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA = [
    # Core identifiers
    'experiment_id',
    'well_id',
    'well_index',
    'image_id',
    'time_int',
    'frame_index',
    'embryo_id',

    # From plate_metadata
    'genotype',
    'treatment',
    'temperature_c',
    'embryos_per_well',

    # From scope_metadata
    'micrometers_per_pixel',
    'frame_interval_s',
    'absolute_start_time',
    'image_width_px',
    'image_height_px',
]
```

#### `segmentation.py`
```python
REQUIRED_COLUMNS_SEGMENTATION_TRACKING = [
    # Core IDs
    'experiment_id',
    'video_id',
    'well_id',              # Well identifier for grouping
    'well_index',
    'image_id',
    'embryo_id',
    'snip_id',
    'frame_index',
    'time_int',

    # Mask data
    'mask_rle',             # Compressed mask as RLE string
    'area_px',              # Raw pixel area from SAM2
    'bbox_x_min',
    'bbox_y_min',
    'bbox_x_max',
    'bbox_y_max',
    'mask_confidence',

    # Geometry (will be converted to μm in features)
    'centroid_x_px',
    'centroid_y_px',

    # SAM2 metadata
    'is_seed_frame',        # Boolean - was this a SAM2 seed frame?

    # File references
    'source_image_path',    # Path to original stitched FF image
    'exported_mask_path',   # Path to exported PNG mask
]
```

#### `snip_processing.py`
```python
REQUIRED_COLUMNS_SNIP_MANIFEST = [
    # Core IDs
    'snip_id',
    'embryo_id',
    'experiment_id',
    'frame_index',
    'time_int',

    # File paths
    'source_image_path',    # Path to stitched FF image
    'cropped_snip_path',    # Path to extracted snip JPG

    # Extraction metadata
    'rotation_angle',       # PCA rotation applied (degrees)
    'crop_x_min',           # Crop bounding box
    'crop_y_min',
    'crop_x_max',
    'crop_y_max',
]
```

#### `features.py`
```python
REQUIRED_COLUMNS_FEATURES = [
    # Core IDs
    'snip_id',
    'embryo_id',
    'experiment_id',        # For cross-experiment analysis
    'well_id',              # Well identifier
    'frame_index',

    # Calibration (document what was used for conversions)
    'micrometers_per_pixel',
    'frame_interval_s',     # For velocity calculations

    # Geometry features (μm-based)
    'area_um2',             # Critical - must use μm², not pixels
    'perimeter_um',
    'centroid_x_um',
    'centroid_y_um',

    # Developmental stage
    'predicted_stage_hpf',  # Critical for QC and downstream analysis

    # Fraction alive detection (used to determine dead_flag in QC)
    'fraction_alive',

    # Pose/kinematics
    'orientation_angle',
    'bbox_width_um',
    'bbox_height_um',
]
```

#### `quality_control.py`
```python
REQUIRED_COLUMNS_QC = [
    # Core ID
    'snip_id',
    'embryo_id',
    'experiment_id',
    'time_int',

    # Gating flag
    'use_embryo',           # Final gate for embeddings/analysis

    # Viability QC
    'dead_flag',
    'fraction_alive',
    'death_inflection_time_int',  # When embryo died  
    'death_predicted_stage_hpf',        # Developmental stage at death(age  at death)

    # Imaging QC 
    'focus_flag', #(from_auxiliary masks)
    'bubble_flag', #(from_auxiliary masks)
    'yolk_flag', #(from_auxiliary masks)
    'edge_flag', 
    'discontinuous_mask_flag',
    'overlapping_mask_flag',
    #...etc check sam2 for snip_level flags and the restof codebase. 
    

    # Segmentation QC
    'mask_quality_flag',

    # Morphology QC
    'sa_outlier_flag',      # Surface area outlier
]

QC_FAIL_FLAGS = [
    'dead_flag',
    'sa_outlier_flag',
    'yolk_flag',
    'edge_flag',
    'discontinuous_mask_flag',
    'overlapping_mask_flag',
    'focus_flag',
    'bubble_flag',
]
```

#### `analysis_ready.py`
```python
REQUIRED_COLUMNS_ANALYSIS_READY = [
    # Core IDs (must be preserved from features)
    'snip_id',
    'embryo_id',
    'experiment_id',
    'time_int',
    'well_id',
    'well_index',

    # Embedding status
    'embedding_calculated',  # Boolean: True if embeddings present

    # Note: Embedding columns (z0...z{dim-1}) are optional and checked separately
] + REQUIRED_COLUMNS_FEATURES + REQUIRED_COLUMNS_QC + REQUIRED_COLUMNS_PLATE_METADATA + REQUIRED_COLUMNS_SCOPE_METADATA #we want these as well 
```

#### `embeddings.py`
```python
REQUIRED_COLUMNS_EMBEDDING_MANIFEST = [
    'snip_id',
    'processed_snip_path',
    'use_embryo_flag',
]

REQUIRED_COLUMNS_LATENTS = [
    'snip_id',
    'embedding_model',
    # Followed by z0 … z{dim-1}; validator checks expected dimensionality per model
]
```
---

### 2. Integration with Inline Validation

**Consistent validation pattern used everywhere:**

```python
# Step 1: Check required columns exist
missing = set(REQUIRED_COLUMNS) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {stage}: {sorted(missing)}")

# Step 2: Check ALL required columns are non-empty
for col in REQUIRED_COLUMNS:
    if df[col].isna().any():
        raise ValueError(f"Column '{col}' contains null/empty values in {stage}")
```

---

### 2a. Plate Metadata Consolidation (NEW)

**File:** `metadata/plate_processing.py`

```python
from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA

def process_plate_layout(input_excel: Path, experiment_id: str, output_csv: Path) -> None:
    """
    Load, normalize, validate, and write plate metadata.

    Handles various input formats (96-well, 24-well, different column names).
    """
    # Load and normalize
    df = pd.read_excel(input_excel)
    df = _normalize_plate_columns(df)  # Map "Well" → "well_id", etc.

    # Add experiment_id if not present
    if 'experiment_id' not in df.columns:
        df['experiment_id'] = experiment_id

    # Validate
    missing = set(REQUIRED_COLUMNS_PLATE_METADATA) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in plate_metadata: {sorted(missing)}")

    for col in REQUIRED_COLUMNS_PLATE_METADATA:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/empty values in plate_metadata")

    df.to_csv(output_csv, index=False)
```

---

### 2b. Scope Metadata Consolidation (NEW)

**File:** `preprocessing/keyence/extract_scope_metadata.py` (microscope-specific)

```python
from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA

def extract_keyence_scope_metadata(raw_image_dir: Path, output_csv: Path) -> None:
    """Extract scope metadata from Keyence ND2/BZ-X800 files."""

    # Extract from Keyence-specific formats
    df = _read_keyence_headers(raw_image_dir)

    # Validate
    missing = set(REQUIRED_COLUMNS_SCOPE_METADATA) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in scope_metadata: {sorted(missing)}")

    for col in REQUIRED_COLUMNS_SCOPE_METADATA:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/empty values in scope_metadata")

    df.to_csv(output_csv, index=False)
```

**File:** `preprocessing/yx1/extract_scope_metadata.py` (microscope-specific)

Same pattern, but YX1-specific extraction logic.

---

### 2c. Scope & Plate Metadata Consolidation (NEW)

**File:** `preprocessing/consolidate_plate_n_scope_metadata.py` (SHARED - microscope-agnostic)

```python
from data_pipeline.schemas.scope_and_plate_metadata import REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA

def consolidate_metadata(
    scope_metadata_csv: Path,   # Already validated (from any microscope)
    plate_metadata_csv: Path,   # Already validated
    output_csv: Path
) -> None:
    """
    Join pre-validated scope and plate metadata.

    Works for any microscope since inputs are already standardized.
    """
    scope_df = pd.read_csv(scope_metadata_csv)
    plate_df = pd.read_csv(plate_metadata_csv)

    # Standard join (microscope-agnostic)
    merged_df = scope_df.merge(
        plate_df,
        on=['experiment_id', 'well_id'],
        how='left',
        validate='many_to_one'
    )

    # Validate
    missing = set(REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA) - set(merged_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in scope_and_plate_metadata: {sorted(missing)}")

    for col in REQUIRED_COLUMNS_SCOPE_AND_PLATE_METADATA:
        if merged_df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/empty values in scope_and_plate_metadata")

    merged_df.to_csv(output_csv, index=False)
```

---

### 2d. Segmentation Consolidation

**File:** `segmentation/grounded_sam2/csv_formatter.py`

**Rename output:** `tracking_table.csv` → `segmentation_tracking.csv`

```python
from data_pipeline.schemas.segmentation import REQUIRED_COLUMNS_SEGMENTATION_TRACKING

def write_segmentation_tracking(df: pd.DataFrame, output_csv: Path) -> None:
    """Write segmentation tracking table with validation."""

    # Check required columns exist
    missing = set(REQUIRED_COLUMNS_SEGMENTATION_TRACKING) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in segmentation_tracking: {sorted(missing)}")

    # Check all required columns are non-empty
    for col in REQUIRED_COLUMNS_SEGMENTATION_TRACKING:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/empty values in segmentation_tracking")

    df.to_csv(output_csv, index=False)
```

**Updates to `_generate_csv_rows()` in same file:**
- Extract `mask_rle = seg_data.get('rle', '')` (line ~451)
- Extract `source_image_path` from image metadata
- Extract `well_id` from video_id
- Extract `is_seed_frame` boolean
- Add all to row data (line ~506)

---

### 2e. Snip Processing Consolidation

**File:** `snip_processing/io.py`

```python
from data_pipeline.schemas.snip_processing import REQUIRED_COLUMNS_SNIP_MANIFEST

def write_snip_manifest(snips: list[dict], output_csv: Path) -> None:
    """Write snip manifest with validation."""
    df = pd.DataFrame(snips)

    # Check required columns exist
    missing = set(REQUIRED_COLUMNS_SNIP_MANIFEST) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in snip_manifest: {sorted(missing)}")

    # Check all required columns are non-empty
    for col in REQUIRED_COLUMNS_SNIP_MANIFEST:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/empty values in snip_manifest")

    df.to_csv(output_csv, index=False)
```

---

### 2f. Feature Extraction Consolidation

**File:** `feature_extraction/consolidate_features.py` (renamed from `consolidate.py`)

```python
from data_pipeline.schemas.features import REQUIRED_COLUMNS_FEATURES

def write_consolidated_features(df: pd.DataFrame, output_csv: Path) -> None:
    """Write consolidated features with validation."""

    # Check required columns exist
    missing = set(REQUIRED_COLUMNS_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in feature_extraction: {sorted(missing)}")

    # Check all required columns are non-empty
    for col in REQUIRED_COLUMNS_FEATURES:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/empty values in feature_extraction")

    df.to_csv(output_csv, index=False)
```

---

### 2g. QC Consolidation

**File:** `quality_control/consolidation/consolidate_qc.py`

```python
from data_pipeline.schemas.quality_control import REQUIRED_COLUMNS_QC

def write_consolidated_qc(df: pd.DataFrame, output_csv: Path) -> None:
    """Write consolidated QC flags with validation."""

    # Check required columns exist
    missing = set(REQUIRED_COLUMNS_QC) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in quality_control: {sorted(missing)}")

    # Check all required columns are non-empty
    for col in REQUIRED_COLUMNS_QC:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/empty values in quality_control")

    df.to_csv(output_csv, index=False)
```

**File:** `quality_control/consolidation/compute_use_embryo.py`

```python
from data_pipeline.schemas.quality_control import QC_FAIL_FLAGS

def derive_use_flags(consolidated_qc: pd.DataFrame) -> pd.DataFrame:
    """
    Apply use_embryo gating: embryo passes if ALL QC_FAIL_FLAGS are False.

    use_embryo = True means embryo passes QC for embeddings and analysis.
    """
    consolidated_qc['use_embryo'] = ~consolidated_qc[QC_FAIL_FLAGS].any(axis=1)
    return consolidated_qc
```

---

### 2h. Analysis-Ready Consolidation

**File:** `analysis_ready/assemble_features_qc_embeddings.py`

```python
from data_pipeline.schemas.analysis_ready import REQUIRED_COLUMNS_ANALYSIS_READY

def write_analysis_table(df: pd.DataFrame, output_csv: Path) -> None:
    """Write analysis-ready table with validation."""

    # Check required columns exist
    missing = set(REQUIRED_COLUMNS_ANALYSIS_READY) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in analysis_ready: {sorted(missing)}")

    # Check all required columns are non-empty
    for col in REQUIRED_COLUMNS_ANALYSIS_READY:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains null/empty values in analysis_ready")

    df.to_csv(output_csv, index=False)
```

---

## Files to Create/Modify

### New files (~250 lines total):
1. `src/data_pipeline/schemas/__init__.py` (empty)
2. `src/data_pipeline/schemas/plate_metadata.py` (~20 lines)
3. `src/data_pipeline/schemas/scope_metadata.py` (~25 lines)
4. `src/data_pipeline/schemas/scope_and_plate_metadata.py` (~25 lines)
5. `src/data_pipeline/schemas/segmentation.py` (~35 lines)
6. `src/data_pipeline/schemas/snip_processing.py` (~20 lines)
7. `src/data_pipeline/schemas/features.py` (~30 lines)
8. `src/data_pipeline/schemas/quality_control.py` (~35 lines)
9. `src/data_pipeline/schemas/analysis_ready.py` (~25 lines)
10. `src/data_pipeline/metadata/plate_processing.py` (~80 lines)

### Renamed + modified files:

**Metadata:** (NEW)
- `metadata/plate_processing.py` (NEW - ~80 lines)

**Preprocessing:**
- `preprocessing/keyence/metadata.py` → `extract_scope_metadata.py` (+10 lines validation)
- `preprocessing/yx1/metadata.py` → `extract_scope_metadata.py` (+10 lines validation)
- `preprocessing/consolidate_plate_n_scope_metadata.py` (NEW - SHARED, not per-microscope, ~40 lines)

**Segmentation:**
- `segmentation/grounded_sam2/csv_formatter.py`:
  - Rename output: `tracking_table.csv` → `segmentation_tracking.csv`
  - Extract `mask_rle`, `source_image_path`, `well_id`, `is_seed_frame` (+8 lines)
  - Add validation (+10 lines)
  - Rename function: `write_tracking_table()` → `write_segmentation_tracking()`

**Snip Processing:**
- `snip_processing/io.py`: Add validation to `write_snip_manifest()` (+10 lines)

**Features:**
- `feature_extraction/consolidate.py` → `consolidate_features.py` (+10 lines validation)

**QC:**
- `quality_control/consolidation/consolidate_qc.py` (+10 lines validation)
- `quality_control/consolidation/compute_use_embryo.py` (+5 lines import)

**Analysis-Ready:**
- `analysis_ready/assemble_features_qc_embeddings.py`: Add validation to `write_analysis_table()` (+10 lines)

---

## Validation Logic (Uniform Across All Consolidation Points)

### Step 1: Column Existence Check
```python
missing = set(REQUIRED_COLUMNS) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {stage}: {sorted(missing)}")
```

### Step 2: Non-Empty Check (ALL required columns)
```python
for col in REQUIRED_COLUMNS:
    if df[col].isna().any():
        raise ValueError(f"Column '{col}' contains null/empty values in {stage}")
```

**Note:** No separate `REQUIRED_NON_EMPTY` lists. ALL required columns must be non-empty.

---

## Why This Design

✅ **MVP**: Minimal code, maximum value
✅ **Uniform validation**: Same pattern everywhere (exist + non-empty)
✅ **Central schema definitions**: Single source of truth for required columns
✅ **Inline validation**: No separate validator module to maintain
✅ **Fail fast**: Errors raised immediately during consolidation
✅ **Clear error messages**: Tells you exactly what's missing and where
✅ **Organized schemas**: Columns grouped by purpose with comments
✅ **RLE in CSV**: Mask data preserved for validation/debugging
✅ **Source paths tracked**: Can trace back to original images
✅ **Snip manifest validated**: Ensures extraction completeness
✅ **Analysis-ready validated**: Critical final hand-off to notebooks
✅ **Clear provenance**: `scope_and_plate_metadata.csv` tells you exactly what's in it
✅ **Shared consolidation**: Microscope-specific extraction, shared joining logic

---

## What This Catches

- **Missing columns** (e.g., forgot to add `micrometers_per_pixel` in scope extraction)
- **Empty critical fields** (e.g., `temperature_c` is null → breaks developmental stage normalization)
- **Bad plate layouts** (e.g., well annotations missing genotype)
- **Empty mask data** (e.g., `mask_rle` is empty → segmentation failed)
- **Missing snip files** (e.g., `cropped_snip_path` is null → extraction incomplete)
- **Schema drift** (e.g., renamed `area_um2` to `surface_area_um2` and forgot to update downstream)
- **Integration errors** (e.g., feature merge dropped required columns)
- **Analysis hand-off failures** (e.g., `use_embryo` missing from final table)

---

## What This Doesn't Do

- ❌ No validation reports or JSON logs (overengineering)
- ❌ No separate validator module (inline checks only)
- ❌ No range checks or type validation (out of scope for MVP)
- ❌ No warnings vs. errors (fail fast only)

---

## Consolidation Points (8 Total)

1. ✅ **Plate Metadata** → `experiment_metadata/{exp}/plate_metadata.csv` (validates input plate layouts)
2. ✅ **Scope Metadata** → `experiment_metadata/{exp}/scope_metadata.csv` (validates microscope extraction - per-microscope)
3. ✅ **Scope & Plate Metadata** → `experiment_metadata/{exp}/scope_and_plate_metadata.csv` (joins validated inputs - SHARED)
4. ✅ **Segmentation** → `segmentation/{exp}/segmentation_tracking.csv` (includes mask_rle + paths + metadata)
5. ✅ **Snip Processing** → `processed_snips/{exp}/snip_manifest.csv` (validates extraction completeness)
6. ✅ **Features** → `computed_features/{exp}/consolidated_snip_features.csv` (includes exp_id + well_id + calibration)
7. ✅ **QC** → `quality_control/{exp}/consolidated/consolidated_qc_flags.csv` (includes death timing)
8. ✅ **Analysis-Ready** → `analysis_ready/{exp}/features_qc_embeddings.csv` (critical final hand-off)

---

## Data Flow

```
INPUT VALIDATION
│
├─ Plate Layout Excel
│   ↓
│   metadata/plate_processing.py
│   ↓
│   experiment_metadata/{exp}/plate_metadata.csv  [VALIDATED]
│
└─ Raw Microscope Files (Keyence or YX1)
    ↓
    preprocessing/{microscope}/extract_scope_metadata.py  [MICROSCOPE-SPECIFIC]
    ↓
    experiment_metadata/{exp}/scope_metadata.csv  [VALIDATED]

CONSOLIDATION (SHARED)
│
└─ plate_metadata.csv + scope_metadata.csv
    ↓
    preprocessing/consolidate_plate_n_scope_metadata.py  [MICROSCOPE-AGNOSTIC]
    ↓
    experiment_metadata/{exp}/scope_and_plate_metadata.csv  [VALIDATED]

DOWNSTREAM PROCESSING
    ↓
    segmentation → features → QC → analysis-ready
    [Each with validation at consolidation points]
```

---

## Implementation Order

1. Create `src/data_pipeline/schemas/` directory + `__init__.py`
2. Create 8 schema files (plate_metadata, scope_metadata, scope_and_plate_metadata, segmentation, snip_processing, features, quality_control, analysis_ready)
3. Create `metadata/plate_processing.py`
4. Rename + modify preprocessing extraction (per-microscope: keyence/extract_scope_metadata.py, yx1/extract_scope_metadata.py)
5. Create shared `preprocessing/consolidate_plate_n_scope_metadata.py`
6. Modify segmentation csv_formatter.py (add mask_rle + source_image_path + well_id + is_seed_frame + validation)
7. Add validation to snip_processing/manifest_generation.py (raw/processed paths + rotation metadata)
8. Add validation to feature_extraction/fraction_alive.py + consolidate_features.py (ensure calibration, viability fraction, metadata included)
9. Modify quality_control/consolidation files (add death timing metadata + consolidate_qc.py + compute_use_embryo.py)
10. Add validation to analysis_ready/assemble_features_qc_embeddings.py
11. Test with one experiment end-to-end

---

## Success Criteria

- [ ] Pipeline fails with clear error if plate layout missing `genotype` or `temperature_c`
- [ ] Pipeline fails with clear error if scope extraction missing `micrometers_per_pixel`
- [ ] Pipeline fails with clear error if `mask_rle` is missing/empty in segmentation
- [ ] Pipeline fails with clear error if `well_id` or `is_seed_frame` missing in segmentation
- [ ] Pipeline fails with clear error if `cropped_snip_path` is missing/null in snip_manifest
- [ ] Pipeline fails with clear error if `predicted_stage_hpf` is null in features
- [ ] Pipeline fails with clear error if `fraction_alive` is null or missing in features
- [ ] Pipeline fails with clear error if `experiment_id` or `well_id` missing in features
- [ ] Pipeline fails with clear error if `dead_inflection_time_int` missing in QC
- [ ] Pipeline fails with clear error if `use_embryo` or `embedding_calculated` missing in analysis_ready
- [ ] Pipeline fails with clear error if required columns are missing at any consolidation point
- [ ] `segmentation_tracking.csv` includes `mask_rle`, `source_image_path`, `well_id`, and `is_seed_frame` columns
- [ ] `scope_and_plate_metadata.csv` successfully joins microscope and plate data
- [ ] `use_embryo` flag is correctly computed using `QC_FAIL_FLAGS`
- [ ] All validation logic lives inline in consolidation functions
- [ ] Consolidation function is shared (not duplicated per-microscope)
- [ ] Embedding manifest contains only gated snips and verified file paths
- [ ] Latent CSVs include `embedding_model` and expected latent dimensions for each snip

---

**Status:** Ready for approval
