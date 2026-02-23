# MorphSeq Snakemake Rules and Data Flow

**Status:** Target Rules + Data Flow Spec (refactor)
**Audience:** Scientists and developers wiring/maintaining the pipeline
**Last Updated:** 2026-02-10

**Note:** This describes the intended refactor end-state; the repo may still contain legacy paths (for example `experiment_image_manifest.json`) until the implementation is complete.

## 2026-02-10 - Addendum, highlighting what we need to change in the original doc
This addendum only updates ingest/handoff interpretation. Existing downstream rule logic remains unchanged.

Rule-level clarifications:
1. Keep phase names and ordering intact.
2. Keep ingest scope-first through extraction and mapping (YX1 and Keyence remain separate there).
3. Keep `materialize_stitched_images_yx1` / `materialize_stitched_images_keyence` as stage names.
4. Ensure builders emit `stitched_image_index.csv` rows during processing (reporter pattern).
5. Keep the canonical pre-segmentation handoff as:
   - `stitched_image_index.csv`
   - `frame_manifest.csv`
6. For frame-level contracts, standardize on:
   - `channel_id`
   - `channel_name_raw`
   - `temperature`
   - required `micrometers_per_pixel`

## TL;DR
Use this sequence for pre-segmentation data flow:

1. Plate metadata ingest
2. Scope metadata ingest (YX1/Keyence)
3. Scope-specific series mapping
4. Apply mapping
5. Scope-specific stitched image materialization + emit `stitched_image_index.csv`
6. Validate stitched index
7. Build `frame_manifest.csv` from:
   - `scope_metadata_mapped.csv`
   - `stitched_image_index.csv`
   - `plate_metadata.csv`

Downstream segmentation consumes `frame_manifest.csv`.

The old `experiment_image_manifest.json` flow is deprecated and removed.

---

## Rule Flow (High Level)

```
PHASE 1: METADATA
  normalize_plate_metadata
  extract_scope_metadata_yx1 | extract_scope_metadata_keyence
  map_series_to_wells_yx1 | map_series_to_wells_keyence
  apply_series_mapping_yx1 | apply_series_mapping_keyence

PHASE 2: IMAGES + FRAME CONTRACT
  materialize_stitched_images_yx1 | materialize_stitched_images_keyence
  validate_stitched_image_index
  build_frame_manifest

PHASE 3+
  segmentation_and_downstream (consumes frame_manifest.csv)
```

---

## Phase 1 Rules

### `rule normalize_plate_metadata`
**Input**
- `data_pipeline_output/inputs/plate_metadata/{experiment}_well_metadata.xlsx`

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/plate_metadata.csv` `[VALIDATED]`

**Module**
- `metadata_ingest/plate/plate_processing.py`

**Purpose**
- Parse/normalize plate annotations.
- Ensure `temperature` and `start_age_hpf` are available for downstream joining.

---

### `rule extract_scope_metadata_yx1`
### `rule extract_scope_metadata_keyence`
**Input**
- `data_pipeline_output/inputs/raw_image_data/YX1/{experiment}/` or
- `data_pipeline_output/inputs/raw_image_data/Keyence/{experiment}/`

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/scope_metadata_raw.csv` `[VALIDATED]`

**Modules**
- `metadata_ingest/scope/yx1/extract_scope_metadata.py`
- `metadata_ingest/scope/keyence/extract_scope_metadata.py`

**Purpose**
- Extract microscope timing, calibration, dimensions, and raw channel provenance.
- Keep this microscope-specific, but emit a shared column contract.

---

### `rule map_series_to_wells_yx1`
### `rule map_series_to_wells_keyence`
**Input**
- `plate_metadata.csv`
- `scope_metadata_raw.csv`
- (and raw Keyence path for Keyence mapping)

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/series_well_mapping.csv` `[VALIDATED]`
- `data_pipeline_output/experiment_metadata/{experiment}/series_well_mapping_provenance.json`

**Modules**
- `metadata_ingest/scope/yx1/map_series_to_wells.py`
- `metadata_ingest/scope/keyence/map_series_to_wells.py`

**Purpose**
- Resolve microscope series/position IDs into plate well IDs.
- Keep logic scope-specific.

---

### `rule apply_series_mapping_yx1`
### `rule apply_series_mapping_keyence`
**Input**
- `scope_metadata_raw.csv`
- `series_well_mapping.csv`

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/scope_metadata_mapped.csv` `[VALIDATED]`

**Purpose**
- Produce final well-linked scope metadata used by all later joins.
- Modules:
  - `metadata_ingest/scope/yx1/apply_series_mapping.py`
  - `metadata_ingest/scope/keyence/apply_series_mapping.py`

---

## Phase 2 Rules

### `rule materialize_stitched_images_yx1`
### `rule materialize_stitched_images_keyence`
**Input**
- raw scope data path
- `scope_metadata_mapped.csv`

**Output**
- `data_pipeline_output/built_image_data/{experiment}/stitched_ff_images/` (directory)
- `data_pipeline_output/experiment_metadata/{experiment}/stitched_image_index.csv` `[VALIDATED]`

**Modules**
- `image_building/scope/yx1/stitched_ff_builder.py`
- `image_building/scope/keyence/stitched_ff_builder.py`
- shared writer: `image_building/handoff/io.py`

**Purpose**
- Perform scope-specific stitching/materialization.
- Emit reporter rows during write/symlink operations.

**Important**
- Do not crawl the output folder and infer metadata from filenames.
- Builders report rows as they process each frame.

---

### `rule validate_stitched_image_index`
**Input**
- `stitched_image_index.csv`
- `stitched_ff_images/` directory

**Output**
- validation marker file (example):
  - `data_pipeline_output/experiment_metadata/{experiment}/.stitched_image_index.validated`

**Module**
- `image_building/handoff/validate_stitched_index.py`

**Purpose**
- Ensure referenced image paths exist and rows satisfy schema/uniqueness checks.

---

### `rule build_frame_manifest`
**Input**
- `plate_metadata.csv`
- `scope_metadata_mapped.csv`
- `stitched_image_index.csv`
- `.stitched_image_index.validated`

**Output**
- `data_pipeline_output/experiment_metadata/{experiment}/frame_manifest.csv` `[VALIDATED]`

**Module**
- `metadata_ingest/frame_manifest/build_frame_manifest.py`

**Purpose**
- Build one canonical frame-level table for segmentation and downstream logic.
- Join scope calibration/timing with plate annotations and stitched paths.

---

### `rule validate_frame_manifest`
**Input**
- `frame_manifest.csv`

**Output**
- validation marker file (example):
  - `data_pipeline_output/experiment_metadata/{experiment}/.frame_manifest.validated`

**Module**
- `metadata_ingest/frame_manifest/validate_frame_manifest.py`

**Purpose**
- Enforce required columns, non-null checks, uniqueness key, and basic path integrity.

---

## Contract Columns

### `stitched_image_index.csv`
Required columns:
- `experiment_id`
- `microscope_id`
- `well_id`
- `well_index`
- `channel_id`
- `time_int`
- `frame_index`
- `image_id`
- `stitched_image_path`
- `materialization_status`
- `source_artifact_path`
- `source_artifact_kind`

Optional columns:
- `image_width_px`
- `image_height_px`

### `frame_manifest.csv`
Required columns:
- `experiment_id`
- `microscope_id`
- `well_id`
- `well_index`
- `channel_id`
- `channel_name_raw`
- `time_int`
- `frame_index`
- `image_id`
- `stitched_image_path`
- `micrometers_per_pixel`
- `frame_interval_s`
- `absolute_start_time`
- `experiment_time_s`
- `image_width_px`
- `image_height_px`
- `objective_magnification`
- `genotype`
- `treatment`
- `medium`
- `temperature`
- `start_age_hpf`
- `embryos_per_well`

Uniqueness key for both:
- `(experiment_id, well_id, channel_id, time_int)`

---

## Naming and ID Conventions

- `channel_id`: normalized channel (`BF`, `GFP`, etc.)
- `channel_name_raw`: microscope-native channel label
- `image_id`: `{well_id}_{channel_id}_t{frame_index:04d}`

Frame semantics:
- `time_int` = acquisition ordering key
- `frame_index` = contiguous 0-based index after sorting by `time_int` per `(experiment_id, well_id, channel_id)`

---

## Scientist-Friendly Mental Model

If you want to know:
- "What files were produced?" -> read `stitched_image_index.csv`
- "What frame table should segmentation trust?" -> read `frame_manifest.csv`
- "Where embryo IDs start?" -> segmentation stage, not metadata stage

---

## Deprecated Rules and Files (Planned)

Deprecated rule:
- `rule generate_image_manifest`

Removed files:
- `metadata_ingest/manifests/generate_image_manifest.py`
- `schemas/image_manifest.py`

Do not add new dependencies on `experiment_image_manifest.json`.

---

## Acceptance Checks

1. No duplicate frame keys in stitched index.
2. No duplicate frame keys in frame manifest.
3. Every stitched path in frame manifest exists.
4. `micrometers_per_pixel` is non-null for every row.
5. `temperature` and `start_age_hpf` are present where required.
6. No `embryo_id` appears before segmentation outputs.

---

## Appendix: Legacy Detailed Reference (Retained)

This appendix restores the prior long-form rule notes that were accidentally dropped in the short rewrite.

How to read this safely:
- The rules and contracts above this appendix are the canonical current flow.
- If any statement below conflicts with the current flow, follow the current flow.
- Legacy term mapping:
  - `generate_image_manifest` / `experiment_image_manifest.json` -> stitched index + frame manifest flow
  - old flat module references -> scope-first modules + shared handoff validation

> [!WARNING]
> Everything below is retained historical context. The canonical contracts and flow are defined above; if anything below conflicts, follow the sections above.

## (Legacy) Preliminary Snakemake Rules for MorphSeq Pipeline

**Author:** Claude Code + User
**Date:** 2025-10-11
**Status:** DRAFT - Phases 1-5 complete, Phase 6+ pending

---

## Phase 1: Metadata Input Validation

### `rule normalize_plate_metadata`
**Input:**
- `raw_plate_layout.xlsx` (user-provided, various formats)

**Output:**
- `experiment_metadata/{exp}/plate_metadata.csv` [VALIDATED]

**Module:**
- `metadata_ingest/plate/plate_processing.py`
- Schema: `REQUIRED_COLUMNS_PLATE_METADATA`

**Purpose:**
Parse and normalize plate layout spreadsheets into the schema-backed CSV under Phase 1 `raw_inputs/`. Captures optional `series_number_map` and other plate annotations used later in the mapping step.

---

### `rule extract_scope_metadata_keyence`
### `rule extract_scope_metadata_yx1`
**Input:**
- `raw_image_data/{microscope}/{exp}/` (raw microscope files)

**Output:**
- `experiment_metadata/{exp}/scope_metadata_raw.csv` [VALIDATED]

**Module:**
- `metadata_ingest/scope/{microscope}/extract_scope_metadata.py`
- Schema: `REQUIRED_COLUMNS_SCOPE_METADATA_RAW`

**Purpose:**
Pull per-microscope series metadata (micrometers_per_pixel, frame_interval_s, timestamps, raw channel labels) into schema-conformant CSVs stored alongside the plate layout. Keeps parsed scope headers separate from aligned metadata for auditing and remapping.

---

### `rule map_series_to_wells`
**Input:**
- `experiment_metadata/{exp}/plate_metadata.csv`
- `experiment_metadata/{exp}/scope_metadata_raw.csv`

**Output:**
- `experiment_metadata/{exp}/series_well_mapping.csv`
- `experiment_metadata/{exp}/series_well_mapping_provenance.json`

**Module:**
- `metadata_ingest/scope/{microscope}/map_series_to_wells.py`
- Schema: `REQUIRED_COLUMNS_SERIES_MAPPING`

**Purpose:**
Create an explicit series_number → well_index lookup with provenance. Falls back to deterministic implicit mappings when spreadsheets lack explicit maps, and fails fast on ambiguities.

### `rule apply_series_mapping`
**Input:**
- `experiment_metadata/{exp}/plate_metadata.csv`
- `experiment_metadata/{exp}/scope_metadata_raw.csv`
- `experiment_metadata/{exp}/series_well_mapping.csv`

**Output:**
- `experiment_metadata/{exp}/scope_metadata_mapped.csv` [VALIDATED]

**Module:**
- `metadata_ingest/scope/shared/apply_series_mapping.py`
- Schema: `REQUIRED_COLUMNS_SCOPE_METADATA_MAPPED`

**Purpose:**
Join plate and scope metadata using the mapping to provide a schema-checked table that downstream stages consume.

---

## Phase 2a: Image Building

**Input:**
- `raw_image_data/{microscope}/{exp}/`
- `experiment_metadata/{exp}/scope_metadata_mapped.csv`

**Output:**
- `built_image_data/{exp}/stitched_ff_images/` (directory target)
- `build_diagnostics/{exp}/stitching_{microscope}.csv`

**Module:**
- `image_building/scope/{microscope}/stitched_ff_builder.py`

**Purpose:**
Perform microscope-specific z-stack collapse and tile stitching to produce normalized FF images (`{well_index}/{channel_id}/{image_id}.tif`), emit `stitched_image_index.csv` rows via reporter pattern, and emit QA logs for each experiment.

**Note:** Diagnostics CSVs are for human review and do not require schema validation. See Phase 2 docs for rationale.

---

### `rule validate_stitched_image_index`
**Input:**
- `experiment_metadata/{exp}/stitched_image_index.csv` (emitted by builders via reporter pattern)

**Output:**
- `experiment_metadata/{exp}/stitched_image_index.csv` [VALIDATED]

**Module:**
- `image_building/handoff/validate_stitched_index.py`
- Schema: `schemas/stitched_image_index.py`

**Purpose:**
- Validate stitched image index has no duplicate frame keys
- Validate all stitched image paths exist
- Validate channel normalization (BF must be present, all names in VALID_CHANNEL_NAMES)

---

### `rule build_frame_manifest`
**Input:**
- `experiment_metadata/{exp}/stitched_image_index.csv` [VALIDATED]
- `experiment_metadata/{exp}/scope_metadata_mapped.csv` [VALIDATED]
- `experiment_metadata/{exp}/plate_metadata.csv` [VALIDATED]

**Output:**
- `experiment_metadata/{exp}/frame_manifest.csv` [VALIDATED]

**Module:**
- `metadata_ingest/frame_manifest/build_frame_manifest.py`
- Schema: `schemas/frame_manifest.py` (REQUIRED_COLUMNS_FRAME_MANIFEST)

**Purpose:**
- Join stitched image index + scope metadata + plate metadata into canonical frame-level table
- Sort frames by time_int per (experiment_id, well_id, channel_id)
- Assign contiguous `frame_index`
- Single source of truth for frame inventory consumed by segmentation

**Key point:** This is where channel normalization is validated. All downstream rules use normalized channel names from this manifest.

---

## Phase 3: Segmentation (SAM2 Pipeline)

**Note:** Processing happens **per-well** basis using `frame_manifest.csv` to get per-well frame lists.

---

### `rule gdino_detection`
**Input:**
- `built_image_data/{exp}/stitched_ff_images/`
- `experiment_metadata/{exp}/frame_manifest.csv` [VALIDATED]

**Output:**
- `segmentation/{exp}/gdino_detections.json` (per-well)

**Module:**
- `segmentation/grounded_sam2/gdino_detection.py`

**Purpose:**
- Run GroundingDINO on **all frames** in the well
- Detect embryos (count, bounding boxes)
- Determine **seed frame** (good quality frame with clear embryo detection)
- Generate bounding boxes to prompt SAM2

**Key point:** Runs on ALL frames to assess embryo presence/count and select best seed frame. Uses manifest to get per-well frame lists.

---

### `rule sam2_segmentation_and_tracking`
**Input:**
- `gdino_detections.json` (seed frame bboxes)
- `experiment_metadata/{exp}/frame_manifest.csv` [VALIDATED]
- `built_image_data/{exp}/stitched_ff_images/`

**Output:**
- `segmentation/{exp}/sam2_raw_output.json` (nested: video/embryo/frame structure)

**Modules:**
- `segmentation/grounded_sam2/propagation.py` (main entry point)
- `segmentation/grounded_sam2/frame_organization_for_sam2.py` (utility functions - NOT a separate rule)

**Purpose:**
- Track embryos across time using SAM2 video propagation
- Uses seed frame bboxes from GroundingDINO as prompts
- **Custom bidirectional propagation:** backward + forward from seed frame to accommodate SAM2's strict ordering requirements
- **Internal workflow:**
  1. `frame_organization_for_sam2.py` creates temp directory with SAM2-compatible frame ordering
  2. Runs bidirectional propagation (backward from seed, then forward from seed)
  3. Cleans up temp directory
  4. Outputs nested JSON with tracking results

**Key point:** `organize_frames_for_sam2` is a utility function called internally, NOT a separate Snakemake rule.

---

### `rule export_sam2_masks`
**Input:**
- `segmentation/{exp}/sam2_raw_output.json`

**Output:**
- `segmentation/{exp}/mask_images/{image_id}_masks.png` (integer-labeled PNGs)

**Module:**
- `segmentation/grounded_sam2/mask_export.py`

**Purpose:**
- Export masks as integer-labeled PNG images for visualization/QC
- Each embryo gets a unique integer label
- Useful for debugging, visual inspection, and downstream QC

---

### `rule flatten_sam2_to_csv`
**Input:**
- `segmentation/{exp}/sam2_raw_output.json`
- `experiment_metadata/{exp}/scope_metadata_mapped.csv` (to inject well_id, experiment_id, calibration)

**Output:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]

**Module:**
- `segmentation/grounded_sam2/csv_formatter.py`
- Schema: `REQUIRED_COLUMNS_SEGMENTATION_TRACKING`

**Purpose:**
- Flatten nested JSON → row-per-mask CSV
- Add critical columns:
  - `mask_rle` (compressed mask string)
  - `well_id` (from metadata join)
  - `experiment_id` (from metadata join)
  - `is_seed_frame` (boolean flag)
  - `source_image_path` (original stitched image)
  - `exported_mask_path` (PNG mask path)
- Validate against schema (column existence + non-empty checks)

**Key point:** This is the authoritative segmentation output consumed by all downstream steps (snip processing, features, QC).

---

## Phase 3b: UNet Auxiliary Masks

### `rule unet_auxiliary_masks`
**Input:**
- `built_image_data/{exp}/stitched_ff_images/`

**Output:**
- `segmentation/{exp}/unet_masks/via/{image_id}_via.png` (viability/dead regions)
- `segmentation/{exp}/unet_masks/yolk/{image_id}_yolk.png` (yolk sac)
- `segmentation/{exp}/unet_masks/focus/{image_id}_focus.png` (out-of-focus)
- `segmentation/{exp}/unet_masks/bubble/{image_id}_bubble.png` (air bubbles)
- `segmentation/{exp}/unet_masks/mask/{image_id}_mask.png` (embryo - alternative segmentation)

**Modules:**
- `segmentation/unet/inference.py` (main entry point)
- `segmentation/unet/model_loader.py` (loads 5 different checkpoints)

**Purpose:**
- Generate auxiliary masks for QC purposes ONLY (not primary segmentation)
- All 5 models use same inference pipeline, just different checkpoints:
  - `mask_v0_0100` (embryo)
  - `via_v1_0100` (viability/dead regions)
  - `yolk_v1_0050` (yolk sac)
  - `focus_v0_0100` (out-of-focus regions)
  - `bubble_v0_0100` (air bubbles)

**Key point:** UNet masks are used downstream in `auxiliary_mask_qc` module (imaging quality, viability detection). SAM2 remains the authoritative embryo segmentation.

---

---

## Image Manifest Design Discussion

### **Channel Normalization Strategy**

**Problem:** Microscopes use inconsistent channel naming:
- YX1: "BF", "EYES - Dia", "Empty", "GFP" (all need normalization)
- Keyence: "BF", "Brightfield", "GFP" (need standardization)

**Solution:** Normalize during preprocessing, validate in manifest, use normalized names everywhere downstream.

### **Channel Normalization Mapping**

```python
# schemas/channel_normalization.py (NEW)

# Universal channel mapping (all microscopes)
CHANNEL_NORMALIZATION_MAP = {
    # Brightfield variants
    "bf": "BF",
    "brightfield": "BF",
    "eyes - dia": "BF",      # YX1 mislabeling
    "empty": "BF",           # YX1 mislabeling
    "phase": "Phase",

    # Fluorescence channels
    "gfp": "GFP",
    "rfp": "RFP",
    "cfp": "CFP",
    "yfp": "YFP",
    "mcherry": "mCherry",
    "dapi": "DAPI",
}

# Valid normalized names (microscope-agnostic)
VALID_CHANNEL_NAMES = [
    "BF", "Phase",                    # Brightfield
    "GFP", "RFP", "CFP", "YFP",      # Fluorescence
    "mCherry", "DAPI"
]

# BF must always be present
BRIGHTFIELD_CHANNELS = {"BF", "Phase"}
```

### **Structure (Experiment-level, indexed by well_index + normalized channel name)**

```json
{
  "experiment_id": "20250529_30hpf_ctrl",
  "microscope": "yx1",
  "wells": {
    "A01": {
      "well_index": "A01",
      "well_id": "20250529_30hpf_ctrl_A01",
      "genotype": "WT",
      "treatment": "ctrl",
      "temperature": 28.5,
      "embryos_per_well": 1,
      "micrometers_per_pixel": 0.65,
      "pixels_per_micrometer": 1.538,
      "frame_interval_s": 180,
      "image_width_px": 2048,
      "image_height_px": 2048,
      "channels": {
        "BF": {
          "channel_id": "BF",
          "channel_name_raw": "EYES - Dia",
          "microscope_channel_index": 0,
          "frames": [
            {
              "image_id": "20250529_30hpf_ctrl_A01_BF_t0000",
              "time_int": 0,
              "absolute_start_time": "2025-05-29T10:00:00",
              "experiment_time_s": 0,
              "image_path": "built_image_data/20250529_30hpf_ctrl/stitched_ff_images/A01/BF/A01_BF_t0000.tif"
            },
            {
              "image_id": "20250529_30hpf_ctrl_A01_BF_t0001",
              "time_int": 1,
              "absolute_start_time": "2025-05-29T10:03:00",
              "experiment_time_s": 180,
              "image_path": "built_image_data/20250529_30hpf_ctrl/stitched_ff_images/A01/BF/A01_BF_t0001.tif"
            }
          ]
        },
        "GFP": {
          "channel_id": "GFP",
          "channel_name_raw": "GFP",
          "microscope_channel_index": 1,
          "frames": [
            {
              "image_id": "20250529_30hpf_ctrl_A01_GFP_t0000",
              "time_int": 0,
              "image_path": "built_image_data/20250529_30hpf_ctrl/stitched_ff_images/A01/GFP/A01_GFP_t0000.tif"
            }
          ]
        }
      }
    },
    "A02": {...}
  }
}
```

### **Key Design Points:**
1. **`wells` indexed by `well_index`** (e.g., "A01", not full well_id)
2. **`channels` indexed by normalized name** ("BF", "GFP" - NOT "ch00", "ch01")
3. **`image_id` uses normalized channel name** ("..._BF_t0000" - self-documenting!)
4. **`well_id`** = `experiment_id_{well_index}` (full identifier)
5. **Full metadata** from `scope_metadata_mapped.csv` at well level
6. **Provenance preserved:** `channel_name_raw` + `microscope_channel_index` track original values
7. **Frames list per channel** (chronological order for SAM2)

### **Why normalized channel names in image_id?**

**Before (unclear):**
```
image_id: "20250529_30hpf_ctrl_A01_ch00_t0000"  # What is ch00?
image_id: "20250529_30hpf_ctrl_A01_ch01_t0000"  # What is ch01?
```

**After (self-documenting):**
```
image_id: "20250529_30hpf_ctrl_A01_BF_t0000"   # Brightfield
image_id: "20250529_30hpf_ctrl_A01_GFP_t0000"  # GFP channel
```

**Benefits:**
- ✅ Self-documenting (no channel lookup needed)
- ✅ Easier debugging (grep for "_BF_" or "_GFP_")
- ✅ Microscope-agnostic (YX1 and Keyence both use "BF")
- ✅ Biologically meaningful (channel matters, not index)

### **Storage Location:**
```
experiment_metadata/{exp}/
  ├── plate_metadata.csv
  ├── scope_metadata_raw.csv
  ├── scope_metadata_mapped.csv
  ├── stitched_image_index.csv
  └── frame_manifest.csv  ← Single tabular manifest per experiment
```

### **Schema Validation:**
```python
# schemas/frame_manifest.py (replaces deprecated schemas/image_manifest.py)

REQUIRED_EXPERIMENT_FIELDS = [
    'experiment_id',
    'microscope',
    'wells'
]

REQUIRED_WELL_FIELDS = [
    'well_index',
    'well_id',
    'genotype',
    'treatment',
    'temperature',
    'embryos_per_well',
    'micrometers_per_pixel',
    'frame_interval_s',
    'image_width_px',
    'image_height_px',
    'channels'
]

REQUIRED_CHANNEL_FIELDS = [
    'channel_id',              # Normalized: "BF", "GFP", etc.
    'channel_name_raw',          # Original: "EYES - Dia", "Empty", etc.
    # 'microscope_channel_index',  # Original index (0, 1, 2...) for provenance
    'frames'
]

REQUIRED_FRAME_FIELDS = [
    'image_id',
    'frame_index', 
    'absolute_start_time',
    'experiment_time_s',
    'image_path'
]

def validate_channels(channels_dict):
    """
    Validate that channels have been properly normalized.

    Args:
        channels_dict: {"BF": {...}, "GFP": {...}}  # Indexed by normalized name
    """
    from data_pipeline.schemas.channel_normalization import (
        VALID_CHANNEL_NAMES,
        BRIGHTFIELD_CHANNELS
    )

    # 1. Must have at least one brightfield channel
    bf_present = any(ch in BRIGHTFIELD_CHANNELS for ch in channels_dict.keys())
    if not bf_present:
        raise ValueError(f"Missing brightfield channel. Found: {list(channels_dict.keys())}")

    # 2. All channel names must be normalized
    for ch_name, ch_data in channels_dict.items():
        if ch_name not in VALID_CHANNEL_NAMES:
            raise ValueError(f"Invalid channel name: {ch_name}. Must be one of {VALID_CHANNEL_NAMES}")

        # Check required fields
        missing = set(REQUIRED_CHANNEL_FIELDS) - set(ch_data.keys())
        if missing:
            raise ValueError(f"Channel {ch_name} missing fields: {missing}")
```

### **How it's generated:**
```
rule validate_stitched_image_index:
    input:
        - experiment_metadata/{exp}/stitched_image_index.csv  [emitted by builders]
    output:
        - experiment_metadata/{exp}/stitched_image_index.csv [VALIDATED]

    # Module: image_building/handoff/validate_stitched_index.py
    # 1. Read stitched_image_index.csv rows emitted by scope-specific builders
    # 2. Check for duplicate frame keys
    # 3. Validate all image paths exist
    # 4. Validate channel normalization (BF must be present)

rule build_frame_manifest:
    input:
        - experiment_metadata/{exp}/stitched_image_index.csv [VALIDATED]
        - experiment_metadata/{exp}/scope_metadata_mapped.csv [VALIDATED]
        - experiment_metadata/{exp}/plate_metadata.csv [VALIDATED]
    output:
        - experiment_metadata/{exp}/frame_manifest.csv [VALIDATED]

    # Module: metadata_ingest/frame_manifest/build_frame_manifest.py
    # 1. Read scope_metadata_mapped.csv (includes normalized channel info from preprocessing)
    # 2. Join with stitched_image_index.csv and plate_metadata.csv
    # 3. Sort frames by time_int per (experiment_id, well_id, channel_id)
    # 4. Assign contiguous frame_index
    # 5. Validate against schema (REQUIRED_COLUMNS_FRAME_MANIFEST)
    # 6. Write frame_manifest.csv [VALIDATED]
```

### **Pipeline Flow for Channel Normalization:**

```
1. Preprocessing (microscope-specific normalization)
   ├─ metadata_ingest/scope/yx1/extract_scope_metadata.py
   │  ├─ Import CHANNEL_NORMALIZATION_MAP from schemas/
   │  ├─ Detect raw channel names from ND2 metadata
   │  ├─ Normalize: "EYES - Dia" → "BF", "GFP" → "GFP"
   │  └─ Write scope_metadata_raw.csv with channel_id + channel_name_raw columns
   │
   └─ metadata_ingest/scope/keyence/extract_scope_metadata.py
      ├─ Import CHANNEL_NORMALIZATION_MAP from schemas/
      ├─ Detect raw channel names from Keyence file structure
      ├─ Normalize: "Brightfield" → "BF", "gfp" → "GFP"
      └─ Write scope_metadata_raw.csv with channel_id + channel_name_raw columns

2. Frame Manifest Generation (shared validation)
   └─ metadata_ingest/frame_manifest/build_frame_manifest.py
      ├─ Read scope_metadata_mapped.csv (includes normalized channel info from preprocessing)
      ├─ Join with stitched_image_index.csv + plate_metadata.csv
      ├─ Validate channels using schemas/frame_manifest.py
      │  ├─ Check BF channel_id present (BRIGHTFIELD_CHANNELS)
      │  ├─ Check all channel_id values in VALID_CHANNEL_NAMES
      │  └─ Check REQUIRED_COLUMNS_FRAME_MANIFEST present
      └─ Write frame_manifest.csv [VALIDATED]

3. Downstream Rules (consume normalized names)
   └─ All rules use normalized channel_id values ("BF", "GFP")
      └─ No microscope-specific logic needed
```

---

## Phase 4: Snip Processing

**Note:** Snips are **processed** embryo crops, not just extracted. Processing includes: crop + rotation + noise augmentation + CLAHE equalization + Gaussian blending for training data quality.

**Current Implementation:** `src/build/build03A_process_images.py` lines 257-414 (export_embryo_snips function)

---

### `rule extract_snips`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]
- `built_image_data/{exp}/stitched_ff_images/`

**Output:**
- `processed_snips/{exp}/raw_crops/{snip_id}.tif` (unprocessed crops)

**Module:**
- `snip_processing/extraction.py`

**Purpose:**
- Crop embryo regions using SAM2 masks + bounding boxes from segmentation_tracking.csv
- No rotation or augmentation applied
- Save as raw TIF files for subsequent processing
- Useful for debugging and provenance (can inspect pre-processing crops)

**Key point:** Creates raw crops only. All processing (crop + rotation + augmentation) happens in next rule.

---

### `rule process_snips`
**Input:**
- `processed_snips/{exp}/raw_crops/{snip_id}.tif`
- `segmentation/{exp}/segmentation_tracking.csv` [needed for mask_rle data]

**Output:**
- `processed_snips/{exp}/processed/{snip_id}.jpg` (fully processed)

**Module:**
- `snip_processing/rotation.py` (PCA-based orientation)
- `snip_processing/augmentation.py` (noise + CLAHE + blending)

**Purpose:**
- Apply crop + PCA-based rotation for standardized orientation
- Add Gaussian noise to background regions (training data augmentation)
- Apply CLAHE histogram equalization (contrast enhancement)
- Gaussian blending at edges (smooth transitions)
- Save as JPEGs with snip_id naming

**Key processing steps (from build03A lines 367-384):**
1. Crop to bounding box region
2. PCA rotation using mask contour (angle stored for manifest)
3. CLAHE equalization (clipLimit=2.0, tileGridSize=(8,8))
4. Gaussian noise addition to background (mean=0, std=10)
5. Gaussian blur blending at edges (sigma=3)

**Key point:** Only saves processed JPEGs. Manifest generation happens separately to allow validation without reprocessing.

---

### `rule generate_snip_manifest`
**Input:**
- `processed_snips/{exp}/processed/` (directory of processed snips)
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]

**Output:**
- `processed_snips/{exp}/snip_manifest.csv` [VALIDATED]

**Module:**
- `snip_processing/manifest_generation.py`
- Schema: `REQUIRED_COLUMNS_SNIP_MANIFEST`

**Purpose:**
- Scan processed_snips/ directory to inventory all processed JPEGs
- Join with segmentation_tracking.csv to get experiment_id, well_id, embryo_id, time_int
- Validate completeness (all expected snips present, no missing files)
- Add file metadata (file size, dimensions, processing timestamp)
- Validate schema and write snip_manifest.csv [VALIDATED]

**Required manifest columns:**
```python
REQUIRED_COLUMNS_SNIP_MANIFEST = [
    'snip_id',
    'experiment_id',
    'well_id',
    'embryo_id',
    'time_int',
    'raw_crop_path',          # Path to raw crop TIF
    'processed_snip_path',    # Path to processed JPEG
    'file_size_bytes',        # Validate files exist and are non-empty
    'image_width_px',         # Actual snip dimensions
    'image_height_px',
    'processing_timestamp',   # When processing occurred
]
```

**Output structure:**
```
processed_snips/{exp}/
├── raw_crops/
│   └── {snip_id}.tif         # Unprocessed crops (for debugging)
├── processed/
│   └── {snip_id}.jpg         # Fully processed (crop + rotate + augment)
└── snip_manifest.csv         # [VALIDATED] - Authoritative snip inventory
```

**Key point:** Separate manifest generation allows validation without reprocessing. Can regenerate manifest to add new columns or verify file integrity.

---

## Phase 5: Feature Extraction

**Note:** Features are computed from validated segmentation data and consolidated into a single analysis-ready table. All feature modules run in parallel (independent computations), then consolidation merges results.

**Current Implementation:** `src/build/build03A_process_images.py` (compile_embryo_stats function, lines 771-863)

---

### `rule compute_mask_geometry`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]
- `experiment_metadata/{exp}/scope_metadata_mapped.csv` [for pixel_size calibration]

**Output:**
- `computed_features/{exp}/mask_geometry_metrics.csv`

**Module:**
- `feature_extraction/mask_geometry_metrics.py`

**Purpose:**
- Compute geometric features from SAM2 masks:
  - `area_px`, `area_um2` (using micrometers_per_pixel)
  - `perimeter_px`, `perimeter_um`
  - `length_um`, `width_um` (via PCA on mask contour)
  - `centroid_x_um`, `centroid_y_um`
- **Critical:** Must convert area_px → area_um2 using micrometers_per_pixel from scope_metadata_mapped.csv
- **Critical:** Fail if pixel-based areas are used without calibration (downstream stage inference requires um2)

**Key columns:**
```python
OUTPUT_COLUMNS_MASK_GEOMETRY = [
    'snip_id',
    'area_px',
    'area_um2',           # Required for stage inference
    'perimeter_px',
    'perimeter_um',
    'length_um',
    'width_um',
    'centroid_x_um',
    'centroid_y_um',
]
```

---

### `rule compute_pose_kinematics`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]
- `experiment_metadata/{exp}/scope_metadata_mapped.csv` [for pixel_size + frame_interval_s]

**Output:**
- `computed_features/{exp}/pose_kinematics_metrics.csv`

**Module:**
- `feature_extraction/pose_kinematics_metrics.py`

**Purpose:**
- Compute pose and motion features:
  - Bounding box dimensions (bbox_width_um, bbox_height_um)
  - Orientation angle (from PCA or SAM2 mask)
  - Frame-to-frame deltas:
    - `displacement_um` (Euclidean distance between centroids)
    - `speed_um_per_s` (displacement / frame_interval_s)
    - `angular_velocity_deg_per_s`
- Requires temporal ordering (sort by embryo_id + time_int)

**Key columns:**
```python
OUTPUT_COLUMNS_POSE_KINEMATICS = [
    'snip_id',
    'bbox_width_um',
    'bbox_height_um',
    'orientation_deg',
    'displacement_um',
    'speed_um_per_s',
    'angular_velocity_deg_per_s',
]
```

---

### `rule compute_fraction_alive`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED]
- `segmentation/{exp}/unet_masks/viability/`

**Output:**
- `computed_features/{exp}/fraction_alive.csv`

**Module:**
- `feature_extraction/fraction_alive.py`

**Purpose:**
- Measure the proportion of viable pixels per snip using UNet viability masks
- Aggregate by `snip_id` using SAM2 masks to normalize for embryo area
- Emits continuous `fraction_alive` plus helper metadata (e.g., total viability pixels)

---

### `rule predict_developmental_stage`
**Input:**
- `computed_features/{exp}/mask_geometry_metrics.csv` [needs area_um2]

**Output:**
- `computed_features/{exp}/stage_predictions.csv`

**Module:**
- `feature_extraction/stage_inference.py`

**Purpose:**
- Predict developmental stage (HPF - hours post fertilization) from area_um2 growth curves
- Uses Kimmel et al. (1995) formula or trained model
- **Must use area_um2** - fail if pixel-based areas detected
- Outputs predicted_stage_hpf for each snip

**Key columns:**
```python
OUTPUT_COLUMNS_STAGE_PREDICTIONS = [
    'snip_id',
    'predicted_stage_hpf',
    'stage_confidence',      # Optional confidence score
]
```

**Key point:** This rule depends on `compute_mask_geometry` completing first (needs area_um2 input).

---

### `rule consolidate_features`
**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [base table with snip_id]
- `computed_features/{exp}/mask_geometry_metrics.csv`
- `computed_features/{exp}/pose_kinematics_metrics.csv`
- `computed_features/{exp}/fraction_alive.csv`
- `computed_features/{exp}/stage_predictions.csv`
- `experiment_metadata/{exp}/scope_metadata_mapped.csv` [for joining experiment_id, well_id]

**Output:**
- `computed_features/{exp}/consolidated_snip_features.csv` [VALIDATED]

**Module:**
- `feature_extraction/consolidate_features.py`
- Schema: `REQUIRED_COLUMNS_CONSOLIDATED_FEATURES`

**Purpose:**
- Merge all feature tables on snip_id
- Add experiment metadata (experiment_id, well_id, genotype, treatment, temperature)
- Validate completeness:
  - All snips from segmentation_tracking.csv have features
  - No missing critical columns (area_um2, predicted_stage_hpf)
  - No NaN values in required fields
- This is the **single source of truth** consumed by all QC and analysis modules

---

## Phase 6: Quality Control

**Key principle:** QC modules compute quality flags from pure data sources. Phase 3 outputs NO QC flags - all quality assessment happens here in Phase 6.

QC modules are organized by data provenance:
- **Segmentation QC:** Flags from SAM2 mask analysis
- **Auxiliary Mask QC:** Flags from UNet masks (imaging + viability)
- **Morphology QC:** Flags from feature analysis

---

### `rule compute_segmentation_quality_qc`

**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED] (pure segmentation data, includes image dimensions)

**Output:**
- `quality_control/{exp}/segmentation_quality_qc.csv`

**Module:**
- `quality_control/segmentation_qc/segmentation_quality_qc.py`

**Purpose:**
Run SAM2 mask quality checks using functions extracted from `gsam_qc_class.py`:
- `check_segmentation_variability()` - Detect area variance across frames
- `check_mask_on_edge()` - Detect masks touching image boundaries
- `check_discontinuous_masks()` - Detect fragmented/disconnected masks
- `check_overlapping_masks()` - Detect embryo mask overlaps (bbox check + IoU)

**Flags generated:**
- `edge_flag` - Mask bbox within 5px of image edge
- `discontinuous_mask_flag` - Mask has multiple disconnected components
- `overlapping_mask_flag` - Masks overlap (IoU-based after bbox check)
- (Note: high_segmentation_var flags computed but not used in QC_FAIL_FLAGS - informational only)

**Output columns:**
`snip_id`, `edge_flag`, `discontinuous_mask_flag`, `overlapping_mask_flag`

**Key point:** `gsam_qc_class.py` is deprecated - only the QC check functions are extracted and reused here.

---

### `rule compute_auxiliary_mask_qc`

**Input:**
- `segmentation/{exp}/segmentation_tracking.csv` [VALIDATED] (for snip_ids, mask locations, image dims)
- `segmentation/{exp}/unet_masks/` (yolk, focus, bubble subdirectories)

**Output:**
- `quality_control/{exp}/auxiliary_mask_qc.csv`

**Module:**
- `quality_control/auxiliary_mask_qc/imaging_quality_qc.py`

**Purpose:**
Analyze UNet auxiliary masks to detect imaging quality issues:
- Yolk sac detection (missing or abnormal)
- Out-of-focus regions
- Air bubble artifacts

**Flags generated:**
- `yolk_flag` - No yolk sac detected or abnormal yolk
- `focus_flag` - Out-of-focus regions detected
- `bubble_flag` - Air bubble artifacts detected

**Output columns:**
`snip_id`, `yolk_flag`, `focus_flag`, `bubble_flag`

---

### `rule compute_embryo_death_qc`

**Input:**
- `computed_features/{exp}/fraction_alive.csv` (from Phase 5 feature extraction)

**Output:**
- `quality_control/{exp}/embryo_death_qc.csv`

**Module:**
- `quality_control/auxiliary_mask_qc/death_detection.py`

**Purpose:**
Generate the **ONLY** death flag by thresholding `fraction_alive` metric.

**Flags generated:**
- `dead_flag` - THE ONLY SOURCE of death detection (threshold: fraction_alive < 0.9)

**Output columns:**
`snip_id`, `embryo_id`, `time_int`, `fraction_alive`, `dead_flag`, `dead_inflection_time_int`, `death_predicted_stage_hpf`

**Key point:** This is the single authoritative source for `dead_flag`. No other module generates death-related flags.

---

### `rule compute_surface_area_outliers`

**Input:**
- `computed_features/{exp}/consolidated_snip_features.csv` (needs `area_um2` + `predicted_stage_hpf`)
- `metadata/sa_reference_curves.csv` (reference growth curves)

**Output:**
- `quality_control/{exp}/surface_area_outliers_qc.csv`

**Module:**
- `quality_control/morphology_qc/size_validation_qc.py`

**Purpose:**
Flag embryos with abnormal surface area for their developmental stage.

Two-sided outlier detection:
- Upper bound: `area_um2 > k_upper × reference(stage_hpf)` (k = 1.2)
- Lower bound: `area_um2 < k_lower × reference(stage_hpf)` (k = 0.9)

Uses control embryos (wt, control_flag) to build reference curve; falls back to `stage_ref.csv` if insufficient controls.

**Flags generated:**
- `sa_outlier_flag` - Abnormal area for developmental stage

**Output columns:**
`snip_id`, `sa_outlier_flag`

---

### `rule consolidate_qc_flags`

**Input:**
- `quality_control/{exp}/segmentation_quality_qc.csv`
- `quality_control/{exp}/auxiliary_mask_qc.csv`
- `quality_control/{exp}/embryo_death_qc.csv`
- `quality_control/{exp}/surface_area_outliers_qc.csv`

**Output:**
- `quality_control/{exp}/consolidated_qc_flags.csv` [VALIDATED]

**Module:**
- `quality_control/consolidation/consolidate_qc.py`
- Schema: `REQUIRED_COLUMNS_QC_FLAGS` + `QC_FAIL_FLAGS` from `schemas/quality_control.py`

**Purpose:**
Merge all QC flags on `snip_id` and compute final quality gate.

**Compute `use_embryo_flag`:**
```python
# From schemas/quality_control.py
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

# Final gate
use_embryo_flag = NOT (any flag in QC_FAIL_FLAGS is True)
```

**Validation:**
1. All snips from consolidated_features present
2. No duplicate snip_ids
3. All flags are boolean (fillna with False)
4. `use_embryo_flag` correctly computed
5. QC summary statistics generated (counts per flag)

**Output columns:**
All QC flags + `use_embryo_flag` + QC provenance metadata

**Key point:** This is the authoritative QC table consumed by embeddings and analysis modules

---

## Phase 7: Embeddings

Embeddings run only on snips that pass QC (`use_embryo_flag == True`). We stage a manifest of eligible snips, launch the VAE in a Python 3.9 subprocess, and validate the resulting latent vectors.

---

### `rule prepare_embedding_manifest`
**Input:**
- `processed_snips/{exp}/processed/` (final JPEG crops)
- `quality_control/{exp}/use_embryo_flags.csv` `[VALIDATED]`

**Output:**
- `latent_embeddings/{model_name}/{exp}_embedding_manifest.csv` `[VALIDATED]`

**Module:**
- `embeddings/prepare_manifest.py`
- Schema: `REQUIRED_COLUMNS_EMBEDDING_MANIFEST` (snip_id, processed_snip_path, use_embryo_flag, file_size_bytes)

**Purpose:**
- Join the QC gate with processed snip paths, keeping only rows where `use_embryo_flag == True`
- Validate file existence/size so the inference step never dereferences stale paths
- Emit one manifest per experiment/model, making reruns idempotent even when additional snips are added later

---

### `rule generate_embeddings`
**Input:**
- `latent_embeddings/{model_name}/{exp}_embedding_manifest.csv` `[VALIDATED]`
- `models/embeddings/{model_name}/` (checkpoint + config)

**Output:**
- `latent_embeddings/{model_name}/{exp}_latents.csv` `[VALIDATED]`

**Module:**
- `embeddings/inference.py` (invoked via `embeddings/subprocess_wrapper.py`)
- Post-checks handled by `embeddings/file_validation.py`
- Schema: `REQUIRED_COLUMNS_LATENTS` (`snip_id`, `embedding_model`, `z0` … `z{dim-1}`)

**Purpose:**
- Launch VAE inference inside an isolated Python 3.9 environment to match training dependencies
- Stream manifest rows into the model, produce latent vectors, and write a CSV aligned with manifest ordering
- Validate that column counts match model dimensionality, every manifest row appears once, and no latent entry is NaN
- Tag each row with `embedding_model` so downstream aggregation can mix models safely

**Key parameters:**
- `model_name` (Snakemake config; default `morphology_vae_2024`)
- `batch_size` (configurable, defaults to 256)
- `device` resolved via `config/runtime.py`

---

## Phase 8: Analysis-Ready - PENDING

---

## Summary of New Files and Data Outputs

### **New Schema Files**
```
src/data_pipeline/schemas/
├── __init__.py
├── channel_normalization.py          # Channel name mappings
├── plate_metadata.py
├── scope_metadata_raw.py             # RENAMED from scope_metadata.py
├── scope_metadata_mapped.py          # RENAMED from scope_and_plate_metadata.py
├── stitched_image_index.py           # NEW
├── frame_manifest.py                 # NEW (replaces deprecated image_manifest.py)
├── segmentation.py
├── snip_processing.py
├── features.py
├── quality_control.py
└── analysis_ready.py
```

### **New Processing Modules**
```
src/data_pipeline/
├── metadata_ingest/
│   ├── scope/
│   │   ├── yx1/
│   │   │   ├── extract_scope_metadata.py
│   │   │   └── map_series_to_wells.py
│   │   ├── keyence/
│   │   │   ├── extract_scope_metadata.py
│   │   │   └── map_series_to_wells.py
│   │   └── shared/
│   │       └── apply_series_mapping.py
│   └── frame_manifest/
│       └── build_frame_manifest.py
├── image_building/
│   ├── scope/
│   │   ├── yx1/
│   │   │   └── stitched_ff_builder.py
│   │   └── keyence/
│   │       └── stitched_ff_builder.py
│   └── handoff/
│       ├── io.py
│       └── validate_stitched_index.py
├── snip_processing/
│   ├── extraction.py
│   ├── rotation.py
│   ├── augmentation.py
│   └── manifest_generation.py
├── feature_extraction/
│   ├── mask_geometry_metrics.py
│   ├── pose_kinematics_metrics.py
│   ├── fraction_alive.py
│   ├── stage_inference.py
│   └── consolidate_features.py
└── embeddings/
    ├── prepare_manifest.py
    ├── inference.py
    ├── subprocess_wrapper.py
    └── file_validation.py
```

### **New Data Outputs**
```
experiment_metadata/{exp}/
├── plate_metadata.csv [VALIDATED]
├── scope_metadata_raw.csv [VALIDATED]
├── scope_metadata_mapped.csv [VALIDATED]
├── stitched_image_index.csv [VALIDATED]
└── frame_manifest.csv [VALIDATED]  # Single tabular manifest per experiment


segmentation/{exp}/
├── gdino_detections.json
├── sam2_raw_output.json
├── segmentation_tracking.csv [VALIDATED]
└── mask_images/

processed_snips/{exp}/
├── raw_crops/{snip_id}.tif
├── processed/{snip_id}.jpg
└── snip_manifest.csv [VALIDATED]

computed_features/{exp}/
├── mask_geometry_metrics.csv
├── pose_kinematics_metrics.csv
├── fraction_alive.csv
├── stage_predictions.csv
└── consolidated_snip_features.csv [VALIDATED]

latent_embeddings/{model_name}/
├── {experiment_id}_embedding_manifest.csv [VALIDATED]
└── {experiment_id}_latents.csv [VALIDATED]
```

### **Key Changes from Original Plan**
1. ✅ **Renamed directory:** `processed_metadata/` → `experiment_metadata/`
2. ✅ **Replaced rule:** `generate_image_manifest` → `validate_stitched_image_index` + `build_frame_manifest`
3. ✅ **Channel normalization:** YX1/Keyence extract + normalize → `channel_id` + `channel_name_raw` columns; frame manifest validates → downstream consumes
4. ✅ **Self-documenting image_ids:** `_BF_t0000` instead of `_ch00_t0000`
5. ✅ **Provenance preserved:** `channel_name_raw` + `microscope_channel_index` track original values
6. ✅ **Single tabular manifest:** `frame_manifest.csv` (flat CSV; replaces nested JSON)

---

## Notes
- [VALIDATED] = Schema enforcement at consolidation point
- Microscope-specific vs shared logic explicitly noted
- Explicit consolidation steps tracked
- Channel normalization happens in Phase 1 (extract_scope_metadata)
- Manifest generation validates normalization in Phase 2
