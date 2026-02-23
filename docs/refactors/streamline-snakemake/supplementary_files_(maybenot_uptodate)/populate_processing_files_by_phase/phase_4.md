# Phase 4 – Snip Processing

Goal: extract embryo crops from stitched FF images using SAM2 masks,
apply standardized processing (rotation, augmentation, CLAHE), and
generate a validated manifest. All snips must use canonical IDs from
`identifiers/parsing.py` for downstream feature extraction and QC.

---

## Inputs

- `segmentation/{exp}/segmentation_tracking.csv` (Phase 3) – SAM2 mask
  locations, bounding boxes, `snip_id`.
- `built_image_data/{exp}/stitched_ff_images/` (Phase 2a) – source
  imagery.
- Aligned metadata (Phase 1) for calibration if needed.

---

## Outputs

- `processed_snips/{exp}/raw_crops/{snip_id}.tif` – unprocessed embryo
  crops (for debugging/provenance).
- `processed_snips/{exp}/processed/{snip_id}.jpg` – fully processed
  snips (crop + rotate + augment + CLAHE).
- `processed_snips/{exp}/snip_manifest.csv` – schema-backed inventory
  (`REQUIRED_COLUMNS_SNIP_MANIFEST`).

---

## Modules to Populate

### `snip_processing/extraction.py`

- Responsibility: crop embryo regions using SAM2 masks + bounding boxes
  from `segmentation_tracking.csv`.
- No rotation or augmentation applied at this stage.
- Outputs raw TIF crops under `raw_crops/`.
- Must preserve `snip_id` naming via `identifiers/parsing`.

### `snip_processing/rotation.py`

- Responsibility: PCA-based orientation correction for standardized
  embryo alignment.
- Computes rotation angle from mask contour and applies transformation.
- Stores rotation metadata for manifest.

### `snip_processing/augmentation.py`

- Responsibility: apply training data augmentation and contrast
  enhancement.
- Key steps:
  - CLAHE histogram equalization (clipLimit=2.0, tileGridSize=(8,8))
  - Gaussian noise addition to background regions (mean=0, std=10)
  - Gaussian blur blending at edges (sigma=3)
- Outputs processed JPEGs under `processed/`.

### `snip_processing/manifest_generation.py`

- Responsibility: scan `processed/` directory, join with
  `segmentation_tracking.csv`, validate completeness.
- Emits `snip_manifest.csv` with schema enforcement
  (`REQUIRED_COLUMNS_SNIP_MANIFEST`).
- Key columns:
  - `snip_id`, `experiment_id`, `well_id`, `embryo_id`, `time_int`
  - `raw_crop_path`, `processed_snip_path`
  - `file_size_bytes`, `image_width_px`, `image_height_px`
  - `processing_timestamp`

---

## Contracts & Validation

- **ID consistency:** use `snip_id` from `segmentation_tracking.csv`
  (derived via `identifiers/parsing`); no ad-hoc ID generation.
- **Schema enforcement:** `snip_manifest.csv` must validate against
  `REQUIRED_COLUMNS_SNIP_MANIFEST`.
- **Completeness check:** manifest generation fails if expected snips
  are missing or files are empty.
- **Separate extraction/processing:** raw crops preserved for debugging;
  manifest can be regenerated without reprocessing images.

---

## Handoff

- Phase 4 outputs feed directly into feature extraction (Phase 5) and
  embeddings (Phase 7, after QC filtering in Phase 6).
- Snip manifest provides the authoritative inventory for downstream
  joins and validation.
