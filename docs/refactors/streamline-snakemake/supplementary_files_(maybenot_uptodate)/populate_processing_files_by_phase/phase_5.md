# Phase 5 – Feature Extraction

Goal: compute geometric, kinematic, and biological features from SAM2
masks and auxiliary data, then consolidate into a single analysis-ready
table. All features must be calibrated (µm, µm/s) and validated against
schemas.

---

## Inputs

- `segmentation/{exp}/segmentation_tracking.csv` (Phase 3) – base table
  with mask RLE, `snip_id`.
- `segmentation/{exp}/unet_masks/via/` (Phase 3b) – viability masks for
  fraction alive computation.
- `experiment_metadata/{exp}/scope_and_plate_metadata.csv` (Phase 1) –
  calibration (`micrometers_per_pixel`, `frame_interval_s`).

---

## Outputs

- `computed_features/{exp}/mask_geometry_metrics.csv` – area, perimeter,
  length, width (px and µm).
- `computed_features/{exp}/pose_kinematics_metrics.csv` – bounding box
  dimensions, orientation, displacement, speed, angular velocity.
- `computed_features/{exp}/fraction_alive.csv` – viability metric per
  snip (from UNet via masks).
- `computed_features/{exp}/stage_predictions.csv` – predicted
  developmental stage (HPF) from area growth curves.
- `computed_features/{exp}/consolidated_snip_features.csv` – schema-backed
  merge of all feature tables (`REQUIRED_COLUMNS_CONSOLIDATED_FEATURES`).

---

## Modules to Populate

### `feature_extraction/mask_geometry_metrics.py`

- Responsibility: compute geometric features from SAM2 masks.
- Key outputs:
  - `area_px`, `area_um2` (using `micrometers_per_pixel`)
  - `perimeter_px`, `perimeter_um`
  - `length_um`, `width_um` (via PCA on mask contour)
  - `centroid_x_um`, `centroid_y_um`
- **Critical:** Must convert pixel-based measures to µm² using
  calibration metadata; fail if uncalibrated areas are detected
  (downstream stage inference requires µm²).

### `feature_extraction/pose_kinematics_metrics.py`

- Responsibility: compute pose and motion features across time.
- Key outputs:
  - Bounding box dimensions (`bbox_width_um`, `bbox_height_um`)
  - Orientation angle (from PCA or SAM2 mask)
  - Frame-to-frame deltas:
    - `displacement_um` (Euclidean distance between centroids)
    - `speed_um_per_s` (displacement / `frame_interval_s`)
    - `angular_velocity_deg_per_s`
- Requires temporal ordering (sort by `embryo_id` + `time_int`).

### `feature_extraction/fraction_alive.py`

- Responsibility: measure proportion of viable pixels per snip using
  UNet viability masks.
- Aggregates by `snip_id` using SAM2 masks to normalize for embryo area.
- Emits continuous `fraction_alive` plus helper metadata (e.g., total
  viability pixels).
- **Critical:** This is the ONLY source for viability data fed into
  Phase 6 death detection.

### `feature_extraction/stage_inference.py`

- Responsibility: predict developmental stage (HPF) from `area_um2`
  growth curves.
- Uses Kimmel et al. (1995) formula or trained model.
- **Must use `area_um2`** – fail if pixel-based areas detected.
- Outputs `predicted_stage_hpf`, `stage_confidence` (optional).

### `feature_extraction/consolidate_features.py`

- Responsibility: merge all feature tables on `snip_id`, join with
  metadata, validate completeness.
- Schema enforcement: `REQUIRED_COLUMNS_CONSOLIDATED_FEATURES`.
- Validation checks:
  - All snips from `segmentation_tracking.csv` have features
  - No missing critical columns (`area_um2`, `predicted_stage_hpf`)
  - No NaN values in required fields
- This is the **single source of truth** consumed by Phase 6 QC modules.

---

## Contracts & Validation

- **Parallel execution:** All feature modules run independently (no
  dependencies between geometry/pose/viability/stage computations).
- **Calibration enforcement:** All spatial measures must be in µm, all
  velocities in µm/s; pixel-only outputs must fail validation.
- **Schema enforcement:** `consolidated_snip_features.csv` must validate
  against schema; consolidation fails if any feature table is incomplete.
- **Temporal consistency:** Kinematic features require correct
  `time_int` ordering; any temporal gaps must be flagged.

---

## Handoff

- Phase 5 outputs feed directly into QC flag generation (Phase 6), which
  consumes `consolidated_snip_features.csv` as the authoritative feature
  table.
- `fraction_alive.csv` specifically feeds Phase 6 death detection (only
  source for `dead_flag`).
