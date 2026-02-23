# Phase 6 – Quality Control

Goal: compute quality flags from pure data sources (segmentation, masks,
features) and consolidate into a single QC table with final
`use_embryo_flag` gate. Phase 3 outputs NO QC flags – all quality
assessment happens here in Phase 6.

---

## Inputs

- `segmentation/{exp}/segmentation_tracking.csv` (Phase 3) – pure
  segmentation data (for mask quality checks, image dimensions).
- `segmentation/{exp}/unet_masks/` (Phase 3b) – auxiliary masks (yolk,
  focus, bubble subdirectories).
- `computed_features/{exp}/fraction_alive.csv` (Phase 5) – viability
  metric for death detection.
- `computed_features/{exp}/consolidated_snip_features.csv` (Phase 5) –
  area and stage for outlier detection.
- `metadata/sa_reference_curves.csv` – reference growth curves for
  surface area validation.

---

## Outputs

- `quality_control/{exp}/segmentation_quality_qc.csv` – SAM2 mask
  quality flags (`edge_flag`, `discontinuous_mask_flag`,
  `overlapping_mask_flag`).
- `quality_control/{exp}/auxiliary_mask_qc.csv` – imaging quality flags
  (`yolk_flag`, `focus_flag`, `bubble_flag`).
- `quality_control/{exp}/embryo_death_qc.csv` – death detection flags
  (`dead_flag`, `dead_inflection_time_int`, `death_predicted_stage_hpf`).
- `quality_control/{exp}/surface_area_outliers_qc.csv` – morphology
  outlier flags (`sa_outlier_flag`).
- `quality_control/{exp}/consolidated_qc_flags.csv` – schema-backed
  merge of all QC flags plus final `use_embryo_flag`
  (`REQUIRED_COLUMNS_QC_FLAGS`).

---

## Modules to Populate

### `quality_control/segmentation_qc/segmentation_quality_qc.py`

- Responsibility: run SAM2 mask quality checks.
- Functions (extracted from deprecated `gsam_qc_class.py`):
  - `check_segmentation_variability()` – detect area variance across
    frames (informational only, not in QC_FAIL_FLAGS)
  - `check_mask_on_edge()` – detect masks touching image boundaries
    (bbox within 5px of edge)
  - `check_discontinuous_masks()` – detect fragmented/disconnected masks
  - `check_overlapping_masks()` – detect embryo mask overlaps (bbox
    check + IoU-based)
- Flags generated: `edge_flag`, `discontinuous_mask_flag`,
  `overlapping_mask_flag`

### `quality_control/auxiliary_mask_qc/imaging_quality_qc.py`

- Responsibility: analyze UNet auxiliary masks for imaging quality
  issues.
- Detections:
  - Yolk sac detection (missing or abnormal)
  - Out-of-focus regions
  - Air bubble artifacts
- Flags generated: `yolk_flag`, `focus_flag`, `bubble_flag`

### `quality_control/auxiliary_mask_qc/death_detection.py`

- Responsibility: generate the **ONLY** death flag by thresholding
  `fraction_alive` metric.
- Threshold: `fraction_alive < 0.9` → `dead_flag = True`
- Computes inflection time (`dead_inflection_time_int`) and death stage
  (`death_predicted_stage_hpf`).
- **Critical:** This is the single authoritative source for `dead_flag`;
  no other module generates death-related flags.

### `quality_control/morphology_qc/size_validation_qc.py`

- Responsibility: flag embryos with abnormal surface area for their
  developmental stage.
- Two-sided outlier detection:
  - Upper bound: `area_um2 > k_upper × reference(stage_hpf)` (k = 1.2)
  - Lower bound: `area_um2 < k_lower × reference(stage_hpf)` (k = 0.9)
- Uses control embryos (WT, control_flag) to build reference curve;
  falls back to `stage_ref.csv` if insufficient controls.
- Flags generated: `sa_outlier_flag`

### `quality_control/consolidation/consolidate_qc.py`

- Responsibility: merge all QC flags on `snip_id` and compute final
  quality gate.
- Schema enforcement: `REQUIRED_COLUMNS_QC_FLAGS` + `QC_FAIL_FLAGS` from
  `schemas/quality_control.py`.
- Compute `use_embryo_flag`:
  ```python
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

  use_embryo_flag = NOT (any flag in QC_FAIL_FLAGS is True)
  ```
- Validation:
  - All snips from `consolidated_features` present
  - No duplicate `snip_ids`
  - All flags are boolean (fillna with False)
  - `use_embryo_flag` correctly computed
  - QC summary statistics generated (counts per flag)

---

## Contracts & Validation

- **Data provenance organization:**
  - Segmentation QC: from SAM2 mask analysis
  - Auxiliary Mask QC: from UNet masks (imaging + viability)
  - Morphology QC: from feature analysis
- **No QC in Phase 3:** Phase 3 outputs pure segmentation data; all
  quality assessment happens here in Phase 6.
- **Schema enforcement:** `consolidated_qc_flags.csv` must validate
  against schema; consolidation fails if any QC module output is
  incomplete.
- **Boolean flags:** All QC flags must be boolean; missing values
  treated as False.
- **Single death source:** Only `death_detection.py` generates
  `dead_flag`; no other module should compute death-related flags.

---

## Handoff

- Phase 6 outputs feed directly into embeddings (Phase 7), which filters
  to only snips where `use_embryo_flag == True`.
- `consolidated_qc_flags.csv` is the authoritative QC table for all
  downstream analysis and reporting.
