# Feature Extraction Module Inventory

Living note for the feature-extraction split. This file should track:
- what each module is responsible for
- what each module consumes
- what each module emits
- where the implementation logic currently lives

The goal is to keep the feature phase parallelizable, contract-driven, and easy to extend.

## Canonical upstream contracts

These are the shared inputs that feature entrypoints should consume before any module-specific extras are considered:

- `experiment_metadata/{experiment}/frame_contract.csv`
- `segmentation_and_tracking/{experiment}/contracts/segmentation_tracking.csv`
- `processed_snips/{experiment}/contracts/snip_manifest.csv`
- `auxiliary_masks/{experiment}/contracts/auxiliary_masks.csv`
- `segmentation_and_tracking/{experiment}/per_well/{well_id}/masks/embryo_mask/`

Notes:
- `frame_contract.csv` is the primary experiment-level boundary.
- `embryo_mask` is the canonical source for body-axis and pose-style calculations.
- `auxiliary_masks.csv` is the canonical source for `via`, `yolk`, `focus`, and `bubble` consumers.

## Module inventory

### 1. `mask_geometry`

Purpose:
- Measure embryo geometry from the embryo mask.
- Produce geometry descriptors that downstream analysis can consume without re-reading raw masks.

Inputs:
- `segmentation_and_tracking/{experiment}/per_well/{well_id}/masks/embryo_mask/`
- `segmentation_and_tracking/{experiment}/contracts/segmentation_tracking.csv`
- `experiment_metadata/{experiment}/frame_contract.csv`

Primary outputs:
- `computed_features/{experiment}/mask_geometry/mask_geometry_metrics.csv`

Typical extracted quantities:
- area
- perimeter
- centroid
- major/minor axis approximations
- length and width estimates
- body-axis or centerline-derived geometry

Implementation source of truth:
- `segmentation_sandbox/scripts/body_axis_analysis/`
  - `centerline_extraction.py`
  - `curvature_metrics.py`
  - `geodesic_method.py`
  - `geodesic_method_optimized.py`
  - `mask_preprocessing.py`
  - `pca_method.py`
  - `spline_utils.py`

Notes:
- This module should be treated as the body-axis/geometry layer, not as segmentation.
- If a geometry calculation needs embryo masks, that dependency should be explicit in the entrypoint contract.

### 2. `curvature_metrics`

Purpose:
- Compute curvature as a separate, potentially slow geometry-derived feature.
- Keep curvature isolated from the faster geometry/pose modules so long-running calculations do not block the rest of the feature DAG.

Inputs:
- `segmentation_and_tracking/{experiment}/per_well/{well_id}/masks/embryo_mask/`
- `computed_features/{experiment}/mask_geometry/mask_geometry_metrics.csv`
- `segmentation_and_tracking/{experiment}/contracts/segmentation_tracking.csv`
- `experiment_metadata/{experiment}/frame_contract.csv`

Primary outputs:
- `computed_features/{experiment}/curvature_metrics/curvature_metrics.csv`

Typical extracted quantities:
- mean curvature
- median curvature
- curvature distribution summaries
- per-axis or per-centerline curvature traces if we choose to keep them

Implementation source of truth:
- `segmentation_sandbox/scripts/body_axis_analysis/curvature_metrics.py`

Notes:
- This should remain a separate file and separate Snakemake rule because it can be slow.
- If curvature depends on a centerline or spline representation, that dependency should come from `mask_geometry` rather than re-deriving the geometry path.

### 3. `pose_kinematics`

Purpose:
- Measure pose and motion-related quantities from the embryo mask over time.
- Capture orientation and temporal movement, not mask generation.

Inputs:
- `segmentation_and_tracking/{experiment}/per_well/{well_id}/masks/embryo_mask/`
- `segmentation_and_tracking/{experiment}/contracts/segmentation_tracking.csv`
- `experiment_metadata/{experiment}/frame_contract.csv`

Primary outputs:
- `computed_features/{experiment}/pose_kinematics/pose_kinematics_metrics.csv`

Typical extracted quantities:
- orientation
- body-axis angle
- centroid displacement
- speed
- frame-to-frame motion summaries
- bounding-box-derived pose proxies
- velocity field summaries from the OT canonical-grid layer
- barycentric displacement / source-to-target vector summaries when available

Implementation source of truth:
- `src/analyze/optimal_transport_morphometrics/uot_masks/run_transport.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/feature_compaction/features.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/feature_compaction/storage.py`
- `segmentation_sandbox/scripts/body_axis_analysis/`
- any future pose helpers promoted into `src/data_pipeline/feature_extraction/core/`

Notes:
- This module is closely related to `mask_geometry`, but its outputs should be temporal/kinematic rather than static geometry.
- Prefer explicit vector-field names such as `velocity_field` or `barycentric_velocity_yx` over an ambiguous “back vector” label.
- If we keep a source-directed vector view, it should be a derived representation of the OT coupling, not a separate model.

### 4. `fraction_alive`

Purpose:
- Estimate whether an embryo is alive using embryo mask plus `via` auxiliary mask overlap.

Inputs:
- `segmentation_and_tracking/{experiment}/per_well/{well_id}/masks/embryo_mask/`
- `auxiliary_masks/{experiment}/contracts/auxiliary_masks.csv`
- `experiment_metadata/{experiment}/frame_contract.csv`
- `segmentation_and_tracking/{experiment}/contracts/segmentation_tracking.csv`

Primary outputs:
- `computed_features/{experiment}/fraction_alive/fraction_alive.csv`

Typical extracted quantities:
- alive / not alive classification
- mask overlap summaries
- per-frame viability score or proxy

Implementation source of truth:
- `src/data_pipeline/feature_extraction/fraction_alive.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_fraction_alive.py`

Notes:
- This module should consume auxiliary masks, not create them.
- `via` is the only auxiliary mask family required for the baseline implementation.

### 5. `stage_predictions`

Purpose:
- Estimate developmental stage from time, temperature, and feature context.
- Keep the stage logic separate from segmentation and mask generation.

Inputs:
- `experiment_metadata/{experiment}/frame_contract.csv`
- `experiment_metadata/{experiment}/scope_metadata_mapped.csv` 
- `computed_features/{experiment}/mask_geometry/mask_geometry_metrics.csv`
- `computed_features/{experiment}/pose_kinematics/pose_kinematics_metrics.csv`
- `computed_features/{experiment}/fraction_alive/fraction_alive.csv`

Primary outputs:
- `computed_features/{experiment}/stage_predictions/stage_predictions.csv`

Typical extracted quantities:
- stage label
- stage confidence or score
- time-aligned stage trajectory

Implementation source of truth:
- `src/data_pipeline/feature_extraction/stage_inference.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_stage_predictions.py`

Notes:
- Stage prediction may start from time/temperature heuristics and later absorb more feature context.
- Keep this module independent from the final consolidation step.

### 6. `consolidate_features`

Purpose:
- Merge all feature tables into the final experiment-level feature contract.
- Reintroduce plate metadata only here.

Inputs:
- `computed_features/{experiment}/mask_geometry/mask_geometry_metrics.csv`
- `computed_features/{experiment}/pose_kinematics/pose_kinematics_metrics.csv`
- `computed_features/{experiment}/fraction_alive/fraction_alive.csv`
- `computed_features/{experiment}/stage_predictions/stage_predictions.csv`
- `experiment_metadata/{experiment}/plate_metadata.csv`

Primary outputs:
- `computed_features/{experiment}/consolidated/consolidated_snip_features.csv`
- `computed_features/{experiment}/consolidated/consolidated_snip_features.csv.validated`
- optional schema marker or sidecar for the consolidated contract

Typical extracted quantities:
- merged feature table
- canonical downstream analysis table
- schema/version marker for consumers

Implementation source of truth:
- `src/data_pipeline/feature_extraction/consolidate_features.py`
- `src/data_pipeline/feature_extraction/entrypoints/consolidate_features.py`

Notes:
- This should be the only place that joins plate metadata back in.
- The consolidated artifact is the contract, not just a convenience output.

## Output layout

Target layout for feature outputs:

```text
computed_features/{experiment}/
  mask_geometry/
    mask_geometry_metrics.csv
    mask_geometry_metrics.csv.validated
  curvature_metrics/
    curvature_metrics.csv
    curvature_metrics.csv.validated
  pose_kinematics/
    pose_kinematics_metrics.csv
    pose_kinematics_metrics.csv.validated
  fraction_alive/
    fraction_alive.csv
    fraction_alive.csv.validated
  stage_predictions/
    stage_predictions.csv
    stage_predictions.csv.validated
  consolidated/
    consolidated_snip_features.csv
    consolidated_snip_features.csv.validated
```

## Open questions

- Which geometry outputs should be considered stable contract fields versus debug-only diagnostics?
- Whether `pose_kinematics` should expose only summary metrics or also per-frame pose traces.
- Whether `stage_predictions` should depend only on time/temperature at first, or also on geometry/viability features in the initial implementation.
- Whether the consolidated contract should store a `schema_version` column or a sidecar JSON schema file.

## Minimal file map

This is the smallest file set we should keep as the modular feature layout.

### Shared plumbing

- `src/data_pipeline/feature_extraction/io/paths.py`
- `src/data_pipeline/feature_extraction/io/loaders.py`
- `src/data_pipeline/feature_extraction/io/writers.py`
- `src/data_pipeline/segmentation_and_tracking/utils/mask_processing.py`

### Mask geometry

- `src/data_pipeline/feature_extraction/core/mask_geometry.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_mask_geometry.py`

### Curvature metrics

- `src/data_pipeline/feature_extraction/core/curvature_skeletonization.py`
- `src/data_pipeline/feature_extraction/core/curvature_metrics.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_curvature_metrics.py`
- `src/data_pipeline/segmentation_and_tracking/utils/mask_processing.py`

### Pose kinematics

- `src/data_pipeline/feature_extraction/core/pose_kinematics.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_pose_kinematics.py`

### Fraction alive

- `src/data_pipeline/feature_extraction/core/fraction_alive.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_fraction_alive.py`

### Stage predictions

- `src/data_pipeline/feature_extraction/core/stage_inference.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_stage_predictions.py`

### Consolidation

- `src/data_pipeline/feature_extraction/core/consolidate_features.py`
- `src/data_pipeline/feature_extraction/entrypoints/consolidate_features.py`

## Implementation split by feature

This is the current touch/create map for the code side of the refactor.

### Shared mask cleanup

Create:
- `src/data_pipeline/segmentation_and_tracking/utils/mask_processing.py`

Touch:
- any feature consumer that needs cleaned embryo masks

### `mask_geometry`

Touch:
- `src/data_pipeline/feature_extraction/core/mask_geometry.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_mask_geometry.py`
- `src/data_pipeline/feature_extraction/io/loaders.py`
- `src/data_pipeline/feature_extraction/io/writers.py`

### `curvature_metrics`

Create:
- `src/data_pipeline/feature_extraction/core/curvature_skeletonization.py`
- `src/data_pipeline/feature_extraction/core/curvature_metrics.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_curvature_metrics.py`

Touch:
- `src/data_pipeline/segmentation_and_tracking/utils/mask_processing.py`
- `src/data_pipeline/feature_extraction/io/loaders.py`
- `src/data_pipeline/feature_extraction/io/writers.py`

### `pose_kinematics`

Touch:
- `src/data_pipeline/feature_extraction/core/pose_kinematics.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_pose_kinematics.py`
- `src/data_pipeline/feature_extraction/io/loaders.py`
- `src/data_pipeline/feature_extraction/io/writers.py`

### `fraction_alive`

Touch:
- `src/data_pipeline/feature_extraction/core/fraction_alive.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_fraction_alive.py`
- `src/data_pipeline/feature_extraction/io/loaders.py`
- `src/data_pipeline/feature_extraction/io/writers.py`

### `stage_predictions`

Touch:
- `src/data_pipeline/feature_extraction/core/stage_inference.py`
- `src/data_pipeline/feature_extraction/entrypoints/compute_stage_predictions.py`
- `src/data_pipeline/feature_extraction/io/loaders.py`
- `src/data_pipeline/feature_extraction/io/writers.py`

### `consolidate_features`

Touch:
- `src/data_pipeline/feature_extraction/core/consolidate_features.py`
- `src/data_pipeline/feature_extraction/entrypoints/consolidate_features.py`
- `src/data_pipeline/feature_extraction/io/loaders.py`
- `src/data_pipeline/feature_extraction/io/writers.py`
