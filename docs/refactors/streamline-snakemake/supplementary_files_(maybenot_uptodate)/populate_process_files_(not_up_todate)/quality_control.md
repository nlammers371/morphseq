# Quality Control Module Population Plan

Goal: replace the sprawling Build04 QC script with dependency-scoped modules that each consume explicit CSV inputs and emit tidy outputs. QC functions should be pure (no hidden filesystem ops), unit-testable, and rely on μm-based metrics where applicable.

---

## `quality_control/auxiliary_mask_qc/imaging_quality_qc.py`
**Responsibilities**
- Combine UNet auxiliary masks (yolk, focus, bubbles) with SAM2 pose data to flag imaging issues.
- Operate on per-snip records keyed by `snip_id`.

**Functions to implement**
- `load_aux_masks(mask_root: Path, experiment_id: str) -> dict[str, list[np.ndarray]]`
- `compute_imaging_qc_flags(pose_table: pd.DataFrame, aux_masks: dict[str, list[np.ndarray]], params: ImagingQCParams) -> pd.DataFrame`
- `write_imaging_qc(df: pd.DataFrame, output_csv: Path) -> None`

**Source material**
- `build04_perform_embryo_qc.py` (imaging flags)
- UNet helper utilities in `build02B_segment_bf_main.py`

**Cleanup notes**
- Accept pose metrics (centroids in μm) from `feature_extraction`; do not recompute geometry.
- Provide configurable radii/distances via `config.qc_thresholds`.
- Ensure flags are boolean columns (e.g., `focus_flag`, `bubble_flag`, `edge_flag`).

---

## `quality_control/auxiliary_mask_qc/embryo_death_qc.py`
**Responsibilities**
- Compute `fraction_alive`, persistent death detection, and `dead_flag` using UNet viability masks plus SAM2 tracks.
- Respect the approved Option 1 architecture (persistence + 2 hour buffer).

**Functions to implement**
- `load_viability_masks(mask_root: Path, experiment_id: str) -> dict[str, list[np.ndarray]]`
- `compute_fraction_alive(mask_geometry: pd.DataFrame, viability_masks: dict[str, list[np.ndarray]]) -> pd.DataFrame`
- `detect_persistent_death(fraction_alive: pd.DataFrame, stage_table: pd.DataFrame, params: ViabilityParams) -> pd.DataFrame`
- `write_embryo_death_qc(df: pd.DataFrame, output_csv: Path) -> None`

**Source material**
- `build04_perform_embryo_qc.py` (`dead_flag`, persistence logic)
- Legacy notebooks validating death detection

**Cleanup notes**
- Fraction alive must use mask areas in μm² (sourced from mask geometry module).
- Stage table provides HPF to compute the 2 hr buffer; enforce presence of `predicted_stage_hpf`.
- Output schema: `snip_id`, `embryo_id`, `time_int`, `fraction_alive`, `dead_flag`, `dead_inflection_time_int`, `death_predicted_stage_hpf`.

---

## `quality_control/segmentation_qc/segmentation_quality_qc.py`
**Responsibilities**
- Analyze SAM2 mask quality (edge contact, overlaps, discontinuities, size anomalies).

**Functions to implement**
- `compute_segmentation_quality(tracking_table: pd.DataFrame, mask_paths: list[Path], params: SegmentationQCParams) -> pd.DataFrame`
- `flag_edge_contact(mask: np.ndarray, margin_px: int) -> bool`
- `flag_overlap(current_mask: np.ndarray, others: list[np.ndarray], iou_threshold: float) -> bool`

**Source material**
- `segmentation_sandbox/scripts/pipelines/05_sam2_qc_analysis.py`
- Build03 QC heuristics

**Cleanup notes**
- Operate on mask files already exported; avoid re-running SAM2 inside QC.
- Emit clear boolean flags (e.g., `mask_on_edge`, `overlapping_masks`).
- Provide vectorized helpers where possible; otherwise keep loops explicit and well-commented.

---

## `quality_control/segmentation_qc/tracking_metrics_qc.py`
**Responsibilities**
- Evaluate trajectory smoothness, speed jumps, ID swaps using pose/kinematics metrics.

**Functions to implement**
- `compute_tracking_metrics(pose_table: pd.DataFrame, params: TrackingQCParams) -> pd.DataFrame`
- `flag_speed_outliers(speed_series: pd.Series, zscore_cutoff: float) -> pd.Series`
- `flag_tracking_discontinuities(trajectory: pd.DataFrame, params: TrackingQCParams) -> pd.Series`

**Source material**
- `build04_perform_embryo_qc.py` (speed/z-score logic)
- Legacy notebooks analyzing tracking failures

**Cleanup notes**
- Input is the pose table (μm-centric). Do not recompute centroids/speeds.
- Ensure outputs align with new naming (`tracking_error_flag`, `speed_outlier_flag`).
- Add summary metrics (mean speed, max displacement) for debugging.

---

## `quality_control/morphology_qc/size_validation_qc.py`
**Responsibilities**
- Detect abnormal surface-area growth using μm² metrics and stage predictions.

**Functions to implement**
- `flag_surface_area_outliers(mask_geometry: pd.DataFrame, stage_table: pd.DataFrame, params: SizeQCParams) -> pd.DataFrame`
- `compute_reference_distribution(stage_table: pd.DataFrame, params: SizeQCParams) -> dict`

**Source material**
- Surface area outlier logic (SA outlier) in `build04_perform_embryo_qc.py`
- Statistical references used in SA validation notebooks

**Cleanup notes**
- Use `area_um2` exclusively; pixel values cannot capture biology.
- Expose both z-score and percentile thresholds for tuning.
- Record a rationale column (e.g., `sa_outlier_reason`) to ease debugging.

---

## `quality_control/consolidation/consolidate_qc.py`
**Responsibilities**
- Merge individual QC tables into `consolidated_qc_flags.csv`.

**Functions to implement**
- `merge_qc_tables(imaging: pd.DataFrame, viability: pd.DataFrame, tracking: pd.DataFrame, segmentation: pd.DataFrame, size: pd.DataFrame) -> pd.DataFrame`
- `write_consolidated_qc(df: pd.DataFrame, output_csv: Path) -> None`
- `validate_qc_columns(df: pd.DataFrame) -> None`

**Source material**
- Consolidation steps in Build04 and later notebooks

**Cleanup notes**
- Join on `snip_id`; fail loudly if mismatched sets appear.
- Ensure boolean columns remain boolean (no NaN). Use explicit fill defaults (`False`).
- Keep QC provenance columns (e.g., `dead_inflection_time_int`, `death_predicted_stage_hpf`) intact for downstream analysis.

---

## `quality_control/consolidation/compute_use_embryo.py`
**Responsibilities**
- Apply gating logic across consolidated QC flags to determine `use_embryo`.
- Optionally compute debugging columns (e.g., `use_embryo_reason`).

**Functions to implement**
- `derive_use_flags(consolidated_qc: pd.DataFrame, params: UseEmbryoParams) -> pd.DataFrame`
- `write_use_flags(df: pd.DataFrame, output_csv: Path) -> None`

**Source material**
- Build04 gating logic (`use_embryo`, `reasons`) and later automation scripts

**Cleanup notes**
- Provide composable policy: threshold definitions live in `config.qc_thresholds`.
- Emit `use_embryo` boolean plus optional enumerated reason for exclusion.
- Keep row order consistent with consolidated QC file for easy diffing.

---

## Cross-cutting tasks
- Centralize parameter structures (`dataclasses` or TypedDict) so Snakemake config merges cleanly.
- Unit test each module with small synthetic data (e.g., a few snips, minimal masks).
- Standardize logging via the shared logger (`data_pipeline.logging.get_logger`).
- Document expected columns in module docstrings; keep comments light but precise.
