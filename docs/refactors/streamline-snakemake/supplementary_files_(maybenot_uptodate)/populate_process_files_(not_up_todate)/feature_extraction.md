# Feature Extraction Module Population Plan

Goal: provide thin, composable functions that transform SAM2 tracking output and snip metadata into a consolidated feature table. Every module must consume explicit inputs (DataFrames, dicts) and emit schema-documented outputs. Surface-area metrics are always expressed in μm² using microscope metadata; any pixel-only logic is considered a bug.

---

## `feature_extraction/mask_geometry_metrics.py`
**Responsibilities**
- Derive per-snip mask geometry from SAM2 PNGs or RLE data.
- Convert all surface-area figures to μm² using pixel size metadata provided by preprocessing.
- Emit a tidy DataFrame keyed by `snip_id`.

**Functions to implement**
- `compute_mask_geometry(mask: np.ndarray, pixel_size_um: float) -> dict`
- `mask_geometry_from_csv(tracking_table: pd.DataFrame, masks_dir: Path, pixel_size_um: float) -> pd.DataFrame`
- `summarize_contour(contour: np.ndarray) -> dict`

**Source material**
- `build03A_process_images.py` (mask geometry + contour stats)
- `segmentation_sandbox/scripts/utils/mask_utils.py` (geometry helpers)

**Cleanup notes**
- Return both `area_px` and `area_um2`, but document that downstream consumers must prefer μm².
- Keep computation NumPy/scikit-image only; no torch dependency required.
- Validate that `pixel_size_um` is supplied—raise immediately if missing.

---

## `feature_extraction/pose_kinematics_metrics.py`
**Responsibilities**
- Use SAM2 tracking_table rows to compute pose (centroid, bounding boxes) and motion deltas.
- Bridge pixels ↔ μm for positional values using microscope metadata.

**Functions to implement**
- `compute_pose_metrics(record: dict, pixel_size_um: float) -> dict`
- `compute_frame_deltas(current: dict, previous: dict, interval_minutes: float) -> dict`
- `build_pose_table(tracking_table: pd.DataFrame, mask_geometry: pd.DataFrame, pixel_size_um: float, frame_interval_minutes: float) -> pd.DataFrame`

**Source material**
- `build03A_process_images.py` (centroid/orientation/motion features)
- `build04_perform_embryo_qc.py` (speed / delta calculations)

**Cleanup notes**
- Accept pre-sorted tracking table (by `embryo_id`, `time_int`); assert ordering before computing deltas.
- Output both pixel and μm values where helpful (`centroid_x_px`, `centroid_x_um`).
- Provide velocity in μm/hour (or μm/min) using explicit frame interval from metadata.

---

## `feature_extraction/stage_inference.py`
**Responsibilities**
- Map per-snip surface area (μm²) to developmental stage (HPF) using reference curves.
- Enforce that inputs are already converted to μm².

**Functions to implement**
- `load_stage_reference(csv_path: Path) -> pd.DataFrame`
- `infer_stage(area_um2: float, reference: pd.DataFrame, params: StageParams) -> dict`
- `annotate_with_stage(mask_geometry: pd.DataFrame, reference: pd.DataFrame, params: StageParams) -> pd.DataFrame`

**Source material**
- `build04_perform_embryo_qc.py` (stage logic + buffers)
- `surface_area_ref.csv` / `surface_area_ref_df.csv` legacy files

**Cleanup notes**
- Fail fast if `area_um2` column is missing—this protects the “no raw pixel area” guarantee.
- Keep the interpolation/lookup implementation simple (likely pandas/NumPy).
- Record confidence intervals or residuals as separate columns for downstream QC.

---

## `feature_extraction/consolidate.py`
**Responsibilities**
- Merge tracking table, mask geometry, pose/kinematics, and stage outputs into a single per-snip DataFrame.
- Write `consolidated_snip_features.csv` with a predictable column order.

**Functions to implement**
- `merge_feature_tables(tracking: pd.DataFrame, mask_geom: pd.DataFrame, pose: pd.DataFrame, stage: pd.DataFrame) -> pd.DataFrame`
- `write_consolidated_features(df: pd.DataFrame, output_csv: Path) -> None`
- `validate_consolidated_features(df: pd.DataFrame) -> None`

**Source material**
- Manual merges in `build03A_process_images.py`
- Recent notebooks producing feature aggregation CSVs

**Cleanup notes**
- Use `snip_id` as the only join key; assert uniqueness before writing.
- Preserve mandatory columns (`embryo_id`, `time_int`, `area_um2`, `centroid_x_um`, etc.).
- Leave space for embeddings by including `embedding_calculated` column defaulted to `False`.
- Return a DataFrame so consolidation can be unit tested before writing to disk.

---

## Cross-cutting tasks
- Document column schemas via module-level constants (e.g., `MASK_GEOMETRY_COLUMNS`).
- Feed pixel-size metadata from `preprocessing` through Snakemake config to avoid magic numbers; keep the plumbing explicit until we confirm a shared helper is worth the extra surface area.
- Write unit tests that exercise: mask with known geometry, multi-frame pose table, stage inference boundaries, and full consolidation join.
- Provide docstrings referencing legacy script sections so reviewers can trace parity.
