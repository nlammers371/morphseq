# Snip Processing Module Population Plan

Goal: split the Build03/04 monolith into focused, importable modules. Snip extraction stays separate from feature computation. Feature modules live under `feature_extraction/` so QC can consume a single consolidated table. Surface-area metrics must always be converted to μm² (pixel size from metadata) before any biology-facing logic runs.

---

## `snip_processing/extraction.py`
**Responsibilities**
- Crop embryo regions from stitched FF images using SAM2 mask metadata.
- Manage padding, bounding boxes, and crop alignment.

**Functions to implement**
- `load_frame(image_path: Path, device: torch.device | str = "cpu") -> np.ndarray`
- `extract_crop(image: np.ndarray, mask: np.ndarray, padding: int, max_size: tuple[int, int]) -> np.ndarray`
- `extract_snips_for_frame(image_path: Path, masks: list[np.ndarray], config: dict) -> list[dict]`
- `extract_snips_for_experiment(sam2_csv: Path, image_root: Path, output_dir: Path, config: dict) -> list[dict]`

**Source material**
- `build03A_process_images.py` (cropping helpers)
- `segmentation_sandbox/scripts/pipelines/06_export_masks.py` (mask file naming)

**Cleanup notes**
- Avoid in-function globbing; rely on explicit paths passed from Snakemake.
- Return plain dicts describing each snip (`snip_id`, `frame_path`, `crop_path`, metadata).
- Handle GPU acceleration only where necessary; majority can stay in NumPy.
- When torch is required, obtain devices via `data_pipeline.config.runtime.resolve_device`.

---

## `snip_processing/rotation.py`
**Responsibilities**
- Standardize snip orientation (e.g., PCA principal axis alignment).
- Calculate angle metadata stored alongside snips.

**Functions to implement**
- `compute_principal_axis(mask: np.ndarray) -> float`
- `rotate_snip(image: np.ndarray, angle: float, fill_value: int = 0) -> np.ndarray`
- `align_snips(snips: list[dict], config: dict) -> list[dict]`

**Source material**
- `build03A_process_images.py` (PCA rotation logic)
- `build04_perform_embryo_qc.py` (orientation metadata usage)

**Cleanup notes**
- Keep math in NumPy/scikit-image; no need for torch here.
- Update snip dicts with `rotation_angle`, `rotation_quality` fields.

---

## `snip_processing/augmentation.py`
**Responsibilities**
- Generate synthetic snips for training (noise, flips, brightness, etc.).

**Functions to implement**
- `augment_snip(image: np.ndarray, mask: np.ndarray, config: dict) -> np.ndarray`
- `augment_snip_batch(snips: list[dict], config: dict) -> list[dict]`

**Source material**
- `build03A_process_images.py` augmentation blocks

**Cleanup notes**
- Use deterministic random seeds when provided for reproducibility.
- Keep augmentation optional; config toggles all behaviours.

---

## `snip_processing/io.py`
**Responsibilities**
- Save cropped/rotated snips and write manifest CSVs.

**Functions to implement**
- `save_snip(image: np.ndarray, destination: Path, image_format: str = "png") -> None`
- `write_snip_manifest(snips: list[dict], output_csv: Path) -> None`
- `load_snip_manifest(manifest_csv: Path) -> list[dict]`

**Source material**
- `build03A_process_images.py`
- Existing analysis scripts consuming snip manifests

**Cleanup notes**
- Centralize file naming (`snip_id` convention) via `identifiers.parsing`.
- Ensure manifests include references to masks, rotations, feature rows, and QC-ready flags.

---

## `feature_extraction/mask_geometry_metrics.py`
**Responsibilities**
- Compute SAM2-mask intrinsic geometry, converting all areas to μm² using microscope metadata.

**Functions to implement**
- `compute_mask_geometry(mask: np.ndarray, pixel_size_um: float) -> dict`
- `summarize_contour(contour: np.ndarray) -> dict`

**Source material**
- Geometry calculations in `build03A_process_images.py`
- Pixel-size lookups in Build01 metadata code

**Cleanup notes**
- Return both `area_px` and `area_um2`; downstream biology must use `area_um2`.
- Keep implementation pure NumPy/scikit-image; no torch dependencies.
- Document schema clearly (keys, units) so QC/analysis stay consistent.

---

## `feature_extraction/pose_kinematics_metrics.py`
**Responsibilities**
- Derive pose and motion metrics from SAM2 tracking table + mask geometry output.

**Functions to implement**
- `compute_pose_metrics(record: dict) -> dict`
- `compute_frame_deltas(current: dict, previous: dict) -> dict`
- `build_pose_table(tracking_table: pd.DataFrame, pixel_size_um: float) -> pd.DataFrame`

**Source material**
- Positional/motion logic in `build03A_process_images.py`
- Tracking QC helpers from `build04_perform_embryo_qc.py`

**Cleanup notes**
- Accept explicit tracking rows; never re-read CSV inside loops.
- Expose centroid/bbox in both px and μm for flexibility.
- Provide velocity in μm/hour when stage metadata available.

---

## `feature_extraction/stage_inference.py`
**Responsibilities**
- Infer developmental stage (HPF) from surface-area curves expressed in μm².

**Functions to implement**
- `load_stage_reference(csv_path: Path) -> pd.DataFrame`
- `infer_stage(area_um2: float, reference: pd.DataFrame, params: StageParams) -> dict`
- `annotate_features_with_stage(features: pd.DataFrame, reference: pd.DataFrame, params: StageParams) -> pd.DataFrame`

**Source material**
- Stage inference block in `build04_perform_embryo_qc.py`
- Legacy `surface_area_ref.csv` usage

**Cleanup notes**
- Treat stage model as data-driven lookup (no heavy ML assumption).
- Raise if area values are still in pixels—hard fail avoids grave regressions.

---

## `feature_extraction/consolidate.py`
**Responsibilities**
- Merge tracking table, mask geometry, pose/kinematics, and stage outputs into `consolidated_snip_features.csv`.

**Functions to implement**
- `merge_feature_tables(tracking: pd.DataFrame, mask_geom: pd.DataFrame, pose: pd.DataFrame, stage: pd.DataFrame) -> pd.DataFrame`
- `write_consolidated_features(df: pd.DataFrame, output_csv: Path) -> None`

**Source material**
- Manual merges inside `build03A_process_images.py`
- Recent notebooks creating per-experiment feature tables

**Cleanup notes**
- Use `snip_id` as the only join key—validate presence/uniqueness before writing.
- Preserve `area_um2` and `embedding_calculated` placeholders so QC + embeddings can extend later.
- Keep function side-effect free apart from writing the final CSV.

---

## Cross-cutting refactor tasks
- Align naming conventions with `identifiers` module (`snip_id`, `well_id`, etc.).
- Ensure all functions return plain dicts/lists/DataFrames suitable for CSV serialization.
- Factor shared config (padding, rotation defaults) into `data_pipeline.config.snips`.
- Provide smoke tests that run extraction → rotation → feature computation on a tiny sample.
- Document in code comments where pixel-size metadata enters the flow so area conversions stay explicit.
