# Segmentation Module Population Plan

Goal: extract the minimal functions needed to run GroundingDINO + SAM2 and the auxiliary UNet masks without legacy scaffolding (BaseFileHandler, entity trackers, etc.). Modules should expose clear, importable functions that operate on explicit arguments (paths, tensors) and return structured data.

---

## `segmentation/grounded_sam2/frame_organization_for_sam2.py`
**Responsibilities**
- Convert stitched FF images into temporally ordered video tensors per well.
- Handle padding, channel alignment, and resizing consistent with SAM2 expectations.

**Functions to implement**
- `list_frames(input_dir: Path, experiment_id: str, well_id: str) -> list[Path]`
- `load_frames(frame_paths: list[Path], device: torch.device) -> torch.Tensor`
- `build_video_tensor(frames: torch.Tensor, config: SAM2VideoConfig) -> torch.Tensor`
- `prepare_sam2_inputs(raw_dir: Path, output_dir: Path, config: SAM2VideoConfig, device: torch.device) -> tuple[list[Path], Path]`

**Source material**
- `segmentation_sandbox/scripts/pipelines/01_prepare_videos.py`
- `segmentation_sandbox/scripts/utils/mask_utils.py` (for geometry helpers)

**Cleanup notes**
- Remove filesystem discovery; accept explicit directories.
- Ensure deterministic frame ordering (sort by `time_int`).
- Device preference handled via shared `data_pipeline.config.runtime.resolve_device`.
- Emit simple JSON/CSV manifest describing generated video tensors.

---

## `segmentation/grounded_sam2/gdino_detection.py`
**Responsibilities**
- Run GroundingDINO on representative frames to produce seed detections.
- Convert detections to bounding boxes compatible with SAM2 prompts.

**Functions to implement**
- `load_gdino_model(checkpoint: Path, device: torch.device) -> GDINOModel`
- `detect_embryos(video_tensor: torch.Tensor, model: GDINOModel, params: GDINOParams) -> list[Detection]`
- `filter_detections(detections: list[Detection], params: GDINOParams) -> list[Detection]`
- `write_detections(detections: list[Detection], output_json: Path) -> None`

**Source material**
- `segmentation_sandbox/scripts/pipelines/03_gdino_detection.py`
- GDINO utilities under `segmentation_sandbox/models`

**Cleanup notes**
- Return plain dictionaries/lists for detections (no helper classes).
- Ensure thresholds (confidence, NMS) come from `config.segmentation`.
- No global state; model passed explicitly.

---

## `segmentation/grounded_sam2/propagation.py`
**Responsibilities**
- Run SAM2 propagation using GroundingDINO seeds.
- Manage bidirectional tracking, temporal windows, and interpolation.

**Functions to implement**
- `load_sam2_model(checkpoint: Path, device: torch.device) -> SAM2Model`
- `run_forward_pass(model: SAM2Model, video_tensor: torch.Tensor, seeds: list[Detection], params: SAM2Params) -> SAM2Result`
- `run_backward_pass(model: SAM2Model, video_tensor: torch.Tensor, seeds: list[Detection], params: SAM2Params) -> SAM2Result`
- `merge_bidirectional_tracks(forward: SAM2Result, backward: SAM2Result, params: SAM2Params) -> SAM2Result`
- `propagate_masks(video_tensor: torch.Tensor, seeds: list[Detection], params: SAM2Params, *, bidirectional: bool = True) -> SAM2Result`
- `postprocess_tracks(result: SAM2Result, params: SAM2Params) -> SAM2Result`
- `save_sam2_json(result: SAM2Result, output_json: Path) -> None`

**Source material**
- `segmentation_sandbox/scripts/pipelines/04_sam2_video_processing.py`
- `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`

**Cleanup notes**
- Flatten class hierarchies (no `GroundedSamAnnotations`).
- Return simple dicts describing masks, scores, and per-frame metadata.
- GPU-first implementation with CPU fallback for testing.
- Separate I/O from inference: propagation returns data; writing handled in `mask_export`/`csv_formatter`.
- Bidirectional handling controlled via `SAM2Params`; forward-only path still supported for quick tests.

---

## `segmentation/grounded_sam2/bounding_box_utils.py`
**Responsibilities**
- Provide shared conversions between GroundingDINO detection boxes and SAM2 prompt formats.
- Keep box math isolated from model code to simplify testing.

**Functions to implement**
- `gdino_to_sam2_boxes(detections: list[Detection], frame_size: tuple[int, int]) -> list[SAM2Box]`
- `clip_boxes_to_frame(boxes: list[SAM2Box], width: int, height: int) -> list[SAM2Box]`
- `scale_boxes(boxes: list[SAM2Box], scale_x: float, scale_y: float) -> list[SAM2Box]`
- `expand_boxes(boxes: list[SAM2Box], padding: float, max_width: int, max_height: int) -> list[SAM2Box]`
- `boxes_to_tracking_prompts(boxes: list[SAM2Box]) -> list[Prompt]`

**Source material**
- `segmentation_sandbox/scripts/pipelines/03_gdino_detection.py` (detection JSON schema)
- `segmentation_sandbox/scripts/pipelines/04_sam2_video_processing.py` (SAM2 prompt expectations)

**Cleanup notes**
- Represent boxes/prompts as straightforward dictionaries or tuples; keep conversion logic explicit in this module.
- Ensure conversions account for image orientation/rotation applied during preprocessing.
- Avoid implicit normalization – keep coordinate systems explicit and documented.

---

## `segmentation/grounded_sam2/mask_export.py`
**Responsibilities**
- Export SAM2 masks to PNG/NumPy files and write ancillary manifests.
- Handle label remapping, palette assignment, compression.

**Functions to implement**
- `export_masks(result: SAM2Result, output_dir: Path, palette: Optional[List[int]] = None) -> list[Path]`
- `write_manifest(result: SAM2Result, mask_paths: list[Path], output_json: Path) -> None`
- `validate_exports(mask_paths: list[Path]) -> None`

**Source material**
- `segmentation_sandbox/scripts/pipelines/06_export_masks.py`

**Cleanup notes**
- Reuse `mask_utilities` for RLE/geometry conversions as needed.
- Avoid `BaseFileHandler`; operate on explicit paths.
- Provide optional compression settings via config.

---

## `segmentation/grounded_sam2/csv_formatter.py`
**Responsibilities**
- Flatten SAM2 JSON outputs into row-per-embryo-per-frame CSVs.
- Include detection scores, mask areas, centroid coordinates.

**Functions to implement**
- `sam2_result_to_records(result: SAM2Result, pixel_size_um: float) -> list[dict]`
- `merge_with_detection_metadata(records: list[dict], detections: list[Detection]) -> list[dict]`
- `assign_snip_ids(records: list[dict], experiment_id: str) -> list[dict]`
- `write_tracking_table(records: list[dict], output_csv: Path) -> None`

**Source material**
- `segmentation_sandbox/scripts/utils/export_sam2_metadata_to_csv.py`
- `build03A_process_images.py` (for ID/component extraction)

**Cleanup notes**
- Use `identifiers.parsing` for ID parsing instead of regex inline.
- Compute both `area_px` and `area_um2` (using microscope metadata) so downstream features don't guess pixel size.
- Tracking table owns `snip_id = {embryo_id}_s{frame:04d}`; enforce uniqueness before writing.
- Output schema documented in docstring (columns, dtypes). Optional Parquet export can follow once CSV path is validated.

---

## `segmentation/unet/inference.py`
**Responsibilities**
- Run batched inference for all UNet heads on stitched FF images.
- Share pre/post-processing across heads.

**Functions to implement**
- `load_unet_heads(model_dir: Path, device: torch.device) -> dict[str, UNetModel]`
- `run_unet_batch(frames: list[Path], models: dict[str, UNetModel], params: UNetParams) -> dict[str, list[np.ndarray]]`
- `save_unet_outputs(outputs: dict[str, list[np.ndarray]], output_root: Path) -> dict[str, list[Path]]`

**Source material**
- `src/build/build02B_segment_bf_main.py`

**Cleanup notes**
- Collapse duplicate pipelines per head into a single loop.
- Replace environment variable toggles with explicit params (model names, batch size).
- Device selection managed via shared helper.

---

## `segmentation/unet/model_loader.py`
**Responsibilities**
- Discover/load UNet checkpoints defined in config.
- Handle device transfer (GPU/CPU) and eval mode setup.

**Functions to implement**
- `list_unet_models(model_root: Path) -> dict[str, Path]`
- `load_unet_model(model_path: Path, device: torch.device) -> UNetModel`
- `load_models_from_config(config: UNetModelConfig, device: torch.device) -> dict[str, UNetModel]`

**Source material**
- Portions of `build02B_segment_bf_main.py`
- Legacy `model_loader` utilities in segmentation sandbox

**Cleanup notes**
- Remove global caches; return dictionaries of models.
- Support torchscript or standard `state_dict` with minimal branching.

---

## `segmentation/mask_utilities.py`
**Responsibilities**
- Shared geometry helpers for masks (RLE, polygons, bounding boxes).

**Functions to implement**
- `mask_to_rle(mask: np.ndarray) -> str`
- `rle_to_mask(rle: str, shape: tuple[int, int]) -> np.ndarray`
- `mask_to_polygon(mask: np.ndarray) -> list[list[float]]`
- `compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int]`
- `mask_area(mask: np.ndarray) -> int`

**Source material**
- `segmentation_sandbox/scripts/utils/mask_utils.py`

**Cleanup notes**
- Pure NumPy/skimage implementation (no torch dependency here).
- Ensure compatibility with both SAM2 and UNet outputs.
- Add unit tests covering degenerate cases (empty mask, full mask).

---

## Cross-cutting refactor tasks
- Pull configuration defaults from `data_pipeline.config.segmentation`, but allow overrides via Snakemake `config`.
- Keep all GPU inference optional but default-on; include CPU fallback in docs for testing.
- Provide clear logging (per experiment, per well) without the previous logging hierarchy.
- Add smoke tests that mock small tensors to ensure the modules compose end-to-end.
- Ensure pixel-size metadata from preprocessing reaches the CSV formatter (probably by threading values through Snakemake config); revisit a shared helper only if duplication becomes painful.
