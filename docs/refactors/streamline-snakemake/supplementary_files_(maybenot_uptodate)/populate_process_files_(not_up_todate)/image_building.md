# Image Building Module Population Plan

Goal: convert Phase 1 aligned metadata plus raw microscope dumps into
normalized stitched FF images while generating consistent entity IDs and
diagnostics. Everything here maps directly to
`built_image_data/{experiment_id}/stitched_ff_images/…` and
`build_diagnostics/{experiment_id}/stitching_{microscope}.csv`, and must
interoperate with `identifiers/parsing.py` so downstream stages receive
canonical `well_id`, `image_id`, and `channel` naming.

---

## Shared Expectations

- **Inputs:** raw microscope data under
  `raw_image_data/{microscope}/{experiment_id}/` and Phase 1 hand-offs
  (`input_metadata_alignment/{exp}/aligned_metadata/scope_and_plate.csv`).
- **Outputs:**
  - `built_image_data/{exp}/stitched_ff_images/{well_id}/{channel}/{image_id}.tif`
  - `build_diagnostics/{exp}/stitching_{microscope}.csv`
- **ID generation:** use `identifiers/parsing.py` helpers for
  `experiment_id`, `well_id`, `image_id`, and to normalize channels
  (`BF`, `GFP`, …). No `ch00`/`ch01` legacy names.
- **Diagnostics columns (recommended, not enforced):**
  - Required for provenance: `experiment_id`, `well_id`, `channel`, `frame_index`
  - Optional QA metrics: focus scores, tile counts, warnings, processing timestamps
  - Per-microscope columns encouraged (e.g., Keyence z-plane selection, YX1 bit depth)
  - No formal schema validation - diagnostics are for human review, not pipeline automation

---

## `image_building/shared/layout.py`

- Wraps `identifiers/parsing.py` helpers so every builder produces paths
  and IDs the same way.
- Suggested helpers:
  - `make_well_id(experiment_id: str, well_index: str) -> str`
  - `make_image_id(experiment_id: str, well_index: str, channel: str, frame_index: int) -> str`
  - `make_image_path(exp_id: str, well_index: str, channel: str, frame_index: int, root: Path) -> Path`
- All builders must rely on these helpers (rather than manual string
  formatting) to guarantee `built_image_data/{exp}/stitched_ff_images/`
  remains consistent across microscopes.

---

## `image_building/keyence/stitched_ff_builder.py`

**Responsibilities**
- Read Phase 1 aligned metadata to resolve wells, channels, and frame
  order.
- Collapse tile stacks via `z_stacking.py`, stitch into FF images, apply
  flat-field corrections.
- Emit per-frame diagnostics rows (focus score, tiles consumed, warnings).

**Key functions**
- `build_keyence_stitched_ff(experiment_ctx: ExperimentContext, output_root: Path, diagnostics_csv: Path) -> None`
- `stitch_frame(well_ctx: WellContext, frame_ctx: FrameContext) -> np.ndarray`
- `write_stitched_frame(image: np.ndarray, image_id: str, channel: str, destination: Path) -> None`
- `record_diagnostics(record: DiagnosticRecord, csv_path: Path) -> None`

**Source material**
- `src/build/build01A_compile_keyence_torch.py`
- `src/build/build01AB_stitch_keyence_z_slices.py` (focus metrics)

**Implementation notes**
- Pull device configuration from `data_pipeline.config.runtime.resolve_device`.
- Consume normalized channel names from aligned metadata (no renaming in this stage).
- Ensure output layout matches manifest expectations exactly.

---

## `image_building/keyence/z_stacking.py`

**Responsibilities**
- Provide tile-level focus selection for Keyence stacks.
- Surface focus metrics back to the builder for diagnostics.

**Key functions**
- `compute_focus_metric(stack: np.ndarray, method: FocusMethod) -> np.ndarray`
- `select_focus_plane(stack: np.ndarray, method: FocusMethod) -> np.ndarray`
- `collapse_stack(stack_dir: Path, temp_dir: Path) -> list[np.ndarray]`

**Implementation notes**
- Deterministic ordering of tiles (sorted filenames).
- Expose focus method/threshold configuration via `config.microscopes`.
- Return metrics in a structured form so the builder can log them.

---

## `image_building/yx1/stitched_ff_builder.py`

**Responsibilities**
- Handle YX1-specific preprocessing (channel ordering, bit depth, flat-field).
- Produce the same stitched FF layout as the Keyence builder, with provenance columns
  in diagnostics (experiment_id, well_id, channel, frame_index) plus YX1-specific metrics.

**Key functions**
- `build_yx1_stitched_ff(experiment_ctx: ExperimentContext, output_root: Path, diagnostics_csv: Path) -> None`
- `process_frame(raw_frame: np.ndarray, channel: str, config: YX1ProcessingConfig) -> np.ndarray`
- `write_stitched_frame(...)` (shared signature with Keyence implementation)

**Source material**
- `src/build/build01B_compile_yx1_images_torch.py`

**Implementation notes**
- Reuse shared ID helpers, do not embed YX1-specific naming logic in downstream stages.
- Support GPU acceleration via torch where practical, but provide CPU fallback.
- Include provenance columns in diagnostics for consistency with Keyence builder.

---

## Future Shared Utilities (optional)

If common functionality emerges (diagnostic record writer, ID helper wrappers, flat-field utilities), collect them under `image_building/shared/`. For now, keep each microscope self-contained to avoid premature abstraction.

---

## Validation & Testing

- Smoke test builders with synthetic or small real experiments to ensure:
  - Output directory structure matches expectations.
  - Diagnostics CSVs include provenance columns (experiment_id, well_id, channel, frame_index).
  - IDs parse correctly via `identifiers/parsing`.
  - Channel normalization is preserved (`BF`, `GFP`, …).
- Capture representative before/after examples for documentation once migration completes.

**Note:** Diagnostics CSVs do not require formal schema validation. See Phase 2 docs
for rationale on keeping diagnostics validation lightweight.
